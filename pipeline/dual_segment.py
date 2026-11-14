"""② 영역 분할 (두 모델 통합): 블록 모델 + 장면 모델 → MaskClass 라벨 하나. (담당: 은재)

    블록 모델 (v4/v6 …)   linear, dot
    장면 모델 (scene_v*)   crosswalk, road  (+ 5클래스 모델은 alley, sidewalk, tactile)

사진 종류(가까운/넓은)와 상관없이 모든 사진을 똑같이 처리한다.

순서
1. 장면 모델 결과를 칠한다 (횡단보도·차도). 겹치면 신뢰도 높은 쪽이 이긴다
2. 줄무늬 필터: 줄무늬 없는 '횡단보도'(가까운 아스팔트·콘크리트 오인식)를 지운다 (crosswalk.py)
   → 블록 제거보다 먼저 해야 가짜 횡단보도 때문에 진짜 블록이 지워지지 않는다
3. 블록 모델 결과를 위에 칠한다. 단, 블록 조각의 대부분이 횡단보도·차도 안에 있으면 버린다
   (횡단보도 안 노란 칠을 선형으로 오인식하는 경우). 연석에 걸쳐 일부만 겹친 블록은 그대로 둔다
4. 블록 신뢰도 지도 → Segmentation.mean_confidence (H1)

이중 검출 (장면 모델에 tactile 클래스가 있을 때만. 없으면 위 순서 그대로)
    T = 장면 모델 tactile(점형·선형 구분 없는 점자블록 영역) 마스크의 합집합
    경우 1 (동의)        블록 조각이 T와 겹침                         → 그대로 사용
    경우 2 (블록 모델만) 블록 조각 면적 중 T와 겹치는 비율 < 기준     → 버림 (간판·나무·횡단보도 줄무늬 오인식)
    경우 3 (tactile만)   T인데 통합 라벨에 블록이 없는 덩어리          → "종류 모름 블록" 면적
                         → Segmentation.unknown_block_area_px (규칙에서 C4·C5 의심을 보류로 바꿈)
기준값은 실제 사진 25장으로 정한 값이라 MergeConfig로 조정한다.

클래스는 모델 파일에 기록된 이름으로 매핑한다 (번호 +1 방식은 모델마다 달라서 쓰지 않음).
segment.Segmenter 와 같은 predict(image, photo) -> (label, Segmentation) 형식이라 run.py에 그대로 꽂힌다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .crosswalk import StripeConfig, drop_unstriped
from .schema import MaskClass, PhotoKind, Segmentation
from .segment import summarize

NAME_TO_CLASS: dict[str, MaskClass] = {
    "linear": MaskClass.LINEAR,
    "dot": MaskClass.DOT,
    "crosswalk": MaskClass.CROSSWALK,
    "zebra": MaskClass.CROSSWALK,
    "road": MaskClass.ROAD,
}
# 5클래스 장면 모델의 나머지 클래스: 라벨에는 칠하지 않는다 (tactile은 이중 검출에만 씀)
SCENE_EXTRA_NAMES = frozenset({"alley", "sidewalk", "tactile"})
TACTILE_NAME = "tactile"

# 검출 하나 = (H×W bool 마스크, MaskClass, 신뢰도)
Detection = tuple[np.ndarray, MaskClass, float]


@dataclass
class MergeConfig:
    block_conf: float = 0.25
    scene_conf: float = 0.25
    imgsz: int = 640
    max_block_in_vehicle: float = 0.6  # 블록 조각의 이 비율 이상이 횡단보도·차도 안이면 버림
    min_block_area: float = 0.0005  # 사진 면적 대비. 더 작은 블록 조각은 버림
    stripe: StripeConfig = field(default_factory=StripeConfig)
    # 이중 검출 (tactile 클래스가 있는 장면 모델일 때만)
    min_block_in_tactile: float = 0.10  # 경우 2: 블록 조각 면적 중 tactile과 겹치는 비율이 이보다 낮으면 버림
    unknown_margin: float = 0.012  # 경우 3: 블록 주변 여유·잡티 제거 크기 (사진 짧은 변 대비)
    min_unknown_area: float = 0.005  # 경우 3: 종류 모름 덩어리 최소 면적 (사진 면적 대비)


def _names(names) -> list[tuple[int, str]]:
    items = names.items() if isinstance(names, dict) else enumerate(names)
    return [(int(i), str(n).strip().lower()) for i, n in items]


def class_mapping(names, allowed: set[MaskClass], skip: frozenset[str] = frozenset()) -> dict[int, MaskClass]:
    """YOLO model.names → {클래스 id: MaskClass}.

    skip에 있는 이름은 조용히 건너뛴다 (장면 모델의 alley·sidewalk·tactile).
    그 밖의 모르는 이름이나 다른 모델용 클래스가 있으면 에러.
    """
    out, bad = {}, []
    for i, n in _names(names):
        if n in skip:
            continue
        cls = NAME_TO_CLASS.get(n)
        if cls is None or cls not in allowed:
            bad.append(n)
        else:
            out[i] = cls
    if bad:
        raise ValueError(f"이 모델에 맞지 않는 클래스 이름: {bad} (허용: {[c.name.lower() for c in allowed]})")
    return out


def _filter_by_tactile(blocks: list[Detection], tactile: np.ndarray, cfg: MergeConfig) -> list[Detection]:
    """경우 1·2: tactile과 거의 안 겹치는 블록 조각은 버린다 (신뢰도와 무관)."""
    kept = []
    for m, cls, s in blocks:
        area = m.sum()
        if area > 0 and (m & tactile).sum() / area < cfg.min_block_in_tactile:
            continue
        kept.append((m, cls, s))
    return kept


def unknown_blocks(label: np.ndarray, tactile: np.ndarray, cfg: MergeConfig) -> np.ndarray:
    """경우 3: tactile인데 통합 라벨에 블록(선형·점형)이 없는 큰 덩어리 = 종류 모름 블록 마스크."""
    h, w = label.shape[:2]
    blocks = np.isin(label, [int(MaskClass.LINEAR), int(MaskClass.DOT)])
    k = max(3, round(cfg.unknown_margin * min(h, w)) | 1)
    kernel = np.ones((k, k), np.uint8)
    near = cv2.dilate(blocks.astype(np.uint8), kernel) > 0  # 블록 경계 어긋남은 무시
    u = cv2.morphologyEx((tactile & ~near).astype(np.uint8), cv2.MORPH_OPEN, kernel)
    n, cc, st, _ = cv2.connectedComponentsWithStats(u)
    big = [i for i in range(1, n) if st[i, cv2.CC_STAT_AREA] >= cfg.min_unknown_area * h * w]
    return np.isin(cc, big)


def merge_detections(
    image_bgr: np.ndarray,
    scene: list[Detection],
    blocks: list[Detection],
    photo: PhotoKind,
    cfg: Optional[MergeConfig] = None,
    tactile: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, Segmentation]:
    """두 모델의 검출 결과를 라벨 하나로 합친다. 모델 없이도 돌아가서 테스트에 쓴다.

    tactile(장면 모델 tactile 마스크 합집합)이 있으면 이중 검출을 켠다. None이면 예전과 똑같다.
    """
    cfg = cfg or MergeConfig()
    h, w = image_bgr.shape[:2]
    if tactile is not None:
        blocks = _filter_by_tactile(blocks, tactile, cfg)  # 경우 2
    label = np.zeros((h, w), np.uint8)
    conf_map = np.zeros((h, w), np.float32)

    for m, cls, _ in sorted(scene, key=lambda d: d[2]):  # 1. 장면
        label[m] = int(cls)
    label, _ = drop_unstriped(image_bgr, label, cfg.stripe)  # 2. 줄무늬 필터
    vehicle = np.isin(label, [int(MaskClass.CROSSWALK), int(MaskClass.ROAD)])

    min_px = cfg.min_block_area * h * w
    for m, cls, score in sorted(blocks, key=lambda d: d[2]):  # 3. 블록
        area = m.sum()
        if area < min_px or (m & vehicle).sum() / area >= cfg.max_block_in_vehicle:
            continue
        label[m] = int(cls)
        conf_map[m] = score
    seg = summarize(label, photo, conf_map)
    if tactile is not None:  # 경우 3
        seg.unknown_block_area_px = int(unknown_blocks(label, tactile, cfg).sum())
    return label, seg


def _to_masks(result, mapping: dict[int, MaskClass], h: int, w: int,
              tactile_ids: frozenset[int] = frozenset()) -> tuple[list[Detection], Optional[np.ndarray]]:
    """YOLO 결과 → (매핑된 검출들, tactile 합집합 마스크 또는 None). 매핑에 없는 클래스는 건너뛴다."""
    tactile = np.zeros((h, w), bool) if tactile_ids else None
    if result.masks is None or result.boxes is None:
        return [], tactile
    out = []
    for m, c, s in zip(result.masks.data.cpu().numpy(), result.boxes.cls.int().cpu().numpy(),
                       result.boxes.conf.cpu().numpy()):
        c = int(c)
        if c not in mapping and c not in tactile_ids:
            continue
        if m.shape != (h, w):
            m = cv2.resize(m.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
        if c in tactile_ids:
            tactile |= m > 0.5
        else:
            out.append((m > 0.5, mapping[c], float(s)))
    return out, tactile


class DualSegmenter:
    """블록 모델 + 장면 모델. segment.Segmenter 와 같은 predict 형식."""

    def __init__(self, block_weights: str | Path, scene_weights: str | Path, cfg: Optional[MergeConfig] = None):
        from ultralytics import YOLO  # requirements-ml.txt

        for p in (block_weights, scene_weights):
            if not Path(p).is_file():
                raise FileNotFoundError(f"가중치 파일이 없음: {p}")
        self.cfg = cfg or MergeConfig()
        self.block_model = YOLO(str(block_weights))
        self.scene_model = YOLO(str(scene_weights))
        self.block_map = class_mapping(self.block_model.names, {MaskClass.LINEAR, MaskClass.DOT})
        self.scene_map = class_mapping(self.scene_model.names, {MaskClass.CROSSWALK, MaskClass.ROAD},
                                       skip=SCENE_EXTRA_NAMES)
        self.tactile_ids = frozenset(i for i, n in _names(self.scene_model.names) if n == TACTILE_NAME)

    @property
    def dual_detection(self) -> bool:
        """장면 모델에 tactile 클래스가 있어 이중 검출이 켜졌는가."""
        return bool(self.tactile_ids)

    def _run(self, model, mapping, image_bgr, conf, tactile_ids=frozenset()):
        h, w = image_bgr.shape[:2]
        r = model.predict(image_bgr, conf=conf, imgsz=self.cfg.imgsz, retina_masks=True, verbose=False)[0]
        return _to_masks(r, mapping, h, w, tactile_ids)

    def predict(self, image_bgr: np.ndarray, photo: PhotoKind = PhotoKind.WIDE) -> tuple[np.ndarray, Segmentation]:
        scene, tactile = self._run(self.scene_model, self.scene_map, image_bgr, self.cfg.scene_conf, self.tactile_ids)
        blocks, _ = self._run(self.block_model, self.block_map, image_bgr, self.cfg.block_conf)
        return merge_detections(image_bgr, scene, blocks, photo, self.cfg, tactile)
