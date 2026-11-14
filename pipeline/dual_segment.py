"""② 영역 분할 (두 모델 통합): 블록 모델 + 장면 모델 → MaskClass 라벨 하나. (담당: 은재)

    블록 모델 (v4/v6 …)   linear, dot
    장면 모델 (scene_v*)   crosswalk, road

사진 종류(가까운/넓은)와 상관없이 모든 사진을 똑같이 처리한다.

순서
1. 장면 모델 결과를 칠한다 (횡단보도·차도). 겹치면 신뢰도 높은 쪽이 이긴다
2. 줄무늬 필터: 줄무늬 없는 '횡단보도'(가까운 아스팔트·콘크리트 오인식)를 지운다 (crosswalk.py)
   → 블록 제거보다 먼저 해야 가짜 횡단보도 때문에 진짜 블록이 지워지지 않는다
3. 블록 모델 결과를 위에 칠한다. 단, 블록 조각의 대부분이 횡단보도·차도 안에 있으면 버린다
   (횡단보도 안 노란 칠을 선형으로 오인식하는 경우). 연석에 걸쳐 일부만 겹친 블록은 그대로 둔다
4. 블록 신뢰도 지도 → Segmentation.mean_confidence (H1)

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


def class_mapping(names, allowed: set[MaskClass]) -> dict[int, MaskClass]:
    """YOLO model.names → {클래스 id: MaskClass}. 모르는 이름이나 다른 모델용 클래스가 있으면 에러."""
    items = names.items() if isinstance(names, dict) else enumerate(names)
    out, bad = {}, []
    for i, n in items:
        cls = NAME_TO_CLASS.get(str(n).strip().lower())
        if cls is None or cls not in allowed:
            bad.append(str(n))
        else:
            out[int(i)] = cls
    if bad:
        raise ValueError(f"이 모델에 맞지 않는 클래스 이름: {bad} (허용: {[c.name.lower() for c in allowed]})")
    return out


def merge_detections(
    image_bgr: np.ndarray,
    scene: list[Detection],
    blocks: list[Detection],
    photo: PhotoKind,
    cfg: Optional[MergeConfig] = None,
) -> tuple[np.ndarray, Segmentation]:
    """두 모델의 검출 결과를 라벨 하나로 합친다. 모델 없이도 돌아가서 테스트에 쓴다."""
    cfg = cfg or MergeConfig()
    h, w = image_bgr.shape[:2]
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
    return label, summarize(label, photo, conf_map)


def _to_masks(result, mapping: dict[int, MaskClass], h: int, w: int) -> list[Detection]:
    if result.masks is None or result.boxes is None:
        return []
    out = []
    for m, c, s in zip(result.masks.data.cpu().numpy(), result.boxes.cls.int().cpu().numpy(),
                       result.boxes.conf.cpu().numpy()):
        if m.shape != (h, w):
            m = cv2.resize(m.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
        out.append((m > 0.5, mapping[int(c)], float(s)))
    return out


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
        self.scene_map = class_mapping(self.scene_model.names, {MaskClass.CROSSWALK, MaskClass.ROAD})

    def _run(self, model, mapping, image_bgr, conf) -> list[Detection]:
        h, w = image_bgr.shape[:2]
        r = model.predict(image_bgr, conf=conf, imgsz=self.cfg.imgsz, retina_masks=True, verbose=False)[0]
        return _to_masks(r, mapping, h, w)

    def predict(self, image_bgr: np.ndarray, photo: PhotoKind = PhotoKind.WIDE) -> tuple[np.ndarray, Segmentation]:
        scene = self._run(self.scene_model, self.scene_map, image_bgr, self.cfg.scene_conf)
        blocks = self._run(self.block_model, self.block_map, image_bgr, self.cfg.block_conf)
        return merge_detections(image_bgr, scene, blocks, photo, self.cfg)
