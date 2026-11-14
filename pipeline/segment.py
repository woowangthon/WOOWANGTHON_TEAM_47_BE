"""② 영역 분할: 원본 BGR 사진 → 단일 채널 MaskClass 라벨 마스크.

기존 schema.py / skeleton.py 와 호환되는 Segmenter 인터페이스를 유지한다.
현재 v4_best.pt 클래스: 0=linear, 1=dot. 결과 라벨: 0=배경, 1=linear, 2=dot.
추후 모델 클래스가 늘어나면 '클래스 이름'에 따라 3=crosswalk, 4=road로 매핑한다.

주의: 횡단보도/차도를 검출할 수 없는 2클래스 모델에서 이들 면적이 0이라는 사실은
그 객체가 실제로 없다는 증거가 아니다. C계열 규칙에서 별도의 모델 결과가 필요하다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .schema import BLOCK_CLASSES, MaskClass, PhotoKind, Segmentation


# YOLO 클래스 ID를 믿고 +1 하지 말고, 모델에 기록된 이름으로 매핑한다.
_NAME_TO_MASK_CLASS: dict[str, MaskClass] = {
    "linear": MaskClass.LINEAR,
    "dot": MaskClass.DOT,
    "crosswalk": MaskClass.CROSSWALK,
    "zebra": MaskClass.CROSSWALK,
    "road": MaskClass.ROAD,
}


class Segmenter:
    def __init__(self, weights: str | Path, conf: float = 0.25, imgsz: int = 640):
        weights = Path(weights)
        if not weights.is_file():
            raise FileNotFoundError(f"YOLO 가중치 파일을 찾을 수 없습니다: {weights}")
        if not 0.0 < conf <= 1.0:
            raise ValueError("conf는 0보다 크고 1 이하여야 합니다.")

        from ultralytics import YOLO  # requirements-ml.txt

        self.model = YOLO(str(weights))
        self.conf = conf
        self.imgsz = imgsz
        self.class_mapping = self._get_class_mapping(self.model.names)
        self.available_classes = sorted({cls.name.lower() for cls in self.class_mapping.values()})

    @staticmethod
    def _get_class_mapping(names: dict | list) -> dict[int, MaskClass]:
        items = names.items() if isinstance(names, dict) else enumerate(names)
        mapping: dict[int, MaskClass] = {}
        unknown = []
        for class_id, class_name in items:
            normalized = str(class_name).strip().lower()
            if normalized not in _NAME_TO_MASK_CLASS:
                unknown.append(str(class_name))
            else:
                mapping[int(class_id)] = _NAME_TO_MASK_CLASS[normalized]
        if unknown:
            raise ValueError(
                f"모델에 정의되지 않은 클래스가 있습니다: {unknown}. "
                "pipeline/schema.py의 MaskClass와 학습 클래스명을 확인하세요."
            )
        return mapping

    def predict(self, image_bgr: np.ndarray, photo: PhotoKind) -> tuple[np.ndarray, Segmentation]:
        if not isinstance(image_bgr, np.ndarray) or image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
            raise ValueError("입력 이미지는 H×W×3 형태의 OpenCV BGR 배열이어야 합니다.")
        if image_bgr.dtype != np.uint8:
            raise ValueError("이미지 dtype은 uint8이어야 합니다.")
        photo = PhotoKind(photo)
        h, w = image_bgr.shape[:2]
        label = np.zeros((h, w), dtype=np.uint8)
        conf_map = np.zeros((h, w), dtype=np.float32)

        result = self.model.predict(
            source=image_bgr,
            conf=self.conf,
            imgsz=self.imgsz,
            retina_masks=True,
            verbose=False,
        )[0]

        if result.masks is not None and result.boxes is not None:
            instance_masks = result.masks.data.cpu().numpy()
            class_ids = result.boxes.cls.int().cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            if len(instance_masks) != len(class_ids) or len(scores) != len(class_ids):
                raise RuntimeError("YOLO 마스크와 클래스/신뢰도 개수가 일치하지 않습니다.")

            # 동일 위치에서 겹치는 인스턴스는 높은 신뢰도 결과가 우선한다.
            for instance_mask, class_id, score in sorted(
                zip(instance_masks, class_ids, scores), key=lambda x: float(x[2])
            ):
                cls = self.class_mapping.get(int(class_id))
                if cls is None:
                    raise ValueError(f"알 수 없는 모델 클래스 ID: {class_id}")

                binary = np.asarray(instance_mask) > 0.5
                if binary.shape != (h, w):
                    binary = cv2.resize(
                        binary.astype(np.uint8),
                        (w, h),
                        interpolation=cv2.INTER_NEAREST,
                    ).astype(bool)
                label[binary] = int(cls)
                conf_map[binary] = float(score)

        return label, summarize(label, photo, conf_map)


def summarize(
    label: np.ndarray,
    photo: PhotoKind,
    conf_map: Optional[np.ndarray] = None,
) -> Segmentation:
    """정답/추론 마스크 공용 요약. 기존 schema.py의 반환 형식 유지."""
    if label.ndim != 2:
        raise ValueError("라벨 마스크는 H×W 단일 채널이어야 합니다.")
    h, w = label.shape
    areas = {
        cls.name.lower(): int(np.count_nonzero(label == int(cls)))
        for cls in MaskClass if cls != MaskClass.BACKGROUND
    }
    mean_conf = None
    if conf_map is not None:
        if conf_map.shape != label.shape:
            raise ValueError("conf_map 크기가 라벨 마스크와 다릅니다.")
        blocks = np.isin(label, [int(cls) for cls in BLOCK_CLASSES])
        if blocks.any():
            mean_conf = float(conf_map[blocks].mean())
        # 검출 0개면 평균 신뢰도를 계산할 수 없으므로 None.

    return Segmentation(
        photo=PhotoKind(photo),
        width=w,
        height=h,
        class_area_px=areas,
        mean_confidence=mean_conf,
    )
