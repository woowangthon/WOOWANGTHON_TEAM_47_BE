"""② 영역 분할: 사진 → 라벨 마스크(MaskClass 값). (학습: A 기본, B 미세조정)

YOLOv8-seg 클래스 순서는 training/data.yaml 과 같아야 한다:
    0 linear, 1 dot, 2 crosswalk, 3 road  →  라벨 값 = 클래스 id + 1
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .schema import BLOCK_CLASSES, MaskClass, PhotoKind, Segmentation


class Segmenter:
    def __init__(self, weights: str | Path, conf: float = 0.25, imgsz: int = 1024):
        from ultralytics import YOLO  # requirements-ml.txt

        self.model = YOLO(str(weights))
        self.conf = conf
        self.imgsz = imgsz

    def predict(self, image_bgr: np.ndarray, photo: PhotoKind) -> tuple[np.ndarray, Segmentation]:
        h, w = image_bgr.shape[:2]
        label = np.zeros((h, w), np.uint8)
        conf_map = np.zeros((h, w), np.float32)

        res = self.model.predict(image_bgr, conf=self.conf, imgsz=self.imgsz, retina_masks=True, verbose=False)[0]
        if res.masks is not None:
            masks = res.masks.data.cpu().numpy()  # (N, H, W)
            classes = res.boxes.cls.cpu().numpy().astype(int)
            scores = res.boxes.conf.cpu().numpy()
            # 겹치면 신뢰도가 높은 쪽이 이긴다
            for m, c, s in sorted(zip(masks, classes, scores), key=lambda t: t[2]):
                if m.shape != (h, w):
                    m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
                sel = m > 0.5
                label[sel] = c + 1
                conf_map[sel] = s

        return label, summarize(label, photo, conf_map)


def summarize(label: np.ndarray, photo: PhotoKind, conf_map: Optional[np.ndarray] = None) -> Segmentation:
    """라벨 마스크에서 Segmentation 요약을 만든다. 정답 마스크로 개발할 때도 쓴다."""
    h, w = label.shape[:2]
    areas = {c.name.lower(): int((label == c).sum()) for c in MaskClass if c != MaskClass.BACKGROUND}
    mean_conf = None
    if conf_map is not None:
        blk = np.isin(label, [int(c) for c in BLOCK_CLASSES])
        mean_conf = float(conf_map[blk].mean()) if blk.any() else 0.0
    return Segmentation(photo=photo, width=w, height=h, class_area_px=areas, mean_confidence=mean_conf)
