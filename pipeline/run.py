"""파이프라인 진입점. ② → ③ → ④ → ⑤.

모델 없이 정답 마스크로 개발할 때는 analyze_label() + judge() 를 바로 쓴다.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .crosswalk import drop_unstriped
from .rectify import analyze_rois, rectify_by_crosswalk
from .rules import judge
from .schema import InspectionResult, PhotoAnalysis, PhotoKind
from .segment import Segmenter, summarize
from .skeleton import SkeletonConfig, extract_path


def analyze_label(
    label: np.ndarray,
    photo: PhotoKind,
    image_bgr: Optional[np.ndarray] = None,
    segmentation=None,
    skel_cfg: Optional[SkeletonConfig] = None,
) -> PhotoAnalysis:
    seg = segmentation or summarize(label, photo)
    path = extract_path(label, photo, skel_cfg)
    rect = None
    rois = []
    if image_bgr is not None:
        if photo == PhotoKind.WIDE:
            rect = rectify_by_crosswalk(image_bgr, label)
        rois = analyze_rois(image_bgr, label, path, photo, rect)
    return PhotoAnalysis(segmentation=seg, path=path, rois=rois, rectification=rect)


def run_inspection(
    segmenter: Segmenter,
    close_bgr: Optional[np.ndarray],
    wide_bgr: Optional[np.ndarray],
    inspection_id: Optional[str] = None,
) -> InspectionResult:
    close = wide = wide_label = None
    if close_bgr is not None:
        label, seg = segmenter.predict(close_bgr, PhotoKind.CLOSE)
        close = analyze_label(label, PhotoKind.CLOSE, close_bgr, seg)
    if wide_bgr is not None:
        wide_label, seg = segmenter.predict(wide_bgr, PhotoKind.WIDE)
        # 줄무늬 없는 횡단보도 오검출 제거 후 면적 다시 계산 (H3 판정에 쓰임)
        wide_label, _ = drop_unstriped(wide_bgr, wide_label)
        seg = seg.model_copy(update={"class_area_px": summarize(wide_label, PhotoKind.WIDE).class_area_px})
        wide = analyze_label(wide_label, PhotoKind.WIDE, wide_bgr, seg)
    result = judge(close, wide, wide_label)
    result.inspection_id = inspection_id
    return result
