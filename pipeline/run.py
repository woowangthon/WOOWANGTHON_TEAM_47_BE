"""파이프라인 진입점. ② → ③ → ④ → ⑤.

모델 없이 정답 마스크로 개발할 때는 analyze_label() + judge() 를 바로 쓴다.
segmenter는 dual_segment.DualSegmenter(블록+장면 모델)를 쓴다. segment.Segmenter(모델 하나)도 형식은 같다.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .rectify import analyze_rois, rectify_by_crosswalk
from .rules import judge, judge_photo
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


def run_photo(segmenter, image_bgr: np.ndarray, inspection_id: Optional[str] = None) -> InspectionResult:
    """사진 한 장 검사 (서비스 기본). 사진 종류를 묻지 않고, 보이는 것에 따라 M·C 계열을 판정한다.

    분석은 넓은 사진 기준(시점 보정·진입부 ROI 포함)으로 한 번만 하고 M·C 계열이 같이 쓴다.
    """
    label, seg = segmenter.predict(image_bgr, PhotoKind.WIDE)
    analysis = analyze_label(label, PhotoKind.WIDE, image_bgr, seg)
    result = judge_photo(analysis, label)
    result.inspection_id = inspection_id
    return result


def run_inspection(
    segmenter: Segmenter,
    close_bgr: Optional[np.ndarray],
    wide_bgr: Optional[np.ndarray],
    inspection_id: Optional[str] = None,
) -> InspectionResult:
    """예전 사진 두 장 구조 (호환용)."""
    close = wide = wide_label = None
    if close_bgr is not None:
        label, seg = segmenter.predict(close_bgr, PhotoKind.CLOSE)
        close = analyze_label(label, PhotoKind.CLOSE, close_bgr, seg)
    if wide_bgr is not None:
        wide_label, seg = segmenter.predict(wide_bgr, PhotoKind.WIDE)  # 줄무늬 필터는 DualSegmenter 안에서
        wide = analyze_label(wide_label, PhotoKind.WIDE, wide_bgr, seg)
    result = judge(close, wide, wide_label)
    result.inspection_id = inspection_id
    return result
