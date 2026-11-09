"""④ 정밀 분석: 시점 보정 + ROI 블록 종류·돌출선 방향. (담당: B)"""

from __future__ import annotations

import numpy as np

from .schema import PathStructure, PhotoKind, Rectification, RoiAnalysis


def rectify_by_crosswalk(image_bgr: np.ndarray, label: np.ndarray) -> Rectification:
    """횡단보도 줄무늬로 호모그래피를 구해 위에서 본 평면으로 보정한다.

    TODO(B):
    - 횡단보도 마스크에서 줄무늬 경계선 검출 (Canny + HoughLinesP)
    - 줄무늬 방향의 소실점 / 평행선 2쌍 → 호모그래피
    - 진행축 각도(crosswalk_axis_deg) 계산
    - 실패 조건(줄무늬 부족, 비정상 스케일) → ok=False, reason 기록 → H2
    """
    return Rectification(ok=False, reason="not implemented")


def analyze_rois(
    image_bgr: np.ndarray,
    label: np.ndarray,
    path: PathStructure,
    photo: PhotoKind,
    rect: Rectification | None = None,
) -> list[RoiAnalysis]:
    """꺾임·분기점, 횡단보도 진입부 ROI의 블록 종류와 돌출선 방향.

    TODO(B):
    - ROI 잘라내기 (보정 성공 시 보정 평면에서)
    - 2D FFT 스펙트럼의 주 방향 → ridge_angle_deg (선형), 등방성이면 점형
    - confidence: 스펙트럼 피크 강도 비
    """
    return []
