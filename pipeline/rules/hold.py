"""H 계열: 판정 보류 조건. H1·H4 담당 A, H2·H3 담당 B."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..schema import BLOCK_CLASSES, Finding, PhotoAnalysis, PhotoKind, Verdict


@dataclass
class HoldConfig:
    min_confidence: float = 0.5  # H1
    min_crosswalk_ratio: float = 0.01  # H3: 사진 면적 대비
    min_block_ratio: float = 0.02  # H4: 가장 큰 블록 덩어리 면적 비
    min_block_width: float = 0.035  # M 관문: 블록 줄 폭이 사진 짧은 변의 이 비율 이상이어야 점형·선형 구분을 믿음


def check_h1(p: PhotoAnalysis, cfg: HoldConfig) -> list[Finding]:
    """가림·그림자·분할 신뢰도 낮음 → 해당 사진 전체 보류, 재촬영 요청."""
    c = p.segmentation.mean_confidence
    if c is None or c >= cfg.min_confidence:
        return []
    return [Finding(code="H1", verdict=Verdict.HOLD, photo=p.segmentation.photo,
                    message="블록 인식 신뢰도가 낮음 (가림·그림자). 다시 촬영해 주세요", confidence=c)]


def has_class(wide: PhotoAnalysis, name: str, cfg: HoldConfig) -> bool:
    s = wide.segmentation
    return s.class_area_px.get(name, 0) >= cfg.min_crosswalk_ratio * s.width * s.height


def check_h3(wide: PhotoAnalysis, cfg: HoldConfig) -> list[Finding]:
    """횡단보도·차도가 모두 없음 → C 계열 전체 보류, M만 진행."""
    if has_class(wide, "crosswalk", cfg) or has_class(wide, "road", cfg):
        return []
    return [Finding(code="H3", verdict=Verdict.HOLD, photo=PhotoKind.WIDE,
                    message="넓은 사진에 횡단보도·차도가 보이지 않음. 3~5m 뒤에서 횡단보도가 함께 나오게 찍어 주세요")]


def check_h3_partial(wide: PhotoAnalysis, cfg: HoldConfig) -> list[Finding]:
    """차도는 있고 횡단보도만 없음 → C2만 판정, C1·C3·C4 보류."""
    if has_class(wide, "crosswalk", cfg):
        return []
    return [Finding(code="H3", verdict=Verdict.HOLD, photo=PhotoKind.WIDE,
                    message="횡단보도가 보이지 않아 연결 판정(C1·C3·C4) 보류. 차도 직진(C2)만 확인함")]


def check_h2(wide: PhotoAnalysis) -> list[Finding]:
    """시점 보정 실패 → C4만 보류."""
    r = wide.rectification
    if r is not None and r.ok:
        return []
    return [Finding(code="H2", verdict=Verdict.HOLD, photo=PhotoKind.WIDE,
                    message="시점 보정 실패로 횡단보도 진입 방향(C4) 판정 보류"
                            + (f" ({r.reason})" if r is not None and r.reason else ""),
                    confidence=0.0)]


def check_h4(close: PhotoAnalysis, cfg: HoldConfig) -> list[Finding]:
    """고립된 작은 블록 영역 → 비교 대상이 없어 M 계열 보류."""
    s, areas = close.segmentation, close.path.component_area_px
    if areas and max(areas.values()) >= cfg.min_block_ratio * s.width * s.height:
        return []
    return [Finding(code="H4", verdict=Verdict.HOLD, photo=PhotoKind.CLOSE,
                    message="블록 영역이 너무 작거나 없음. 블록이 화면 아래 절반을 채우게 찍어 주세요")]


def block_width_px(label: np.ndarray) -> float:
    """블록 줄의 대표 폭(px): 중심선 위 거리변환 값(=반폭)의 중앙값 × 2."""
    import cv2
    from skimage.morphology import skeletonize

    blocks = np.isin(label, [int(c) for c in BLOCK_CLASSES])
    if not blocks.any():
        return 0.0
    dt = cv2.distanceTransform(blocks.astype(np.uint8), cv2.DIST_L2, 5)
    sk = skeletonize(blocks)
    return float(2 * np.median(dt[sk])) if sk.any() else 0.0


def m_gate(p: PhotoAnalysis, label: Optional[np.ndarray], cfg: HoldConfig) -> tuple[bool, str]:
    """M 계열(모양) 판정 관문: 블록이 충분히 크고 선명한가. (통과 여부, 못 넘은 이유)"""
    s, areas = p.segmentation, p.path.component_area_px
    if not areas or max(areas.values()) < cfg.min_block_ratio * s.width * s.height:
        return False, "블록 영역이 너무 작거나 없음"
    if label is not None and block_width_px(label) < cfg.min_block_width * min(s.width, s.height):
        return False, "블록이 작게 찍혀 점형·선형 구분이 어려움"
    return True, ""


def h4_from_gate(why: str) -> list[Finding]:
    return [Finding(code="H4", verdict=Verdict.HOLD, photo=PhotoKind.CLOSE,
                    message=f"{why}. 블록이 화면을 크게 채우게 가까이 찍어 주세요")]
