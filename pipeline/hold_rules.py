"""Conservative H1/H4 hold-candidate rules for TactileCheck.

Pure post-analysis function. Does not mutate masks, paths, or existing M3 findings.
H1 cannot establish shadows/occlusion from a class label mask alone: it flags
missing/low-confidence predictions, not the physical reason for uncertainty.
H4 flags only *small* isolated structures, not every standalone DOT tile.
"""
from __future__ import annotations

from dataclasses import dataclass

from .schema import Finding, MaskClass, NodeType, PhotoAnalysis, Verdict


@dataclass(frozen=True)
class HoldConfig:
    min_mean_confidence: float = 0.45
    # Only flag isolated nodes whose entire connected component is small.
    small_component_area_px: int = 1200
    small_component_image_ratio: float = 0.0002

    def __post_init__(self) -> None:
        if not 0 <= self.min_mean_confidence <= 1:
            raise ValueError("min_mean_confidence must be between 0 and 1")
        if self.small_component_area_px < 0 or self.small_component_image_ratio < 0:
            raise ValueError("component size thresholds must be nonnegative")


def check_h1_h4(
    analysis: PhotoAnalysis, cfg: HoldConfig | None = None
) -> list[Finding]:
    """Return local hold findings; never overwrite M3/C findings or overall verdict.

    Caller is responsible for merging and selecting overall verdict.
    Call on CLOSE photo for the M-side rules in the initial MVP.
    """
    cfg = cfg or HoldConfig()
    seg, path = analysis.segmentation, analysis.path
    if (seg.width, seg.height) != (path.width, path.height):
        raise ValueError("segmentation/path image dimensions do not match")
    if seg.photo != path.photo:
        raise ValueError("segmentation/path photo kinds do not match")

    findings: list[Finding] = []
    conf = seg.mean_confidence
    if conf is not None and not 0 <= conf <= 1:
        raise ValueError("mean_confidence must be between 0 and 1")

    tactile_px = sum(
        max(0, int(seg.class_area_px.get(cls.name.lower(), 0)))
        for cls in (MaskClass.LINEAR, MaskClass.DOT)
    )
    # If class_area_px was not populated, do not infer no-detection from empty metadata.
    has_class_data = any(cls.name.lower() in seg.class_area_px for cls in (MaskClass.LINEAR, MaskClass.DOT))
    missing = has_class_data and tactile_px == 0
    low_conf = conf is not None and conf < cfg.min_mean_confidence
    if missing or low_conf:
        reason = (
            "점자블록이 검출되지 않아 설치 상태를 판단할 수 없습니다."
            if missing else
            f"점자블록 평균 검출 신뢰도가 낮아({conf:.2f}) 판정을 보류합니다."
        )
        findings.append(Finding(
            code="H1", verdict=Verdict.HOLD, photo=seg.photo,
            message=reason,
            suggestion="점자블록이 화면에 충분히 보이도록 가까이서 다시 촬영해 주세요.",
            confidence=0.0 if conf is None else conf,
        ))

    min_area = max(
        cfg.small_component_area_px,
        int(seg.width * seg.height * cfg.small_component_image_ratio),
    )
    seen: set[int] = set()
    for node in path.nodes:
        if node.type != NodeType.ISOLATED or node.block not in (MaskClass.LINEAR, MaskClass.DOT):
            continue
        component = node.component
        if component in seen:
            continue
        area = path.component_area_px.get(component)
        # Missing component metadata is unknown, not proof that region is small.
        if area is None or not 0 < area < min_area:
            continue
        seen.add(component)
        findings.append(Finding(
            code="H4", verdict=Verdict.HOLD, photo=seg.photo,
            location=node.pt,
            message=f"작게 분리된 점자블록 후보({area}px²)가 있어 연결 상태 판단을 보류합니다.",
            suggestion="해당 블록과 주변 유도블록이 함께 보이도록 다시 촬영해 주세요.",
            confidence=0.0 if conf is None else conf,
        ))
    return findings
