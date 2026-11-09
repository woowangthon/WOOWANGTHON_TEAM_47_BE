"""M 계열: 근접 사진 모양 판정. (담당: A)"""

from __future__ import annotations

from ..schema import Finding, MaskClass, NodeType, PhotoAnalysis, PhotoKind, Verdict
from .clauses import CLAUSES, SUGGESTIONS


def check_m3(close: PhotoAnalysis) -> list[Finding]:
    """M3: 중심선의 꺾임·분기점 위치가 점형인가."""
    targets = [n for n in close.path.nodes if n.type in (NodeType.TURN, NodeType.BRANCH) and n.block is not None]
    if not targets:
        return []
    bad = [n for n in targets if n.block != MaskClass.DOT]
    if not bad:
        return [Finding(code="M3", verdict=Verdict.NORMAL, photo=PhotoKind.CLOSE,
                        message="꺾임·분기점이 점형으로 설치됨", confidence=0.7)]
    return [
        Finding(
            code="M3",
            verdict=Verdict.SUSPECTED,
            photo=PhotoKind.CLOSE,
            message=f"{'분기점' if n.type == NodeType.BRANCH else '꺾임점'}이 점형이 아님",
            location=n.pt,
            clause=CLAUSES["M3"],
            suggestion=SUGGESTIONS["M3"],
            confidence=0.7,  # TODO(A): ROI 분류 신뢰도 반영
        )
        for n in bad
    ]


def check_m4(close: PhotoAnalysis) -> list[Finding]:
    """M4: 선형 경로 중간 끊김 ("끊김 의심").

    TODO(A): 같은 방향 선형 덩어리(component) 사이 끝점 쌍을 찾고,
    두 끝점 방향이 서로 마주 보며 거리가 임계값 이하이면 끊김 의심.
    """
    return []
