"""A(모양 판정)와 B(문맥 판정)의 접점: 마스크 + 경로 구조 JSON 형식.

이 파일이 11/10 합의본이다. 형식을 바꾸면 SCHEMA_VERSION을 올리고 양쪽에 알린다.
좌표는 모두 원본 사진 픽셀 기준 (x: 오른쪽, y: 아래쪽).
"""

from __future__ import annotations

from enum import Enum, IntEnum
from typing import Literal, Optional

from pydantic import BaseModel, Field

SCHEMA_VERSION = "0.1.1"  # 0.1.1: Rectification.px_per_m, RoiAnalysis.role 추가 (C4)


# ---------------------------------------------------------------------------
# 마스크
# ---------------------------------------------------------------------------


class MaskClass(IntEnum):
    """라벨 마스크 PNG(uint8, 1채널)의 픽셀 값.

    YOLO 클래스 id는 여기서 1을 뺀 값이다 (linear=0, dot=1, crosswalk=2, road=3).
    """

    BACKGROUND = 0
    LINEAR = 1  # 선형 블록 (유도)
    DOT = 2  # 점형 블록 (경고·꺾임·분기)
    CROSSWALK = 3
    ROAD = 4  # 차도


BLOCK_CLASSES = (MaskClass.LINEAR, MaskClass.DOT)


class PhotoKind(str, Enum):
    CLOSE = "close"  # 근접: M3·M4, 선형/점형 구분
    WIDE = "wide"  # 넓은: C1~C4


class Point(BaseModel):
    x: float
    y: float


class Segmentation(BaseModel):
    """② 영역 분할 결과."""

    photo: PhotoKind
    width: int
    height: int
    mask_path: Optional[str] = Field(
        None, description="라벨 마스크 PNG 경로 (값은 MaskClass). 메모리로만 넘길 때는 None"
    )
    class_area_px: dict[str, int] = Field(
        default_factory=dict, description="클래스 이름 → 픽셀 수. H3(횡단보도·차도 없음) 판정에 사용"
    )
    mean_confidence: Optional[float] = Field(
        None, description="블록 영역의 평균 신뢰도 0~1. H1(가림·그림자) 판정에 사용"
    )


# ---------------------------------------------------------------------------
# 경로 구조 (③ 중심선 추출 결과)
# ---------------------------------------------------------------------------


class NodeType(str, Enum):
    END = "end"  # 끝점 (차수 1)
    TURN = "turn"  # 꺾임점 (차수 2, 방향 변화 큼)
    BRANCH = "branch"  # 분기점 (차수 3 이상)
    PASS = "pass"  # 고리 경로의 기준점 등, 꺾이지 않는 차수 2 노드
    ISOLATED = "isolated"  # 연결된 엣지가 없는 점


class PathNode(BaseModel):
    id: int
    type: NodeType
    pt: Point
    block: Optional[MaskClass] = Field(
        None, description="노드 주변 블록 종류 (최빈값). 넓은 사진에서는 신뢰하지 않는다"
    )
    component: int = Field(description="연결 성분 번호. 같은 번호면 이어진 블록 덩어리")


class PathEdge(BaseModel):
    id: int
    u: int
    v: int
    polyline: list[Point] = Field(description="u → v 순서의 단순화된 중심선")
    length_px: float
    block: Optional[MaskClass] = Field(None, description="엣지를 따라 가장 많은 블록 종류")


class PathStructure(BaseModel):
    photo: PhotoKind
    width: int
    height: int
    nodes: list[PathNode] = Field(default_factory=list)
    edges: list[PathEdge] = Field(default_factory=list)
    component_area_px: dict[int, int] = Field(
        default_factory=dict, description="연결 성분 번호 → 블록 마스크 면적. H4(고립 영역)에 사용"
    )

    def node(self, node_id: int) -> PathNode:
        return next(n for n in self.nodes if n.id == node_id)

    def edges_of(self, node_id: int) -> list[PathEdge]:
        return [e for e in self.edges if node_id in (e.u, e.v)]


# ---------------------------------------------------------------------------
# ④ 정밀 분석
# ---------------------------------------------------------------------------


class Rectification(BaseModel):
    """횡단보도 줄무늬 기반 시점 보정. 실패하면 C4만 보류 (H2)."""

    ok: bool
    homography: Optional[list[list[float]]] = Field(None, description="3x3, 원본 → 위에서 본 평면")
    crosswalk_axis_deg: Optional[float] = Field(
        None, description="보정 평면에서 횡단보도 진행축 각도 (0=오른쪽, 반시계 +)"
    )
    px_per_m: Optional[float] = Field(None, description="보정 평면에서 1m가 몇 단위인지 (횡단보도 줄 간격으로 추정)")
    reason: Optional[str] = None


class RoiAnalysis(BaseModel):
    """꺾임·분기점·진입부 ROI의 블록 종류와 돌출선 방향."""

    photo: PhotoKind
    role: Optional[Literal["node", "entrance", "approach"]] = Field(
        None, description="node=꺾임·분기점(M3), entrance=횡단보도 진입부, approach=진입부 직전 선형 구간(C4)"
    )
    node_id: Optional[int] = None
    center: Point
    size_px: int
    block: Optional[MaskClass] = None
    ridge_angle_deg: Optional[float] = Field(
        None, description="선형 돌출선 방향 (보정 평면 기준, 0~180). 점형이면 None"
    )
    confidence: float = 0.0


class PhotoAnalysis(BaseModel):
    """한 사진에 대한 ②~④ 결과 묶음. 규칙 판정(⑤)의 입력."""

    segmentation: Segmentation
    path: PathStructure
    rois: list[RoiAnalysis] = Field(default_factory=list)
    rectification: Optional[Rectification] = None


# ---------------------------------------------------------------------------
# ⑤ 판정 결과 (FE가 받는 형식)
# ---------------------------------------------------------------------------


class Verdict(str, Enum):
    SUSPECTED = "suspected"  # 설치 오류 의심
    NORMAL = "normal"  # 정상 추정
    HOLD = "hold"  # 판정 보류 (재촬영 요청)


Code = Literal["M3", "M4", "C1", "C2", "C3", "C4", "H1", "H2", "H3", "H4"]


class Finding(BaseModel):
    code: Code
    verdict: Verdict
    photo: PhotoKind
    message: str
    location: Optional[Point] = Field(None, description="오버레이 표시 위치")
    clause: Optional[str] = Field(None, description="근거 조항 (rules/clauses.py)")
    suggestion: Optional[str] = Field(None, description="개선 방향")
    confidence: float = 0.0
    primary: bool = Field(True, description="C2와 C3가 함께 뜨면 C3는 primary=False (보조 표시)")


class InspectionResult(BaseModel):
    schema_version: str = SCHEMA_VERSION
    inspection_id: Optional[str] = None
    overall: Verdict
    findings: list[Finding] = Field(default_factory=list)
    close: Optional[PhotoAnalysis] = None
    wide: Optional[PhotoAnalysis] = None
