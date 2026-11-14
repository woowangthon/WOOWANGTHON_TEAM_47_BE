"""A(모양 판정)와 B(문맥 판정)의 접점: 마스크 + 경로 구조 JSON 형식.

이 파일이 11/10 합의본이다. 형식을 바꾸면 SCHEMA_VERSION을 올리고 양쪽에 알린다.
좌표는 모두 원본 사진 픽셀 기준 (x: 오른쪽, y: 아래쪽).
"""

from __future__ import annotations

from enum import Enum, IntEnum
from typing import Literal, Optional

from pydantic import BaseModel, Field

SCHEMA_VERSION = "0.3.2"  # 0.3.2: 시민 신고 접수 ReportSubmit, Location.memo 1000자 / 0.3.1: Segmentation.unknown_block_area_px (이중 검출) / 0.1.1: px_per_m·role (C4) / 0.1.2: C5·C6, walk_vp / 0.2.0: 사진 한 장 (photo, scope) / 0.3.0: 신고 위치·사진·처리 상태


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
    unknown_block_area_px: Optional[int] = Field(
        None, description="장면 모델은 점자블록이라고 했지만 블록 모델이 못 찾은 '종류 모름 블록' 면적. "
        "이중 검출(5클래스 장면 모델)일 때만 채움, 아니면 None. 0보다 크면 C4·C5 의심을 보류로 바꿈 (0.3.1)"
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
    walk_vp: Optional[list[float]] = Field(
        None, description="원본 사진에서 '횡단보도 건너는 방향'의 소실점 (동차좌표 x,y,w). "
        "카메라가 기울지 않았다고 가정한 대략값으로, 보정이 실패해도 흰 줄이 3개 이상이면 채움 (C6)"
    )
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


Code = Literal["M3", "M4", "C1", "C2", "C3", "C4", "C5", "C6", "H1", "H2", "H3", "H4"]


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


class Location(BaseModel):
    """신고 위치. 모두 선택 입력."""

    address: Optional[str] = Field(None, max_length=200, description="주소 또는 장소 이름 (사용자 입력)")
    memo: Optional[str] = Field(None, max_length=1000, description="장소 설명·신고 내용 메모 (예: OO초 앞 횡단보도)")
    lat: Optional[float] = Field(None, ge=-90, le=90, description="위도 (브라우저 위치 권한으로 받은 값)")
    lng: Optional[float] = Field(None, ge=-180, le=180, description="경도")


class ReportStatus(str, Enum):
    """관리자 처리 상태. 화면 표시: 접수 / 확인 중 / 처리 완료 / 반려."""

    RECEIVED = "received"
    REVIEWING = "reviewing"
    RESOLVED = "resolved"
    REJECTED = "rejected"


class InspectionResult(BaseModel):
    schema_version: str = SCHEMA_VERSION
    inspection_id: Optional[str] = None
    overall: Verdict
    findings: list[Finding] = Field(default_factory=list)
    photo: Optional[PhotoAnalysis] = Field(None, description="사진 한 장 구조의 분석 결과 (0.2.0부터 기본)")
    scope: dict[str, str] = Field(
        default_factory=dict,
        description='계열별 판정 여부와 이유. 예: {"C": "판정", "M": "건너뜀 (블록이 작게 찍혀 ...)"}',
    )
    location: Optional[Location] = Field(None, description="신고 위치 (0.3.0)")
    photo_url: Optional[str] = Field(None, description="저장된 사진 주소 (API 상대 경로, 0.3.0)")
    created_at: Optional[str] = Field(None, description="접수 시각 ISO 8601 (0.3.0)")
    close: Optional[PhotoAnalysis] = Field(None, description="예전 사진 두 장 구조 (호환용)")
    wide: Optional[PhotoAnalysis] = Field(None, description="예전 사진 두 장 구조 (호환용)")


class ReportSummary(BaseModel):
    """관리자 목록 한 줄."""

    inspection_id: str
    created_at: Optional[str] = None
    overall: Verdict
    status: ReportStatus = ReportStatus.RECEIVED
    codes: list[str] = Field(default_factory=list, description="'오류 의심' 판정 코드 (중복 제거)")
    location: Optional[Location] = None
    photo_url: Optional[str] = None


class ReportDetail(ReportSummary):
    """관리자 상세: 판정 결과 전체 + 처리 정보."""

    admin_memo: Optional[str] = None
    updated_at: Optional[str] = None
    result: InspectionResult


class ReportSubmit(BaseModel):
    """시민 신고 접수 (결과 화면 → 신고 화면). 점검 때 저장한 위치를 이 값으로 바꾼다."""

    address: str = Field(..., min_length=1, max_length=200, description="발견 위치 (주소·장소 이름)")
    memo: Optional[str] = Field(None, max_length=1000, description="확인이 필요한 내용")
    lat: Optional[float] = Field(None, ge=-90, le=90)
    lng: Optional[float] = Field(None, ge=-180, le=180)


class ReportUpdate(BaseModel):
    status: Optional[ReportStatus] = None
    admin_memo: Optional[str] = Field(None, max_length=1000)
