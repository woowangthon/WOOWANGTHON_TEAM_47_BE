"""FastAPI 서버.

실행: uvicorn app.main:app --reload
설정은 환경변수 또는 레포 최상위 .env (예시: .env.example). 이미 설정된 환경변수가 .env보다 우선한다.
"""

from __future__ import annotations

import os
import time
import uuid
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from pipeline.schema import (
    SCHEMA_VERSION,
    InspectionResult,
    Location,
    ReportDetail,
    ReportStatus,
    ReportSummary,
    ReportUpdate,
    Verdict,
)



def _load_dotenv(path: Path = Path(__file__).resolve().parents[1] / ".env") -> None:
    """.env의 KEY=VALUE를 환경변수로 (python-dotenv 없이). 이미 있는 값은 덮어쓰지 않는다."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        v = v.split(" #", 1)[0].strip().strip('"').strip("'")
        if v:
            os.environ.setdefault(k.strip(), v)


_load_dotenv()

from . import storage  # noqa: E402  (.env를 읽은 뒤에 불러옴)

state: dict = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    # 두 모델 통합 (블록 + 장면). 둘 다 있으면 DualSegmenter, 아니면 예전 방식(모델 하나)
    block = Path(os.getenv("BLOCK_WEIGHTS", "weights/v4_best.pt"))
    scene = Path(os.getenv("SCENE_WEIGHTS", "weights/scene_v1_best.pt"))
    legacy = Path(os.getenv("WEIGHTS_PATH", "weights/best.pt"))
    if block.exists() and scene.exists():
        from pipeline.dual_segment import DualSegmenter

        state["segmenter"] = DualSegmenter(block, scene)
    elif legacy.exists():
        from pipeline.segment import Segmenter

        state["segmenter"] = Segmenter(legacy)
    yield
    state.clear()


app = FastAPI(title="TactileCheck API", version=SCHEMA_VERSION, lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:5173").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)


MAX_UPLOAD_MB = float(os.getenv("MAX_UPLOAD_MB", "15"))
RATE_LIMIT_PER_MIN = int(os.getenv("RATE_LIMIT_PER_MIN", "10"))
_hits: dict[str, deque] = defaultdict(deque)


def _client_ip(request: Request) -> str:
    fwd = request.headers.get("x-forwarded-for")  # 배포 서버(프록시) 뒤에서는 실제 IP가 여기 있음
    return fwd.split(",")[0].strip() if fwd else (request.client.host if request.client else "unknown")


def _rate_limit(request: Request) -> None:
    """IP당 분당 점검 요청 수 제한 (서버 메모리 기준, 서버가 하나일 때 유효)."""
    if RATE_LIMIT_PER_MIN <= 0:
        return
    now, q = time.time(), _hits[_client_ip(request)]
    while q and now - q[0] > 60:
        q.popleft()
    if len(q) >= RATE_LIMIT_PER_MIN:
        raise HTTPException(429, "요청이 너무 많아요. 1분 뒤에 다시 시도해 주세요")
    q.append(now)


def _require_admin(x_admin_token: Optional[str] = Header(None)) -> None:
    token = os.getenv("ADMIN_TOKEN")
    if not token:
        raise HTTPException(503, "관리자 기능이 설정되지 않음 (ADMIN_TOKEN)")
    if x_admin_token != token:
        raise HTTPException(401, "관리자 토큰이 필요함 (X-Admin-Token 헤더)")


@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "schema_version": SCHEMA_VERSION, "model_loaded": "segmenter" in state,
            "storage": storage.backend()}


@app.get("/api/health/deep")
def health_deep() -> dict:
    """모델과 저장소 연결까지 확인. 모니터링(UptimeRobot 등)이 주기적으로 불러 서버·Supabase를 깨워 둔다."""
    try:
        store = storage.ping()
    except Exception as e:
        raise HTTPException(503, f"저장소 연결 실패: {type(e).__name__}")
    if "segmenter" not in state:
        raise HTTPException(503, "모델이 로드되지 않음")
    return {"ok": True, "schema_version": SCHEMA_VERSION, "model_loaded": True, "storage": store}


async def _read_image(f: UploadFile) -> np.ndarray:
    data = await f.read()
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"사진이 너무 커요 (최대 {MAX_UPLOAD_MB:g}MB)")
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, f"이미지를 읽을 수 없음: {f.filename}")
    return img


def _blank(v: Optional[str]) -> Optional[str]:
    return v.strip() or None if v is not None else None


@app.post("/api/inspections", response_model=InspectionResult, dependencies=[Depends(_rate_limit)])
async def create_inspection(
    photo: Optional[UploadFile] = File(None, description="점검할 사진 한 장 (구도·거리 제한 없음)"),
    address: Optional[str] = Form(None, description="주소 또는 장소 이름"),
    memo: Optional[str] = Form(None, description="장소 설명 메모"),
    lat: Optional[float] = Form(None, description="위도 (브라우저 위치 권한)"),
    lng: Optional[float] = Form(None, description="경도"),
    close: Optional[UploadFile] = File(None, description="(호환용) 예전 근접 사진 칸"),
    wide: Optional[UploadFile] = File(None, description="(호환용) 예전 넓은 사진 칸"),
) -> InspectionResult:
    """사진 한 장(+ 선택 위치)을 받아 판정하고, 사진·위치·결과를 함께 저장한다 (처리 상태: 접수)."""
    upload = photo or wide or close
    if upload is None:
        raise HTTPException(400, "사진이 필요함 (photo)")
    if "segmenter" not in state:
        raise HTTPException(503, "분할 모델 가중치가 없음 (BLOCK_WEIGHTS·SCENE_WEIGHTS 확인)")
    try:
        location = Location(address=_blank(address), memo=_blank(memo), lat=lat, lng=lng)
    except ValueError as e:
        raise HTTPException(422, f"위치 값이 올바르지 않음: {e}")
    if not any(v is not None for v in location.model_dump().values()):
        location = None

    from pipeline.run import run_photo

    image = await _read_image(upload)
    result = run_photo(state["segmenter"], image, str(uuid.uuid4()))
    return storage.save_report(result, image, location)


@app.get("/api/inspections/{inspection_id}", response_model=InspectionResult)
def get_inspection(inspection_id: str) -> InspectionResult:
    result = storage.load_result(inspection_id)
    if result is None:
        raise HTTPException(404, "점검 건을 찾을 수 없음")
    return result


@app.get("/api/inspections/{inspection_id}/photo", responses={200: {"content": {"image/jpeg": {}}}})
def get_photo(inspection_id: str) -> Response:
    """저장된 사진 (긴 변 1600px JPEG). 판정 위치 좌표는 원본 사진 기준이라
    화면에서는 photo.segmentation.width/height 대비 비율로 바꿔 쓴다."""
    jpg = storage.load_photo(inspection_id)
    if jpg is None:
        raise HTTPException(404, "사진을 찾을 수 없음")
    return Response(jpg, media_type="image/jpeg", headers={"Cache-Control": "private, max-age=3600"})


# ---------------------------------------------------------------- 관리자 (X-Admin-Token 헤더 필요)


@app.get("/api/admin/inspections", response_model=list[ReportSummary], dependencies=[Depends(_require_admin)])
def admin_list(status: Optional[ReportStatus] = None, overall: Optional[Verdict] = None,
               limit: int = 50, offset: int = 0) -> list[ReportSummary]:
    """신고 목록 (최근 접수 순). status·overall로 거르기."""
    return storage.list_reports(status, overall, max(1, min(limit, 200)), max(0, offset))


@app.get("/api/admin/inspections/{inspection_id}", response_model=ReportDetail,
         dependencies=[Depends(_require_admin)])
def admin_get(inspection_id: str) -> ReportDetail:
    r = storage.get_report(inspection_id)
    if r is None:
        raise HTTPException(404, "점검 건을 찾을 수 없음")
    return r


@app.patch("/api/admin/inspections/{inspection_id}", response_model=ReportDetail,
           dependencies=[Depends(_require_admin)])
def admin_update(inspection_id: str, upd: ReportUpdate) -> ReportDetail:
    """처리 상태·관리자 메모 변경."""
    try:
        r = storage.update_report(inspection_id, upd)
    except Exception as e:
        raise HTTPException(502, f"저장소 업데이트 실패: {type(e).__name__}")
    if r is None:
        raise HTTPException(404, "점검 건을 찾을 수 없음")
    return r
