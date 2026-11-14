"""FastAPI 서버. (담당: A)

실행: uvicorn app.main:app --reload
"""

from __future__ import annotations

import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from pipeline.schema import SCHEMA_VERSION, InspectionResult

from . import storage

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


@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "schema_version": SCHEMA_VERSION, "model_loaded": "segmenter" in state}


async def _decode(f: Optional[UploadFile]) -> Optional[np.ndarray]:
    if f is None:
        return None
    img = cv2.imdecode(np.frombuffer(await f.read(), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, f"이미지를 읽을 수 없음: {f.filename}")
    return img


@app.post("/api/inspections", response_model=InspectionResult)
async def create_inspection(
    close: Optional[UploadFile] = File(None, description="근접 사진"),
    wide: Optional[UploadFile] = File(None, description="넓은 사진"),
) -> InspectionResult:
    if close is None and wide is None:
        raise HTTPException(400, "근접 사진과 넓은 사진 중 하나 이상 필요")
    if "segmenter" not in state:
        raise HTTPException(503, "분할 모델 가중치가 없음 (BLOCK_WEIGHTS·SCENE_WEIGHTS 확인)")

    from pipeline.run import run_inspection

    result = run_inspection(state["segmenter"], await _decode(close), await _decode(wide), str(uuid.uuid4()))
    storage.save_result(result)
    return result


@app.get("/api/inspections/{inspection_id}", response_model=InspectionResult)
def get_inspection(inspection_id: str) -> InspectionResult:
    result = storage.load_result(inspection_id)
    if result is None:
        raise HTTPException(404, "점검 건을 찾을 수 없음")
    return result
