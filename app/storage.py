"""점검 결과 저장.

SUPABASE_URL·SUPABASE_KEY가 있으면 Supabase `inspections` 테이블에도 저장하고 조회한다
(테이블 만들기: docs/supabase.sql). 없으면 서버 메모리에만 둔다 (재시작하면 사라짐).

- 키는 서버 전용 service_role 키를 쓴다. .env에만 두고 커밋하지 않는다 (.gitignore에 등록됨).
- Supabase가 잠깐 안 돼도 점검 API는 실패하지 않는다. 경고만 남기고 메모리 결과를 쓴다.
- 원본 사진은 저장하지 않는다 (얼굴·차량 번호가 찍힐 수 있음). 판정 결과 JSON만 저장한다.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from pipeline.schema import InspectionResult

log = logging.getLogger("tactilecheck.storage")

_memory: dict[str, InspectionResult] = {}
_cached_client = None


def _table() -> str:
    return os.getenv("SUPABASE_TABLE", "inspections")


def backend() -> str:
    """지금 쓰는 저장소: "supabase" 또는 "memory"."""
    return "supabase" if os.getenv("SUPABASE_URL") and os.getenv("SUPABASE_KEY") else "memory"


def _client():
    global _cached_client
    if backend() != "supabase":
        return None
    if _cached_client is None:
        from supabase import create_client

        _cached_client = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])
    return _cached_client


def save_result(result: InspectionResult) -> None:
    _memory[result.inspection_id] = result
    client = _client()
    if client is None:
        return
    try:
        client.table(_table()).upsert({
            "id": result.inspection_id,
            "overall": result.overall.value,
            "schema_version": result.schema_version,
            "result": result.model_dump(mode="json"),
        }).execute()
    except Exception as e:  # 저장 실패해도 점검 결과는 돌려준다
        log.warning("Supabase 저장 실패 (%s): %s", result.inspection_id, e)


def load_result(inspection_id: str) -> Optional[InspectionResult]:
    if inspection_id in _memory:
        return _memory[inspection_id]
    client = _client()
    if client is None:
        return None
    try:
        rows = client.table(_table()).select("result").eq("id", inspection_id).limit(1).execute().data
    except Exception as e:
        log.warning("Supabase 조회 실패 (%s): %s", inspection_id, e)
        return None
    if not rows:
        return None
    result = InspectionResult.model_validate(rows[0]["result"])
    _memory[inspection_id] = result
    return result
