"""점검 결과 저장. SUPABASE_URL/KEY가 없으면 메모리에만 둔다. (담당: A)

TODO(A): Supabase 테이블 `inspections (id uuid pk, created_at, overall text, result jsonb)`
와 사진 업로드용 Storage 버킷 연결.
"""

from __future__ import annotations

import os
from typing import Optional

from pipeline.schema import InspectionResult

_memory: dict[str, InspectionResult] = {}


def _client():
    url, key = os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY")
    if not (url and key):
        return None
    from supabase import create_client

    return create_client(url, key)


def save_result(result: InspectionResult) -> None:
    _memory[result.inspection_id] = result
    client = _client()
    if client is not None:
        client.table("inspections").insert(
            {"id": result.inspection_id, "overall": result.overall.value, "result": result.model_dump(mode="json")}
        ).execute()


def load_result(inspection_id: str) -> Optional[InspectionResult]:
    if inspection_id in _memory:
        return _memory[inspection_id]
    client = _client()
    if client is None:
        return None
    rows = client.table("inspections").select("result").eq("id", inspection_id).execute().data
    return InspectionResult.model_validate(rows[0]["result"]) if rows else None
