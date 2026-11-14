"""점검(신고) 저장: 판정 결과 + 사진 + 위치 + 관리자 처리 상태.

SUPABASE_URL·SUPABASE_KEY가 있으면
    - DB: Supabase `inspections` 테이블 (만들기·변경: docs/supabase.sql)
    - 사진: Supabase Storage 비공개 버킷 (SUPABASE_BUCKET, 기본 inspection-photos)
없으면 서버 메모리에만 둔다 (재시작하면 사라짐, 개발·테스트용).

- 키는 서버 전용 secret(service_role) 키. .env·배포 서버 Secrets에만 두고 커밋하지 않는다.
- 사진은 긴 변 1600px JPEG로 줄여 저장한다 (관리자 확인용). 얼굴·차량 번호가 찍힐 수 있으니
  버킷은 비공개로 두고, 사진은 API(서버)를 통해서만 내보낸다.
- 시민 쪽 저장(점검 접수)은 Supabase가 잠깐 안 돼도 실패하지 않는다. 경고만 남기고 메모리에 둔다.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Optional

import cv2
import numpy as np

from pipeline.schema import (
    InspectionResult,
    Location,
    ReportDetail,
    ReportStatus,
    ReportSummary,
    ReportUpdate,
    Verdict,
)

log = logging.getLogger("tactilecheck.storage")

PHOTO_LONG_SIDE = 1600
PHOTO_QUALITY = 85
PHOTO_CACHE_MAX = 100

_memory: dict[str, dict] = {}  # id → DB 행과 같은 모양
_photos: dict[str, bytes] = {}  # id → JPEG (Supabase 없을 때만)
_cached_client = None


def _table() -> str:
    return os.getenv("SUPABASE_TABLE", "inspections")


def _bucket() -> str:
    return os.getenv("SUPABASE_BUCKET", "inspection-photos")


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


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def photo_url(inspection_id: str) -> str:
    return f"/api/inspections/{inspection_id}/photo"


# ---------------------------------------------------------------- 저장 (시민 접수)


def encode_photo(image_bgr: np.ndarray) -> bytes:
    h, w = image_bgr.shape[:2]
    s = min(1.0, PHOTO_LONG_SIDE / max(h, w))
    if s < 1.0:
        image_bgr = cv2.resize(image_bgr, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, PHOTO_QUALITY])
    if not ok:
        raise ValueError("사진 인코딩 실패")
    return buf.tobytes()


def _codes(result: InspectionResult) -> list[str]:
    return sorted({f.code for f in result.findings if f.verdict == Verdict.SUSPECTED})


def save_report(result: InspectionResult, image_bgr: Optional[np.ndarray] = None,
                location: Optional[Location] = None) -> InspectionResult:
    """판정 결과 + 사진 + 위치를 저장하고, photo_url·location·created_at을 채운 결과를 돌려준다."""
    rid = result.inspection_id
    result.created_at = result.created_at or _now()
    result.location = location
    photo_path = None
    if image_bgr is not None:
        jpg = encode_photo(image_bgr)
        photo_path = f"{rid}.jpg"
        client = _client()
        if client is not None:
            try:
                client.storage.from_(_bucket()).upload(
                    photo_path, jpg, {"content-type": "image/jpeg", "upsert": "true"})
            except Exception as e:  # 업로드 실패해도 서버가 살아 있는 동안은 메모리 사진으로 보여 준다
                log.warning("사진 업로드 실패 (%s): %s", rid, e)
        _cache_photo(rid, jpg)  # 접수 직후 결과 화면이 바로 사진을 부르므로 메모리에도 둔다
        result.photo_url = photo_url(rid)

    loc = location or Location()
    row = {
        "id": rid,
        "created_at": result.created_at,
        "overall": result.overall.value,
        "schema_version": result.schema_version,
        "result": result.model_dump(mode="json"),
        "photo_path": photo_path,
        "status": ReportStatus.RECEIVED.value,
        "codes": _codes(result),
        "address": loc.address,
        "place_memo": loc.memo,
        "lat": loc.lat,
        "lng": loc.lng,
        "admin_memo": None,
        "updated_at": result.created_at,
    }
    _memory[rid] = row
    client = _client()
    if client is not None:
        try:
            client.table(_table()).upsert(row).execute()
        except Exception as e:  # 저장 실패해도 시민에게는 판정 결과를 돌려준다
            log.warning("Supabase 저장 실패 (%s): %s", rid, e)
    return result


def save_result(result: InspectionResult) -> None:
    """(호환용) 사진·위치 없이 결과만 저장."""
    save_report(result)


# ---------------------------------------------------------------- 조회


def _get_row(rid: str) -> Optional[dict]:
    if rid in _memory:
        return _memory[rid]
    client = _client()
    if client is None:
        return None
    try:
        rows = client.table(_table()).select("*").eq("id", rid).limit(1).execute().data
    except Exception as e:
        log.warning("Supabase 조회 실패 (%s): %s", rid, e)
        return None
    if rows:
        _memory[rid] = rows[0]
        return rows[0]
    return None


def _location(row: dict) -> Optional[Location]:
    loc = Location(address=row.get("address"), memo=row.get("place_memo"), lat=row.get("lat"), lng=row.get("lng"))
    return loc if any(v is not None for v in loc.model_dump().values()) else None


def _result(row: dict) -> InspectionResult:
    r = InspectionResult.model_validate(row["result"])
    r.location = _location(row)
    r.photo_url = photo_url(row["id"]) if row.get("photo_path") else None
    r.created_at = row.get("created_at") or r.created_at
    return r


def load_result(rid: str) -> Optional[InspectionResult]:
    row = _get_row(rid)
    return _result(row) if row else None


def load_photo(rid: str) -> Optional[bytes]:
    if rid in _photos:
        return _photos[rid]
    row = _get_row(rid)
    client = _client()
    if not row or not row.get("photo_path") or client is None:
        return None
    try:
        jpg = client.storage.from_(_bucket()).download(row["photo_path"])
    except Exception as e:
        log.warning("사진 다운로드 실패 (%s): %s", rid, e)
        return None
    _cache_photo(rid, jpg)
    return jpg


def _cache_photo(rid: str, jpg: bytes) -> None:
    """서버와 Supabase가 멀어(시드니↔아시아) 다운로드가 느리므로 최근 사진은 메모리에 보관 (약 100장 ≈ 10MB).
    Supabase가 없으면 이게 유일한 사진 저장소라 지우지 않는다."""
    _photos[rid] = jpg
    if _client() is None:
        return
    while len(_photos) > PHOTO_CACHE_MAX:
        _photos.pop(next(iter(_photos)))


# ---------------------------------------------------------------- 관리자


def _summary(row: dict) -> ReportSummary:
    return ReportSummary(
        inspection_id=row["id"], created_at=row.get("created_at"), overall=row["overall"],
        status=row.get("status") or ReportStatus.RECEIVED, codes=row.get("codes") or [],
        location=_location(row), photo_url=photo_url(row["id"]) if row.get("photo_path") else None,
    )


def list_reports(status: Optional[ReportStatus] = None, overall: Optional[Verdict] = None,
                 limit: int = 50, offset: int = 0) -> list[ReportSummary]:
    """최근 접수 순. Supabase가 있으면 Supabase가 기준 (서버 재시작 후에도 전체 목록)."""
    client = _client()
    if client is None:
        rows = sorted(_memory.values(), key=lambda r: r.get("created_at") or "", reverse=True)
        if status:
            rows = [r for r in rows if r.get("status") == status.value]
        if overall:
            rows = [r for r in rows if r.get("overall") == overall.value]
        return [_summary(r) for r in rows[offset:offset + limit]]
    cols = "id,created_at,overall,status,codes,address,place_memo,lat,lng,photo_path"
    q = client.table(_table()).select(cols)
    if status:
        q = q.eq("status", status.value)
    if overall:
        q = q.eq("overall", overall.value)
    rows = q.order("created_at", desc=True).range(offset, offset + limit - 1).execute().data
    return [_summary(r) for r in rows]


def get_report(rid: str) -> Optional[ReportDetail]:
    row = _get_row(rid)
    if not row:
        return None
    return ReportDetail(**_summary(row).model_dump(), admin_memo=row.get("admin_memo"),
                        updated_at=row.get("updated_at"), result=_result(row))


class ReportClosed(Exception):
    """관리자가 이미 처리를 시작한 신고라 시민이 위치·내용을 바꿀 수 없음."""


def submit_report(rid: str, loc: Location) -> Optional[ReportSummary]:
    """시민 신고 접수: 위치·내용을 저장한다. 처리 상태가 '접수'일 때만 (관리자가 보기 시작하면 잠김).
    Supabase 저장이 실패하면 예외를 그대로 올린다 (시민에게 '접수됨'을 잘못 보여 주면 안 됨)."""
    row = _get_row(rid)
    if not row:
        return None
    if (row.get("status") or ReportStatus.RECEIVED.value) != ReportStatus.RECEIVED.value:
        raise ReportClosed(rid)
    patch = {"address": loc.address, "place_memo": loc.memo, "lat": loc.lat, "lng": loc.lng, "updated_at": _now()}
    client = _client()
    if client is not None:
        client.table(_table()).update(patch).eq("id", rid).execute()
    row.update(patch)
    return _summary(row)


def update_report(rid: str, upd: ReportUpdate) -> Optional[ReportDetail]:
    row = _get_row(rid)
    if not row:
        return None
    patch = {"updated_at": _now()}
    if upd.status is not None:
        patch["status"] = upd.status.value
    if upd.admin_memo is not None:
        patch["admin_memo"] = upd.admin_memo
    client = _client()
    if client is not None:  # 관리자 변경은 실패하면 에러로 알린다 (조용히 잃어버리면 안 됨)
        client.table(_table()).update(patch).eq("id", rid).execute()
    row.update(patch)
    return get_report(rid)


# ---------------------------------------------------------------- 점검 (깨우기용)


def ping() -> str:
    """저장소 연결 확인. Supabase에 가벼운 조회를 해서 '활동 중'으로 유지하는 역할도 한다."""
    client = _client()
    if client is None:
        return "memory"
    client.table(_table()).select("id").limit(1).execute()
    return "supabase"
