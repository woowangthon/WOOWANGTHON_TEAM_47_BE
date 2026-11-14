"""저장소: Supabase 키가 없으면 메모리, 있으면 Supabase DB·Storage. 진짜 Supabase 대신 가짜 클라이언트를 쓴다."""

import numpy as np
import pytest

from app import storage
from pipeline.schema import InspectionResult, Location, ReportStatus, ReportUpdate, Verdict


class _FakeQuery:
    def __init__(self, db, table):
        self.db, self.table, self.op, self.payload, self.filters, self.rng = db, table, None, None, {}, None

    def upsert(self, row):
        self.op, self.payload = "upsert", row
        return self

    def update(self, patch):
        self.op, self.payload = "update", patch
        return self

    def select(self, cols):
        self.op = self.op or "select"
        return self

    def eq(self, col, val):
        self.filters[col] = val
        return self

    def limit(self, n):
        self.rng = (0, n - 1)
        return self

    def order(self, col, desc=False):
        return self

    def range(self, a, b):
        self.rng = (a, b)
        return self

    def execute(self):
        if self.db.get("fail"):
            raise RuntimeError("network down")
        rows = self.db.setdefault(self.table, {})
        if self.op == "upsert":
            rows[self.payload["id"]] = dict(self.payload)
            return type("R", (), {"data": [self.payload]})
        hit = [r for r in rows.values() if all(r.get(k) == v for k, v in self.filters.items())]
        if self.op == "update":
            for r in hit:
                r.update(self.payload)
            return type("R", (), {"data": hit})
        hit.sort(key=lambda r: r.get("created_at") or "", reverse=True)
        if self.rng:
            hit = hit[self.rng[0]:self.rng[1] + 1]
        return type("R", (), {"data": hit})


class _FakeBucket:
    def __init__(self, files):
        self.files = files

    def upload(self, path, data, opts=None):
        self.files[path] = data

    def download(self, path):
        return self.files[path]


class _FakeStorage:
    def __init__(self):
        self.files = {}

    def from_(self, bucket):
        return _FakeBucket(self.files)


class _FakeClient:
    def __init__(self):
        self.db = {}
        self.storage = _FakeStorage()

    def table(self, name):
        return _FakeQuery(self.db, name)


@pytest.fixture
def supa(monkeypatch):
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_KEY", "test")
    fake = _FakeClient()
    monkeypatch.setattr(storage, "_cached_client", fake)
    return fake


def _result(i="00000000-0000-0000-0000-000000000001", overall=Verdict.HOLD):
    return InspectionResult(inspection_id=i, overall=overall, scope={"C": "판정"})


IMG = np.full((2000, 1500, 3), 120, np.uint8)


def test_memory_only_without_keys():
    assert storage.backend() == "memory"
    r = storage.save_report(_result(), IMG, Location(address="청주시 우암동"))
    assert r.photo_url.endswith("/photo") and r.location.address == "청주시 우암동"
    assert storage.load_result(r.inspection_id).overall == Verdict.HOLD
    assert storage.load_photo(r.inspection_id)[:2] == b"\xff\xd8"  # JPEG


def test_photo_is_resized():
    import cv2

    jpg = storage.encode_photo(np.zeros((5712, 4284, 3), np.uint8))
    img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
    assert max(img.shape[:2]) == storage.PHOTO_LONG_SIDE


def test_saves_and_loads_from_supabase(supa):
    r = storage.save_report(_result(), IMG, Location(address="A", lat=36.6, lng=127.5))
    row = supa.db["inspections"][r.inspection_id]
    assert row["overall"] == "hold" and row["status"] == "received" and row["photo_path"] == f"{r.inspection_id}.jpg"
    assert row["lat"] == 36.6 and supa.storage.files[row["photo_path"]][:2] == b"\xff\xd8"
    storage._memory.clear()  # 서버 재시작 흉내: 메모리가 비어도 Supabase에서 읽음
    got = storage.load_result(r.inspection_id)
    assert got.scope == {"C": "판정"} and got.location.address == "A" and got.photo_url
    storage._memory.clear()
    assert storage.load_photo(r.inspection_id)[:2] == b"\xff\xd8"


def test_admin_list_get_update(supa):
    a = storage.save_report(_result("00000000-0000-0000-0000-00000000000a", Verdict.SUSPECTED))
    a.created_at = None
    storage.save_report(_result("00000000-0000-0000-0000-00000000000b"))
    assert len(storage.list_reports()) == 2
    assert [s.inspection_id for s in storage.list_reports(overall=Verdict.SUSPECTED)] == [a.inspection_id]
    d = storage.update_report(a.inspection_id, ReportUpdate(status=ReportStatus.REVIEWING, admin_memo="현장 확인"))
    assert d.status == ReportStatus.REVIEWING and d.admin_memo == "현장 확인"
    assert supa.db["inspections"][a.inspection_id]["status"] == "reviewing"
    assert [s.inspection_id for s in storage.list_reports(status=ReportStatus.REVIEWING)] == [a.inspection_id]


def test_supabase_failure_does_not_break_report(supa):
    supa.db["fail"] = True
    r = storage.save_report(_result(), IMG)  # 예외가 나지 않아야 함
    assert storage.load_result(r.inspection_id) is not None  # 메모리에 있음
    storage._memory.clear()
    assert storage.load_result(r.inspection_id) is None


def test_ping(supa):
    assert storage.ping() == "supabase"
    supa.db["fail"] = True
    with pytest.raises(RuntimeError):
        storage.ping()
