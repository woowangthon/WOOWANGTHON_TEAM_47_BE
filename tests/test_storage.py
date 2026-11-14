"""저장소: Supabase 키가 없으면 메모리, 있으면 Supabase에도 저장·조회. 진짜 Supabase 대신 가짜 클라이언트를 쓴다."""

import pytest

from app import storage
from pipeline.schema import InspectionResult, Verdict


class _FakeQuery:
    def __init__(self, db, table):
        self.db, self.table, self.op, self.payload, self.filter = db, table, None, None, None

    def upsert(self, row):
        self.op, self.payload = "upsert", row
        return self

    def select(self, cols):
        self.op = "select"
        return self

    def eq(self, col, val):
        self.filter = val
        return self

    def limit(self, n):
        return self

    def execute(self):
        if self.db.get("fail"):
            raise RuntimeError("network down")
        rows = self.db.setdefault(self.table, {})
        if self.op == "upsert":
            rows[self.payload["id"]] = self.payload
            return type("R", (), {"data": [self.payload]})
        return type("R", (), {"data": [rows[self.filter]] if self.filter in rows else []})


class _FakeClient:
    def __init__(self):
        self.db = {}

    def table(self, name):
        return _FakeQuery(self.db, name)


@pytest.fixture
def supa(monkeypatch):
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_KEY", "test")
    fake = _FakeClient()
    monkeypatch.setattr(storage, "_cached_client", fake)
    monkeypatch.setattr(storage, "_memory", {})
    return fake


def _result(i="00000000-0000-0000-0000-000000000001"):
    return InspectionResult(inspection_id=i, overall=Verdict.HOLD, scope={"C": "판정"})


def test_memory_only_without_keys(monkeypatch):
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_KEY", raising=False)
    monkeypatch.setattr(storage, "_memory", {})
    assert storage.backend() == "memory"
    storage.save_result(_result())
    assert storage.load_result(_result().inspection_id).overall == Verdict.HOLD


def test_saves_and_loads_from_supabase(supa):
    r = _result()
    storage.save_result(r)
    row = supa.db["inspections"][r.inspection_id]
    assert row["overall"] == "hold" and row["schema_version"] == r.schema_version
    storage._memory.clear()  # 서버 재시작 흉내: 메모리가 비어도 Supabase에서 읽음
    assert storage.load_result(r.inspection_id).scope == {"C": "판정"}


def test_supabase_failure_does_not_break(supa):
    supa.db["fail"] = True
    r = _result()
    storage.save_result(r)  # 예외가 나지 않아야 함
    assert storage.load_result(r.inspection_id) is not None  # 메모리에 있음
    storage._memory.clear()
    assert storage.load_result(r.inspection_id) is None
