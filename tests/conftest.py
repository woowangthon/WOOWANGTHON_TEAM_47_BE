"""모든 테스트 공통: 실제 Supabase·관리자 토큰을 쓰지 않게 막는다.

app.main이 .env를 읽어 SUPABASE 키가 환경변수에 들어와도, 테스트에서는 항상 메모리 저장소를 쓴다.
Supabase 동작은 tests/test_storage.py의 가짜 클라이언트로만 확인한다.
"""

import pytest


@pytest.fixture(autouse=True)
def _isolate_storage(monkeypatch):
    for k in ("SUPABASE_URL", "SUPABASE_KEY", "ADMIN_TOKEN"):
        monkeypatch.delenv(k, raising=False)
    from app import main, storage

    monkeypatch.setattr(storage, "_cached_client", None)
    monkeypatch.setattr(storage, "_memory", {})
    monkeypatch.setattr(storage, "_photos", {})
    main._hits.clear()
    yield
