from fastapi.testclient import TestClient

from app.main import app


def test_health():
    with TestClient(app) as c:
        r = c.get("/api/health")
        assert r.status_code == 200
        assert r.json()["ok"] is True


def test_inspection_requires_photo():
    with TestClient(app) as c:
        assert c.post("/api/inspections").status_code == 400
