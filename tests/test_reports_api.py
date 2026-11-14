"""신고 API: 위치 입력, 사진 저장·조회, 관리자 API, 업로드 제한, 깨우기용 점검. 진짜 모델 대신 합성 장면 라벨."""

import cv2
import pytest
from fastapi.testclient import TestClient

from app import main
from pipeline import synth
from pipeline.segment import summarize


class _FakeSegmenter:
    def __init__(self, label):
        self.label = label

    def predict(self, image_bgr, photo):
        return self.label, summarize(self.label, photo)


@pytest.fixture
def client(monkeypatch):
    img, label, _ = synth.render_wide_scene()
    ok, buf = cv2.imencode(".jpg", img)
    with TestClient(main.app) as c:
        main.state["segmenter"] = _FakeSegmenter(label)
        c.jpg = buf.tobytes()
        yield c


def _post(c, **form):
    return c.post("/api/inspections", files={"photo": ("a.jpg", c.jpg, "image/jpeg")}, data=form)


def test_report_with_location_and_photo(client):
    r = _post(client, address="청주시 청원구 우암동", memo="우암초 앞 횡단보도", lat="36.65", lng="127.49")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["location"] == {"address": "청주시 청원구 우암동", "memo": "우암초 앞 횡단보도", "lat": 36.65, "lng": 127.49}
    assert body["photo_url"] == f"/api/inspections/{body['inspection_id']}/photo" and body["created_at"]
    p = client.get(body["photo_url"])
    assert p.status_code == 200 and p.headers["content-type"] == "image/jpeg"
    again = client.get(f"/api/inspections/{body['inspection_id']}").json()
    assert again["location"]["address"] == "청주시 청원구 우암동"


def test_location_is_optional_and_validated(client):
    assert _post(client).json()["location"] is None
    assert _post(client, address="   ").json()["location"] is None  # 빈칸은 없는 것으로
    assert _post(client, lat="123").status_code == 422  # 위도 범위 밖


def test_admin_requires_token(client, monkeypatch):
    assert client.get("/api/admin/inspections").status_code == 503  # 토큰 미설정
    monkeypatch.setenv("ADMIN_TOKEN", "s3cret")
    assert client.get("/api/admin/inspections").status_code == 401
    assert client.get("/api/admin/inspections", headers={"X-Admin-Token": "wrong"}).status_code == 401
    assert client.get("/api/admin/inspections", headers={"X-Admin-Token": "s3cret"}).status_code == 200


def test_admin_list_detail_update(client, monkeypatch):
    monkeypatch.setenv("ADMIN_TOKEN", "s3cret")
    h = {"X-Admin-Token": "s3cret"}
    rid = _post(client, address="A").json()["inspection_id"]
    _post(client, address="B")
    lst = client.get("/api/admin/inspections", headers=h).json()
    assert len(lst) == 2 and {x["status"] for x in lst} == {"received"}
    d = client.get(f"/api/admin/inspections/{rid}", headers=h).json()
    assert d["location"]["address"] == "A" and d["result"]["inspection_id"] == rid
    u = client.patch(f"/api/admin/inspections/{rid}", headers=h, json={"status": "resolved", "admin_memo": "보수 완료"})
    assert u.status_code == 200 and u.json()["status"] == "resolved" and u.json()["admin_memo"] == "보수 완료"
    only = client.get("/api/admin/inspections", headers=h, params={"status": "resolved"}).json()
    assert [x["inspection_id"] for x in only] == [rid]
    assert client.patch(f"/api/admin/inspections/{rid}", headers=h, json={"status": "done"}).status_code == 422
    assert client.get("/api/admin/inspections/00000000-0000-0000-0000-000000000000", headers=h).status_code == 404


def test_upload_limits(client, monkeypatch):
    monkeypatch.setattr(main, "MAX_UPLOAD_MB", 0.0001)
    assert _post(client).status_code == 413
    monkeypatch.setattr(main, "MAX_UPLOAD_MB", 15.0)
    monkeypatch.setattr(main, "RATE_LIMIT_PER_MIN", 2)
    main._hits.clear()
    assert [_post(client).status_code for _ in range(3)] == [200, 200, 429]


def test_health_deep(client):
    r = client.get("/api/health/deep")
    assert r.status_code == 200 and r.json()["storage"] == "memory"
    main.state.pop("segmenter")
    assert client.get("/api/health/deep").status_code == 503


def test_citizen_report_submit(client, monkeypatch):
    """결과 화면 → 신고 화면: 발견 위치·내용을 접수하면 결과·관리자 목록에 위치가 채워진다."""
    rid = _post(client).json()["inspection_id"]
    body = {"address": " 청주시 우암동 우암초 앞 ", "memo": "선형 블록이 횡단보도로 이어지지 않음", "lat": 36.65, "lng": 127.49}
    r = client.post(f"/api/inspections/{rid}/report", json=body)
    assert r.status_code == 200, r.text
    assert r.json()["status"] == "received"
    assert r.json()["location"] == {"address": "청주시 우암동 우암초 앞", "memo": "선형 블록이 횡단보도로 이어지지 않음",
                                    "lat": 36.65, "lng": 127.49}
    assert client.get(f"/api/inspections/{rid}").json()["location"]["address"] == "청주시 우암동 우암초 앞"

    # 접수 상태인 동안은 다시 보내면 덮어쓰기
    r = client.post(f"/api/inspections/{rid}/report", json={"address": "우암동 횡단보도"})
    assert r.json()["location"] == {"address": "우암동 횡단보도", "memo": None, "lat": None, "lng": None}

    monkeypatch.setenv("ADMIN_TOKEN", "s3cret")
    h = {"X-Admin-Token": "s3cret"}
    listed = {x["inspection_id"]: x for x in client.get("/api/admin/inspections", headers=h).json()}
    assert listed[rid]["location"]["address"] == "우암동 횡단보도"

    # 관리자가 확인을 시작하면 시민 쪽 수정은 잠김
    client.patch(f"/api/admin/inspections/{rid}", json={"status": "reviewing"}, headers=h)
    assert client.post(f"/api/inspections/{rid}/report", json={"address": "다른 곳"}).status_code == 409


def test_citizen_report_validation(client):
    rid = _post(client).json()["inspection_id"]
    assert client.post("/api/inspections/nope/report", json={"address": "x"}).status_code == 404
    assert client.post(f"/api/inspections/{rid}/report", json={"address": ""}).status_code == 422
    assert client.post(f"/api/inspections/{rid}/report", json={"address": "   "}).status_code == 422
    assert client.post(f"/api/inspections/{rid}/report", json={"address": "x", "lat": 123}).status_code == 422
    assert client.post(f"/api/inspections/{rid}/report", json={"address": "x" * 201}).status_code == 422
