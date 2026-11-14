"""사진 한 장 판정(judge_photo)과 API 테스트."""

import cv2
import numpy as np
from fastapi.testclient import TestClient

from pipeline import synth
from pipeline.rules import judge_photo
from pipeline.run import analyze_label, run_photo
from pipeline.schema import MaskClass, PhotoKind, Verdict
from pipeline.segment import summarize


def _photo(label, img=None, conf=None):
    seg = summarize(label, PhotoKind.WIDE, conf)
    return judge_photo(analyze_label(label, PhotoKind.WIDE, img, seg), label)


def _codes(r, verdict=None):
    return {f.code for f in r.findings if verdict is None or f.verdict == verdict}


def test_close_up_blocks_without_crosswalk_runs_m_only():
    label, _ = synth.l_turn(dot_at_turn=False)
    r = _photo(label)
    assert r.scope["M"] == "판정" and r.scope["C"].startswith("해당 없음")
    assert "M3" in _codes(r, Verdict.SUSPECTED)
    assert not _codes(r) & {"C1", "C2", "C3", "C4", "C5", "C6", "H3"}  # C는 해당 없음, H3로 보류하지 않음


def test_close_up_normal_turn_is_normal_overall():
    label, _ = synth.l_turn(dot_at_turn=True)
    r = _photo(label)
    assert r.overall == Verdict.NORMAL


def test_crosswalk_scene_runs_c():
    img, label, _ = synth.render_wide_scene()
    r = _photo(label, img)
    assert r.scope["C"] == "판정"
    assert "C1" in _codes(r, Verdict.NORMAL) and "C4" in _codes(r, Verdict.NORMAL)


def test_m_skipped_not_held_when_blocks_too_small_but_c_judged():
    """C를 판정했으면 M 관문을 못 넘어도 보류로 세지 않고 건너뛴다."""
    from pipeline.rules.hold import HoldConfig

    img, label, _ = synth.render_wide_scene()
    a = analyze_label(label, PhotoKind.WIDE, img, summarize(label, PhotoKind.WIDE))
    r = judge_photo(a, label, HoldConfig(min_block_width=0.5))  # 관문을 일부러 높여 M을 못 넘게
    assert r.scope["M"].startswith("건너뜀")
    assert "H4" not in _codes(r)
    assert r.overall == Verdict.NORMAL


def test_both_families_when_crosswalk_and_big_turn():
    """횡단보도 + 크게 찍힌 꺾임점: C와 M 위반이 함께 나온다."""
    label, _ = synth.approach_crosswalk(entrance_dot=False)  # 진입부 점형 누락 (C4)
    label = label.copy()
    cv2.line(label, (320, 620), (600, 620), int(MaskClass.LINEAR), 30)  # 아래쪽 ㄱ자 꺾임, 꺾임점도 선형 (M3)
    r = _photo(label)
    assert r.scope["C"] == "판정" and r.scope["M"] == "판정"
    assert {"C4", "M3"} <= _codes(r, Verdict.SUSPECTED)


def test_tiny_blocks_without_crosswalk_holds_h4():
    label = np.zeros((640, 640), np.uint8)
    label[300:310, 300:340] = MaskClass.LINEAR
    r = _photo(label)
    assert "H4" in _codes(r, Verdict.HOLD) and r.overall == Verdict.HOLD


def test_low_confidence_holds_whole_photo():
    label, _ = synth.l_turn()
    conf = np.where(np.isin(label, [1, 2]), 0.2, 0).astype(np.float32)
    r = _photo(label, conf=conf)
    assert _codes(r) == {"H1"} and r.overall == Verdict.HOLD


class _FakeSegmenter:
    """API 테스트용: 진짜 모델 대신 합성 장면 라벨을 돌려준다."""

    def __init__(self, label):
        self.label = label

    def predict(self, image_bgr, photo):
        return self.label, summarize(self.label, photo)


def test_run_photo_and_api_single_upload(tmp_path, monkeypatch):
    img, label, _ = synth.render_wide_scene()
    r = run_photo(_FakeSegmenter(label), img, "t1")
    assert r.photo is not None and r.inspection_id == "t1"

    from app import main, storage

    monkeypatch.setattr(storage, "DATA_DIR", tmp_path, raising=False)
    ok, buf = cv2.imencode(".jpg", img)
    with TestClient(main.app) as c:
        main.state["segmenter"] = _FakeSegmenter(label)
        res = c.post("/api/inspections", files={"photo": ("a.jpg", buf.tobytes(), "image/jpeg")})
        assert res.status_code == 200, res.text
        body = res.json()
        assert body["schema_version"] == "0.2.0"
        assert body["scope"]["C"] == "판정"
        assert body["photo"] is not None and body["close"] is None and body["wide"] is None
        assert c.get(f"/api/inspections/{body['inspection_id']}").status_code == 200
