from pipeline import synth
from pipeline.rules import judge
from pipeline.run import analyze_label
from pipeline.schema import PhotoKind, Verdict


def _m3(result):
    return [f for f in result.findings if f.code == "M3"]


def test_m3_normal_when_turn_is_dot():
    label, _ = synth.l_turn(dot_at_turn=True)
    r = judge(analyze_label(label, PhotoKind.CLOSE), None)
    assert [f.verdict for f in _m3(r)] == [Verdict.NORMAL]
    assert r.overall == Verdict.NORMAL


def test_m3_suspected_when_turn_is_linear():
    label, _ = synth.l_turn(dot_at_turn=False)
    r = judge(analyze_label(label, PhotoKind.CLOSE), None)
    bad = [f for f in _m3(r) if f.verdict == Verdict.SUSPECTED]
    assert len(bad) == 1 and bad[0].location is not None
    assert r.overall == Verdict.SUSPECTED


def test_m3_suspected_at_branch():
    label, _ = synth.t_branch(dot_at_branch=False)
    r = judge(analyze_label(label, PhotoKind.CLOSE), None)
    assert any(f.verdict == Verdict.SUSPECTED for f in _m3(r))


def test_h3_holds_c_when_no_crosswalk():
    label, _ = synth.straight()
    r = judge(None, analyze_label(label, PhotoKind.WIDE), label)
    assert [f.code for f in r.findings] == ["H3"]
    assert r.overall == Verdict.HOLD


def test_h2_holds_c4_without_rectification():
    label, _ = synth.approach_crosswalk()
    r = judge(None, analyze_label(label, PhotoKind.WIDE), label)
    assert "H2" in {f.code for f in r.findings}


# ---------------------------------------------------------------- C 계열


def _wide(label):
    return judge(None, analyze_label(label, PhotoKind.WIDE), label)


def _codes(r, verdict=None):
    return {f.code for f in r.findings if verdict is None or f.verdict == verdict}


def test_c1_normal_when_path_reaches_crosswalk():
    label, _ = synth.approach_crosswalk()
    r = _wide(label)
    assert [f.verdict for f in r.findings if f.code == "C1"] == [Verdict.NORMAL]
    assert not _codes(r, Verdict.SUSPECTED)


def test_c2_when_path_heads_into_road():
    label, _ = synth.road_straight()
    r = _wide(label)
    assert "C2" in _codes(r, Verdict.SUSPECTED)
    c3 = [f for f in r.findings if f.code == "C3"]
    assert c3 and all(not f.primary for f in c3)  # C2와 같이 뜨면 C3는 보조
    assert r.overall == Verdict.SUSPECTED


def test_c2_even_if_curb_blocks_link_to_crosswalk():
    label, _ = synth.road_straight(curb_link=True)
    r = _wide(label)
    assert "C2" in _codes(r, Verdict.SUSPECTED)
    assert "C1" not in _codes(r, Verdict.NORMAL)  # 연결돼 있어도 방향 오류가 우선


def test_c3_when_crosswalk_is_beside_and_no_turn():
    label, _ = synth.side_crosswalk_no_turn()
    r = _wide(label)
    assert _codes(r, Verdict.SUSPECTED) == {"C3"}
    assert all(f.primary for f in r.findings if f.code == "C3")


def test_c1_when_path_stops_short():
    label, _ = synth.short_of_crosswalk()
    r = _wide(label)
    assert _codes(r, Verdict.SUSPECTED) == {"C1"}


def test_road_only_checks_c2_and_holds_rest():
    label, _ = synth.road_only_straight()
    r = _wide(label)
    assert "C2" in _codes(r, Verdict.SUSPECTED)
    assert "H3" in _codes(r, Verdict.HOLD)
    assert r.overall == Verdict.SUSPECTED


# ---------------------------------------------------------------- C4 (사진까지 그린 장면)


def _wide_img(**kw):
    img, label, meta = synth.render_wide_scene(**kw)
    return judge(None, analyze_label(label, PhotoKind.WIDE, img), label), meta


def test_c4_normal_and_overall_normal():
    r, _ = _wide_img()
    assert [f.verdict for f in r.findings if f.code == "C4"] == [Verdict.NORMAL]
    assert r.overall == Verdict.NORMAL  # C1 정상 + C4 정상


def test_c4_when_ridges_misaligned():
    r, _ = _wide_img(ridge_offset_deg=40)
    c4 = [f for f in r.findings if f.code == "C4"]
    assert c4 and all(f.verdict == Verdict.SUSPECTED for f in c4)
    assert "어긋남" in c4[0].message


def test_c4_small_offset_is_tolerated():
    r, _ = _wide_img(ridge_offset_deg=10)
    assert "C4" in _codes(r, Verdict.NORMAL)


def test_c4_when_entrance_has_no_dot():
    r, _ = _wide_img(entrance_dot=False)
    assert any(f.code == "C4" and "점형" in f.message for f in r.findings if f.verdict == Verdict.SUSPECTED)


def test_c4_robust_to_camera_yaw_and_roll():
    assert "C4" in _codes(_wide_img(yaw_deg=15, roll_deg=-6)[0], Verdict.NORMAL)
    assert "C4" in _codes(_wide_img(yaw_deg=15, roll_deg=-6, ridge_offset_deg=30)[0], Verdict.SUSPECTED)


def test_h2_when_crosswalk_has_no_stripes():
    r, _ = _wide_img(stripes=False)
    assert "H2" in _codes(r, Verdict.HOLD)
    assert "C4" not in _codes(r)


def test_rectified_axis_matches_true_direction():
    import numpy as np

    from pipeline.rectify import angle_diff, angle_std, apply_h, rectify_by_crosswalk

    img, label, meta = synth.render_wide_scene(yaw_deg=12, roll_deg=5)
    rect = rectify_by_crosswalk(img, label)
    assert rect.ok
    q = apply_h(np.array(rect.homography) @ meta["top_to_img"], np.array([[600.0, 1500], [600.0, 1000]]))
    assert angle_diff(rect.crosswalk_axis_deg, angle_std(*(q[1] - q[0]))) < 2


# ---------------------------------------------------------------- 규칙위반케이스로 고친 것


def test_path_along_curb_is_not_c1_normal():
    """횡단보도 옆을 스쳐 지나가는 선형은 입구에 닿아도 정상이 아니다 (C3)."""
    label, _ = synth.along_curb_past_crosswalk()
    r = _wide(label)
    assert "C1" not in _codes(r, Verdict.NORMAL)
    assert "C3" in _codes(r, Verdict.SUSPECTED)


def test_c4_entrance_without_dot_detected_without_rectification():
    label, _ = synth.approach_crosswalk(entrance_dot=False)
    r = _wide(label)  # 사진 없이 라벨만 → 시점 보정 실패(H2)
    assert "H2" in _codes(r, Verdict.HOLD)
    assert "C4" in _codes(r, Verdict.SUSPECTED)


def test_c4_entrance_with_dot_not_flagged_without_rectification():
    label, _ = synth.approach_crosswalk(entrance_dot=True)
    r = _wide(label)
    assert "C4" not in _codes(r, Verdict.SUSPECTED)
    assert "C1" in _codes(r, Verdict.NORMAL)


# ---------------------------------------------------------------- C5 선형 유도 없음 · C6 선형 방향 불일치


def test_c5_dot_at_entrance_but_no_linear():
    r, _ = _wide_img(linear=False)
    assert "C5" in _codes(r, Verdict.SUSPECTED)
    assert "C1" not in _codes(r, Verdict.NORMAL)


def test_c5_not_raised_for_normal_scene():
    r, _ = _wide_img()
    assert "C5" not in _codes(r)
    assert "C6" not in _codes(r)


def test_c6_when_linear_path_is_slanted():
    r, _ = _wide_img(path_angle_deg=35)
    c6 = [f for f in r.findings if f.code == "C6"]
    assert c6 and c6[0].verdict == Verdict.SUSPECTED


def test_c6_small_slant_tolerated():
    r, _ = _wide_img(path_angle_deg=10)
    assert "C6" not in _codes(r)


def test_c6_rough_without_rectification():
    """보정 실패여도 흰 줄로 구한 대략 소실점으로 비스듬한 줄을 잡는다."""
    from pipeline.rectify import rectify_by_crosswalk

    img, label, _ = synth.render_wide_scene(path_angle_deg=40, yaw_deg=10)
    a = analyze_label(label, PhotoKind.WIDE, img)
    rect = rectify_by_crosswalk(img, label)
    a.rectification = rect.model_copy(update={"ok": False, "reason": "시험: 보정 실패로 가정"})
    from pipeline.rules.c_rules import C6Config, check_c6

    rough = C6Config(use_rough=True)
    assert any(f.code == "C6" and "대략" in f.message for f in check_c6(a, label, rough))
    assert check_c6(a, label) == []  # 기본은 꺼져 있음
    img, label, _ = synth.render_wide_scene(yaw_deg=10)
    a = analyze_label(label, PhotoKind.WIDE, img)
    a.rectification = rectify_by_crosswalk(img, label).model_copy(update={"ok": False})
    assert check_c6(a, label, rough) == []


def test_rectify_without_crosswalk_does_not_crash():
    import numpy as np

    from pipeline.rectify import rectify_by_crosswalk

    r = rectify_by_crosswalk(np.zeros((100, 100, 3), np.uint8), np.zeros((100, 100), np.uint8))
    assert not r.ok and r.walk_vp is None
