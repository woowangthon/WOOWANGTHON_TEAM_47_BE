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
