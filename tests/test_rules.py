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
