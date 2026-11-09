import pytest

from pipeline import synth
from pipeline.schema import NodeType, PhotoKind
from pipeline.skeleton import end_direction, extract_path


def _count(path, t):
    return sum(n.type == t for n in path.nodes)


@pytest.mark.parametrize(
    "scene",
    [synth.straight, synth.l_turn, lambda: synth.l_turn(False), synth.t_branch, lambda: synth.t_branch(False)],
)
def test_node_counts(scene):
    label, exp = scene()
    path = extract_path(label, PhotoKind.CLOSE)
    assert _count(path, NodeType.END) == exp["ends"]
    assert _count(path, NodeType.TURN) == exp["turns"]
    assert _count(path, NodeType.BRANCH) == exp["branches"]
    assert len(path.component_area_px) == 1


def test_turn_location():
    label, _ = synth.l_turn()
    path = extract_path(label, PhotoKind.CLOSE)
    turn = next(n for n in path.nodes if n.type == NodeType.TURN)
    assert abs(turn.pt.x - 200) < 15 and abs(turn.pt.y - 250) < 15


def test_end_direction_points_outward():
    label, _ = synth.straight()
    path = extract_path(label, PhotoKind.CLOSE)
    top = min((n for n in path.nodes if n.type == NodeType.END), key=lambda n: n.pt.y)
    dx, dy = end_direction(path, top.id)
    assert dy < -0.95  # 위쪽(바깥)을 향함


def test_path_json_serializable():
    label, _ = synth.t_branch()
    path = extract_path(label, PhotoKind.CLOSE)
    assert path.model_dump_json()
