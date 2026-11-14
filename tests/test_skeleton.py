"""Path structure regression tests (dot-tile junctions are PASS nodes)."""

import pytest

from pipeline import synth
from pipeline.schema import MaskClass, NodeType, PhotoKind
from pipeline.skeleton import end_direction, extract_path


def _count(path, t):
    return sum(n.type == t for n in path.nodes)


@pytest.mark.parametrize(
    "scene, dot_corner",
    [
        (synth.straight, False),
        (synth.l_turn, True),
        (lambda: synth.l_turn(False), False),
        (synth.t_branch, False),
        (lambda: synth.t_branch(False), False),
    ],
)
def test_node_counts(scene, dot_corner):
    label, exp = scene()
    path = extract_path(label, PhotoKind.CLOSE)
    assert _count(path, NodeType.END) == exp["ends"]
    assert _count(path, NodeType.BRANCH) == exp["branches"]
    assert len(path.component_area_px) == 1

    if dot_corner:
        # 점형이 L자 모서리에 있으면 선형 끝점 2개가 점형 노드에 연결된다.
        # 점형 노드는 degree=2 이므로 PASS이고, TURN 노드가 아니다.
        assert _count(path, NodeType.TURN) == 0
        dot_passes = [
            n for n in path.nodes
            if n.block == MaskClass.DOT and n.type == NodeType.PASS
        ]
        assert len(dot_passes) == 1
        assert len(path.edges_of(dot_passes[0].id)) == 2
    else:
        assert _count(path, NodeType.TURN) == exp["turns"]


@pytest.mark.parametrize(
    "with_dot, expected_type, expected_block",
    [
        (True, NodeType.PASS, MaskClass.DOT),
        (False, NodeType.TURN, MaskClass.LINEAR),
    ],
)
def test_turn_location(with_dot, expected_type, expected_block):
    label, _ = synth.l_turn(with_dot)
    path = extract_path(label, PhotoKind.CLOSE)
    candidates = [
        n for n in path.nodes
        if n.type == expected_type and n.block == expected_block
    ]
    assert len(candidates) == 1
    corner = candidates[0]
    assert abs(corner.pt.x - 200) < 15
    assert abs(corner.pt.y - 250) < 15


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
