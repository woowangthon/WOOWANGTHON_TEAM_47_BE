"""Regression for the long horizontal linear tile meeting a single dot patch.

Do not require exact skeleton topology for scenes with real branches.
"""
import numpy as np
from pipeline.schema import MaskClass, NodeType, PhotoKind
from pipeline.skeleton import extract_path


def test_single_dot_straight_corridor_has_no_fake_turn_or_branch():
    mask = np.zeros((640, 640), dtype=np.uint8)
    mask[320:440, :420] = MaskClass.LINEAR
    mask[295:465, 420:580] = MaskClass.DOT
    path = extract_path(mask, PhotoKind.CLOSE)
    suspicious = [n for n in path.nodes if n.type in (NodeType.TURN, NodeType.BRANCH) and n.block == MaskClass.LINEAR]
    assert not suspicious, [(n.type, n.pt) for n in suspicious]
    assert any(n.block == MaskClass.DOT for n in path.nodes)


def test_true_l_without_dot_keeps_turn_candidate():
    mask = np.zeros((640, 640), dtype=np.uint8)
    mask[120:530, 190:245] = MaskClass.LINEAR
    mask[120:175, 190:530] = MaskClass.LINEAR
    path = extract_path(mask, PhotoKind.CLOSE)
    assert any(n.type == NodeType.TURN for n in path.nodes)


def test_true_t_without_dot_keeps_branch_candidate():
    mask = np.zeros((640, 640), dtype=np.uint8)
    mask[100:530, 290:345] = MaskClass.LINEAR
    mask[100:155, 120:520] = MaskClass.LINEAR
    path = extract_path(mask, PhotoKind.CLOSE)
    assert any(n.type == NodeType.BRANCH for n in path.nodes)
