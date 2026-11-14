import cv2
import numpy as np

from pipeline.schema import PhotoKind, Verdict
from pipeline.shape_rules import analyze_m3, find_m3_candidates
from pipeline.skeleton import extract_path


def make_mask(kind, dot=False):
    m = np.zeros((480, 480), dtype=np.uint8)
    if kind == 'L':
        cv2.line(m, (120, 400), (120, 120), 1, 33)
        cv2.line(m, (120, 120), (400, 120), 1, 33)
        if dot: cv2.rectangle(m, (82, 82), (158, 158), 2, -1)
    elif kind == 'T':
        cv2.line(m, (80, 120), (400, 120), 1, 33)
        cv2.line(m, (240, 120), (240, 400), 1, 33)
        if dot: cv2.rectangle(m, (202, 82), (278, 158), 2, -1)
    else:
        cv2.line(m, (240, 70), (240, 400), 1, 33)
    return m


def check(kind, dot, expected):
    mask = make_mask(kind, dot)
    path = extract_path(mask, PhotoKind.CLOSE)
    findings = analyze_m3(mask, path)
    assert any(f.verdict == expected for f in findings), [f.model_dump() for f in findings]


def test_L_missing_dot(): check('L', False, Verdict.SUSPECTED)
def test_L_with_dot(): check('L', True, Verdict.NORMAL)
def test_T_missing_dot(): check('T', False, Verdict.SUSPECTED)
def test_T_with_dot(): check('T', True, Verdict.NORMAL)
def test_straight_no_turn(): check('straight', False, Verdict.HOLD)


def test_mismatch_rejected():
    m = make_mask('L')
    path = extract_path(m, PhotoKind.CLOSE)
    try:
        find_m3_candidates(m[:300], path)
    except ValueError:
        pass
    else:
        raise AssertionError('해상도 불일치를 검사하지 못함')
