"""합성 라벨 마스크 생성기: 규칙 단위 테스트용. (담당: B)

각 함수는 (label, expected) 를 돌려준다. label 값은 MaskClass.
블록 폭 기본 30px ≒ 실제 블록 한 줄.

TODO: 진입부 점형 누락(C4), 끊김(M4) 장면 추가.
"""

from __future__ import annotations

import cv2
import numpy as np

from .schema import MaskClass

W, H = 640, 640
BLOCK = 30


def _canvas(w: int = W, h: int = H) -> np.ndarray:
    return np.zeros((h, w), np.uint8)


def _line(label, p0, p1, cls: MaskClass, width: int = BLOCK) -> None:
    cv2.line(label, p0, p1, int(cls), thickness=width, lineType=cv2.LINE_8)


def _square(label, center, cls: MaskClass, size: int = BLOCK * 2) -> None:
    x, y = center
    r = size // 2
    label[max(0, y - r) : y + r, max(0, x - r) : x + r] = int(cls)


def straight() -> tuple[np.ndarray, dict]:
    """직선 선형 경로. 꺾임·분기 없음."""
    lab = _canvas()
    _line(lab, (320, 600), (320, 60), MaskClass.LINEAR)
    return lab, {"ends": 2, "turns": 0, "branches": 0, "m3": None}


def l_turn(dot_at_turn: bool = True) -> tuple[np.ndarray, dict]:
    """ㄱ자 경로. dot_at_turn=False 면 M3 오류."""
    lab = _canvas()
    _line(lab, (200, 600), (200, 250), MaskClass.LINEAR)
    _line(lab, (200, 250), (580, 250), MaskClass.LINEAR)
    if dot_at_turn:
        _square(lab, (200, 250), MaskClass.DOT)
    return lab, {"ends": 2, "turns": 1, "branches": 0, "m3": "normal" if dot_at_turn else "suspected"}


def t_branch(dot_at_branch: bool = True) -> tuple[np.ndarray, dict]:
    """T자 분기. dot_at_branch=False 면 M3 오류."""
    lab = _canvas()
    _line(lab, (320, 600), (320, 250), MaskClass.LINEAR)
    _line(lab, (60, 250), (580, 250), MaskClass.LINEAR)
    if dot_at_branch:
        _square(lab, (320, 250), MaskClass.DOT)
    return lab, {"ends": 3, "turns": 0, "branches": 1, "m3": "normal" if dot_at_branch else "suspected"}


def approach_crosswalk(with_road: bool = True) -> tuple[np.ndarray, dict]:
    """넓은 사진: 선형 경로 → 진입부 점형 → 횡단보도. 정상 장면."""
    lab = _canvas()
    if with_road:
        lab[:200, :] = int(MaskClass.ROAD)
    lab[:200, 240:400] = int(MaskClass.CROSSWALK)
    _line(lab, (320, 620), (320, 250), MaskClass.LINEAR)
    lab[200:260, 260:380] = int(MaskClass.DOT)
    return lab, {"ends": 2, "turns": 0, "branches": 0}


def road_straight(curb_link: bool = False) -> tuple[np.ndarray, dict]:
    """넓은 사진: 선형 줄이 횡단보도 없는 차도로 직진, 횡단보도는 오른쪽 옆. C2 (+C3 보조).

    curb_link=True 면 연석을 따라 점형 줄이 옆 횡단보도까지 이어진다.
    연결만 보면 정상이지만 선형이 차도를 향하므로 C2 (노션 5장 예시).
    """
    lab = _canvas()
    lab[:200, :] = int(MaskClass.ROAD)
    lab[:200, 440:600] = int(MaskClass.CROSSWALK)
    _line(lab, (200, 620), (200, 235), MaskClass.LINEAR)
    if curb_link:
        lab[205:245, 185:470] = int(MaskClass.DOT)
    return lab, {"c2": True, "c3": not curb_link}


def side_crosswalk_no_turn() -> tuple[np.ndarray, dict]:
    """넓은 사진: 횡단보도가 오른쪽 옆에 있는데 경로가 꺾이지 않고 앞(인도)으로 직진. C3."""
    lab = _canvas()
    lab[:, 420:] = int(MaskClass.ROAD)
    lab[100:420, 420:] = int(MaskClass.CROSSWALK)
    _line(lab, (150, 620), (150, 60), MaskClass.LINEAR)
    return lab, {"c3": True}


def short_of_crosswalk() -> tuple[np.ndarray, dict]:
    """넓은 사진: 경로가 횡단보도를 향하지만 입구 훨씬 앞에서 끝남. C1."""
    lab = _canvas()
    lab[:200, :] = int(MaskClass.ROAD)
    lab[:200, 240:400] = int(MaskClass.CROSSWALK)
    _line(lab, (320, 620), (320, 380), MaskClass.LINEAR)
    return lab, {"c1": "suspected"}


def road_only_straight() -> tuple[np.ndarray, dict]:
    """넓은 사진: 횡단보도 없이 차도만 보이고 경로가 차도로 직진. C2 + H3(부분 보류)."""
    lab = _canvas()
    lab[:200, :] = int(MaskClass.ROAD)
    _line(lab, (320, 620), (320, 235), MaskClass.LINEAR)
    return lab, {"c2": True}
