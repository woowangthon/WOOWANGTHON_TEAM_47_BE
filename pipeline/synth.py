"""합성 라벨 마스크 생성기: 규칙 단위 테스트용. (담당: B)

각 함수는 (label, expected) 를 돌려준다. label 값은 MaskClass.
블록 폭 기본 30px ≒ 실제 블록 한 줄.

render_wide_scene()은 C4용으로 라벨뿐 아니라 사진(줄무늬·블록 무늬)까지 그린다.

TODO: 끊김(M4) 장면 추가.
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


# ---------------------------------------------------------------------------
# C4: 사진까지 그리는 넓은 장면
# ---------------------------------------------------------------------------

PX_PER_M = 200  # 위에서 본 평면 그림의 해상도
SCENE_W_M, SCENE_D_M = 6.0, 9.0  # 폭(X), 앞쪽 깊이(Y)
CURB_Y, CW_Y0 = 4.85, 5.0  # 연석 시작, 횡단보도 시작 (카메라 발밑이 Y=0)
YELLOW, YELLOW_HI = (0, 175, 225), (110, 235, 255)


def _top_px(x_m: float, y_m: float) -> tuple[int, int]:
    return round(x_m * PX_PER_M), round((SCENE_D_M - y_m) * PX_PER_M)


def _camera_homography(img_w, img_h, cam_x, cam_h, pitch_deg, yaw_deg=0.0, roll_deg=0.0, focal_ratio=0.75):
    """위에서 본 평면 그림 픽셀 → 폰 사진 픽셀 호모그래피. 카메라는 +Y(앞)를 보며 아래로 숙임.

    yaw_deg: 왼쪽(+)으로 돌림, roll_deg: 사진이 시계 방향(+)으로 기울어짐.
    """
    f = focal_ratio * max(img_w, img_h)
    K = np.array([[f, 0, img_w / 2], [0, f, img_h / 2], [0, 0, 1.0]])
    t = np.radians(pitch_deg)
    R = np.array([[1, 0, 0], [0, -np.sin(t), -np.cos(t)], [0, np.cos(t), -np.sin(t)]])  # 월드(X,Y,Z위) → 카메라
    y, r = np.radians(yaw_deg), np.radians(roll_deg)
    Rz_world = np.array([[np.cos(y), np.sin(y), 0], [-np.sin(y), np.cos(y), 0], [0, 0, 1]])
    Rroll = np.array([[np.cos(r), -np.sin(r), 0], [np.sin(r), np.cos(r), 0], [0, 0, 1]])
    R = Rroll @ R @ Rz_world
    C = np.array([cam_x, 0.0, cam_h])
    ground_to_img = K @ np.column_stack([R[:, 0], R[:, 1], -R @ C])  # (X, Y, 1) → 사진
    top_to_ground = np.array([[1 / PX_PER_M, 0, 0], [0, -1 / PX_PER_M, SCENE_D_M], [0, 0, 1]])
    return ground_to_img @ top_to_ground


def _ridges(img, lab, x0, x1, y0, y1, offset_deg, period_m=0.075, width_m=0.035):
    """선형 블록: 돌출선이 진행 방향(Y)에서 offset_deg 만큼 돌아간 줄무늬."""
    (u0, v1), (u1, v0) = _top_px(x0, y0), _top_px(x1, y1)
    vv, uu = np.mgrid[v0:v1, u0:u1]
    o = np.radians(offset_deg)
    s = uu * np.cos(o) + vv * np.sin(o)
    on = (s % (period_m * PX_PER_M)) < width_m * PX_PER_M
    img[v0:v1, u0:u1] = YELLOW
    img[v0:v1, u0:u1][on] = YELLOW_HI
    lab[v0:v1, u0:u1] = int(MaskClass.LINEAR)


def _dots(img, lab, x0, x1, y0, y1, pitch_m=0.05, r_m=0.0125):
    (u0, v1), (u1, v0) = _top_px(x0, y0), _top_px(x1, y1)
    img[v0:v1, u0:u1] = YELLOW
    step, r = pitch_m * PX_PER_M, max(1, round(r_m * PX_PER_M))
    for v in np.arange(v0 + step / 2, v1, step):
        for u in np.arange(u0 + step / 2, u1, step):
            cv2.circle(img, (round(u), round(v)), r, YELLOW_HI, -1)
    lab[v0:v1, u0:u1] = int(MaskClass.DOT)


def render_wide_scene(
    ridge_offset_deg: float = 0.0,
    entrance_dot: bool = True,
    stripes: bool = True,
    cam_x: float = 3.0,
    cam_h: float = 1.5,
    pitch_deg: float = 25.0,
    yaw_deg: float = 0.0,
    roll_deg: float = 0.0,
    focal_ratio: float = 0.75,
    img_size: tuple[int, int] = (1050, 1400),
) -> tuple[np.ndarray, np.ndarray, dict]:
    """넓은 사진: 인도 위 선형 경로 → 진입부 점형 → 연석 → 횡단보도.

    ridge_offset_deg  선형 돌출선을 진행 방향에서 돌린 각도 (0이면 정상, C4 방향 불일치 시험용)
    entrance_dot      False면 진입부까지 선형만 깔림 (C4 점형 누락 시험용)
    stripes           False면 횡단보도 라벨만 있고 줄무늬는 없음 (시점 보정 실패 → H2 시험용)

    돌려주는 것: (사진 BGR, 라벨, meta). meta["top_to_img"]는 평면 그림 → 사진 호모그래피.
    """
    tw, th = round(SCENE_W_M * PX_PER_M), round(SCENE_D_M * PX_PER_M)
    rng = np.random.default_rng(47)
    top = np.full((th, tw, 3), 120, np.uint8)  # 인도
    lab = np.zeros((th, tw), np.uint8)
    for y in np.arange(0.3, CURB_Y, 0.3):  # 인도 보도블록 줄눈
        top[_top_px(0, y)[1], :] = 95
    for x in np.arange(0.3, SCENE_W_M, 0.3):
        top[_top_px(0, CURB_Y)[1]:, _top_px(x, 0)[0]] = 95
    top[_top_px(0, CW_Y0)[1]:_top_px(0, CURB_Y)[1], :] = 175  # 연석
    cw_v = _top_px(0, CW_Y0)[1]
    top[:cw_v, :] = 70  # 아스팔트
    lab[:cw_v, :] = int(MaskClass.CROSSWALK)
    if stripes:
        y = CW_Y0 + 0.2
        while y + 0.5 < SCENE_D_M:  # 흰 줄 50cm, 간격 45cm, 차도 방향(X)으로 길게
            top[_top_px(0, y + 0.5)[1]:_top_px(0, y)[1], :] = 225
            y += 0.95
    _ridges(top, lab, 2.7, 3.3, 0.0, 4.25 if entrance_dot else CURB_Y, ridge_offset_deg)
    if entrance_dot:
        _dots(top, lab, 2.4, 3.6, 4.25, CURB_Y)
    top = np.clip(top.astype(np.int16) + rng.normal(0, 4, top.shape).astype(np.int16), 0, 255).astype(np.uint8)

    w, h = img_size
    Hm = _camera_homography(w, h, cam_x, cam_h, pitch_deg, yaw_deg, roll_deg, focal_ratio)
    img = cv2.warpPerspective(top, Hm, (w, h), flags=cv2.INTER_LINEAR)
    label = cv2.warpPerspective(lab, Hm, (w, h), flags=cv2.INTER_NEAREST)
    return img, label, {"top_to_img": Hm, "c4": entrance_dot and abs(ridge_offset_deg) < 10}
