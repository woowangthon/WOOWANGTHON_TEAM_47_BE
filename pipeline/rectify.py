"""④ 정밀 분석: 시점 보정 + 횡단보도 진입부 ROI 돌출선 방향. (C4, 담당: 은재)

시점 보정 (rectify_by_crosswalk)
    횡단보도 흰 줄은 실제로는 서로 평행하고 같은 간격(약 95cm)으로 놓여 있다.
    1. 흰 줄 하나하나를 찾아 긴 축(차도 방향 T)을 구한다
    2. 사진 속 긴 축들이 모이는 점 → T 방향 소실점
    3. 줄들을 가로지르는 선 위에서 '같은 간격이 점점 좁아지는 비율' → 두 번째 소실점
    4. 두 소실점을 이은 선이 지평선. 폰 초점거리(긴 변의 0.75배로 가정)와 지평선으로
       지면의 기울기를 구하고, 카메라를 지면 바로 위에서 내려다보도록 돌리는 호모그래피를 만든다
    5. 보정 평면에서 줄들이 평행하고 간격이 고른지 다시 확인한다 (아니면 실패 → H2)
    횡단보도가 사진 밖으로 잘려도 흰 줄이 4개 이상 보이면 된다.
    횡단보도 진행축(사람이 건너는 방향 P)은 보정 평면에서 줄의 긴 축에 수직이다.

진입부 ROI (analyze_rois, 넓은 사진)
    블록이 횡단보도에 닿는 지점(진입부)을 보정 평면으로 옮겨, 횡단보도에서 멀어지는 방향으로
    entrance(진입부 바로 앞)와 approach(1m쯤 뒤 선형 구간) 두 칸을 위에서 본 모습으로 잘라낸다.
    각 칸을 2D FFT 해서
      - 가장 강한 주파수와 같은 간격의 직각 방향이 약함 → 선형. 돌출선 방향 = 주파수 방향에 수직
      - 직각 방향도 비슷하게 강함(정사각 격자) → 점형
      - 무늬가 약함(멀거나 흐림) → 판단 안 함

가까운 사진의 꺾임·분기점 ROI(M3용)는 아직 구현하지 않았다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from .schema import BLOCK_CLASSES, MaskClass, PathStructure, PhotoKind, Point, Rectification, RoiAnalysis


@dataclass
class RectifyConfig:
    focal_ratio: float = 0.75  # 초점거리 / 사진 긴 변 (폰 기본 카메라 26mm 환산 ≈ 0.75)
    work_long_side: int = 1000
    min_bars: int = 4
    stripe_period_m: float = 0.95  # 흰 줄 50cm + 간격 45~50cm
    max_tilt_deg: float = 80.0  # 지면과 거의 나란히 찍으면 실패
    max_bar_angle_std: float = 6.0  # 보정 후 줄들이 평행해야 함
    max_gap_cv: float = 0.3  # 보정 후 줄 간격이 고르게 나와야 함
    min_bar_area: float = 0.0004  # 사진 면적 대비


def angle_std(dx: float, dy: float) -> float:
    """보정 평면(이미지 좌표, y 아래) 벡터 → 각도 (0=오른쪽, 반시계 +), 0~180."""
    return math.degrees(math.atan2(-dy, dx)) % 180


def angle_diff(a: float, b: float) -> float:
    """두 방향(0~180) 사이 각도 차, 0~90."""
    d = abs(a - b) % 180
    return min(d, 180 - d)


def apply_h(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    p = np.c_[pts, np.ones(len(pts))] @ H.T
    return p[:, :2] / p[:, 2:3]


def _fail(reason: str) -> Rectification:
    return Rectification(ok=False, reason=reason)


def _find_bars(gray: np.ndarray, cw: np.ndarray, min_area: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """횡단보도 안 흰 페인트 덩어리 중 길쭉한 것 → (중심, 긴 축 단위벡터)."""
    vals = gray[cw]
    t, _ = cv2.threshold(vals.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    paint = ((gray > t) & cw).astype(np.uint8)
    paint = cv2.morphologyEx(paint, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, cc, st, _ = cv2.connectedComponentsWithStats(paint)
    bars = []
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] < min_area:
            continue
        x, y, w, h = st[i, :4]
        ys, xs = np.nonzero(cc[y:y + h, x:x + w] == i)
        pts = np.c_[xs + x, ys + y].astype(np.float64)
        c = pts.mean(0)
        ev, evec = np.linalg.eigh(np.cov((pts - c).T))
        if ev[0] <= 1e-6 or ev[1] / ev[0] < 6.0:  # 길이/폭 2.5배 이상만
            continue
        bars.append((c, evec[:, 1]))
    return bars


def _vanishing_point(lines: list[np.ndarray]) -> np.ndarray:
    """여러 직선의 최소제곱 교점 (동차좌표, 무한원점 가능)."""
    L = np.array([l / np.hypot(l[0], l[1]) for l in lines])
    return np.linalg.svd(L)[2][-1]


def rectify_by_crosswalk(image_bgr: np.ndarray, label: np.ndarray,
                         cfg: Optional[RectifyConfig] = None) -> Rectification:
    cfg = cfg or RectifyConfig()
    H0, W0 = label.shape[:2]
    if not (label == MaskClass.CROSSWALK).any():
        return _fail("횡단보도 없음")
    s = min(1.0, cfg.work_long_side / max(H0, W0))
    w, h = round(W0 * s), round(H0 * s)
    gray = cv2.cvtColor(cv2.resize(image_bgr, (w, h), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    cw = cv2.resize((label == MaskClass.CROSSWALK).astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    cw = cv2.erode(cw, np.ones((3, 3), np.uint8)).astype(bool)

    bars = _find_bars(gray, cw, round(cfg.min_bar_area * w * h))
    if len(bars) < cfg.min_bars:
        return _fail(f"횡단보도 흰 줄이 {len(bars)}개만 보임 (최소 {cfg.min_bars}개)")

    # 1. 긴 축 방향을 한쪽으로 맞추고, 줄 방향에 수직인 '가로지르는 선' 위 위치 t로 정렬
    #    (같은 간격 평행선은 어떤 가로지르는 선이든 같은 간격으로 자른다)
    ref = bars[0][1]
    bars = [(c, u if u @ ref >= 0 else -u) for c, u in bars]
    zu = np.exp(2j * np.array([math.atan2(u[1], u[0]) for _, u in bars])).mean()
    mu = np.angle(zu) / 2
    d = np.array([-math.sin(mu), math.cos(mu)])
    p0 = np.mean([c for c, _ in bars], axis=0)
    ts = []
    for c, u in bars:
        A = np.column_stack([u, -d])
        if abs(np.linalg.det(A)) < 0.5:  # 평균 방향과 크게 다른 조각(오검출)
            continue
        ts.append(np.linalg.solve(A, p0 - c)[1])
    ts = list(np.sort(ts))
    # 그림자·차선에 끊겨 여러 조각으로 잡힌 같은 줄은 하나로 합친다:
    # 양옆 간격보다 훨씬 좁은 간격은 같은 줄로 본다
    while len(ts) >= 3:
        g = np.diff(ts)
        ratio = [g[i] / min(g[j] for j in (i - 1, i + 1) if 0 <= j < len(g)) for i in range(len(g))]
        i = int(np.argmin(ratio))
        if ratio[i] >= 0.45 and g[i] >= 3.0:
            break
        ts[i:i + 2] = [(ts[i] + ts[i + 1]) / 2]
    ts = np.array(ts)
    if len(ts) < cfg.min_bars:
        return _fail(f"서로 다른 흰 줄이 {len(ts)}개만 보임 (최소 {cfg.min_bars}개)")

    # 2. 같은 간격 줄들의 사진 속 위치 t_k = (a k + b) / (c k + 1) → k→∞ 가 두 번째 소실점
    #    간격이 이웃의 약 2배면 줄 하나를 못 찾은 것으로 보고 k를 2칸 건너뛴다
    gaps = np.diff(ts)
    steps = []
    for i, g in enumerate(gaps):
        nb = [gaps[j] for j in (i - 1, i + 1) if 0 <= j < len(gaps)]
        ratio = g / (min(nb) if nb else g)
        steps.append(2 if 1.7 < ratio < 2.4 else 1)
    k = np.r_[0, np.cumsum(steps)].astype(float)
    (a, b, c_), *_ = np.linalg.lstsq(np.column_stack([k, np.ones_like(k), -k * ts]), ts, rcond=None)
    pred = (a * k + b) / (c_ * k + 1)
    unit = np.diff(ts) / np.diff(k)  # 줄 한 칸 간격 (멀수록 좁음)
    local = np.r_[unit[0], (unit[:-1] + unit[1:]) / 2, unit[-1]]
    if (np.abs(pred - ts) > 0.3 * local).any():
        return _fail("흰 줄 간격이 고르지 않음 (빠진 줄 또는 오검출)")
    v_cross = np.array([c_ * p0[0] + a * d[0], c_ * p0[1] + a * d[1], c_])
    v_bar = _vanishing_point([np.cross([*c, 1.0], [*(c + u), 1.0]) for c, u in bars])

    # 3. 지평선 → 지면 법선 → 내려다보는 카메라로 회전
    f = cfg.focal_ratio * max(w, h)
    K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]])
    horizon = np.cross(v_bar, v_cross)
    if np.linalg.norm(horizon) < 1e-12:
        return _fail("지평선 계산 실패")
    n = K.T @ horizon
    n /= np.linalg.norm(n)
    ys, xs = np.nonzero(cw)
    side = np.c_[xs, ys, np.ones_like(xs)] @ horizon
    if min((side > 0).mean(), (side < 0).mean()) > 0.02:
        return _fail("지평선이 횡단보도를 가로지름 (줄 검출 오류)")
    if n @ np.linalg.solve(K, [xs.mean(), ys.mean(), 1.0]) < 0:
        n = -n
    tilt = math.degrees(math.acos(np.clip(n[2], -1, 1)))
    if tilt > cfg.max_tilt_deg:
        return _fail(f"지면과 거의 나란히 찍힘 ({tilt:.0f}°)")
    axis = np.cross(n, [0, 0, 1.0])
    ang = math.acos(np.clip(n[2], -1, 1))
    R = cv2.Rodrigues(axis / np.linalg.norm(axis) * ang)[0] if np.linalg.norm(axis) > 1e-9 else np.eye(3)
    Hw = K @ R @ np.linalg.inv(K)  # 작업 크기 사진 → 보정 평면

    # 4. 보정 평면에서 검증: 줄들이 평행하고 간격이 고른가
    angs, offs = [], []
    for c, u in bars:
        q = apply_h(Hw, np.array([c - 20 * u, c + 20 * u]))
        angs.append(math.atan2(q[1, 1] - q[0, 1], q[1, 0] - q[0, 0]))
    z = np.exp(2j * np.array(angs)).mean()
    t_ang = np.angle(z) / 2
    ang_std = math.degrees(math.sqrt(max(0.0, -2 * math.log(max(abs(z), 1e-9)))) / 2)
    if ang_std > cfg.max_bar_angle_std:
        return _fail(f"보정 후에도 흰 줄이 평행하지 않음 (흩어짐 {ang_std:.1f}°)")
    nrm = np.array([-math.sin(t_ang), math.cos(t_ang)])
    offs = np.sort(apply_h(Hw, np.array([c for c, _ in bars])) @ nrm)
    gaps = np.diff(offs)
    if gaps.mean() <= 0 or gaps.std() / gaps.mean() > cfg.max_gap_cv:
        return _fail("보정 후 흰 줄 간격이 고르지 않음")

    S = np.diag([s, s, 1.0])
    return Rectification(
        ok=True,
        homography=(Hw @ S).tolist(),  # 원본 사진 픽셀 → 보정 평면
        crosswalk_axis_deg=angle_std(*nrm),  # 줄 긴 축(T)에 수직 = 건너는 방향(P)
        px_per_m=float(gaps.mean() / cfg.stripe_period_m),
    )


# ---------------------------------------------------------------------------
# ROI
# ---------------------------------------------------------------------------


@dataclass
class RoiConfig:
    patch_px: int = 128
    entrance_gap: float = 0.03  # 사진 대각선 대비: 블록이 이만큼 안에 있으면 진입부
    entrance_offset_m: float = 0.15
    entrance_size_m: float = 0.4
    approach_offset_m: float = 1.0
    approach_size_m: float = 0.6
    approach_scan_m: float = 1.5  # 선형 구간을 찾을 때 옆으로 훑는 범위
    min_block_cover: float = 0.5  # ROI 안 블록 비율
    min_source_px: int = 24  # ROI가 원본 사진에서 이 폭보다 작으면 무늬를 못 읽음
    period_range_m: tuple[float, float] = (0.03, 0.12)  # 돌출선·점 간격
    min_texture: float = 0.12  # 위 주파수 대역 에너지 비율
    linear_ratio: float = 3.0  # 같은 간격에서 주 방향 / 직각 방향 에너지
    dot_ratio: float = 1.8


def _cover(blocks, Hinv, center, side_m, px_per_m, n=32) -> float:
    k = side_m * px_per_m / n
    M = Hinv @ np.array([[k, 0, center[0] - k * n / 2], [0, k, center[1] - k * n / 2], [0, 0, 1.0]])
    return float(cv2.warpPerspective(blocks.astype(np.uint8), M, (n, n),
                                     flags=cv2.WARP_INVERSE_MAP | cv2.INTER_NEAREST).mean())


def _texture(gray, blocks, Hinv, center, side_m, px_per_m, cfg: RoiConfig):
    """보정 평면의 정사각형 칸 → (블록 종류, 돌출선 각도, 신뢰도)."""
    N = cfg.patch_px
    k = side_m * px_per_m / N
    A = np.array([[k, 0, center[0] - k * N / 2], [0, k, center[1] - k * N / 2], [0, 0, 1.0]])
    M = Hinv @ A  # 칸 픽셀 → 원본 사진
    corners = apply_h(M, np.array([[0, 0], [N, 0], [N, N], [0, N]], float))
    src_side = math.sqrt(abs(cv2.contourArea(corners.astype(np.float32))))
    flags = cv2.WARP_INVERSE_MAP
    bm = cv2.warpPerspective(blocks.astype(np.uint8), M, (N, N), flags=flags | cv2.INTER_NEAREST)
    if bm.mean() < cfg.min_block_cover or src_side < cfg.min_source_px:
        return None, None, 0.0
    patch = cv2.warpPerspective(gray, M, (N, N), flags=flags | cv2.INTER_LINEAR).astype(np.float32)
    patch -= cv2.GaussianBlur(patch, (0, 0), N / 16)
    win = np.outer(np.hanning(N), np.hanning(N)) * cv2.erode(bm, np.ones((5, 5), np.uint8))
    P = np.abs(np.fft.fftshift(np.fft.fft2(patch * win))) ** 2
    fy, fx = np.mgrid[-N // 2:N // 2, -N // 2:N // 2]
    r = np.hypot(fx, fy)
    lo, hi = side_m / cfg.period_range_m[1], side_m / cfg.period_range_m[0]
    band = (r >= lo) & (r <= hi)
    total = P[r >= 2].sum()
    if total <= 0 or P[band].sum() / total < cfg.min_texture:
        return None, None, 0.0
    # 가장 강한 주파수 하나를 찾고, 같은 간격(반지름)에서 직각 방향 에너지와 비교한다.
    # 선형: 직각 쪽이 훨씬 약함 / 점형(정사각 격자): 두 방향이 비슷함.
    # 블록 경계(30cm)나 돌출선 끝의 가로 성분은 다른 반지름에 있어 영향을 덜 받는다.
    ang = np.arctan2(fy, fx)
    Pb = np.where(band, P, 0)
    iy, ix = np.unravel_index(np.argmax(Pb), P.shape)
    r0, phi0 = r[iy, ix], ang[iy, ix]
    ring = band & (np.abs(r - r0) <= 1.5)

    def wedge(phi, deg=15):
        d = np.abs((ang - phi + np.pi / 2) % np.pi - np.pi / 2)
        return ring & (d <= np.radians(deg))

    w_par = wedge(phi0)
    e_par, e_orth = P[w_par].sum(), P[wedge(phi0 + np.pi / 2)].sum()
    ratio = e_par / (e_orth + 1e-9)
    if ratio >= cfg.linear_ratio:
        z = (P[w_par] * np.exp(2j * ang[w_par])).sum()  # 쐐기 안 가중 평균으로 각도를 세밀하게
        phi = np.angle(z) / 2  # 주파수 방향 → 돌출선은 그에 수직
        return MaskClass.LINEAR, angle_std(-math.sin(phi), math.cos(phi)), float(1 - 1 / ratio)
    if ratio <= cfg.dot_ratio:
        return MaskClass.DOT, None, float(np.clip(1 - (ratio - 1) / (cfg.dot_ratio - 1 + 1e-9) / 2, 0.5, 1.0))
    return None, None, 0.0


def _vote(votes):
    """[(블록, 각도, 신뢰도)] → 과반이 같은 종류면 그 종류 (각도는 평균), 아니면 판단 안 함."""
    for kind in (MaskClass.LINEAR, MaskClass.DOT):
        same = [v for v in votes if v[0] == kind]
        if len(same) * 2 > len(votes):
            conf = float(np.mean([v[2] for v in same]))
            if kind == MaskClass.DOT:
                return kind, None, conf
            z = np.mean([np.exp(2j * np.radians(v[1])) for v in same])
            return kind, float(np.degrees(np.angle(z) / 2) % 180), conf
    return None, None, 0.0


def analyze_rois(
    image_bgr: np.ndarray,
    label: np.ndarray,
    path: PathStructure,
    photo: PhotoKind,
    rect: Rectification | None = None,
    cfg: Optional[RoiConfig] = None,
) -> list[RoiAnalysis]:
    """넓은 사진: 횡단보도 진입부(entrance)와 그 뒤 선형 구간(approach) ROI.

    TODO(M3): 가까운 사진의 꺾임·분기점 ROI (role="node")
    """
    cfg = cfg or RoiConfig()
    if photo != PhotoKind.WIDE or rect is None or not rect.ok or not rect.px_per_m:
        return []
    h, w = label.shape[:2]
    gap = max(1, round(cfg.entrance_gap * math.hypot(w, h)))
    cw = (label == MaskClass.CROSSWALK).astype(np.uint8)
    blocks = np.isin(label, [int(c) for c in BLOCK_CLASSES])
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * gap + 1, 2 * gap + 1))
    near = cv2.dilate(cw, kern).astype(bool) & blocks
    if not near.any() or not cw.any():
        return []

    H = np.array(rect.homography)
    Hinv = np.linalg.inv(H)
    rng = np.random.default_rng(0)

    def centroid(mask):
        ys, xs = np.nonzero(mask)
        i = rng.choice(len(xs), min(len(xs), 4000), replace=False)
        return apply_h(H, np.c_[xs[i], ys[i]].astype(float)).mean(0)

    E, C = centroid(near), centroid(cw.astype(bool))
    a = math.radians(rect.crosswalk_axis_deg)
    axis = np.array([math.cos(a), -math.sin(a)])
    away = axis if (E - C) @ axis >= 0 else -axis  # 횡단보도에서 인도 쪽으로

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    side_dir = np.array([-away[1], away[0]])
    out = []
    for role, off, size in (("entrance", cfg.entrance_offset_m, cfg.entrance_size_m),
                            ("approach", cfg.approach_offset_m, cfg.approach_size_m)):
        center = E + away * off * rect.px_per_m
        if role == "approach":  # 진입부 점형 띠가 넓으면 선형 길이 띠 중심과 어긋나므로 옆으로 훑어 블록이 가장 많은 곳
            cands = [center + side_dir * dl * rect.px_per_m for dl in np.arange(-cfg.approach_scan_m, cfg.approach_scan_m + 1e-6, 0.1)]
            center = max(cands, key=lambda c: _cover(blocks, Hinv, c, size, rect.px_per_m))
        # 칸 크기에 따라 블록 경계가 걸치는 정도가 달라 결과가 흔들리므로 3가지 크기로 읽고 다수결
        votes = [_texture(gray, blocks, Hinv, center, size * f, rect.px_per_m, cfg) for f in (0.75, 1.0, 1.25)]
        block, ridge, conf = _vote(votes)
        cx, cy = apply_h(Hinv, center[None])[0]
        out.append(RoiAnalysis(
            photo=photo, role=role, center=Point(x=float(cx), y=float(cy)),
            size_px=round(size * rect.px_per_m), block=block, ridge_angle_deg=ridge, confidence=conf,
        ))
    return out
