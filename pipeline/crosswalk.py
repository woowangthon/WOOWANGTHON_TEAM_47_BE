"""횡단보도 마스크 검증: 줄무늬가 없는 '횡단보도'는 버린다.

장면 모델이 가까이 찍힌 아스팔트·콘크리트를 횡단보도로 잘못 잡는 경우가 있다.
횡단보도는 흰 줄이 한 방향으로 반복되므로, 조각마다 두 가지를 잰다.

    coherence     경계선(그래디언트) 방향이 한 방향으로 모인 정도 (0~1)
    edge_density  뚜렷한 경계선 픽셀 비율

둘 다 낮으면(방향도 흩어지고 경계도 거의 없음) 줄무늬 없는 바닥으로 보고 배경으로 바꾼다.
기준값은 AI Hub 검증 세트 정답 횡단보도 301조각에서 진짜를 버리는 비율 0.7%,
실제 사진 오검출 2건(빨간 아스팔트, 밤 콘크리트)은 모두 걸러지는 값으로 정했다.

stripe_angle_deg(줄무늬가 뻗은 방향)는 C4 시점 보정에서 다시 쓴다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from .schema import MaskClass

WORK_LONG_SIDE = 800  # 사진 크기와 무관하게 같은 기준으로 재기 위해 이 크기로 줄여서 계산


@dataclass
class StripeConfig:
    min_coherence: float = 0.3
    min_edge_density: float = 0.08
    edge_mag: float = 40.0
    min_area: float = 0.002  # 사진 면적 대비. 이보다 작은 조각은 판단하지 않고 그대로 둔다


@dataclass
class StripeStats:
    coherence: float
    edge_density: float
    stripe_angle_deg: float  # 줄무늬가 뻗은 방향, 이미지 좌표 (x 오른쪽, y 아래), 0~180
    striped: bool


def stripe_stats(gray: np.ndarray, mask: np.ndarray, cfg: StripeConfig) -> StripeStats | None:
    """gray, mask는 WORK_LONG_SIDE 크기 기준."""
    inner = cv2.erode(mask.astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool)  # 마스크 테두리(연석·차선 경계 한 줄)는 빼고 잼
    if inner.sum() < 500:
        return None
    g = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 1.5)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)[inner]
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)[inner]
    mag = np.hypot(gx, gy)
    # 방향을 2배 각도로 더하면 반대 방향(밝→어둡, 어둡→밝) 경계가 서로 상쇄되지 않는다
    z = (mag * np.exp(2j * np.arctan2(gy, gx))).sum() / (mag.sum() + 1e-6)
    coh = float(abs(z))
    edge = float((mag > cfg.edge_mag).mean())
    grad_deg = math.degrees(np.angle(z) / 2)  # 경계선 법선 = 줄무늬에 수직
    return StripeStats(
        coherence=coh,
        edge_density=edge,
        stripe_angle_deg=(grad_deg + 90) % 180,
        striped=not (coh < cfg.min_coherence and edge < cfg.min_edge_density),
    )


def drop_unstriped(image_bgr: np.ndarray, label: np.ndarray, cfg: StripeConfig | None = None
                   ) -> tuple[np.ndarray, list[StripeStats]]:
    """줄무늬 없는 횡단보도 조각을 배경으로 바꾼 라벨과, 남은 조각들의 통계를 돌려준다.

    차도로 바꾸지 않는 이유: 잘못 바꾸면 C2(차도 직진) 오탐이 생긴다.
    """
    cfg = cfg or StripeConfig()
    h, w = label.shape[:2]
    cw = (label == MaskClass.CROSSWALK).astype(np.uint8)
    if not cw.any():
        return label, []

    s = min(1.0, WORK_LONG_SIDE / max(h, w))
    small = (round(w * s), round(h * s))
    gray = cv2.cvtColor(cv2.resize(image_bgr, small, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)

    out = label.copy()
    kept: list[StripeStats] = []
    n, cc, st, _ = cv2.connectedComponentsWithStats(cw)
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] < cfg.min_area * h * w:
            continue
        comp = cc == i
        stats = stripe_stats(gray, cv2.resize(comp.astype(np.uint8), small, interpolation=cv2.INTER_NEAREST), cfg)
        if stats is None:
            continue
        if stats.striped:
            kept.append(stats)
        else:
            out[comp] = MaskClass.BACKGROUND
    return out, kept
