"""C 계열: 넓은 사진 문맥 판정. (담당: 은재)

판정 순서 (노션 5장)
1. 방향 판정 먼저: 블록 경로 끝점의 연장선(레이)을 긋는다
   - 연장선이 횡단보도 밖 차도에 닿음 → C2 (다른 블록으로 횡단보도에 이어져 있어도 오류)
2. 연결 판정: 블록이 횡단보도 입구에 닿는가
   - 닿음 → 정상 (C2가 없을 때만 C1 정상 표시)
   - 연장선은 횡단보도를 향하는데 못 미침 → C1
   - 연장선이 횡단보도를 비켜가고(옆에 있음) 그쪽으로 이어진 블록 없음 → C3
     (C2와 함께 뜨면 judge에서 C3를 primary=False로 바꾼다)
연결·방향 판정은 시점 보정 없이 진행한다. C4만 보정이 필요하다.

넓은 사진에서는 선형·점형 구분을 믿지 않으므로 블록 전체(선형+점형)의 경로로 판정한다.
횡단보도가 없고 차도만 있으면 C2만 판정한다 (C1·C3는 judge에서 H3로 보류).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from ..schema import BLOCK_CLASSES, Finding, MaskClass, NodeType, PhotoAnalysis, PhotoKind, Point, Verdict
from ..skeleton import end_direction
from .clauses import CLAUSES, SUGGESTIONS


@dataclass
class ContextConfig:
    """길이는 모두 사진 대각선 대비 비율."""

    ray_len: float = 0.5  # 연장선 최대 길이
    entrance_gap: float = 0.03  # 블록이 횡단보도에서 이만큼 안에 있으면 '입구에 닿음'
    junction_ray_len: float = 0.08  # 꺾임·분기점에서 직진 연장선 길이 (연석 바로 앞에서 꺾인 경우)
    ray_step_px: float = 2.0
    end_dir_px: float = 80.0  # 끝 방향을 잴 중심선 구간 길이


@dataclass
class RayHit:
    node_id: int
    tip: tuple[float, float]
    direction: tuple[float, float]
    hit: Optional[MaskClass]  # CROSSWALK / ROAD / None
    hit_pt: Optional[tuple[float, float]]


def _masks(label: np.ndarray, gap_px: int):
    cw = (label == MaskClass.CROSSWALK).astype(np.uint8)
    road = label == MaskClass.ROAD
    blocks = np.isin(label, [int(c) for c in BLOCK_CLASSES])
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * gap_px + 1, 2 * gap_px + 1))
    cw_near = cv2.dilate(cw, k).astype(bool)  # 횡단보도 + 입구 허용 거리
    return cw.astype(bool), cw_near, road, blocks


def _edge_continuation(path, node_id: int, edge, length_px: float) -> Optional[tuple[float, float]]:
    """엣지를 따라 node로 들어온 방향 그대로 node 너머로 이어지는 단위 벡터."""
    pts = edge.polyline if edge.u == node_id else list(reversed(edge.polyline))
    tip, ref, acc = pts[0], pts[-1], 0.0
    for a, b in zip(pts, pts[1:]):
        acc += math.dist((a.x, a.y), (b.x, b.y))
        if acc >= length_px:
            ref = b
            break
    dx, dy = tip.x - ref.x, tip.y - ref.y
    n = math.hypot(dx, dy)
    return (dx / n, dy / n) if n > 0 else None


def _march(x0, y0, d, max_len, cw_near, road, step):
    h, w = road.shape
    t = 0.0
    while t <= max_len:
        x, y = x0 + d[0] * t, y0 + d[1] * t
        xi, yi = int(round(x)), int(round(y))
        if not (0 <= xi < w and 0 <= yi < h):
            break
        if cw_near[yi, xi]:  # 횡단보도 가장자리 근처는 횡단보도로 본다
            return MaskClass.CROSSWALK, (x, y)
        if road[yi, xi]:
            return MaskClass.ROAD, (x, y)
        t += step
    return None, None


def cast_rays(wide: PhotoAnalysis, label: np.ndarray, cfg: ContextConfig) -> list[RayHit]:
    """경로 끝점마다 바깥 방향 연장선을 따라가며 처음 닿는 횡단보도/차도를 찾는다.

    꺾임·분기점에서도 각 엣지의 직진 연장선을 짧게 본다. 선형 줄이 연석 바로 앞에서
    옆으로 꺾이면(연석 따라 깔린 블록) 끝점이 사라지므로, 이 경우의 차도 직진을 잡기 위함.
    """
    h, w = label.shape[:2]
    diag = math.hypot(w, h)
    gap = max(1, round(cfg.entrance_gap * diag))
    _, cw_near, road, _ = _masks(label, gap)

    out = []
    for n in wide.path.nodes:
        if n.type == NodeType.END:
            d = end_direction(wide.path, n.id, cfg.end_dir_px)
            dirs, max_len = ([d] if d else []), cfg.ray_len * diag
        elif n.type in (NodeType.TURN, NodeType.BRANCH):
            dirs = [d for e in wide.path.edges_of(n.id) if (d := _edge_continuation(wide.path, n.id, e, cfg.end_dir_px))]
            max_len = cfg.junction_ray_len * diag
        else:
            continue
        for d in dirs:
            hit, hit_pt = _march(n.pt.x, n.pt.y, d, max_len, cw_near, road, cfg.ray_step_px)
            out.append(RayHit(n.id, (n.pt.x, n.pt.y), d, hit, hit_pt))
    return out


def check_direction_and_connection(
    wide: PhotoAnalysis, wide_label: Optional[np.ndarray], cfg: Optional[ContextConfig] = None
) -> list[Finding]:
    """C1·C2·C3."""
    if wide_label is None:
        return []
    cfg = cfg or ContextConfig()
    h, w = wide_label.shape[:2]
    gap = max(1, round(cfg.entrance_gap * math.hypot(w, h)))
    cw, cw_near, _, blocks = _masks(wide_label, gap)
    if not blocks.any():
        return []

    rays = cast_rays(wide, wide_label, cfg)
    findings: list[Finding] = []

    # 1. 방향: 연장선이 횡단보도 밖 차도에 닿음 → C2
    seen: set[int] = set()
    for r in rays:
        if r.hit == MaskClass.ROAD and r.node_id not in seen:
            seen.add(r.node_id)
            findings.append(Finding(
                code="C2", verdict=Verdict.SUSPECTED, photo=PhotoKind.WIDE,
                message="블록이 횡단보도가 없는 차도 쪽으로 향함 (차도 직진 위험)",
                location=Point(x=r.tip[0], y=r.tip[1]),
                clause=CLAUSES["C2"], suggestion=SUGGESTIONS["C2"], confidence=0.7,
            ))
    has_c2 = bool(findings)

    if not cw.any():  # 횡단보도 없음: C1·C3는 판정하지 않음 (judge에서 H3)
        return findings

    # 2. 연결: 블록이 횡단보도 입구에 닿는가
    touching = bool((blocks & cw_near).any())
    if touching:
        if not has_c2:
            findings.append(Finding(
                code="C1", verdict=Verdict.NORMAL, photo=PhotoKind.WIDE,
                message="블록 경로가 횡단보도 입구까지 이어짐", confidence=0.6,
            ))
        return findings

    toward = [r for r in rays if r.hit == MaskClass.CROSSWALK]
    if toward:
        r = min(toward, key=lambda r: math.dist(r.tip, r.hit_pt))
        findings.append(Finding(
            code="C1", verdict=Verdict.SUSPECTED, photo=PhotoKind.WIDE,
            message="블록이 횡단보도 쪽을 향하지만 입구까지 이어지지 않음",
            location=Point(x=r.tip[0], y=r.tip[1]),
            clause=CLAUSES["C1"], suggestion=SUGGESTIONS["C1"], confidence=0.6,
        ))
        return findings

    # 연장선이 횡단보도를 비켜감 → 횡단보도에 가장 가까운 끝점에서 C3
    ys, xs = np.nonzero(cw)
    cx, cy = float(xs.mean()), float(ys.mean())
    ends = [n for n in wide.path.nodes if n.type == NodeType.END]
    if ends:
        n = min(ends, key=lambda n: math.dist((n.pt.x, n.pt.y), (cx, cy)))
        loc = Point(x=n.pt.x, y=n.pt.y)
    else:
        loc = None
    findings.append(Finding(
        code="C3", verdict=Verdict.SUSPECTED, photo=PhotoKind.WIDE,
        message="옆에 횡단보도가 있는데 그쪽으로 꺾는 안내 없이 직진만 표시됨",
        location=loc, clause=CLAUSES["C3"], suggestion=SUGGESTIONS["C3"], confidence=0.6,
    ))
    return findings


def check_c4(wide: PhotoAnalysis) -> list[Finding]:
    """C4: 진입부 ROI 돌출선 방향 vs 횡단보도 진행축, 진입부 점형 유무.

    TODO: wide.rectification.crosswalk_axis_deg 와 진입부 RoiAnalysis.ridge_angle_deg 비교.
    시점 보정(rectify.py)이 구현되기 전까지는 H2로 보류된다.
    """
    return []
