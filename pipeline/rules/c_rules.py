"""C 계열: 넓은 사진 문맥 판정. (담당: 은재)

판정 순서 (노션 5장)
1. 방향 판정 먼저: 블록 경로 끝점의 연장선(레이)을 긋는다
   - 연장선이 횡단보도 밖 차도에 닿음 → C2 (다른 블록으로 횡단보도에 이어져 있어도 오류)
2. 연결 판정: 블록이 횡단보도 입구에 닿는가
   - 닿고, 경로가 횡단보도 안으로 들어가는 방향 → 정상 (C2가 없을 때만 C1 정상 표시)
   - 닿긴 하지만 옆을 스쳐 지나갈 뿐 (연석 따라 지나가는 선형 등) → C3
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
    arrive_len: float = 0.15  # 끝점·분기점에서 이 거리 안에 횡단보도(부풀리지 않은 원래 영역)가 있으면 '들어감'
    arrive_min_edge: float = 0.04  # '들어감' 판단에 쓰는 중심선 구간의 최소 길이 (계단 모양 점형 띠의 짧은 꺾임 제외)
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


def _path_rays(wide: PhotoAnalysis, cfg: ContextConfig, min_edge_px: float = 0.0):
    """(노드, 방향) 목록: 끝점은 바깥 방향, 꺾임·분기점은 각 엣지의 직진 방향.

    min_edge_px 보다 짧은 중심선 구간에서 나온 방향은 뺀다.
    """
    for n in wide.path.nodes:
        if n.type == NodeType.END:
            edges = wide.path.edges_of(n.id)
            if edges and edges[0].length_px < min_edge_px:
                continue
            d = end_direction(wide.path, n.id, cfg.end_dir_px)
            if d:
                yield n, d
        elif n.type in (NodeType.TURN, NodeType.BRANCH):
            for e in wide.path.edges_of(n.id):
                if e.length_px < min_edge_px:
                    continue
                d = _edge_continuation(wide.path, n.id, e, cfg.end_dir_px)
                if d:
                    yield n, d


def arrives_at_crosswalk(wide: PhotoAnalysis, label: np.ndarray, cfg: ContextConfig) -> bool:
    """경로가 횡단보도 '안으로 들어가는' 방향인가.

    입구에 닿았는지(cw_near)만 보면 연석을 따라 옆으로 지나가는 선형도 정상이 된다.
    끝점·분기점에서 진행 방향으로 짧게 그은 선이 실제 횡단보도에 닿아야 들어간다고 본다.
    """
    h, w = label.shape[:2]
    cw = label == MaskClass.CROSSWALK
    none = np.zeros_like(cw)
    diag = math.hypot(w, h)
    max_len = cfg.arrive_len * diag
    for n, d in _path_rays(wide, cfg, cfg.arrive_min_edge * diag):
        hit, _ = _march(n.pt.x, n.pt.y, d, max_len, cw, none, cfg.ray_step_px)
        if hit == MaskClass.CROSSWALK:
            return True
    return False


def _nearest_end(wide: PhotoAnalysis, cw: np.ndarray) -> Optional[Point]:
    ys, xs = np.nonzero(cw)
    cx, cy = float(xs.mean()), float(ys.mean())
    ends = [n for n in wide.path.nodes if n.type == NodeType.END]
    if not ends:
        return None
    n = min(ends, key=lambda n: math.dist((n.pt.x, n.pt.y), (cx, cy)))
    return Point(x=n.pt.x, y=n.pt.y)


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
        if arrives_at_crosswalk(wide, wide_label, cfg):
            if not has_c2:
                findings.append(Finding(
                    code="C1", verdict=Verdict.NORMAL, photo=PhotoKind.WIDE,
                    message="블록 경로가 횡단보도 입구까지 이어짐", confidence=0.6,
                ))
        elif not has_c2:  # 닿긴 하지만 옆을 스쳐 지나감
            findings.append(Finding(
                code="C3", verdict=Verdict.SUSPECTED, photo=PhotoKind.WIDE,
                message="블록이 횡단보도 옆을 지나갈 뿐, 횡단보도 쪽으로 들어가는 안내가 없음",
                location=_nearest_end(wide, cw), clause=CLAUSES["C3"], suggestion=SUGGESTIONS["C3"],
                confidence=0.55,
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
    findings.append(Finding(
        code="C3", verdict=Verdict.SUSPECTED, photo=PhotoKind.WIDE,
        message="옆에 횡단보도가 있는데 그쪽으로 꺾는 안내 없이 직진만 표시됨",
        location=_nearest_end(wide, cw), clause=CLAUSES["C3"], suggestion=SUGGESTIONS["C3"], confidence=0.6,
    ))
    return findings


@dataclass
class C4Config:
    max_ridge_dev_deg: float = 20.0  # 선형 돌출선과 횡단보도 진행축의 허용 각도 차
    min_conf: float = 0.5
    # 시점 보정 실패 시 라벨로 진입부 점형 확인
    entrance_gap: float = 0.03  # 사진 대각선 대비 (ContextConfig.entrance_gap과 같음)
    min_entrance_px: float = 0.002  # 진입부 블록이 사진 면적의 이만큼은 있어야 판단
    max_dot_share: float = 0.15  # 진입부 블록 중 점형 비율이 이보다 낮으면 점형 누락


def check_c4_from_label(wide: PhotoAnalysis, label: Optional[np.ndarray],
                        cfg: Optional[C4Config] = None) -> list[Finding]:
    """시점 보정이 실패했을 때: 횡단보도 입구에 닿은 블록이 거의 선형뿐이면 점형 누락(C4).

    무늬(FFT) 대신 모델이 준 선형/점형 라벨을 믿는 방식이라 신뢰도를 낮게 둔다.
    돌출선 방향은 보정 없이는 못 보므로 H2(방향 판정 보류)는 그대로 남는다.
    """
    if label is None:
        return []
    cfg = cfg or C4Config()
    h, w = label.shape[:2]
    gap = max(1, round(cfg.entrance_gap * math.hypot(w, h)))
    cw = (label == MaskClass.CROSSWALK).astype(np.uint8)
    if not cw.any():
        return []
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * gap + 1, 2 * gap + 1))
    near = cv2.dilate(cw, k).astype(bool)
    lin = int((near & (label == MaskClass.LINEAR)).sum())
    dot = int((near & (label == MaskClass.DOT)).sum())
    if lin + dot < cfg.min_entrance_px * h * w or dot / (lin + dot) >= cfg.max_dot_share:
        return []
    ys, xs = np.nonzero(near & (label == MaskClass.LINEAR))
    return [Finding(
        code="C4", verdict=Verdict.SUSPECTED, photo=PhotoKind.WIDE,
        message="횡단보도 진입부에 점형 블록이 없고 선형만 깔려 있음 (시점 보정 없이 블록 종류로 판단)",
        location=Point(x=float(xs.mean()), y=float(ys.mean())),
        clause=CLAUSES["C4"], suggestion=SUGGESTIONS["C4"], confidence=0.45,
    )]


def check_c4(wide: PhotoAnalysis, cfg: Optional[C4Config] = None) -> list[Finding]:
    """C4: 횡단보도 진입부 점형 유무 + 진입 직전 선형 돌출선 방향 vs 횡단보도 진행축.

    시점 보정(rectify.py)이 성공했을 때만 judge에서 불린다. 진입부 무늬를 읽지 못하면 H2로 보류.
    """
    from ..rectify import angle_diff  # 순환 import 방지

    cfg = cfg or C4Config()
    rect = wide.rectification
    if rect is None or not rect.ok or rect.crosswalk_axis_deg is None:
        return []
    ent = next((r for r in wide.rois if r.role == "entrance"), None)
    app = next((r for r in wide.rois if r.role == "approach"), None)
    if ent is None and app is None:  # 블록이 횡단보도에 닿지 않음 → C1에서 이미 다룸
        return []

    findings: list[Finding] = []
    checked: list[str] = []
    if ent and ent.block is not None and ent.confidence >= cfg.min_conf:
        checked.append("진입부 점형")
        if ent.block == MaskClass.LINEAR:
            findings.append(Finding(
                code="C4", verdict=Verdict.SUSPECTED, photo=PhotoKind.WIDE,
                message="횡단보도 진입부에 점형 블록이 없고 선형만 깔려 있음",
                location=ent.center, clause=CLAUSES["C4"], suggestion=SUGGESTIONS["C4"],
                confidence=round(0.7 * ent.confidence, 2),
            ))
    if app and app.block == MaskClass.LINEAR and app.ridge_angle_deg is not None and app.confidence >= cfg.min_conf:
        checked.append("진행 방향")
        dev = angle_diff(app.ridge_angle_deg, rect.crosswalk_axis_deg)
        if dev > cfg.max_ridge_dev_deg:
            findings.append(Finding(
                code="C4", verdict=Verdict.SUSPECTED, photo=PhotoKind.WIDE,
                message=f"진입부 선형 블록 돌출선이 횡단보도 건너는 방향과 {dev:.0f}° 어긋남",
                location=app.center, clause=CLAUSES["C4"], suggestion=SUGGESTIONS["C4"],
                confidence=round(0.7 * app.confidence, 2),
            ))
    if findings:
        return findings
    if checked:
        return [Finding(
            code="C4", verdict=Verdict.NORMAL, photo=PhotoKind.WIDE,
            message=f"횡단보도 진입부 확인 정상 ({', '.join(checked)})", confidence=0.6,
        )]
    return [Finding(
        code="H2", verdict=Verdict.HOLD, photo=PhotoKind.WIDE,
        message="진입부 블록 무늬가 작거나 흐려 진입 방향(C4) 판정 보류. 진입부가 더 크게 나오게 찍어 주세요",
    )]
