"""③ 경로 구조 추출: 블록 마스크 → 중심선 → 끝점·꺾임점·분기점 그래프. (담당: A)

순서
1. 선형+점형 영역을 합쳐 모폴로지 닫기(작은 틈 메우기), 작은 덩어리 제거
2. 골격화(skeletonize) → 픽셀 그래프
3. 차수 2 픽셀 사슬을 엣지 하나로 묶기 (차수 ≠ 2 픽셀이 노드)
4. 짧은 가지(끝점–분기점) 제거, 남은 차수 2 노드는 합치기
5. 엣지를 Douglas–Peucker로 단순화하고 각도 변화가 큰 꼭짓점을 꺾임점으로 분리
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import cv2
import networkx as nx
import numpy as np
from skimage.morphology import remove_small_objects, skeletonize

from .schema import (
    BLOCK_CLASSES,
    MaskClass,
    NodeType,
    PathEdge,
    PathNode,
    PathStructure,
    PhotoKind,
    Point,
)

Pixel = tuple[int, int]  # (y, x)


@dataclass
class SkeletonConfig:
    close_px: int = 15  # 이 크기 이하의 틈은 메움 (리스크: 마스크 끊김 → 연결 오판)
    min_area_px: int = 400  # 이보다 작은 블록 덩어리는 버림
    spur_px: float = 40.0  # 이보다 짧은 끝점–분기점 가지는 가짜 가지로 보고 제거
    dp_epsilon_px: float = 6.0  # 중심선 단순화 허용 오차
    turn_deg: float = 30.0  # 방향 변화가 이보다 크면 꺾임점
    node_window_px: int = 9  # 노드 블록 종류를 볼 때 창 크기 (반지름)


def block_mask(label: np.ndarray) -> np.ndarray:
    return np.isin(label, [int(c) for c in BLOCK_CLASSES])


def extract_path(
    label: np.ndarray, photo: PhotoKind, cfg: Optional[SkeletonConfig] = None
) -> PathStructure:
    """라벨 마스크(MaskClass 값)에서 경로 구조를 뽑는다."""
    cfg = cfg or SkeletonConfig()
    h, w = label.shape[:2]

    mask = block_mask(label).astype(np.uint8)
    if cfg.close_px > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg.close_px, cfg.close_px))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    mask = remove_small_objects(mask.astype(bool), max_size=cfg.min_area_px)

    n_comp, comp_map = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    component_area = {
        int(c): int((comp_map == c).sum()) for c in range(1, n_comp)
    }

    skel = skeletonize(mask)
    graph = _pixel_graph(skel)
    nodes, edges = _contract(graph)
    nodes, edges = _prune_spurs(nodes, edges, cfg.spur_px)
    nodes, edges = _merge_degree2(nodes, edges)
    nodes, edges, turn_ids = _split_turns(nodes, edges, cfg)

    return _to_schema(label, photo, w, h, nodes, edges, turn_ids, comp_map, component_area, cfg)


def end_direction(path: PathStructure, node_id: int, length_px: float = 80.0) -> Optional[tuple[float, float]]:
    """끝점에서 바깥쪽을 향하는 단위 벡터 (dx, dy). 중심선 마지막 구간 기준.

    방향 판정(C2·C3)에서 선형 줄의 연장선을 그을 때 쓴다.
    """
    edges = path.edges_of(node_id)
    if len(edges) != 1:
        return None
    e = edges[0]
    pts = e.polyline if e.u == node_id else list(reversed(e.polyline))
    tip = pts[0]
    acc, ref = 0.0, pts[-1]
    for a, b in zip(pts, pts[1:]):
        acc += math.dist((a.x, a.y), (b.x, b.y))
        if acc >= length_px:
            ref = b
            break
    dx, dy = tip.x - ref.x, tip.y - ref.y
    n = math.hypot(dx, dy)
    return (dx / n, dy / n) if n > 0 else None


# ---------------------------------------------------------------------------
# 내부 구현
# ---------------------------------------------------------------------------


def _pixel_graph(skel: np.ndarray) -> nx.Graph:
    """골격 픽셀 8-이웃 그래프. 계단 모양에서 가짜 분기가 생기지 않게,
    공유하는 4-이웃이 있으면 대각선 연결은 만들지 않는다."""
    pix = set(zip(*np.nonzero(skel)))
    g = nx.Graph()
    g.add_nodes_from(pix)
    for y, x in pix:
        for dy, dx in ((0, 1), (1, 0)):
            q = (y + dy, x + dx)
            if q in pix:
                g.add_edge((y, x), q)
        for dy, dx in ((1, 1), (1, -1)):
            q = (y + dy, x + dx)
            if q in pix and (y + dy, x) not in pix and (y, x + dx) not in pix:
                g.add_edge((y, x), q)
    return g


# 작업용 표현: nodes = {id: [pixels]}, edges = {id: (u, v, [pixels u→v])}
Nodes = dict[int, list[Pixel]]
Edges = dict[int, tuple[int, int, list[Pixel]]]


def _contract(g: nx.Graph) -> tuple[Nodes, Edges]:
    cluster_of: dict[Pixel, int] = {}
    nodes: Nodes = {}

    junction = g.subgraph([p for p in g if g.degree(p) >= 3])
    for comp in nx.connected_components(junction):
        nid = len(nodes)
        nodes[nid] = list(comp)
        for p in comp:
            cluster_of[p] = nid
    for p in g:
        if g.degree(p) <= 1:
            nid = len(nodes)
            nodes[nid] = [p]
            cluster_of[p] = nid
    # 끝점·분기점이 없는 고리: 아무 픽셀이나 기준점으로
    for comp in nx.connected_components(g):
        if not any(p in cluster_of for p in comp):
            p = min(comp)
            nid = len(nodes)
            nodes[nid] = [p]
            cluster_of[p] = nid

    edges: Edges = {}
    seen: set[frozenset] = set()
    for start in list(cluster_of):
        for q in g.neighbors(start):
            if cluster_of.get(q) == cluster_of[start] or frozenset((start, q)) in seen:
                continue
            seen.add(frozenset((start, q)))
            path = [start, q]
            prev, cur = start, q
            while cur not in cluster_of:
                nxt = [n for n in g.neighbors(cur) if n != prev]
                if not nxt:
                    break
                seen.add(frozenset((cur, nxt[0])))
                path.append(nxt[0])
                prev, cur = cur, nxt[0]
            if cur in cluster_of:
                edges[len(edges)] = (cluster_of[start], cluster_of[cur], path)
    return nodes, edges


def _degree(edges: Edges, nid: int) -> int:
    return sum((u == nid) + (v == nid) for u, v, _ in edges.values())


def _length(path: list[Pixel]) -> float:
    return sum(math.dist(a, b) for a, b in zip(path, path[1:]))


def _prune_spurs(nodes: Nodes, edges: Edges, spur_px: float) -> tuple[Nodes, Edges]:
    changed = True
    while changed:
        changed = False
        for eid, (u, v, path) in list(edges.items()):
            du, dv = _degree(edges, u), _degree(edges, v)
            if u == v or _length(path) >= spur_px:
                continue
            if (du == 1 and dv >= 3) or (dv == 1 and du >= 3):
                del edges[eid]
                nodes.pop(u if du == 1 else v)
                changed = True
                break
    return nodes, edges


def _merge_degree2(nodes: Nodes, edges: Edges) -> tuple[Nodes, Edges]:
    """가지 제거 후 차수 2가 된 노드를 없애고 양쪽 엣지를 이어 붙인다."""
    changed = True
    while changed:
        changed = False
        for nid in list(nodes):
            inc = [eid for eid, (u, v, _) in edges.items() if nid in (u, v)]
            if len(inc) != 2:
                continue
            (u1, v1, p1), (u2, v2, p2) = edges[inc[0]], edges[inc[1]]
            if u1 == v1 or u2 == v2:
                continue  # 고리는 그대로 둔다
            a = p1 if v1 == nid else list(reversed(p1))  # ... → nid
            b = p2 if u2 == nid else list(reversed(p2))  # nid → ...
            start = u1 if v1 == nid else v1
            end = v2 if u2 == nid else u2
            del edges[inc[0]], edges[inc[1]]
            edges[max(edges, default=-1) + 1] = (start, end, a + b[1:])
            nodes.pop(nid)
            changed = True
            break
    return nodes, edges


def _turn_angle(a: Pixel, b: Pixel, c: Pixel) -> float:
    v1 = (b[0] - a[0], b[1] - a[1])
    v2 = (c[0] - b[0], c[1] - b[1])
    n1, n2 = math.hypot(*v1), math.hypot(*v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    cos = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)))
    return math.degrees(math.acos(cos))


def _simplify(path: list[Pixel], eps: float) -> list[Pixel]:
    if len(path) < 3:
        return path
    arr = np.array([[x, y] for y, x in path], dtype=np.float32).reshape(-1, 1, 2)
    approx = cv2.approxPolyDP(arr, eps, closed=False).reshape(-1, 2)
    return [(int(y), int(x)) for x, y in approx]


def _split_turns(nodes: Nodes, edges: Edges, cfg: SkeletonConfig) -> tuple[Nodes, Edges, set[int]]:
    turn_ids: set[int] = set()
    out: Edges = {}
    for u, v, path in edges.values():
        pts = _simplify(path, cfg.dp_epsilon_px)
        cur_start, seg = u, [pts[0]]
        for i in range(1, len(pts) - 1):
            seg.append(pts[i])
            if _turn_angle(pts[i - 1], pts[i], pts[i + 1]) > cfg.turn_deg:
                nid = max(nodes) + 1
                nodes[nid] = [pts[i]]
                turn_ids.add(nid)
                out[len(out)] = (cur_start, nid, seg)
                cur_start, seg = nid, [pts[i]]
        seg.append(pts[-1])
        out[len(out)] = (cur_start, v, seg)
    return nodes, out, turn_ids


def _mode_block(values: np.ndarray) -> Optional[MaskClass]:
    vals = values[np.isin(values, [int(c) for c in BLOCK_CLASSES])]
    if vals.size == 0:
        return None
    return MaskClass(int(np.bincount(vals).argmax()))


def _to_schema(
    label: np.ndarray,
    photo: PhotoKind,
    w: int,
    h: int,
    nodes: Nodes,
    edges: Edges,
    turn_ids: set[int],
    comp_map: np.ndarray,
    component_area: dict[int, int],
    cfg: SkeletonConfig,
) -> PathStructure:
    r = cfg.node_window_px
    out_nodes: list[PathNode] = []
    for nid, pix in nodes.items():
        cy = float(np.mean([p[0] for p in pix]))
        cx = float(np.mean([p[1] for p in pix]))
        deg = _degree(edges, nid)
        if deg == 0:
            ntype = NodeType.ISOLATED
        elif deg == 1:
            ntype = NodeType.END
        elif deg >= 3:
            ntype = NodeType.BRANCH
        else:
            ntype = NodeType.TURN if nid in turn_ids else NodeType.PASS
        iy, ix = int(round(cy)), int(round(cx))
        window = label[max(0, iy - r) : iy + r + 1, max(0, ix - r) : ix + r + 1]
        out_nodes.append(
            PathNode(
                id=nid,
                type=ntype,
                pt=Point(x=cx, y=cy),
                block=_mode_block(window.ravel()),
                component=int(comp_map[iy, ix]),
            )
        )

    out_edges: list[PathEdge] = []
    for eid, (u, v, pts) in edges.items():
        samples = _dense(pts)
        vals = np.array([label[y, x] for y, x in samples if 0 <= y < h and 0 <= x < w])
        out_edges.append(
            PathEdge(
                id=eid,
                u=u,
                v=v,
                polyline=[Point(x=float(x), y=float(y)) for y, x in pts],
                length_px=_length(pts),
                block=_mode_block(vals) if vals.size else None,
            )
        )
    return PathStructure(
        photo=photo,
        width=w,
        height=h,
        nodes=out_nodes,
        edges=out_edges,
        component_area_px=component_area,
    )


def _dense(pts: list[Pixel]) -> list[Pixel]:
    """단순화된 꼭짓점 사이를 1px 간격으로 채운다 (블록 종류 샘플링용)."""
    out: list[Pixel] = []
    for (y0, x0), (y1, x1) in zip(pts, pts[1:]):
        n = max(abs(y1 - y0), abs(x1 - x0), 1)
        out += [(round(y0 + (y1 - y0) * t / n), round(x0 + (x1 - x0) * t / n)) for t in range(n)]
    out.append(pts[-1])
    return out
