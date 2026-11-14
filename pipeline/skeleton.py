"""③ 경로 구조 추출: 선형 마스크 골격 + 점형 영역 노드. (담당: A)

- 선형(linear)만 skeletonize하여 점형 타일 내부의 가짜 분기 제거
- 점형(dot) 각 연결 성분을 그래프의 한 노드로 취급
- 선형과 점형이 실제로 접하거나 작은 분할 경계 틈 이내인 경우만 연결
- 기존 PathStructure / PathNode / PathEdge JSON 스키마 유지
- 원본 라벨은 변경하지 않으며, M4의 실제 단절을 임의로 메우지 않음

주의: 본 모듈은 '구조 후보' 검출이다. M3 오류 판정은 별도 규칙에서 처리한다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import cv2
import networkx as nx
import numpy as np
from skimage.morphology import skeletonize

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
    close_px: int = 0  # 기본 꺼짐: M4 실제 단절 보존 (선형에만 적용)
    min_area_px: int = 400  # 선형 노이즈 제거 절대 최소 면적
    min_area_ratio: float = 0.00015  # 이미지 면적 대비 작은 선형 영역 제거 (3k 사진에서 1396px)
    min_dot_area_px: int = 400  # 점형 노이즈 제거 최소 면적
    dot_attach_px: int = 8  # 점형-선형 사이 허용할 최대 픽셀 틈 (보수적)
    spur_px: float = 0.0  # 0=선형 폭 기반 자동 결정, 양수=해당 픽셀 값으로 고정
    dp_epsilon_px: float = 6.0  # 중심선 단순화 허용 오차
    turn_deg: float = 45.0  # 선형의 꺾임 판단 각도
    turn_min_arm_px: float = 35.0  # 꺾임 앞뒤 팔이 최소한 이 길이는 되어야 함
    node_window_px: int = 9
    corridor_solidity: float = 0.90  # 점형 2영역에 접하는 볼록한 넓은 선형은 중앙 경로 사용
    corridor_min_eigen_ratio: float = 1.15  # 주 진행축이 식별되는 경우만
    corridor_turn_deg: float = 70.0  # 보간된 넓은 띠에서는 가짜 꺾임 후보를 더 엄격하게
    one_dot_corridor_eigen_ratio: float = 5.0  # 한쪽만 점형에 닿는 직선 띠에 한해 강화
    one_dot_corridor_solidity: float = 0.94  # ㄱ/T자나 울퉁불퉁한 영역의 오보정 방지


def block_mask(label: np.ndarray) -> np.ndarray:
    return np.isin(label, [int(c) for c in BLOCK_CLASSES])


def _keep_large_components(binary: np.ndarray, area_px: int) -> np.ndarray:
    """uint8 0/1 mask의 작은 연결 성분 제거. skimage 버전과 무관."""
    binary = (binary != 0).astype(np.uint8)
    if area_px <= 1:
        return binary
    count, component, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    keep = np.zeros(count, dtype=np.uint8)
    keep[1:] = (stats[1:, cv2.CC_STAT_AREA] >= area_px).astype(np.uint8)
    return keep[component]


def prepare_linear_mask(label: np.ndarray, cfg: Optional[SkeletonConfig] = None) -> np.ndarray:
    """Skeletonization에 사용하는 선형 마스크. 원본 label은 변경하지 않는다."""
    cfg = cfg or SkeletonConfig()
    binary = (label == MaskClass.LINEAR).astype(np.uint8)
    if cfg.close_px > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg.close_px, cfg.close_px))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k)
    min_area = max(cfg.min_area_px, int(binary.size * cfg.min_area_ratio))
    return _keep_large_components(binary, min_area)


def extract_path(
    label: np.ndarray, photo: PhotoKind, cfg: Optional[SkeletonConfig] = None
) -> PathStructure:
    """기존 함수 계약 유지: label(H,W; 0/1/2/3/4) -> PathStructure.

    선형 skeleton과 점형 연결 성분을 별개로 분석한다. 점형 접점은
    선형의 실제 끝점 근처만 연결한다(멀리 떨어진 설치 단절은 메우지 않음).
    """
    cfg = cfg or SkeletonConfig()
    if label.ndim != 2 or label.dtype != np.uint8:
        raise ValueError("label은 (H,W) uint8 단일 채널이어야 합니다")
    h, w = label.shape
    linear = prepare_linear_mask(label, cfg)
    dot = _keep_large_components((label == MaskClass.DOT).astype(np.uint8), cfg.min_dot_area_px)

    # 선형만으로 중심선 구성 → 점형 타일 내부의 가짜 가지/분기 자체가 생기지 않음
    n_line, line_components = cv2.connectedComponents(linear, connectivity=8)
    areas_line = {i: int(np.count_nonzero(line_components == i)) for i in range(1, n_line)}
    skel = skeletonize(linear.astype(bool))
    # 점형 두 영역을 잇는 넓고 볼록한 선형 띠는 medial-axis의 가짜 가지를
    # 단일 중앙 경로 후보로 대체한다. T/L 형태 등 복잡한 영역에는 적용하지 않는다.
    skel, refined_components = _refine_simple_corridors(skel, linear, dot, line_components, cfg)
    graph = _pixel_graph(skel)
    nodes, edges = _contract(graph)
    # 접촉 구역별로 가장 가까운 실제 선형 끝점을 먼저 결정한다.
    # 이 끝점만 가지치기로부터 보호해야 주변 가짜 가지들이 보존되지 않는다.
    attached_ends = _find_endpoint_attachments(nodes, edges, linear, dot, line_components, cfg)
    protected_ends = set(attached_ends)
    # 폭이 넓은 점자블록의 모서리에서 생긴 짧은 가짜 가지는
    # 단순 40px 고정보다 선형 폭에 비례해 제거하는 편이 안정적이다.
    if cfg.spur_px > 0:
        effective_spur_px = cfg.spur_px
    else:
        dist_map = cv2.distanceTransform(linear, cv2.DIST_L2, 3)
        radius_values = dist_map[skel]
        effective_spur_px = max(20.0, float(np.median(radius_values)) * 1.6) if radius_values.size else 20.0
    nodes, edges = _prune_spurs(nodes, edges, effective_spur_px, protected_ends)
    nodes, edges = _merge_degree2(nodes, edges)
    nodes, edges, turn_ids = _split_turns(nodes, edges, cfg, line_components, refined_components)
    path = _to_schema(label, photo, w, h, nodes, edges, turn_ids,
                      line_components, areas_line, cfg)
    for node in path.nodes:
        node.block = MaskClass.LINEAR

    # 점형 1 연결 성분 → 하나의 노드. 선형 끝점에 실제 접촉한 경우에만 연결.
    return _attach_dot_regions(path, linear, dot, line_components, attached_ends, cfg)


def _refine_simple_corridors(
    skeleton: np.ndarray,
    linear: np.ndarray,
    dot: np.ndarray,
    line_components: np.ndarray,
    cfg: SkeletonConfig,
) -> tuple[np.ndarray, set[int]]:
    """점형에 접한 충분히 곧고 볼록한 선형 띠만 중심축으로 근사한다.

    넓은 분할 영역의 medial axis가 가지를 만드는 경우에 대한 제한적 보정이다.
    참된 T/Y자 분기나 누락 사례를 모두 해결하지 못하므로 M3의 최종 정상 판정
    근거로 이 결과 하나만 사용하면 안 된다.
    """
    n_dot, dot_labels = cv2.connectedComponents(dot, connectivity=8)
    n_line = int(line_components.max())
    if n_dot < 2 or n_line < 1:
        return skeleton, set()
    out = skeleton.astype(np.uint8).copy()
    refined_components: set[int] = set()
    # YOLO가 만든 8px 안팎의 분할 경계 틈은 '연결 후보' 판단에만 사용.
    sz = 2 * max(cfg.dot_attach_px, 0) + 1
    dilated_dot = cv2.dilate(dot, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (sz, sz)))
    candidates = (dilated_dot != 0) & (linear != 0)
    for li in range(1, n_line + 1):
        region = (line_components == li).astype(np.uint8)
        if int(region.sum()) < 1500:
            continue
        # 인접 점형 연결 성분이 2개 이상인지 확인한다.
        # dot 마스크와 닿는 선형 경계에서 최근접 dot ID를 받는다.
        touching = (region != 0) & candidates
        if not np.any(touching):
            continue
        # dilated_dot 결과는 class ID를 잃었으므로 labels를 함께 팽창시켜 연결 ID 보존
        n_touch = 0
        for di in range(1, n_dot):
            dilated_region = cv2.dilate(
                (dot_labels == di).astype(np.uint8),
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (sz, sz)),
            )
            if np.any((dilated_region != 0) & touching):
                n_touch += 1
        if n_touch < 1:
            continue
        contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        hull_area = cv2.contourArea(cv2.convexHull(contour))
        required_solidity = (cfg.one_dot_corridor_solidity if n_touch == 1
                             else cfg.corridor_solidity)
        if hull_area <= 0 or cv2.contourArea(contour) / hull_area < required_solidity:
            continue
        ys, xs = np.nonzero(region)
        coords = np.column_stack([xs, ys]).astype(np.float64)
        origin = coords.mean(axis=0)
        eigvals, eigvecs = np.linalg.eigh(np.cov(coords.T))
        required_ratio = (cfg.one_dot_corridor_eigen_ratio if n_touch == 1
                          else cfg.corridor_min_eigen_ratio)
        if eigvals[0] <= 0 or eigvals[1] / eigvals[0] < required_ratio:
            continue
        axis = eigvecs[:, 1]
        perp = np.array([-axis[1], axis[0]])
        delta = coords - origin
        along = delta @ axis
        sideways = delta @ perp
        # 주 진행축 직교 절편의 평균을 이어 중심선 근사 (20px 간격)
        step_px = 20.0
        lo = float(along.min())
        bin_ids = np.floor((along - lo) / step_px).astype(np.int32)
        counts = np.bincount(bin_ids)
        sums = np.bincount(bin_ids, weights=sideways)
        good = counts > 10
        if good.sum() < 5:
            continue
        bins = np.flatnonzero(good)
        lateral = sums[bins] / counts[bins]
        if len(lateral) >= 5:
            # 좌우 경계의 작은 울퉁불퉁함으로 인한 흔들림 완화
            lateral = cv2.GaussianBlur(lateral.reshape(-1, 1).astype(np.float32), (1, 11), sigmaX=0).ravel()
        longitudinal = lo + (bins.astype(float) + 0.5) * step_px
        centers = origin + longitudinal[:, None] * axis + lateral[:, None] * perp
        centers = np.rint(centers).astype(np.int32)
        centerline = np.zeros_like(out)
        # centerline이 분리된 영역을 무리하게 잇지 않도록 인접 bin만 연결
        for i in range(len(centers) - 1):
            if bins[i + 1] - bins[i] > 1:
                continue
            p0, p1 = tuple(centers[i]), tuple(centers[i + 1])
            cv2.line(centerline, p0, p1, 1, 1)
        if not np.any(centerline):
            continue
        # 중앙 근사 선이 마스크 바깥으로 크게 벗어나면 이 대체는 포기
        coords_line = np.argwhere(centerline != 0)
        if np.mean(region[coords_line[:, 0], coords_line[:, 1]] != 0) < 0.97:
            continue
        # 경계 근방에서는 평균 중심선이 마스크 밖/점형 안으로 나갈 수 있다.
        # 실제 선형 영역에 속하는 픽셀만 남겨 접점·노드 클래스 불일치를 방지한다.
        centerline[region == 0] = 0
        # 해당 선형 연결 성분의 원래 medial-axis 가짜 가지를 제거하고 교체
        out[region != 0] = 0
        out[centerline != 0] = 1
        refined_components.add(li)
    return out.astype(bool), refined_components


def _find_endpoint_attachments(
    nodes: Nodes,
    edges: Edges,
    linear: np.ndarray,
    dot: np.ndarray,
    line_components: np.ndarray,
    cfg: SkeletonConfig,
) -> dict[int, int]:
    """접촉 영역마다 가장 가까운 선형 끝점을 dot connected component에 배정.

    skeleton 끝은 두꺼운 띠의 가운데서 끝나므로 표면 접점에서 조금 떨어질 수 있다.
    실제 접촉이 확인된 영역만 후보로 사용하고, 멀리 떨어진 마스크는 연결하지 않는다.
    """
    if not np.any(dot) or not nodes:
        return {}
    h, w = linear.shape
    _, dots = cv2.connectedComponents(dot, connectivity=8)
    n_dot = int(dots.max()) + 1
    line_radius = cv2.distanceTransform(linear, cv2.DIST_L2, 3)
    ends = {}
    for nid, pix in nodes.items():
        if _degree(edges, nid) == 1:
            y, x = pix[0]
            if 0 <= x < w and 0 <= y < h and line_components[y, x] > 0:
                ends[nid] = (x, y, int(line_components[y, x]))
    if not ends:
        return {}
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * max(0, cfg.dot_attach_px) + 1,) * 2,
    )
    chosen: dict[int, tuple[int, float]] = {}
    for di in range(1, n_dot):
        region = (dots == di).astype(np.uint8)
        expanded = cv2.dilate(region, kernel) if cfg.dot_attach_px > 0 else region
        contact = cv2.bitwise_and(expanded, linear)
        if not np.any(contact):
            continue
        num, contact_ids = cv2.connectedComponents(contact, connectivity=8)
        for ci in range(1, num):
            ys, xs = np.nonzero(contact_ids == ci)
            if not len(xs):
                continue
            touched = line_components[ys, xs]
            best = None
            for nid, (x, y, component_id) in ends.items():
                valid = touched == component_id
                if not np.any(valid):
                    continue
                d2 = (xs[valid] - x) ** 2 + (ys[valid] - y) ** 2
                dist = float(np.sqrt(d2.min()))
                allowed = max(24.0, 1.7 * float(line_radius[y, x]) + cfg.dot_attach_px + 12)
                if dist <= allowed and (best is None or dist < best[0]):
                    best = (dist, nid)
            if best is not None:
                dist, nid = best
                if nid not in chosen or dist < chosen[nid][1]:
                    chosen[nid] = (di, dist)
    return {nid: di for nid, (di, _) in chosen.items()}


def _attach_dot_regions(
    path: PathStructure,
    linear: np.ndarray,
    dot: np.ndarray,
    line_components: np.ndarray,
    attached_ends: dict[int, int],
    cfg: SkeletonConfig,
) -> PathStructure:
    n_dot, dots, stats, centroids = cv2.connectedComponentsWithStats(dot, connectivity=8)
    if n_dot <= 1:
        return path
    h, w = linear.shape
    endpoint_by_id = {n.id: n for n in path.nodes if n.type == NodeType.END and n.block == MaskClass.LINEAR}
    # dot 노드의 기준점은 분할 영역 내부에 위치하도록 (도형 중심이 바깥에 있을 수 있음)
    dot_nodes: dict[int, PathNode] = {}
    next_id = max((n.id for n in path.nodes), default=-1) + 1
    for di in range(1, n_dot):
        cx, cy = centroids[di]
        iy, ix = int(round(cy)), int(round(cx))
        if dots[min(max(iy, 0), h - 1), min(max(ix, 0), w - 1)] != di:
            ys, xs = np.nonzero(dots == di)
            j = int(np.argmin((xs - cx) ** 2 + (ys - cy) ** 2))
            cx, cy = float(xs[j]), float(ys[j])
        dot_nodes[di] = PathNode(
            id=next_id, type=NodeType.ISOLATED,
            pt=Point(x=float(cx), y=float(cy)),
            block=MaskClass.DOT, component=0,
        )
        next_id += 1

    # 점형 영역으로 들어오는 실제 선형의 끝점을 점형 노드로 합친다.
    # 경로 좌표는 점형 노드 중심까지 연장하되 독립적인 가짜 분기는 만들지 않는다.
    removed: set[int] = set()
    attachment_pairs: list[tuple[int, int]] = []  # (linear cc, dot cc)
    for eid, di in attached_ends.items():
        if eid not in endpoint_by_id:
            continue
        endpoint = endpoint_by_id[eid]
        incident = [e for e in path.edges if eid in (e.u, e.v)]
        if len(incident) != 1:
            continue
        edge = incident[0]
        target = dot_nodes[di]
        if edge.u == eid:
            edge.u = target.id
            edge.polyline.insert(0, target.pt)
        else:
            edge.v = target.id
            edge.polyline.append(target.pt)
        edge.length_px += math.dist(
            (endpoint.pt.x, endpoint.pt.y), (target.pt.x, target.pt.y)
        )
        removed.add(eid)
        attachment_pairs.append((endpoint.component, di))

    path.nodes = [n for n in path.nodes if n.id not in removed]
    path.nodes.extend(dot_nodes.values())
    for node in dot_nodes.values():
        degree = sum(int(e.u == node.id) + int(e.v == node.id) for e in path.edges)
        node.type = (NodeType.ISOLATED if degree == 0 else
                     NodeType.END if degree == 1 else
                     NodeType.BRANCH if degree >= 3 else NodeType.PASS)

    # 점형과 선형이 연결되면 같은 component 번호가 되도록 재번호 매김
    n_line = int(line_components.max())
    parent = {('L', i): ('L', i) for i in range(1, n_line + 1)}
    parent.update({('D', i): ('D', i) for i in range(1, n_dot)})

    def root(key):
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    for li, di in attachment_pairs:
        if li > 0:
            parent[root(('D', di))] = root(('L', li))

    root_to_id: dict[tuple[str, int], int] = {}
    area_by_id: dict[int, int] = {}
    area_line = np.bincount(line_components.ravel(), minlength=n_line + 1)
    for (kind, idx) in parent:
        group = root((kind, idx))
        cid = root_to_id.setdefault(group, len(root_to_id) + 1)
        amount = int(area_line[idx] if kind == 'L' else stats[idx, cv2.CC_STAT_AREA])
        area_by_id[cid] = area_by_id.get(cid, 0) + amount
    for n in path.nodes:
        old = ('D', next(di for di, dn in dot_nodes.items() if dn.id == n.id)) if n.block == MaskClass.DOT else ('L', n.component)
        if old in parent:
            n.component = root_to_id[root(old)]
    path.component_area_px = area_by_id
    return path


def end_direction(path: PathStructure, node_id: int, length_px: float = 80.0) -> Optional[tuple[float, float]]:
    """끝점에서 바깥쪽을 향하는 단위 벡터 (dx, dy). 중심선 마지막 구간 기준.

    방향 판정(C2·C3)에서 선형 줄의 연장선을 그을 때 쓴다.
    """
    edges = path.edges_of(node_id)
    if len(edges) != 1:
        return None
    e = edges[0]
    pts = e.polyline if e.u == node_id else list(reversed(e.polyline))
    # 점형 노드는 연결 영역의 중심에 있으므로, 방향은 점형 중심→선형 끝점
    # 가상 연결선이 아니라 실제 선형 중심선의 접선으로 추정한다.
    if path.node(node_id).block == MaskClass.DOT:
        if len(pts) < 3:
            return None
        pts = pts[1:]
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


def _prune_spurs(nodes: Nodes, edges: Edges, spur_px: float, protected_ends: Optional[set[int]] = None) -> tuple[Nodes, Edges]:
    protected_ends = protected_ends or set()
    changed = True
    while changed:
        changed = False
        for eid, (u, v, path) in list(edges.items()):
            du, dv = _degree(edges, u), _degree(edges, v)
            if u == v or _length(path) >= spur_px:
                continue
            if (du == 1 and dv >= 3) or (dv == 1 and du >= 3):
                leaf = u if du == 1 else v
                if leaf in protected_ends:
                    continue
                del edges[eid]
                nodes.pop(leaf)
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


def _split_turns(
    nodes: Nodes, edges: Edges, cfg: SkeletonConfig,
    line_components: Optional[np.ndarray] = None,
    refined_components: Optional[set[int]] = None,
) -> tuple[Nodes, Edges, set[int]]:
    turn_ids: set[int] = set()
    out: Edges = {}
    for u, v, path in edges.values():
        pts = _simplify(path, cfg.dp_epsilon_px)
        cur_start, seg = u, [pts[0]]
        threshold = cfg.turn_deg
        if line_components is not None and refined_components:
            ry, rx = pts[len(pts) // 2]
            if int(line_components[ry, rx]) in refined_components:
                threshold = max(cfg.turn_deg, cfg.corridor_turn_deg)
        for i in range(1, len(pts) - 1):
            seg.append(pts[i])
            if (_turn_angle(pts[i - 1], pts[i], pts[i + 1]) > threshold
                and _length([pts[i - 1], pts[i]]) >= cfg.turn_min_arm_px
                and _length([pts[i], pts[i + 1]]) >= cfg.turn_min_arm_px):
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
