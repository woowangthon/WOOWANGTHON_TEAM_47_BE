"""TactileCheck: YOLO label_mask -> skeleton/path graph visualization.

Run from the repository root after run_mask_extract.py:
    python run_path_debug.py --image "test.jpg"

Requires existing pipeline/schema.py and pipeline/skeleton.py.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from skimage.morphology import skeletonize

from pipeline.schema import MaskClass, NodeType, PhotoKind
from pipeline.skeleton import SkeletonConfig, extract_path, prepare_linear_mask


NODE_COLORS = {
    NodeType.END: (30, 220, 30),        # green, BGR
    NodeType.BRANCH: (40, 40, 255),      # red
    NodeType.TURN: (0, 180, 255),        # orange
    NodeType.PASS: (255, 255, 0),        # cyan
    NodeType.ISOLATED: (255, 0, 255),    # magenta
}


def read_image(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"원본 사진이 없습니다: {path}")
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"사진을 읽을 수 없습니다: {path}")
    return image


def save_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError(f"이미지 인코딩 실패: {path}")
    encoded.tofile(str(path))  # Windows Korean pathname safe


def read_label(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"라벨 마스크가 없습니다: {path}")
    if path.suffix.lower() == ".npy":
        label = np.load(path, allow_pickle=False)
    elif path.suffix.lower() == ".png":
        label = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    else:
        raise ValueError("마스크 파일은 .npy 또는 .png만 지원합니다")
    if label is None or label.ndim != 2 or label.dtype != np.uint8:
        raise ValueError("마스크는 H×W, uint8 단일 채널이어야 합니다")
    valid = np.array([int(c) for c in MaskClass], dtype=np.uint8)
    if not np.all(np.isin(label, valid)):
        raise ValueError("MaskClass에 정의되지 않은 라벨 픽셀이 있습니다")
    return label


def draw_path(image: np.ndarray, label: np.ndarray, path) -> tuple[np.ndarray, np.ndarray]:
    mask_overlay = image.copy()
    mask_overlay[label == MaskClass.LINEAR] = (255, 0, 0)
    mask_overlay[label == MaskClass.DOT] = (0, 0, 255)
    overlay = cv2.addWeighted(image, 0.78, mask_overlay, 0.22, 0)
    graph_only = np.zeros_like(image)

    for edge in path.edges:
        pts = np.array(
            [[round(p.x), round(p.y)] for p in edge.polyline], dtype=np.int32
        ).reshape(-1, 1, 2)
        if len(pts) > 1:
            cv2.polylines(overlay, [pts], isClosed=False, color=(255, 255, 255), thickness=3)
            cv2.polylines(graph_only, [pts], isClosed=False, color=(255, 255, 255), thickness=2)
        elif len(pts) == 1:
            xy = tuple(pts[0, 0])
            cv2.circle(overlay, xy, 2, (255, 255, 255), -1)
            cv2.circle(graph_only, xy, 2, (255, 255, 255), -1)

    for node in path.nodes:
        xy = (round(node.pt.x), round(node.pt.y))
        if node.block == MaskClass.DOT:
            # 점형 영역은 연결/전환 노드로 표현; 내부 skeleton 가짜 분기 표시 안 함
            color = (0, 255, 255)  # 노랑 BGR
            cv2.circle(overlay, xy, 11, color, 3)
            cv2.circle(graph_only, xy, 9, color, 2)
            node_label = f"D{node.id}"
        else:
            color = NODE_COLORS[node.type]
            cv2.circle(overlay, xy, 7, color, -1)
            cv2.circle(overlay, xy, 7, (0, 0, 0), 1)
            cv2.circle(graph_only, xy, 6, color, -1)
            node_label = str(node.id)
        cv2.putText(
            overlay, node_label, (xy[0] + 8, xy[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA
        )

    return overlay, graph_only


def main() -> None:
    parser = argparse.ArgumentParser(description="TactileCheck 중심선·끝점·분기점 시각화")
    parser.add_argument("--image", required=True, help="마스크 생성에 사용한 원본 사진")
    parser.add_argument("--mask", default="outputs/masks/label_mask.npy", help="0/1/2/3/4 라벨 마스크")
    parser.add_argument("--out", default="outputs/path", help="시각화 출력 폴더")
    parser.add_argument("--photo", choices=["close", "wide"], default="close")
    parser.add_argument("--close-px", type=int, default=0, help="작은 틈 메우기 커널; 0=비활성(M4 단절 보존)")
    parser.add_argument("--min-area-px", type=int, default=400)
    parser.add_argument("--spur-px", type=float, default=0.0, help="0=선형 폭 기반 자동 제거, 양수=고정 픽셀")
    parser.add_argument("--dot-attach-px", type=int, default=8, help="점형-선형 작은 분할 간격 허용치")
    parser.add_argument("--min-dot-area-px", type=int, default=400, help="점형 노이즈 최소 면적")
    parser.add_argument("--turn-deg", type=float, default=45.0, help="선형 꺾임 판정 각도")
    parser.add_argument("--turn-min-arm-px", type=float, default=35.0, help="꺾임 앞뒤 최소 팔 길이")
    args = parser.parse_args()

    image = read_image(Path(args.image))
    label = read_label(Path(args.mask))
    if label.shape != image.shape[:2]:
        raise ValueError(
            f"원본/마스크 크기가 다릅니다: image={image.shape[:2]}, mask={label.shape}"
        )

    cfg = SkeletonConfig(
        close_px=args.close_px,
        min_area_px=args.min_area_px,
        spur_px=args.spur_px,
        dot_attach_px=args.dot_attach_px,
        min_dot_area_px=args.min_dot_area_px,
        turn_deg=args.turn_deg,
        turn_min_arm_px=args.turn_min_arm_px,
    )
    path = extract_path(label, PhotoKind(args.photo), cfg)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    raw_skeleton = skeletonize(prepare_linear_mask(label, cfg).astype(bool)).astype(np.uint8) * 255
    # 기존 파일명 유지 (이제 선형 마스크만 skeletonize함)
    save_png(out / "raw_skeleton.png", raw_skeleton)
    save_png(out / "linear_skeleton.png", raw_skeleton)
    dots = np.zeros((label.shape[0], label.shape[1], 3), dtype=np.uint8)
    dots[label == MaskClass.DOT] = (0, 0, 180)
    for node in path.nodes:
        if node.block == MaskClass.DOT:
            cv2.circle(dots, (round(node.pt.x), round(node.pt.y)), 11, (0, 255, 255), 2)
    save_png(out / "dot_regions.png", dots)
    overlay, graph = draw_path(image, label, path)
    save_png(out / "path_overlay.png", overlay)
    save_png(out / "path_graph.png", graph)
    (out / "path_structure.json").write_text(
        json.dumps(path.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    counts = Counter(node.type.value for node in path.nodes)
    print("마스크 크기:", label.shape)
    print("검출된 선형 픽셀:", int(np.count_nonzero(label == MaskClass.LINEAR)))
    print("검출된 점형 픽셀:", int(np.count_nonzero(label == MaskClass.DOT)))
    linear_nodes = [n for n in path.nodes if n.block == MaskClass.LINEAR]
    dot_nodes = [n for n in path.nodes if n.block == MaskClass.DOT]
    line_counts = Counter(n.type.value for n in linear_nodes)
    print("선형 경로 끝점:", line_counts.get("end", 0),
          "분기점:", line_counts.get("branch", 0),
          "꺾임점:", line_counts.get("turn", 0))
    print("점형 연결/전환 영역 노드:", len(dot_nodes))
    print("(점형 내부의 가지들은 구조 노드로 세지 않습니다.)")
    print("경로 엣지:", len(path.edges))
    print("시각화 저장 위치:", out.resolve())
    print("※ 꺾임·분기 결과는 후보 검출이며 설치 오류 판정이 아닙니다.")


if __name__ == "__main__":
    main()
