"""v4_best.pt로 한 장의 사진 마스크를 저장하는 실험용 스크립트.

프로젝트 루트에서 실행:
    python run_mask_extract.py --image "tests/images/sample.jpg" \
        --weights "weights/v4_best.pt" --photo close
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from pipeline.schema import MaskClass, PhotoKind
from pipeline.segment import Segmenter


CLASS_COLORS = {
    MaskClass.LINEAR: (255, 0, 0),       # 파랑 (BGR)
    MaskClass.DOT: (0, 0, 255),          # 빨강
    MaskClass.CROSSWALK: (0, 255, 255),  # 노랑
    MaskClass.ROAD: (0, 200, 0),        # 초록
}


def read_image(path: Path) -> np.ndarray:
    # 윈도우 한글 경로 처리
    raw = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"사진을 읽을 수 없습니다: {path}")
    return image


def save_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    success, data = cv2.imencode(".png", image)
    if not success:
        raise IOError(f"PNG 생성 실패: {path}")
    data.tofile(str(path))


def export_masks(image: np.ndarray, label: np.ndarray, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    # 실제 후속 skeleton.extract_path()에 넣을 라벨 마스크: 0/1/2/3/4 값 유지
    save_png(out_dir / "label_mask.png", label)
    np.save(out_dir / "label_mask.npy", label)

    overlay = image.copy()
    color_layer = image.copy()
    for cls in (MaskClass.LINEAR, MaskClass.DOT, MaskClass.CROSSWALK, MaskClass.ROAD):
        mask = label == int(cls)
        save_png(out_dir / f"{cls.name.lower()}_mask.png", mask.astype(np.uint8) * 255)
        color_layer[mask] = CLASS_COLORS[cls]
    mixed = cv2.addWeighted(image, 0.55, color_layer, 0.45, 0.0)
    overlay[label > 0] = mixed[label > 0]
    save_png(out_dir / "overlay.png", overlay)


def main() -> None:
    parser = argparse.ArgumentParser(description="TactileCheck YOLO v4 마스크 추출")
    parser.add_argument("--image", required=True, help="입력 사진 경로")
    parser.add_argument("--weights", default="weights/v4_best.pt", help="모델 경로")
    parser.add_argument("--photo", choices=["close", "wide"], default="close")
    parser.add_argument("--out", default="outputs/masks", help="출력 폴더")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.is_file():
        raise FileNotFoundError(f"입력 사진이 없습니다: {image_path}")
    image = read_image(image_path)

    segmenter = Segmenter(args.weights, conf=args.conf, imgsz=args.imgsz)
    label, info = segmenter.predict(image, PhotoKind(args.photo))

    out_dir = Path(args.out)
    export_masks(image, label, out_dir)
    info.mask_path = str(out_dir / "label_mask.png")
    metadata = {
        "segmentation": info.model_dump(mode="json"),
        "available_classes": segmenter.available_classes,
        "weights": args.weights,
        "input_image": str(image_path),
        "note": "미지원 클래스는 면적이 0이어도 미검출과 구별해야 합니다.",
    }
    (out_dir / "segmentation.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"사진 크기: {image.shape[1]}×{image.shape[0]}")
    print("학습된 클래스:", ", ".join(segmenter.available_classes))
    print("클래스별 픽셀 수:", info.class_area_px)
    print("점자블록 평균 confidence:", info.mean_confidence)
    print("결과 폴더:", out_dir.resolve())


if __name__ == "__main__":
    main()
