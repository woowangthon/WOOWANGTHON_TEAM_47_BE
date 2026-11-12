"""사진 + AI Hub 원본 JSON이 나란히 있는 폴더마다 CVAT용 COCO zip을 만든다.

사진이 줄어 있으면(aihub_extract.py 결과) 실제 사진 크기에 맞게 좌표를 바꾼다.
클래스는 `tactile` 하나 (점자블럭 정상 34, 불량전체 35). 불량부분(36)은 --with-damage 일 때만 `damage`.

출력: <out>/<폴더이름>.zip  (안에 annotations/instances_default.json)
CVAT: 폴더 사진으로 Task 생성 → Upload annotations → COCO 1.0 → 이 zip

사용 예
    python training/aihub_folder_coco.py datasets/aihub_candidates/images \
        "datasets/aihub_candidates/점형 및 혼합" "datasets/aihub_candidates/파손 및 육안확인어려움"
"""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import cv2
import numpy as np

BLOCK_IDS = {"34", "35"}  # 점자블럭 정상, 불량전체
DAMAGE_ID = "36"


def build(folder: Path, with_damage: bool) -> tuple[dict, list[str]]:
    cats = [{"id": 1, "name": "tactile", "supercategory": ""}]
    if with_damage:
        cats.append({"id": 2, "name": "damage", "supercategory": ""})
    coco = {"images": [], "annotations": [], "categories": cats}
    problems: list[str] = []

    for jp in sorted(folder.glob("*.json")):
        j = json.loads(jp.read_text(encoding="utf-8-sig"))
        fn = j["info"]["filename"]
        img_path = folder / fn
        if not img_path.exists():
            problems.append(f"사진 없음: {fn}")
            continue
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)  # EXIF 회전 반영 (CVAT 표시와 같음)
        if img is None:
            problems.append(f"사진을 읽을 수 없음: {fn}")
            continue
        h, w = img.shape[:2]
        W, H = int(j["info"]["width"]), int(j["info"]["height"])
        sx, sy = w / W, h / H
        if abs(sx - sy) > 0.01:
            problems.append(f"비율 불일치 (회전?): {fn} JSON {W}x{H}, 사진 {w}x{h}")
            continue

        image_id = len(coco["images"]) + 1
        coco["images"].append({"id": image_id, "file_name": fn, "width": w, "height": h})
        for a in j["annotations"]:
            lid = a["label_id"]
            cat = 1 if lid in BLOCK_IDS else 2 if (with_damage and lid == DAMAGE_ID) else None
            if cat is None or a["annotation_type"] != "polygon" or len(a["annotation_info"]) < 3:
                continue
            pts = np.array(a["annotation_info"], np.float64) * [sx, sy]
            pts[:, 0] = pts[:, 0].clip(0, w - 1)
            pts[:, 1] = pts[:, 1].clip(0, h - 1)
            x0, y0 = pts.min(0)
            x1, y1 = pts.max(0)
            coco["annotations"].append(
                {
                    "id": len(coco["annotations"]) + 1,
                    "image_id": image_id,
                    "category_id": cat,
                    "segmentation": [[round(float(v), 2) for v in pts.ravel()]],
                    "area": float(abs(cv2.contourArea(pts.astype(np.float32)))),
                    "bbox": [float(x0), float(y0), float(x1 - x0), float(y1 - y0)],
                    "iscrowd": 0,
                }
            )
    return coco, problems


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("folders", type=Path, nargs="+")
    ap.add_argument("--out", type=Path, default=None, help="기본: 첫 폴더의 상위 폴더/cvat")
    ap.add_argument("--with-damage", action="store_true")
    args = ap.parse_args()
    out = args.out or args.folders[0].parent / "cvat"
    out.mkdir(parents=True, exist_ok=True)

    for folder in args.folders:
        coco, problems = build(folder, args.with_damage)
        z = out / f"{folder.name}.zip"
        with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("annotations/instances_default.json", json.dumps(coco, ensure_ascii=False))
        print(f"{folder.name}: 사진 {len(coco['images'])}장, 폴리곤 {len(coco['annotations'])}개 → {z}")
        for p in problems:
            print(f"  [건너뜀] {p}")


if __name__ == "__main__":
    main()
