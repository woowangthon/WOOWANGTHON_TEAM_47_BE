"""AI Hub '보행 안전을 위한 도로 시설물' zip에서 점자블럭 후보만 골라 꺼낸다. (담당: A)

압축을 풀지 않고 zip 안의 파일을 하나씩 읽는다 (TS9.zip은 무압축이라 빠르다).

하는 일
1. TL9.zip에서 '09. 점자블럭' JSON만 읽는다 (도로안내표지 지주는 건너뜀)
2. 지역(city_id)·날짜 묶음이 고르게 섞이게 후보 N장을 고른다
3. TS9.zip에서 그 사진만 꺼내 긴 변을 줄여 저장한다
4. 줄인 좌표로 CVAT용 COCO(클래스 `tactile` 하나)를 만든다
5. 후보를 훑어볼 gallery.html, 메타데이터 candidates.csv를 만든다

출력 (datasets/는 .gitignore에 등록됨)
    datasets/aihub_candidates/
    ├── images/                       줄인 사진
    ├── json/                         원본 JSON (수정하지 않음)
    ├── annotations/instances_default.json
    ├── cvat_coco.zip                 CVAT "Upload annotations → COCO 1.0" 용
    ├── candidates.csv                파일명, group(city_date), 폴리곤 수 등
    └── gallery.html                  폴리곤을 겹쳐 그린 미리보기

사용 예
    python training/aihub_extract.py --n 300
    python training/aihub_extract.py --n 5 --out datasets/aihub_test   # 시험
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import random
import zipfile
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "_data" / "119.보행 안전을 위한 도로 시설물 데이터" / "01.데이터" / "1.Training"
LABEL_ZIP = DATA / "라벨링데이터" / "TL9.zip"
IMAGE_ZIP = DATA / "원천데이터" / "TS9.zip"
TACTILE_DIR = "09."  # '09. 점자블럭/...'
WHOLE_ID = "35"  # 점자블럭 불량전체 → 블록 영역
DAMAGE_ID = "36"  # 점자블럭 불량부분 → 파손 (확장 계획 X1)


def open_zip(p: Path) -> zipfile.ZipFile:
    # 파일명이 CP949로 저장되어 있다
    return zipfile.ZipFile(p, metadata_encoding="cp949")


def load_labels(zl: zipfile.ZipFile) -> list[dict]:
    out = []
    for name in zl.namelist():
        if not (name.startswith(TACTILE_DIR) and name.endswith(".json")):
            continue
        j = json.loads(zl.read(name).decode("utf-8-sig"))
        info = j["info"]
        whole = [a for a in j["annotations"] if a["label_id"] == WHOLE_ID]
        out.append(
            {
                "zip_name": name,
                "json": j,
                "filename": info["filename"],
                "group": f"{info['city_id']}_{info['date']}",
                "n_whole": len(whole),
                "max_points": max((len(a["annotation_info"]) for a in whole), default=0),
            }
        )
    return out


def pick(items: list[dict], n: int, min_points: int, seed: int) -> list[dict]:
    """city_date 묶음을 돌아가며 하나씩 뽑는다. 꼭짓점이 많은(정교한) 폴리곤 우선."""
    rng = random.Random(seed)
    groups: dict[str, list[dict]] = defaultdict(list)
    for it in items:
        if it["n_whole"] and it["max_points"] >= min_points:
            groups[it["group"]].append(it)
    for g in groups.values():
        rng.shuffle(g)
        g.sort(key=lambda it: -it["max_points"])
    keys = list(groups)
    rng.shuffle(keys)
    chosen: list[dict] = []
    while len(chosen) < n and any(groups[k] for k in keys):
        for k in keys:
            if groups[k] and len(chosen) < n:
                chosen.append(groups[k].pop(0))
    return chosen


def image_index(zi: zipfile.ZipFile) -> dict[str, str]:
    return {n.rsplit("/", 1)[-1]: n for n in zi.namelist() if n.startswith(TACTILE_DIR)}


def polygon(points, scale: float, w: int, h: int) -> list[float]:
    flat = []
    for x, y in points:
        flat += [min(max(x * scale, 0), w - 1), min(max(y * scale, 0), h - 1)]
    return [round(v, 2) for v in flat]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300, help="후보 장수")
    ap.add_argument("--long-side", type=int, default=1280, help="저장할 사진의 긴 변 (px)")
    ap.add_argument("--min-points", type=int, default=5, help="블록 폴리곤 최소 꼭짓점 수 (4점 사각형 제외)")
    ap.add_argument("--with-damage", action="store_true", help="불량부분 폴리곤도 `damage` 클래스로 포함")
    ap.add_argument("--seed", type=int, default=47)
    ap.add_argument("--out", type=Path, default=ROOT / "datasets" / "aihub_candidates")
    args = ap.parse_args()

    for d in ("images", "json", "annotations"):
        (args.out / d).mkdir(parents=True, exist_ok=True)

    zl, zi = open_zip(LABEL_ZIP), open_zip(IMAGE_ZIP)
    items = load_labels(zl)
    chosen = pick(items, args.n, args.min_points, args.seed)
    idx = image_index(zi)
    print(f"점자블럭 JSON {len(items)}개 중 {len(chosen)}장 선택 "
          f"(묶음 {len({c['group'] for c in chosen})}개)")

    cats = [{"id": 1, "name": "tactile", "supercategory": ""}]
    if args.with_damage:
        cats.append({"id": 2, "name": "damage", "supercategory": ""})
    coco = {"images": [], "annotations": [], "categories": cats}
    rows, cards = [], []

    for i, it in enumerate(chosen, 1):
        j, fn = it["json"], it["filename"]
        if fn not in idx:
            print(f"  [건너뜀] 원천 zip에 없음: {fn}")
            continue
        img = cv2.imdecode(np.frombuffer(zi.read(idx[fn]), np.uint8), cv2.IMREAD_COLOR)
        W, H = int(j["info"]["width"]), int(j["info"]["height"])
        if img is None or img.shape[:2] != (H, W):
            got = None if img is None else img.shape[:2][::-1]
            print(f"  [건너뜀] 크기 불일치 {fn}: JSON {W}x{H}, 사진 {got}")
            continue
        scale = min(1.0, args.long_side / max(W, H))
        w, h = round(W * scale), round(H * scale)
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(args.out / "images" / fn), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        (args.out / "json" / Path(fn).with_suffix(".json").name).write_bytes(zl.read(it["zip_name"]))

        image_id = len(coco["images"]) + 1
        coco["images"].append({"id": image_id, "file_name": fn, "width": w, "height": h})
        polys = []
        for a in j["annotations"]:
            cat = 1 if a["label_id"] == WHOLE_ID else 2 if (args.with_damage and a["label_id"] == DAMAGE_ID) else None
            if cat is None or a["annotation_type"] != "polygon" or len(a["annotation_info"]) < 3:
                continue
            seg = polygon(a["annotation_info"], scale, w, h)
            pts = np.array(seg, np.float32).reshape(-1, 2)
            x0, y0 = pts.min(0)
            x1, y1 = pts.max(0)
            coco["annotations"].append(
                {
                    "id": len(coco["annotations"]) + 1,
                    "image_id": image_id,
                    "category_id": cat,
                    "segmentation": [seg],
                    "area": float(abs(cv2.contourArea(pts))),
                    "bbox": [float(x0), float(y0), float(x1 - x0), float(y1 - y0)],
                    "iscrowd": 0,
                }
            )
            polys.append((cat, seg))

        rows.append({"file_name": fn, "group": it["group"], "width": w, "height": h,
                     "n_block_polygons": it["n_whole"], "max_points": it["max_points"],
                     "type": "", "keep": ""})
        cards.append(_card(fn, w, h, it["group"], polys))
        if i % 25 == 0:
            print(f"  {i}/{len(chosen)}")

    (args.out / "annotations" / "instances_default.json").write_text(json.dumps(coco, ensure_ascii=False))
    with zipfile.ZipFile(args.out / "cvat_coco.zip", "w") as zf:
        zf.write(args.out / "annotations" / "instances_default.json", "annotations/instances_default.json")
    with open(args.out / "candidates.csv", "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["file_name"])
        wr.writeheader()
        wr.writerows(rows)
    (args.out / "gallery.html").write_text(_GALLERY.replace("{{CARDS}}", "\n".join(cards)), encoding="utf-8")

    size_mb = sum(p.stat().st_size for p in (args.out / "images").iterdir()) / 1e6
    print(f"완료: 사진 {len(rows)}장 ({size_mb:.0f}MB), 폴리곤 {len(coco['annotations'])}개 → {args.out}")


def _card(fn: str, w: int, h: int, group: str, polys) -> str:
    color = {1: "#00e0ff", 2: "#ff3b6b"}
    shapes = "".join(
        f'<polygon points="{" ".join(f"{seg[k]},{seg[k + 1]}" for k in range(0, len(seg), 2))}" '
        f'fill="{color[c]}33" stroke="{color[c]}" stroke-width="{max(w, h) / 300:.1f}"/>'
        for c, seg in polys
    )
    name = html.escape(fn)
    return (
        f'<figure><svg viewBox="0 0 {w} {h}"><image href="images/{name}" width="{w}" height="{h}"/>'
        f"{shapes}</svg><figcaption>{name}<br><span>{html.escape(group)}</span></figcaption></figure>"
    )


_GALLERY = """<!doctype html><meta charset="utf-8"><title>AI Hub 점자블럭 후보</title>
<style>
body{font:13px system-ui;margin:16px;background:#111;color:#ddd}
main{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:12px}
figure{margin:0}svg{width:100%;height:auto;display:block;background:#222}
figcaption{word-break:break-all;padding:4px 0}span{color:#888}
</style>
<p>하늘색: 점자블럭(불량전체) 폴리곤, 빨강: 불량부분(--with-damage). 고른 사진은 candidates.csv의 type(mixed/linear/dot/background)·keep 열에 표시</p>
<main>
{{CARDS}}
</main>"""


if __name__ == "__main__":
    main()
