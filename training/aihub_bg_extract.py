"""AI Hub '인도보행 영상' 서피스마스킹에서 블록 모델(v4)용 배경 사진(빈 라벨)을 뽑는다. (담당: A)

점자블럭(braille_guide_blocks) 폴리곤이 없는 사진만 고른다.
v4가 노란 물체(은행나무, 블라인드, 노란 칠)와 횡단보도 줄무늬를 블록으로 오인식하므로
노란 픽셀이 많은 사진과 횡단보도 사진을 우선 넣는다.

구성 (기본 200장)
    yellow     노란 픽셀 비율이 높은 사진 (--n-yellow)
    crosswalk  횡단보도가 있는 사진 (--n-crosswalk)
    random     나머지 일반 인도 사진

출력
    datasets/aihub_bg_v1/
    ├── images/        긴 변 1280px
    ├── labels/        빈 .txt (YOLO 배경 이미지)
    ├── manifest.csv
    └── gallery.html

사용 예
    python training/aihub_bg_extract.py --n 200
"""

from __future__ import annotations

import argparse
import csv
import html
import random
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ZIP_DIR = ROOT / "_data" / "인도보행 영상" / "서피스마스킹"


def yellow_ratio(img: np.ndarray) -> float:
    hsv = cv2.cvtColor(cv2.resize(img, (320, 180)), cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, (15, 90, 90), (35, 255, 255))
    return float(m.mean() / 255)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--n-yellow", type=int, default=100)
    ap.add_argument("--n-crosswalk", type=int, default=50)
    ap.add_argument("--pool", type=int, default=1500, help="노란 비율을 계산할 후보 수")
    ap.add_argument("--long-side", type=int, default=1280)
    ap.add_argument("--seed", type=int, default=47)
    ap.add_argument("--out", type=Path, default=ROOT / "datasets" / "aihub_bg_v1")
    args = ap.parse_args()
    rng = random.Random(args.seed)

    # 1. 점자블럭 없는 사진 목록 (zip, 경로, 횡단보도 여부)
    items = []
    zips = {}
    for k in range(1, 6):
        z = zipfile.ZipFile(ZIP_DIR / f"Surface_{k}.zip", metadata_encoding="cp949")
        zips[k] = z
        for xml_name in sorted(n for n in z.namelist() if n.endswith(".xml")):
            folder = xml_name.split("/")[0]
            for im in ET.fromstring(z.read(xml_name)).iter("image"):
                labels = {(p.get("label"), (p.findtext("attribute") or "").strip()) for p in im}
                if any(lb == "braille_guide_blocks" for lb, _ in labels):
                    continue
                items.append({"zip": k, "folder": folder, "src": f"{folder}/{im.get('name')}",
                              "crosswalk": any(a == "crosswalk" for _, a in labels)})
    print(f"점자블럭 없는 사진 {len(items)}장")

    # 2. 영상 폴더마다 한 장씩 돌아가며 후보 pool 구성 (비슷한 연속 프레임 방지)
    def spread(cands, n):
        by = {}
        for it in cands:
            by.setdefault(it["folder"], []).append(it)
        for v in by.values():
            rng.shuffle(v)
        keys = list(by)
        rng.shuffle(keys)
        out = []
        while len(out) < n and any(by[k] for k in keys):
            for k in keys:
                if by[k] and len(out) < n:
                    out.append(by[k].pop())
        return out

    def load(it):
        return cv2.imdecode(np.frombuffer(zips[it["zip"]].read(it["src"]), np.uint8), cv2.IMREAD_COLOR)

    chosen, used = [], set()
    pool = spread([it for it in items if not it["crosswalk"]], args.pool)
    for it in pool:
        it["yellow"] = yellow_ratio(load(it))
    pool.sort(key=lambda it: -it["yellow"])
    for it in pool[: args.n_yellow]:
        chosen.append({**it, "kind": "yellow"})
        used.add(it["src"])
    for it in spread([it for it in items if it["crosswalk"]], args.n_crosswalk):
        chosen.append({**it, "kind": "crosswalk"})
        used.add(it["src"])
    rest = spread([it for it in items if it["src"] not in used and not it["crosswalk"]], args.n - len(chosen))
    chosen += [{**it, "kind": "random"} for it in rest]

    # 3. 저장
    (args.out / "images").mkdir(parents=True, exist_ok=True)
    (args.out / "labels").mkdir(parents=True, exist_ok=True)
    rows, cards = [], []
    for it in chosen:
        img = load(it)
        h, w = img.shape[:2]
        s = min(1.0, args.long_side / max(h, w))
        img = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
        stem = f"bg_{it['folder']}_{Path(it['src']).stem}"
        cv2.imwrite(str(args.out / "images" / f"{stem}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        (args.out / "labels" / f"{stem}.txt").write_text("")
        y = it.get("yellow")
        rows.append({"file": f"{stem}.jpg", "kind": it["kind"], "folder": it["folder"],
                     "yellow_ratio": "" if y is None else f"{y:.3f}"})
        cards.append(f'<figure><img src="images/{html.escape(stem)}.jpg" loading="lazy">'
                     f'<figcaption>{it["kind"]} · {html.escape(stem)}</figcaption></figure>')

    with open(args.out / "manifest.csv", "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    (args.out / "gallery.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>배경 사진</title><style>'
        "body{font:13px system-ui;margin:16px;background:#111;color:#ddd}"
        "main{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:10px}"
        "figure{margin:0}img{width:100%}figcaption{word-break:break-all}</style><main>"
        + "\n".join(cards) + "</main>", encoding="utf-8")
    kinds = {k: sum(r["kind"] == k for r in rows) for k in ("yellow", "crosswalk", "random")}
    print(f"완료: {len(rows)}장 {kinds}, 영상 폴더 {len({r['folder'] for r in rows})}개 → {args.out}")


if __name__ == "__main__":
    main()
