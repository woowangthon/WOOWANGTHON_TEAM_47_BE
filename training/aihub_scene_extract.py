"""AI Hub '인도보행 영상' 서피스마스킹 zip에서 장면 모델(횡단보도·차도) 학습 데이터를 만든다. (담당: A)

압축을 풀지 않고 zip 안의 XML(CVAT 1.1)과 사진만 골라 읽는다.

클래스 (YOLO id → 파이프라인 MaskClass)
    0 crosswalk  ← roadway/crosswalk, alley/crosswalk          → MaskClass.CROSSWALK(3)
    1 road       ← roadway/normal (+ --alley-as-road 이면 alley) → MaskClass.ROAD(4)

하는 일
1. XML 폴리곤을 z_order 순서로 라벨 맵에 칠한다 (위에 덮인 인도·시설물이 차도를 가리는 것을 반영)
2. 라벨 맵에서 crosswalk / road 영역만 뽑아 외곽선 → YOLO-seg 폴리곤으로 바꾼다
3. 연속 프레임이라 비슷한 사진이 많다. 횡단보도 사진은 전부, 차도만 있는 사진과
   둘 다 없는 사진(배경)은 간격(stride)을 두고 고른다
4. 같은 영상 폴더(Surface_xxx)는 train/val 한쪽에만 들어가게 나눈다 (누수 방지)

출력 (datasets/는 .gitignore에 등록됨)
    datasets/aihub_scene_v1/
    ├── images/{train,val}/
    ├── labels/{train,val}/
    ├── data.yaml
    ├── manifest.csv
    └── check/                 라벨을 겹쳐 그린 확인용 사진 + index.html

사용 예
    python training/aihub_scene_extract.py                       # Surface_1~5 전부
    python training/aihub_scene_extract.py --zips 1 --limit 50 --out datasets/scene_test   # 시험
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ZIP_DIR = ROOT / "_data" / "인도보행 영상" / "서피스마스킹"
NAMES = ["crosswalk", "road"]
COLORS = [(0, 220, 255), (255, 80, 60)]  # BGR: crosswalk 노랑, road 파랑


def open_zip(p: Path) -> zipfile.ZipFile:
    return zipfile.ZipFile(p, metadata_encoding="cp949")


def scene_class(label: str, attr: str, alley_as_road: bool) -> int:
    """폴리곤 → 0 crosswalk / 1 road / -1 그 외(가림막으로만 쓰임)."""
    if label in ("roadway", "alley") and attr == "crosswalk":
        return 0
    if label == "roadway":
        return 1
    if label == "alley" and alley_as_road:
        return 1
    return -1


def parse_xml(data: bytes, folder: str, alley_as_road: bool) -> list[dict]:
    out = []
    for im in ET.fromstring(data).iter("image"):
        polys = []
        for p in im.findall("polygon"):
            a = p.find("attribute")
            attr = (a.text or "").strip() if a is not None else ""
            pts = np.array([[float(v) for v in xy.split(",")] for xy in p.get("points").split(";")], np.float32)
            if len(pts) < 3:
                continue
            polys.append((int(p.get("z_order", 0)), scene_class(p.get("label"), attr, alley_as_road), pts))
        polys.sort(key=lambda t: t[0])
        cls = {c for _, c, _ in polys if c >= 0}
        out.append({"folder": folder, "name": im.get("name"), "W": int(im.get("width")),
                    "H": int(im.get("height")), "polys": polys,
                    "kind": "crosswalk" if 0 in cls else "road" if 1 in cls else "background"})
    return out


def label_map(polys, scale: float, w: int, h: int) -> np.ndarray:
    """z_order 순서로 칠한 라벨 맵. 255 = 비대상(인도 등), 0/1 = 클래스."""
    m = np.full((h, w), 255, np.uint8)
    for _, c, pts in polys:
        cv2.fillPoly(m, [np.round(pts * scale).astype(np.int32)], 255 if c < 0 else c)
    return m


def to_yolo(m: np.ndarray, min_area: float, eps_ratio: float) -> list[tuple[int, np.ndarray]]:
    h, w = m.shape
    out = []
    for c in range(len(NAMES)):
        cnts, _ = cv2.findContours((m == c).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in cnts:
            if cv2.contourArea(cnt) < min_area * w * h:
                continue
            cnt = cv2.approxPolyDP(cnt, eps_ratio * cv2.arcLength(cnt, True), True).reshape(-1, 2)
            if len(cnt) >= 3:
                out.append((c, cnt / [w, h]))
    return out


def split_of(folder: str, val_ratio: float) -> str:
    return "val" if int(hashlib.md5(folder.encode()).hexdigest(), 16) % 1000 < val_ratio * 1000 else "train"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zips", type=int, nargs="+", default=[1, 2, 3, 4, 5], help="사용할 Surface_N.zip 번호")
    ap.add_argument("--long-side", type=int, default=1280)
    ap.add_argument("--road-stride", type=int, default=6, help="차도만 있는 사진은 N장마다 1장")
    ap.add_argument("--bg-stride", type=int, default=30, help="둘 다 없는 사진(배경)은 N장마다 1장")
    ap.add_argument("--crosswalk-stride", type=int, default=1)
    ap.add_argument("--alley-as-road", action="store_true", help="골목길(alley)도 road로 칠함")
    ap.add_argument("--min-area", type=float, default=0.002, help="사진 면적 대비 최소 조각 크기")
    ap.add_argument("--eps", type=float, default=0.002, help="외곽선 단순화 (둘레 대비)")
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--limit", type=int, default=0, help="시험용: 최대 장수")
    ap.add_argument("--n-check", type=int, default=60, help="확인용 겹쳐 그린 사진 수")
    ap.add_argument("--out", type=Path, default=ROOT / "datasets" / "aihub_scene_v1")
    args = ap.parse_args()

    for s in ("train", "val"):
        (args.out / "images" / s).mkdir(parents=True, exist_ok=True)
        (args.out / "labels" / s).mkdir(parents=True, exist_ok=True)
    (args.out / "check").mkdir(exist_ok=True)

    stride = {"crosswalk": args.crosswalk_stride, "road": args.road_stride, "background": args.bg_stride}
    rows, seen = [], {"crosswalk": 0, "road": 0, "background": 0}
    check_every = None
    for k in args.zips:
        z = open_zip(ZIP_DIR / f"Surface_{k}.zip")
        names = set(z.namelist())
        for xml_name in sorted(n for n in names if n.endswith(".xml")):
            folder = xml_name.split("/")[0]
            split = split_of(folder, args.val)
            for it in parse_xml(z.read(xml_name), folder, args.alley_as_road):
                kind = it["kind"]
                seen[kind] += 1
                if (seen[kind] - 1) % stride[kind]:
                    continue
                src = f"{folder}/{it['name']}"
                if src not in names:
                    continue
                img = cv2.imdecode(np.frombuffer(z.read(src), np.uint8), cv2.IMREAD_COLOR)
                if img is None or img.shape[:2] != (it["H"], it["W"]):
                    print(f"  [건너뜀] 읽기 실패/크기 불일치: {src}")
                    continue
                scale = min(1.0, args.long_side / max(it["W"], it["H"]))
                w, h = round(it["W"] * scale), round(it["H"] * scale)
                img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
                polys = to_yolo(label_map(it["polys"], scale, w, h), args.min_area, args.eps)

                stem = f"{folder}_{Path(it['name']).stem}"
                cv2.imwrite(str(args.out / "images" / split / f"{stem}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
                (args.out / "labels" / split / f"{stem}.txt").write_text(
                    "".join(f"{c} " + " ".join(f"{v:.5f}" for v in p.ravel()) + "\n" for c, p in polys))
                n_cls = [sum(c == i for c, _ in polys) for i in range(len(NAMES))]
                rows.append({"file": f"{stem}.jpg", "split": split, "zip": k, "folder": folder,
                             "kind": kind, "n_crosswalk": n_cls[0], "n_road": n_cls[1]})
                if len(rows) % 500 == 0:
                    print(f"  {len(rows)}장 (Surface_{k})")
                if args.limit and len(rows) >= args.limit:
                    break
            if args.limit and len(rows) >= args.limit:
                break
        if args.limit and len(rows) >= args.limit:
            break

    with open(args.out / "manifest.csv", "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    (args.out / "data.yaml").write_text(
        "train: images/train\nval: images/val\nnames:\n"
        + "".join(f"  {i}: {n}\n" for i, n in enumerate(NAMES)))
    make_check(args.out, rows, args.n_check)

    by = lambda key, val: sum(r[key] == val for r in rows)
    print(f"완료: {len(rows)}장 (train {by('split', 'train')}, val {by('split', 'val')}) → {args.out}")
    print(f"  종류: 횡단보도 {by('kind', 'crosswalk')}, 차도만 {by('kind', 'road')}, 배경 {by('kind', 'background')}")
    print(f"  폴리곤: crosswalk {sum(r['n_crosswalk'] for r in rows)}, road {sum(r['n_road'] for r in rows)}")


def make_check(out: Path, rows: list[dict], n: int) -> None:
    """종류별로 고르게 n장을 골라 라벨을 겹쳐 그린다."""
    picks = []
    for kind in ("crosswalk", "road", "background"):
        rs = [r for r in rows if r["kind"] == kind]
        m = n // 2 if kind == "crosswalk" else n // 4
        picks += rs[:: max(1, len(rs) // max(1, m))][:m]
    cards = []
    for r in picks:
        img = cv2.imread(str(out / "images" / r["split"] / r["file"]))
        h, w = img.shape[:2]
        over = img.copy()
        lines = (out / "labels" / r["split"] / r["file"]).with_suffix(".txt").read_text().splitlines()
        for ln in lines:
            v = ln.split()
            pts = (np.array(v[1:], np.float32).reshape(-1, 2) * [w, h]).astype(np.int32)
            cv2.fillPoly(over, [pts], COLORS[int(v[0])])
            cv2.polylines(img, [pts], True, COLORS[int(v[0])], 2)
        img = cv2.addWeighted(over, 0.35, img, 0.65, 0)
        name = f"{r['kind']}_{r['file']}"
        cv2.imwrite(str(out / "check" / name), cv2.resize(img, (w // 2, h // 2)), [cv2.IMWRITE_JPEG_QUALITY, 85])
        cards.append(f'<figure><img src="{html.escape(name)}"><figcaption>{html.escape(name)}'
                     f' · cw {r["n_crosswalk"]} · road {r["n_road"]}</figcaption></figure>')
    (out / "check" / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>장면 라벨 확인</title><style>'
        "body{font:13px system-ui;margin:16px;background:#111;color:#ddd}"
        "main{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:12px}"
        "figure{margin:0}img{width:100%}figcaption{word-break:break-all}</style>"
        "<p>노랑 = crosswalk, 파랑 = road</p><main>" + "\n".join(cards) + "</main>", encoding="utf-8")


if __name__ == "__main__":
    main()
