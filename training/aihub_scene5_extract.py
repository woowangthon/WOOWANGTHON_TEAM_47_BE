"""AI Hub '인도보행 영상' 서피스마스킹 zip에서 5클래스 장면 모델 학습 데이터를 만든다.

aihub_scene_extract.py(2클래스)에서 갈라져 나옴. 차도와 헷갈리는 인도·골목길·점자블록을
따로 가르쳐서 차도 오탐(가짜 C2)을 줄이는 것이 목적이다. 서버는 crosswalk·road만 쓰고
나머지(alley, sidewalk, tactile)는 무시한다.

클래스 (YOLO id ← AI Hub 라벨)
    0 crosswalk ← roadway/crosswalk, alley/crosswalk
    1 road      ← roadway/normal (그 밖의 roadway)
    2 alley     ← alley/normal·damaged·speed_bump·blocks
    3 sidewalk  ← sidewalk/전부, bike_lane
    4 tactile   ← braille_guide_blocks/전부 (점형·선형 구분 없이 블록 영역)
    caution_zone(맨홀·배수구 등)은 라벨 없이 가림막으로만 쓴다.

사진 고르기 (연속 프레임이라 비슷한 사진이 많아 간격을 둔다. 위에서부터 먼저 걸리는 종류로 분류)
    crosswalk  횡단보도가 있는 사진            --crosswalk-stride (기본 1 = 전부)
    hard       빨간 포장(urethane)·자전거도로·아스팔트 인도가 있는 사진 (차도 오탐 원인)  --hard-stride (기본 2)
    road       차도가 있는 사진                 --road-stride (기본 3)
    alley      골목길이 있는 사진               --alley-stride (기본 8)
    other      그 밖의 사진 (인도·블록만 등)    --other-stride (기본 20)

같은 영상 폴더(Surface_xxx)는 train/val 한쪽에만 들어가게 나눈다 (누수 방지, 2클래스판과 같은 규칙).

사용 예 (레포 최상위에서)
    .venv/bin/python training/aihub_scene5_extract.py --dry-run          # 장수만 세기
    .venv/bin/python training/aihub_scene5_extract.py --limit 80 --out datasets/scene5_test
    .venv/bin/python training/aihub_scene5_extract.py                    # 전체 → datasets/aihub_scene5
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import time
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ZIP_DIR = ROOT / "_data" / "인도보행 영상" / "서피스마스킹"
NAMES = ["crosswalk", "road", "alley", "sidewalk", "tactile"]
COLORS = [(255, 0, 255), (255, 120, 40), (60, 200, 60), (230, 230, 0), (0, 220, 255)]  # BGR
KINDS = ["crosswalk", "hard", "road", "alley", "other"]
HARD = {("sidewalk", "urethane"), ("sidewalk", "asphalt"), ("bike_lane", "")}
IGNORE = 255


def open_zip(p: Path) -> zipfile.ZipFile:
    return zipfile.ZipFile(p, metadata_encoding="cp949")


def scene_class(label: str, attr: str) -> int:
    """폴리곤 → 클래스 id, 그 외는 -1 (가림막으로만 쓰임)."""
    if label in ("roadway", "alley") and attr == "crosswalk":
        return 0
    if label == "roadway":
        return 1
    if label == "alley":
        return 2
    if label in ("sidewalk", "bike_lane"):
        return 3
    if label == "braille_guide_blocks":
        return 4
    return -1


def parse_xml(data: bytes, folder: str) -> list[dict]:
    out = []
    for im in ET.fromstring(data).iter("image"):
        polys, hard = [], False
        for p in im.findall("polygon"):
            a = p.find("attribute")
            attr = (a.text or "").strip() if a is not None else ""
            label = p.get("label")
            pts = np.array([[float(v) for v in xy.split(",")] for xy in p.get("points").split(";")], np.float32)
            if len(pts) < 3:
                continue
            hard |= (label, attr) in HARD
            polys.append((int(p.get("z_order", 0)), scene_class(label, attr), pts))
        polys.sort(key=lambda t: t[0])
        cls = {c for _, c, _ in polys if c >= 0}
        kind = ("crosswalk" if 0 in cls else "hard" if hard else "road" if 1 in cls
                else "alley" if 2 in cls else "other")
        out.append({"folder": folder, "name": im.get("name"), "W": int(im.get("width")),
                    "H": int(im.get("height")), "polys": polys, "kind": kind})
    return out


def label_map(polys, scale: float, w: int, h: int) -> np.ndarray:
    """z_order 순서로 칠한 라벨 맵. 255 = 라벨 없음(가림막 포함), 0~4 = 클래스."""
    m = np.full((h, w), IGNORE, np.uint8)
    for _, c, pts in polys:
        cv2.fillPoly(m, [np.round(pts * scale).astype(np.int32)], IGNORE if c < 0 else c)
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
    ap.add_argument("--crosswalk-stride", type=int, default=1)
    ap.add_argument("--hard-stride", type=int, default=2)
    ap.add_argument("--road-stride", type=int, default=3)
    ap.add_argument("--alley-stride", type=int, default=8)
    ap.add_argument("--other-stride", type=int, default=20)
    ap.add_argument("--min-area", type=float, default=0.002, help="사진 면적 대비 최소 조각 크기")
    ap.add_argument("--eps", type=float, default=0.002, help="외곽선 단순화 (둘레 대비)")
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--limit", type=int, default=0, help="시험용: 최대 장수")
    ap.add_argument("--dry-run", action="store_true", help="사진을 읽지 않고 고를 장수만 센다")
    ap.add_argument("--n-check", type=int, default=60, help="확인용 겹쳐 그린 사진 수")
    ap.add_argument("--out", type=Path, default=ROOT / "datasets" / "aihub_scene5")
    args = ap.parse_args()

    stride = {"crosswalk": args.crosswalk_stride, "hard": args.hard_stride, "road": args.road_stride,
              "alley": args.alley_stride, "other": args.other_stride}
    if not args.dry_run:
        for s in ("train", "val"):
            (args.out / "images" / s).mkdir(parents=True, exist_ok=True)
            (args.out / "labels" / s).mkdir(parents=True, exist_ok=True)
        (args.out / "check").mkdir(exist_ok=True)

    rows, seen, picked = [], Counter(), Counter()
    t0 = time.time()
    for k in args.zips:
        z = open_zip(ZIP_DIR / f"Surface_{k}.zip")
        names = set(z.namelist())
        for xml_name in sorted(n for n in names if n.endswith(".xml")):
            folder = xml_name.split("/")[0]
            split = split_of(folder, args.val)
            for it in parse_xml(z.read(xml_name), folder):
                kind = it["kind"]
                seen[kind] += 1
                if (seen[kind] - 1) % stride[kind]:
                    continue
                src = f"{folder}/{it['name']}"
                if src not in names:
                    continue
                if args.dry_run:
                    picked[(kind, split)] += 1
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
                row = {"file": f"{stem}.jpg", "split": split, "zip": k, "folder": folder, "kind": kind}
                row.update({f"n_{n}": sum(c == i for c, _ in polys) for i, n in enumerate(NAMES)})
                rows.append(row)
                if len(rows) % 500 == 0:
                    print(f"  {len(rows)}장 (Surface_{k}, {time.time() - t0:.0f}초)", flush=True)
                if args.limit and len(rows) >= args.limit:
                    break
            if args.limit and len(rows) >= args.limit:
                break
        if args.limit and len(rows) >= args.limit:
            break

    if args.dry_run:
        print("전체 프레임 종류:", dict(seen))
        for kind in KINDS:
            print(f"  {kind:10s} train {picked[(kind, 'train')]:6d} · val {picked[(kind, 'val')]:6d}")
        print(f"  합계 {sum(picked.values())}장")
        return

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
    print("  종류: " + ", ".join(f"{kd} {by('kind', kd)}" for kd in KINDS))
    print("  폴리곤: " + ", ".join(f"{n} {sum(r[f'n_{n}'] for r in rows)}" for n in NAMES))


def make_check(out: Path, rows: list[dict], n: int) -> None:
    """종류별로 고르게 n장을 골라 라벨을 겹쳐 그린다. z 순서대로 칠해 블록·횡단보도가 위에 보이게 한다."""
    picks = []
    for kind in KINDS:
        rs = [r for r in rows if r["kind"] == kind]
        m = n // len(KINDS)
        picks += rs[:: max(1, len(rs) // max(1, m))][:m]
    order = {3: 0, 2: 1, 1: 2, 0: 3, 4: 4}  # sidewalk → alley → road → crosswalk → tactile
    cards = []
    for r in picks:
        img = cv2.imread(str(out / "images" / r["split"] / r["file"]))
        h, w = img.shape[:2]
        over = img.copy()
        lines = (out / "labels" / r["split"] / r["file"]).with_suffix(".txt").read_text().splitlines()
        polys = [(int(v[0]), np.array(v[1:], np.float32)) for v in (ln.split() for ln in lines)]
        for c, pts in sorted(polys, key=lambda t: order[t[0]]):
            pts = (pts.reshape(-1, 2) * [w, h]).astype(np.int32)
            cv2.fillPoly(over, [pts], COLORS[c])
            cv2.polylines(img, [pts], True, COLORS[c], 2)
        img = cv2.addWeighted(over, 0.4, img, 0.6, 0)
        name = f"{r['kind']}_{r['file']}"
        cv2.imwrite(str(out / "check" / name), cv2.resize(img, (w // 2, h // 2)), [cv2.IMWRITE_JPEG_QUALITY, 85])
        cnt = " · ".join(f"{nm} {r[f'n_{nm}']}" for nm in NAMES)
        cards.append(f'<figure><img src="{html.escape(name)}"><figcaption>{html.escape(name)}'
                     f"<br>{html.escape(cnt)}</figcaption></figure>")
    legend = "분홍 crosswalk · 파랑 road · 초록 alley · 하늘색 sidewalk · 노랑 tactile"
    (out / "check" / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>장면 5클래스 라벨 확인</title><style>'
        "body{font:13px system-ui;margin:16px;background:#111;color:#ddd}"
        "main{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:12px}"
        "figure{margin:0}img{width:100%}figcaption{word-break:break-all}</style>"
        f"<p>{legend}</p><main>" + "\n".join(cards) + "</main>", encoding="utf-8")


if __name__ == "__main__":
    main()
