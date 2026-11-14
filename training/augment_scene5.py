"""5클래스 장면 데이터에 실제 촬영 조건(하늘·세로 구도·야간)을 흉내 낸 사진을 더한다.

AI Hub 인도보행 사진은 전부 낮에 가로(1280×720)로 바닥을 보고 찍어서 하늘·야간·세로 사진이 거의 없다.
그래서 모델이 하늘을 차도로 칠하거나 밤 사진에서 약해진다. 학습(train) 쪽에만 아래 사진을 추가한다.

    sky    가로 사진에서 720폭을 잘라 위에 하늘(맑음·흐림·노을·밤)을 붙인 세로 사진. 하늘에는 라벨이 없다
    night  어둡게 + 색 틀어짐 + 노이즈 + 가로등 번짐
    검증(val)은 원본 그대로 둔다 (이전 모델과 점수를 비교할 수 있게).

출력은 긴 변 960px로 줄여 저장한다 (학습은 640이라 충분하고 zip이 작아진다).

사용 예 (레포 최상위에서)
    .venv/bin/python training/augment_scene5.py --limit 60 --out datasets/scene5b_test   # 시험
    .venv/bin/python training/augment_scene5.py                                          # → datasets/aihub_scene5b
"""

from __future__ import annotations

import argparse
import html
import random
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
NAMES = ["crosswalk", "road", "alley", "sidewalk", "tactile"]
COLORS = [(255, 0, 255), (255, 120, 40), (60, 200, 60), (230, 230, 0), (0, 220, 255)]
IGNORE = 255


def read_label_map(txt: Path, w: int, h: int) -> np.ndarray:
    m = np.full((h, w), IGNORE, np.uint8)
    polys = [ln.split() for ln in txt.read_text().splitlines() if ln.strip()]
    for v in sorted(polys, key=lambda v: {3: 0, 2: 1, 1: 2, 0: 3, 4: 4}[int(v[0])]):
        pts = np.round(np.array(v[1:], np.float32).reshape(-1, 2) * [w, h]).astype(np.int32)
        cv2.fillPoly(m, [pts], int(v[0]))
    return m


def to_yolo_text(m: np.ndarray, min_area: float = 0.002, eps_ratio: float = 0.002) -> str:
    h, w = m.shape
    out = []
    for c in range(len(NAMES)):
        cnts, _ = cv2.findContours((m == c).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in cnts:
            if cv2.contourArea(cnt) < min_area * w * h:
                continue
            cnt = cv2.approxPolyDP(cnt, eps_ratio * cv2.arcLength(cnt, True), True).reshape(-1, 2)
            if len(cnt) >= 3:
                out.append(f"{c} " + " ".join(f"{v:.5f}" for v in (cnt / [w, h]).ravel()))
    return "".join(s + "\n" for s in out)


def smooth_noise(h: int, w: int, rng: np.random.Generator, cells: int) -> np.ndarray:
    """0~1 사이의 부드러운 무늬 (구름·밝기 얼룩)."""
    g = rng.random((max(2, cells * h // max(h, w) + 1), max(2, cells * w // max(h, w) + 1))).astype(np.float32)
    return cv2.resize(g, (w, h), interpolation=cv2.INTER_CUBIC).clip(0, 1)


def make_sky(h: int, w: int, rng: np.random.Generator) -> tuple[np.ndarray, bool]:
    """(하늘 그림 BGR, 밤 여부). 위에서 아래로 색이 변하고 구름 얼룩이 있다."""
    kind = rng.choice(["clear", "overcast", "dusk", "night"], p=[0.35, 0.3, 0.1, 0.25])
    top, bot = {
        "clear": ((225, 150, 80), (240, 215, 185)),
        "overcast": ((205, 200, 195), (235, 232, 228)),
        "dusk": ((150, 110, 90), (150, 190, 235)),
        "night": ((28, 16, 12), (55, 42, 38)),
    }[kind]
    jit = rng.normal(0, 12, 3)
    t = np.linspace(0, 1, h, dtype=np.float32)[:, None, None]
    sky = (1 - t) * (np.array(top, np.float32) + jit) + t * (np.array(bot, np.float32) + jit)
    sky = np.repeat(sky, w, axis=1)
    cloud = smooth_noise(h, w, rng, int(rng.integers(3, 9)))[..., None]
    amount = {"clear": 0.35, "overcast": 0.12, "dusk": 0.3, "night": 0.25}[kind] * rng.random()
    target = np.array((70, 60, 60) if kind == "night" else (245, 245, 245), np.float32)
    sky = sky * (1 - amount * cloud) + target * amount * cloud
    sky += rng.normal(0, 2.5, sky.shape)
    return sky.clip(0, 255).astype(np.uint8), kind == "night"


def night(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """낮 사진을 밤처럼: 어둡게, 채도 낮춤, 가로등 색, 번짐, 노이즈."""
    h, w = img.shape[:2]
    x = img.astype(np.float32) / 255
    gray = x.mean(axis=2, keepdims=True)
    x = gray + (x - gray) * rng.uniform(0.45, 0.8)  # 채도
    x = x ** rng.uniform(1.6, 2.6) * rng.uniform(0.5, 0.8)
    tint = np.array([(0.8, 0.95, 1.12), (1.08, 1.0, 0.92), (1.0, 1.0, 1.0)][int(rng.integers(3))], np.float32)
    x *= tint
    light = np.zeros((h, w), np.float32)  # 가로등이 비춘 밝은 곳
    for _ in range(int(rng.integers(1, 4))):
        cx, cy, r = rng.uniform(0, w), rng.uniform(0, h), rng.uniform(0.25, 0.6) * max(h, w)
        yy, xx = np.ogrid[:h, :w]
        light = np.maximum(light, np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * (r / 2) ** 2)))
    x *= (0.55 + 0.9 * light)[..., None]
    x += rng.normal(0, rng.uniform(0.012, 0.035), x.shape)
    return (x.clip(0, 1) * 255).astype(np.uint8)


def sky_portrait(img: np.ndarray, lab: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, bool]:
    """가로 사진에서 정사각형에 가까운 창을 잘라 위에 하늘을 붙인다 → 세로 사진."""
    h, w = img.shape[:2]
    cw = min(w, int(h * rng.uniform(0.95, 1.25)))
    x0 = int(rng.integers(0, w - cw + 1))
    img, lab = img[:, x0:x0 + cw], lab[:, x0:x0 + cw]
    sh = int(h * rng.uniform(0.3, 0.75))
    sky, is_night = make_sky(sh, cw, rng)
    if is_night:
        img = night(img, rng)
    out = np.vstack([sky, img])
    k = max(3, (h // 40) | 1)  # 이음매를 흐리게
    band = out[sh - k:sh + k].copy()
    out[sh - k:sh + k] = cv2.GaussianBlur(band, (0, 0), k / 2)
    return out, np.vstack([np.full((sh, cw), IGNORE, np.uint8), lab]), is_night


def save(out: Path, split: str, stem: str, img: np.ndarray, label_text: str, long_side: int) -> None:
    s = min(1.0, long_side / max(img.shape[:2]))
    if s < 1:
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(out / "images" / split / f"{stem}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    (out / "labels" / split / f"{stem}.txt").write_text(label_text)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=ROOT / "datasets" / "aihub_scene5")
    ap.add_argument("--out", type=Path, default=ROOT / "datasets" / "aihub_scene5b")
    ap.add_argument("--sky-ratio", type=float, default=0.2, help="학습 사진 중 하늘 붙인 세로 사진을 만들 비율")
    ap.add_argument("--night-ratio", type=float, default=0.2, help="학습 사진 중 야간 사본을 만들 비율")
    ap.add_argument("--long-side", type=int, default=960)
    ap.add_argument("--limit", type=int, default=0, help="시험용: 학습 사진 최대 장수")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)
    for s in ("train", "val"):
        (args.out / "images" / s).mkdir(parents=True, exist_ok=True)
        (args.out / "labels" / s).mkdir(parents=True, exist_ok=True)
    (args.out / "check").mkdir(exist_ok=True)

    n = {"train": 0, "val": 0, "sky": 0, "night": 0}
    checks = []
    for split in ("val", "train"):
        files = sorted((args.src / "images" / split).glob("*.jpg"))
        if args.limit:
            files = files[:: max(1, len(files) // args.limit)][: args.limit]
        for i, f in enumerate(files):
            img = cv2.imread(str(f))
            txt = args.src / "labels" / split / f"{f.stem}.txt"
            save(args.out, split, f.stem, img, txt.read_text(), args.long_side)
            n[split] += 1
            if split == "val":
                continue
            r = rng.random()
            if r < args.sky_ratio:
                h, w = img.shape[:2]
                im2, lab2, _ = sky_portrait(img, read_label_map(txt, w, h), rng)
                save(args.out, split, f"{f.stem}_sky", im2, to_yolo_text(lab2), args.long_side)
                n["sky"] += 1
                if len(checks) < 40 and n["sky"] % 7 == 1:
                    checks.append(f"{f.stem}_sky")
            elif r < args.sky_ratio + args.night_ratio:
                save(args.out, split, f"{f.stem}_night", night(img, rng), txt.read_text(), args.long_side)
                n["night"] += 1
                if len(checks) < 40 and n["night"] % 7 == 1:
                    checks.append(f"{f.stem}_night")
            if (i + 1) % 1000 == 0:
                print(f"  train {i + 1}/{len(files)}", flush=True)

    (args.out / "data.yaml").write_text(
        "train: images/train\nval: images/val\nnames:\n" + "".join(f"  {i}: {x}\n" for i, x in enumerate(NAMES)))
    cards = []
    for stem in checks:  # 라벨이 사진과 맞는지 눈으로 확인
        img = cv2.imread(str(args.out / "images" / "train" / f"{stem}.jpg"))
        h, w = img.shape[:2]
        over = img.copy()
        for ln in (args.out / "labels" / "train" / f"{stem}.txt").read_text().splitlines():
            v = ln.split()
            cv2.fillPoly(over, [(np.array(v[1:], np.float32).reshape(-1, 2) * [w, h]).astype(np.int32)], COLORS[int(v[0])])
        both = np.hstack([img, cv2.addWeighted(over, 0.4, img, 0.6, 0)])
        cv2.imwrite(str(args.out / "check" / f"{stem}.jpg"), cv2.resize(both, None, fx=0.5, fy=0.5), [cv2.IMWRITE_JPEG_QUALITY, 85])
        cards.append(f'<figure><img src="{html.escape(stem)}.jpg"><figcaption>{html.escape(stem)}</figcaption></figure>')
    (args.out / "check" / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>보강 사진 확인</title><style>'
        "body{font:13px system-ui;margin:16px;background:#111;color:#ddd}"
        "main{display:grid;grid-template-columns:repeat(auto-fill,minmax(420px,1fr));gap:12px}"
        "figure{margin:0}img{width:100%}figcaption{word-break:break-all}</style>"
        "<p>왼쪽 사진 · 오른쪽 라벨 | 분홍 crosswalk · 파랑 road · 초록 alley · 하늘색 sidewalk · 노랑 tactile</p><main>"
        + "\n".join(cards) + "</main>", encoding="utf-8")
    print(f"완료 → {args.out}")
    print(f"  train 원본 {n['train']} + 하늘 세로 {n['sky']} + 야간 {n['night']} = {n['train'] + n['sky'] + n['night']} · val {n['val']}")


if __name__ == "__main__":
    main()
