"""BDD100K 10K 분할 라벨에서 '하늘·건물·나무만 있고 땅은 없는' 윗부분을 잘라 배경 사진으로 만든다 (맥에서 실행).

    .venv/bin/python training/bdd_sky_crops.py --bdd datasets/bdd100k --n 800

왜: 실제 사진(세로)은 위쪽에 하늘이 크게 나오는데, AI Hub 학습 사진(가로, 보행자 시점)에는 하늘이 거의 없어서
장면 모델이 하늘을 차도로 칠한다. 합성 하늘(scene5_v2)로는 안 고쳐져서 실제 하늘 사진을 '아무것도 아님'으로 넣는다.

- BDD 클래스 id (Cityscapes와 같음): 0 road, 1 sidewalk, 8 vegetation, 9 terrain, 10 sky, 2 building ...
- 자르는 곳: 땅(road·sidewalk·terrain)과 차·사람이 시작되는 줄 위쪽만. 하늘 15% 이상, 높이가 원본의 30% 이상
- 가로세로 비율을 섞어서 자름 (세로 0.6 ~ 가로 1.6). 라벨 파일은 빈 파일 = 배경 사진
- 밝기로 낮·밤을 나눠 밤이 약 25%가 되게 고른다
- 결과: <out>/images/*.jpg, <out>/labels/*.txt(빈 파일), <out>/check/index.html → zip으로 묶어 서버에 올린다

BDD100K는 연구용(비상업) 라이선스. 다시 배포하지 말 것.
"""

import argparse
import glob
import html
import random
import shutil
from pathlib import Path

import cv2
import numpy as np

GROUND = [0, 1, 9]  # road, sidewalk, terrain
OBJECTS = [11, 12, 13, 14, 15, 16, 17, 18]  # person, rider, car, truck, bus, train, motorcycle, bicycle
SKY = 10

ap = argparse.ArgumentParser()
ap.add_argument("--bdd", type=Path, default=Path("datasets/bdd100k"))
ap.add_argument("--out", type=Path, default=Path("datasets/bdd_sky"))
ap.add_argument("--n", type=int, default=800, help="만들 배경 사진 수")
ap.add_argument("--night", type=float, default=0.25, help="밤(어두운) 사진 비율")
ap.add_argument("--long-side", type=int, default=960)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
rng = random.Random(args.seed)

# 사진과 마스크 짝 찾기 (배포판마다 폴더 구조가 달라서 이름으로 맞춘다)
imgs = {Path(p).stem: p for p in glob.glob(str(args.bdd / "**/*.jpg"), recursive=True)}
masks = {}
for p in glob.glob(str(args.bdd / "**/*.png"), recursive=True):
    if "color" in p or "colormap" in p:  # 색칠 미리보기는 건너뜀
        continue
    masks[Path(p).stem.replace("_train_id", "")] = p
pairs = [(imgs[k], masks[k]) for k in sorted(set(imgs) & set(masks))]
print(f"사진 {len(imgs)} · 마스크 {len(masks)} · 짝 {len(pairs)}")
if not pairs:
    raise SystemExit("짝이 없음. images/10k/... 와 labels/sem_seg/masks/... 를 --bdd 아래에 풀었는지 확인")
rng.shuffle(pairs)


def crop_one(img, m):
    h, w = m.shape
    stop = np.isin(m, GROUND + OBJECTS)
    rows = np.where(stop.any(1))[0]
    top = int(rows.min()) if len(rows) else h  # 땅·차가 처음 나오는 줄
    # 기둥처럼 위로 튀어나온 물체 때문에 너무 위에서 잘리지 않게, 열마다 첫 줄을 보고 10% 분위로
    first = np.where(stop.any(0), stop.argmax(0), h)
    top = max(top, int(np.percentile(first, 10)))
    if top < 0.3 * h:
        return None
    ch = rng.randint(int(0.3 * h), top)  # 자를 높이
    cw = min(w, int(ch * rng.uniform(0.6, 1.6)))
    x = rng.randint(0, w - cw)
    y = rng.randint(0, top - ch)
    cm = m[y:y + ch, x:x + cw]
    if np.isin(cm, GROUND + OBJECTS).mean() > 0.003 or (cm == SKY).mean() < 0.15:
        return None
    return img[y:y + ch, x:x + cw], float(img[y:y + ch, x:x + cw].mean())


want_night = round(args.n * args.night)
day, night = [], []
for ip, mp in pairs:
    if len(day) >= args.n - want_night and len(night) >= want_night:
        break
    img, m = cv2.imread(ip), cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
    if img is None or m is None or img.shape[:2] != m.shape:
        continue
    for _ in range(3):  # 자리를 몇 번 바꿔 시도
        r = crop_one(img, m)
        if r:
            break
    if not r:
        continue
    c, bright = r
    (night if bright < 70 else day).append((Path(ip).stem, c, bright))

pick = night[:want_night] + day[:args.n - min(len(night), want_night)]
if args.out.exists():
    shutil.rmtree(args.out)
(args.out / "images").mkdir(parents=True)
(args.out / "labels").mkdir()
(args.out / "check").mkdir()
cards = []
for i, (stem, c, bright) in enumerate(pick):
    s = min(1.0, args.long_side / max(c.shape[:2]))
    c = cv2.resize(c, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    name = f"bddsky_{i:04d}_{stem}"
    cv2.imwrite(str(args.out / "images" / f"{name}.jpg"), c, [cv2.IMWRITE_JPEG_QUALITY, 90])
    (args.out / "labels" / f"{name}.txt").write_text("")
    if i % 5 == 0:  # 미리보기는 5장에 1장
        t = cv2.resize(c, None, fx=240 / c.shape[0], fy=240 / c.shape[0])
        cv2.imwrite(str(args.out / "check" / f"{name}.jpg"), t, [cv2.IMWRITE_JPEG_QUALITY, 80])
        cards.append(f'<figure><img src="{html.escape(name)}.jpg"><figcaption>{"밤" if bright < 70 else "낮"} · 밝기 {bright:.0f}</figcaption></figure>')
(args.out / "check" / "index.html").write_text(
    '<!doctype html><meta charset="utf-8"><title>BDD 하늘 배경</title><style>body{font:13px system-ui;background:#111;color:#ddd;margin:16px}'
    "main{display:flex;flex-wrap:wrap;gap:8px}figure{margin:0}img{height:200px}</style>"
    f"<h3>BDD 하늘 배경 {len(pick)}장 (밤 {min(len(night), want_night)} · 낮 {len(pick) - min(len(night), want_night)}) — 5장에 1장만 표시. "
    "땅·차·사람이 보이면 안 됨</h3><main>" + "\n".join(cards) + "</main>", encoding="utf-8")
print(f"배경 {len(pick)}장 (밤 {min(len(night), want_night)}, 낮 {len(pick) - min(len(night), want_night)}) → {args.out}")
print(f"미리보기: {args.out / 'check/index.html'}")
shutil.make_archive(str(args.out), "zip", args.out.parent, args.out.name)
print(f"zip: {args.out}.zip")
