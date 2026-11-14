"""aihub_scene5(5클래스)에서 골목길을 빼고 BDD 하늘 배경을 더해 4클래스 학습 폴더를 만든다.
표준 라이브러리만 씀 → 서버에서 바로 실행 (사진은 복사하지 않고 하드링크).

    python3 make_scene4.py --src ~/aihub_scene5 --sky ~/bdd_sky --out ~/aihub_scene4

- 클래스: 0 crosswalk · 1 road · 2 sidewalk · 3 tactile  (alley 삭제, 뒤 번호를 당김)
- 골목길 자리는 라벨 없음(배경)이 된다. 골목길 안 횡단보도는 원래대로 crosswalk
  → 골목길은 차도·인도 구분이 애매하고 정답 라벨도 흔들려서, '차도가 아닌 곳'으로만 배우게 한다
- 골목길만 있던 사진은 라벨이 비어 배경 사진이 된다 (약 1,260장). 배경이 너무 많으면 차도를 덜 잡게 되므로
  --empty-keep 비율(기본 0.5)만 남긴다. 하늘 배경 800장과 합쳐 train의 약 13%
- --sky 의 배경 사진(빈 라벨)은 train에만 넣는다. val은 AI Hub 그대로라 점수는 이전 모델과 비교 가능
"""

import argparse
import hashlib
import os
import shutil
from pathlib import Path

NAMES = ["crosswalk", "road", "sidewalk", "tactile"]
REMAP = {"0": "0", "1": "1", "2": None, "3": "2", "4": "3"}  # 5클래스 id → 4클래스 id (None = 버림)

ap = argparse.ArgumentParser()
ap.add_argument("--src", type=Path, default=Path("~/aihub_scene5").expanduser())
ap.add_argument("--sky", type=Path, default=None, help="bdd_sky 폴더 (images/, labels/)")
ap.add_argument("--empty-keep", type=float, default=0.5, help="골목길만 있어 빈 라벨이 된 train 사진 중 남길 비율")
ap.add_argument("--out", type=Path, default=Path("~/aihub_scene4").expanduser())
args = ap.parse_args()
src, out = args.src.expanduser(), args.out.expanduser()


def put(a, b):
    try:
        os.link(a, b)
    except OSError:  # 다른 디스크면 복사
        shutil.copy2(a, b)


if out.exists():
    shutil.rmtree(out)
dropped = {"train": 0, "val": 0}
skipped = 0
for split in ("train", "val"):
    (out / "images" / split).mkdir(parents=True)
    (out / "labels" / split).mkdir(parents=True)
    for img in (src / "images" / split).glob("*.jpg"):
        lp = src / "labels" / split / (img.stem + ".txt")
        lines = []
        if lp.exists():
            for line in lp.read_text().splitlines():
                c, _, rest = line.partition(" ")
                new = REMAP.get(c)
                if new is None:
                    dropped[split] += 1
                    continue
                lines.append(f"{new} {rest}")
        if split == "train" and not lines:  # 이름 해시로 고르므로 다시 돌려도 같은 사진이 남는다
            if int(hashlib.md5(img.stem.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF >= args.empty_keep:
                skipped += 1
                continue
        put(img, out / "images" / split / img.name)
        (out / "labels" / split / (img.stem + ".txt")).write_text("\n".join(lines) + ("\n" if lines else ""))

n_sky = 0
if args.sky:
    sky = args.sky.expanduser()
    for img in (sky / "images").glob("*.jpg"):
        put(img, out / "images" / "train" / img.name)
        (out / "labels" / "train" / (img.stem + ".txt")).write_text("")
        n_sky += 1

(out / "data.yaml").write_text(f"path: {out}\ntrain: images/train\nval: images/val\nnames:\n"
                               + "".join(f"  {i}: {n}\n" for i, n in enumerate(NAMES)))
n_tr = len(list((out / "images" / "train").glob("*.jpg")))
n_va = len(list((out / "images" / "val").glob("*.jpg")))
print(f"train {n_tr} (그중 하늘 배경 {n_sky}) · val {n_va}")
print(f"뺀 골목길 전용 사진(빈 라벨): {skipped}")
print(f"지운 골목길 다각형: train {dropped['train']} · val {dropped['val']}")
print(f"→ {out}/data.yaml  ({', '.join(NAMES)})")
