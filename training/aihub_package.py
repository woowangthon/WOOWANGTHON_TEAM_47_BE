"""aihub_extract.py로 고른 후보를 원본 해상도 사진 + 원본 JSON 묶음 zip으로 만든다.

팀원에게 넘길 때 쓴다. 팀원 폴더(_data/datasets_aihub)와 같은 형식:
zip 최상위에 <이름>.jpg + <이름>.json 이 나란히 있고, 둘 다 AI Hub 원본 그대로다.
--exclude-dir 에 있는 파일과 같은 사진(파일명·image_id)은 뺀다.

사용 예
    python training/aihub_package.py --exclude-dir _data/datasets_aihub
"""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

from aihub_extract import IMAGE_ZIP, LABEL_ZIP, ROOT, TACTILE_DIR, open_zip


def _info(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8-sig"))["info"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", type=Path, default=ROOT / "datasets" / "aihub_candidates" / "json")
    ap.add_argument("--exclude-dir", type=Path, action="append", default=[])
    ap.add_argument("--out", type=Path, default=ROOT / "datasets" / "aihub_defect_originals.zip")
    args = ap.parse_args()

    skip_names, skip_ids = set(), set()
    for d in args.exclude_dir:
        for p in d.glob("*.json"):
            info = _info(p)
            skip_names.add(Path(info["filename"]).stem)
            skip_ids.add(info["image_id"])

    zl, zi = open_zip(LABEL_ZIP), open_zip(IMAGE_ZIP)
    labels = {n.rsplit("/", 1)[-1]: n for n in zl.namelist() if n.startswith(TACTILE_DIR)}
    images = {n.rsplit("/", 1)[-1]: n for n in zi.namelist() if n.startswith(TACTILE_DIR)}

    n_ok = n_skip = 0
    with zipfile.ZipFile(args.out, "w", zipfile.ZIP_STORED) as out:
        for p in sorted(args.candidates.glob("*.json")):
            info = _info(p)
            fn = info["filename"]
            if Path(fn).stem in skip_names or info["image_id"] in skip_ids:
                n_skip += 1
                continue
            jname = Path(fn).with_suffix(".json").name
            if fn not in images or jname not in labels:
                print(f"  [건너뜀] 원본 zip에 없음: {fn}")
                continue
            out.writestr(jname, zl.read(labels[jname]))  # JSON 원본 바이트 그대로
            out.writestr(fn, zi.read(images[fn]))  # 사진 원본 바이트 그대로
            n_ok += 1

    print(f"완료: {n_ok}쌍 (겹쳐서 뺀 것 {n_skip}) → {args.out} ({args.out.stat().st_size / 1e9:.2f}GB)")


if __name__ == "__main__":
    main()
