
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]

SOURCE = ROOT / "datasets" / "roboflow_original"
TARGET = ROOT / "datasets" / "tactile_blocks"

CLASS_MAP = {
    0: 0,  # 가로 선형 -> linear
    1: 0,  # 세로 선형 -> linear
    2: 1,  # 점형 -> dot
}

SPLITS = {
    "train": "train",
    "valid": "val",
    "test": "test",
}

for src_split, dst_split in SPLITS.items():
    src_images = SOURCE / src_split / "images"
    src_labels = SOURCE / src_split / "labels"

    if not src_images.is_dir() or not src_labels.is_dir():
        raise FileNotFoundError(
            f"데이터 폴더를 찾을 수 없습니다: {src_split}"
        )

    dst_images = TARGET / "images" / dst_split
    dst_labels = TARGET / "labels" / dst_split

    dst_images.mkdir(parents=True, exist_ok=True)
    dst_labels.mkdir(parents=True, exist_ok=True)

    for image in src_images.iterdir():
        if image.is_file():
            shutil.copy2(image, dst_images / image.name)

    for label in src_labels.glob("*.txt"):
        converted = []

        for line in label.read_text(
            encoding="utf-8"
        ).splitlines():
            parts = line.split()

            if not parts:
                continue

            old_id = int(parts[0])
            new_id = CLASS_MAP[old_id]

            converted.append(
                " ".join([str(new_id), *parts[1:]])
            )

        (dst_labels / label.name).write_text(
            "\n".join(converted),
            encoding="utf-8",
        )

    print(f"{src_split} -> {dst_split} 변환 완료")

print("전체 데이터셋 변환 완료!")
