"""5클래스 장면 모델 학습 (학교 GPU 서버용). aihub_scene5.zip을 푼 폴더에서 실행.

    python3 -u train_scene5.py --data ~/aihub_scene5b --name scene5_v2 --device 0 --batch 16

- 처음부터 학습한다 (yolov8s-seg.pt). v3(2클래스)에서 이 방식이 1에폭 최고 문제 없이 안정적이었음
- 결과: <project>/<name>/weights/best.pt  (기본 ~/runs/scene5_v2/weights/best.pt)
- v1(scene5_v1): aihub_scene5, 50에폭, 기본 증강
- v2(scene5_v2): aihub_scene5b(하늘 세로·야간 사본 추가), 70에폭, 밝기·회전 증강을 조금 키움
- 끊기면 --resume 으로 last.pt에서 이어서 학습
- scene4_v1: aihub_scene4(골목길 삭제 + BDD 하늘 배경 800장), 기본 증강(--plain-aug)
    python3 -u train_scene5.py --data ~/aihub_scene4 --name scene4_v1 --plain-aug --epochs 50 --patience 15
- 클래스 이름은 데이터 폴더의 data.yaml에서 읽는다 (4클래스·5클래스 모두 됨)
- 서버는 crosswalk·road만 판정에 쓰고 나머지는 이름으로 건너뛴다 (tactile은 이중검출에 씀)
"""

import argparse
from pathlib import Path

from ultralytics import YOLO

NAMES = ["crosswalk", "road", "alley", "sidewalk", "tactile"]

ap = argparse.ArgumentParser()
ap.add_argument("--data", type=Path, default=Path("~/aihub_scene5b").expanduser())
ap.add_argument("--model", default="yolov8s-seg.pt")
ap.add_argument("--epochs", type=int, default=70)
ap.add_argument("--patience", type=int, default=20)
ap.add_argument("--imgsz", type=int, default=640)
ap.add_argument("--batch", type=int, default=16, help="GPU를 같이 쓰면 줄일 것. -1이면 자동")
ap.add_argument("--device", default="0")
ap.add_argument("--workers", type=int, default=8)
ap.add_argument("--project", default=str(Path("~/runs").expanduser()))
ap.add_argument("--name", default="scene5_v2")
ap.add_argument("--plain-aug", action="store_true", help="증강을 ultralytics 기본값으로 (v1과 같은 조건)")
ap.add_argument("--cache", default="", choices=["", "ram", "disk"], help="사진을 미리 풀어 둠 (사진 읽기가 느릴 때). ram은 메모리 약 10GB 필요")
ap.add_argument("--resume", action="store_true")
args = ap.parse_args()

data = args.data.expanduser().resolve()
src_yaml = (data / "data.yaml").read_text().splitlines()  # names: 아래 "  0: crosswalk" 줄들
i = next((k for k, l in enumerate(src_yaml) if l.strip() == "names:"), None)
if i is not None:
    NAMES = [l.split(":", 1)[1].strip() for l in src_yaml[i + 1:] if l.startswith(" ") and ":" in l]
print("클래스", NAMES)
yaml = data / "data_abs.yaml"  # 서버 절대 경로로 다시 씀
yaml.write_text(f"path: {data}\ntrain: images/train\nval: images/val\nnames:\n"
                + "".join(f"  {i}: {n}\n" for i, n in enumerate(NAMES)))
print("train", len(list((data / "images/train").glob("*.jpg"))), "· val", len(list((data / "images/val").glob("*.jpg"))))

if args.resume:
    YOLO(f"{args.project}/{args.name}/weights/last.pt").train(resume=True)
else:
    YOLO(args.model).train(
        data=str(yaml), imgsz=args.imgsz, epochs=args.epochs, patience=args.patience,
        batch=args.batch, device=args.device, workers=args.workers, cache=args.cache or False,
        project=args.project, name=args.name, exist_ok=True, save_period=5,
        **({} if args.plain_aug else dict(hsv_v=0.5, hsv_s=0.6, degrees=5.0, scale=0.6)),
    )

best = YOLO(f"{args.project}/{args.name}/weights/best.pt")
m = best.val(data=str(yaml), imgsz=args.imgsz, split="val", device=args.device)
for i, n in best.names.items():
    print(f"{n:10s} mask mAP50 {m.seg.ap50[i]:.3f}  mAP50-95 {m.seg.ap[i]:.3f}")
