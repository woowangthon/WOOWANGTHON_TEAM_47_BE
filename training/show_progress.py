"""학습 진행 상황을 표로 보여 준다 (서버에서 실행, 표준 라이브러리만 씀).

    python3 show_progress.py                      # ~/runs/scene5_v2
    python3 show_progress.py --name scene5_v1     # 다른 학습
    watch -n 30 python3 show_progress.py          # 30초마다 새로 고침 (Ctrl+C로 종료, 학습은 계속됨)

results.csv(에폭마다 한 줄)를 읽어 손실·mask mAP·남은 시간을 보여 준다.
"""

import argparse
import csv
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--project", default=str(Path("~/runs").expanduser()))
ap.add_argument("--name", default="scene5_v2")
ap.add_argument("--epochs", type=int, default=50, help="전체 에폭 (남은 시간 계산용)")
ap.add_argument("--last", type=int, default=15, help="표에 보여 줄 최근 에폭 수")
args = ap.parse_args()

path = Path(args.project) / args.name / "results.csv"
if not path.exists():
    raise SystemExit(f"아직 results.csv가 없음: {path}\n(첫 에폭이 끝나야 생김. 로그: tail -f ~/{args.name}.log)")

rows = [{k.strip(): v.strip() for k, v in r.items()} for r in csv.DictReader(open(path))]
if not rows:
    raise SystemExit("아직 끝난 에폭이 없음")


def f(r, key):
    try:
        return float(r.get(key, "nan"))
    except ValueError:
        return float("nan")


B, R, G, Y, D = "\033[1m", "\033[0m", "\033[32m", "\033[33m", "\033[2m"
best_i = max(range(len(rows)), key=lambda i: f(rows[i], "metrics/mAP50(M)"))
done = int(f(rows[-1], "epoch"))
bar_n = 30
fill = round(bar_n * done / args.epochs)

print(f"{B}{args.name}{R}  {G}{'█' * fill}{D}{'░' * (bar_n - fill)}{R}  {done}/{args.epochs} 에폭")
t = f(rows[-1], "time")
if t == t and done:  # time 칸이 있으면 남은 시간 계산
    per = t / done
    left = per * max(args.epochs - done, 0)
    print(f"경과 {t / 60:.0f}분 · 에폭당 {per:.0f}초 · 남은 시간 약 {left / 60:.0f}분")
print()
print(f"{B}{'에폭':>4}  {'box':>6} {'seg':>6} {'cls':>6}   {'mask P':>7} {'mask R':>7} {'mAP50':>7} {'mAP50-95':>9}{R}")
for i, r in list(enumerate(rows))[-args.last:]:
    mark = f" {Y}★ 최고{R}" if i == best_i else ""
    print(f"{int(f(r, 'epoch')):>4}  {f(r, 'train/box_loss'):6.3f} {f(r, 'train/seg_loss'):6.3f} {f(r, 'train/cls_loss'):6.3f}"
          f"   {f(r, 'metrics/precision(M)'):7.3f} {f(r, 'metrics/recall(M)'):7.3f} {f(r, 'metrics/mAP50(M)'):7.3f}"
          f" {f(r, 'metrics/mAP50-95(M)'):9.3f}{mark}")
b = rows[best_i]
print()
print(f"최고: {int(f(b, 'epoch'))}에폭 · mask mAP50 {f(b, 'metrics/mAP50(M)'):.3f} · mAP50-95 {f(b, 'metrics/mAP50-95(M)'):.3f}"
      f"  {D}(5클래스 평균. 클래스별 점수는 학습이 끝나면 로그 맨 끝에 나옴){R}")
