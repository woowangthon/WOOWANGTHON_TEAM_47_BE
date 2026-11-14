"""기존 outputs/masks/label_mask.npy → 구조 분석 → M3 후보 판정·시각화.

python run_m3_debug.py --mask outputs/masks/label_mask.npy --image test.jpg
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from pipeline.schema import PhotoKind, Verdict
from pipeline.skeleton import extract_path
from pipeline.shape_rules import analyze_m3, find_m3_candidates


def _save_png(path: Path, image: np.ndarray):
    ok, buf = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError(f"이미지 저장 실패: {path}")
    buf.tofile(str(path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask", default="outputs/masks/label_mask.npy")
    parser.add_argument("--image", required=True)
    parser.add_argument("--out", default="outputs/m3")
    args = parser.parse_args()
    label = np.load(args.mask, allow_pickle=False)
    image = cv2.imdecode(np.fromfile(args.image, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"사진을 읽을 수 없습니다: {args.image}")
    if label.shape != image.shape[:2]:
        raise ValueError(f"사진 크기 {image.shape[:2]}, 마스크 {label.shape} 불일치")

    path = extract_path(label, PhotoKind.CLOSE)
    candidates = find_m3_candidates(label, path)
    findings = analyze_m3(label, path)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    overlay = image.copy()
    colors = {Verdict.NORMAL: (40, 190, 40), Verdict.SUSPECTED: (20, 20, 245), Verdict.HOLD: (0, 180, 255)}
    for candidate in candidates:
        pt = (round(candidate.location.x), round(candidate.location.y))
        c = colors[candidate.verdict]
        cv2.circle(overlay, pt, 32, c, 5)
        cv2.putText(overlay, f"M3 {candidate.verdict.value}: {candidate.kind}",
                    (max(0, pt[0] - 200), max(30, pt[1] - 42)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, c, 3, cv2.LINE_AA)
    _save_png(out / "m3_overlay.png", overlay)
    (out / "m3_findings.json").write_text(
        json.dumps({"findings": [v.model_dump(mode="json") for v in findings],
                    "candidates": [vars(c) | {"location": {"x": c.location.x, "y": c.location.y},
                                              "verdict": c.verdict.value} for c in candidates]},
                   ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"검증된 M3 후보: {len(candidates)}개")
    for f in findings:
        print(f"  [{f.verdict.value}] {f.message}")
    print(f"결과 위치: {out.resolve()}")


if __name__ == "__main__":
    main()
