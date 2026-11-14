"""Standalone mask postprocessing/visualization: run from repo root.

python -m scripts.run_mask_refine --image datasets/test/test.jpg \\
  --mask outputs/masks/label_mask.npy --out outputs/mask_refined
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from pipeline.mask_refine import MaskRefineConfig, refine_mask
from pipeline.schema import MaskClass


COLORS_BGR = {
    int(MaskClass.LINEAR): (255, 90, 20),
    int(MaskClass.DOT): (20, 90, 250),
    int(MaskClass.CROSSWALK): (40, 210, 210),
    int(MaskClass.ROAD): (50, 220, 50),
}


def read_image(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f'Image missing: {path}')
    arr = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if arr is None:
        raise ValueError(f'Image decode failed: {path}')
    return arr


def read_label(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f'Mask missing: {path}')
    if path.suffix.lower() == '.npy':
        arr = np.load(path, allow_pickle=False)
    elif path.suffix.lower() == '.png':
        arr = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    else:
        raise ValueError('Mask must be .npy or .png')
    if arr is None or arr.dtype != np.uint8 or arr.ndim != 2:
        raise ValueError('Mask must be H x W uint8 single channel')
    return arr


def save_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, data = cv2.imencode('.png', image)
    if not ok:
        raise IOError(f'Could not encode image: {path}')
    data.tofile(str(path))


def color_overlay(image: np.ndarray, label: np.ndarray) -> np.ndarray:
    colors = image.copy()
    for k, color in COLORS_BGR.items():
        colors[label == k] = color
    out = cv2.addWeighted(image, 0.58, colors, 0.42, 0)
    out[label == 0] = image[label == 0]
    return out


def main() -> None:
    p = argparse.ArgumentParser(description='TactileCheck class-preserving mask cleanup')
    p.add_argument('--image', required=True, help='Same original photo used to infer this mask')
    p.add_argument('--mask', default='outputs/masks/label_mask.npy')
    p.add_argument('--out', default='outputs/mask_refined')
    p.add_argument('--min-area-px', type=int, default=80)
    p.add_argument('--min-largest-ratio', type=float, default=0.002)
    args = p.parse_args()
    img_path = Path(args.image)
    mask_path = Path(args.mask)
    image = read_image(img_path)
    raw = read_label(mask_path)
    if raw.shape != image.shape[:2]:
        raise ValueError(f'Image {image.shape[:2]} and mask {raw.shape} do not match; regenerate the mask!')

    # Detect stale-mask mismatch when segmentation.json records image source.
    info_path = mask_path.parent / 'segmentation.json'
    recorded_image = None
    if info_path.is_file():
        try:
            recorded_image = json.loads(info_path.read_text(encoding='utf-8')).get('input_image')
            if recorded_image and Path(recorded_image).as_posix().casefold() != img_path.as_posix().casefold():
                print('WARNING: input_image in segmentation.json differs from --image:', recorded_image)
                print('         Re-run mask extraction for this photo before trusting geometry.')
        except (ValueError, OSError):
            pass

    refined, audit = refine_mask(raw, MaskRefineConfig(
        min_area_px=args.min_area_px,
        min_largest_ratio=args.min_largest_ratio,
    ))
    audit['input_image'] = str(img_path)
    audit['input_mask'] = str(mask_path)
    audit['input_mask_sha256'] = hashlib.sha256(raw.tobytes()).hexdigest()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / 'refined_label_mask.npy', refined)
    save_png(out / 'refined_label_mask.png', refined)
    save_png(out / 'overlay_before.png', color_overlay(image, raw))
    save_png(out / 'overlay_after.png', color_overlay(image, refined))
    # Lightweight side-by-side preview (avoid huge 3K x 6K files)
    scale = min(1.0, 900 / max(image.shape[:2]))
    size = (max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale)))
    before = cv2.resize(color_overlay(image, raw), size)
    after = cv2.resize(color_overlay(image, refined), size)
    preview = np.concatenate([before, after], axis=1)
    cv2.putText(preview, 'BEFORE', (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (250,250,250), 2)
    cv2.putText(preview, 'AFTER', (size[0]+15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (250,250,250), 2)
    save_png(out / 'comparison.png', preview)
    (out / 'refine_report.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Output:', out.resolve())
    for name, s in audit['classes'].items():
        print(f"  {name}: {s['original_components']} -> {s['kept_components']} components, "
              f"{s['removed_px']} pixels removed (threshold {s['threshold_px']})")
    print('Changed pixels:', audit['pixels_changed'])
    print('NOTE: No morphology/gap closing; large false positives need model improvement.')


if __name__ == '__main__':
    main()
