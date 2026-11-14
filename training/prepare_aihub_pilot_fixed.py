#!/usr/bin/env python3
"""Prepare a CVAT COCO draft using EXIF-normalized AI Hub images and polygons.

Important: original JPG pixels may be rotated, with EXIF Orientation specifying
how to display them. This program FIRST normalizes them to upright orientation,
then rescales the JSON polygons to that coordinate system. It refuses images
with mismatched aspect ratios instead of silently corrupting labels.
Only block-region labels 34/35 are imported (damage-only label 36 is excluded).
All imported polygons have a temporary single class 'tactile'. In CVAT,
relabel/split the shapes into 'linear' and 'dot' BEFORE YOLO training.

Usage (PowerShell):
 py -m pip install pillow
 py prepare_aihub_pilot.py --images datasets/aihub_raw/images --json datasets/aihub_raw/json --out datasets/aihub_pilot --limit 10
"""

from __future__ import annotations
import argparse
import csv
import json
import re
from pathlib import Path
from PIL import Image, ImageDraw, ImageOps

IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
NAME_ALIASES = ('점자블럭', '점자블록')


def normalize_stem(stem: str) -> str:
    return re.sub(r'\(\d+\)$', '', stem)


def extract_polygons(data):
    """Return polygons in the AI Hub annotation_info=[[x,y],...] format."""
    if not isinstance(data, list) or not data:
        return []
    if len(data) >= 3 and all(
        isinstance(p, (list, tuple)) and len(p) >= 2
        and isinstance(p[0], (int, float)) and isinstance(p[1], (int, float))
        for p in data
    ):
        return [[(float(p[0]), float(p[1])) for p in data]]
    result = []
    for part in data:
        result += extract_polygons(part)
    return result


def area(pts):
    return abs(sum(x * pts[(i+1)%len(pts)][1] - pts[(i+1)%len(pts)][0] * y
                   for i, (x, y) in enumerate(pts))) / 2


def main():
    parser = argparse.ArgumentParser(description='AI Hub JSON -> COCO CVAT draft + overlay previews')
    parser.add_argument('--images', required=True, type=Path, help='Selected AI Hub JPGs')
    parser.add_argument('--json', required=True, type=Path, help='Corresponding AI Hub original JSON files')
    parser.add_argument('--out', default=Path('datasets/aihub_pilot'), type=Path)
    parser.add_argument('--limit', default=None, type=int, help='Limit valid matches (omit for all)')
    args = parser.parse_args()

    if not args.images.is_dir() or not args.json.is_dir():
        parser.error('Image or JSON folder not found: check --images and --json')
    if args.limit is not None and args.limit < 1:
        parser.error('--limit must be >= 1')
    image_files = sorted(p for p in args.images.rglob('*') if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    if not image_files:
        parser.error('No images found')

    json_files = list(args.json.rglob('*.json'))
    json_by_stem = {normalize_stem(p.stem): p for p in json_files}
    (args.out / 'images_for_cvat').mkdir(parents=True, exist_ok=True)
    (args.out / 'previews').mkdir(parents=True, exist_ok=True)

    coco = {
        'info': {'description': 'AI Hub temporary tactile polygon draft; split into linear / dot', 'version': '1.0'},
        'licenses': [],
        'categories': [{'id': 1, 'name': 'tactile', 'supercategory': 'tactile_paving'}],
        'images': [], 'annotations': []
    }
    reports = []
    img_id = ann_id = 1
    used = set()
    for img_path in image_files:
        stem = normalize_stem(img_path.stem)
        if stem in used:
            reports.append({'image': img_path.name, 'status': 'DUPLICATE_STEM', 'json': '', 'actual_size': '', 'source_size': '', 'polygons': 0, 'scale_x': '', 'scale_y': ''})
            continue
        label_path = json_by_stem.get(stem)
        if not label_path:
            reports.append({'image': img_path.name, 'status': 'NO_JSON', 'json': '', 'actual_size': '', 'source_size': '', 'polygons': 0, 'scale_x': '', 'scale_y': ''})
            continue
        try:
            data = json.loads(label_path.read_text(encoding='utf-8-sig'))
            with Image.open(img_path) as im:
                # Many portrait AI Hub JPGs store pixels sideways and rely on
                # EXIF Orientation for upright display in Windows / photo viewers.
                # PIL.convert() and PIL.size alone do NOT apply that orientation.
                source_exif_orientation = im.getexif().get(274, 1)
                upright = ImageOps.exif_transpose(im)
                actual_w, actual_h = upright.size
                base_img = upright.convert('RGB')
            info = data.get('info', {})
            source_w, source_h = int(info['width']), int(info['height'])
            if source_w <= 0 or source_h <= 0:
                raise ValueError('Invalid JSON dimensions')
            expected_ratio = source_w / source_h
            actual_ratio = actual_w / actual_h
            # Reject swapped/cropped images instead of drawing dangerous false labels.
            if abs(actual_ratio / expected_ratio - 1) > 0.035:
                raise ValueError(
                    f'ORIENTATION_OR_ASPECT_MISMATCH: JSON={source_w}x{source_h}, '
                    f'EXIF-corrected image={actual_w}x{actual_h}, '
                    f'EXIF Orientation={source_exif_orientation}. '
                    'If the photo is sideways without EXIF, fix the image orientation first.'
                )
            filename = info.get('filename')
            if filename and normalize_stem(Path(filename).stem) != stem:
                raise ValueError('JSON info.filename does not match image filename')
            scale_x, scale_y = actual_w / source_w, actual_h / source_h
            anns = []
            for item in data.get('annotations', []):
                if (str(item.get('label_id')) not in {'34', '35'}
                        or item.get('label_name') not in NAME_ALIASES
                        or item.get('annotation_type') != 'polygon'):
                    continue
                for source_pts in extract_polygons(item.get('annotation_info', [])):
                    pts = [(max(0.0, min(float(actual_w), x * scale_x)),
                            max(0.0, min(float(actual_h), y * scale_y))) for x, y in source_pts]
                    if len(pts) >= 3 and area(pts) >= 1.0:
                        anns.append(pts)
            if not anns:
                reports.append({'image': img_path.name, 'status': 'NO_VALID_POLYGONS', 'json': str(label_path), 'actual_size': f'{actual_w}x{actual_h}', 'source_size': f'{source_w}x{source_h}', 'polygons': 0, 'scale_x': scale_x, 'scale_y': scale_y})
                continue
            # Save actual pixels in upright orientation without the old EXIF
            # Orientation tag. CVAT, COCO coordinates, and thumbnails now all
            # use the exact same pixel coordinate system.
            out_image = args.out / 'images_for_cvat' / img_path.name
            if img_path.suffix.lower() in ('.jpg', '.jpeg'):
                base_img.save(out_image, format='JPEG', quality=95, subsampling=0)
            elif img_path.suffix.lower() == '.png':
                base_img.save(out_image, format='PNG')
            elif img_path.suffix.lower() == '.webp':
                base_img.save(out_image, format='WEBP', quality=95)
            elif img_path.suffix.lower() == '.bmp':
                base_img.save(out_image, format='BMP')
            coco['images'].append({'id': img_id, 'file_name': img_path.name, 'width': actual_w, 'height': actual_h})
            for pts in anns:
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                coco['annotations'].append({
                    'id': ann_id, 'image_id': img_id, 'category_id': 1,
                    'segmentation': [[round(v, 4) for pt in pts for v in pt]],
                    'bbox': [round(min(xs), 4), round(min(ys), 4), round(max(xs)-min(xs), 4), round(max(ys)-min(ys), 4)],
                    'area': round(area(pts), 4), 'iscrowd': 0
                })
                ann_id += 1
            # Preview only; do not use resized previews for training.
            overlay = Image.new('RGBA', base_img.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)
            width = max(3, round(actual_w / 220))
            for poly in anns:
                draw.polygon(poly, fill=(0, 195, 255, 64), outline=(220, 30, 40, 245), width=width)
            preview = Image.alpha_composite(base_img.convert('RGBA'), overlay).convert('RGB')
            preview.thumbnail((900, 1200))
            preview.save(args.out / 'previews' / f'{img_path.stem}_overlay.jpg', quality=85)
            reports.append({'image': img_path.name, 'status': 'OK', 'json': str(label_path), 'actual_size': f'{actual_w}x{actual_h}', 'source_size': f'{source_w}x{source_h}', 'polygons': len(anns), 'scale_x': round(scale_x, 6), 'scale_y': round(scale_y, 6)})
            used.add(stem)
            img_id += 1
            if args.limit is not None and len(coco['images']) >= args.limit:
                break
        except (ValueError, KeyError, TypeError, OSError, json.JSONDecodeError) as e:
            reports.append({'image': img_path.name, 'status': f'ERROR: {e}', 'json': str(label_path), 'actual_size': '', 'source_size': '', 'polygons': 0, 'scale_x': '', 'scale_y': ''})

    if coco['images']:
        (args.out / 'instances_cvat.json').write_text(json.dumps(coco, ensure_ascii=False, indent=2), encoding='utf-8')
    with (args.out / 'review_report.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=['image', 'status', 'json', 'actual_size', 'source_size', 'polygons', 'scale_x', 'scale_y'])
        writer.writeheader()
        writer.writerows(reports)

    print('Finished:', args.out.resolve())
    print('Matched images:', len(coco['images']))
    print('Imported tactile polygons:', len(coco['annotations']))
    print('Unmatched/errors:', sum(row['status'] != 'OK' for row in reports))
    if coco['images']:
        print('1. Check previews/ for alignment, and inspect review_report.csv')
        print('2. CVAT: upload images_for_cvat/ and import instances_cvat.json')
        print('3. CVAT: replace/split tactile polygons into linear or dot before YOLO export')
    else:
        print('No matched images. Inspect review_report.csv and original filenames.')


if __name__ == '__main__':
    main()
