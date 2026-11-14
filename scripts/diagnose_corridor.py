"""Diagnose why the one-dot straight-corridor refinement was not activated.

Run from repository root:
  python -m scripts.diagnose_corridor --mask outputs/inspections/RUN_ID/masks/refined_label_mask.png

Read-only: this script never modifies masks, weights or source files.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import inspect
import cv2
import numpy as np

from pipeline import skeleton as sk
from pipeline.schema import MaskClass


def _eligible_coverage(region: np.ndarray, axis: np.ndarray, origin: np.ndarray, coords: np.ndarray) -> float:
    """Reproduce the 97% centerline-inside-region check from the patched skeleton."""
    perp = np.array([-axis[1], axis[0]])
    delta = coords - origin
    along = delta @ axis
    sideways = delta @ perp
    step = 20.0
    lo = float(along.min())
    bins = np.floor((along - lo) / step).astype(np.int32)
    counts = np.bincount(bins)
    sums = np.bincount(bins, weights=sideways)
    valid = counts > 10
    if valid.sum() < 5:
        return -1.0
    ids = np.flatnonzero(valid)
    lateral = sums[ids] / counts[ids]
    if len(lateral) >= 5:
        lateral = cv2.GaussianBlur(lateral.reshape(-1, 1).astype(np.float32), (1, 11), sigmaX=0).ravel()
    longitudinal = lo + (ids.astype(float) + 0.5) * step
    centers = np.rint(origin + longitudinal[:, None] * axis + lateral[:, None] * perp).astype(np.int32)
    line = np.zeros(region.shape, dtype=np.uint8)
    for i in range(len(centers) - 1):
        if ids[i + 1] - ids[i] == 1:
            cv2.line(line, tuple(centers[i]), tuple(centers[i + 1]), 1, 1)
    pts = np.argwhere(line != 0)
    if len(pts) == 0:
        return -1.0
    return float(np.mean(region[pts[:, 0], pts[:, 1]] != 0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mask', required=True, type=Path, help='refined_label_mask.png of the v5 run')
    args = parser.parse_args()

    print('\n[1] 실제 로드된 skeleton.py')
    print(Path(inspect.getfile(sk)).resolve())
    cfg = sk.SkeletonConfig()
    patched = hasattr(cfg, 'one_dot_corridor_eigen_ratio')
    print('one-dot 패치 적용:', patched)
    if not patched:
        print('=> 패치가 실제 프로젝트의 pipeline/skeleton.py에 반영되지 않았습니다.')
        return
    print('one-dot 기준: PCA >=', cfg.one_dot_corridor_eigen_ratio,
          ', solidity >=', cfg.one_dot_corridor_solidity)

    path = args.mask.resolve()
    print('\n[2] 마스크:', path)
    if not path.is_file():
        print('ERROR: 마스크 파일이 없습니다. v5 결과 폴더명을 확인하세요.')
        return
    raw = np.fromfile(str(path), dtype=np.uint8)
    label = cv2.imdecode(raw, cv2.IMREAD_UNCHANGED)
    if label is None:
        print('ERROR: 마스크를 읽지 못했습니다.')
        return
    if label.ndim == 3:
        if not np.array_equal(label[..., 0], label[..., 1]) or not np.array_equal(label[..., 0], label[..., 2]):
            print('ERROR: 이 파일은 컬러 오버레이입니다. 원본 label_mask.png를 지정하세요.')
            return
        label = label[..., 0]
    if label.dtype != np.uint8:
        print('ERROR: uint8 마스크가 아닙니다:', label.dtype)
        return
    values, counts = np.unique(label, return_counts=True)
    print('shape:', label.shape, 'class counts:', dict(zip(values.tolist(), counts.tolist())))
    if not set(values.tolist()).issubset({0, 1, 2, 3, 4}):
        print('ERROR: 클래스 ID 0-4 마스크가 아닙니다.')
        return

    linear = sk.prepare_linear_mask(label, cfg)
    dot = sk._keep_large_components((label == int(MaskClass.DOT)).astype(np.uint8), cfg.min_dot_area_px)
    n_line, line_cc, stats, centers = cv2.connectedComponentsWithStats(linear, connectivity=8)
    n_dot, dot_cc = cv2.connectedComponents(dot, connectivity=8)
    print('\n[3] 연결 요소: LINEAR', n_line - 1, '/ DOT', n_dot - 1)
    if n_line <= 1 or n_dot <= 1:
        print('LINEAR 또는 DOT가 없어서 보정할 수 없습니다.')
        return
    gap_map = cv2.distanceTransform((dot == 0).astype(np.uint8), cv2.DIST_L2, 3)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * cfg.dot_attach_px + 1,) * 2)

    for li in range(1, n_line):
        area = int(stats[li, cv2.CC_STAT_AREA])
        if area < 1500:
            print(f'LINEAR#{li}: area={area}, skip: 면적<1500')
            continue
        region = (line_cc == li).astype(np.uint8)
        x,y,w,h = (int(x) for x in stats[li, :4])
        gap = float(gap_map[region != 0].min())
        n_touch = 0
        for di in range(1, n_dot):
            expanded = cv2.dilate((dot_cc == di).astype(np.uint8), kernel)
            if np.any((region != 0) & (expanded != 0)):
                n_touch += 1
        contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnt = max(contours, key=cv2.contourArea)
        hull_area = cv2.contourArea(cv2.convexHull(cnt))
        solidity = float(cv2.contourArea(cnt) / hull_area) if hull_area else 0.
        ys, xs = np.nonzero(region)
        coords = np.column_stack((xs, ys)).astype(np.float64)
        origin = coords.mean(axis=0)
        eigenvals, eigenvecs = np.linalg.eigh(np.cov(coords.T))
        ratio = float(eigenvals[1] / eigenvals[0]) if eigenvals[0] > 0 else 0.
        required_solidity = cfg.one_dot_corridor_solidity if n_touch == 1 else cfg.corridor_solidity
        required_ratio = cfg.one_dot_corridor_eigen_ratio if n_touch == 1 else cfg.corridor_min_eigen_ratio
        coverage = _eligible_coverage(region, eigenvecs[:, 1], origin, coords) if n_touch else None
        print(f'\nLINEAR#{li}: bbox=(x{x},y{y},w{w},h{h}), area={area}, center=({centers[li,0]:.0f},{centers[li,1]:.0f})')
        print(f'  점형 연결 수={n_touch}, 최소 간격≈{gap:.1f}px (허용 {cfg.dot_attach_px}px)')
        print(f'  convex solidity={solidity:.4f} (요구 {required_solidity:.2f})')
        print(f'  PCA ratio={ratio:.3f} (요구 {required_ratio:.2f})')
        if coverage is not None:
            print(f'  중심선 내부 포함률={coverage:.4f} (요구 0.97)')
        reasons = []
        if n_touch == 0: reasons.append('점형 접속 0개')
        if solidity < required_solidity: reasons.append('solidity 부족')
        if ratio < required_ratio: reasons.append('PCA ratio 부족')
        if coverage is not None and coverage < 0.97: reasons.append('중심선 포함률 부족')
        print('  판정:', ', '.join(reasons) if reasons else '조건 만족(추가로 원본과 교체 중심선 비교 필요)')

    print('\n[4] 실제 패치 함수 결과')
    n, comp = cv2.connectedComponents(linear, connectivity=8)
    new_skel, changed = sk._refine_simple_corridors(sk.skeletonize(linear.astype(bool)), linear, dot, comp, cfg)
    print('refined_components:', sorted(changed))
    print('=> []라면 조건 때문에 모든 LINEAR가 건너뛰어진 것입니다.')


if __name__ == '__main__':
    main()
