"""TactileCheck: conservative connected-component cleanup for 0/1/2/3/4 masks.

The original label array is not changed. Only LINEAR (1) and DOT (2) can be
filtered. This step never bridges gaps, changes class IDs, or makes legality
claims. Large false positives need better YOLO predictions, not cleanup.

Public contract:
    refined, stats = refine_mask(label, MaskRefineConfig())
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import cv2
import numpy as np

from .schema import MaskClass


@dataclass(frozen=True)
class MaskRefineConfig:
    # Minimum area uses the larger of absolute area
    # and the fraction of the largest component, per class.
    min_area_px: int = 80
    min_largest_ratio: float = 0.002
    # Do not fill holes or close disconnected paths: M3/M4 depend on topology.
    # When a class has no foreground, it is left empty.

    def __post_init__(self) -> None:
        if self.min_area_px < 0:
            raise ValueError('min_area_px must be >= 0')
        if not (0.0 <= self.min_largest_ratio < 1.0):
            raise ValueError('min_largest_ratio must be in [0,1)')


def _validate(label: np.ndarray) -> None:
    if not isinstance(label, np.ndarray) or label.ndim != 2 or label.dtype != np.uint8:
        raise ValueError('label must be a 2-D uint8 array')
    valid = np.array([int(x) for x in MaskClass], dtype=np.uint8)
    if not np.all(np.isin(label, valid)):
        present = np.unique(label).tolist()
        raise ValueError(f'Unknown class ID in label mask: {present}')


def refine_mask(
    label: np.ndarray, cfg: MaskRefineConfig | None = None
) -> tuple[np.ndarray, dict[str, Any]]:
    """Remove very small isolated predictions without changing connection topology.

    For each tactile class, threshold=max(min_area_px, largest_area*ratio).
    Connected components use 8-way connectivity; each removed component is
    replaced with background only. Crosswalk/road labels are never touched.

    Returns a NEW uint8 array and a JSON-serializable component audit.
    """
    cfg = cfg or MaskRefineConfig()
    _validate(label)
    out = label.copy()
    class_stats: dict[str, dict[str, Any]] = {}
    for cls in (MaskClass.LINEAR, MaskClass.DOT):
        binary = (label == int(cls)).astype(np.uint8)
        count, cc, sizes, _centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        areas = sizes[1:, cv2.CC_STAT_AREA] if count > 1 else np.empty(0, dtype=np.int32)
        largest = int(areas.max()) if areas.size else 0
        threshold = max(cfg.min_area_px, int(np.ceil(largest * cfg.min_largest_ratio)))
        removed: list[dict[str, int]] = []
        keep = np.zeros(count, dtype=np.uint8)
        if count > 1:
            keep[1:] = (areas >= threshold).astype(np.uint8)
            removed_idx = np.flatnonzero(keep[1:] == 0) + 1
            for i in removed_idx:
                removed.append({
                    'area_px': int(sizes[i, cv2.CC_STAT_AREA]),
                    'x': int(sizes[i, cv2.CC_STAT_LEFT]),
                    'y': int(sizes[i, cv2.CC_STAT_TOP]),
                    'w': int(sizes[i, cv2.CC_STAT_WIDTH]),
                    'h': int(sizes[i, cv2.CC_STAT_HEIGHT]),
                })
            # cc==0 means background of this CLASS; cannot overwrite other labels.
            out[(binary != 0) & (keep[cc] == 0)] = int(MaskClass.BACKGROUND)
        class_stats[cls.name.lower()] = {
            'original_components': int(max(0, count - 1)),
            'kept_components': int(np.count_nonzero(keep[1:])),
            'removed_components': int(len(removed)),
            'removed_px': int(sum(i['area_px'] for i in removed)),
            'largest_area_px': largest,
            'threshold_px': threshold,
            'removed_regions': removed,
        }
    info = {
        'image_shape_hw': list(label.shape),
        'config': asdict(cfg),
        'classes': class_stats,
        'pixels_changed': int(np.count_nonzero(label != out)),
        'note': (
            'Geometry-only conservative cleanup; not a classifier. '
            'Large, confidently predicted false positives can remain; '
            'no gap closing/merging/filling is performed.'
        ),
    }
    return out, info
