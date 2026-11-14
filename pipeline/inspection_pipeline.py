"""TactileCheck: one-shot CLOSE-photo segmentation -> refinement -> path -> M3.

Additive module; preserves pipeline/schema.py and existing debug tools. This is
NOT a legal installation-compliance checker. M3 normal means only that a dot
was detected at a candidate; unimplemented rules keep overall at HOLD.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import uuid
from typing import Any
from pydantic import BaseModel
from enum import Enum

import cv2
import numpy as np

from .mask_refine import MaskRefineConfig, refine_mask
from .hold_rules import check_h1_h4
from .schema import (
    Finding, InspectionResult, MaskClass, NodeType, PhotoAnalysis, PhotoKind,
    Segmentation, Verdict,
)
from .segment import Segmenter, summarize
from .shape_rules import analyze_m3, find_m3_candidates
from .skeleton import SkeletonConfig, extract_path


@dataclass(frozen=True)
class PipelineConfig:
    conf: float = 0.25
    imgsz: int = 1024
    min_area_px: int = 80
    min_largest_ratio: float = 0.002
    close_px: int = 0   # Do NOT bridge true physical gaps.


def read_image(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"이미지 파일이 없습니다: {path}")
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None or image.ndim != 3:
        raise ValueError(f"이미지를 읽을 수 없습니다: {path}")
    return image


def read_label(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"마스크 파일이 없습니다: {path}")
    if path.suffix.lower() == '.npy':
        label = np.load(path, allow_pickle=False)
    elif path.suffix.lower() == '.png':
        label = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    else:
        raise ValueError('마스크 파일은 .npy 또는 .png여야 합니다')
    if not isinstance(label, np.ndarray) or label.ndim != 2 or label.dtype != np.uint8:
        raise ValueError('마스크는 uint8 단일 채널 (H,W)이어야 합니다')
    unknown = set(np.unique(label).tolist()) - set(int(x) for x in MaskClass)
    if unknown:
        raise ValueError(f'마스크에 정의되지 않은 클래스가 있습니다: {sorted(unknown)}')
    return label


def save_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode('.png', image)
    if not ok:
        raise RuntimeError(f'PNG 저장 실패: {path}')
    encoded.tofile(str(path))  # Supports Korean Windows paths.


def _json_safe(obj):
    """Pydantic 모델과 NumPy 값을 JSON 저장 가능한 자료형으로 변환."""

    if isinstance(obj, BaseModel):
        return obj.model_dump(mode="json")

    if isinstance(obj, Enum):
        return obj.value

    if isinstance(obj, Path):
        return str(obj)

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, np.generic):
        return obj.item()

    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple, set)):
        return [_json_safe(v) for v in obj]

    return obj


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text(
        json.dumps(
            _json_safe(data),
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _overlay_mask(image: np.ndarray, label: np.ndarray) -> np.ndarray:
    colors = {
        MaskClass.LINEAR: (255, 80, 20),  # BGR blue
        MaskClass.DOT: (20, 90, 240),
        MaskClass.CROSSWALK: (40, 190, 190),
        MaskClass.ROAD: (55, 210, 45),
    }
    colored = image.copy()
    for cls, rgb in colors.items():
        colored[label == cls] = rgb
    out = cv2.addWeighted(image, 0.60, colored, 0.40, 0)
    out[label == MaskClass.BACKGROUND] = image[label == MaskClass.BACKGROUND]
    return out


def _overlay_path(image: np.ndarray, label: np.ndarray, path) -> np.ndarray:
    out = _overlay_mask(image, label)
    node_colors = {
        NodeType.END: (35, 220, 35),
        NodeType.BRANCH: (50, 50, 250),
        NodeType.TURN: (0, 180, 250),
        NodeType.PASS: (255, 240, 0),
        NodeType.ISOLATED: (255, 0, 255),
    }
    for edge in path.edges:
        pts = np.asarray([(round(p.x), round(p.y)) for p in edge.polyline], dtype=np.int32)
        if len(pts) >= 2:
            cv2.polylines(out, [pts.reshape(-1, 1, 2)], False, (255, 255, 255), 3)
    for node in path.nodes:
        pt = (round(node.pt.x), round(node.pt.y))
        clr = node_colors[node.type]
        cv2.circle(out, pt, 7, clr, -1)
        cv2.circle(out, pt, 7, (10, 10, 10), 1)
        cv2.putText(out, str(node.id), (pt[0] + 9, pt[1] - 9),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, clr, 1, cv2.LINE_AA)
    return out


def _overlay_m3(image: np.ndarray, candidates) -> np.ndarray:
    out = image.copy()
    colors = {
        Verdict.NORMAL: (40, 190, 40),
        Verdict.SUSPECTED: (20, 20, 245),
        Verdict.HOLD: (0, 180, 255),
    }
    for c in candidates:
        pt = (round(c.location.x), round(c.location.y))
        color = colors[c.verdict]
        cv2.circle(out, pt, 30, color, 5)
        cv2.putText(out, f'M3 {c.verdict.value}: {c.kind}',
                    (max(0, pt[0]-180), max(30, pt[1]-40)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
    return out


def _validate_names(model_names: dict | list | tuple) -> None:
    """Model index 0/1 MUST represent linear/dot, else mask IDs are invalid."""
    if isinstance(model_names, dict):
        values = [model_names.get(i, model_names.get(str(i))) for i in (0, 1)]
    else:
        values = list(model_names)[:2]
    names = [str(x).strip().lower() for x in values]
    if names != ['linear', 'dot']:
        raise ValueError(
            f'모델 클래스 순서가 [linear, dot]가 아닙니다: {model_names}. '
            '클래스를 확인한 뒤 분석하세요.'
        )


def _candidate_to_json(c) -> dict[str, Any]:
    item = asdict(c)
    item['verdict'] = c.verdict.value
    return item


def _run_steps(
    image_path: Path,
    run_dir: Path,
    *,
    weights: Path | None,
    input_mask: Path | None,
    cfg: PipelineConfig,
    segmenter: object | None,
) -> dict[str, Any]:
    image = read_image(image_path)
    photo = PhotoKind.CLOSE

    if input_mask is not None:
        raw_label = read_label(input_mask)
        raw_summary = summarize(raw_label, photo)
        source_type = 'existing_mask'
        model_file = None
    else:
        if weights is None and segmenter is None:
            raise ValueError('YOLO 추론에는 --weights가 필요합니다. 또는 --mask를 지정하세요.')
        if segmenter is None:
            if not weights.is_file():
                raise FileNotFoundError(f'가중치 파일이 없습니다: {weights}')
            segmenter = Segmenter(weights=weights, conf=cfg.conf, imgsz=cfg.imgsz)
        if hasattr(segmenter, 'model') and hasattr(segmenter.model, 'names'):
            _validate_names(segmenter.model.names)
        raw_label, raw_summary = segmenter.predict(image, photo)
        source_type = 'yolo'
        model_file = str(weights.resolve()) if weights else '(injected test segmenter)'

    if raw_label.ndim != 2 or raw_label.dtype != np.uint8:
        raise ValueError('분할 결과는 uint8 단일 채널이어야 합니다')
    if raw_label.shape != image.shape[:2]:
        raise ValueError(
            f'원본/마스크 크기 불일치: image={image.shape[:2]}, mask={raw_label.shape}. '
            '새 사진에서 마스크를 다시 추출해야 합니다.'
        )
    unknown = set(np.unique(raw_label).tolist()) - set(int(x) for x in MaskClass)
    if unknown:
        raise ValueError(f'마스크 클래스 값 오류: {sorted(unknown)}')

    mask_dir = run_dir / 'masks'
    path_dir = run_dir / 'path'
    m3_dir = run_dir / 'm3'
    mask_dir.mkdir(parents=True, exist_ok=True)
    path_dir.mkdir(parents=True, exist_ok=True)
    m3_dir.mkdir(parents=True, exist_ok=True)

    np.save(mask_dir / 'label_mask.npy', raw_label)
    save_png(mask_dir / 'label_mask.png', raw_label)
    save_png(mask_dir / 'raw_overlay.png', _overlay_mask(image, raw_label))
    write_json(mask_dir / 'segmentation.json', {
        **raw_summary.model_dump(mode='json'),
        'input_image': str(image_path.resolve()),
        'input_image_sha256': sha256_file(image_path),
        'weights': model_file,
        'source_type': source_type,
    })

    refined, audit = refine_mask(raw_label, MaskRefineConfig(
        min_area_px=cfg.min_area_px, min_largest_ratio=cfg.min_largest_ratio,
    ))
    np.save(mask_dir / 'refined_label_mask.npy', refined)
    save_png(mask_dir / 'refined_label_mask.png', refined)
    save_png(mask_dir / 'refined_overlay.png', _overlay_mask(image, refined))
    write_json(mask_dir / 'refine_report.json', audit)

    path = extract_path(refined, photo, SkeletonConfig(close_px=cfg.close_px))
    save_png(path_dir / 'path_overlay.png', _overlay_path(image, refined, path))
    write_json(path_dir / 'path_structure.json', path.model_dump(mode='json'))

    candidates = find_m3_candidates(refined, path)
    findings: list[Finding] = analyze_m3(refined, path)
    save_png(m3_dir / 'm3_overlay.png', _overlay_m3(image, candidates))
    write_json(m3_dir / 'm3_findings.json', {
        'findings': [f.model_dump(mode='json') for f in findings],
        'candidates': [_candidate_to_json(c) for c in candidates],
    })

    refined_summary = Segmentation(
        photo=photo,
        width=image.shape[1],
        height=image.shape[0],
        mask_path='masks/refined_label_mask.png',
        class_area_px=summarize(refined, photo).class_area_px,
        mean_confidence=raw_summary.mean_confidence,  # model detection mean, not M3 rule confidence
    )
    # H1/H4 return independent hold candidates; never rewrite M3 findings.
    close_analysis = PhotoAnalysis(segmentation=refined_summary, path=path)
    hold_findings = check_h1_h4(close_analysis)
    findings = [*findings, *hold_findings]
    hold_dir = run_dir / 'hold'
    hold_dir.mkdir(parents=True, exist_ok=True)
    write_json(hold_dir / 'hold_findings.json', {
        'findings': [f.model_dump(mode='json') for f in hold_findings],
    })
    # A definite M3 suspicion is retained even if H1/H4 request recapture.
    # Never assert overall NORMAL until all required C/M rules have been assessed.
    overall = Verdict.SUSPECTED if any(f.verdict == Verdict.SUSPECTED for f in findings) else Verdict.HOLD
    result = InspectionResult(
        inspection_id=run_dir.name,
        overall=overall,
        findings=findings,
        close=close_analysis,
    )
    write_json(run_dir / 'result.json', result.model_dump(mode='json'))

    manifest = {
        'status': 'complete',
        'photo': photo.value,
        'input_image': str(image_path.resolve()),
        'input_sha256': sha256_file(image_path),
        'input_mask': str(input_mask.resolve()) if input_mask else None,
        'source_type': source_type,
        'weights': model_file,
        'model_conf_threshold': cfg.conf if source_type == 'yolo' else None,
        'image_shape_hw': list(image.shape[:2]),
        'refine_changed_px': audit['pixels_changed'],
        'path_nodes': len(path.nodes),
        'path_edges': len(path.edges),
        'm3_candidate_count': len(candidates),
        'm3_verdicts': [f.verdict.value for f in findings],
        'overall': overall.value,
        'coverage': ['M3', 'H1', 'H4'],
        'hold_verdicts': [f.code for f in hold_findings],
        'not_evaluated': ['M4', 'C1', 'C2', 'C3', 'C4', 'H2', 'H3'],
        'warning': (
            'This is an M3+H1/H4 heuristic. overall=hold means not fully evaluated; '
            'normal means dot coverage at a candidate, not legal compliance. '
            + ('--mask mode cannot prove the mask belongs to this photo.' if input_mask else '')
        ),
        'files': {
            'raw_mask': 'masks/label_mask.npy',
            'refined_mask': 'masks/refined_label_mask.npy',
            'mask_overlay': 'masks/refined_overlay.png',
            'path_overlay': 'path/path_overlay.png',
            'path_graph': 'path/path_structure.json',
            'm3_overlay': 'm3/m3_overlay.png',
            'm3_findings': 'm3/m3_findings.json',
            'hold_findings': 'hold/hold_findings.json',
            'result': 'result.json',
        }
    }
    write_json(run_dir / 'manifest.json', manifest)
    return manifest


def run_inspection(
    image_path: str | Path,
    *,
    weights: str | Path | None = None,
    mask: str | Path | None = None,
    out_root: str | Path = 'outputs/inspections',
    config: PipelineConfig | None = None,
    segmenter: object | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Run all steps in an isolated directory and publish it only on success.

    In --mask mode the mask MUST come from the same image; dimensions alone do
    not prove provenance. For normal operation prefer --weights.
    """
    img = Path(image_path)
    if not img.is_file():
        raise FileNotFoundError(f'사진이 없습니다: {img}')
    weights_path = Path(weights) if weights is not None else None
    mask_path = Path(mask) if mask is not None else None
    if weights_path is None and mask_path is None and segmenter is None:
        raise ValueError('--weights 또는 --mask 중 하나를 지정하세요.')
    if weights_path is not None and mask_path is not None:
        raise ValueError('--weights와 --mask는 동시에 지정하지 마세요.')

    out = Path(out_root)
    out.mkdir(parents=True, exist_ok=True)
    stem = re.sub(r'[^\w가-힣.-]+', '_', img.stem, flags=re.UNICODE).strip('._')[:40] or 'image'
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    uid = uuid.uuid4().hex[:8]
    name = f'{stem}_{stamp}_{uid}'
    scratch = out / f'.{name}.partial'
    final = out / name
    scratch.mkdir(parents=False, exist_ok=False)
    try:
        result = _run_steps(img, scratch, weights=weights_path, input_mask=mask_path,
                            cfg=config or PipelineConfig(), segmenter=segmenter)
        # Result inspection_id should use final run name instead of temporary folder name.
        res_path = scratch / 'result.json'
        res = json.loads(res_path.read_text(encoding='utf-8'))
        res['inspection_id'] = name
        write_json(res_path, res)
        scratch.rename(final)
        return final, result
    except Exception:
        shutil.rmtree(scratch, ignore_errors=True)
        raise
