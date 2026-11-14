"""Path 그래프의 점형 연결 여부와 M3 후보 탈락 원인을 점검.

프로젝트 루트에서:
python -m scripts.run_m3_diagnose --mask outputs/masks/label_mask.npy --path outputs/path/path_structure.json
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np

from pipeline.schema import MaskClass, PathStructure
from pipeline.shape_rules import M3Config, _angle_degrees, _arm_outward_points, _linear_arm_from_dot, _vector_and_length, find_m3_candidates


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mask', default='outputs/masks/label_mask.npy')
    ap.add_argument('--path', default='outputs/path/path_structure.json')
    args = ap.parse_args()
    label = np.load(args.mask, allow_pickle=False)
    path = PathStructure.model_validate(json.loads(Path(args.path).read_text(encoding='utf-8')))
    assert label.shape == (path.height, path.width), 'path와 mask 크기가 다릅니다. Path를 다시 실행하세요.'
    cfg = M3Config()
    print('Path nodes=', len(path.nodes), 'edges=', len(path.edges), 'DOT nodes=', sum(n.block == MaskClass.DOT for n in path.nodes))
    for node in path.nodes:
        related = path.edges_of(node.id)
        if len(related) < 2:
            continue
        parts = []
        raw_parts = []
        for edge in related:
            coords = _arm_outward_points(path, node, edge.id)
            raw_parts.append(_vector_and_length(coords, cfg))
            parts.append(_linear_arm_from_dot(label, path, node, edge.id, cfg) if node.block == MaskClass.DOT else _vector_and_length(coords, cfg))
        print(f'node={node.id} type={node.type.value} block={node.block} degree={len(related)} x={node.pt.x:.0f} y={node.pt.y:.0f}')
        for edge, raw, corrected in zip(related, raw_parts, parts):
            print(f'  edge={edge.id}: raw_length={raw[1] if raw else None}; linear_length={corrected[1] if corrected else None}')
        if len(parts)==2 and all(p is not None for p in parts):
            deg = 180-_angle_degrees(parts[0][0],parts[1][0])
            print(f'  실제 선형 기준 꺾임 각도: {deg:.1f}°, threshold={cfg.min_turn_deg}°')
        elif len(parts)==2:
            print('  직선/꺾임 검증 불가: 충분히 긴 실제 선형 팔을 두 개 확보하지 못함')
    candidates = find_m3_candidates(label, path)
    print('M3 candidates=', len(candidates))
    for c in candidates:
        print(f'  {c.kind} {c.verdict.value} node={c.node_id} angle={c.direction_change_deg}°: {c.reason}')

if __name__ == '__main__':
    main()
