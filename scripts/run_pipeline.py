"""Run from project root: python -m scripts.run_pipeline --image ... --weights ..."""
from __future__ import annotations

import argparse
from pathlib import Path

from pipeline.inspection_pipeline import PipelineConfig, run_inspection


def main() -> None:
    parser = argparse.ArgumentParser(
        description='TactileCheck close photo: segmentation -> mask refinement -> path -> M3'
    )
    parser.add_argument('--image', required=True, help='원본 사진 경로')
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--weights', help='YOLO 점형·선형(0=linear, 1=dot) 모델 .pt 경로')
    source.add_argument('--mask', help='기존 사진에서 생성된 .npy/.png 마스크 (YOLO 없이 테스트)')
    parser.add_argument('--out-root', default='outputs/inspections', help='실행별 결과 폴더의 상위 위치')
    parser.add_argument('--conf', type=float, default=0.25)
    parser.add_argument('--imgsz', type=int, default=1024)
    parser.add_argument('--min-area-px', type=int, default=80)
    parser.add_argument('--min-largest-ratio', type=float, default=0.002)
    args = parser.parse_args()

    run_dir, report = run_inspection(
        Path(args.image),
        weights=Path(args.weights) if args.weights else None,
        mask=Path(args.mask) if args.mask else None,
        out_root=Path(args.out_root),
        config=PipelineConfig(
            conf=args.conf, imgsz=args.imgsz,
            min_area_px=args.min_area_px,
            min_largest_ratio=args.min_largest_ratio,
        ),
    )
    print('\n=== TactileCheck 통합 분석 완료 ===')
    print('결과 폴더:', run_dir.resolve())
    print('마스크 후처리 변경 픽셀:', report['refine_changed_px'])
    print('Path 노드/엣지:', report['path_nodes'], '/', report['path_edges'])
    print('M3 후보:', report['m3_candidate_count'])
    print('M3 판정:', ', '.join(report['m3_verdicts']))
    print('전체 판정:', report['overall'], '(M3만 검사하므로 미검사 규칙은 보류)')
    print('확인 파일:', run_dir / 'result.json')
    if args.mask:
        print('주의: --mask 사용 시 동일 사진에서 생성된 마스크인지 직접 확인해야 합니다.')


if __name__ == '__main__':
    main()
