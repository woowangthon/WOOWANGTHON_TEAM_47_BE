"""Offline integration tests: no Ultralytics weights or GPU needed."""
from pathlib import Path
import json

import cv2
import numpy as np
import pytest

from pipeline.inspection_pipeline import PipelineConfig, run_inspection
from pipeline.schema import PhotoKind
from pipeline.segment import summarize


class StubSegmenter:
    def __init__(self, label):
        self.label = label
        self.model = type('Model', (), {'names': {0: 'linear', 1: 'dot'}})()

    def predict(self, image, photo):
        return self.label.copy(), summarize(self.label, photo)


def _sample(tmp_path):
    image = np.zeros((340, 360, 3), dtype=np.uint8)
    image[:, :] = (80, 90, 100)
    label = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.rectangle(label, (135, 30), (185, 140), 1, -1)
    cv2.rectangle(label, (125, 140), (195, 215), 2, -1)
    cv2.rectangle(label, (135, 215), (185, 318), 1, -1)
    label[10:12, 10:12] = 1  # removable tiny speck
    imgpath = tmp_path / '테스트 사진.jpg'
    assert cv2.imencode('.jpg', image)[0]
    cv2.imencode('.jpg', image)[1].tofile(str(imgpath))
    return imgpath, label


def test_all_steps_segmenter_and_isolated_runs(tmp_path):
    imgpath, label = _sample(tmp_path)
    results = []
    for _ in range(2):
        output, manifest = run_inspection(
            imgpath, out_root=tmp_path / 'runs',
            segmenter=StubSegmenter(label), config=PipelineConfig(),
        )
        results.append(output)
        assert manifest['source_type'] == 'yolo'
        assert manifest['refine_changed_px'] == 4
        assert manifest['coverage'] == ['M3', 'H1', 'H4']
        assert manifest['overall'] in ('hold', 'suspected')
        assert (output / 'result.json').is_file()
        assert (output / 'masks/refined_label_mask.npy').is_file()
        assert (output / 'path/path_overlay.png').is_file()
        assert (output / 'path/path_structure.json').is_file()
        assert (output / 'm3/m3_overlay.png').is_file()
        assert (output / 'm3/m3_findings.json').is_file()
        assert (output / 'hold/hold_findings.json').is_file()
        result = json.loads((output / 'result.json').read_text(encoding='utf-8'))
        assert result['inspection_id'] == output.name
        assert result['close']['segmentation']['mask_path'] == 'masks/refined_label_mask.png'
        assert result['schema_version']
        assert result['overall'] != 'normal'  # M3-only must not certify entire scene.
        mask = np.load(output / 'masks/refined_label_mask.npy')
        assert mask[10:12, 10:12].sum() == 0
        assert mask[170, 155] == 2
    assert results[0] != results[1]
    assert all(x.exists() for x in results)


def test_mask_mode_and_mismatch_failure(tmp_path):
    imgpath, label = _sample(tmp_path)
    maskfile = tmp_path / 'mask.npy'
    np.save(maskfile, label)
    output, report = run_inspection(
        imgpath, mask=maskfile, out_root=tmp_path / 'runs'
    )
    assert report['source_type'] == 'existing_mask'
    assert report['weights'] is None
    assert output.exists()

    np.save(maskfile, np.zeros((50, 70), dtype=np.uint8))
    with pytest.raises(ValueError, match='크기 불일치'):
        run_inspection(imgpath, mask=maskfile, out_root=tmp_path / 'runs')
    assert not list((tmp_path / 'runs').glob('.*.partial'))
    assert len([x for x in (tmp_path / 'runs').iterdir() if x.is_dir()]) == 1


def test_reject_wrong_class_order(tmp_path):
    imgpath, label = _sample(tmp_path)
    stub = StubSegmenter(label)
    stub.model.names = {0: 'dot', 1: 'linear'}
    with pytest.raises(ValueError, match='클래스 순서'):
        run_inspection(imgpath, segmenter=stub, out_root=tmp_path / 'runs')
    assert not list((tmp_path / 'runs').iterdir())


def test_requires_weights_or_mask(tmp_path):
    imgpath, label = _sample(tmp_path)
    with pytest.raises(ValueError, match='--weights 또는 --mask'):
        run_inspection(imgpath, out_root=tmp_path / 'runs')


def test_missing_tactile_blocks_returns_h1_hold(tmp_path):
    imgpath, _ = _sample(tmp_path)
    blank = np.zeros((340, 360), dtype=np.uint8)
    output, manifest = run_inspection(imgpath, segmenter=StubSegmenter(blank), out_root=tmp_path / 'runs')
    result = json.loads((output / 'result.json').read_text(encoding='utf-8'))
    assert result['overall'] == 'hold'
    assert any(f['code'] == 'H1' and f['verdict'] == 'hold' for f in result['findings'])
    assert 'H1' in manifest['hold_verdicts']


def test_low_confidence_returns_h1_hold(tmp_path):
    imgpath, label = _sample(tmp_path)
    stub = StubSegmenter(label)
    original_predict = stub.predict
    def predict_low_conf(image, photo):
        mask, summary = original_predict(image, photo)
        summary.mean_confidence = 0.2
        return mask, summary
    stub.predict = predict_low_conf
    output, _ = run_inspection(imgpath, segmenter=stub, out_root=tmp_path / 'runs')
    result = json.loads((output / 'result.json').read_text(encoding='utf-8'))
    assert any(f['code'] == 'H1' for f in result['findings'])
