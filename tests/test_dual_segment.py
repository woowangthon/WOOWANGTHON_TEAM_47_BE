"""두 모델 통합(merge_detections) 테스트. 진짜 YOLO 대신 '모델이 이렇게 답했다'는 검출 결과를 직접 만들어 넣는다."""

import numpy as np
import pytest

from pipeline.dual_segment import class_mapping, merge_detections
from pipeline.schema import MaskClass, PhotoKind

H, W = 600, 800


def _image(striped_top: bool):
    """위쪽 절반: 흰 줄무늬 횡단보도(또는 민짜 아스팔트), 아래쪽: 인도."""
    rng = np.random.default_rng(0)
    img = np.full((H, W, 3), 120, np.uint8) + rng.integers(0, 10, (H, W, 1), dtype=np.uint8)
    img[:300] = 70
    if striped_top:
        for y in range(10, 300, 60):
            img[y:y + 30] = 225
    return img


def _rect(y0, y1, x0, x1):
    m = np.zeros((H, W), bool)
    m[y0:y1, x0:x1] = True
    return m


CROSSWALK = (_rect(0, 300, 0, W), MaskClass.CROSSWALK, 0.9)
PATH = (_rect(300, 600, 370, 430), MaskClass.LINEAR, 0.8)  # 인도 위 선형 블록
PAINT = (_rect(100, 250, 100, 140), MaskClass.LINEAR, 0.7)  # 횡단보도 안 노란 칠 오인식
CURB = (_rect(260, 340, 500, 560), MaskClass.DOT, 0.75)  # 연석에 걸친 점형 (절반만 횡단보도와 겹침)


def test_classes_mapped_and_blocks_painted_on_scene():
    label, seg = merge_detections(_image(True), [CROSSWALK], [PATH], PhotoKind.WIDE)
    assert label[150, 50] == MaskClass.CROSSWALK
    assert label[450, 400] == MaskClass.LINEAR
    assert seg.class_area_px["crosswalk"] > 0 and seg.class_area_px["linear"] > 0
    assert seg.mean_confidence == pytest.approx(0.8)


def test_block_inside_crosswalk_is_dropped_but_curb_block_kept():
    label, _ = merge_detections(_image(True), [CROSSWALK], [PATH, PAINT, CURB], PhotoKind.WIDE)
    assert label[175, 120] == MaskClass.CROSSWALK  # 노란 칠 오인식은 버림
    assert label[300, 530] == MaskClass.DOT  # 절반만 겹친 연석 점형은 남김
    assert label[270, 530] == MaskClass.DOT  # 겹친 부분도 블록으로 칠함


def test_unstriped_crosswalk_removed_before_block_check():
    """가까운 사진에서 민짜 바닥을 횡단보도로 잡아도, 그 위의 진짜 블록은 지워지지 않아야 한다."""
    block_on_fake = (_rect(100, 250, 300, 360), MaskClass.DOT, 0.8)
    label, seg = merge_detections(_image(False), [CROSSWALK], [block_on_fake], PhotoKind.CLOSE)
    assert seg.class_area_px["crosswalk"] == 0  # 줄무늬 없는 횡단보도는 지움
    assert label[175, 330] == MaskClass.DOT  # 그래서 블록은 남음


def test_tiny_block_pieces_dropped():
    tiny = (_rect(400, 405, 100, 105), MaskClass.DOT, 0.9)
    label, _ = merge_detections(_image(True), [], [tiny], PhotoKind.CLOSE)
    assert not (label == MaskClass.DOT).any()


def test_no_detections():
    label, seg = merge_detections(_image(True), [], [], PhotoKind.WIDE)
    assert not label.any()
    assert not seg.mean_confidence  # 블록이 없으면 0.0 또는 None (main의 segment.summarize는 None)


def test_class_mapping_by_name():
    allowed = {MaskClass.LINEAR, MaskClass.DOT}
    assert class_mapping({0: "linear", 1: "dot"}, allowed) == {0: MaskClass.LINEAR, 1: MaskClass.DOT}
    assert class_mapping({0: "Dot", 1: "LINEAR"}, allowed) == {0: MaskClass.DOT, 1: MaskClass.LINEAR}
    with pytest.raises(ValueError):  # 장면 모델을 블록 자리에 잘못 넣은 경우
        class_mapping({0: "crosswalk", 1: "road"}, allowed)
