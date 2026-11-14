import numpy as np

from pipeline.crosswalk import drop_unstriped
from pipeline.schema import MaskClass

H, W = 600, 800


def _scene(striped: bool, vertical_stripes: bool = True):
    rng = np.random.default_rng(0)
    img = np.full((H, W, 3), 90, np.uint8) + rng.integers(0, 12, (H, W, 1), dtype=np.uint8)
    label = np.zeros((H, W), np.uint8)
    label[100:400, 100:700] = MaskClass.CROSSWALK
    if striped:
        for k in range(0, 600, 60):
            if vertical_stripes:
                img[100:400, 100 + k:130 + k] = 230
            else:
                img[100 + k // 2:115 + k // 2, 100:700] = 230
    return img, label


def test_striped_crosswalk_is_kept():
    img, label = _scene(striped=True)
    out, kept = drop_unstriped(img, label)
    assert (out == MaskClass.CROSSWALK).sum() == (label == MaskClass.CROSSWALK).sum()
    assert len(kept) == 1 and kept[0].striped


def test_plain_ground_is_dropped():
    img, label = _scene(striped=False)
    out, kept = drop_unstriped(img, label)
    assert not (out == MaskClass.CROSSWALK).any()
    assert kept == []


def test_stripe_angle():
    img, label = _scene(striped=True, vertical_stripes=True)
    a = drop_unstriped(img, label)[1][0].stripe_angle_deg
    assert min(abs(a - 90), abs(a - 270)) < 5  # 세로 줄무늬
    img, label = _scene(striped=True, vertical_stripes=False)
    a = drop_unstriped(img, label)[1][0].stripe_angle_deg
    assert min(a, 180 - a) < 5  # 가로 줄무늬
