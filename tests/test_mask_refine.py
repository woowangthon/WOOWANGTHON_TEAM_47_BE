import numpy as np
import cv2
import pytest
from pipeline.mask_refine import MaskRefineConfig, refine_mask


def test_removes_small_isolated_noise_but_preserves_large_shapes():
    a = np.zeros((400, 500), dtype=np.uint8)
    cv2.rectangle(a, (50, 90), (350, 160), 1, -1)
    cv2.rectangle(a, (355, 90), (380, 160), 2, -1)
    a[300:302, 300:303] = 1
    a[250:254, 250:254] = 2
    b, report = refine_mask(a, MaskRefineConfig(min_area_px=50, min_largest_ratio=0.002))
    assert b[300,300] == 0 and b[250,250] == 0
    assert b[120,100] == 1 and b[120,365] == 2
    assert a[300,300] == 1  # no input mutation
    assert report['classes']['linear']['removed_components'] == 1
    assert report['classes']['dot']['removed_components'] == 1


def test_no_gap_filling_or_class_overwrite():
    a = np.zeros((200,250),np.uint8)
    a[50:120, 20:100] = 1
    a[50:120, 103:190] = 1
    a[50:120, 100:103] = 3  # crosswalk output from team B must be preserved
    a[130:190, 20:220] = 2
    b,_=refine_mask(a,MaskRefineConfig(min_area_px=5,min_largest_ratio=0))
    assert np.array_equal(a,b)
    assert np.all(b[50:120,100:103]==3)


def test_real_gap_preserved_for_m4():
    a=np.zeros((200,300),np.uint8)
    a[30:80,20:130]=1
    a[30:80,136:260]=1
    b,_=refine_mask(a,MaskRefineConfig(min_area_px=10,min_largest_ratio=0.001))
    assert np.all(b[30:80,130:136]==0)
    assert np.all(b[30:80,20:130]==1)
    assert np.all(b[30:80,136:260]==1)


def test_empty_and_no_tactile_classes_are_safe():
    a=np.zeros((30,30),np.uint8)
    a[10:20,10:20]=4
    b,report=refine_mask(a)
    assert np.array_equal(a,b)
    assert report['pixels_changed']==0


def test_invalid_mask_rejected():
    with pytest.raises(ValueError):
        refine_mask(np.zeros((5,5,3),np.uint8))
    with pytest.raises(ValueError):
        refine_mask(np.ones((5,5),np.int32))
    with pytest.raises(ValueError):
        refine_mask(np.full((5,5),255,np.uint8))
