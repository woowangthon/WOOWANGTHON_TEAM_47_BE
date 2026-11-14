from pipeline.hold_rules import HoldConfig, check_h1_h4
from pipeline.schema import (
    MaskClass, NodeType, PathNode, PathStructure, PhotoAnalysis, PhotoKind,
    Point, Segmentation, Verdict,
)


def make_analysis(conf=0.9, areas=None, nodes=None, components=None):
    return PhotoAnalysis(
        segmentation=Segmentation(
            photo=PhotoKind.CLOSE, width=1000, height=1000,
            class_area_px={"linear": 40000, "dot": 10000} if areas is None else areas,
            mean_confidence=conf,
        ),
        path=PathStructure(
            photo=PhotoKind.CLOSE, width=1000, height=1000,
            nodes=[] if nodes is None else nodes,
            component_area_px={} if components is None else components,
        ),
    )


def isolated(component, block=MaskClass.DOT):
    return PathNode(id=1, type=NodeType.ISOLATED,
                    pt=Point(x=100, y=200), block=block, component=component)


def test_confident_mask_has_no_holds():
    assert check_h1_h4(make_analysis()) == []


def test_low_confidence_h1():
    result = check_h1_h4(make_analysis(conf=0.2))
    assert len(result) == 1 and result[0].code == "H1"
    assert result[0].verdict == Verdict.HOLD


def test_no_detection_h1():
    assert check_h1_h4(make_analysis(areas={"linear": 0, "dot": 0}))[0].code == "H1"


def test_missing_confidence_is_not_automatically_low():
    assert check_h1_h4(make_analysis(conf=None)) == []


def test_unknown_class_area_is_not_proof_of_no_detection():
    assert check_h1_h4(make_analysis(areas={})) == []


def test_small_isolated_component_h4():
    result = check_h1_h4(make_analysis(nodes=[isolated(3)], components={3: 200}))
    assert len(result) == 1 and result[0].code == "H4"
    assert result[0].location == Point(x=100, y=200)


def test_large_standalone_dot_is_not_h4():
    assert check_h1_h4(make_analysis(nodes=[isolated(3)], components={3: 6000})) == []


def test_missing_area_does_not_trigger_h4():
    assert check_h1_h4(make_analysis(nodes=[isolated(3)])) == []


def test_config_validation():
    import pytest
    with pytest.raises(ValueError):
        HoldConfig(min_mean_confidence=1.5)
