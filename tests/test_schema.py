from pipeline.schema import (
    SCHEMA_VERSION,
    Finding,
    InspectionResult,
    PhotoKind,
    Point,
    Verdict,
)


def test_result_json_roundtrip():
    r = InspectionResult(
        overall=Verdict.SUSPECTED,
        findings=[
            Finding(code="M3", verdict=Verdict.SUSPECTED, photo=PhotoKind.CLOSE,
                    message="꺾임점이 점형이 아님", location=Point(x=1, y=2)),
        ],
    )
    data = r.model_dump(mode="json")
    assert data["schema_version"] == SCHEMA_VERSION
    assert data["findings"][0]["verdict"] == "suspected"
    assert InspectionResult.model_validate(data) == r
