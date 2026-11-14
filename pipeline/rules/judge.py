"""⑤ 규칙 판정.

judge_photo(): 사진 한 장 구조 (서비스 기본).
    사진에 보이는 것으로 적용할 규칙을 정한다. 둘 다 해당하면 둘 다 판정한다.
    - H1 블록 신뢰도 낮음 → 사진 전체 보류
    - 횡단보도·차도가 보임 → C 계열
    - 블록이 크고 선명함 (M 관문) → M 계열
      관문을 못 넘으면: C 판정을 했으면 M은 건너뛰고, 아무것도 못 봤으면 H4 보류(가까이 찍기 안내)

이중 검출로 "종류 모름 블록"(장면 모델은 점자블록이라는데 블록 모델이 못 찾은 영역)이 있으면
C4·C5 '오류 의심'을 '보류'로 바꾼다. C4·C5는 "점형/선형이 없다"는 규칙이라 블록 모델이 블록을 놓치면
정상 설치도 오류로 나오기 때문. C1·C2·C3·C6·M 계열은 건드리지 않는다.

judge(): 예전 사진 두 장(근접·넓은) 구조. 테스트와 호환용으로 남겨 둔다.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..schema import Finding, InspectionResult, PhotoAnalysis, Verdict
from . import c_rules, hold, m_rules


UNKNOWN_BLOCK_HOLD = ("블록 모델이 읽지 못한 점자블록이 있어 판정 보류. 블록이 더 크게 나오게 찍어 주세요")


def _hold_if_unknown_blocks(findings: list[Finding], photo: Optional[PhotoAnalysis]) -> list[str]:
    """종류 모름 블록이 있으면 C4·C5 의심을 보류로 바꾼다. 바꾼 코드 목록을 돌려준다."""
    area = photo.segmentation.unknown_block_area_px if photo is not None else None
    if not area:
        return []
    held = []
    for f in findings:
        if f.code in ("C4", "C5") and f.verdict == Verdict.SUSPECTED:
            f.message = f"{UNKNOWN_BLOCK_HOLD} (원래 판정: {f.message})"
            f.verdict, f.suggestion, f.clause = Verdict.HOLD, None, None
            held.append(f.code)
    return held


def _c_family(wide: PhotoAnalysis, wide_label: Optional[np.ndarray], cfg: hold.HoldConfig) -> list[Finding]:
    """C 계열 (H1·H3 검사는 호출하는 쪽에서)."""
    findings: list[Finding] = []
    c = c_rules.check_direction_and_connection(wide, wide_label)
    findings += c
    if any(f.code == "C2" and f.verdict == Verdict.SUSPECTED for f in c):  # C2가 있으면 C3는 보조 표시
        for f in c:
            if f.code == "C3":
                f.primary = False
    partial = hold.check_h3_partial(wide, cfg)
    if partial:  # 횡단보도 없음: C4·C6도 볼 수 없음
        return findings + partial
    findings += c_rules.check_c6(wide, wide_label)
    h2 = hold.check_h2(wide)
    if h2:  # 보정 실패: 방향은 보류, 점형 유무만 라벨로 확인
        findings += h2 + c_rules.check_c4_from_label(wide, wide_label)
    else:
        findings += c_rules.check_c4(wide)
    return findings


def judge_photo(
    photo: PhotoAnalysis,
    label: Optional[np.ndarray],
    cfg: Optional[hold.HoldConfig] = None,
) -> InspectionResult:
    """사진 한 장 판정. scope에 M·C 계열을 판정했는지, 왜 안 했는지 남긴다."""
    cfg = cfg or hold.HoldConfig()
    h1 = hold.check_h1(photo, cfg)
    if h1:
        return InspectionResult(overall=Verdict.HOLD, findings=h1, photo=photo,
                                scope={"C": "보류 (블록 인식 신뢰도 낮음)", "M": "보류 (블록 인식 신뢰도 낮음)"})

    findings: list[Finding] = []
    scope: dict[str, str] = {}

    c_on = hold.has_class(photo, "crosswalk", cfg) or hold.has_class(photo, "road", cfg)
    if c_on:
        findings += _c_family(photo, label, cfg)
        scope["C"] = "판정"
    else:
        scope["C"] = "해당 없음 (횡단보도·차도가 보이지 않음)"

    m_ok, why = hold.m_gate(photo, label, cfg)
    if m_ok:
        m = m_rules.check_m(photo, label)
        if c_on and m and all(f.verdict == Verdict.HOLD for f in m):
            m = []  # 횡단보도 사진에 꺾임·분기점이 없으면 M은 대상 없음 (보류로 세지 않음)
            scope["M"] = "대상 없음 (꺾임·분기점이 보이지 않음)"
        else:
            scope["M"] = "판정"
        findings += m
    elif c_on:
        scope["M"] = f"건너뜀 ({why})"
    else:
        findings += hold.h4_from_gate(why)
        scope["M"] = f"보류 ({why})"

    _hold_if_unknown_blocks(findings, photo)  # 종합 판정 계산 전에
    return InspectionResult(overall=_overall(findings), findings=findings, photo=photo, scope=scope)


def judge(
    close: Optional[PhotoAnalysis],
    wide: Optional[PhotoAnalysis],
    wide_label: Optional[np.ndarray] = None,
    cfg: Optional[hold.HoldConfig] = None,
) -> InspectionResult:
    """예전 사진 두 장 구조."""
    cfg = cfg or hold.HoldConfig()
    findings: list[Finding] = []

    if close is not None:
        holds = hold.check_h1(close, cfg) or hold.check_h4(close, cfg)
        findings += holds
        if not holds:
            findings += m_rules.check_m3(close)
            findings += m_rules.check_m4(close)

    if wide is not None:
        holds = hold.check_h1(wide, cfg) or hold.check_h3(wide, cfg)
        findings += holds
        if not holds:
            findings += _c_family(wide, wide_label, cfg)
        _hold_if_unknown_blocks(findings, wide)

    return InspectionResult(overall=_overall(findings), findings=findings, close=close, wide=wide)


def _overall(findings: list[Finding]) -> Verdict:
    verdicts = {f.verdict for f in findings}
    if Verdict.SUSPECTED in verdicts:
        return Verdict.SUSPECTED
    if Verdict.NORMAL in verdicts and Verdict.HOLD not in verdicts:
        return Verdict.NORMAL
    return Verdict.HOLD
