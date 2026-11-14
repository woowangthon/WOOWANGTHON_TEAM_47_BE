"""⑤ 규칙 판정: 보류 조건 → M 계열 → C 계열 → 종합."""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..schema import Finding, InspectionResult, PhotoAnalysis, Verdict
from . import c_rules, hold, m_rules


def judge(
    close: Optional[PhotoAnalysis],
    wide: Optional[PhotoAnalysis],
    wide_label: Optional[np.ndarray] = None,
    cfg: Optional[hold.HoldConfig] = None,
) -> InspectionResult:
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
            c = c_rules.check_direction_and_connection(wide, wide_label)
            findings += c
            # C2가 있으면 C3는 보조 표시
            if any(f.code == "C2" and f.verdict == Verdict.SUSPECTED for f in c):
                for f in c:
                    if f.code == "C3":
                        f.primary = False
            partial = hold.check_h3_partial(wide, cfg)
            if partial:  # 횡단보도 없음: C4도 볼 수 없음
                findings += partial
            else:
                h2 = hold.check_h2(wide)
                if h2:  # 보정 실패: 방향은 보류, 점형 유무만 라벨로 확인
                    findings += h2 + c_rules.check_c4_from_label(wide, wide_label)
                else:
                    findings += c_rules.check_c4(wide)

    return InspectionResult(overall=_overall(findings), findings=findings, close=close, wide=wide)


def _overall(findings: list[Finding]) -> Verdict:
    verdicts = {f.verdict for f in findings}
    if Verdict.SUSPECTED in verdicts:
        return Verdict.SUSPECTED
    if Verdict.NORMAL in verdicts and Verdict.HOLD not in verdicts:
        return Verdict.NORMAL
    return Verdict.HOLD
