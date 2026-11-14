"""문서용 파이프라인 그림 생성. 레포 최상위에서 실행 → docs/pipeline.png

    .venv/bin/python docs/make_pipeline_figure.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

FONT_DIR = Path.home() / "Library" / "Fonts"
for w in ("Medium", "Bold", "SemiBold"):
    p = FONT_DIR / f"Pretendard-{w}.otf"
    if p.exists():
        font_manager.fontManager.addfont(str(p))
plt.rcParams["font.family"] = "Pretendard" if (FONT_DIR / "Pretendard-Medium.otf").exists() else "AppleGothic"

# 색: 입출력 / AI 학습 모델 / 영상처리 / 규칙
C = {
    "io": ("#EEF0F3", "#5B6270"),
    "ai": ("#E3EEFF", "#2F6FDB"),
    "cv": ("#E2F5EA", "#1F8A4C"),
    "rule": ("#FFF1DE", "#C76A00"),
    "gate": ("#F6E8FF", "#8A3FC7"),
}
INK = "#1D2230"
SUB = "#586074"

fig, ax = plt.subplots(figsize=(12, 15.5), dpi=200)
ax.set_xlim(-17, 120)
ax.set_ylim(0, 155)
ax.axis("off")
fig.patch.set_facecolor("white")


def box(x, y, w, h, kind, title, lines=(), step=None, tsize=12.5):
    fc, ec = C[kind]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.6",
                                fc=fc, ec=ec, lw=1.6))
    ty = y + h - 3.2
    label = f"{step}  {title}" if step else title
    ax.text(x + w / 2, ty, label, ha="center", va="center", fontsize=tsize, color=INK, weight="bold")
    for i, ln in enumerate(lines):
        ax.text(x + w / 2, ty - 3.4 - i * 2.9, ln, ha="center", va="center", fontsize=9.6, color=SUB)
    return (x, y, w, h)


def arrow(p0, p1, color="#8A91A0", label=None, lx=0, ly=0):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=14, lw=1.6, color=color,
                                 shrinkA=0, shrinkB=0))
    if label:
        ax.text((p0[0] + p1[0]) / 2 + lx, (p0[1] + p1[1]) / 2 + ly, label, ha="center", va="center",
                fontsize=9.2, color=color, weight="bold",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none"))


def stage(y, text):
    ax.text(-16, y, text, ha="left", va="center", fontsize=10, color="#9AA1AE", weight="bold")


# 제목
ax.text(60, 151.5, "TactileCheck 분석 파이프라인", ha="center", fontsize=19, weight="bold", color=INK)
ax.text(60, 147.6, "사진 한 장 → 두 모델로 영역 분할 → 경로 구조 → 사진 내용에 맞는 규칙 판정 → 근거 조항과 함께 결과",
        ha="center", fontsize=10.5, color=SUB)

# ① 입력
box(38, 133, 44, 10, "io", "사진 1장 업로드", ["구도·거리 제한 없음 · React 모바일 웹"], "①")
arrow((60, 133), (60, 128.5))

# ② 영역 분할
stage(122, "② 영역 분할")
box(14, 113, 40, 15, "ai", "블록 모델", ["YOLOv8-seg", "선형 · 점형"])
box(66, 113, 40, 15, "ai", "장면 모델", ["YOLOv8-seg", "횡단보도 · 차도"])
ax.plot([60, 60], [128.5, 128.5])
arrow((60, 128.5), (34, 128.2), color="#8A91A0")
arrow((60, 128.5), (86, 128.2), color="#8A91A0")
box(22, 94, 76, 14, "cv", "라벨 통합 + 오검출 필터",
    ["① 줄무늬 없는 '횡단보도' 제거 (가까운 아스팔트·콘크리트 오인식)",
     "② 횡단보도·차도 안에 잡힌 블록 제거 (횡단보도 노란 칠 오인식)"])
arrow((34, 113), (40, 108))
arrow((86, 113), (80, 108))
arrow((60, 94), (60, 89.5))

# ③ 경로 구조
stage(84, "③ 경로 구조")
box(28, 78, 64, 11.5, "cv", "중심선 추출",
    ["블록 마스크 → 중심선 → 꺾임점 · 분기점 · 끝점 (짧은 가지 제거)"])

# 관문
arrow((44, 78), (30, 71.5), color=C["gate"][1])
arrow((76, 78), (90, 71.5), color=C["gate"][1])
box(8, 62, 44, 9.5, "gate", "블록이 크고 선명한가?", ["예 → M 계열 · 아니요 → H4 보류"], tsize=11.5)
box(68, 62, 44, 9.5, "gate", "횡단보도·차도가 보이는가?", ["예 → C 계열 · 아니요 → 해당 없음(H3)"], tsize=11.5)

# ④ 정밀 분석 (C쪽)
stage(55.5, "④ 정밀 분석")
arrow((90, 62), (90, 57))
box(68, 41, 44, 16, "cv", "시점 보정 + 진입부 분석",
    ["횡단보도 흰 줄 → 지평선 → 위에서 본 평면",
     "진입부 ROI 2D FFT: 점형/선형 · 돌출선 방향",
     "실패하면 방향 판정만 보류 (H2)"], tsize=11.5)

# ⑤ 규칙 판정
stage(34.5, "⑤ 규칙 판정")
arrow((30, 62), (30, 37))
box(8, 20, 44, 17, "rule", "M 계열 (모양)",
    ["H1 신뢰도 낮음 → 보류", "M3 꺾임·분기점 점형 누락", "M4 선형 경로 끊김"], tsize=11.5)
arrow((90, 41), (90, 37))
box(60, 12, 52, 25, "rule", "C 계열 (문맥)",
    ["H1 신뢰도 낮음 → 보류", "C2 차도로 직진 (가장 먼저)", "C5 점형만 있고 선형 유도 없음",
     "C1 입구까지 안 이어짐 · C3 옆으로 비켜감", "C4 진입부 점형·돌출선 방향", "C6 선형 줄 방향 불일치"], tsize=11.5)

# ⑥ 결과
arrow((30, 20), (48, 10.5))
arrow((86, 12), (72, 10.5))
box(22, 1, 76, 9.5, "io", "결과: 오류 의심 / 정상 추정 / 판정 보류",
    ["위치 오버레이 · 근거 조항 · 개선 방향 · 재촬영 안내 (FastAPI · 결과 페이지)"], "⑥", tsize=12)

# 범례
lx, ly = -15, 146
for i, (k, name) in enumerate([("ai", "AI 학습 모델"), ("cv", "영상처리"), ("gate", "규칙 적용 관문"), ("rule", "규칙 판정")]):
    fc, ec = C[k]
    ax.add_patch(FancyBboxPatch((lx + i * 0, ly - 3.2 - i * 3.0), 3.2, 2.2, boxstyle="round,pad=0,rounding_size=0.5",
                                fc=fc, ec=ec, lw=1.2))
    ax.text(lx + 4.2, ly - 2.1 - i * 3.0, name, va="center", fontsize=9, color=SUB)

ax.text(118, 146 - 2.1, "AI는 영역만 인식하고,\n오류 판정은 규칙이 한다\n→ 판정마다 근거 조항", ha="right", va="top",
        fontsize=9, color=SUB, linespacing=1.4)

out = Path(__file__).with_name("pipeline.png")
fig.savefig(out, bbox_inches="tight", pad_inches=0.3, facecolor="white")
print(out)
