"""C 계열: 넓은 사진 문맥 판정. (담당: B)

판정 순서 (노션 5장)
1. 방향 판정 먼저: 선형 줄 끝 구간 연장선
   - 연장선이 횡단보도 밖 차도에 닿음 → C2 (다른 블록으로 횡단보도에 이어져 있어도 오류)
   - 연장선이 횡단보도를 비켜가고, 옆 횡단보도 쪽 꺾임 안내 없음 → C3
     (C2와 함께 뜨면 C3는 primary=False)
2. 방향 정상이면 연결 판정: 경로가 횡단보도 입구에 닿는가
   - 닿음 → 정상, C4 검사로
   - 같은 방향 덩어리 사이 틈 → M4 끊김 의심
   - 그 외 → C1
연결·방향 판정은 시점 보정 없이 진행한다. C4만 보정이 필요하다.
"""

from __future__ import annotations

from ..schema import Finding, PhotoAnalysis


def check_direction_and_connection(wide: PhotoAnalysis, wide_label) -> list[Finding]:
    """C1·C2·C3 (+ 넓은 사진 M4).

    TODO(B):
    - skeleton.end_direction() 으로 선형 줄 끝점의 바깥 방향 벡터
    - 연장선(레이)을 라벨 마스크 위에서 따라가며 처음 닿는 클래스 확인 (ROAD / CROSSWALK)
    - 횡단보도 입구 = 횡단보도 마스크와 보도 쪽 경계, 경로 끝점과의 거리로 연결 판정
    """
    return []


def check_c4(wide: PhotoAnalysis) -> list[Finding]:
    """C4: 진입부 ROI 돌출선 방향 vs 횡단보도 진행축, 진입부 점형 유무.

    TODO(B): wide.rectification.crosswalk_axis_deg 와 진입부 RoiAnalysis.ridge_angle_deg 비교.
    """
    return []
