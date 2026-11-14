"""대각선 가상 연결선에 의한 M3 정상 꺾임 후보 누락 재현 테스트."""
import cv2
import numpy as np

from pipeline.schema import MaskClass, NodeType, PathEdge, PathNode, PathStructure, PhotoKind, Point, Verdict
from pipeline.shape_rules import analyze_m3, find_m3_candidates


def test_dot_2arm_virtual_connector_angle():
    # 실제 점형 영역으로 연결된 선형은 왼쪽/위쪽의 직각 L자이지만,
    # 점형 중심→상부 선형 끝 사이의 가상 연결선은 오른쪽 대각선으로 우회함.
    mask = np.zeros((650, 650), dtype=np.uint8)
    cv2.rectangle(mask, (260, 260), (500, 490), 2, -1)
    cv2.rectangle(mask, (20, 348), (260, 418), 1, -1)
    cv2.rectangle(mask, (345, 70), (425, 260), 1, -1)
    # 2-arm DOT connection; top virtual connector aims up-right before rejoining linear
    path = PathStructure(
        photo=PhotoKind.CLOSE, width=650, height=650,
        nodes=[
            PathNode(id=1,type=NodeType.PASS,pt=Point(x=380,y=375),block=MaskClass.DOT,component=1),
            PathNode(id=2,type=NodeType.END,pt=Point(x=40,y=383),block=MaskClass.LINEAR,component=1),
            PathNode(id=3,type=NodeType.END,pt=Point(x=385,y=90),block=MaskClass.LINEAR,component=1),
        ],
        edges=[
            PathEdge(id=1,u=1,v=2,polyline=[Point(x=380,y=375),Point(x=250,y=383),Point(x=40,y=383)],length_px=340,block=MaskClass.LINEAR),
            PathEdge(id=2,u=1,v=3,polyline=[Point(x=380,y=375),Point(x=490,y=276),Point(x=390,y=245),Point(x=385,y=90)],length_px=375,block=MaskClass.LINEAR),
        ]
    )
    candidates = find_m3_candidates(mask, path)
    assert len(candidates)==1, candidates
    assert candidates[0].kind=='turn'
    assert candidates[0].verdict==Verdict.NORMAL
    assert candidates[0].direction_change_deg>=55.0


def test_straight_through_dot_does_not_create_false_turn():
    mask=np.zeros((500,500),np.uint8)
    cv2.rectangle(mask,(220,180),(280,320),2,-1)
    cv2.rectangle(mask,(235,30),(265,180),1,-1)
    cv2.rectangle(mask,(235,320),(265,480),1,-1)
    path=PathStructure(
        photo=PhotoKind.CLOSE,width=500,height=500,
        nodes=[PathNode(id=0,type=NodeType.PASS,pt=Point(x=250,y=250),block=MaskClass.DOT,component=1),
               PathNode(id=1,type=NodeType.END,pt=Point(x=250,y=40),block=MaskClass.LINEAR,component=1),
               PathNode(id=2,type=NodeType.END,pt=Point(x=250,y=460),block=MaskClass.LINEAR,component=1)],
        edges=[PathEdge(id=0,u=0,v=1,polyline=[Point(x=250,y=250),Point(x=250,y=40)],length_px=210),
               PathEdge(id=1,u=0,v=2,polyline=[Point(x=250,y=250),Point(x=250,y=460)],length_px=210)]
    )
    assert find_m3_candidates(mask,path)==[]
