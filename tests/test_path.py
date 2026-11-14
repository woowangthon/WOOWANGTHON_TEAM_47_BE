import unittest
import numpy as np
import cv2
from pipeline.schema import MaskClass, PhotoKind, NodeType
from pipeline.skeleton import extract_path, SkeletonConfig


class PathTests(unittest.TestCase):
    def setUp(self):
        self.cfg = SkeletonConfig(min_area_px=20, min_dot_area_px=20, close_px=0,
                                  dot_attach_px=4, spur_px=5, turn_deg=45)

    def test_dot_only_no_fake_branch(self):
        mask = np.zeros((320, 320), dtype=np.uint8)
        cv2.rectangle(mask, (75,75), (245,245), 2, -1)
        result = extract_path(mask, PhotoKind.CLOSE, self.cfg)
        self.assertEqual(len(result.edges), 0)
        self.assertEqual(len(result.nodes), 1)
        self.assertEqual(result.nodes[0].block, MaskClass.DOT)
        self.assertEqual(result.nodes[0].type, NodeType.ISOLATED)

    def test_line_and_two_dot_tiles(self):
        mask = np.zeros((400, 400), dtype=np.uint8)
        cv2.rectangle(mask,(175,100),(225,290),1,-1)
        cv2.rectangle(mask,(165,50),(235,110),2,-1)
        cv2.rectangle(mask,(165,285),(235,350),2,-1)
        result = extract_path(mask, PhotoKind.CLOSE, self.cfg)
        dots=[n for n in result.nodes if n.block == MaskClass.DOT]
        self.assertEqual(len(dots),2)
        self.assertEqual(len([n for n in result.nodes if n.type == NodeType.BRANCH]),0)
        self.assertTrue(any(n.type == NodeType.END for n in dots))
        self.assertEqual(len(result.edges),1)
        self.assertEqual(result.edges[0].block, MaskClass.LINEAR)

    def test_three_linear_arms_meet_at_dot(self):
        mask = np.zeros((420, 420), dtype=np.uint8)
        cv2.rectangle(mask, (40, 170), (155, 210), 1, -1)
        cv2.rectangle(mask, (265, 170), (380, 210), 1, -1)
        cv2.rectangle(mask, (190, 265), (230, 380), 1, -1)
        cv2.rectangle(mask, (155, 145), (265, 265), 2, -1)
        result = extract_path(mask, PhotoKind.CLOSE, self.cfg)
        dot_nodes = [n for n in result.nodes if n.block == MaskClass.DOT]
        self.assertEqual(len(dot_nodes), 1)
        self.assertEqual(dot_nodes[0].type, NodeType.BRANCH)
        self.assertEqual(len(result.edges), 3)
        self.assertEqual(len([n for n in result.nodes if n.block == MaskClass.LINEAR and n.type == NodeType.BRANCH]), 0)

    def test_two_separate_linear_parts_no_bridging(self):
        mask = np.zeros((320, 320), np.uint8)
        cv2.rectangle(mask, (110, 50), (160, 130), 1, -1)
        cv2.rectangle(mask, (110, 195), (160, 290), 1, -1)
        result = extract_path(mask, PhotoKind.CLOSE, self.cfg)
        comps = set(n.component for n in result.nodes)
        self.assertEqual(len(comps), 2)

    def test_disconnected_dot_keeps_gap(self):
        mask = np.zeros((400,400), np.uint8)
        cv2.rectangle(mask,(175,100),(225,220),1,-1)
        cv2.rectangle(mask,(165,290),(235,345),2,-1) # real 70px gap
        path = extract_path(mask, PhotoKind.CLOSE, self.cfg)
        dots=[n for n in path.nodes if n.block == MaskClass.DOT]
        self.assertEqual(len(dots),1)
        self.assertEqual(dots[0].type,NodeType.ISOLATED)

    def test_l_corner_without_dot_is_turn(self):
        mask=np.zeros((400,400),np.uint8)
        pts=np.array([[110,300],[110,120],[300,120]],np.int32)
        cv2.polylines(mask,[pts],False,1,45)
        path=extract_path(mask,PhotoKind.CLOSE,self.cfg)
        self.assertGreaterEqual(len([n for n in path.nodes if n.type==NodeType.TURN]),1)

    def test_empty_mask(self):
        mask=np.zeros((100,100),np.uint8)
        path=extract_path(mask,PhotoKind.CLOSE,self.cfg)
        self.assertEqual(path.nodes,[])
        self.assertEqual(path.edges,[])
        self.assertEqual(path.component_area_px,{})

if __name__=='__main__':unittest.main()
