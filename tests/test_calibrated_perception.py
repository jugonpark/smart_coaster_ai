import math
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import config
from perception.camera import WorldFrame
from perception.marker_object_detector import MarkerObjectDetector
from perception.marker_scanner import MarkerScan
from perception.object_detector import Detection, ObjectDetector
from perception.robot_tracker import RobotTracker
from perception.pose_tracker import PoseTracker
from perception.table_calibration import MarkerDefinition, TableCalibration


def world():
    markers = [MarkerDefinition(i, x, y, 5.0) for i, x, y in
               ((41, -29.5, 17), (42, 29.5, 17),
                (43, 29.5, -17), (44, -29.5, -17))]
    H = [[0.07, 0.02, -40], [0.015, -0.06, 21],
         [0.00008, 0.00004, 1]]
    value = WorldFrame(calibration=TableCalibration((1280, 720), (65, 40), markers, H))
    value.observe_frame((720, 1280, 3))
    return value


class CalibratedPerceptionTests(unittest.TestCase):
    def test_robot_heading_uses_transformed_marker_edge(self):
        frame_world = world()
        corners = np.array([[300., 280.], [370., 300.],
                            [365., 370.], [295., 350.]], dtype=np.float32)
        scan = MarkerScan(by_id={0: corners})
        pose = RobotTracker().process(np.zeros((720, 1280, 3), dtype=np.uint8),
                                      frame_world, scan)
        tl, tr = frame_world.to_world(*corners[0]), frame_world.to_world(*corners[1])
        self.assertTrue(pose.detected)
        np.testing.assert_allclose((pose.x_cm, pose.y_cm),
                                   frame_world.to_world(*MarkerScan.center_px(corners)))
        self.assertAlmostEqual(pose.heading_rad,
                               math.atan2(tr[1]-tl[1], tr[0]-tl[0]), places=6)

    def test_yolo_box_radius_uses_world_edge_points(self):
        frame_world = world()
        detector = ObjectDetector.__new__(ObjectDetector)
        detector.backend = 'stub'
        detector._frame_counter = 0
        detector._cached = [Detection('cup', 1.0, 1050, 500, 80, 60)]
        with patch.object(config, 'YOLO_EVERY_N_FRAMES', 100):
            item = detector.detect(np.zeros((720, 1280, 3), dtype=np.uint8),
                                   frame_world)[0]
        center = np.array(frame_world.to_world(1050, 500))
        expected = max(np.linalg.norm(np.array(frame_world.to_world(*p)) - center)
                       for p in ((1010, 500), (1090, 500), (1050, 470), (1050, 530)))
        self.assertAlmostEqual(item.radius_cm, expected, places=6)

    def test_aruco_object_uses_configured_cm_radius(self):
        frame_world = world()
        corners = np.array([[600., 300.], [650., 300.],
                            [650., 350.], [600., 350.]], dtype=np.float32)
        detections = MarkerObjectDetector().detect(
            np.zeros((720, 1280, 3), dtype=np.uint8), frame_world,
            scan=MarkerScan(by_id={1: corners}), now=100.0)
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].radius_cm, config.MARKER_OBJECTS[1][1])
        np.testing.assert_allclose((detections[0].x_cm, detections[0].y_cm),
                                   frame_world.to_world(*MarkerScan.center_px(corners)))

    def test_pose_landmark_uses_decoded_frame_size(self):
        tracker = PoseTracker.__new__(PoseTracker)
        tracker._last_ts_ms = -1
        landmark = SimpleNamespace(x=0.5, y=0.5, visibility=1.0, presence=1.0)
        tracker._landmarker = SimpleNamespace(
            detect_for_video=lambda image, timestamp: SimpleNamespace(
                pose_landmarks=[[landmark] * 33]))
        fake_mp = SimpleNamespace(Image=lambda **kwargs: None,
                                  ImageFormat=SimpleNamespace(SRGB=1))
        with patch('perception.pose_tracker.mp', fake_mp):
            pose = tracker.process(np.zeros((480, 640, 3), dtype=np.uint8),
                                   SimpleNamespace(to_world=lambda x, y: (x, y)), now=1.0)
        self.assertEqual(pose.left_wrist.px, (320.0, 240.0))


if __name__ == '__main__':
    unittest.main()
