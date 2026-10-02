import json
import math
import sys
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "raspberry_pi"))

from perception.object_detector import Detection, ObjectDetector
from perception.robot_tracker import MarkerScan, RobotTracker, WorldFrame
from perception.table_calibration import MarkerDefinition, TableCalibration
from perception.vision_state import build_vision_state
from planning.transform import world_to_robot
from core.shared_state import SharedState
from fusion.sensor_fusion import SensorFusion
from main import perception_step
from perception.radar import RadarState


def calibration(H=None, size=(1280, 720)):
    H = np.eye(3) if H is None else np.asarray(H, dtype=np.float64)
    return TableCalibration(size[0], size[1], H, np.linalg.inv(H), [], 0.0, 0.0)


def test_identity_and_round_trip():
    c = calibration()
    assert c.pixel_to_table(12.5, 30.25) == pytest.approx((12.5, 30.25))
    assert c.table_to_pixel(*c.pixel_to_table(81.2, 43.7)) == pytest.approx((81.2, 43.7))


def test_synthetic_perspective_known_points():
    image = np.array([[100, 80], [1100, 100], [1200, 650], [70, 690]], np.float64)
    table = np.array([[-50, 30], [50, 30], [50, -30], [-50, -30]], np.float64)
    H, _ = cv2.findHomography(image, table)
    c = calibration(H)
    for px, expected in zip(image, table):
        assert c.pixel_to_table(*px) == pytest.approx(expected, abs=1e-7)


@pytest.mark.parametrize("matrix", [np.zeros((3, 3)), [[1, 0, math.nan], [0, 1, 0], [0, 0, 1]]])
def test_invalid_singular_and_nonfinite_h_rejected(matrix):
    with pytest.raises(ValueError):
        TableCalibration(1280, 720, matrix)


def test_nonfinite_input_and_zero_homogeneous_w_are_rejected():
    c = calibration([[1, 0, 0], [0, 1, 0], [1, 0, -5]])
    with pytest.raises(ValueError):
        c.pixel_to_table(5, 0)
    with pytest.raises(ValueError):
        calibration().pixel_to_table(math.inf, 0)


def test_missing_and_resolution_mismatch_are_invalid():
    path = ROOT / "tests" / "_generated_table_calibration.json"
    assert not TableCalibration.load_optional(path, (1280, 720)).valid
    try:
        calibration().save(path)
        assert not TableCalibration.load_optional(path, (640, 480)).valid
        assert TableCalibration.load_optional(path, (1280, 720)).valid
    finally:
        path.unlink(missing_ok=True)


def test_example_calibration_cannot_enable_motion():
    example = ROOT / "calibration" / "table_calibration.example.json"
    assert not TableCalibration.load_optional(example, (1280, 720)).valid


def test_four_markers_provide_sixteen_corner_correspondences():
    markers = [
        MarkerDefinition(41, -50, 30, 4, 0), MarkerDefinition(42, 50, 30, 4, 0),
        MarkerDefinition(43, 50, -30, 4, 0), MarkerDefinition(44, -50, -30, 4, 0),
    ]
    table_points = np.vstack([m.table_corners() for m in markers])
    image_points = table_points * 7 + np.array([640, 360])
    scan = MarkerScan({m.marker_id: image_points[i * 4:(i + 1) * 4] for i, m in enumerate(markers)})
    c = TableCalibration.from_marker_scan(scan, markers, 1280, 720)
    assert c.valid and c.correspondence_count == 16
    assert c.rms_error_cm < 1e-8


def test_robot_center_and_heading_are_transformed_on_table_plane():
    H = np.array([[0.08, 0.015, -50], [0.01, -0.09, 30], [0.0001, 0.0002, 1]], float)
    world = WorldFrame(calibration=calibration(H), require_calibration=True)
    corners = np.array([[500, 300], [580, 315], [570, 390], [490, 375]], float)
    tracker = RobotTracker()
    tracker._scanner.scan = lambda frame: MarkerScan({0: corners})
    pose = tracker.process(None, world)
    center = world.to_world(*MarkerScan.center_px(corners))
    tl, tr = world.to_world(*corners[0]), world.to_world(*corners[1])
    assert (pose.x_cm, pose.y_cm) == pytest.approx(center)
    assert pose.heading_rad == pytest.approx(math.atan2(tr[1] - tl[1], tr[0] - tl[0]))
    assert not math.isclose(pose.heading_rad, MarkerScan.heading_rad(corners))


def test_detector_radius_uses_local_transformed_bbox_edges():
    H = np.array([[0.1, 0, 0], [0, 0.1, 0], [0.001, 0, 1]], float)
    world = WorldFrame(calibration=calibration(H), require_calibration=True)
    with patch("config.YOLO_BACKEND", "stub"):
        detector = ObjectDetector()
    detector._cached = [Detection("box", 0.9, 100, 100, 20, 20),
                        Detection("box", 0.9, 900, 100, 20, 20)]
    detector._inferred_once = True
    detector._frame_counter = 0
    out = detector.detect(None, world)
    assert out[0].radius_cm != pytest.approx(out[1].radius_cm)
    assert all(d.radius_cm > 0 and math.isfinite(d.radius_cm) for d in out)


def test_calibration_invalid_makes_vision_unknown_and_transform_convention_remains():
    state = build_vision_state(camera_ok=True, captured_at=10, robot=tracker_pose(),
                               detections=[], detector_available=True, now=10,
                               calibration_valid=False)
    assert not state.calibration_valid
    assert all(value == "UNKNOWN" for value in state.sectors.values())
    assert world_to_robot(0, 10, math.pi / 2) == pytest.approx((10, 0))


def test_missing_runtime_calibration_publishes_invalid_world_without_running_detectors():
    now = __import__("time").monotonic()
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    camera = type("Camera", (), {"last_ok_t": now, "read": lambda self: (True, frame)})()
    forbidden = type("Forbidden", (), {"process": lambda *args: (_ for _ in ()).throw(
        AssertionError("tracker must not run without calibration"))})()
    radar = type("Radar", (), {"read": lambda self: RadarState()})()
    shared = SharedState()
    world = WorldFrame(calibration=TableCalibration.invalid(1280, 720, reason="missing"),
                       require_calibration=True)
    assert perception_step(camera, world, forbidden, None, radar, SensorFusion(), shared)
    snapshot = shared.snapshot().world
    assert snapshot is not None
    assert not snapshot.vision.calibration_valid
    assert not snapshot.vision_valid
    assert not snapshot.sensor_valid


def tracker_pose():
    from perception.robot_tracker import RobotPose
    return RobotPose(True, 0, 0, 0, timestamp=10)
