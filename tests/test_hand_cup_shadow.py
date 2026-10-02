import json
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "raspberry_pi"))
sys.path.insert(0, str(ROOT))

import config
from core.shared_state import SharedState
from decision.risk_arbiter import choose_risk
from decision.risk_evaluator import RiskEvaluator
from fusion.sensor_fusion import SensorFusion
from fusion.sensor_fusion import SectorState
from main import _hand_debug, _wrist_inputs, perception_step
from perception.radar import parse_fake_packet
from perception.robot_tracker import MarkerScanner, RobotTracker, WorldFrame
from perception.table_calibration import TableCalibration
from planning.escape_planner import EscapePlanner
from safety.safety_manager import SafetyManager
from shared_ai.hand_cup import HandCupEvaluator


def _frame():
    frame = np.full((480, 640, 3), 255, dtype=np.uint8)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    for marker_id, x in ((0, 80), (1, 360)):
        marker = cv2.aruco.generateImageMarker(dictionary, marker_id, 150)
        frame[150:300, x:x + 150] = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
    return frame


def _perceive(wrist_x_cm=None, *, pose_x_cm=None):
    frame = _frame()
    now = time.monotonic()
    camera = SimpleNamespace(last_ok_t=now, read=lambda: (True, frame))
    calibration = TableCalibration(640, 480,
        np.array([[0.1, 0, 0], [0, -0.1, 48], [0, 0, 1]], dtype=float))
    world_frame = WorldFrame(calibration=calibration, require_calibration=True)
    detector = SimpleNamespace(available=False, backend="stub", detect=lambda *args: [])
    hand_list = ([] if wrist_x_cm is None else [SimpleNamespace(
        wrist_x_cm=wrist_x_cm, wrist_y_cm=25.55,
        px=[(wrist_x_cm * 10, 224.5)], handedness="Left")])
    hand = SimpleNamespace(process=lambda *args: SimpleNamespace(hands=hand_list,
                                                                 any_grasp_ready=False))
    pose_wrist = (None if pose_x_cm is None else
                  SimpleNamespace(x_cm=pose_x_cm, y_cm=25.55,
                                  px=(pose_x_cm * 10, 224.5)))
    pose = SimpleNamespace(process=lambda *args: SimpleNamespace(
        left_wrist=pose_wrist, right_wrist=None,
        wrists=[] if pose_wrist is None else [pose_wrist]))
    radar = SimpleNamespace(read=lambda: parse_fake_packet(
        json.dumps({"targets": []}), received_at=time.monotonic()))
    shared = SharedState()
    assert perception_step(camera, world_frame, RobotTracker(), detector, radar,
                           SensorFusion(), shared, scanner=MarkerScanner(),
                           pose_tracker=pose, hand_tracker=hand,
                           hand_cup_eval=HandCupEvaluator())
    return shared.snapshot()


@pytest.mark.parametrize("wrist_x,level", [(3.45, "SAFE"), (18.45, "WARN"),
                                            (33.45, "DANGER")])
def test_stub_keeps_marker_hand_evidence_and_unknown_obstacle_sectors(wrist_x, level):
    state = _perceive(wrist_x)
    world = state.world
    assert world.robot.detected and world.vision_valid and world.sensor_valid
    assert not world.vision.detector_available
    assert world.vision.wrist_source == "HAND" and world.vision.hands_count == 1
    assert any(obj.detection.label == "cup" for obj in world.vision.objects)
    assert world.hand_cup.level == level
    assert set(world.sectors.values()) == {"UNKNOWN"}
    if level != "SAFE":
        risk = choose_risk(world, RiskEvaluator().evaluate(world), now=time.monotonic())
        assert risk.level == level and risk.source == "HAND"
        assert risk.threat_direction is not None
        assert EscapePlanner().plan(world, risk).status == "STOP"
    debug = _hand_debug(state)
    assert "frame=640x480" in debug and "ID0=True ID1=True hands=1" in debug
    assert f"hand_risk={level}" in debug and "detector_available=False" in debug


def test_hand_primary_pose_fallback_and_no_wrist():
    primary = _perceive(33.45, pose_x_cm=3.45)
    assert primary.world.vision.wrist_source == "HAND"
    assert primary.world.hand_cup.level == "DANGER"
    fallback = _perceive(None, pose_x_cm=18.45)
    assert fallback.world.vision.wrist_source == "POSE"
    assert fallback.world.vision.hands_count == 0
    assert fallback.world.hand_cup.level == "WARN"
    absent = _perceive(None)
    assert absent.world.vision.wrist_source == "NONE"
    assert absent.world.vision.trigger_wrist_px is None
    assert absent.world.hand_cup is None and not absent.world.vision_valid


def test_two_hands_select_closest_to_id1_and_keep_trigger():
    cup = SimpleNamespace(x_cm=30, y_cm=20)
    hands = SimpleNamespace(hands=[
        SimpleNamespace(wrist_x_cm=0, wrist_y_cm=20, px=[(0, 200)], handedness="Left"),
        SimpleNamespace(wrist_x_cm=25, wrist_y_cm=20, px=[(250, 200)], handedness="Right"),
    ])
    human, source = _wrist_inputs(hands, None, cup)
    evidence = HandCupEvaluator().update(human, cup, now=time.monotonic())
    assert source == "HAND" and evidence.trigger_wrist_side == "RIGHT"
    assert evidence.trigger_wrist_x_cm == 25 and evidence.distance_cm == 5


def test_hand_landmarker_wrist_uses_actual_640x480_frame_shape():
    from shared_ai.hand_tracker import HandTracker

    result = SimpleNamespace(hand_landmarks=[[
        SimpleNamespace(x=0.5, y=0.25) for _ in range(21)]],
        handedness=[], hand_world_landmarks=[])
    tracker = HandTracker.__new__(HandTracker)
    tracker._last_ts_ms = -1
    tracker._landmarker = SimpleNamespace(detect_for_video=lambda *args: result)
    rgb = np.zeros((480, 640, 3), dtype=np.uint8)
    world = SimpleNamespace(to_world=lambda x, y: (x / 10, y / 10))
    with patch.object(config, "FRAME_WIDTH", 1280), patch.object(config, "FRAME_HEIGHT", 720):
        hand = tracker.process(rgb, world).hands[0]
    assert hand.px[0] == (320, 120)
    assert (hand.wrist_x_cm, hand.wrist_y_cm) == (32, 12)


def test_shadow_mode_still_stops_even_with_clear_sector():
    state = _perceive(33.45)
    world = state.world
    world.sectors = dict.fromkeys(world.sectors, "CLEAR")
    world.blocked = SectorState()
    world.vision.detector_available = True  # Exercise the motion-disabled gate itself.
    risk = choose_risk(world, RiskEvaluator().evaluate(world), now=time.monotonic())
    plan = EscapePlanner().plan(world, risk)
    with patch.object(config, "ENABLE_ESCAPE_MOTION", False):
        command = SafetyManager().validate(
            camera_ok=True, world=world, plan=plan,
            telemetry=SimpleNamespace(received=True, age_s=0), risk=risk)
    assert command.status == "STOP"
    assert "motion disabled" in command.reason
