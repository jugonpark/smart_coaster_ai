import json
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "raspberry_pi"))
sys.path.insert(0, str(ROOT))

import config
from decision.risk_arbiter import choose_risk
from decision.risk_evaluator import RiskEvaluator, RiskState
from fusion.sensor_fusion import SensorFusion
from core.shared_state import SharedState
from core.control_loop import ControlLoop
from core.motion_goal import MotionGoals
from main import perception_step
from perception.marker_objects import detect_marker_objects
from perception.radar import RadarState, parse_fake_packet
from perception.robot_tracker import MarkerScanner, RobotPose, RobotTracker, WorldFrame
from perception.table_calibration import TableCalibration
from perception.vision_state import build_vision_state
from planning.escape_planner import EscapePlanner
from safety.safety_manager import SafetyManager
from shared_ai.hand_cup import HandCupEvidence, HandCupEvaluator


def test_same_aruco_scan_detects_robot_zero_and_current_cup_one():
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    frame = np.full((480, 640, 3), 255, dtype=np.uint8)
    for marker_id, x in ((0, 80), (1, 360)):
        marker = cv2.aruco.generateImageMarker(dictionary, marker_id, 150)
        frame[150:300, x:x + 150] = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
    scanner = MarkerScanner()
    scan = scanner.scan(frame)
    world = WorldFrame(px_per_cm=10, frame_height=480)
    robot = RobotTracker().process(frame, world, scan=scan)
    cup = next(item for item in detect_marker_objects(scan, world)
               if item.label == "cup")
    assert robot.detected and robot.x_cm < cup.x_cm
    assert cup.radius_cm == 4
    frame[150:300, 360:510] = 255
    assert not any(item.label == "cup" for item in
                   detect_marker_objects(scanner.scan(frame), world))


def _world(level="DANGER", wrist=(20.0, 0.0), radar_targets=(), *, now=None):
    now = time.monotonic() if now is None else now
    robot = RobotPose(True, 0, 0, 0, timestamp=now)
    evidence = HandCupEvidence(level=level, distance_cm=10, ttc_s=None,
                               trigger_wrist_x_cm=wrist[0],
                               trigger_wrist_y_cm=wrist[1],
                               trigger_wrist_side="LEFT", cup_x_cm=30,
                               cup_y_cm=0, timestamp=now, reason="DISTANCE")
    vision = build_vision_state(camera_ok=True, captured_at=now, robot=robot,
                                detections=[], detector_available=True, now=now,
                                hand_cup=evidence)
    radar = parse_fake_packet(json.dumps({"targets": list(radar_targets)}), received_at=now)
    return SensorFusion().build_from_states(vision, radar, now=now), now


def test_hand_danger_without_radar_target_plans_bounded_backwards_move():
    world, now = _world()
    radar_risk = RiskEvaluator().evaluate(world, now=now)
    risk = choose_risk(world, radar_risk, now=now)
    plan = EscapePlanner().plan(world, risk, now=now)
    assert (risk.level, risk.source, risk.threat_direction) == ("DANGER", "HAND", "FRONT")
    assert (plan.direction, plan.status, plan.speed_cm_s, plan.target_distance_cm) == (
        "BACK", "RUN", 15, 15)
    assert plan.vx == -15 and plan.vy == 0


def test_hand_warn_and_grip_stays_warn():
    human = SimpleNamespace(left_wrist=SimpleNamespace(x_cm=25, y_cm=0))
    human.wrists = [human.left_wrist]
    evidence = HandCupEvaluator().update(human, SimpleNamespace(x_cm=0, y_cm=0),
                                          now=1.0,
                                          hands=SimpleNamespace(any_grasp_ready=True),
                                          gaze=SimpleNamespace(looking_at_target=True))
    assert (evidence.level, evidence.trigger_wrist_side, evidence.grip, evidence.gaze) == (
        "WARN", "LEFT", True, True)
    world, now = _world(level="WARN")
    risk = choose_risk(world, RiskEvaluator().evaluate(world, now=now), now=now)
    plan = EscapePlanner().plan(world, risk, now=now)
    assert (plan.status, plan.speed_cm_s, plan.target_distance_cm) == ("SLOW", 10, 10)


def test_primary_trigger_wrist_kept_for_direction_and_radar_priority():
    world, now = _world(wrist=(0, 20))  # selected wrist LEFT of robot
    risk = choose_risk(world, RiskEvaluator().evaluate(world, now=now), now=now)
    assert (risk.source, risk.threat_direction) == ("HAND", "LEFT")
    left = SimpleNamespace(x_cm=20, y_cm=0)  # closer to cup (30, 0)
    right = SimpleNamespace(x_cm=0, y_cm=5)  # closer to robot (0, 0)
    human = SimpleNamespace(left_wrist=left, right_wrist=right, wrists=[left, right])
    world.hand_cup = HandCupEvaluator().update(human,
        SimpleNamespace(x_cm=30, y_cm=0), now=now)
    risk = choose_risk(world, RiskState("SAFE", timestamp=now), now=now)
    assert (world.hand_cup.trigger_wrist_side, risk.threat_direction) == ("LEFT", "FRONT")
    radar_world, now = _world(level="WARN", radar_targets=[
        {"distance_cm": 10, "angle_deg": 0, "approach_speed_cm_s": 0}], now=now)
    radar_risk = RiskEvaluator().evaluate(radar_world, now=now)
    assert choose_risk(radar_world, radar_risk, now=now).source == "RADAR"


def test_unknown_hand_does_not_downgrade_radar_and_ties_use_ttc_then_distance():
    world, now = _world()
    world.hand_cup = HandCupEvidence(timestamp=now)
    radar = RiskState("DANGER", distance_cm=20, ttc_s=0.4,
                      threat_direction="FRONT", timestamp=now)
    assert choose_risk(world, radar, now=now) is radar
    world.hand_cup = HandCupEvidence(level="DANGER", distance_cm=10, ttc_s=0.3,
        trigger_wrist_x_cm=0, trigger_wrist_y_cm=20, timestamp=now, reason="TTC")
    assert choose_risk(world, radar, now=now).source == "HAND"
    radar.ttc_s = 0.2
    assert choose_risk(world, radar, now=now).source == "RADAR"


def test_radar_obstacle_blocks_opposite_hand_escape_sector():
    world, now = _world(radar_targets=[
        {"distance_cm": 20, "angle_deg": 180, "approach_speed_cm_s": 0}])
    risk = choose_risk(world, RiskEvaluator().evaluate(world, now=now), now=now)
    assert risk.source == "HAND"
    assert world.sectors["BACK"] == "BLOCKED"
    assert EscapePlanner().plan(world, risk, now=now).direction == "BACK_LEFT"


def test_blocked_unknown_stale_and_motion_disabled_stop():
    world, now = _world()
    risk = choose_risk(world, RiskEvaluator().evaluate(world, now=now), now=now)
    with patch.object(config, "ENABLE_ESCAPE_MOTION", True):
        for direction in world.sectors:
            world.sectors[direction] = "UNKNOWN"
        assert EscapePlanner().plan(world, risk, now=now).status == "STOP"
        for direction in world.sectors:
            world.sectors[direction] = "BLOCKED"
        assert EscapePlanner().plan(world, risk, now=now).status == "STOP"
        for direction in world.sectors:
            world.sectors[direction] = "CLEAR"
        plan = EscapePlanner().plan(world, risk, now=now)
        fresh = SimpleNamespace(received=True, age_s=0)
        assert SafetyManager().validate(camera_ok=True, world=world, plan=plan,
                                        telemetry=fresh, risk=risk, now=now).status == "RUN"
        assert SafetyManager().validate(camera_ok=True, world=world, plan=plan,
                                        telemetry=SimpleNamespace(received=True, age_s=10),
                                        risk=risk, now=now).status == "STOP"
    assert SafetyManager().validate(camera_ok=True, world=world, plan=plan,
                                    telemetry=fresh, risk=risk, now=now).status == "STOP"
    stale = HandCupEvidence(level="DANGER", distance_cm=10,
                            trigger_wrist_x_cm=20, trigger_wrist_y_cm=0,
                            timestamp=now - 1)
    world.hand_cup = stale
    assert choose_risk(world, RiskState("SAFE"), now=now).level == "INVALID"


def test_test_only_radar_bypass_requires_hand_and_valid_vision():
    radar_world, now = _world()
    with patch.object(config, "ALLOW_MOTION_WITHOUT_RADAR", True), patch.object(
            config, "ENABLE_ESCAPE_MOTION", True):
        world = SensorFusion().build_from_states(radar_world.vision, RadarState(), now=now)
        assert world.sensor_valid and world.vision_valid and not world.radar_valid
        risk = choose_risk(world, RiskEvaluator().evaluate(world, now=now), now=now)
        plan = EscapePlanner().plan(world, risk, now=now)
        assert risk.source == "HAND" and plan.status == "RUN"
        assert SafetyManager().validate(camera_ok=True, world=world, plan=plan,
                telemetry=SimpleNamespace(received=True, age_s=0), risk=risk, now=now).status == "RUN"
        world.vision_valid = False
        assert SafetyManager().validate(camera_ok=True, world=world, plan=plan,
                telemetry=SimpleNamespace(received=True, age_s=0), risk=risk, now=now).status == "STOP"


def test_current_frame_markers_and_wrist_reach_worldstate():
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    frame = np.full((480, 640, 3), 255, dtype=np.uint8)
    for marker_id, x in ((0, 80), (1, 360)):
        marker = cv2.aruco.generateImageMarker(dictionary, marker_id, 150)
        frame[150:300, x:x + 150] = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
    matrix = np.array([[0.1, 0, 0], [0, -0.1, 48], [0, 0, 1]], dtype=float)
    calibration = TableCalibration(640, 480, matrix)
    world_frame = WorldFrame(calibration=calibration, require_calibration=True)
    camera = SimpleNamespace(last_ok_t=time.monotonic(), read=lambda: (True, frame))
    radar = SimpleNamespace(read=lambda: parse_fake_packet(
        json.dumps({"targets": []}), received_at=time.monotonic()))
    human = SimpleNamespace(wrists=[SimpleNamespace(
        x_cm=33.45, y_cm=25.55, px=(334.5, 224.5))])
    pose = SimpleNamespace(process=lambda rgb, world: human)
    shared = SharedState()
    assert perception_step(camera, world_frame, RobotTracker(), None, radar,
                           SensorFusion(), shared, scanner=MarkerScanner(),
                           pose_tracker=pose, hand_cup_eval=HandCupEvaluator())
    state = shared.snapshot()
    assert state.world.sensor_valid
    assert state.world.hand_cup.level == "DANGER"
    assert state.world.robot.detected
    assert any(obj.detection.label == "cup" for obj in state.world.vision.objects)
    frame[150:300, 360:510] = 255
    assert perception_step(camera, world_frame, RobotTracker(), None, radar,
                           SensorFusion(), shared, scanner=MarkerScanner(),
                           pose_tracker=pose, hand_cup_eval=HandCupEvaluator())
    state = shared.snapshot()
    assert not state.world.vision_valid
    assert state.world.hand_cup is None


def test_hand_risk_reaches_existing_bounded_motion_goal_and_stop_gate():
    world, now = _world()
    shared = SharedState()
    shared.publish_world(world, captured_at=now)
    packets = []

    class Sender:
        def send(self, vx, vy, w, status, force=False, **kwargs):
            packets.append((vx, vy, w, status, kwargs))

    telemetry = SimpleNamespace(received=True, age_s=0, boot_id=1,
                                motion_id=None, goal_active=False,
                                goal_reached=False, status="STOP")
    receiver = SimpleNamespace(latest=lambda: telemetry)
    loop = ControlLoop(shared, receiver, Sender())
    loop.motion_goals = MotionGoals(initial_id=100)
    with patch.object(config, "ENABLE_ESCAPE_MOTION", True):
        assert loop.tick(now=now).status == "STOP"  # boot STOP handshake
        second = loop.tick()
        assert second.status == "RUN", second.reason
        move = packets[-1]
        assert move[0:4] == (-15, 0, 0, "RUN")
        assert move[4] == {"motion_id": 101, "target_distance_cm": 15}
        assert loop.tick().status == "RUN"
        assert packets[-1][4]["motion_id"] == 101
        telemetry.motion_id = 101
        telemetry.goal_reached = True
        assert loop.tick().status == "STOP"
        assert loop.tick().status == "STOP"
        shared.publish_world(world, captured_at=time.monotonic())
        assert loop.tick().status == "RUN"
        assert packets[-1][4]["motion_id"] == 102
    assert packets[-1][3] == "RUN"
