from __future__ import annotations

import time
import threading
import sys
import math
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2

import config
from comm import MonitorSender, TelemetryReceiver, UdpSender
from core.control_loop import ControlLoop, run_monitor_loop
from core.shared_state import SharedState
from fusion import SensorFusion
from perception import NetworkCamera, ObjectDetector, RadarReceiver, RobotTracker, WorldFrame
from perception.robot_tracker import MarkerScanner
from perception.marker_objects import detect_marker_objects
from shared_ai.hand_cup import HandCupEvaluator
from perception.vision_state import build_vision_state
from perception.table_calibration import TableCalibration
from perception.robot_tracker import RobotPose
from perception.vision_state import DIRECTIONS, sector_for
from planning.transform import world_to_robot


def _build_detector():
    try:
        return ObjectDetector()
    except Exception as e:
        print(f"[YOLO] init failed -> detector unavailable: {e}")
        return None


def _wrist_inputs(hands, pose, cup):
    """Use current-frame HandLandmarker wrists; Pose is only a fallback."""
    candidates = []
    for hand in getattr(hands, "hands", ()) if hands is not None else ():
        pixels = getattr(hand, "px", ())
        if not pixels:
            continue
        try:
            px = tuple(pixels[0])  # HandLandmarker landmark 0 = WRIST.
            point = SimpleNamespace(x_cm=float(hand.wrist_x_cm),
                                    y_cm=float(hand.wrist_y_cm), px=px)
            if not all(math.isfinite(v) for v in (point.x_cm, point.y_cm, *px)):
                continue
        except (AttributeError, TypeError, ValueError):
            continue
        candidates.append((str(getattr(hand, "handedness", "?")).upper(), point))
    if candidates:
        # Keep the closest wrist of each labeled side; unlabeled wrists remain
        # candidates, so the evaluator can still select the closest to ID1.
        selected = []
        for side in ("LEFT", "RIGHT"):
            group = [point for name, point in candidates if name == side]
            if group:
                selected.append((side, min(group, key=lambda p:
                    math.hypot(p.x_cm - cup.x_cm, p.y_cm - cup.y_cm))))
        selected.extend((name, point) for name, point in candidates
                        if name not in ("LEFT", "RIGHT"))
        left = next((p for name, p in selected if name == "LEFT"), None)
        right = next((p for name, p in selected if name == "RIGHT"), None)
        human = SimpleNamespace(left_wrist=left, right_wrist=right,
                                wrists=[point for _, point in selected])
        return human, "HAND"
    if pose is not None and getattr(pose, "wrists", ()):
        return pose, "POSE"
    return None, "NONE"


def _hand_debug(state) -> str:
    world = state.world
    vision = getattr(world, "vision", None)
    evidence = getattr(world, "hand_cup", None)
    size = getattr(vision, "frame_size", None)
    threat = "NONE"
    preferred = "NONE"
    if (world is not None and evidence is not None and
            evidence.level in ("WARN", "DANGER") and
            evidence.trigger_wrist_x_cm is not None and
            evidence.trigger_wrist_y_cm is not None):
        dx, dy = world_to_robot(evidence.trigger_wrist_x_cm - world.robot.x_cm,
                                evidence.trigger_wrist_y_cm - world.robot.y_cm,
                                world.robot.heading_rad)
        if math.hypot(dx, dy) > 1e-6:
            threat = sector_for(math.atan2(dy, dx))
            preferred = DIRECTIONS[(DIRECTIONS.index(threat) + 4) % 8]
    def pair(value):
        return "NONE" if value is None else f"({value[0]:.1f},{value[1]:.1f})"
    def number(value):
        return "NONE" if value is None else f"{value:.2f}"
    wrist_cm = (None if evidence is None or evidence.trigger_wrist_x_cm is None else
                (evidence.trigger_wrist_x_cm, evidence.trigger_wrist_y_cm))
    cup_cm = (None if evidence is None or evidence.cup_x_cm is None else
              (evidence.cup_x_cm, evidence.cup_y_cm))
    return (f"[HANDDBG] frame={size[0]}x{size[1]} " if size else "[HANDDBG] frame=NONE ") + (
        f"ID0={bool(world and world.robot.detected)} "
        f"ID1={bool(vision and any(o.detection.label == 'cup' for o in vision.objects))} "
        f"hands={getattr(vision, 'hands_count', 0)} "
        f"wrist_source={getattr(vision, 'wrist_source', 'NONE')} "
        f"wrist_px={pair(getattr(vision, 'trigger_wrist_px', None))} "
        f"wrist_cm={pair(wrist_cm)} cup_cm={pair(cup_cm)} "
        f"distance_cm={number(getattr(evidence, 'distance_cm', None))} "
        f"approach_cm_s={number(getattr(evidence, 'approach_speed_cm_s', None))} "
        f"ttc={number(getattr(evidence, 'ttc_s', None))} "
        f"hand_risk={getattr(evidence, 'level', 'UNKNOWN')} threat={threat} "
        f"escape={getattr(state.plan, 'direction', 'NONE')} preferred_escape={preferred} "
        f"world_valid={bool(world and world.sensor_valid)} "
        f"vision_valid={bool(world and world.vision_valid)} "
        f"detector_available={bool(vision and vision.detector_available)}")


def perception_step(camera, world_frame, robot_tracker, detector, radar, fusion,
                    shared: SharedState, *, scanner=None, pose_tracker=None,
                    hand_tracker=None, gaze_tracker=None, hand_cup_eval=None) -> bool:
    """Acquire a frame and publish only a complete WorldState from that frame."""
    camera_ok = False
    try:
        camera_ok, frame = camera.read()
        now = time.monotonic()
        captured_at = getattr(camera, "last_ok_t", 0.0) or now
        if not camera_ok or frame is None:
            shared.publish_failure("camera offline", camera_ok=False)
            return False
        if captured_at > now or now - captured_at > config.WORLD_STALE_S:
            shared.publish_failure("camera frame stale", camera_ok=False)
            return False
        calibration_valid = True
        if world_frame is not None and getattr(world_frame, "require_calibration", False):
            shape = getattr(frame, "shape", ())
            calibration_valid = (len(shape) >= 2 and
                                 world_frame.validate_resolution(shape[1], shape[0]))
        if not calibration_valid:
            vision = build_vision_state(
                camera_ok=True, captured_at=captured_at, robot=RobotPose(),
                detections=[], detector_available=False, now=time.monotonic(),
                calibration_valid=False,
            )
            radar_state = radar.read()
            world = fusion.build_from_states(vision, radar_state, now=time.monotonic())
            shared.publish_world(world, captured_at=captured_at)
            return True
        scan = scanner.scan(frame) if scanner is not None else None
        robot = (robot_tracker.process(frame, world_frame, scan=scan) if scan is not None
                 else robot_tracker.process(frame, world_frame))
        detector_available = bool(detector and getattr(detector, "available", detector.backend != "stub"))
        try:
            detections = detector.detect(frame, world_frame) if detector else []
        except Exception as exc:
            print(f"[YOLO] inference failed: {exc}")
            detections = []
            detector_available = False
        if detector:
            detector_available = detector_available and getattr(detector, "available", True)
        hand_cup = None
        hand_cup_available = False
        hands_count = 0
        wrist_source = "NONE"
        trigger_wrist_px = None
        if scan is not None:
            marker_objects = detect_marker_objects(scan, world_frame, captured_at=captured_at)
            cup = next((item for item in marker_objects if item.label == "cup"), None)
            # Cup ID 1 is authoritative for hand risk; do not use a YOLO cup
            # when its marker is missing or obscured.
            detections = [item for item in detections if item.role != "cup"] + marker_objects
            if cup is not None and hand_cup_eval is not None:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                try:
                    hands = hand_tracker.process(rgb, world_frame) if hand_tracker else None
                except Exception as exc:
                    print(f"[HAND] inference failed: {exc}")
                    hands = None
                hands_count = len(getattr(hands, "hands", ())) if hands else 0
                try:
                    pose = pose_tracker.process(rgb, world_frame) if pose_tracker else None
                except Exception as exc:
                    print(f"[POSE] inference failed: {exc}")
                    pose = None
                human, wrist_source = _wrist_inputs(hands, pose, cup)
                try:
                    gaze = (gaze_tracker.process(rgb, world_frame,
                                target_cm=(cup.x_cm, cup.y_cm)) if gaze_tracker else None)
                except Exception as exc:
                    print(f"[GAZE] inference failed: {exc}")
                    gaze = None
                hand_cup = hand_cup_eval.update(human, cup, now=captured_at,
                                                 hands=hands, gaze=gaze)
                hand_cup_available = hand_cup.level != "UNKNOWN"
                if hand_cup_available:
                    trigger = min(human.wrists, key=lambda p: math.hypot(
                        p.x_cm - cup.x_cm, p.y_cm - cup.y_cm))
                    trigger_wrist_px = getattr(trigger, "px", None)
        vision = build_vision_state(
            camera_ok=True, captured_at=captured_at, robot=robot,
            detections=detections,
            detector_available=detector_available,
            now=time.monotonic(),
            calibration_valid=True, hand_cup=hand_cup,
            hand_cup_available=hand_cup_available,
        )
        vision.hands_count = hands_count
        vision.wrist_source = wrist_source
        vision.trigger_wrist_px = trigger_wrist_px
        if hasattr(frame, "shape") and len(frame.shape) >= 2:
            vision.frame_size = (frame.shape[1], frame.shape[0])
        radar_state = radar.read()
        world = fusion.build_from_states(vision, radar_state, now=time.monotonic())
        shared.publish_world(world, captured_at=captured_at)
        return True
    except Exception as exc:
        reason = f"perception exception: {exc}"
        print(f"[PERCEPTION] {reason}")
        shared.publish_failure(reason, camera_ok=camera_ok)
        return False


def main() -> None:
    camera = NetworkCamera()
    table_calibration = TableCalibration.load_optional(
        config.TABLE_CALIBRATION_PATH, (config.FRAME_WIDTH, config.FRAME_HEIGHT)
    )
    if (table_calibration.valid and
            table_calibration.aruco_dictionary != config.ARUCO_DICT_NAME):
        table_calibration = TableCalibration.invalid(
            config.FRAME_WIDTH, config.FRAME_HEIGHT,
            reason="calibration ArUco dictionary mismatch",
        )
    world_frame = WorldFrame(calibration=table_calibration, require_calibration=True)
    robot_tracker = RobotTracker()
    scanner = MarkerScanner()
    detector = _build_detector()
    pose_tracker = hand_tracker = gaze_tracker = None
    for name in ("Hand", "Pose", "Gaze"):
        try:
            if name == "Hand":
                from shared_ai.hand_tracker import HandTracker as constructor
            elif name == "Pose":
                from shared_ai.pose_tracker import PoseTracker as constructor
            else:
                from shared_ai.gaze_tracker import GazeTracker as constructor
            tracker = constructor()
            if name == "Hand":
                hand_tracker = tracker
            elif name == "Pose":
                pose_tracker = tracker
            else:
                gaze_tracker = tracker
        except Exception as exc:
            print(f"[{name}] tracker unavailable: {exc}")
    hand_cup_eval = HandCupEvaluator()
    radar = RadarReceiver()
    fusion = SensorFusion()
    shared = SharedState()
    sender = UdpSender()
    telemetry_rx = TelemetryReceiver()
    monitor = MonitorSender()
    controller = ControlLoop(shared, telemetry_rx, sender)
    stopped = threading.Event()
    control_thread = threading.Thread(target=controller.run, args=(stopped,), daemon=True)
    monitor_thread = threading.Thread(
        target=run_monitor_loop, args=(shared, telemetry_rx, monitor, stopped), daemon=True
    )

    last_log = last_hand_debug = 0.0
    print("[PI] central controller started")
    print(f"[PI] camera={config.NETWORK_CAMERA_URL}")
    print(f"[PI] ESP32={config.ESP32_IP}:{config.ESP32_PORT}")
    print(f"[PI] automatic motion enabled={config.ENABLE_ESCAPE_MOTION}")
    if config.ALLOW_MOTION_WITHOUT_RADAR:
        print("[PI] TEST ONLY - RADAR BYPASS ACTIVE")
    print(f"[PI] table calibration valid={world_frame.valid} "
          f"reason={table_calibration.invalid_reason or 'OK'}")

    try:
        control_thread.start()
        monitor_thread.start()
        while True:
            ok = perception_step(camera, world_frame, robot_tracker, detector,
                                 radar, fusion, shared, scanner=scanner,
                                 pose_tracker=pose_tracker, hand_tracker=hand_tracker,
                                 gaze_tracker=gaze_tracker, hand_cup_eval=hand_cup_eval)
            if not ok:
                time.sleep(0.03)
            now = time.monotonic()
            if not config.ENABLE_ESCAPE_MOTION and now - last_hand_debug >= 0.5:
                print(_hand_debug(shared.snapshot()), flush=True)
                last_hand_debug = now
            if now - last_log >= 1.0 / config.PRINT_HZ:
                state = shared.snapshot()
                print(f"[PI] state={state.health.state} cmd={state.command.status} "
                      f"reason={state.command.reason} risk={getattr(state.risk, 'level', '-')}/{getattr(state.risk, 'source', '-')} "
                      f"ID0={bool(state.world and state.world.robot.detected)} "
                      f"ID1={bool(state.world and state.world.vision and any(item.detection.label == 'cup' for item in state.world.vision.objects))} "
                      f"wrist={getattr(getattr(state.world, 'hand_cup', None), 'trigger_wrist_side', None)} "
                      f"hand={getattr(getattr(state.world, 'hand_cup', None), 'distance_cm', None)} "
                      f"threat={getattr(state.risk, 'threat_direction', None)} "
                      f"escape={getattr(state.plan, 'direction', None)} ticks={state.timing.ticks} "
                      f"control_ms={state.timing.last_duration_s * 1000:.1f} "
                      f"missed={state.timing.missed_deadlines}")
                last_log = now
    except KeyboardInterrupt:
        pass
    finally:
        stopped.set()
        control_thread.join(timeout=1)
        monitor_thread.join(timeout=1)
        sender.close()
        monitor.close()
        telemetry_rx.close()
        radar.close()
        for tracker in (pose_tracker, hand_tracker, gaze_tracker):
            if tracker is not None:
                tracker.close()
        camera.close()
        print("[PI] stopped")


if __name__ == "__main__":
    main()
