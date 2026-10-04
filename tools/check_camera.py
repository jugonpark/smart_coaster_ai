"""
카메라 + 인식 계층 단계별 진단 도구.

main.py는 모델이 하나라도 없으면 바로 종료된다. 이 도구는 반대로,
없는 계층은 끄고 나머지만 돌려서 "어디까지 되는지"를 눈으로 확인하게 해준다.

확인할 수 있는 것:
  - 웹캠이 열리는지, 실제 해상도가 config 설정과 맞는지
  - ArUco 마커가 잡히는지, ID / heading / px_per_cm 스케일이 맞는지
  - 사람 관절(손목/팔꿈치/어깨)이 잡히는지
  - YOLO가 컵/장애물을 잡는지
  - 각 계층이 몇 ms씩 먹는지 (느리면 어디가 범인인지)

UDP는 전혀 보내지 않는다. 로봇이 연결돼 있어도 움직이지 않는다.

사용법:
    python tools/check_camera.py
    python tools/check_camera.py --camera 1     # 다른 카메라로

키:
    1 : ArUco 켜기/끄기
    2 : Pose 켜기/끄기
    3 : YOLO 켜기/끄기
    t : 설정 패널 열기/닫기 (화면 우상단 SETTINGS 버튼 클릭도 동일)
    s : 현재 화면을 PNG로 저장
    q : 종료
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import cv2  # noqa: E402
import numpy as np  # noqa: E402

import config  # noqa: E402
from ui import SettingsPanel, load_tuning  # noqa: E402

GREEN = (0, 220, 0)
RED = (0, 0, 255)
GRAY = (160, 160, 160)
WHITE = (255, 255, 255)
YELLOW = (0, 255, 255)

WIN = "D.I.G - camera check"


def _in_rect(x, y, rect) -> bool:
    x1, y1, x2, y2 = rect
    return x1 <= x <= x2 and y1 <= y <= y2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=config.CAMERA_INDEX)
    args = ap.parse_args()

    print("=" * 60)
    print("카메라 / 인식 계층 진단")
    print("=" * 60)

    load_tuning()

    # ---------------- 카메라 ----------------
    config.CAMERA_INDEX = args.camera
    print(f"\n[카메라] {config.CAMERA_URL or args.camera} 여는 중...")
    from perception import Camera
    try:
        camera = Camera()
    except (RuntimeError, ValueError) as exc:
        print(f"  실패: {exc}")
        print("  - 다른 프로그램(줌/팀즈 등)이 카메라를 쓰고 있지 않은지 확인")
        print("  - Windows 설정 > 개인 정보 > 카메라 > 데스크톱 앱 액세스 허용 확인")
        print("  - 다른 인덱스로 시도:  python tools/check_camera.py --camera 1")
        return 1

    ok, frame = camera.read()
    if not ok:
        print("  실패: 카메라는 열렸지만 프레임을 읽지 못했습니다.")
        camera.release()
        return 1

    actual_h, actual_w = frame.shape[:2]
    print(f"  OK. 실제 해상도 {actual_w}x{actual_h}")
    if (actual_w, actual_h) != (config.FRAME_WIDTH, config.FRAME_HEIGHT):
        print(f"  ※ config 설정({config.FRAME_WIDTH}x{config.FRAME_HEIGHT})과 다릅니다.")
        print(f"     config.py의 FRAME_WIDTH/HEIGHT를 {actual_w}/{actual_h}로 바꾸세요.")
        print("     (좌표 변환이 이 값을 쓰기 때문에 안 맞으면 위치가 전부 틀어집니다)")

    # ---------------- 각 계층 로드 (실패해도 계속) ----------------
    world = camera.world
    world.observe_frame(frame.shape)
    if not world.valid:
        print(f"  calibration invalid: {world.invalid_reason}")
        camera.release()
        return 1

    robot_tracker = None
    try:
        from perception import RobotTracker
        robot_tracker = RobotTracker()
        print("[ArUco] OK")
    except Exception as e:
        print(f"[ArUco] 사용 불가: {e}")

    pose_tracker = None
    try:
        from perception import PoseTracker
        pose_tracker = PoseTracker()
    except Exception as e:
        print(f"[Pose] 사용 불가: {str(e).splitlines()[0]}")

    hand_tracker = None
    if config.USE_HAND:
        try:
            from perception import HandTracker
            hand_tracker = HandTracker()
        except Exception as e:
            print(f"[Hand] 사용 불가: {str(e).splitlines()[0]}")
            print("       python tools/download_models.py hand")

    gaze_tracker = None
    if config.USE_GAZE:
        try:
            from perception import GazeTracker
            gaze_tracker = GazeTracker()
        except Exception as e:
            print(f"[Gaze] 사용 불가: {str(e).splitlines()[0]}")
            print("       python tools/download_models.py face")

    from perception import MarkerObjectDetector, MarkerScanner
    scanner = MarkerScanner()
    marker_detector = MarkerObjectDetector()

    detector = None
    try:
        from perception import ObjectDetector
        detector = ObjectDetector()
    except Exception as e:
        print(f"[YOLO] 사용 불가: {str(e).splitlines()[0]}")
        print("       테스트 모드(ArUco)로 물체를 인식합니다. 설정 '모드' 탭에서 전환 가능.")
        config.TEST_MODE_ARUCO = True

    use_aruco = robot_tracker is not None
    use_pose = pose_tracker is not None
    use_yolo = detector is not None
    use_hand = hand_tracker is not None
    use_gaze = gaze_tracker is not None

    panel = SettingsPanel()
    btn_rect = (actual_w - 150, 8, actual_w - 10, 44)
    mouse = {"hover": False}

    def on_mouse(event, x, y, flags, _p):
        mouse["hover"] = _in_rect(x, y, btn_rect)
        if event == cv2.EVENT_LBUTTONDOWN and mouse["hover"]:
            panel.toggle()

    cv2.namedWindow(WIN)
    cv2.setMouseCallback(WIN, on_mouse)

    print("\n키:  1=ArUco  2=Pose  3=YOLO  4=Hand  5=Gaze  t=설정  s=저장  q=종료\n")

    fps = 0.0
    last_t = time.time()
    shot_n = 0

    while True:
        ok, frame = camera.read()
        if not ok:
            print("프레임 읽기 실패")
            break

        if config.FLIP_HORIZONTAL:
            frame = cv2.flip(frame, 1)
        world.observe_frame(frame.shape)
        if not world.valid:
            print(f"calibration invalid: {world.invalid_reason}")
            break

        now = time.time()
        dt = now - last_t
        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt)
        last_t = now

        lines = []

        # ---- ArUco (스캔은 한 번, 로봇 추적과 물체 검출이 공유) ----
        scan = scanner.scan(frame)

        if use_aruco:
            t0 = time.time()
            robot = robot_tracker.process(frame, world, scan)
            ms = (time.time() - t0) * 1000
            if robot.detected:
                robot_tracker.draw(frame, robot, world)
                import math
                lines.append((f"ArUco  OK   id={config.ROBOT_MARKER_ID} "
                              f"pos=({robot.x_cm:.1f},{robot.y_cm:.1f})cm "
                              f"heading={math.degrees(robot.heading_rad):+.0f}deg  {ms:.0f}ms", GREEN))
            else:
                lines.append((f"ArUco  marker not found (ID {config.ROBOT_MARKER_ID})  {ms:.0f}ms", RED))
        else:
            lines.append(("ArUco  off", GRAY))

        # Pose/Hand/Gaze는 같은 RGB 프레임을 쓴다. 변환을 한 번만 한다.
        rgb = None
        if use_pose or use_hand or use_gaze:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # ---- Pose ----
        if use_pose:
            t0 = time.time()
            human = pose_tracker.process(rgb, world, now)
            ms = (time.time() - t0) * 1000
            if human.detected:
                pose_tracker.draw(frame, human)
                n_w = len(human.wrists)
                for j in human.wrists:
                    cv2.circle(frame, (int(j.px[0]), int(j.px[1])), 12, YELLOW, 2)
                lines.append((f"Pose   OK   wrists {n_w}  joints {len(human.repulsion_points)}  {ms:.0f}ms", GREEN))
            else:
                lines.append((f"Pose   no person  {ms:.0f}ms", RED))
        else:
            lines.append(("Pose   off", GRAY))

        # ---- Hand (스켈레톤 + 그립) ----
        if use_hand:
            t0 = time.time()
            hands = hand_tracker.process(rgb, world, now)
            ms = (time.time() - t0) * 1000
            if hands.detected:
                hand_tracker.draw(frame, hands)
                parts = []
                for h in hands.hands:
                    if h.valid:
                        parts.append(f"{h.handedness[:1]}(ap{h.aperture:.2f} "
                                     f"op{h.openness:.2f}{' GRASP' if h.grasp_ready else ''})")
                lines.append((f"Hand   hands {len(hands.hands)}  "
                              f"{'  '.join(parts) if parts else 'no metrics'}  {ms:.0f}ms",
                              (0, 200, 255) if hands.any_grasp_ready else GREEN))
            else:
                lines.append((f"Hand   no hand  {ms:.0f}ms", RED))
        else:
            lines.append(("Hand   off (needs models/hand_landmarker.task)", GRAY))

        # ---- Gaze (머리 방향) ----
        if use_gaze:
            t0 = time.time()
            # 컵 위치를 알면 '컵을 보는지'까지 판정한다. 이 단계에선 화면 중앙을 임시 목표로.
            target = None
            gaze = gaze_tracker.process(rgb, world, target, now)
            ms = (time.time() - t0) * 1000
            if gaze.detected:
                gaze_tracker.draw(frame, gaze, world)
                lines.append((f"Gaze   yaw {gaze.yaw_deg:+.0f}  pitch {gaze.pitch_deg:+.0f}  "
                              f"roll {gaze.roll_deg:+.0f}  {ms:.0f}ms", GREEN))
                lines.append(("       if arrow points backwards, set GAZE_FORWARD_SIGN=-1",
                              (150, 150, 150)))
            else:
                lines.append((f"Gaze   no face  {ms:.0f}ms", RED))
        else:
            lines.append(("Gaze   off (needs models/face_landmarker.task)", GRAY))

        # ---- 물체 인식 (YOLO 또는 테스트 모드 ArUco) ----
        use_marker_obj = config.TEST_MODE_ARUCO or detector is None
        if use_marker_obj or use_yolo:
            src = marker_detector if use_marker_obj else detector
            name = "ArUco " if use_marker_obj else "YOLO  "
            t0 = time.time()
            dets = src.detect(frame, world, scan=scan, now=now)
            ms = (time.time() - t0) * 1000
            cup = src.pick_cup(dets)
            obs = src.pick_obstacles(dets)

            for d in dets:
                x1 = int(d.cx_px - d.w_px / 2); y1 = int(d.cy_px - d.h_px / 2)
                x2 = int(d.cx_px + d.w_px / 2); y2 = int(d.cy_px + d.h_px / 2)
                col = YELLOW if d.role == "cup" else (80, 80, 255) if d.role == "obstacle" else GRAY
                cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
                cv2.putText(frame, f"{d.label} r={d.radius_cm:.0f}cm", (x1, y1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)

            if use_marker_obj:
                scanner.draw(frame, scan)
                marker_detector.draw(frame, world, dets)

            col = GREEN if dets else RED
            lines.append((f"{name} detections {len(dets)} (cup={'O' if cup else 'X'} "
                          f"obstacle={len(obs)})  {ms:.0f}ms", col))
            if use_marker_obj:
                lines.append((f"       registered IDs {sorted(config.MARKER_OBJECTS)} / "
                              f"visible markers {scan.ids}", (150, 150, 150)))
            elif dets and cup is None and not obs:
                lines.append(("       detected but not classified as cup/obstacle -> "
                              "check config.CLASS_CUP name", YELLOW))
        else:
            lines.append(("YOLO   off", GRAY))

        # ---- HUD ----
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], 40 + 26 * len(lines)), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.45, frame, 0.55, 0)

        cv2.putText(frame, f"{frame.shape[1]}x{frame.shape[0]}  {fps:.1f}fps  "
                           "TABLE VALID",
                    (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 1)
        y = 56
        for text, col in lines:
            cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 1)
            y += 26

        cv2.putText(frame, "1=ArUco 2=Pose 3=YOLO  t=settings  s=save  q=quit",
                    (10, frame.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, GRAY, 1)

        # 설정 버튼
        x1, y1, x2, y2 = btn_rect
        cv2.rectangle(frame, (x1, y1), (x2, y2),
                      (110, 110, 110) if mouse["hover"] else (70, 70, 70), -1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (200, 200, 200), 1)
        cv2.putText(frame, "SETTINGS [T]", (x1 + 10, y2 - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)

        # 설정 패널: 바뀐 값 반영 + 현재 상태 전송
        # 키/값 모두 영문으로. readout 라벨 폰트(Consolas)에는 한글 글리프가 없다.
        panel.pump({
            "resolution": f"{actual_w}x{actual_h}",
            "FPS": f"{fps:.1f}",
            "scale": "TABLE homography",
            **{f"L{i + 1}": t for i, (t, _c) in enumerate(lines)},
        })

        cv2.imshow(WIN, frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key == ord('t'):
            panel.toggle()
        elif key == ord('1') and robot_tracker is not None:
            use_aruco = not use_aruco
        elif key == ord('2') and pose_tracker is not None:
            use_pose = not use_pose
        elif key == ord('3') and detector is not None:
            use_yolo = not use_yolo
        elif key == ord('4') and hand_tracker is not None:
            use_hand = not use_hand
        elif key == ord('5') and gaze_tracker is not None:
            use_gaze = not use_gaze
        elif key == ord('s'):
            shot_n += 1
            p = Path(f"check_{shot_n:02d}.png")
            cv2.imwrite(str(p), frame)
            print(f"저장: {p.resolve()}")

    panel.close()
    camera.release()
    cv2.destroyAllWindows()
    for t in (pose_tracker, hand_tracker, gaze_tracker):
        if t is not None:
            t.close()
    print("종료.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
