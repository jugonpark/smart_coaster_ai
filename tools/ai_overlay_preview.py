"""Preview Pose -> Hand -> Gaze and ArUco IDs on a Galaxy MJPEG feed.

This program only reads camera frames and displays overlays. It does not
import the Pi controller, open UDP sockets, or issue ESP32 commands.

PowerShell:
    $env:GRISE_CAMERA_URL = "http://10.232.69.154:8080/"
    python tools/ai_overlay_preview.py
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class PreviewWorld:
    """Image coordinates only; no physical cm calibration is claimed here."""

    @staticmethod
    def to_world(x: float, y: float) -> tuple[float, float]:
        return x, -y


def make_aruco_detector(cv2):
    if not hasattr(cv2, "aruco") or not hasattr(cv2.aruco, "ArucoDetector"):
        raise RuntimeError("OpenCV ArUco is unavailable; install opencv-contrib-python")
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    return cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())


def draw_aruco(frame, detector, cv2) -> list[int]:
    corners, ids, _ = detector.detectMarkers(frame)
    if ids is None:
        return []
    cv2.aruco.drawDetectedMarkers(frame, corners, ids)
    return [int(marker_id) for marker_id in ids.flatten()]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=os.getenv("GRISE_CAMERA_URL", "http://10.232.69.154:8080/"),
                        help="Galaxy MJPEG URL (also GRISE_CAMERA_URL)")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="Exit after N frames (0: run until q/Esc)")
    parser.add_argument("--headless", action="store_true",
                        help="Process frames without opening a display window")
    return parser.parse_args(argv)


def run(args) -> int:
    import cv2
    from shared_ai import GazeTracker, HandTracker, PoseTracker

    if args.max_frames < 0:
        raise ValueError("--max-frames must be >= 0")

    trackers = []
    capture = None
    try:
        aruco = make_aruco_detector(cv2)
        # Fail clearly if any required model is absent; never silently show a
        # camera-only preview as though all three AI stages were active.
        pose = PoseTracker(); trackers.append(pose)
        hand = HandTracker(); trackers.append(hand)
        gaze = GazeTracker(); trackers.append(gaze)
        capture = cv2.VideoCapture(
            args.url, cv2.CAP_FFMPEG,
            [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000,
             cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000],
        )
        if not capture.isOpened():
            raise RuntimeError(f"camera unavailable: {args.url}")
        world = PreviewWorld()
        count = 0
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                raise RuntimeError("Galaxy MJPEG frame unavailable")
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            human = pose.process(rgb, world)
            hands = hand.process(rgb, world)
            gaze_info = gaze.process(rgb, world)
            pose.draw(frame, human)
            hand.draw(frame, hands)
            gaze.draw(frame, gaze_info, world)
            marker_ids = draw_aruco(frame, aruco, cv2)
            cv2.putText(frame, f"ArUco 4X4_50 IDs: {marker_ids or '-'}",
                        (10, frame.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (0, 255, 0), 2)
            count += 1
            if count == 1 or count % 30 == 0:
                print(f"[AI] frames={count} pose={human.detected} "
                      f"hands={len(hands.hands)} gaze={gaze_info.detected} "
                      f"grip_evidence={hands.any_grasp_ready} "
                      f"aruco_ids={marker_ids}", flush=True)
            if not args.headless:
                cv2.imshow("GRISE AI Preview - Pose / Hand / Gaze / ArUco", frame)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
            if args.max_frames and count >= args.max_frames:
                break
        return 0
    finally:
        if capture is not None:
            capture.release()
        for tracker in reversed(trackers):
            tracker.close()
        if not args.headless:
            cv2.destroyAllWindows()


def main(argv=None) -> int:
    try:
        return run(parse_args(argv))
    except (RuntimeError, ValueError) as exc:
        print(f"[AI] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
