import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import ai_overlay_preview as preview


def test_preview_runs_shared_trackers_without_motion(monkeypatch):
    import cv2
    import shared_ai

    calls = []

    class Tracker:
        def __init__(self, name):
            self.name = name

        def process(self, frame, world, *args):
            calls.append((self.name, "process", frame.shape))
            if self.name == "pose":
                return SimpleNamespace(detected=True)
            if self.name == "hand":
                return SimpleNamespace(hands=[], any_grasp_ready=False)
            return SimpleNamespace(detected=False)

        def draw(self, *args):
            calls.append((self.name, "draw"))

        def close(self):
            calls.append((self.name, "close"))

    class Capture:
        def isOpened(self):
            return True

        def read(self):
            return True, np.zeros((480, 640, 3), dtype=np.uint8)

        def release(self):
            calls.append(("capture", "release"))

    monkeypatch.setattr(shared_ai, "PoseTracker", lambda: Tracker("pose"))
    monkeypatch.setattr(shared_ai, "HandTracker", lambda: Tracker("hand"))
    monkeypatch.setattr(shared_ai, "GazeTracker", lambda: Tracker("gaze"))
    monkeypatch.setattr(cv2, "VideoCapture", lambda *args: Capture())
    assert preview.run(preview.parse_args(["--headless", "--max-frames", "1"])) == 0
    assert [call[0] for call in calls if call[1] == "process"] == ["pose", "hand", "gaze"]
    assert ("capture", "release") in calls


def test_preview_coordinates_follow_actual_frame():
    from shared_ai.pose_tracker import PoseTracker

    landmark = SimpleNamespace(x=0.5, y=0.5, visibility=1.0, presence=1.0)
    tracker = PoseTracker.__new__(PoseTracker)
    tracker._last_ts_ms = -1
    tracker._landmarker = SimpleNamespace(
        detect_for_video=lambda frame, timestamp: SimpleNamespace(
            pose_landmarks=[[landmark] * 17]))
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    pose = tracker.process(frame, preview.PreviewWorld(), now=1.0)
    assert pose.left_wrist.px == (320, 240)
    assert pose.left_wrist.x_cm == 320


def test_aruco_4x4_50_marker_id_is_detected_and_drawn():
    import cv2

    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    marker = cv2.aruco.generateImageMarker(dictionary, 0, 180)
    frame = np.full((300, 300, 3), 255, dtype=np.uint8)
    frame[60:240, 60:240] = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
    detector = preview.make_aruco_detector(cv2)
    assert preview.draw_aruco(frame, detector, cv2) == [0]
