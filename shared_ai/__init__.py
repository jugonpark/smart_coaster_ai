"""Shared, perception-only MediaPipe trackers adapted from 32200362-sys/grise.

Both the Windows preview and the future Pi perception pipeline import these
trackers. This package never sends commands or decides robot safety states.
"""

__all__ = ["PoseTracker", "HandTracker", "GazeTracker"]


def __getattr__(name):
    if name == "PoseTracker":
        from .pose_tracker import PoseTracker
        return PoseTracker
    if name == "HandTracker":
        from .hand_tracker import HandTracker
        return HandTracker
    if name == "GazeTracker":
        from .gaze_tracker import GazeTracker
        return GazeTracker
    raise AttributeError(name)
