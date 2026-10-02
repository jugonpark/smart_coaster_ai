"""Current-frame ArUco objects in the calibrated table frame."""

from __future__ import annotations

import time

import config
from perception.object_detector import Detection
from perception.robot_tracker import MarkerScan


def detect_marker_objects(scan: MarkerScan, world, *, captured_at: float | None = None):
    timestamp = time.monotonic() if captured_at is None else captured_at
    out = []
    markers = {config.CUP_MARKER_ID: ("cup", config.CUP_RADIUS_CM)}
    markers.update({mid: ("obstacle", radius)
                    for mid, radius in config.MARKER_OBSTACLE_RADII_CM.items()})
    for marker_id, (label, radius) in markers.items():
        corners = scan.get(marker_id)
        if corners is None:
            continue
        cx, cy = MarkerScan.center_px(corners)
        x_cm, y_cm = world.to_world(cx, cy)
        out.append(Detection(label, 1.0, cx, cy, 0.0, 0.0,
                             x_cm=x_cm, y_cm=y_cm, radius_cm=radius,
                             timestamp=timestamp))
    return out
