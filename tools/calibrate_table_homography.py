#!/usr/bin/env python3
"""Build and inspect a TABLE-plane homography from four or more ArUco markers.

Examples:
  python tools/calibrate_table_homography.py --image table.jpg
  python tools/calibrate_table_homography.py --camera 0

Click the preview to print Pixel -> TABLE coordinates. Press S to save and Q
to quit. The camera and marker layout must remain fixed after calibration.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "raspberry_pi"))

from perception.robot_tracker import MarkerScan, MarkerScanner
from perception.table_calibration import MarkerDefinition, TableCalibration


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--image", type=Path, help="saved calibration image")
    source.add_argument("--camera", type=int, default=None, help="local camera index")
    parser.add_argument("--config", type=Path,
                        default=ROOT / "calibration" / "table_calibration.example.json")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "calibration" / "table_calibration.json")
    parser.add_argument("--no-gui", action="store_true", help="calculate and save without preview")
    return parser.parse_args()


def load_layout(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    markers = [MarkerDefinition.from_json(item) for item in data["markers"]]
    if len(markers) < 4 or len({item.marker_id for item in markers}) != len(markers):
        raise ValueError("config needs at least four unique marker IDs")
    return markers, data.get("aruco_dictionary", "DICT_4X4_50")


def draw_overlay(frame, scan, calibration, markers):
    out = frame.copy()
    marker_by_id = {marker.marker_id: marker for marker in markers}
    for marker_id, corners in scan.by_id.items():
        points = np.rint(corners).astype(np.int32)
        color = (0, 220, 0) if marker_id in marker_by_id else (0, 180, 255)
        cv2.polylines(out, [points], True, color, 2)
        center = tuple(np.rint(np.mean(corners, axis=0)).astype(int))
        label = f"ID {marker_id}"
        if marker_id in marker_by_id:
            marker = marker_by_id[marker_id]
            label += f" ({marker.x_cm:+.1f},{marker.y_cm:+.1f})cm"
        cv2.circle(out, center, 4, color, -1)
        cv2.putText(out, label, (center[0] + 5, center[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    if calibration is None:
        cv2.putText(out, "CALIBRATION INVALID - need 4 configured markers", (15, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2, cv2.LINE_AA)
        return out

    cv2.putText(out, f"VALID RMS={calibration.rms_error_cm:.3f}cm "
                f"MAX={calibration.max_error_cm:.3f}cm", (15, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 0), 2, cv2.LINE_AA)
    xs = [marker.x_cm for marker in markers]
    ys = [marker.y_cm for marker in markers]
    grid = 10.0
    for x in np.arange(np.floor(min(xs) / grid) * grid,
                       np.ceil(max(xs) / grid) * grid + 0.1, grid):
        points = [calibration.table_to_pixel(x, y) for y in np.linspace(min(ys), max(ys), 20)]
        cv2.polylines(out, [np.rint(points).astype(np.int32)], False, (90, 90, 90), 1)
    for y in np.arange(np.floor(min(ys) / grid) * grid,
                       np.ceil(max(ys) / grid) * grid + 0.1, grid):
        points = [calibration.table_to_pixel(x, y) for x in np.linspace(min(xs), max(xs), 20)]
        cv2.polylines(out, [np.rint(points).astype(np.int32)], False, (90, 90, 90), 1)
    origin = tuple(np.rint(calibration.table_to_pixel(0, 0)).astype(int))
    x_axis = tuple(np.rint(calibration.table_to_pixel(15, 0)).astype(int))
    y_axis = tuple(np.rint(calibration.table_to_pixel(0, 15)).astype(int))
    cv2.arrowedLine(out, origin, x_axis, (0, 0, 255), 2, tipLength=0.2)
    cv2.arrowedLine(out, origin, y_axis, (0, 255, 0), 2, tipLength=0.2)
    cv2.putText(out, "+X", x_axis, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(out, "+Y", y_axis, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return out


def calculate(frame, scanner, markers, dictionary):
    scan = scanner.scan(frame)
    try:
        result = TableCalibration.from_marker_scan(
            scan, markers, frame.shape[1], frame.shape[0], aruco_dictionary=dictionary)
    except ValueError:
        result = None
    return scan, result


def main():
    args = arguments()
    markers, dictionary = load_layout(args.config)
    scanner = MarkerScanner(dictionary_name=dictionary)
    capture = None
    if args.image:
        frame = cv2.imread(str(args.image))
        if frame is None:
            raise SystemExit(f"cannot read image: {args.image}")
    else:
        capture = cv2.VideoCapture(0 if args.camera is None else args.camera)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        ok, frame = capture.read()
        if not ok:
            raise SystemExit("camera frame unavailable")

    latest = {"calibration": None}
    window = "GRISE table calibration"

    def click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and latest["calibration"]:
            table = latest["calibration"].pixel_to_table(x, y)
            print(f"Pixel: ({x}, {y})  Table: ({table[0]:+.2f} cm, {table[1]:+.2f} cm)")

    if not args.no_gui:
        cv2.namedWindow(window)
        cv2.setMouseCallback(window, click)
    try:
        while True:
            if capture is not None:
                ok, frame = capture.read()
                if not ok:
                    continue
            scan, result = calculate(frame, scanner, markers, dictionary)
            latest["calibration"] = result
            if args.no_gui:
                if result is None:
                    raise SystemExit("calibration invalid: four configured markers not detected")
                result.save(args.output)
                break
            cv2.imshow(window, draw_overlay(frame, scan, result, markers))
            key = cv2.waitKey(30 if capture is not None else 0) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                if result is None:
                    print("Not saved: calibration invalid")
                else:
                    result.save(args.output)
                    print(f"Saved {args.output} RMS={result.rms_error_cm:.3f}cm "
                          f"MAX={result.max_error_cm:.3f}cm")
            if args.image and key != 255:
                continue
    finally:
        if capture is not None:
            capture.release()
        if not args.no_gui:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
