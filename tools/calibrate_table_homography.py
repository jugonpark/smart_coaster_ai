"""Preview and save a 16-corner table homography. This tool never sends UDP."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from perception.marker_scanner import MarkerScanner  # noqa: E402
from perception.table_calibration import (  # noqa: E402
    DICTIONARY, MarkerDefinition, TableCalibration,
)


ROOT = Path(__file__).resolve().parents[1]


def load_layout(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("aruco_dictionary") != DICTIONARY:
        raise ValueError("layout must use DICT_4X4_50")
    markers = [MarkerDefinition.from_json(item) for item in data["markers"]]
    table_size = (float(data["table_width_cm"]), float(data["table_height_cm"]))
    return markers, table_size


def draw_preview(frame, scan, calibration, error=None):
    out = frame.copy()
    for marker_id, corners in scan.by_id.items():
        pts = np.rint(corners).astype(np.int32)
        cv2.polylines(out, [pts], True, (0, 230, 0), 2)
        center = np.rint(pts.mean(axis=0)).astype(int)
        cv2.putText(out, f"ID {marker_id}", tuple(center + (5, -5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 230, 0), 2)
    if calibration is None:
        cv2.putText(out, f"INVALID: {error or 'need IDs 41-44'}", (12, 27),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
        return out
    cv2.putText(out, f"RMS {calibration.rms_error_cm:.2f}cm  "
                f"MAX {calibration.max_error_cm:.2f}cm", (12, 27),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 230, 0), 2)
    half_w = calibration.table_width_cm / 2
    half_h = calibration.table_height_cm / 2
    for x in np.arange(-half_w, half_w + 0.1, 10):
        points = [calibration.table_to_pixel(x, y) for y in np.linspace(-half_h, half_h, 25)]
        cv2.polylines(out, [np.rint(points).astype(np.int32)], False, (100, 100, 100), 1)
    for y in np.arange(-half_h, half_h + 0.1, 10):
        points = [calibration.table_to_pixel(x, y) for x in np.linspace(-half_w, half_w, 25)]
        cv2.polylines(out, [np.rint(points).astype(np.int32)], False, (100, 100, 100), 1)
    origin = tuple(np.rint(calibration.table_to_pixel(0, 0)).astype(int))
    for label, end_cm, color in (("+X", (10, 0), (0, 0, 255)),
                                 ("+Y", (0, 10), (0, 255, 0))):
        end = tuple(np.rint(calibration.table_to_pixel(*end_cm)).astype(int))
        cv2.arrowedLine(out, origin, end, color, 2)
        cv2.putText(out, label, end, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--image", type=Path, help="saved frame from the real camera path")
    source.add_argument("--camera", type=int, help="USB camera index")
    source.add_argument("--url", help="Galaxy HTTP/MJPEG URL")
    parser.add_argument("--config", type=Path,
                        default=ROOT / "calibration/table_layout.json")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "calibration/table_calibration.json")
    parser.add_argument("--no-gui", action="store_true", help="fit and save one frame")
    args = parser.parse_args()
    markers, table_size = load_layout(args.config)
    capture = None
    if args.image:
        frame = cv2.imread(str(args.image))
        if frame is None:
            raise SystemExit(f"cannot read {args.image}")
    else:
        source_value = args.url or os.getenv("GRISE_CAMERA_URL") or (
            args.camera if args.camera is not None else 0)
        capture = cv2.VideoCapture(source_value)
        if not capture.isOpened():
            raise SystemExit(f"camera unavailable: {source_value}")
        ok, frame = capture.read()
        if not ok:
            raise SystemExit("camera frame unavailable")
    scanner = MarkerScanner()
    window = "D.I.G table calibration"
    latest = {"calibration": None}

    def click(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN and latest["calibration"] is not None:
            point = latest["calibration"].pixel_to_table(x, y)
            print(f"pixel ({x},{y}) -> TABLE ({point[0]:+.2f},{point[1]:+.2f}) cm")

    if not args.no_gui:
        cv2.namedWindow(window)
        cv2.setMouseCallback(window, click)
    try:
        while True:
            if capture is not None:
                ok, frame = capture.read()
                if not ok:
                    raise SystemExit("camera frame unavailable")
            scan = scanner.scan(frame)
            try:
                fit = TableCalibration.from_scan(
                    scan, markers, (frame.shape[1], frame.shape[0]), table_size)
                error = None
            except ValueError as exc:
                fit, error = None, str(exc)
            latest["calibration"] = fit
            if args.no_gui:
                if fit is None:
                    raise SystemExit(f"calibration invalid: {error}")
                fit.save(args.output)
                break
            cv2.imshow(window, draw_preview(frame, scan, fit, error))
            key = cv2.waitKey(30 if capture is not None else 0) & 0xff
            if key in (ord("q"), 27):
                break
            if key == ord("s") and fit is not None:
                fit.save(args.output)
                print(f"Saved {args.output} RMS={fit.rms_error_cm:.3f}cm "
                      f"MAX={fit.max_error_cm:.3f}cm")
    finally:
        if capture is not None:
            capture.release()
        if not args.no_gui:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
