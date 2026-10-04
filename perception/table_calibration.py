"""Planar pixel <-> table-centimetre calibration from four ArUco markers."""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np


MARKER_IDS = (41, 42, 43, 44)
DICTIONARY = "DICT_4X4_50"
_EPS = 1e-12


def _point_transform(points, matrix):
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise ValueError("points must be finite Nx2")
    projected = np.column_stack((points, np.ones(len(points)))) @ matrix.T
    if not np.isfinite(projected).all() or np.any(np.abs(projected[:, 2]) <= _EPS):
        raise ValueError("invalid homogeneous point")
    result = projected[:, :2] / projected[:, 2, None]
    if not np.isfinite(result).all():
        raise ValueError("non-finite transformed point")
    return result


def _matrix(value):
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (3, 3) or not np.isfinite(result).all():
        raise ValueError("homography must be a finite 3x3 matrix")
    if abs(float(np.linalg.det(result))) <= _EPS:
        raise ValueError("singular homography")
    return result


@dataclass(frozen=True)
class MarkerDefinition:
    id: int
    x_cm: float
    y_cm: float
    size_cm: float
    orientation_deg: float = 0.0

    def __post_init__(self):
        if (type(self.id) is not int or self.id not in MARKER_IDS or
                not all(math.isfinite(v) for v in
                        (self.x_cm, self.y_cm, self.size_cm, self.orientation_deg)) or
                self.size_cm <= 0):
            raise ValueError("invalid calibration marker")

    def table_corners(self):
        half = self.size_cm / 2
        local = np.array([[-half, half], [half, half],
                          [half, -half], [-half, -half]], dtype=np.float64)
        a = math.radians(self.orientation_deg)
        rotate = np.array([[math.cos(a), -math.sin(a)],
                           [math.sin(a), math.cos(a)]])
        return local @ rotate.T + (self.x_cm, self.y_cm)

    @classmethod
    def from_json(cls, data):
        return cls(int(data["id"]), float(data["x_cm"]), float(data["y_cm"]),
                   float(data["size_cm"]), float(data["orientation_deg"]))

    def to_json(self):
        return dict(id=self.id, x_cm=self.x_cm, y_cm=self.y_cm,
                    size_cm=self.size_cm, orientation_deg=self.orientation_deg)


class TableCalibration:
    VERSION = 1

    def __init__(self, image_size, table_size, markers, H, inverse=None,
                 rms_error_cm=0.0, max_error_cm=0.0, correspondence_count=16,
                 created_at=None):
        self.image_width, self.image_height = image_size
        self.table_width_cm, self.table_height_cm = table_size
        if (type(self.image_width) is not int or type(self.image_height) is not int or
                self.image_width <= 0 or self.image_height <= 0 or
                not all(math.isfinite(v) and v > 0 for v in table_size)):
            raise ValueError("invalid image or table size")
        self.markers = tuple(markers)
        if len(self.markers) != 4 or sorted(m.id for m in self.markers) != list(MARKER_IDS):
            raise ValueError("calibration requires unique IDs 41-44")
        self.H_image_to_table = _matrix(H)
        self.H_table_to_image = _matrix(
            np.linalg.inv(self.H_image_to_table) if inverse is None else inverse)
        product = self.H_table_to_image @ self.H_image_to_table
        if abs(float(product[2, 2])) <= _EPS or not np.allclose(
                product / product[2, 2], np.eye(3), atol=1e-7):
            raise ValueError("homography matrices are not inverses")
        if (not all(math.isfinite(v) and v >= 0 for v in
                    (rms_error_cm, max_error_cm)) or
                max_error_cm < rms_error_cm or correspondence_count != 16):
            raise ValueError("invalid calibration quality")
        self.rms_error_cm = float(rms_error_cm)
        self.max_error_cm = float(max_error_cm)
        self.correspondence_count = correspondence_count
        self.created_at = created_at or datetime.now(timezone.utc).isoformat()

    @classmethod
    def from_scan(cls, scan, markers, image_size, table_size):
        markers = tuple(markers)
        if len(markers) != 4 or sorted(m.id for m in markers) != list(MARKER_IDS):
            raise ValueError("calibration requires unique IDs 41-44")
        image_points, table_points = [], []
        for marker in markers:
            corners = scan.get(marker.id)
            if corners is None:
                raise ValueError(f"marker {marker.id} missing")
            corners = np.asarray(corners, dtype=np.float64)
            if (corners.shape != (4, 2) or not np.isfinite(corners).all() or
                    cv2.contourArea(corners.astype(np.float32)) < 4):
                raise ValueError(f"invalid corners for marker {marker.id}")
            image_points.extend(corners)
            table_points.extend(marker.table_corners())
        image_points = np.asarray(image_points)
        table_points = np.asarray(table_points)
        H, _ = cv2.findHomography(image_points, table_points, method=0)
        if H is None:
            raise ValueError("homography fit failed")
        errors = np.linalg.norm(_point_transform(image_points, H) - table_points, axis=1)
        return cls(image_size, table_size, markers, H,
                   rms_error_cm=float(np.sqrt(np.mean(errors ** 2))),
                   max_error_cm=float(np.max(errors)))

    def pixel_to_table(self, x, y):
        return tuple(float(v) for v in _point_transform([(x, y)], self.H_image_to_table)[0])

    def table_to_pixel(self, x, y):
        return tuple(float(v) for v in _point_transform([(x, y)], self.H_table_to_image)[0])

    def to_json(self):
        return {
            "version": self.VERSION,
            "image_width": self.image_width, "image_height": self.image_height,
            "table_width_cm": self.table_width_cm,
            "table_height_cm": self.table_height_cm,
            "aruco_dictionary": DICTIONARY,
            "markers": [m.to_json() for m in self.markers],
            "homography_image_to_table": self.H_image_to_table.tolist(),
            "homography_table_to_image": self.H_table_to_image.tolist(),
            "created_at": self.created_at,
            "quality": {"rms_error_cm": self.rms_error_cm,
                        "max_error_cm": self.max_error_cm,
                        "correspondence_count": self.correspondence_count},
        }

    def save(self, path):
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".tmp")
        temporary.write_text(json.dumps(self.to_json(), indent=2, allow_nan=False) + "\n",
                             encoding="utf-8")
        os.replace(temporary, target)

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("version") != cls.VERSION or data.get("aruco_dictionary") != DICTIONARY:
            raise ValueError("unsupported calibration version or dictionary")
        quality = data["quality"]
        return cls((data["image_width"], data["image_height"]),
                   (data["table_width_cm"], data["table_height_cm"]),
                   [MarkerDefinition.from_json(item) for item in data["markers"]],
                   data["homography_image_to_table"], data["homography_table_to_image"],
                   quality["rms_error_cm"], quality["max_error_cm"],
                   quality["correspondence_count"], data["created_at"])
