"""Planar IMAGE <-> TABLE homography calibration.

TABLE coordinates are centimeters, origin at the physical table center, +X and
+Y along the configured table axes, and positive rotation counterclockwise.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np


_W_EPSILON = 1e-12


@dataclass(frozen=True)
class MarkerDefinition:
    marker_id: int
    x_cm: float
    y_cm: float
    size_cm: float
    orientation_deg: float = 0.0

    def __post_init__(self) -> None:
        values = (self.x_cm, self.y_cm, self.size_cm, self.orientation_deg)
        if (not isinstance(self.marker_id, int) or self.marker_id < 0 or
                not all(math.isfinite(value) for value in values) or self.size_cm <= 0):
            raise ValueError("invalid calibration marker definition")

    def table_corners(self) -> np.ndarray:
        """Return TL, TR, BR, BL coordinates for the configured marker pose."""
        half = self.size_cm * 0.5
        local = np.array([[-half, half], [half, half], [half, -half], [-half, -half]],
                         dtype=np.float64)
        angle = math.radians(self.orientation_deg)
        rotation = np.array([[math.cos(angle), -math.sin(angle)],
                             [math.sin(angle), math.cos(angle)]], dtype=np.float64)
        return local @ rotation.T + np.array([self.x_cm, self.y_cm], dtype=np.float64)

    def to_json(self) -> dict:
        return {"id": self.marker_id, "x_cm": self.x_cm, "y_cm": self.y_cm,
                "size_cm": self.size_cm, "orientation_deg": self.orientation_deg}

    @classmethod
    def from_json(cls, data: dict) -> "MarkerDefinition":
        return cls(int(data["id"]), float(data["x_cm"]), float(data["y_cm"]),
                   float(data["size_cm"]), float(data.get("orientation_deg", 0.0)))


class TableCalibration:
    VERSION = 1

    def __init__(self, image_width: int, image_height: int,
                 H_image_to_table=None, H_table_to_image=None,
                 markers: Sequence[MarkerDefinition] | None = None,
                 rms_error_cm: float | None = None,
                 max_error_cm: float | None = None,
                 *, aruco_dictionary: str = "DICT_4X4_50",
                 created_at: str | None = None, invalid_reason: str | None = None,
                 correspondence_count: int = 0) -> None:
        self.image_width = int(image_width)
        self.image_height = int(image_height)
        self.markers = list(markers or [])
        self.aruco_dictionary = aruco_dictionary
        self.created_at = created_at or datetime.now(timezone.utc).isoformat()
        self.rms_error_cm = rms_error_cm
        self.max_error_cm = max_error_cm
        self.correspondence_count = int(correspondence_count)
        self.invalid_reason = invalid_reason
        self.H_image_to_table = None
        self.H_table_to_image = None
        if invalid_reason is not None:
            return
        if self.image_width <= 0 or self.image_height <= 0:
            raise ValueError("image dimensions must be positive")
        self.H_image_to_table = self._validated_matrix(H_image_to_table)
        if H_table_to_image is None:
            try:
                H_table_to_image = np.linalg.inv(self.H_image_to_table)
            except np.linalg.LinAlgError as exc:
                raise ValueError("singular homography") from exc
        self.H_table_to_image = self._validated_matrix(H_table_to_image)
        inverse_product = self.H_table_to_image @ self.H_image_to_table
        if (not np.isfinite(inverse_product).all() or
                abs(float(inverse_product[2, 2])) <= _W_EPSILON):
            raise ValueError("homography matrices are not inverses")
        inverse_product /= inverse_product[2, 2]
        if not np.allclose(inverse_product, np.eye(3), atol=1e-7):
            raise ValueError("homography matrices are not inverses")
        for value in (rms_error_cm, max_error_cm):
            if value is not None and (not math.isfinite(value) or value < 0):
                raise ValueError("invalid calibration quality")

    @staticmethod
    def _validated_matrix(matrix) -> np.ndarray:
        value = np.asarray(matrix, dtype=np.float64)
        if value.shape != (3, 3) or not np.isfinite(value).all():
            raise ValueError("homography must be a finite float64 3x3 matrix")
        if abs(float(np.linalg.det(value))) <= _W_EPSILON:
            raise ValueError("singular homography")
        return value

    @property
    def valid(self) -> bool:
        return self.invalid_reason is None and self.H_image_to_table is not None

    def matches_resolution(self, width: int, height: int) -> bool:
        return self.valid and int(width) == self.image_width and int(height) == self.image_height

    @staticmethod
    def _transform(points: Iterable[Sequence[float]], matrix: np.ndarray) -> np.ndarray:
        values = np.asarray(list(points), dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != 2 or not np.isfinite(values).all():
            raise ValueError("points must be finite Nx2 values")
        homogeneous = np.column_stack((values, np.ones(len(values)))) @ matrix.T
        w = homogeneous[:, 2]
        if not np.isfinite(homogeneous).all() or np.any(np.abs(w) <= _W_EPSILON):
            raise ValueError("invalid homogeneous transform")
        result = homogeneous[:, :2] / w[:, None]
        if not np.isfinite(result).all():
            raise ValueError("non-finite transformed point")
        return result

    def transform_points(self, points, *, inverse: bool = False) -> np.ndarray:
        if not self.valid:
            raise ValueError(self.invalid_reason or "calibration invalid")
        matrix = self.H_table_to_image if inverse else self.H_image_to_table
        return self._transform(points, matrix)

    def pixel_to_table(self, px_x: float, px_y: float) -> tuple[float, float]:
        value = self.transform_points([(px_x, px_y)])[0]
        return float(value[0]), float(value[1])

    def table_to_pixel(self, x_cm: float, y_cm: float) -> tuple[float, float]:
        value = self.transform_points([(x_cm, y_cm)], inverse=True)[0]
        return float(value[0]), float(value[1])

    def local_pixel_length_to_cm(self, px_x: float, px_y: float, length_px: float) -> float:
        if not math.isfinite(length_px) or length_px < 0:
            raise ValueError("invalid pixel length")
        center = np.array(self.pixel_to_table(px_x, px_y))
        references = self.transform_points([(px_x - length_px, px_y),
                                            (px_x + length_px, px_y),
                                            (px_x, px_y - length_px),
                                            (px_x, px_y + length_px)])
        return float(max(np.linalg.norm(point - center) for point in references))

    def bbox_to_table_radius(self, cx_px: float, cy_px: float,
                             width_px: float, height_px: float) -> float:
        values = (cx_px, cy_px, width_px, height_px)
        if not all(math.isfinite(value) for value in values) or width_px < 0 or height_px < 0:
            raise ValueError("invalid bounding box")
        center = np.array(self.pixel_to_table(cx_px, cy_px))
        points = self.transform_points([(cx_px - width_px / 2, cy_px),
                                        (cx_px + width_px / 2, cy_px),
                                        (cx_px, cy_px - height_px / 2),
                                        (cx_px, cy_px + height_px / 2)])
        return float(max(np.linalg.norm(point - center) for point in points))

    @classmethod
    def from_marker_scan(cls, scan, markers: Sequence[MarkerDefinition],
                         image_width: int, image_height: int,
                         *, aruco_dictionary: str = "DICT_4X4_50") -> "TableCalibration":
        image_points, table_points, found = [], [], 0
        for marker in markers:
            corners = scan.get(marker.marker_id)
            if corners is None:
                continue
            corners = np.asarray(corners, dtype=np.float64)
            if corners.shape != (4, 2) or not np.isfinite(corners).all():
                raise ValueError(f"invalid corners for marker {marker.marker_id}")
            image_points.extend(corners)
            table_points.extend(marker.table_corners())
            found += 1
        if found < 4:
            raise ValueError("at least four configured calibration markers are required")
        image_array = np.asarray(image_points, dtype=np.float64)
        table_array = np.asarray(table_points, dtype=np.float64)
        H, _ = cv2.findHomography(image_array, table_array, method=0)
        if H is None:
            raise ValueError("homography calculation failed")
        projected = cls._transform(image_array, H)
        errors = np.linalg.norm(projected - table_array, axis=1)
        rms = float(math.sqrt(float(np.mean(errors ** 2))))
        maximum = float(np.max(errors))
        return cls(image_width, image_height, H, markers=markers,
                   rms_error_cm=rms, max_error_cm=maximum,
                   aruco_dictionary=aruco_dictionary,
                   correspondence_count=len(image_array))

    def to_json(self) -> dict:
        if not self.valid:
            raise ValueError(self.invalid_reason or "calibration invalid")
        return {
            "version": self.VERSION, "image_width": self.image_width,
            "image_height": self.image_height, "aruco_dictionary": self.aruco_dictionary,
            "markers": [marker.to_json() for marker in self.markers],
            "homography_image_to_table": self.H_image_to_table.tolist(),
            "homography_table_to_image": self.H_table_to_image.tolist(),
            "created_at": self.created_at,
            "quality": {"rms_error_cm": self.rms_error_cm,
                        "max_error_cm": self.max_error_cm,
                        "correspondence_count": self.correspondence_count},
        }

    def save(self, path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_json(), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path, runtime_resolution: tuple[int, int] | None = None) -> "TableCalibration":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("created_at") == "EXAMPLE_ONLY_RECALIBRATE_FOR_HARDWARE":
            raise ValueError("example calibration is not a physical calibration")
        if data.get("version") != cls.VERSION:
            raise ValueError("unsupported calibration version")
        quality = data.get("quality") or {}
        result = cls(data["image_width"], data["image_height"],
                     data["homography_image_to_table"], data["homography_table_to_image"],
                     [MarkerDefinition.from_json(item) for item in data.get("markers", [])],
                     quality.get("rms_error_cm"), quality.get("max_error_cm"),
                     aruco_dictionary=data.get("aruco_dictionary", "DICT_4X4_50"),
                     created_at=data.get("created_at"),
                     correspondence_count=quality.get("correspondence_count", 0))
        if runtime_resolution and not result.matches_resolution(*runtime_resolution):
            return cls.invalid(*runtime_resolution, reason="calibration resolution mismatch")
        return result

    @classmethod
    def invalid(cls, image_width: int, image_height: int, *, reason: str) -> "TableCalibration":
        return cls(image_width, image_height, invalid_reason=reason)

    @classmethod
    def load_optional(cls, path, runtime_resolution: tuple[int, int]) -> "TableCalibration":
        try:
            return cls.load(path, runtime_resolution)
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            return cls.invalid(*runtime_resolution, reason=f"calibration unavailable: {exc}")
