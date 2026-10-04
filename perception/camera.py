"""USB/MJPEG capture and calibrated pixel↔TABLE-plane coordinates."""

from __future__ import annotations

import json
import time
from pathlib import Path
from urllib.parse import urlsplit

import cv2
import config
from .table_calibration import TableCalibration


class WorldFrame:
    """Single TABLE-plane transform and live calibration validity owner."""

    def __init__(self, calibration_path=None, calibration=None):
        self.calibration = calibration
        self._load_error = None
        self._frame_error = "no frame received"
        self._last_read_at = None
        if calibration is None:
            path = Path(calibration_path or config.TABLE_CALIBRATION_PATH)
            try:
                self.calibration = TableCalibration.load(path)
            except (OSError, ValueError, TypeError, KeyError, UnicodeError,
                    json.JSONDecodeError) as exc:
                self._load_error = f"calibration unavailable: {exc}"
        if self.calibration is not None and (
                self.calibration.rms_error_cm > config.CALIBRATION_MAX_RMS_CM or
                self.calibration.max_error_cm > config.CALIBRATION_MAX_ERROR_CM):
            self._load_error = "calibration quality exceeds RMS/MAX limits"

    def observe_frame(self, frame_shape, read_at=None):
        self._last_read_at = time.monotonic() if read_at is None else read_at
        if self._load_error or self.calibration is None:
            self._frame_error = self._load_error or "calibration unavailable"
        elif (len(frame_shape) < 2 or
              (frame_shape[1], frame_shape[0]) !=
              (self.calibration.image_width, self.calibration.image_height)):
            self._frame_error = "calibration resolution mismatch"
        else:
            self._frame_error = None

    def invalidate_frame(self, reason):
        self._frame_error = str(reason)
        self._last_read_at = None

    def can_send_motion(self, now=None):
        now = time.monotonic() if now is None else now
        return (self.calibration is not None and self._frame_error is None and
                self._last_read_at is not None and
                0 <= now - self._last_read_at <= config.CAMERA_FRAME_STALE_S)

    @property
    def invalid_reason(self):
        if self._load_error:
            return self._load_error
        if self._frame_error:
            return self._frame_error
        if not self.can_send_motion():
            return "camera frame stale"
        return None

    @property
    def valid(self):
        return self.can_send_motion()

    def to_world(self, px_x: float, px_y: float) -> tuple[float, float]:
        if self.calibration is None:
            raise ValueError(self.invalid_reason)
        return self.calibration.pixel_to_table(px_x, px_y)

    def to_pixel(self, world_x: float, world_y: float) -> tuple[float, float]:
        if self.calibration is None:
            raise ValueError(self.invalid_reason)
        return self.calibration.table_to_pixel(world_x, world_y)


class Camera:
    """OpenCV VideoCapture 래퍼."""

    def __init__(self) -> None:
        url = config.CAMERA_URL
        if url:
            parsed = urlsplit(url)
            if parsed.scheme not in ("http", "https") or not parsed.hostname:
                raise ValueError("GRISE_CAMERA_URL must be an HTTP(S) stream URL")
        self.cap = cv2.VideoCapture(url if url else config.CAMERA_INDEX)
        if not url:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
            self.cap.set(cv2.CAP_PROP_FPS, config.TARGET_FPS)
        # 버퍼가 쌓이면 지연이 생겨 실시간 제어에 치명적이다.
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not self.cap.isOpened():
            raise RuntimeError(
                "카메라를 열 수 없습니다. GRISE_CAMERA_URL 또는 CAMERA_INDEX를 확인하세요."
            )

        self.world = WorldFrame()

    def read(self):
        """(ok, frame) 반환. frame은 BGR."""
        ok, frame = self.cap.read()
        if not ok:
            self.world.invalidate_frame("camera read failed")
            return False, None
        if config.FLIP_HORIZONTAL:
            frame = cv2.flip(frame, 1)
        self.world.observe_frame(frame.shape)
        return True, frame

    def release(self) -> None:
        self.cap.release()
