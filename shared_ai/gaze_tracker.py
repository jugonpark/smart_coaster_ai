"""
인식 계층 - MediaPipe FaceLandmarker로 머리 방향(시선 대용) 추정.

왜 홍채가 아니라 머리 방향인가
    1~2m 거리에서 홍채(iris) 기반 시선 추정은 노이즈가 심해 실용성이 없다.
    반면 머리 방향(head pose)은 거리에 훨씬 강건하고, 손을 뻗어 물건을 잡을 때는
    거의 항상 대상 쪽으로 고개가 향한다. 그래서 head pose를 시선 대용으로 쓴다.

어떻게 구하나
    FaceLandmarkerOptions(output_facial_transformation_matrixes=True)를 주면
    표준 얼굴 모델 -> 카메라 좌표계로 가는 4x4 변환행렬이 나온다.
    그 회전부 R에서 머리의 정면 벡터와 yaw/pitch/roll을 뽑는다.

    ★ 부호 규약 주의 ★
    MediaPipe 표준 얼굴 모델에서 얼굴 정면이 +Z인지 -Z인지는 버전에 따라 다를 수 있다.
    화면에 시선 광선을 그려두었으니, 반대로 나오면 config.GAZE_FORWARD_SIGN을
    -1로 바꾸면 된다 (설정 패널에도 노출되어 있다).

카메라 각도와의 관계
    천장 카메라라도 70~80도 정도로 비스듬하면 얼굴이 보인다.
    특히 사람이 작업면의 컵을 내려다볼 때 얼굴 법선이 카메라 쪽으로 돌아서
    "컵을 쳐다보는 순간"이 곧 "얼굴이 가장 잘 보이는 순간"이 된다.

한계 (반드시 인지할 것)
    - 얼굴이 안 보이면 아무 신호도 안 나온다. 즉 fail-safe가 아니다.
      그래서 이 신호는 위험도를 '올리는' 데만 쓰고, 하드 정지를 트리거하지 않는다.
    - 머리의 월드 위치는 시차(parallax) 오차가 크다(높이 80cm면 수십 cm).
      따라서 '컵을 보는가' 판정은 방향 위주로만 쓰고 정밀한 값으로 믿으면 안 된다.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

from . import config

# FaceMesh 468점 중 쓰는 것들
NOSE_TIP = 1
LEFT_EYE_OUTER = 33
RIGHT_EYE_OUTER = 263
CHIN = 152
FOREHEAD = 10


@dataclass
class GazeInfo:
    detected: bool = False
    # 머리 자세 (도)
    yaw_deg: float = 0.0      # 좌우 (+ = 카메라 기준 왼쪽)
    pitch_deg: float = 0.0    # 상하 (+ = 위)
    roll_deg: float = 0.0
    # 머리 정면 벡터를 월드 평면(XY)에 투영한 방향
    dir_x: float = 0.0
    dir_y: float = 0.0
    # 머리 위치 (화면/월드). 월드 값은 시차 오차가 크므로 참고용.
    head_px: tuple = (0.0, 0.0)
    head_x_cm: float = 0.0
    head_y_cm: float = 0.0
    # 목표를 향하는지
    angle_to_target_deg: float | None = None
    looking_at_target: bool = False


def _rot_to_euler(R: np.ndarray) -> tuple[float, float, float]:
    """
    회전행렬 -> (yaw, pitch, roll) 라디안.
    ZYX(yaw-pitch-roll) 순서 분해. 짐벌락 근처는 별도 처리.
    """
    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    if sy > 1e-6:
        roll = math.atan2(R[2, 1], R[2, 2])
        pitch = math.atan2(-R[2, 0], sy)
        yaw = math.atan2(R[1, 0], R[0, 0])
    else:
        roll = math.atan2(-R[1, 2], R[1, 1])
        pitch = math.atan2(-R[2, 0], sy)
        yaw = 0.0
    return yaw, pitch, roll


class GazeTracker:
    def __init__(self) -> None:
        path = config.FACE_MODEL_PATH
        if not os.path.isfile(path):
            raise RuntimeError(
                f"FaceLandmarker 모델 파일이 없습니다: {path}\n"
                "다음 명령으로 내려받으세요:\n"
                "    python tools/download_ai_models.py face"
            )

        options = vision.FaceLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=path),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=config.FACE_MIN_DETECTION_CONF,
            min_face_presence_confidence=config.FACE_MIN_PRESENCE_CONF,
            min_tracking_confidence=config.FACE_MIN_TRACKING_CONF,
            output_face_blendshapes=False,
            # 이게 있어야 머리 자세를 얻을 수 있다.
            output_facial_transformation_matrixes=True,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        self._last_ts_ms = -1
        print(f"[Gaze] FaceLandmarker 로드 완료: {path}")

    def process(self, rgb_frame, world, target_cm=None, now: float | None = None) -> GazeInfo:
        """target_cm: (x_cm, y_cm) 컵 위치. 주면 '그쪽을 보는지'까지 판정한다."""
        import time

        ts_ms = int((time.time() if now is None else now) * 1000)
        if ts_ms <= self._last_ts_ms:
            ts_ms = self._last_ts_ms + 1
        self._last_ts_ms = ts_ms

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        res = self._landmarker.detect_for_video(mp_image, ts_ms)

        if not res.face_landmarks:
            return GazeInfo(detected=False)

        info = GazeInfo(detected=True)

        lms = res.face_landmarks[0]
        if len(lms) > NOSE_TIP:
            n = lms[NOSE_TIP]
            info.head_px = (n.x * rgb_frame.shape[1], n.y * rgb_frame.shape[0])
            info.head_x_cm, info.head_y_cm = world.to_world(*info.head_px)

        mats = getattr(res, "facial_transformation_matrixes", None)
        if mats:
            M = np.array(mats[0]).reshape(4, 4)
            R = M[:3, :3]
            # 스케일 성분 제거 (변환행렬에 크기가 섞여 있을 수 있다)
            for c in range(3):
                nrm = np.linalg.norm(R[:, c])
                if nrm > 1e-9:
                    R[:, c] /= nrm

            yaw, pitch, roll = _rot_to_euler(R)
            info.yaw_deg = math.degrees(yaw)
            info.pitch_deg = math.degrees(pitch)
            info.roll_deg = math.degrees(roll)

            # 얼굴 정면 벡터 (표준 모델 기준 +Z, 부호는 config로 뒤집을 수 있음)
            fwd = R @ np.array([0.0, 0.0, 1.0]) * config.GAZE_FORWARD_SIGN
            # 카메라 좌표의 y는 아래가 +, 월드는 위가 + 이므로 뒤집는다.
            dx, dy = float(fwd[0]), -float(fwd[1])
            nrm = math.hypot(dx, dy)
            if nrm > 1e-6:
                info.dir_x, info.dir_y = dx / nrm, dy / nrm

        # --- 목표를 향하는지 ---
        if target_cm is not None and (info.dir_x or info.dir_y):
            tx, ty = target_cm
            vx = tx - info.head_x_cm
            vy = ty - info.head_y_cm
            n = math.hypot(vx, vy)
            if n > 1e-6:
                cosang = (info.dir_x * vx + info.dir_y * vy) / n
                ang = math.degrees(math.acos(max(-1.0, min(1.0, cosang))))
                info.angle_to_target_deg = ang
                info.looking_at_target = ang <= config.GAZE_CONE_DEG

        return info

    def draw(self, frame, info: GazeInfo, world) -> None:
        if not info.detected:
            return
        px = (int(info.head_px[0]), int(info.head_px[1]))
        col = (0, 255, 255) if info.looking_at_target else (180, 180, 180)
        cv2.circle(frame, px, 6, col, -1)

        # 시선 광선 (월드 방향 -> 화면: y 부호 반전)
        L = 110
        tip = (int(px[0] + L * info.dir_x), int(px[1] - L * info.dir_y))
        cv2.arrowedLine(frame, px, tip, col, 2, tipLength=0.25)

        txt = f"yaw{info.yaw_deg:+.0f} pitch{info.pitch_deg:+.0f}"
        if info.angle_to_target_deg is not None:
            txt += f"  to-cup {info.angle_to_target_deg:.0f}deg"
        cv2.putText(frame, txt, (px[0] + 10, px[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)

    def close(self) -> None:
        self._landmarker.close()
