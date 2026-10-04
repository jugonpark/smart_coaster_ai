"""
인식 계층 - MediaPipe Pose로 사람 손목/팔꿈치/어깨 추적.

★ MediaPipe Tasks API를 사용한다 ★
   기존 프로토타입이 쓰던 `mp.solutions.holistic` / `mp.solutions.pose`
   (legacy Solutions API)는 mediapipe 0.10.30 이후 패키지에서 완전히 제거되었다.
   현재 설치 가능한 모든 버전(0.10.30~1.0.0)에는 `mediapipe.tasks`만 들어 있다.
   그래서 PoseLandmarker Task로 새로 작성했다.

   Tasks API는 모델 파일(.task)을 별도로 받아야 한다:
       python tools/download_pose_model.py
   또는 아래 URL에서 직접 받아 config.POSE_MODEL_PATH 경로에 둔다.
       https://storage.googleapis.com/mediapipe-models/pose_landmarker/
           pose_landmarker_lite/float16/1/pose_landmarker_lite.task

랜드마크 인덱스는 BlazePose 33점 규약 그대로다 (legacy와 동일):
    11 L어깨  12 R어깨  13 L팔꿈치  14 R팔꿈치  15 L손목  16 R손목

출력은 항상 월드 좌표(cm)까지 변환해서 내보낸다.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

import config

# BlazePose 33점 인덱스
L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW, R_ELBOW = 13, 14
L_WRIST, R_WRIST = 15, 16

# 화면에 그릴 상체 연결선
_ARM_CONNECTIONS = (
    (L_SHOULDER, R_SHOULDER),
    (L_SHOULDER, L_ELBOW),
    (L_ELBOW, L_WRIST),
    (R_SHOULDER, R_ELBOW),
    (R_ELBOW, R_WRIST),
)


@dataclass
class Joint:
    x_cm: float
    y_cm: float
    px: tuple[float, float]
    visible: bool = True


@dataclass
class HumanPose:
    """사람 한 명의 상체 관절."""

    detected: bool = False
    left_wrist: Joint | None = None
    right_wrist: Joint | None = None
    left_elbow: Joint | None = None
    right_elbow: Joint | None = None
    left_shoulder: Joint | None = None
    right_shoulder: Joint | None = None
    _by_index: dict = field(default_factory=dict, repr=False)

    @property
    def wrists(self) -> list[Joint]:
        """보이는 손목만 모아서 반환. 판단 계층이 이 중 컵에 가까운 쪽을 고른다."""
        return [j for j in (self.left_wrist, self.right_wrist) if j is not None]

    @property
    def repulsion_points(self) -> list[Joint]:
        """
        경로계산 계층에서 척력원으로 쓸 점들.
        손목이 가장 위험하지만 팔꿈치/어깨도 몸통 부피를 대표하므로 함께 넣는다.
        """
        candidates = (
            self.left_wrist, self.right_wrist,
            self.left_elbow, self.right_elbow,
            self.left_shoulder, self.right_shoulder,
        )
        return [j for j in candidates if j is not None]


class PoseTracker:
    def __init__(self) -> None:
        model_path = config.POSE_MODEL_PATH
        if not os.path.isfile(model_path):
            raise RuntimeError(
                f"PoseLandmarker 모델 파일이 없습니다: {model_path}\n"
                "다음 명령으로 내려받으세요:\n"
                "    python tools/download_pose_model.py"
            )

        options = vision.PoseLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=model_path),
            # VIDEO 모드는 프레임 간 추적을 유지해 IMAGE 모드보다 안정적이고 빠르다.
            running_mode=vision.RunningMode.VIDEO,
            num_poses=config.POSE_NUM_POSES,
            min_pose_detection_confidence=config.POSE_MIN_DETECTION_CONF,
            min_pose_presence_confidence=config.POSE_MIN_PRESENCE_CONF,
            min_tracking_confidence=config.POSE_MIN_TRACKING_CONF,
            output_segmentation_masks=False,
        )
        self._landmarker = vision.PoseLandmarker.create_from_options(options)

        # detect_for_video는 타임스탬프가 '엄격히 증가'해야 한다. 같은 값이면 예외가 난다.
        self._last_ts_ms = -1

        print(f"[Pose] PoseLandmarker 로드 완료: {model_path}")

    def process(self, rgb_frame, world, now: float | None = None) -> HumanPose:
        """rgb_frame은 RGB(uint8) numpy 배열. now는 time.time() 값."""
        import time

        ts_ms = int((time.time() if now is None else now) * 1000)
        if ts_ms <= self._last_ts_ms:
            ts_ms = self._last_ts_ms + 1
        self._last_ts_ms = ts_ms

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = self._landmarker.detect_for_video(mp_image, ts_ms)

        if not result.pose_landmarks:
            return HumanPose(detected=False)

        # num_poses=1이면 첫 번째만 쓴다. 여러 명이면 가장 먼저 잡힌 사람.
        landmarks = result.pose_landmarks[0]

        by_index: dict[int, Joint] = {}

        def joint(idx: int) -> Joint | None:
            if idx >= len(landmarks):
                return None
            l = landmarks[idx]
            # Tasks API는 visibility와 presence를 둘 다 준다. 둘 다 충족해야 신뢰.
            vis = getattr(l, "visibility", 1.0)
            pres = getattr(l, "presence", 1.0)
            if vis < config.POSE_MIN_VISIBILITY or pres < config.POSE_MIN_VISIBILITY:
                return None
            px_x = l.x * rgb_frame.shape[1]
            px_y = l.y * rgb_frame.shape[0]
            wx, wy = world.to_world(px_x, px_y)
            j = Joint(x_cm=wx, y_cm=wy, px=(px_x, px_y))
            by_index[idx] = j
            return j

        pose = HumanPose(
            detected=True,
            left_wrist=joint(L_WRIST),
            right_wrist=joint(R_WRIST),
            left_elbow=joint(L_ELBOW),
            right_elbow=joint(R_ELBOW),
            left_shoulder=joint(L_SHOULDER),
            right_shoulder=joint(R_SHOULDER),
        )
        pose._by_index = by_index
        return pose

    def draw(self, frame, pose: HumanPose) -> None:
        """
        상체 관절을 직접 그린다.
        (legacy의 mp.solutions.drawing_utils도 함께 제거되었으므로 cv2로 그린다)
        """
        if not pose.detected:
            return

        for a, b in _ARM_CONNECTIONS:
            ja, jb = pose._by_index.get(a), pose._by_index.get(b)
            if ja is None or jb is None:
                continue
            cv2.line(
                frame,
                (int(ja.px[0]), int(ja.px[1])),
                (int(jb.px[0]), int(jb.px[1])),
                (245, 117, 66), 3,
            )

        for j in pose.repulsion_points:
            cv2.circle(frame, (int(j.px[0]), int(j.px[1])), 6, (245, 66, 230), -1)

    def close(self) -> None:
        self._landmarker.close()
