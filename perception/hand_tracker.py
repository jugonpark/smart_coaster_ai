"""
인식 계층 - MediaPipe HandLandmarker로 손 스켈레톤 + 그립 모양.

왜 별도 모델인가
    PoseLandmarker는 손목 1점(15/16)만 준다. 손가락 정보가 아예 없다.
    기존 프로토타입이 쓰던 Holistic은 손 21점을 같이 줬지만, legacy Solutions API가
    패키지에서 제거되면서 더 이상 못 쓴다. Tasks API에서는 손을 따로 돌려야 한다.

그립 지표
    hand_world_landmarks(미터 단위, 손 중심 기준)를 쓴다.
    화면 좌표(hand_landmarks)는 카메라와의 거리·각도에 따라 크기가 변해서
    "손 모양"을 재는 데 부적합하다. world 쪽은 시점에 거의 독립적이다.

    hand_scale = |wrist(0) - middle_mcp(9)|          손 크기 정규화 기준
    aperture   = |thumb_tip(4) - index_tip(8)| / hand_scale
                 엄지-검지 벌어짐. 잡기 직전 pre-shaping이 여기 나타난다.
                 참고: 편 손 ~1.5-2.5 / 집기(pinch) ~0.1-0.3 / 컵 잡기 준비 ~0.8-1.5
    openness   = 네 손가락 끝의 손목까지 평균거리 / hand_scale
                 편 손 ~2.5-3.0 / 주먹 ~1.2-1.5

    실제 값은 사람마다 다르므로 설정 패널에서 보면서 임계값을 맞추는 것을 전제로 한다.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

import config

WRIST = 0
THUMB_TIP = 4
INDEX_MCP, INDEX_TIP = 5, 8
MIDDLE_MCP, MIDDLE_TIP = 9, 12
RING_MCP, RING_TIP = 13, 16
PINKY_MCP, PINKY_TIP = 17, 20

# 21점 손 스켈레톤 연결
HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),            # 엄지
    (0, 5), (5, 6), (6, 7), (7, 8),            # 검지
    (9, 10), (10, 11), (11, 12),               # 중지
    (13, 14), (14, 15), (15, 16),              # 약지
    (0, 17), (17, 18), (18, 19), (19, 20),     # 소지
    (5, 9), (9, 13), (13, 17),                 # 손바닥
)

FINGER_TIPS = (INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP)


@dataclass
class HandInfo:
    """손 한 개."""

    handedness: str = "?"          # "Left" / "Right" (카메라 기준)
    score: float = 0.0
    px: list = field(default_factory=list)      # 21개 (x_px, y_px)
    wrist_x_cm: float = 0.0
    wrist_y_cm: float = 0.0

    # 그립 지표 (hand_scale로 정규화된 무차원 값)
    aperture: float = 0.0
    openness: float = 0.0
    valid: bool = False

    @property
    def grasp_ready(self) -> bool:
        """
        '잡으려고 손 모양을 만든 상태'인지 대략 판정.
        완전히 편 손도, 꽉 쥔 주먹도 아닌 중간 상태(컵을 감싸는 C자)를 잡기 준비로 본다.
        openness 하한이 없으면 주먹도 통과한다.
        """
        if not self.valid:
            return False
        return (
            config.GRIP_APERTURE_MIN <= self.aperture <= config.GRIP_APERTURE_MAX
            and config.GRIP_OPENNESS_MIN <= self.openness <= config.GRIP_OPENNESS_MAX
        )


@dataclass
class HandsResult:
    detected: bool = False
    hands: list = field(default_factory=list)

    @property
    def any_grasp_ready(self) -> bool:
        return any(h.grasp_ready for h in self.hands)

    def nearest_to(self, x_cm: float, y_cm: float) -> HandInfo | None:
        if not self.hands:
            return None
        return min(self.hands,
                   key=lambda h: math.hypot(h.wrist_x_cm - x_cm, h.wrist_y_cm - y_cm))


def _dist3(a, b) -> float:
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)


class HandTracker:
    def __init__(self) -> None:
        path = config.HAND_MODEL_PATH
        if not os.path.isfile(path):
            raise RuntimeError(
                f"HandLandmarker 모델 파일이 없습니다: {path}\n"
                "다음 명령으로 내려받으세요:\n"
                "    python tools/download_models.py"
            )

        options = vision.HandLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=path),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=config.HAND_NUM_HANDS,
            min_hand_detection_confidence=config.HAND_MIN_DETECTION_CONF,
            min_hand_presence_confidence=config.HAND_MIN_PRESENCE_CONF,
            min_tracking_confidence=config.HAND_MIN_TRACKING_CONF,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self._last_ts_ms = -1
        print(f"[Hand] HandLandmarker 로드 완료: {path}")

    def process(self, rgb_frame, world, now: float | None = None) -> HandsResult:
        import time

        ts_ms = int((time.time() if now is None else now) * 1000)
        if ts_ms <= self._last_ts_ms:
            ts_ms = self._last_ts_ms + 1
        self._last_ts_ms = ts_ms

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        res = self._landmarker.detect_for_video(mp_image, ts_ms)

        if not res.hand_landmarks:
            return HandsResult(detected=False)

        out: list[HandInfo] = []
        for i, lms in enumerate(res.hand_landmarks):
            info = HandInfo()

            # 화면 좌표 (그리기용)
            info.px = [(l.x * rgb_frame.shape[1], l.y * rgb_frame.shape[0])
                       for l in lms]
            info.wrist_x_cm, info.wrist_y_cm = world.to_world(*info.px[WRIST])

            # 손 종류
            if res.handedness and i < len(res.handedness) and res.handedness[i]:
                c = res.handedness[i][0]
                info.handedness = c.category_name
                info.score = float(c.score)

            # 그립 지표는 world landmarks로 계산 (시점 독립)
            if res.hand_world_landmarks and i < len(res.hand_world_landmarks):
                w = res.hand_world_landmarks[i]
                if len(w) >= 21:
                    scale = _dist3(w[WRIST], w[MIDDLE_MCP])
                    if scale > 1e-6:
                        info.aperture = _dist3(w[THUMB_TIP], w[INDEX_TIP]) / scale
                        info.openness = sum(
                            _dist3(w[t], w[WRIST]) for t in FINGER_TIPS
                        ) / (4.0 * scale)
                        info.valid = True

            out.append(info)

        return HandsResult(detected=True, hands=out)

    def draw(self, frame, result: HandsResult) -> None:
        """21점 스켈레톤 + 그립 수치 표시."""
        if not result.detected:
            return

        for h in result.hands:
            if len(h.px) < 21:
                continue
            ready = h.grasp_ready
            bone = (0, 200, 255) if ready else (200, 200, 200)
            joint = (0, 100, 255) if ready else (245, 66, 230)

            for a, b in HAND_CONNECTIONS:
                pa = (int(h.px[a][0]), int(h.px[a][1]))
                pb = (int(h.px[b][0]), int(h.px[b][1]))
                cv2.line(frame, pa, pb, bone, 2)
            for idx, (x, y) in enumerate(h.px):
                r = 5 if idx in (THUMB_TIP, INDEX_TIP) else 3
                cv2.circle(frame, (int(x), int(y)), r, joint, -1)

            # 엄지-검지 사이를 강조 (aperture가 이 선의 길이)
            pt = (int(h.px[THUMB_TIP][0]), int(h.px[THUMB_TIP][1]))
            pi = (int(h.px[INDEX_TIP][0]), int(h.px[INDEX_TIP][1]))
            cv2.line(frame, pt, pi, (0, 255, 255), 1, cv2.LINE_AA)

            if h.valid:
                wx, wy = int(h.px[WRIST][0]), int(h.px[WRIST][1])
                cv2.putText(
                    frame,
                    f"{h.handedness[:1]} ap{h.aperture:.2f} op{h.openness:.2f}"
                    + ("  GRASP" if ready else ""),
                    (wx - 40, wy + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (0, 200, 255) if ready else (200, 200, 200), 1,
                )

    def close(self) -> None:
        self._landmarker.close()
