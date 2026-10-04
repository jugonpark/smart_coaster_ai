"""
인식 계층 - ArUco로 로봇 위치 + 방향각(heading) 추정.

heading 정의
------------
ArUco corners는 [TL, TR, BR, BL] 순서로 나온다.
마커의 위쪽 변 (TL -> TR) 방향을 로봇 정면(+X_robot)으로 삼는다.
월드 좌표계는 Y가 위로 뒤집혀 있으므로, 픽셀 벡터의 y 부호를 뒤집은 뒤
atan2로 각도를 구하면 반시계(CCW) 양수인 표준 각도가 된다.

로봇에 마커를 붙인 방향이 다르면 config.MARKER_HEADING_OFFSET_DEG로 보정한다.

부수 효과
---------
마커의 실제 픽셀 크기를 알면 px_per_cm을 추정할 수 있다.
매 프레임 WorldFrame.update_scale()을 호출해 스케일을 갱신한다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2

import config
from .marker_scanner import MarkerScan, MarkerScanner
from .table_drawing import cv_points, project_vector


@dataclass
class RobotPose:
    detected: bool = False
    x_cm: float = 0.0
    y_cm: float = 0.0
    heading_rad: float = 0.0     # 월드 +X축 기준 반시계 양수
    px: tuple[float, float] = (0.0, 0.0)


def _wrap_pi(a: float) -> float:
    """각도를 (-pi, pi]로 정규화."""
    return math.atan2(math.sin(a), math.cos(a))


class RobotTracker:
    def __init__(self) -> None:
        # 스캔은 MarkerScanner가 담당한다. 테스트 모드의 물체 검출기와
        # 같은 detectMarkers() 결과를 공유하기 위해서다.
        self._scanner = MarkerScanner()
        self._last = RobotPose()
        self._heading_offset = math.radians(config.MARKER_HEADING_OFFSET_DEG)

    def process(self, frame, world, scan=None) -> RobotPose:
        """scan을 넘기면 재검출하지 않는다 (프레임당 detectMarkers 1회)."""
        if scan is None:
            scan = self._scanner.scan(frame)

        target = scan.get(config.ROBOT_MARKER_ID)

        if target is None:
            # 놓친 프레임은 직전 자세를 유지하되 detected=False로 알린다.
            return RobotPose(
                detected=False,
                x_cm=self._last.x_cm,
                y_cm=self._last.y_cm,
                heading_rad=self._last.heading_rad,
                px=self._last.px,
            )

        # --- 위치: 네 코너의 중심 ---
        cx_px, cy_px = MarkerScan.center_px(target)
        x_cm, y_cm = world.to_world(cx_px, cy_px)

        # --- heading: 위쪽 변 TL -> TR ---
        tl = world.to_world(*target[0])
        tr = world.to_world(*target[1])
        heading = _wrap_pi(math.atan2(tr[1] - tl[1], tr[0] - tl[0])
                           + self._heading_offset)

        # --- EMA 스무딩 ---
        a = config.ROBOT_POSE_EMA_ALPHA
        if self._last.detected:
            x_cm = (1 - a) * self._last.x_cm + a * x_cm
            y_cm = (1 - a) * self._last.y_cm + a * y_cm
            # 각도는 -pi/pi 경계를 넘을 수 있어 차이를 wrap한 뒤 보간해야 한다.
            d = _wrap_pi(heading - self._last.heading_rad)
            heading = _wrap_pi(self._last.heading_rad + a * d)

        self._last = RobotPose(
            detected=True,
            x_cm=x_cm,
            y_cm=y_cm,
            heading_rad=heading,
            px=(cx_px, cy_px),
        )
        return self._last

    def draw(self, frame, pose: RobotPose, world) -> None:
        if not pose.detected:
            return
        px = (int(pose.px[0]), int(pose.px[1]))
        cv2.circle(frame, px, 10, (255, 0, 255), -1)
        # heading 화살표 (월드 각도를 다시 픽셀 방향으로: y 부호 반전)
        start, end = project_vector(world, pose.x_cm, pose.y_cm,
                                    math.cos(pose.heading_rad),
                                    math.sin(pose.heading_rad), 8.0)
        cv2.arrowedLine(frame, *cv_points((start, end)),
                        (255, 0, 255), 3, tipLength=0.3)
        cv2.putText(
            frame,
            f"robot {math.degrees(pose.heading_rad):+.0f}deg",
            (px[0] + 12, px[1] - 12),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1,
        )
