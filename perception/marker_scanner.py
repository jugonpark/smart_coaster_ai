"""
ArUco 마커 스캔 - 프레임당 한 번만 검출하고 여러 소비자가 나눠 쓴다.

왜 분리했나
    detectMarkers()는 한 번 호출로 화면의 모든 마커 ID를 돌려준다.
    RobotTracker(로봇 위치)와 MarkerObjectDetector(테스트 모드의 컵/장애물)가
    각자 호출하면 같은 일을 두 번 하게 된다(프레임당 ~3ms씩).
    스캔을 한 번만 하고 결과를 공유한다.

마커 하나에서 뽑는 정보
    center_px : 네 코너의 중심
    heading   : 위쪽 변(TL->TR) 방향. 월드 좌표계(Y 위쪽 +) 기준 반시계 양수
    side_px   : 네 변의 평균 픽셀 길이 (중복 ID 선택용, 월드 스케일에는 사용하지 않음)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

import config


def wrap_pi(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


@dataclass
class MarkerScan:
    """한 프레임에서 검출된 모든 마커. id -> (4,2) 코너 배열."""

    by_id: dict = field(default_factory=dict)

    def __contains__(self, marker_id: int) -> bool:
        return marker_id in self.by_id

    def get(self, marker_id: int):
        return self.by_id.get(marker_id)

    @property
    def ids(self) -> list[int]:
        return sorted(self.by_id)

    # ---- 코너에서 값 뽑기 ----
    @staticmethod
    def center_px(corners) -> tuple[float, float]:
        return float(np.mean(corners[:, 0])), float(np.mean(corners[:, 1]))

    @staticmethod
    def side_px(corners) -> float:
        return float(np.mean([
            np.linalg.norm(corners[(k + 1) % 4] - corners[k]) for k in range(4)
        ]))

    @staticmethod
    def heading_rad(corners, offset_rad: float = 0.0) -> float:
        """마커 위쪽 변 TL->TR 방향. 픽셀 y는 아래가 +라서 부호를 뒤집는다."""
        tl, tr = corners[0], corners[1]
        return wrap_pi(math.atan2(-(float(tr[1]) - float(tl[1])),
                                  float(tr[0]) - float(tl[0])) + offset_rad)


class MarkerScanner:
    def __init__(self) -> None:
        aruco_dict = cv2.aruco.getPredefinedDictionary(
            getattr(cv2.aruco, config.ARUCO_DICT_NAME)
        )
        params = cv2.aruco.DetectorParameters()
        # 서브픽셀 코너 정밀화 - heading 정확도가 눈에 띄게 좋아진다.
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self._detector = cv2.aruco.ArucoDetector(aruco_dict, params)

    def scan(self, frame) -> MarkerScan:
        corners, ids, _ = self._detector.detectMarkers(frame)
        out: dict[int, np.ndarray] = {}
        if ids is not None:
            for c, i in zip(corners, ids.flatten()):
                # 같은 ID가 여러 개 잡히면 가장 큰 것(가까운 것)을 택한다.
                mid = int(i)
                cand = c[0]
                if mid not in out or MarkerScan.side_px(cand) > MarkerScan.side_px(out[mid]):
                    out[mid] = cand
        return MarkerScan(by_id=out)

    def draw(self, frame, scan: MarkerScan) -> None:
        for mid, c in scan.by_id.items():
            pts = c.astype(int).reshape(-1, 1, 2)
            cv2.polylines(frame, [pts], True, (0, 255, 0), 1)
            cx, cy = MarkerScan.center_px(c)
            cv2.putText(frame, str(mid), (int(cx) - 8, int(cy) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
