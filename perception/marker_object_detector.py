"""
테스트 모드 - ArUco 마커로 컵/장애물을 인식 (YOLO 대체).

YOLO 모델이 아직 없거나, 경로계산을 결정적으로 튜닝하고 싶을 때 쓴다.
ObjectDetector와 완전히 같은 Detection 리스트를 내놓기 때문에
판단/경로계산 계층은 어느 쪽이 물체를 찾았는지 알 필요가 없다.

YOLO 대비 장점
    - 학습/데이터셋 불필요, torch/ultralytics도 필요 없음
    - 결정적이라 게인 튜닝에 적합 (프레임마다 흔들리지 않음)
    - 빠름 (2~4ms). 게다가 로봇 추적과 검출 호출을 공유하므로 추가 비용이 사실상 없음
    - 방향과 실제 크기 스케일이 공짜로 나옴

★ 반드시 알아야 할 한계 ★
    1) 마커는 '점'이지 물체의 크기가 아니다.
       30cm 상자에 5cm 마커를 붙이면, 크기를 마커에서 유추할 경우 척력 반경이
       2.5cm로 잡혀서 로봇이 상자를 긁고 지나간다.
       그래서 config.MARKER_OBJECTS에 ID별 '실제 반경'을 직접 적어야 한다.
       이건 선택이 아니라 필수다.
    2) 가림에 all-or-nothing이다. 귀퉁이만 가려도 완전히 사라진다.
       하필 손이 컵으로 갈 때 컵 마커를 덮기 쉬우므로,
       MARKER_OBJECT_HOLD_S 동안 마지막 위치를 유지한다.
    3) 마커는 작업면과 나란히 붙일 것. 비스듬한 카메라에서 세로로 선 마커는
       스치는 각도라 검출이 안 된다.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import numpy as np

import config
from .marker_scanner import MarkerScan, MarkerScanner
from .object_detector import Detection
from .table_drawing import project_circle


@dataclass
class _Seen:
    """마지막으로 본 위치. 가려졌을 때 잠시 유지하는 데 쓴다."""

    x_cm: float
    y_cm: float
    cx_px: float
    cy_px: float
    heading_rad: float
    t: float


class MarkerObjectDetector:
    """config.MARKER_OBJECTS에 등록된 ID만 물체로 인식한다."""

    def __init__(self) -> None:
        self._scanner = MarkerScanner()
        self._seen: dict[int, _Seen] = {}
        roles = {}
        for mid, (role, radius) in config.MARKER_OBJECTS.items():
            roles.setdefault(role, []).append(mid)
        print("[테스트모드] ArUco 물체 인식 활성 — "
              + ", ".join(f"{r}: ID {sorted(v)}" for r, v in sorted(roles.items())))

    # ObjectDetector와 시그니처를 맞춘다 (호출부에서 구분할 필요가 없도록)
    def detect(self, frame, world, scan: MarkerScan | None = None,
               now: float | None = None) -> list[Detection]:
        now = time.time() if now is None else now
        if scan is None:
            scan = self._scanner.scan(frame)

        out: list[Detection] = []

        for mid, (role, radius_cm) in config.MARKER_OBJECTS.items():
            corners = scan.get(mid)

            if corners is not None:
                cx_px, cy_px = MarkerScan.center_px(corners)
                x_cm, y_cm = world.to_world(cx_px, cy_px)
                heading = MarkerScan.heading_rad(corners)
                self._seen[mid] = _Seen(x_cm, y_cm, cx_px, cy_px, heading, now)
                fresh = True
            else:
                s = self._seen.get(mid)
                # 가려진 동안 마지막 위치를 잠시 유지한다.
                if s is None or (now - s.t) > config.MARKER_OBJECT_HOLD_S:
                    continue
                cx_px, cy_px = s.cx_px, s.cy_px
                x_cm, y_cm = s.x_cm, s.y_cm
                fresh = False

            # 라벨은 Detection.role이 그대로 동작하도록 config 이름을 쓴다.
            if role == "cup":
                label = config.CLASS_CUP
            else:
                label = (config.CLASS_OBSTACLE_NAMES[0]
                         if config.CLASS_OBSTACLE_NAMES else "obstacle")

            # 화면 박스는 '마커 크기'가 아니라 '실제 회피 반경'을 그린다.
            # 그래야 로봇이 실제로 피하는 범위가 눈에 보인다.
            left = world.to_pixel(x_cm - radius_cm, y_cm)
            right = world.to_pixel(x_cm + radius_cm, y_cm)
            top = world.to_pixel(x_cm, y_cm + radius_cm)
            bottom = world.to_pixel(x_cm, y_cm - radius_cm)
            size_x = max(p[0] for p in (left, right, top, bottom)) - min(
                p[0] for p in (left, right, top, bottom))
            size_y = max(p[1] for p in (left, right, top, bottom)) - min(
                p[1] for p in (left, right, top, bottom))

            out.append(Detection(
                label=label,
                confidence=1.0 if fresh else 0.5,
                cx_px=cx_px, cy_px=cy_px,
                w_px=size_x, h_px=size_y,
                x_cm=x_cm, y_cm=y_cm,
                radius_cm=radius_cm,     # ★ 마커 크기가 아니라 설정값을 쓴다
            ))

        return out

    def draw(self, frame, world, detections: list[Detection]) -> None:
        """실제 회피 반경을 원으로 표시 (마커 크기와 혼동하지 않도록)."""
        for d in detections:
            col = (0, 255, 255) if d.role == "cup" else (80, 80, 255)
            if d.confidence < 1.0:
                col = (120, 120, 120)   # hold 중 (실제로는 안 보이는 상태)
            points = np.rint(project_circle(world, d.x_cm, d.y_cm,
                                            d.radius_cm)).astype(np.int32)
            cv2.polylines(frame, [points], True, col, 1)

    @staticmethod
    def pick_cup(detections: list[Detection]) -> Detection | None:
        cups = [d for d in detections if d.role == "cup"]
        return max(cups, key=lambda d: d.confidence) if cups else None

    @staticmethod
    def pick_obstacles(detections: list[Detection]) -> list[Detection]:
        return [d for d in detections if d.role == "obstacle"]
