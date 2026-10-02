# Hand-Cup Escape Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 컵 ArUco ID 1에 손이 접근하면 Pi가 손–컵 위험을 기존 레이더 위험과 병합하고, 검증된 유한 거리 회피 명령을 ESP32에 보낸다.

**Architecture:** Laptop은 카메라와 모니터링만 담당한다. Pi가 한 영상 프레임에서 마커·Pose/Hand/Gaze를 인식해 WorldState를 만들고 위험을 판정한다. Pi의 기존 planner, SafetyManager, MotionGoals, UdpSender를 재사용하며 ESP32 펌웨어는 수정하지 않는다.

**Tech Stack:** Python 3, OpenCV ArUco, MediaPipe Tasks, 기존 Pi typed UDP `cmd_move`/telemetry.

**Spec:** [hand-cup-escape.md](../../design/hand-cup-escape.md)

## Global Constraints

- `GRISE_ENABLE_ESCAPE_MOTION=0`을 기본으로 유지한다.
- `grip_on`과 gaze는 위험 등급을 낮추지 않는다.
- Pi가 직접 wheel target/PWM을 계산하지 않는다.
- ESP32와 Arduino 미커밋 파일은 수정하지 않는다.
- 실제 로봇 구동은 이 소프트웨어 계획의 검증 범위 밖이다.

## Review Focus

- 손이 컵 마커를 가리는 프레임: 과거 컵 위치로 새 회피를 시작하지 않는 테스트.
- MediaPipe 추론 지연이 `WORLD_STALE_S=0.35`를 넘을 때 STOP 테스트.
- 레이더 위험과 손–컵 위험이 동률일 때 source·방향이 결정적으로 선택되는 테스트.
- 목표 완료 직후 같은 perception revision에서 `motion_id`를 재발행하지 않는 테스트.
- grip/gaze만 감지되고 손–컵 거리가 안전할 때 위험 등급이 내려가지 않는 테스트.

---

### Task 1: 한 프레임의 컵·로봇·손 인식

**Files:** `raspberry_pi/perception/robot_tracker.py`, `raspberry_pi/perception/marker_objects.py`, `raspberry_pi/main.py`, `raspberry_pi/requirements.txt`, `tests/test_hand_cup_escape.py`

**Interfaces:** `detect_marker_objects(scan, world, captured_at=None) -> list[Detection]`; `RobotTracker.process(frame, world, scan=None) -> RobotPose`. `shared_ai`의 Pose/Hand/Gaze 모듈을 Pi에서도 그대로 import한다.

- [ ] 한 프레임의 ID 0과 ID 1이 올바른 월드 좌표를 만들고, ID 1이 빠진 프레임은 cup=None인 테스트를 작성하고 실패를 확인한다.
- [ ] `MarkerScanner.scan()`을 프레임당 한 번 실행해 두 인식기에 전달한다. 컵 반경 초기값은 4cm이고 현재 `DICT_4X4_50`을 유지한다.
- [ ] Pose/Hand/Gaze 모델 파일 부재 또는 추론 실패를 `detector unavailable`로 기록하고 WorldState를 invalid로 발행한다. 모델 경로와 Pi 의존성을 문서화한다.
- [ ] `python -m pytest -q -p no:cacheprovider tests/test_hand_cup_perception.py`가 통과하는지 확인한다.

### Task 2: 손–컵 evidence와 WorldState

**Files:** `shared_ai/hand_cup.py`, `raspberry_pi/perception/vision_state.py`, `raspberry_pi/fusion/sensor_fusion.py`, `tests/test_hand_cup_evidence.py`, `tests/test_hand_cup_escape.py`

**Interfaces:** `HandCupEvaluator.update(human, cup, now, hands=None, gaze=None) -> HandCupEvidence`; `VisionState.hand_cup`와 `WorldState.hand_cup`은 동일 프레임의 불변 snapshot이다.

- [ ] 가까움 10cm + grip=True가 DANGER이고, 빠른 접근 TTC, 입력 상실 후 미분 이력 초기화, NaN, stale timestamp를 검사한다.
- [ ] 보정이 유효한 경우에만 손–컵 cm 값을 WorldState로 내보낸다. 미보정/해상도 불일치/모델 실패는 UNKNOWN·invalid이다.
- [x] `python -m pytest -q -p no:cacheprovider tests/test_hand_cup_evidence.py tests/test_hand_cup_escape.py`를 실행한다.

### Task 3: 레이더와 손 위험을 병합해 방향 선택

**Files:** `raspberry_pi/decision/risk_evaluator.py`, `raspberry_pi/planning/escape_planner.py`, `raspberry_pi/core/control_loop.py`, `tests/test_hand_cup_escape.py` (new)

**Interfaces:** `RiskEvaluator.evaluate(world) -> RiskState`가 source, level, threat_direction, timestamp를 제공한다. Planner는 선택된 위험의 direction을 사용하되 기존 8방향 CLEAR 검사와 거리/속도 제한을 그대로 적용한다.

- [ ] 손 DANGER + 레이더 NO_TARGET이 반대 방향 회피 계획을 만들고, radar DANGER가 손 WARN보다 우선하며, 동률이 결정적인 테스트를 먼저 작성한다.
- [ ] grip/gaze가 위험도를 내리지 않도록 병합하고, 유효한 위협 방향이 없거나 모든 sector가 막히면 STOP을 반환한다.
- [ ] 기존 레이더 테스트를 포함해 `python -m pytest -q -p no:cacheprovider tests/test_hand_cup_escape.py tests/test_fusion_risk.py tests/test_escape_safety.py`를 실행한다.

### Task 4: ESP32 명령 및 STOP 통합

**Files:** `raspberry_pi/safety/safety_manager.py`, `raspberry_pi/core/motion_goal.py` (기존 재사용), `raspberry_pi/core/control_loop.py`, `tests/test_hand_cup_escape.py`

**Interfaces:** 기존 `UdpSender.send(..., motion_id, target_distance_cm)`를 사용한다. HAND WARN은 초기 10cm/s·10cm·SLOW, HAND DANGER는 15cm/s·15cm·RUN이며 선택된 방향의 body-frame `vx/vy`로 보낸다.

- [ ] 유효한 calibration·camera·AI·radar·telemetry·CLEAR sector·motion enabled 모두 충족 시에만 `cmd_move`가 나오는 모의 UDP 테스트를 작성한다.
- [ ] 각 결함, `goal_reached`, 위협 소멸, shutdown에서 STOP과 목표 재시작 방지를 검증한다.
- [ ] 기존 SafetyManager/MotionGoals gate를 통과시키고 `session_id`·`seq`·`motion_id` 소유권은 기존 코드에 둔다.
- [ ] `python -m pytest -q -p no:cacheprovider tests`와 `git diff --check`를 실행한다.

### Task 5: Shadow run과 실물 검증 준비

**Files:** `raspberry_pi/README.md`, `tools/AI_PREVIEW.md`, 필요 시 monitor payload/test.

- [ ] 모션 비활성 상태에서 손–컵 거리·위험 등급·선택 방향·STOP 이유가 로그에 보이도록 한다.
- [ ] Galaxy 영상에 실제 마커 ID 0/1과 손이 보이는 조건에서 overlay, calibration, Pi shadow 상태를 확인한다.
- [ ] 모델/마커/레이더/telemetry/camera 경계를 기록하고, 소프트웨어 검증과 하드웨어 미검증을 구분해 보고한다.

## 현재 완료와 미완료

- 완료: Windows 공용 Pose/Hand/Gaze 오버레이, ID0/ID1 단일 스캔, HandCupEvaluator, Pi WorldState, 레이더/손 위험 병합, 기존 Planner/SafetyManager/MotionGoals/UdpSender를 통한 소프트웨어 `cmd_move` 경로와 회귀 테스트.
- 미완료: 실제 테이블 보정, Pi 모델 설치, 실제 손/컵 인식 시연, ESP32 텔레메트리 및 물리 이동 시험. 모션 기본값은 계속 0이다.
