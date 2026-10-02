# GRISE Raspberry Pi Central Controller

이 폴더가 기존 노트북 `grise-master`의 **중앙 관제 역할을 Raspberry Pi로 이전하기 위한 기준 구조**입니다.

```text
main.py
config.py
perception/
├─ object_detector.py      # 기존 grise 기반
├─ robot_tracker.py        # 기존 camera/marker_scanner/robot_tracker 통합
├─ network_camera.py       # 노트북 MJPEG 수신
└─ radar.py                # 현재는 인터페이스/UDP placeholder
fusion/
└─ sensor_fusion.py
decision/
└─ risk_evaluator.py       # 기존 위험 기준을 radar 입력 형태로 이전
planning/
├─ escape_planner.py       # 8방향 후보 기반 MVP
└─ transform.py            # 기존 grise 그대로
safety/
└─ safety_manager.py
comm/
├─ udp_sender.py           # 기존 grise 그대로
├─ telemetry_receiver.py
└─ monitor_sender.py
```

## 설계 원칙

- Pi: 인식, 센서융합, 위험판단, 회피방향/거리 결정, 최종 안전검사
- ESP32: 3WD 역기구학, Encoder, PID, PWM, watchdog
- Laptop: 카메라 스트리밍 + 모니터링만

## 최초 실행 순서

1. 노트북 `main.py` 실행 후 MJPEG 주소 확인.
2. Pi에서 `GRISE_LAPTOP_IP`를 실제 노트북 IP로 설정.
3. 처음에는 `GRISE_ENABLE_ESCAPE_MOTION=0` 상태로 관제 로그만 검증.
4. ESP32 telemetry 수신까지 확인한 뒤 실제 모션을 단계적으로 활성화.
5. radar.py는 이후 IWR6843AOP 실제 parser/transport로 교체.

### 중요한 값

기존 grise에서 유지한 초기값: 1280x720@30fps, ArUco ID 0/8cm, YOLO conf 0.45,
ESP32 10.182.7.50:8888, command 30Hz, risk 15/35cm, approach 25cm/s, TTC 0.8/1.6s,
Pi max speed 35cm/s / 1.8rad/s.

새 MVP 초기값: WARN 15cm 회피, DANGER 30cm 회피. 이 값들은 실제 실험 후 반드시 튜닝합니다.

## 컵 ID1 + 손목 접근 회피 (소프트웨어 연결)

현재 Pi는 한 카메라 프레임의 `DICT_4X4_50` 스캔을 공유합니다. ID0은 로봇,
ID1은 컵입니다. `shared_ai`의 Pose/Hand/Gaze를 Windows 미리보기와 그대로
공유합니다. 컵과 가장 가까운 손목을 기준으로 거리·접근속도·TTC를 계산하고,
기존 radar risk와 더 위험한 쪽을 선택합니다. Grip/gaze는 기록용 보조 evidence이며
WARN/DANGER를 SAFE로 낮추지 않습니다.

HAND WARN의 첫 실물 시험 값은 `SLOW`, 10cm/s, 10cm이고, HAND DANGER는
`RUN`, 15cm/s, 15cm입니다. Pi의 기존 `MotionGoals`와 `UdpSender`가
`cmd_move`를 30Hz로 보내며 ESP32가 IK/encoder/PID/거리 정지를 담당합니다.
기본 `GRISE_ENABLE_ESCAPE_MOTION=0`이면 계획은 계산하지만 STOP만 송신합니다.
레이더가 없을 때의 `GRISE_ALLOW_MOTION_WITHOUT_RADAR=1`은 **TEST ONLY**이며
기본값 0입니다. 켜면 시작 로그에 `TEST ONLY - RADAR BYPASS ACTIVE`가 나옵니다.
이 모드에서도 유효한 카메라·테이블 calibration·ID0·ID1·손목·ESP32 telemetry·
CLEAR 회피 방향이 필요합니다. ArUco 장애물은 ID10~13만 인식하므로 첫 시험은
장애물이 없는 격리된 공간에서 수행해야 합니다.

현재 Galaxy 영상이 640x480이라면 Pi 실행 전에 `GRISE_FRAME_WIDTH=640`,
`GRISE_FRAME_HEIGHT=480`을 설정하고, **그 해상도로 실제 테이블에서 만든**
`calibration/table_calibration.json`을 배치합니다. 예제 calibration은 실측값이
아닙니다. MediaPipe 모델 3개는 `python tools/download_ai_models.py`로 받아
Pi의 저장소 루트 `models/`에 둡니다. `pip install -r raspberry_pi/requirements.txt`가
필요합니다.

### 단계별 실물 검증 (PowerShell 예시)

아래 IP는 현재 예시입니다. `ipconfig`와 ESP32 `STATUS`에서 실제 IP를 확인해
바꿉니다. 먼저 노트북에서 Galaxy 스트림을 중계합니다.

```powershell
$env:GRISE_CAMERA_SOURCE="network"
$env:GRISE_CAMERA_URL="http://10.232.69.154:8080/"
$env:NO_PROXY="10.232.69.154"
python laptop/main.py
```

Pi 또는 동일 네트워크의 테스트 컴퓨터에서 다른 터미널로 중앙 제어를 실행합니다.
Pi Linux라면 아래 환경변수를 `export NAME=value` 형식으로 설정합니다.

```powershell
$env:GRISE_CAMERA_URL="http://10.232.69.213:8080/stream.mjpg"
$env:NO_PROXY="10.232.69.213"
$env:GRISE_ESP32_IP="10.232.69.103"
$env:GRISE_FRAME_WIDTH="640"
$env:GRISE_FRAME_HEIGHT="480"
$env:GRISE_TABLE_CALIBRATION="calibration/table_calibration.json"
$env:GRISE_ALLOW_MOTION_WITHOUT_RADAR="1" # TEST ONLY; 실제 radar 준비 시 0
$env:GRISE_ENABLE_ESCAPE_MOTION="0"      # Phase A: shadow / STOP only
python raspberry_pi/main.py
```

- **Phase A:** motion=0에서 ID0, ID1, wrist, hand 거리, risk source/level,
  threat direction, escape direction, STOP 이유를 확인합니다. 하나라도 UNKNOWN이면
  다음 단계로 넘어가지 않습니다.
- **Phase B:** 바퀴를 공중에 띄우고 ESP32 telemetry가 최신이며 장애물이 없는
  상태에서만 motion=1로 재시작합니다. WARN 10cm/s·10cm, DANGER 15cm/s·15cm,
  encoder/goal progress/`goal_reached`를 확인합니다.
- **Phase C:** 바닥에서 10/15cm bounded move와 STOP 거리를 확인합니다.
- **Phase D:** 손 접근에 따른 자동 회피를 시험합니다. 각 단계가 실패하면 STOP하고
  다음 단계로 넘어가지 않습니다.

소프트웨어 테스트 통과는 실제 cm 정확도, 손의 높이에 따른 시차, 바닥 마찰,
장애물 회피 및 모터 동작을 검증하지 않습니다.
