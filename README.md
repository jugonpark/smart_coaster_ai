# D.I.G — 사람 회피 옴니 로봇

천장 카메라로 **컵 / 장애물 / 사람 / 로봇**을 동시에 추적하고,
사람 손이 컵에 위험하게 접근하면 로봇이 감속·정지·우회하도록 만드는 시스템.

```
┌─────────────────────── PC (Python) ────────────────────────┐   UDP    ┌── ESP32 (C++) ──┐
│ [1] 인식      [2] 판단      [3] 경로계산      [4] 전송      │ ───────► │ [5] 제어         │
│  YOLO          손목-컵      potential field   JSON 직렬화   │  30Hz    │  ArduinoJson     │
│  Pose          거리 +       월드 속도벡터      UDP 소켓      │          │  3륜 역기구학     │
│  Hand          접근속도     → 로봇좌표 회전                 │          │  엔코더 PID       │
│  Face          (+그립/시선                                 │          │  타임아웃 정지     │
│  ArUco          = 보조신호)                                │          │                  │
└────────────────────────────────────────────────────────────┘          └──────────────────┘
```

---

## 파일 구조

```
config.py                 ← 모든 튜닝 값이 여기 한 곳에 있다. 대부분의 조정은 이 파일만 건드리면 됨
main.py                   ← 통합 루프 (1~4계층 실행)

perception/               [1] 인식 계층
  camera.py                 카메라 캡처 + 픽셀↔월드(cm) 좌표 변환
  object_detector.py        YOLO (Roboflow 학습 모델) → 컵 / 장애물
  pose_tracker.py           MediaPipe Pose → 손목 / 팔꿈치 / 어깨
  hand_tracker.py           MediaPipe Hand → 손 21점 스켈레톤 + 그립 모양
  gaze_tracker.py           MediaPipe Face → 머리 자세 (시선 대용)
  robot_tracker.py          ArUco → 로봇 위치 + heading + 스케일 추정

decision/                 [2] 판단 계층
  risk_evaluator.py         손목-컵 거리 + 접근속도 + TTC → SAFE / WARN / DANGER
                            (그립·시선은 임계값을 넓히는 보조 신호로만 사용)

planning/                 [3] 경로계산 계층
  potential_field.py        인력(컵) + 척력(장애물·사람) → 월드 속도벡터
  transform.py              월드 → 로봇 좌표계 회전변환 + heading P제어

comm/                     [4] 전송 계층
  udp_sender.py             {vx, vy, w, status} JSON → UDP (논블로킹, 30Hz)

ui/settings_panel.py      실시간 튜닝 패널 (슬라이더). 별도 프로세스로 실행
tuning.json               패널에서 저장한 값. 다음 실행 때 자동 적용 (없으면 config 기본값)

esp32/esp32_omni_controller/
  esp32_omni_controller.ino [5] 제어 계층 (Arduino IDE로 업로드)

tools/selftest.py            하드웨어 없이 [2][3][5] 로직 검증 (47개 테스트)
tools/check_camera.py        카메라 + 인식 5종을 계층별로 켜고 끄며 눈으로 확인
tools/make_marker.py         인쇄용 ArUco 마커 생성 (실제 크기 보장)
tools/geometry_check.py      카메라 설치 각도/높이 검토 (스케일 편차·시차 계산)
tools/udp_monitor.py         ESP32 없이 전송 계층 확인용
tools/download_models.py     MediaPipe 모델 다운로드 (pose/hand/face)
models/                   Roboflow .pt 와 MediaPipe .task 파일들을 여기 둔다
legacy/                   이전 빠름/느림 분류 프로토타입 (참고용, 현재 파이프라인에서 미사용)
```

---

## 좌표계 규약 (가장 중요)

세 계층이 전부 이 규약 위에서 동작한다. 여기가 어긋나면 로봇이 반대로 간다.

**월드 좌표계** — 카메라 이미지 평면, 단위 cm, **Y축은 위쪽이 양수**

```
world_x = px_x / px_per_cm
world_y = (FRAME_HEIGHT - px_y) / px_per_cm
```

`px_per_cm`은 매 프레임 ArUco 마커의 실제 픽셀 크기에서 추정된다
(`config.MARKER_SIZE_CM`을 **반드시 실측값으로** 맞출 것).

**로봇 heading** — 월드 +X축 기준 반시계(CCW) 양수.
ArUco 마커의 위쪽 변 `TL→TR` 방향이 로봇 정면.
실제 부착 방향이 다르면 `config.MARKER_HEADING_OFFSET_DEG`로 보정.

**로봇 좌표계** — `vx` = 정면 전진, `vy` = 좌측 횡이동, `w` = 반시계 회전.

> `config.FLIP_HORIZONTAL = False`를 유지할 것. 좌우 반전하면 월드가 왼손 좌표계가 되어
> heading과 회전 방향이 전부 뒤집힌다. (기존 프로토타입은 셀피 뷰라 반전했었다)

---

## 준비

### 1) 파이썬 환경

**검증된 조합** (Python 3.14.6 / Windows 11에서 실제 설치·임포트 확인):

| 패키지 | 버전 | 비고 |
|---|---|---|
| Python | 3.14.6 | 3.10 이상이면 동작. 3.14도 전부 wheel 제공 |
| opencv-python | 4.14.0.94 | **contrib 아님.** ArUco는 4.7부터 기본 패키지 포함 |
| mediapipe | 1.0.0 | **0.10.30 이상 필수** (Tasks API) |
| numpy | 2.5.1 | 버전 고정 금지 |
| ultralytics | 8.4.115 | YOLO backend용 |
| torch | 2.13.0 | ultralytics가 자동 설치 (~2~3GB) |

```bash
python -m venv .venv
```
```bash
.venv\Scripts\activate
```
```bash
pip install -r requirements.txt
```

> **`opencv-contrib-python`을 따로 깔지 말 것.** ArUco는 4.7부터 contrib에서 main
> 저장소(objdetect)로 옮겨져 일반 `opencv-python`에 포함되어 있다. contrib를 추가로 깔면
> ultralytics가 요구하는 `opencv-python`과 같은 `cv2` 이름을 두고 충돌해서,
> 나중에 설치된 쪽이 이기는 예측 불가능한 상태가 된다.

**MediaPipe API 관련 (중요)** — 이 프로젝트는 **Tasks API**로 작성되어 있다.
기존 프로토타입이 쓰던 `mp.solutions.pose` / `mp.solutions.holistic` (legacy Solutions API)는
mediapipe 0.10.30 이후 패키지에서 **완전히 제거**되었다. 현재 설치 가능한 모든 버전
(0.10.30 ~ 1.0.0)에는 `mediapipe.tasks`만 들어 있고 `solutions` 디렉터리 자체가 없다.
`drawing_utils`도 함께 사라져서 관절 그리기는 `cv2`로 직접 한다.

### 2) MediaPipe 모델 다운로드

Tasks API는 모델 가중치가 패키지에 포함되지 않는다. 한 번만 받으면 된다:

```bash
python tools/download_models.py
```

`models/`에 세 개가 생긴다 (총 약 18MB):

| 파일 | 용도 | 없으면 |
|---|---|---|
| `pose_landmarker_lite.task` | 손목/팔꿈치/어깨 | **필수** — 없으면 실행 불가 |
| `hand_landmarker.task` | 손 21점 스켈레톤 + 그립 모양 | 그립 신호만 꺼짐 |
| `face_landmarker.task` | 머리 자세 (시선 대용) | 시선 신호만 꺼짐 |

뒤의 둘은 **선택**이다. 안전 판정은 거리·속도만으로 이뤄지므로 없어도 시스템은 정상
동작하고, 콘솔에 "비활성화" 안내만 뜬다. 특정 모델만 받으려면 `pose` / `hand` / `face`
인자를 주면 되고, 더 정확한 pose가 필요하면 `pose_full` / `pose_heavy`도 있다.

### 3) Roboflow 모델 준비

1. Roboflow에서 `cup`, `obstacle` 등 클래스를 라벨링해 학습
2. **Deploy → Download weights → YOLOv8/v11 (.pt)**
3. 받은 파일을 `models/dig_yolo.pt`로 저장
4. `config.py`에서 클래스 이름을 실제 데이터셋에 맞춘다

```python
CLASS_CUP = "cup"
CLASS_OBSTACLE_NAMES = ("obstacle", "box", "bottle", "chair")
```

모델이 아직 없으면 `YOLO_BACKEND = "stub"`로 두고 나머지 계층부터 검증할 수 있다.

### 4) ArUco 마커

- `DICT_4X4_50`의 **ID 0**을 출력해 로봇 위에 붙인다
- 출력된 마커 한 변을 자로 재서 `config.MARKER_SIZE_CM`에 입력 (거리 계산 정확도가 전부 여기 달려 있다)

### 5) ESP32

`esp32/esp32_omni_controller/esp32_omni_controller.ino`를 열고 수정:

| 항목 | 위치 |
|---|---|
| WiFi SSID / 비밀번호 | `WIFI_SSID`, `WIFI_PASS` |
| 고정 IP (`config.ESP32_IP`와 일치) | `LOCAL_IP` |
| 바퀴 반지름 / 로봇 반지름 | `WHEEL_RADIUS_CM`, `ROBOT_RADIUS_CM` |
| 엔코더 CPR / 감속비 | `ENCODER_CPR`, `GEAR_RATIO` |
| 모터·엔코더 핀 | `PIN_PWM`, `PIN_IN1/IN2`, `PIN_ENC_A/B` |
| 바퀴 장착각 | `WHEEL_ANGLE_DEG` (기본 0/120/240°) |

라이브러리 매니저에서 **ArduinoJson v7** 설치 필요.

---

## 실행

먼저 하드웨어 없이 로직부터 검증한다 (외부 패키지 불필요, 표준 라이브러리만 사용):

```bash
python tools/selftest.py
```

좌표 회전변환 부호, potential field 게인 스케일, 위험 판정 히스테리시스,
3륜 역기구학, JSON 패킷 형식까지 47개 항목을 검사한다. 전부 PASS여야 한다.

본 실행:

```bash
python main.py
```

`q` / `ESC` 종료 · `스페이스` 일시정지

ESP32 없이 먼저 확인하려면 `config.ESP32_IP = "127.0.0.1"`로 바꾸고:

```bash
python tools/udp_monitor.py
```

---

## 실시간 튜닝 패널

`main.py`나 `tools/check_camera.py` 실행 중 화면 우상단 **SETTINGS** 버튼을 클릭하거나
**`t` 키**를 누르면 슬라이더 창이 열린다. 값을 움직이면 **다음 프레임부터 즉시 반영**된다.

| 탭 | 내용 |
|---|---|
| 판단 | DANGER/WARN 거리, 접근속도 임계, TTC, 스무딩, hold 시간 |
| 경로계산 | 인력·척력 게인, 영향 반경, 힘→속도 게인, 지역최소 탈출 |
| 속도·자세 | 최대 속도/가속도/각속도, heading P게인·불감대 |
| 인식 | YOLO 신뢰도, 관절 신뢰도, 로봇자세 스무딩, YOLO 추론 주기 |
| 속도배율 | SAFE / WARN 배율 |

창 하단에 **현재 측정값**(손-컵 거리, 접근속도, TTC, 위험등급, 로봇속도)이 실시간으로
표시되므로, 실제 숫자를 보면서 임계값을 정할 수 있다.

- **저장** → `tuning.json`. 다음 실행 때 자동으로 불러온다
- **기본값으로 초기화** → `config.py` 원본 값으로 되돌린다
- `config.py`는 절대 수정되지 않는다. `tuning.json`을 지우면 기본값으로 복귀

> **DANGER 속도배율은 UI에 없다.** 0으로 고정이고, `tuning.json`을 손으로 고쳐 넣어도
> 로드 시점에 0으로 강제된다. 안전 정지 기능이라 실수로 올릴 수 있는 경로를 두지 않았다.

### 구현 메모 — 왜 별도 프로세스인가

패널은 자식 프로세스로 뜬다. 부모(영상 루프)는 `tkinter`를 import조차 하지 않는다.

처음엔 워커 스레드에서 Tk를 띄웠는데, tkinter 위젯들이 서로 참조 순환을 만들기 때문에
참조 카운트로는 해제되지 않고 결국 **순환 GC**가 치운다. 순환 GC는 어느 스레드에서든
돌 수 있어서, 메인 스레드가 Tcl 객체를 해제하는 순간
`Tcl_AsyncDelete: async handler deleted by the wrong thread`로 **프로세스가 즉시 죽는다**
(`gc.collect()` 한 번으로 재현됨). 정리 코드로는 막을 수 없다 — GC 타이밍을 통제할 수 없으니까.

프로세스를 나누면 이 문제가 원천적으로 사라진다. 통신은 stdout/stdin JSON 한 줄씩이고,
오가는 건 순수 문자열·숫자뿐이라 스레드 안전 문제가 없다.

## 계층별 동작

### [2] 판단 계층 — 위험 판정

두 손목 중 컵에 더 가까운 쪽의 거리 `d`와 접근속도 `v = -dd/dt`, 그리고 `TTC = d/v`로 판정한다.

| 등급 | 조건 | 로봇 |
|---|---|---|
| `DANGER` | `d < 15cm` 또는 (`v > 25cm/s` 且 `TTC < 0.8s`) | 정지 (`STOP`) |
| `WARN` | `d < 35cm` 또는 (`v > 25cm/s` 且 `TTC < 1.6s`) | 40% 감속 (`SLOW`) |
| `SAFE` | 그 외 | 정상 주행 (`RUN`) |

거리·속도는 EMA로 스무딩하고, **등급 상승은 즉시 / 하강은 hold 시간 뒤에** 반영한다
(안전 방향을 우선하고 SAFE↔DANGER 채터링을 막기 위함).

#### 의도 신호 (그립 모양 · 시선)

그립과 시선은 **임계값을 넓히는 역할만** 한다. 둘 중 하나라도 감지되면
거리 임계에 `INTENT_DIST_BOOST`(1.5), TTC 임계에 `INTENT_TTC_BOOST`(1.4)를 곱해
**더 일찍** 반응한다. 예를 들어 WARN 거리가 35cm → 52.5cm로 넓어진다.

- **그립**: 손 21점의 `hand_world_landmarks`로 엄지-검지 벌림(`aperture`)과
  손 펼침(`openness`)을 손 크기로 정규화해 계산. 편 손도 주먹도 아닌 중간 상태를
  "잡기 준비"로 본다. 시점에 거의 독립적이다.
- **시선**: 홍채가 아니라 **머리 방향**을 쓴다. 1~2m 거리에서 홍채 추정은 노이즈가
  너무 크다. `FaceLandmarker`의 얼굴 변환행렬에서 yaw/pitch/roll과 정면 벡터를 뽑고,
  머리→컵 방향과의 각도가 `GAZE_CONE_DEG`(35°) 안이면 "보는 중"으로 판정한다.

> **의도 신호는 절대 하드 정지를 트리거하지 않는다.** 이유가 두 가지다.
> ① 신뢰도가 낮다 — 사람이 잠깐 쳐다봤다고 로봇이 서면 쓸 수 없는 시스템이 된다.
> ② **fail-safe가 아니다** — 얼굴이 가려지거나 손이 안 잡히면 "위험 없음"으로
> 조용히 읽힌다. 안전 기능이 침묵으로 실패하는 건 최악의 형태다.
> 그래서 실제 판정은 운동학(거리·속도)만 내리고, 의도는 그 임계값을 키우기만 한다.
> 위험도를 낮추는 방향으로는 절대 작용하지 않는다.

### [3] 경로계산 계층 — Potential Field

- **인력(컵)**: 가까우면 quadratic well, `PF_ATTRACT_MAX_CM` 밖은 conic well로 전환해 인력 폭주 방지.
  최대 크기는 `PF_K_ATTRACT × PF_ATTRACT_MAX_CM = 72` — 척력 게인은 전부 이 값과 비교해서 읽으면 된다.
- **척력(장애물·사람)**: **정규화 거리** `d̂ = d/d₀`를 쓴 `F = k(1/d̂ − 1)/d̂²`, 영향 반경 안에서만 작용

  > 교과서 형태인 `k(1/d − 1/d₀)/d²`를 그대로 쓰면 게인이 `d₀`에 **세제곱으로** 의존한다.
  > `k=900, d₀=45cm`로 놔도 30cm에서 척력이 **0.011**밖에 안 나와 인력 72에 완전히 묻히고,
  > 척력이 인력을 넘어서는 건 `d < 2cm`일 때다 — 즉 **로봇이 장애물을 뚫고 돌진한다.**
  > `d̂`로 정규화하면 게인이 영향 반경과 무관해지고 인력과 직접 비교된다
  > (`d̂=0.5 → 4k`, `d̂=0.25 → 48k`). `tools/selftest.py`가 이 회귀를 잡아준다.

- 척력 거리는 **물체 표면까지**(중심거리 − 바운딩박스 반경)로 계산해 큰 물체도 제대로 피한다
- **힘 → 속도**는 고정 게인(`PF_FORCE_TO_SPEED`) 후 clamp. 힘을 항상 최대속도로 정규화하면
  장애물 때문에 합력이 줄어도 여전히 전속력이 나와서 "감속하며 접근"이 사라진다
- 사람 척력은 위험 등급에 따라 증폭 (`SAFE 1.0 / WARN 1.8 / DANGER 3.0`)
- 합력이 거의 0인데 목표에 못 갔으면 **지역 최소점**으로 보고 목표 방향의 수직 성분을 섞어 탈출
- 가속도 제한(`MAX_LINEAR_ACCEL_CM_S2`)으로 명령이 튀지 않게 한다

### [5] 제어 계층 — 3륜 옴니 역기구학

바퀴 `i`의 장착각을 `αᵢ`(정면에서 반시계)라 할 때:

```
vᵢ = −sin(αᵢ)·vx + cos(αᵢ)·vy + L·w
```

0/120/240° 배치 검증:
- 순수 전진 → `v = (0, −0.866vx, +0.866vx)` (0° 바퀴 정지) ✓
- 순수 회전 → 세 바퀴 모두 `+L·w` (같은 방향) ✓

한 바퀴라도 포화하면 **세 바퀴를 같은 비율로** 줄인다. 개별로 자르면 합성 속도의 방향이 틀어진다.

---

## 안전장치

| 계층 | 장치 |
|---|---|
| 판단 | `DANGER` → `SPEED_SCALE_BY_RISK["DANGER"] = 0.0` (속도 0) |
| 경로계산 | ArUco 로봇 마커를 놓치면 속도를 만들지 않음 (위치를 모르는 채 움직이지 않음) |
| 전송 | 논블로킹 소켓 — 네트워크가 막혀도 영상 루프가 멈추지 않음 |
| 전송 | `STOP`은 전송 주기를 무시하고 즉시 전송 |
| 전송 | 종료·예외 시 `finally`에서 STOP을 5회 반복 전송 |
| **제어** | **`CMD_TIMEOUT_MS`(300ms) 무수신 → 즉시 정지.** PC가 죽어도 로봇은 선다 |
| 제어 | WiFi 끊김 감지 시 정지 |
| 제어 | 부팅 직후 `estopped = true` — 첫 패킷 전까지 움직이지 않음 |
| 제어 | 오래된 `seq` 패킷 폐기 (UDP는 순서를 보장하지 않음) |

---

## 튜닝 순서 (권장)

Galaxy HTTP/MJPEG 영상을 PC에서 직접 사용할 때는 카메라 앱이 표시한 실제 주소를
`GRISE_CAMERA_URL`에 지정한다. Windows 명령 프롬프트 예:

```cmd
set GRISE_CAMERA_URL=http://갤럭시IP:포트/영상경로
python tools\check_camera.py
```

주소를 지정하지 않으면 기존 USB `CAMERA_INDEX`를 사용한다. 두 입력 모두
`Camera.read()`에서 같은 BGR 프레임으로 전달된다. 보정 파일의 해상도 검증은
설정값이 아닌 실제 수신 프레임 크기를 사용한다.

1. **좌표계부터.** `YOLO_BACKEND="stub"`, ESP32 끄고 `main.py` 실행 → HUD에서 `scale px/cm`와 로봇 heading 화살표가 맞는지 확인. 마커를 돌려서 화살표가 같이 도는지 본다.
2. **판단 계층.** 손을 컵에 천천히/빠르게 가져가며 HUD의 `hand-cup`, `approach`, `TTC` 값을 보고 `RISK_*` 임계값을 조정.
3. **경로계산.** `tools/udp_monitor.py`로 속도벡터가 사람을 피해 도는지 확인. 사람에 너무 붙으면 `PF_K_REPULSE_HUMAN` ↑, 목표에 못 가면 `PF_K_ATTRACT` ↑.
4. **제어.** 로봇을 들어올린 상태로 PID 튜닝. `KP`부터 올리고 정상상태 오차가 남으면 `KI`, 진동하면 `KD`.
5. 마지막에 바닥에 내려놓고 저속(`MAX_LINEAR_SPEED_CM_S`를 15 정도로 낮춰서)부터 시작.

---

## legacy/

기존 빠름/느림 2클래스 분류 프로토타입(`data_collector.py` / `train.py` / `realtime_predict.py` /
`motion_dataset.csv`)이 들어 있다. 현재 판단 계층은 학습 모델 대신 **거리 + 접근속도 규칙**으로
동작하므로 파이프라인에서 사용하지 않는다. 임계값 결정용 데이터가 필요하면 참고할 것.
