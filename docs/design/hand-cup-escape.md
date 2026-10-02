# 컵 ArUco와 손 접근에 따른 Pi 회피 동작 규격

## 목표와 현재 경계

컵의 ArUco ID 1과 MediaPipe 손목을 같은 보정된 테이블 좌표(cm)로 변환한다.
손목이 컵에 가까워지거나 빠르게 접근하면 Pi가 WARN/DANGER를 판정한다.
기존 레이더 위험과 합쳐 더 높은 위험을 선택하고, Pi의 기존 8방향 회피 계획과
최종 SafetyManager를 거쳐 ESP32에 유한 거리 이동 명령을 보낸다.

현재 `shared_ai/hand_cup.py`와 Pi의 WorldState, 위험 병합, 회피 계획,
기존 UDP `cmd_move` 경로가 소프트웨어로 연결되어 있다. Windows 오버레이에는
Pose/Hand/Gaze 및 ArUco ID가 표시된다. 실제 테이블 보정·손 위치 정확도·
ESP32 이동은 아직 검증되지 않았다.

## 불변 안전 조건

- 팀 저장소의 `grip_on → SAFE` 규칙은 사용하지 않는다. grip/gaze는 로그와
  보조 evidence이며 위험 등급을 낮출 권한이 없다.
- ESP32 firmware, 핀, 모터 PID, UDP typed protocol, watchdog은 변경하지 않는다.
- `GRISE_ENABLE_ESCAPE_MOTION=0`이 기본이다. 이 값이 0이면 Pi는 계획을
  계산하더라도 STOP만 보낸다.
- 손–컵 위험으로 인한 이동에도 유효한 테이블 calibration, 로봇 ArUco ID 0,
  최신 카메라·마커 인식·MediaPipe, 유효한 레이더 상태와 ESP32 telemetry가
  모두 필요하다. 하나라도 빠지면 STOP이다. 레이더만 명시적인
  `GRISE_ALLOW_MOTION_WITHOUT_RADAR=1` TEST ONLY 모드에서 예외로 둘 수 있다.
- 손목은 테이블 평면 위에 있지 않을 수 있어 homography 변환에 시차 오차가
  생긴다. 소프트웨어 테스트만으로 15/35cm 임계값의 물리적 정확도를 주장하지 않는다.

## 데이터 흐름

1. Laptop Galaxy MJPEG → Pi `NetworkCamera`.
2. Pi에서 한 프레임의 ArUco scan을 공유한다. ID 0은 로봇, ID 1은 컵이다.
   Marker ID 1의 좌표는 컵 위치의 측정치이며, 없는 동안 과거 위치로 회피를
   시작하지 않는다.
3. 같은 프레임에 공유 `shared_ai` Pose/Hand/Gaze를 실행한다. 실제 해상도를
   사용하고 `WorldFrame.to_world()`로 손목·컵을 cm로 변환한다.
4. `HandCupEvaluator`가 최소 손목–컵 거리, 접근속도, TTC, grip/gaze evidence를
   계산한다. 15cm 미만 DANGER, 35cm 미만 WARN, 빠른 접근(25cm/s 이상)의
   TTC 0.8/1.6초 기준은 초기 소프트웨어 값이며 실물에서 재보정한다.
5. Pi가 레이더 위험과 손–컵 위험 중 더 높은 등급을 선택한다. 동률이면 더
   짧은 TTC/거리를 우선하고 원인과 방향을 기록한다. 컵/손이 없으면 손–컵
   위험은 UNKNOWN으로 남고 레이더 위험을 SAFE로 덮어쓰지 않는다.
6. 위협 방향은 로봇에서 가장 가까운 손목을 향한 벡터를 로봇 좌표로 변환해
   8방향 sector로 만든다. 회피 계획은 반대 sector부터 탐색하되, 카메라/레이더
   obstacle로 BLOCKED/UNKNOWN인 방향은 선택하지 않는다. 모두 막히면 STOP.
7. 기존 SafetyManager와 MotionGoals를 통과한 계획만 UDP `cmd_move`로 전송한다.

## ESP32 명령 계약

Pi는 30Hz 제어 틱마다 같은 동작의 `motion_id`를 유지하면서 `seq`를 증가시킨다.
예를 들어 손 위협이 FRONT이고 BACK이 CLEAR인 DANGER의 첫 실물 시험 계획은
`vx=-15cm/s, vy=0, w=0, status=RUN, target_distance_cm=15`이다. WARN은
`10cm/s, 10cm, SLOW`가 초기값이다. 검증 후 목표인 25/30, 15/15로 높인다.
실제 전송 형식은 다음과 같다.

```json
{"type":"cmd_move","session_id":123,"seq":45,"vx":-15,"vy":0,"w":0,"status":"RUN","motion_id":678,"target_distance_cm":15}
```

Pi의 `UdpSender`가 `session_id`와 `seq`를 소유한다. `MotionGoals`가
`motion_id`를 소유한다. ESP32는 역기구학·encoder/PID·거리 목표·watchdog을
소유한다. Pi는 wheel PWM을 직접 정하지 않는다.

위험 해소, 마커/모델/보정 상실, telemetry 지연, 레이더 invalid, 선택 가능한
회피 방향 없음, 목표 완료, 사용자 종료 시 `{"type":"stop",...}`을 즉시
보내며 이후 틱에도 STOP을 유지한다. ESP32 `goal_reached`를 확인하고 새
perception revision 전에는 같은 목표를 재시작하지 않는다.

## 검증 순서

1. 손–컵 거리/접근속도와 grip/gaze 비감쇠 단위 테스트.
2. ArUco ID 0/1, calibration, Pose 입력이 한 WorldState로 묶이는 테스트.
3. 위험 병합·방향 선택·모든 sector BLOCKED·stale sensor STOP 테스트.
4. UDP packet/`motion_id`/`seq`/telemetry 완료 상태를 모의한 통합 테스트.
5. Windows Galaxy 영상에서 오버레이와 거리 로그 확인. 사람 손과 컵 마커가
   실제로 보여야 하며, 현재는 이 라이브 검증이 아직 되지 않았다.
6. Pi에서 `GRISE_ENABLE_ESCAPE_MOTION=0`으로 shadow log와 STOP 송신 확인.
7. 보정·레이더·ESP32 telemetry와 바퀴 공중 테스트를 검증한 뒤에만 별도
   실물 회피 시험으로 넘어간다.
