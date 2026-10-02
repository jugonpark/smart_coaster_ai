# Shared MediaPipe perception

`PoseTracker`, `HandTracker`, `GazeTracker`는 팀 저장소
[`32200362-sys/grise`](https://github.com/32200362-sys/grise)의 인식 코드를
Windows/Pi 공용 패키지로 옮긴 것입니다. 원본 확인 시점: `fc9e92b`.

Windows 미리보기와 Pi 인식 경로에서 같은 tracker를 사용합니다.
`hand_cup.py`는 보정된 cm 좌표의 위험 evidence를 계산하지만 ESP32 명령을
전송하지 않습니다. `grasp_ready`와 `looking_at_target`은 보조 evidence이며,
그립 검출로 기존 위험 판정을 `SAFE`로 덮어쓰지 않습니다.

팀 코드의 고정 1280×720 좌표는 실제 입력 프레임 크기를 사용하도록 고쳤습니다.
미리보기의 `PreviewWorld`는 화면 좌표용이며 실측 cm 또는 테이블 보정값이
아닙니다. Pi 연결 시에는 기존 보정된 `WorldFrame`을 전달해야 합니다.
