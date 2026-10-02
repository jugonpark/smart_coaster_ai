# Galaxy MJPEG에서 팀 AI 오버레이 확인

Windows에서 저장소 루트 기준:

```powershell
python -m pip install -r laptop/requirements.txt
python -m pip install -r tools/requirements_ai_preview.txt
python tools/download_ai_models.py
$env:GRISE_CAMERA_URL="http://10.232.69.154:8080/"
python tools/ai_overlay_preview.py
```

`--url`로 주소를 바꿀 수 있습니다. `q`/Esc로 종료합니다. 모델 세 개 중 하나라도
없으면 오류를 출력하고 시작하지 않습니다. 모델 파일은 `models/`에 저장되며
Git에는 포함하지 않습니다.

화면에 Pose 관절, Hand 21점 및 grip evidence, Face 머리 방향(gaze proxy),
ArUco `DICT_4X4_50` 마커 외곽선과 ID가 겹쳐 표시됩니다. ArUco는 현재
OpenCV의 `cv2.aruco`를 사용하므로 별도 모델 파일이 필요 없습니다.
이 단계의 gaze는 컵 위치를 입력받지 않으므로
`looking_at_target`은 평가하지 않습니다. grip과 gaze는 위험 등급을 바꾸지
않으며 ESP32로 UDP 명령을 보내지 않습니다. 화면 좌표는 물리적 cm가 아닙니다.

자동 점검 예: `python tools/ai_overlay_preview.py --headless --max-frames 3`.
Pi 인식 경로도 동일한 `shared_ai` 모듈을 사용합니다. 실제 모터 시험 전에는
테이블 calibration과 Pi shadow mode에서 손–컵 위험 로그를 확인해야 합니다.
