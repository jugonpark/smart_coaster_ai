# GRISE Laptop Node

역할은 **USB 또는 Galaxy HTTP MJPEG 카메라 수신·재송출 + 모니터링**뿐입니다. 판단/경로계획/모터제어는 하지 않습니다.

## 실행

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
python main.py
```

Raspberry Pi는 `http://<노트북IP>:8080/stream.mjpg`를 읽습니다.
Pi의 상태 JSON은 UDP 9001로 이 노트북에 들어옵니다.

초기 카메라 값은 기존 grise 코드와 동일하게 1280x720, 30 FPS, CAMERA_INDEX=0 입니다.

### Windows PowerShell 카메라 선택

환경변수를 지정하지 않으면 USB 카메라 index 0을 사용한다.

```powershell
$env:GRISE_CAMERA_SOURCE="local"
$env:GRISE_CAMERA_INDEX="0"
python laptop/main.py
```

Galaxy의 HTTP MJPEG 영상을 받으려면 Galaxy와 노트북을 서로 접속 가능한 Wi-Fi 또는 Galaxy hotspot에 연결하고, 카메라 앱에 표시된 **실제 Galaxy IP와 stream endpoint**를 사용한다.

```powershell
$env:GRISE_CAMERA_SOURCE="network"
$env:GRISE_CAMERA_URL="http://192.168.x.x:8080/<actual-stream-path>"
python laptop/main.py
```

앱의 `0.0.0.0:8080`은 앱이 기다리는 bind 주소이지 노트북이 접속할 주소가 아니다. Galaxy의 실제 IP와 `/video` 등 앱이 제공하는 정확한 경로를 확인한다. URL에 사용자 이름·암호나 query token이 있으면 프로그램 로그에서는 숨긴다. 현재 테스트는 노트북 Preview만으로 가능하며, 재송출 경로 `/health`, `/snapshot.jpg`, `/stream.mjpg`도 그대로 동작한다. 이후 Pi를 연결할 때는 `http://<노트북IP>:8080/stream.mjpg`를 사용한다.

정상 연결 로그는 `[LAPTOP] camera source: NETWORK`, `[CAMERA] source=network`, `[CAMERA] opened`, `[CAMERA] connected; first frame=1280x720` 순서다. 접속 실패 시 `[CAMERA] network source offline; retrying`이 나오며 설정된 간격으로 다시 연결한다. 실제 수신 해상도는 Galaxy 앱 설정에 따라 다를 수 있다.

## 설정과 상태

`laptop/config.py`의 기본값을 사용하며 아래 환경변수로 변경할 수 있습니다.

| 환경변수 | 기본값 |
|---|---|
| `GRISE_CAMERA_SOURCE` | `local` (`local` 또는 `network`) |
| `GRISE_CAMERA_URL` | 빈 값, `network`일 때 필수 |
| `GRISE_CAMERA_INDEX` | `0` |
| `GRISE_FRAME_WIDTH`, `GRISE_FRAME_HEIGHT` | `1280`, `720` |
| `GRISE_CAMERA_FPS`, `GRISE_JPEG_QUALITY` | `30`, `80` |
| `GRISE_CAMERA_RETRY_S`, `GRISE_CAMERA_FRAME_STALE_S` | `1.0`, `0.5` 초 |
| `GRISE_MJPEG_HOST`, `GRISE_MJPEG_PORT` | `0.0.0.0`, `8080` |
| `GRISE_MONITOR_BIND_HOST`, `GRISE_MONITOR_UDP_PORT` | `0.0.0.0`, `9001` |
| `GRISE_MONITOR_STALE_S` | `1.5` 초 |

Pi가 이 노트북의 IP로 접속하므로 Laptop에 Pi IP 설정은 필요하지 않습니다. 카메라 프레임이 없으면 `/health`는 `{"ok":false}`, `/snapshot.jpg`와 `/stream.mjpg`는 HTTP 503을 반환합니다. 카메라 재연결 후 기존 URL에서 다시 수신할 수 있습니다. 스트림을 받던 중 카메라가 끊기면 연결을 닫으므로 Pi가 재접속해야 합니다.

화면은 현재 노트북 카메라 상태와 Pi의 최신 상태를 표시합니다. Pi UDP가 `GRISE_MONITOR_STALE_S` 동안 오지 않으면 Pi를 OFFLINE으로 표시하고 이전 정보를 숨깁니다. Pi 패킷에 없는 threat direction 등은 `UNKNOWN`/`N/A`로 표시합니다. `q`, Esc, 창 닫기 또는 Ctrl+C로 종료합니다.

로컬 테스트: 작업공간 루트에서 `python -B -m unittest discover -s tests -v`. 가짜 카메라·루프백 UDP 검증이며 실제 USB 카메라, Wi-Fi, Raspberry Pi 검증은 별도입니다.
