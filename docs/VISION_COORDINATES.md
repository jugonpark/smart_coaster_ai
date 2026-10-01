# PROJECT D 영상 좌표와 TABLE calibration

## 좌표계

- **IMAGE FRAME**: 원점은 영상 왼쪽 위, `u`는 오른쪽, `v`는 아래쪽이다. 입력은 1280×720이며 좌우 반전하지 않는다.
- **TABLE FRAME**: 단위 cm, 원점은 물리적 테이블 중심, `+X/+Y`는 사용자가 정한 테이블의 직교 축, `+Z`는 테이블 위쪽, 양의 회전은 CCW다. 카메라 위치와 독립적이다.
- **ROBOT FRAME**: 원점은 ID 0 로봇 마커 중심, `+x`는 전방, `+y`는 왼쪽이다. `world_to_robot()`는 TABLE 벡터를 heading의 음의 각도로 회전한다.

## Planar homography

`TableCalibration`은 IMAGE `(u,v)`와 TABLE `(X_cm,Y_cm)` 사이의 float64 3×3 homography와 역행렬을 소유한다. `WorldFrame.to_world()`와 `to_pixel()` public interface는 유지되며 calibrated mode에서는 homography를 사용한다. homogeneous W가 0에 가깝거나 결과가 NaN/Inf이면 값을 게시하지 않고 calibration을 무효로 취급한다.

테이블의 서로 먼 네 영역에 동일 평면의 `DICT_4X4_50` 마커를 둔다. 각 marker 설정은 ID, 중심 X/Y cm, 물리 크기, TABLE 축 기준 CCW orientation을 가진다. ArUco의 TL/TR/BR/BL 네 corner와 계산된 네 TABLE corner를 대응시켜 4 marker에서 최대 16점을 `cv2.findHomography()`에 넣는다. ID 0 로봇 마커는 calibration marker ID와 별개다.

실측 결과는 `calibration/table_calibration.json`에 저장하며 Git에서 제외한다. `calibration/table_calibration.example.json`은 schema와 예시 배치만 제공하며 실제 보정값이 아니다. 파일에는 image width/height, dictionary, marker 정의, 양방향 H, 생성 시각, RMS/max reprojection error와 correspondence 수가 들어간다. 실행 해상도가 보정 해상도와 다르거나 파일·행렬이 잘못되면 calibration은 invalid다. 자동 resize 보상은 하지 않는다.

## Robot과 object 변환

Robot 중심은 ID 0 marker의 영상 중심을 H로 변환한다. Heading은 영상 edge 각도를 쓰지 않고 TL과 TR corner를 각각 TABLE 좌표로 변환한 벡터의 `atan2(dY,dX)`로 계산한 뒤 `MARKER_HEADING_OFFSET_DEG`를 적용한다. Homography mode에서는 ID 0 marker 크기로 global `px_per_cm`을 갱신하지 않는다.

YOLO bbox 중심 의미는 그대로 유지한다. 반경은 bbox의 left/right/top/bottom 중점을 각각 TABLE로 변환하고, 변환된 중심에서 가장 먼 edge 거리로 근사한다. 따라서 화면 위치마다 다른 local scale을 사용한다. 물체 bottom-center/contact point, segmentation, hand landmark는 후속 작업이다.

`WorldFrame`의 legacy `px_per_cm` mode는 unit test·simulation·migration 호환 전용이다. 실제 `main.py`는 calibration-required mode로 시작한다. 파일 누락, singular/non-finite matrix, 해상도 불일치가 있으면 `calibration_valid=false`, 모든 vision sector가 `UNKNOWN`, fusion `sensor_valid=false`가 되어 기존 safety gate가 STOP한다. `GRISE_ENABLE_ESCAPE_MOTION=0` 기본값은 유지한다.

## 도구와 실기 절차

예시 설정을 복사해 실제 marker 중심·크기·방향을 입력한 뒤 실행한다.

```powershell
Copy-Item calibration/table_calibration.example.json calibration/table_layout.json
python tools/calibrate_table_homography.py --camera 0 --config calibration/table_layout.json
```

저장 이미지도 사용할 수 있다.

```powershell
python tools/calibrate_table_homography.py --image table.jpg --config calibration/table_layout.json
```

도구는 ID/corner/center/TABLE 좌표, valid 상태, RMS/max error, 10cm grid와 TABLE 원점/+X/+Y를 overlay한다. 화면을 클릭하면 Pixel과 TABLE cm를 출력한다. `S`로 `calibration/table_calibration.json`을 저장하고 `Q`/ESC로 종료한다.

실제 절차:

1. 카메라를 테이블 모서리 상단에 단단히 고정한다.
2. 해상도를 1280×720으로 고정하고 autofocus/zoom 변화를 가능한 한 막는다.
3. calibration ArUco 네 개를 동일 테이블 평면에 넓게 배치한다.
4. marker 중심과 크기, orientation을 자로 측정해 config에 입력한다.
5. 도구에서 네 marker와 16 corner 검출, H 계산, RMS/max 값을 확인한다.
6. overlay grid와 원점/축이 실제 테이블과 맞는지 확인한다.
7. calibration에 쓰지 않은 여러 known point를 클릭하고 실제 좌표와 비교해 오차를 기록한다.
8. 초기 hardware acceptance 목표는 RMS 약 1cm 이하, max 약 2cm 이하다. 이는 실측 목표이며 현재 달성된 결과가 아니다.
9. 카메라·해상도·zoom·marker 배치가 움직이면 다시 calibration한다.

향후에는 marker 재검출로 camera movement consistency를 확인하고 RAW → optional undistort → ArUco/homography 순서를 지원할 수 있다. 현재는 임의 camera intrinsic이나 자동 3D 보정을 넣지 않는다.

## 물리적 한계

Homography는 **TABLE PLANE에서만** 정확하다. 로봇 위의 ID 0 마커나 손/물체가 테이블보다 높으면 모서리 카메라에서 parallax가 생기며 ground intersection과 일치하지 않는다. `ROBOT_MARKER_HEIGHT_CM`은 미래 확장 지점일 뿐 현재 3D correction에는 사용하지 않는다. 광각 렌즈 distortion이 크면 가장자리 오차가 homography만으로 제거되지 않는다.

현재 synthetic transform, round-trip, invalid matrix, resolution, robot heading, local bbox scale, fail-safe pipeline은 SOFTWARE_VERIFIED다. 실제 카메라/테이블/ArUco에서 cm 오차, parallax, lens distortion, marker 부착 방향과 재현성은 HARDWARE_UNVERIFIED다.
