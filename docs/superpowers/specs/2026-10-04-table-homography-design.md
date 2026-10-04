# Table-plane calibration for D.I.G

## Outcome and boundary

Use four fixed `DICT_4X4_50` ArUco markers (IDs 41–44) to map camera pixels to centimetres on a 65 × 40 cm table. The PC pipeline in this repository remains responsible for perception, decisions, planning and UDP commands. ESP32 firmware and the UDP packet schema are outside this change. The first deliverable is software-verified calibration and a STOP-safe PC runtime; physical motion remains a separate test.

The working layout starts with a table-centred origin, +X to the right and +Y upward. With 5 cm black markers centred in approximately 6 × 6 cm paper sheets placed flush at the four table corners, the estimated marker centres are ID 41 `(-29.5,+17)`, 42 `(+29.5,+17)`, 43 `(+29.5,-17)` and 44 `(-29.5,-17)` cm. These are initial measurements, not verified calibration. Each marker's measured centre, black-square side and orientation are editable independently.

## Chosen design

Use a planar homography in `WorldFrame`. Its forward matrix maps pixel points to TABLE points; its inverse maps TABLE points to pixels. Every runtime world point uses this conversion. Remove `update_scale()` from the robot runtime path and retire scalar `px_per_cm` conversions from live perception and drawing. Do not silently fall back to the old scalar when calibration is unavailable.

Keep the existing perception interfaces where practical: `Camera.read()` still returns `(ok, BGR frame)`, `WorldFrame.to_world()` returns a TABLE point, and `WorldFrame.to_pixel()` returns an image point. Replace scalar length conversions with local point conversions. For a YOLO box, transform its centre and four edge midpoints, and use the largest resulting TABLE distance as its approximate radius. For a world-space displacement, transform its world start and end points separately; a single global pixel delta is invalid under perspective.

Compute robot heading by transforming its marker TL and TR corners to TABLE coordinates and applying `MARKER_HEADING_OFFSET_DEG` to their `atan2` angle. Transform world points back through the inverse homography when drawing its heading arrow, object exclusion radius and potential-field velocity arrow. A table-plane circle may render as a sampled curve in the image. The marker on top of the robot and MediaPipe joints above the table are projected onto the table plane; their reported centimetres are not a corrected 3D ground intersection.

`Camera` accepts either `GRISE_CAMERA_URL` (HTTP/MJPEG, when set) or the existing USB `CAMERA_INDEX`. Source selection stays inside `Camera`; consumers receive the same BGR frame. Network disconnection, failed reads and stale frames cannot authorize movement. Calibration must match the actual decoded frame resolution, independent of requested capture properties. No horizontal flip is introduced.

## Calibration artifact and workflow

The calibration tool shares the runtime ArUco dictionary and scanner. It accepts the configured camera source or a saved image. It requires IDs 41–44 together and uses each marker's ordered TL/TR/BR/BL corners, producing 16 pixel-to-TABLE correspondences. World corners are calculated from each marker's measured centre, black-square side and TABLE-plane CCW orientation. Fit a homography, compute errors over **all** correspondences, and report RMS and maximum error in cm. The preview displays IDs, table grid and axes; users can inspect independent known points before saving. Invalid/missing markers, non-finite or singular matrices, implausible corners or a failed fit cannot produce a saved calibration.

The saved JSON includes schema version, calibration image width/height, table width/height, `DICT_4X4_50`, all four marker definitions, forward and inverse matrices, creation time, correspondence count, RMS and maximum reprojection error. The example layout and generated calibration are distinct files; the generated calibration is device-specific and ignored by Git. Saving replaces the prior calibration only after a valid fit and visible user action. The fit errors describe agreement with the marker geometry, not independently measured table accuracy. Validation against known points outside the 16 fit points is required before physical motion tests. Initial target: independent-point RMS around 1 cm and maximum around 2 cm; these are goals, not claims.

## Runtime STOP gate

At startup, load and validate the file's version, marker IDs, table size, resolution, finite invertible matrices and quality metadata. At each frame, compare the decoded resolution with the file. Missing or invalid calibration, resolution mismatch, camera read failure or stale frame makes the world invalid. `main.py` then sends `STOP` with zero velocity, clears or ignores cached perception and planner outputs, and cannot send `RUN` until a valid current frame and calibration are restored. The same rule applies before field calculation and again before UDP transmission, so exceptions and stale values cannot bypass it. Existing pause, lost-robot and shutdown STOP behavior remains. The calibration tool sends no UDP packets.

## Code touch points

- `config.py`: calibration path and camera URL, coordinate comments; no firmware limits or UDP changes.
- `perception/table_calibration.py` (new): marker layout, fit, validation, JSON and point transforms.
- `perception/camera.py`: source selection, frame freshness/resolution, `WorldFrame` homography facade.
- `perception/robot_tracker.py`, `object_detector.py`, `marker_object_detector.py`: remove live scalar conversions; use TABLE points and local geometry.
- `main.py`, `tools/check_camera.py`: STOP gate, calibrated drawing and diagnostics.
- `tools/calibrate_table_homography.py` (new), example layout, README: capture, fit, inspect, save and table-plane limitation.
- `tools/selftest.py` or focused tests: synthetic perspective, round trip, heading, radius, drawing projections, invalid-file and STOP cases.

## Verification and rollout

First run calibration/coordinate unit tests and STOP-gate tests with synthetic frames and a fake sender. Run the existing self-test and syntax checks. Use a real Galaxy/USB frame to fit and inspect the calibration, then compare several independent physical table points. Only after the real coordinate error and direction are documented should a separate physical movement test be planned. A passing build or simulation is software evidence only.
