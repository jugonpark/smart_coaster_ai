# Table Homography Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Calibrate the D.I.G PC pipeline to a measured table plane, accept USB or Galaxy MJPEG input, and keep UDP motion stopped whenever calibration or the current frame is invalid.

**Architecture:** A versioned calibration object fits and validates 16 ArUco corner correspondences. `WorldFrame` owns its validity and the pixel↔TABLE transforms; all perception and overlays use those transforms. `main.py` and an injected `UdpSender` motion guard independently enforce STOP.

**Tech Stack:** Python 3, OpenCV ArUco and `findHomography`, NumPy, standard-library `unittest`/`json`, existing PC pipeline.

**Spec:** `docs/superpowers/specs/2026-10-04-table-homography-design.md`

## Global Constraints

- Preserve the existing PC→ESP32 packet schema and do not modify `esp32/`.
- IDs 41–44 use `DICT_4X4_50`; use each marker's four ordered TL/TR/BR/BL corners.
- TABLE origin is the table centre, +X right, +Y up, units cm; current starting table size is 65 × 40 cm.
- Runtime coordinates and drawing use only forward/inverse Homography, never live `px_per_cm` or `update_scale()`.
- Compare calibration resolution with actual decoded `frame.shape`, not configured width/height.
- Invalid/missing calibration, stale or failed frame, resolution mismatch, lost robot and pause require zero-velocity STOP.
- Keep generated calibration device-specific and ignored by Git; physical movement testing is out of scope.

## Review Focus

- A Galaxy stream decodes at a different resolution than `FRAME_WIDTH/HEIGHT`: compare with `frame.shape` and STOP on calibration mismatch (Tasks 2, 6).
- A valid-looking JSON contains a singular or inconsistent inverse matrix: reject it before any world conversion (Tasks 1, 2).
- A marker is rotated or IDs are exchanged: 16 ordered physical corners expose a bad fit rather than silently swapping positions (Task 1).
- A previous valid frame is followed by a failed read or processing delay: motion guard must send STOP despite cached detections (Tasks 5, 6).
- A YOLO box near a perspective-heavy edge has a different local scale: compute radius from transformed edge points (Task 3).

---

### Task 1: Calibration data and fitting tool

**Files:** Create `perception/table_calibration.py`, `tools/calibrate_table_homography.py`, `calibration/table_layout.example.json`, `tests/__init__.py`, `tests/test_table_calibration.py`; modify `.gitignore`, `README.md`.

**Interfaces:** `MarkerDefinition(id, x_cm, y_cm, size_cm, orientation_deg)`; `TableCalibration.from_scan(scan, markers, image_size, table_size) -> TableCalibration`; `TableCalibration.load(path)`, `.save(path)`, `.pixel_to_table(x,y)`, `.table_to_pixel(x,y)`. JSON version 1 carries both 3×3 matrices, image and table dimensions, dictionary, IDs/definitions, UTC timestamp, correspondence count 16, RMS and MAX cm.

- [ ] Write synthetic perspective tests: corner mapping for ID 41–44 using local TL/TR/BR/BL offsets `(-s/2,+s/2),(+s/2,+s/2),(+s/2,-s/2),(-s/2,-s/2)` rotated CCW; fit 16 points and assert point/round-trip error below `1e-3` cm. Reject missing/duplicate IDs, rotated-ID mismatch, nonfinite/singular/inconsistent matrix and malformed metadata.
- [ ] Run `python -m unittest tests.test_table_calibration -v`; confirm failures identify absent interfaces.
- [ ] Implement data validation, fit, all-point RMS/MAX, atomic JSON save and CLI preview/click/`S`/`Q` flow. Use the estimated `(-29.5,+17)` etc. only in the example layout; do not treat them as verified measurements. Add `.gitignore` entry for `calibration/table_calibration.json` and document real-point validation.
- [ ] Rerun the test command and `python tools/calibrate_table_homography.py --help`; expect pass and usable options. Commit the focused change.

### Task 2: WorldFrame bidirectional transform and validity

**Files:** Modify `perception/camera.py`, `config.py`; create `tests/test_world_frame.py`.

**Interfaces:** `WorldFrame(calibration_path=...)`; `.to_world(px_x,px_y) -> (float,float)`; `.to_pixel(x_cm,y_cm) -> (float,float)`; `.observe_frame(frame_shape, read_at)`; `.invalidate_frame(reason)`; `.can_send_motion(now=None) -> bool`; `.invalid_reason`. `Camera.world` remains the facade used by perception.

- [ ] Write tests for forward/inverse round trip, absent file, bad matrix, and actual decoded `(height,width)` mismatch even when config says a different size. Test that `can_send_motion` expires after configured frame age.
- [ ] Run `python -m unittest tests.test_world_frame -v`; expect failure on new validity API.
- [ ] Load Task 1 calibration into `WorldFrame`, record monotonic receipt time, check actual `frame.shape`, and remove scalar conversion methods from the runtime API. `to_pixel` returns unrounded floats; OpenCV callers round at draw time. `Camera.read()` updates/invalidate world on successful/failed reads.
- [ ] Rerun the test command and commit.

### Task 3: Robot heading and local object geometry

**Files:** Modify `perception/robot_tracker.py`, `object_detector.py`, `marker_object_detector.py`, `pose_tracker.py`, `hand_tracker.py`, `gaze_tracker.py`; create `tests/test_calibrated_perception.py`.

**Interfaces:** Keep `RobotTracker.process(frame, world, scan) -> RobotPose` and detector `detect(...) -> list[Detection]`. World coordinates are TABLE points. Normalized landmark coordinates use the current frame's `(width,height)`.

- [ ] Test robot heading from transformed TL/TR (including perspective where image angle differs); confirm no `update_scale()` call. Test YOLO radius as the maximum TABLE distance from transformed box centre to four edge midpoints, marker object radius from its configured real cm, and landmark multiplication by actual frame dimensions.
- [ ] Run `python -m unittest tests.test_calibrated_perception -v`; expect targeted failures.
- [ ] Replace live scalar geometry, keep existing labels and object selection, and make held marker detections retain TABLE positions while the frame is valid. Ensure no old detection is used after invalidation.
- [ ] Rerun the test command and commit.

### Task 4: Homography-based HUD projection

**Files:** Modify `main.py`, `perception/robot_tracker.py`, `marker_object_detector.py`, `tools/check_camera.py`; create `tests/test_calibrated_drawing.py`.

**Interfaces:** Use `WorldFrame.to_pixel()` for world start/end points. Draw a cm-radius TABLE circle by sampling points in world space, projecting each to pixels, then joining the projected points. Replace scalar HUD scale with calibration valid/error/readout.

- [ ] Test projected heading endpoint, potential-field arrow endpoint and object radius curve under a nonuniform homography; assert endpoints equal inverse-projected world points, not a `px_per_cm` offset.
- [ ] Run `python -m unittest tests.test_calibrated_drawing -v`; expect failure.
- [ ] Update overlays and camera diagnostic readouts without changing planner math; use actual frame width for screen positions. Rerun tests and commit.

### Task 5: USB and Galaxy MJPEG capture

**Files:** Modify `config.py`, `perception/camera.py`, `tools/check_camera.py`; create `tests/test_camera_sources.py`; update `README.md`.

**Interfaces:** `GRISE_CAMERA_URL` unset/empty selects `cv2.VideoCapture(config.CAMERA_INDEX)`; a validated HTTP(S) URL selects `cv2.VideoCapture(url)`. `Camera.read()` always returns `(ok, BGR frame)` to downstream code.

- [ ] Test both `VideoCapture` constructor arguments via a fake capture, a failed network read invalidating the world, and decoded dimensions independent of capture `set()` requests. An empty URL selects USB; reject a nonempty URL with an unsupported scheme.
- [ ] Run `python -m unittest tests.test_camera_sources -v`; expect failure.
- [ ] Implement source selection only in `Camera`, document the Galaxy URL and calibration capture sequence, and keep USB behavior. Rerun tests and commit.

### Task 6: Fail-closed integration and UDP send guard

**Files:** Modify `main.py`, `comm/udp_sender.py`; create `tests/test_calibration_stop_gate.py`.

**Interfaces:** `UdpSender(allow_motion: Callable[[], bool] | None = None)` injects the live `WorldFrame.can_send_motion` check. Before making a `cmd_vel`, `send()` checks the guard; denied motion produces the existing `stop` packet immediately, preserving protocol fields. `main.py` also checks world validity before perception/field use and passes the guard to the sender.

- [ ] Test missing/bad calibration, resolution mismatch, failed read and age expiry with fake camera/sender: no `cmd_vel`; repeated frames remain STOP; a fresh matching frame can re-enable only after all normal gates pass. Test guard denial at `UdpSender.send()` even if a caller passes `RUN` and nonzero velocity.
- [ ] Run `python -m unittest tests.test_calibration_stop_gate -v`; expect failure.
- [ ] Add pipeline STOP branch and independent send guard. Preserve pause/lost-robot/shutdown behavior and existing JSON packet type/keys. Rerun tests and commit.

### Task 7: Full software verification and handoff

**Files:** Update `README.md` only if verification reveals missing run steps; no hardware or ESP32 edits.

**Interfaces:** All earlier APIs and acceptance checks.

- [ ] Run `python -m unittest discover -s tests -v`, `python tools/selftest.py`, `python -m compileall config.py main.py perception comm tools`, and `git diff --check`; record exact results.
- [ ] Inspect the branch diff for residual live `px_per_cm`, `update_scale()` and hard-coded normalized-landmark dimensions, and verify no `esp32/` or UDP schema changes. If a check fails, fix the owning task and rerun only the relevant checks.
- [ ] Report software verification and the remaining physical steps: capture actual Galaxy/USB frame, measure independent table points, confirm marker-on-robot and hand height error, then schedule a separate controlled movement test. Do not claim physical calibration or motion accuracy from unit tests.
