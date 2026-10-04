"""Check firmware feedforward against measured, steady encoder RPM."""

import csv
import math
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "arduino" / "esp32_omni_controller" / "robot_config.h"
MOTION = ROOT / "arduino" / "esp32_omni_controller" / "kinematics_motion.cpp"
SAMPLES = ROOT / "calibration" / "motor_pwm_samples.csv"


def read_table(source: str, name: str) -> list[list[float]]:
    match = re.search(rf"\b{name}\[MOTOR_COUNT\]\[FF_POINT_COUNT\]\s*=\s*\{{(.*?)\}};",
                      source, re.DOTALL)
    assert match is not None
    return [[float(value) for value in re.findall(r"-?\d+\.\d+", row)]
            for row in re.findall(r"\{([^{}]+)\}", match.group(1))]


def rpm_at_pwm(samples: list[tuple[float, float]], pwm: float) -> float:
    pairs = list(zip(samples, samples[1:]))
    (p0, r0), (p1, r1) = next(
        ((a, b) for a, b in pairs if a[0] <= pwm <= b[0]),
        pairs[0] if pwm < samples[0][0] else pairs[-1],
    )
    return r0 + (pwm - p0) * (r1 - r0) / (p1 - p0)


def test_feedforward_matches_measured_wheel_rpm():
    source = CONFIG.read_text(encoding="utf-8")
    measured = list(csv.DictReader(SAMPLES.open(encoding="utf-8", newline="")))
    # The 2.5 cm/s samples were intermittent; the revised low-speed entries
    # must be verified on the robot with the bounded startup assist.
    speeds = [5.0, 7.5, 10.0, 12.5, 15.0]
    for sign, name in ((1, "FF_PWM_POS"), (-1, "FF_PWM_NEG")):
        table = read_table(source, name)
        assert len(table) == 3
        for motor, row in enumerate(table, start=1):
            assert len(row) == 7 and row[0] == 0.0
            assert all(a < b <= 255 for a, b in zip(row, row[1:]))
            samples = sorted((abs(float(sample["pwm"])), abs(float(sample["rpm"])))
                             for sample in measured
                             if int(sample["motor"]) == motor and
                             int(sample["pwm"]) * sign > 0)
            assert len(samples) >= 6
            for speed, pwm in zip(speeds, row[2:]):
                target_rpm = speed * 60.0 / (2.0 * math.pi * 2.9)
                assert abs(rpm_at_pwm(samples, pwm) - target_rpm) < 1.5, (
                    motor, sign, speed, pwm)


def test_rotation_uses_same_measured_feedforward_as_translation():
    source = MOTION.read_text(encoding="utf-8")
    assert "feedForwardPwm(i, target)" in source
    assert "feedForward *= 0.75f" not in source
