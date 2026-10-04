#!/usr/bin/env python3
"""Test one bounded ESP32 typed cmd_move. Raise the wheels before first use.

Example: python tools/esp32_cmd_move_test.py --ip 192.168.0.50 --vx 8 --vy 0 --w 0 --distance 10
Only the Python standard library is required. This script does not change firmware.
"""
from __future__ import annotations

import argparse
import ipaddress
import json
import math
import secrets
import socket
import sys
import time
from typing import Any


COMMAND_PORT = 8888
TELEMETRY_PORT = 8889
RATE_HZ = 30.0
INITIAL_STOP_SECONDS = 1.0
FINAL_STOP_SECONDS = 1.0
TELEMETRY_TIMEOUT_SECONDS = 0.8
MAX_LINEAR_CM_S = 15.0  # Conservative first hardware test limit.
MAX_ANGULAR_RAD_S = 1.0
MAX_DISTANCE_CM = 100.0  # Firmware absolute limit.


def finite_float(value: str) -> float:
    try:
        result = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    if not math.isfinite(result):
        raise argparse.ArgumentTypeError("must be finite")
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test one finite-distance ESP32 move")
    parser.add_argument("--ip", required=True, help="ESP32 IPv4 address")
    parser.add_argument("--vx", type=finite_float, default=8.0, help="forward cm/s")
    parser.add_argument("--vy", type=finite_float, default=0.0, help="left cm/s")
    parser.add_argument("--w", type=finite_float, default=0.0, help="CCW rad/s")
    parser.add_argument("--distance", type=finite_float, default=10.0, help="goal distance in cm")
    parser.add_argument("--timeout", type=finite_float, default=10.0, help="goal timeout in seconds")
    args = parser.parse_args(argv)
    try:
        if ipaddress.ip_address(args.ip).version != 4:
            parser.error("--ip must be IPv4")
    except ValueError as exc:
        parser.error(str(exc))
    if not 0 < args.distance <= MAX_DISTANCE_CM:
        parser.error(f"--distance must be > 0 and <= {MAX_DISTANCE_CM:g} cm")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    if math.hypot(args.vx, args.vy) > MAX_LINEAR_CM_S or abs(args.w) > MAX_ANGULAR_RAD_S:
        parser.error("speed exceeds conservative test limit (15 cm/s, 1 rad/s)")
    if math.hypot(args.vx, args.vy) <= 0:
        parser.error("cmd_move requires nonzero translational vx or vy")
    return args


def build_packet(session_id: int, seq: int, motion_id: int | None = None,
                 vx: float = 0.0, vy: float = 0.0, w: float = 0.0,
                 distance: float | None = None) -> dict[str, Any]:
    if type(session_id) is not int or not 0 < session_id <= 0xFFFFFFFF:
        raise ValueError("session_id must be a nonzero uint32")
    if type(seq) is not int or not 0 <= seq <= 0xFFFFFFFF:
        raise ValueError("seq must be uint32")
    packet: dict[str, Any] = {"type": "stop", "session_id": session_id, "seq": seq}
    if motion_id is None:
        return packet
    if type(motion_id) is not int or not 0 < motion_id <= 0x7FFFFFFF:
        raise ValueError("motion_id must be a positive int32")
    if distance is None or not math.isfinite(distance) or not 0 < distance <= MAX_DISTANCE_CM:
        raise ValueError("target distance must be positive and within firmware limit")
    if not all(math.isfinite(value) for value in (vx, vy, w)):
        raise ValueError("velocity must be finite")
    if math.hypot(vx, vy) <= 0:
        raise ValueError("cmd_move requires nonzero translational vx or vy")
    packet.update(type="cmd_move", vx=vx, vy=vy, w=w, status="RUN",
                  motion_id=motion_id, target_distance_cm=distance)
    return packet


def validate_telemetry(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict) or payload.get("type") != "telemetry":
        raise ValueError("unexpected telemetry type")
    for key in ("state", "fault", "mode"):
        if not isinstance(payload.get(key), str):
            raise ValueError(f"invalid {key}")
    for key in ("session_id", "last_seq"):
        if type(payload.get(key)) is not int or not 0 <= payload[key] <= 0xFFFFFFFF:
            raise ValueError(f"invalid {key}")
    motion_id = payload.get("motion_id")
    if motion_id is not None and (type(motion_id) is not int or not 0 <= motion_id <= 0x7FFFFFFF):
        raise ValueError("invalid motion_id")
    for key in ("goal_active", "goal_reached"):
        if type(payload.get(key)) is not bool:
            raise ValueError(f"invalid {key}")
    for key in ("encoder_count", "wheel_target", "wheel_speed", "rpm"):
        values = payload.get(key)
        if not isinstance(values, list) or len(values) != 3 or not all(
                type(value) in (int, float) and math.isfinite(value) for value in values):
            raise ValueError(f"invalid {key}")
    if any(type(value) is not int for value in payload["encoder_count"]):
        raise ValueError("encoder_count must contain integers")
    return payload


def format_telemetry(payload: dict[str, Any]) -> str:
    return (f"state={payload['state']} fault={payload['fault']} "
            f"last_seq={payload['last_seq']} motion_id={payload['motion_id']} "
            f"goal_active={payload['goal_active']} goal_reached={payload['goal_reached']}\n"
            f"encoder_count={payload['encoder_count']} wheel_target={payload['wheel_target']} "
            f"wheel_speed={payload['wheel_speed']} rpm={payload['rpm']}")


class MoveTest:
    def __init__(self, ip: str) -> None:
        self.ip = ip
        self.target = (ip, COMMAND_PORT)
        self.session_id = secrets.randbelow(0xFFFFFFFF) + 1
        self.motion_id = secrets.randbelow(0x7FFFFFFF) + 1
        self.seq = 0
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("0.0.0.0", TELEMETRY_PORT))
        self.sock.setblocking(False)
        self.last_telemetry_at: float | None = None
        self.last_display_at = 0.0

    def close(self) -> None:
        self.sock.close()

    def send(self, *, move: bool = False, vx: float = 0.0, vy: float = 0.0,
             w: float = 0.0, distance: float | None = None) -> int:
        self.seq += 1
        packet = build_packet(self.session_id, self.seq,
                              self.motion_id if move else None, vx, vy, w, distance)
        self.sock.sendto(json.dumps(packet, separators=(",", ":"), allow_nan=False).encode(),
                         self.target)
        return self.seq

    def receive(self) -> list[dict[str, Any]]:
        received = []
        while True:
            try:
                wire, sender = self.sock.recvfrom(8192)
            except BlockingIOError:
                break
            if sender[0] != self.ip:
                continue
            try:
                payload = validate_telemetry(json.loads(wire.decode("utf-8")))
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
                raise RuntimeError(f"invalid ESP32 telemetry: {exc}") from exc
            self.last_telemetry_at = time.monotonic()
            received.append(payload)
            if self.last_telemetry_at - self.last_display_at >= 0.2:
                print(format_telemetry(payload))
                self.last_display_at = self.last_telemetry_at
        return received

    def stop_phase(self, duration: float) -> bool:
        deadline = time.monotonic() + duration
        next_send = time.monotonic()
        acknowledged = False
        while time.monotonic() < deadline:
            now = time.monotonic()
            if now >= next_send:
                self.send()
                next_send = now + 1.0 / RATE_HZ
            for payload in self.receive():
                if payload["fault"] != "NONE" or payload["state"] == "FAULT":
                    raise RuntimeError(f"ESP32 fault: {payload['fault']}")
                if (payload["session_id"] == self.session_id and
                        0 < payload["last_seq"] <= self.seq and payload["mode"] == "NETWORK"):
                    acknowledged = True
            time.sleep(min(0.005, max(0.0, next_send - time.monotonic())))
        return acknowledged

    def run_move(self, vx: float, vy: float, w: float, distance: float,
                 timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        next_send = time.monotonic()
        first_move_seq: int | None = None
        saw_active = False
        while time.monotonic() < deadline:
            now = time.monotonic()
            if now >= next_send:
                sent_seq = self.send(move=True, vx=vx, vy=vy, w=w, distance=distance)
                if first_move_seq is None:
                    first_move_seq = sent_seq
                next_send = now + 1.0 / RATE_HZ
            for payload in self.receive():
                if payload["fault"] != "NONE" or payload["state"] == "FAULT":
                    raise RuntimeError(f"ESP32 fault: {payload['fault']}")
                if payload["session_id"] != self.session_id:
                    raise RuntimeError("telemetry session changed")
                # A delayed STOP response may still have the previous motion_id.
                if payload["last_seq"] >= first_move_seq:
                    if payload["motion_id"] != self.motion_id:
                        raise RuntimeError(f"wrong motion_id: {payload['motion_id']}")
                    if payload["goal_reached"]:
                        return True
                    if payload["state"] in ("DISTANCE_ACTIVE", "DISTANCE_BRAKING"):
                        saw_active = True
                    elif saw_active and payload["state"] == "STOPPED":
                        raise RuntimeError(
                            "ESP32 stopped without goal_reached: "
                            f"progress={payload.get('goal_progress_cm')}cm "
                            f"remaining={payload.get('remaining_cm')}cm "
                            f"overshoot={payload.get('overshoot_cm')}cm "
                            f"encoder_count={payload['encoder_count']}")
                if self.last_telemetry_at is not None and (
                        time.monotonic() - self.last_telemetry_at > TELEMETRY_TIMEOUT_SECONDS):
                    raise TimeoutError("telemetry timeout")
            if self.last_telemetry_at is None or (
                    time.monotonic() - self.last_telemetry_at > TELEMETRY_TIMEOUT_SECONDS):
                raise TimeoutError("telemetry timeout")
            time.sleep(min(0.005, max(0.0, next_send - time.monotonic())))
        raise TimeoutError(f"goal_reached not received within {timeout:g}s")

    def safe_stop(self) -> None:
        # Keep sending even if telemetry is malformed or has stopped arriving.
        deadline = time.monotonic() + FINAL_STOP_SECONDS
        while time.monotonic() < deadline:
            self.send()
            time.sleep(1.0 / RATE_HZ)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    test: MoveTest | None = None
    try:
        test = MoveTest(args.ip)
        print(f"[STOP] Synchronizing with {args.ip}:{COMMAND_PORT} for ~1s")
        if not test.stop_phase(INITIAL_STOP_SECONDS):
            raise RuntimeError("no valid acknowledged NETWORK telemetry; cmd_move was not sent")
        print(f"[MOVE] session_id={test.session_id} motion_id={test.motion_id} "
              f"distance={args.distance:g}cm at {RATE_HZ:g}Hz")
        test.run_move(args.vx, args.vy, args.w, args.distance, args.timeout)
        print("[SUCCESS] Matching goal_reached received; sending STOP")
        return 0
    except KeyboardInterrupt:
        print("[INTERRUPT] Ctrl+C; sending STOP", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"[FAIL] {exc}; sending STOP", file=sys.stderr)
        return 1
    finally:
        if test is not None:
            try:
                test.safe_stop()
            except OSError as exc:
                print(f"[STOP] UDP send failed: {exc}", file=sys.stderr)
            test.close()


if __name__ == "__main__":
    raise SystemExit(main())
