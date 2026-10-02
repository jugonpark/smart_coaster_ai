import importlib.util
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "esp32_cmd_move_test.py"
spec = importlib.util.spec_from_file_location("esp32_cmd_move_test", SCRIPT)
tool = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = tool
spec.loader.exec_module(tool)


def telemetry(*, seq=2, motion_id=7, reached=False, fault="NONE"):
    return tool.validate_telemetry({
        "type": "telemetry", "mode": "NETWORK", "state": "GOAL_REACHED" if reached else "DISTANCE_ACTIVE",
        "fault": fault, "session_id": 9, "last_seq": seq, "motion_id": motion_id,
        "goal_active": not reached, "goal_reached": reached,
        "encoder_count": [1, 2, 3], "wheel_target": [0.0, -7.0, 7.0],
        "wheel_speed": [0.0, -6.0, 6.0], "rpm": [0.0, -20.0, 20.0],
    })


class MoveToolTests(unittest.TestCase):
    def test_typed_cmd_move_schema_and_stop(self):
        packet = tool.build_packet(9, 2, 7, 8.0, 0.0, 0.0, 10.0)
        self.assertEqual(json.loads(json.dumps(packet)), {
            "type": "cmd_move", "session_id": 9, "seq": 2,
            "vx": 8.0, "vy": 0.0, "w": 0.0, "status": "RUN",
            "motion_id": 7, "target_distance_cm": 10.0,
        })
        self.assertEqual(tool.build_packet(9, 1), {"type": "stop", "session_id": 9, "seq": 1})

    def test_distance_and_speed_validation(self):
        for distance in (0, -1, float("nan"), 101):
            with self.subTest(distance=distance), self.assertRaises(ValueError):
                tool.build_packet(9, 1, 7, 8, 0, 0, distance)
        for value in ("0", "-1", "nan", "101"):
            with self.subTest(value=value), self.assertRaises(SystemExit):
                tool.parse_args(["--ip", "10.0.0.1", "--distance", value])
        with self.assertRaises(SystemExit):
            tool.parse_args(["--ip", "10.0.0.1", "--vx", "0", "--w", "1"])

    def test_send_increments_seq_and_keeps_motion_id(self):
        class Sock:
            def __init__(self):
                self.packets = []

            def sendto(self, data, address):
                self.packets.append(json.loads(data))

        test = tool.MoveTest.__new__(tool.MoveTest)
        test.sock = Sock()
        test.target = ("10.0.0.1", 8888)
        test.session_id, test.motion_id, test.seq = 9, 7, 0
        test.send()
        test.send(move=True, vx=8, distance=10)
        test.send(move=True, vx=8, distance=10)
        self.assertEqual([p["seq"] for p in test.sock.packets], [1, 2, 3])
        self.assertEqual([p["motion_id"] for p in test.sock.packets[1:]], [7, 7])

    def test_telemetry_rejects_invalid_fields(self):
        base = telemetry()
        for changed in ({**base, "goal_reached": 1}, {**base, "wheel_speed": [1, 2]},
                        {**base, "motion_id": "7"}, {**base, "encoder_count": [1, 2, 3.0]}):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                tool.validate_telemetry(changed)

    def test_goal_reached_and_wrong_motion_id(self):
        def instance(payload):
            test = tool.MoveTest.__new__(tool.MoveTest)
            test.session_id, test.motion_id, test.seq = 9, 7, 1
            test.last_telemetry_at = 0.0
            test.send = lambda **kwargs: 2
            test.receive = lambda: [payload]
            return test

        with patch.object(tool.time, "monotonic", return_value=0.0):
            self.assertTrue(instance(telemetry(reached=True)).run_move(8, 0, 0, 10, 10))
            with self.assertRaisesRegex(RuntimeError, "wrong motion_id"):
                instance(telemetry(motion_id=8)).run_move(8, 0, 0, 10, 10)

    def test_timeout_and_fault(self):
        test = tool.MoveTest.__new__(tool.MoveTest)
        test.session_id, test.motion_id, test.seq = 9, 7, 1
        test.last_telemetry_at = 0.0
        test.send = lambda **kwargs: 2
        test.receive = lambda: []
        with patch.object(tool.time, "monotonic", return_value=1.0):
            with self.assertRaisesRegex(TimeoutError, "telemetry timeout"):
                test.run_move(8, 0, 0, 10, 10)
        test.receive = lambda: [telemetry(fault="CMD_TIMEOUT")]
        with patch.object(tool.time, "monotonic", return_value=0.0):
            with self.assertRaisesRegex(RuntimeError, "ESP32 fault"):
                test.run_move(8, 0, 0, 10, 10)

    def test_main_finally_stops_on_success_fault_timeout_and_ctrl_c(self):
        instances = []

        class FakeTest:
            def __init__(self, ip):
                self.calls = []
                self.session_id, self.motion_id = 9, 7
                instances.append(self)

            def stop_phase(self, duration):
                self.calls.append("initial_stop")
                return True

            def run_move(self, *args):
                self.calls.append("move")
                if outcome is not None:
                    raise outcome
                return True

            def safe_stop(self):
                self.calls.append("final_stop")

            def close(self):
                self.calls.append("close")

        with patch.object(tool, "MoveTest", FakeTest):
            for outcome, expected in ((None, 0), (RuntimeError("fault"), 1),
                                      (TimeoutError("timeout"), 1), (KeyboardInterrupt(), 130)):
                with self.subTest(outcome=outcome):
                    self.assertEqual(tool.main(["--ip", "10.0.0.1"]), expected)
                    self.assertEqual(instances[-1].calls,
                                     ["initial_stop", "move", "final_stop", "close"])

    def test_safe_stop_repeats_packets(self):
        test = tool.MoveTest.__new__(tool.MoveTest)
        calls = []
        test.send = lambda **kwargs: calls.append(kwargs)
        clock = [0.0]
        with patch.object(tool.time, "monotonic", side_effect=lambda: clock[0]), \
                patch.object(tool.time, "sleep", side_effect=lambda seconds: clock.__setitem__(0, clock[0] + seconds)):
            test.safe_stop()
        self.assertGreaterEqual(len(calls), 20)
        self.assertTrue(all(not call for call in calls))


if __name__ == "__main__":
    unittest.main()
