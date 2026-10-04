import time
import unittest
from unittest.mock import Mock, patch

import numpy as np

from comm.udp_sender import UdpSender
from main import guard_world_or_stop
from perception.camera import WorldFrame
from perception.table_calibration import MarkerDefinition, TableCalibration


def calibration():
    markers = [MarkerDefinition(i, x, y, 5.0) for i, x, y in
               ((41, -29.5, 17), (42, 29.5, 17),
                (43, 29.5, -17), (44, -29.5, -17))]
    return TableCalibration((1280, 720), (65, 40), markers,
                            [[0.05, 0, -32], [0, -0.05, 20], [0, 0, 1]])


class StopGateTests(unittest.TestCase):
    def test_missing_corrupt_nan_singular_and_wrong_size_are_invalid(self):
        missing = WorldFrame(calibration_path='missing-calibration.json')
        missing.observe_frame((720, 1280, 3))
        self.assertFalse(missing.can_send_motion())
        with patch('perception.camera.TableCalibration.load', side_effect=ValueError('corrupt')):
            corrupt = WorldFrame(calibration_path='bad.json')
        corrupt.observe_frame((720, 1280, 3))
        self.assertFalse(corrupt.can_send_motion())
        for H in ([[float('nan'), 0, 0], [0, 1, 0], [0, 0, 1]],
                  [[1, 0, 0], [1, 0, 0], [0, 0, 1]]):
            with self.assertRaises(ValueError):
                TableCalibration((1280, 720), (65, 40), calibration().markers, H)
        wrong = WorldFrame(calibration=calibration())
        wrong.observe_frame((480, 640, 3))
        self.assertFalse(wrong.can_send_motion())

    def test_pipeline_stops_before_perception_when_invalid(self):
        invalid = WorldFrame(calibration=calibration())
        invalid.observe_frame((480, 640, 3))
        sender, planner = Mock(), Mock()
        self.assertFalse(guard_world_or_stop(invalid, sender, planner))
        sender.send.assert_called_once_with(0, 0, 0, 'STOP', force=True)
        planner.reset.assert_called_once()

    def test_high_reprojection_error_keeps_world_invalid(self):
        base = calibration()
        poor_fit = TableCalibration((1280, 720), (65, 40), base.markers,
                                    base.H_image_to_table,
                                    rms_error_cm=12.0, max_error_cm=20.0)
        world = WorldFrame(calibration=poor_fit)
        world.observe_frame((720, 1280, 3))
        self.assertFalse(world.can_send_motion())
        self.assertIn('quality', world.invalid_reason)

    def test_udp_guard_replaces_run_with_stop_even_when_throttled(self):
        sender = UdpSender.__new__(UdpSender)
        sender.allow_motion = lambda: False
        sender.esp_ip = '127.0.0.1'
        sender.session_id = 7
        sender._seq = 0
        sender._last_heartbeat_t = time.time()
        sender._last_send_t = time.time()
        sender._min_interval = 999
        sender.poll = lambda: None
        sent = []
        sender._send_packet = lambda pkt: sent.append(pkt) or True
        packet = sender.send(5, 0, 0, 'RUN')
        self.assertEqual(packet.type, 'stop')
        self.assertEqual(sent[-1].type, 'stop')


if __name__ == '__main__':
    unittest.main()
