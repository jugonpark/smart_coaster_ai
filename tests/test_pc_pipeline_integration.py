import unittest
from unittest.mock import Mock, patch

import numpy as np

import config
import main
from perception.camera import WorldFrame
from perception.marker_scanner import MarkerScan
from perception.pose_tracker import HumanPose
from tests.test_calibration_stop_gate import calibration


class FakeCamera:
    def __init__(self, valid):
        self.world = WorldFrame(calibration=calibration())
        self.frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        self.valid = valid
        self.read_count = 0

    def read(self):
        self.read_count += 1
        if self.read_count > 1:
            raise KeyboardInterrupt
        self.world.observe_frame((720, 1280, 3) if self.valid else (480, 640, 3))
        return True, self.frame

    def release(self):
        pass


class FakeSender:
    def __init__(self):
        self.commands = []

    def send(self, vx, vy, w, status, force=False):
        self.commands.append((vx, vy, w, status, force))

    def link_text(self):
        return 'fake link'

    def close(self):
        pass


class PCPipelineIntegrationTests(unittest.TestCase):
    def run_one_frame(self, valid):
        camera, sender = FakeCamera(valid), FakeSender()
        robot = np.array([[300, 300], [340, 300],
                          [340, 340], [300, 340]], dtype=np.float32)
        cup = np.array([[650, 300], [680, 300],
                        [680, 330], [650, 330]], dtype=np.float32)
        scanner = Mock()
        scanner.scan.return_value = MarkerScan(by_id={0: robot, 1: cup})
        pose = Mock()
        pose.process.return_value = HumanPose(detected=False)
        with patch.object(config, 'SHOW_WINDOW', False), \
             patch.object(config, 'USE_HAND', False), \
             patch.object(config, 'USE_GAZE', False), \
             patch.object(config, 'TEST_MODE_ARUCO', True), \
             patch.object(config, 'YOLO_BACKEND', 'stub'), \
             patch('main.Camera', return_value=camera), \
             patch('main.PoseTracker', return_value=pose), \
             patch('main.MarkerScanner', return_value=scanner), \
             patch('main.UdpSender', return_value=sender), \
             patch('main.SettingsPanel', return_value=Mock()), \
             patch('main.IncidentLogger', return_value=Mock()):
            main.main()
        return sender.commands, scanner

    def test_invalid_frame_stops_before_scanning(self):
        commands, scanner = self.run_one_frame(False)
        self.assertEqual(commands, [(0, 0, 0, 'STOP', True)])
        scanner.scan.assert_not_called()

    def test_valid_frame_reaches_pc_decision_and_fake_sender(self):
        commands, scanner = self.run_one_frame(True)
        scanner.scan.assert_called_once()
        self.assertEqual(len(commands), 1)
        self.assertEqual(commands[0][3], 'RUN')

    def test_failed_read_clears_cached_perception_before_next_frame(self):
        class RecoveringCamera(FakeCamera):
            def read(self):
                self.read_count += 1
                if self.read_count == 1:
                    self.world.invalidate_frame('camera read failed')
                    return False, None
                if self.read_count > 2:
                    raise KeyboardInterrupt
                self.world.observe_frame(self.frame.shape)
                return True, self.frame

        camera, sender = RecoveringCamera(True), FakeSender()
        marker_detector = Mock()
        marker_detector._seen = {1: 'old cup'}
        scanner = Mock()
        scanner.scan.side_effect = lambda frame: (_ for _ in ()).throw(KeyboardInterrupt)
        with patch.object(config, 'SHOW_WINDOW', False), \
             patch.object(config, 'USE_HAND', False), \
             patch.object(config, 'USE_GAZE', False), \
             patch.object(config, 'TEST_MODE_ARUCO', True), \
             patch.object(config, 'YOLO_BACKEND', 'stub'), \
             patch('main.Camera', return_value=camera), \
             patch('main.PoseTracker', return_value=Mock()), \
             patch('main.MarkerScanner', return_value=scanner), \
             patch('main.MarkerObjectDetector', return_value=marker_detector), \
             patch('main.UdpSender', return_value=sender), \
             patch('main.SettingsPanel', return_value=Mock()), \
             patch('main.IncidentLogger', return_value=Mock()), \
             patch('main.time.sleep'):
            main.main()
        self.assertEqual(marker_detector._seen, {})
        self.assertEqual(sender.commands[0], (0, 0, 0, 'STOP', True))


if __name__ == '__main__':
    unittest.main()
