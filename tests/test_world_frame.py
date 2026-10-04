import tempfile
import unittest
from pathlib import Path

import numpy as np

from perception.camera import WorldFrame
from perception.table_calibration import MarkerDefinition, TableCalibration


def make_calibration(path):
    markers = [MarkerDefinition(i, x, y, 5.0) for i, x, y in
               ((41, -29.5, 17), (42, 29.5, 17),
                (43, 29.5, -17), (44, -29.5, -17))]
    H = [[0.05, 0, -32], [0, -0.05, 20], [0, 0, 1]]
    TableCalibration((1280, 720), (65, 40), markers, H).save(path)


class WorldFrameTests(unittest.TestCase):
    def test_round_trip_and_actual_shape(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'table.json'
            make_calibration(path)
            world = WorldFrame(calibration_path=path)
            world.observe_frame((720, 1280, 3), read_at=10.0)
            self.assertTrue(world.can_send_motion(now=10.1))
            p = world.to_world(500, 300)
            np.testing.assert_allclose(world.to_pixel(*p), (500, 300), atol=1e-6)
            world.observe_frame((480, 640, 3), read_at=10.2)
            self.assertFalse(world.can_send_motion(now=10.2))
            self.assertIn('resolution', world.invalid_reason)

    def test_missing_file_and_stale_frame(self):
        world = WorldFrame(calibration_path='does-not-exist.json')
        world.observe_frame((720, 1280, 3), read_at=10.0)
        self.assertFalse(world.can_send_motion(now=10.0))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'table.json'
            make_calibration(path)
            world = WorldFrame(calibration_path=path)
            world.observe_frame((720, 1280, 3), read_at=10.0)
            self.assertFalse(world.can_send_motion(now=11.0))
            self.assertIn('stale', world.invalid_reason)
            world.invalidate_frame('read failed')
            self.assertFalse(world.can_send_motion(now=10.1))


if __name__ == '__main__':
    unittest.main()
