import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from perception.marker_scanner import MarkerScan
from perception.table_calibration import MarkerDefinition, TableCalibration


class TableCalibrationTests(unittest.TestCase):
    def setUp(self):
        self.markers = [
            MarkerDefinition(41, -29.5, 17.0, 5.0, 0.0),
            MarkerDefinition(42, 29.5, 17.0, 5.0, 0.0),
            MarkerDefinition(43, 29.5, -17.0, 5.0, 0.0),
            MarkerDefinition(44, -29.5, -17.0, 5.0, 0.0),
        ]
        self.pixel_to_table = np.array([
            [0.051, 0.002, -32.0],
            [0.001, -0.055, 20.0],
            [0.00002, -0.00001, 1.0],
        ])
        inverse = np.linalg.inv(self.pixel_to_table)
        found = {}
        for marker in self.markers:
            world_corners = marker.table_corners()
            homogeneous = np.column_stack([world_corners, np.ones(4)]) @ inverse.T
            pixels = homogeneous[:, :2] / homogeneous[:, 2:]
            found[marker.id] = pixels.astype(np.float32)
        self.scan = MarkerScan(by_id=found)

    def test_sixteen_corners_fit_and_round_trip(self):
        calibration = TableCalibration.from_scan(
            self.scan, self.markers, (1280, 720), (65.0, 40.0))
        self.assertEqual(calibration.correspondence_count, 16)
        self.assertLess(calibration.rms_error_cm, 1e-3)
        self.assertLess(calibration.max_error_cm, 1e-3)
        world = calibration.pixel_to_table(515.0, 300.0)
        pixel = calibration.table_to_pixel(*world)
        np.testing.assert_allclose(pixel, (515.0, 300.0), atol=1e-2)

    def test_marker_corner_order_and_rotation(self):
        marker = MarkerDefinition(41, 0.0, 0.0, 4.0, 90.0)
        np.testing.assert_allclose(marker.table_corners()[0], (-2.0, -2.0), atol=1e-10)
        np.testing.assert_allclose(marker.table_corners()[1], (-2.0, 2.0), atol=1e-10)

    def test_missing_marker_and_bad_matrix_rejected(self):
        with self.assertRaises(ValueError):
            TableCalibration.from_scan(MarkerScan(by_id={41: self.scan.get(41)}),
                                       self.markers, (1280, 720), (65.0, 40.0))
        calibration = TableCalibration.from_scan(
            self.scan, self.markers, (1280, 720), (65.0, 40.0))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'calibration.json'
            calibration.save(path)
            data = json.loads(path.read_text(encoding='utf-8'))
            data['homography_image_to_table'][0][0] = float('nan')
            path.write_text(json.dumps(data), encoding='utf-8')
            with self.assertRaises(ValueError):
                TableCalibration.load(path)


if __name__ == '__main__':
    unittest.main()
