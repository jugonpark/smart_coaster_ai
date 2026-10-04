import math
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import config
import main
from perception.table_drawing import project_circle, project_vector
from tests.test_calibrated_perception import world


class CalibratedDrawingTests(unittest.TestCase):
    def test_heading_and_velocity_arrows_use_inverse_homography(self):
        transform = world()
        heading_start, heading_end = project_vector(
            transform, 5.0, -4.0, math.cos(0.8), math.sin(0.8), 8.0)
        np.testing.assert_allclose(heading_start, transform.to_pixel(5.0, -4.0))
        np.testing.assert_allclose(heading_end,
                                   transform.to_pixel(5 + 8*math.cos(0.8),
                                                      -4 + 8*math.sin(0.8)))
        _, velocity_end = project_vector(transform, 5, -4, 2, -3, 1.0)
        np.testing.assert_allclose(velocity_end, transform.to_pixel(7, -7))

    def test_cm_radius_is_projected_as_a_curve(self):
        transform = world()
        points = project_circle(transform, 12, -5, 4, samples=16)
        self.assertEqual(len(points), 16)
        np.testing.assert_allclose(points[0], transform.to_pixel(16, -5))
        # Perspective gives different pixel lengths on opposite sides.
        center = np.array(transform.to_pixel(12, -5))
        self.assertNotAlmostEqual(np.linalg.norm(np.array(points[0]) - center),
                                  np.linalg.norm(np.array(points[8]) - center), places=3)

    def test_cup_on_robot_ring_uses_inverse_homography(self):
        transform = world()
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        robot = SimpleNamespace(detected=True, x_cm=12, y_cm=-5,
                                px=(100, 100), heading_rad=0)
        human = SimpleNamespace(wrists=[])
        risk = SimpleNamespace(level='SAFE', reason='test', distance_cm=None)
        field = SimpleNamespace(speed=0, in_local_minima=False,
                                has_goal=False, goal_reached=False)
        with patch.object(config, 'DRAW_DETECTIONS', False), \
             patch.object(config, 'DRAW_FIELD_VECTOR', False), \
             patch.object(main.cv2, 'polylines') as draw_curve:
            main.draw_hud(frame, transform, robot, None, [], human, risk,
                          field, 0, 0, 0, 'STOP', 30, cup_on_robot=True)
        draw_curve.assert_called_once()
        expected = np.asarray(main.cv_points(project_circle(
            transform, 12, -5, 4)), dtype=np.int32)
        np.testing.assert_array_equal(draw_curve.call_args.args[1][0].reshape(-1, 2), expected)


if __name__ == '__main__':
    unittest.main()
