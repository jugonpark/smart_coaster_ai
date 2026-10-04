import math
import unittest

import numpy as np

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


if __name__ == '__main__':
    unittest.main()
