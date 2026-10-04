import unittest
from unittest.mock import patch

import numpy as np

import config
from perception.camera import Camera


class FakeCapture:
    def __init__(self, frame=None):
        self.frame = frame
        self.settings = []

    def set(self, key, value):
        self.settings.append((key, value))
        return True

    def isOpened(self):
        return True

    def read(self):
        return (self.frame is not None, self.frame)

    def release(self):
        pass


class FakeWorld:
    def __init__(self):
        self.shape = None
        self.error = None

    def observe_frame(self, shape):
        self.shape = shape

    def invalidate_frame(self, reason):
        self.error = reason


class CameraSourceTests(unittest.TestCase):
    def test_usb_and_mjpeg_share_read_interface(self):
        for url, expected in (('', config.CAMERA_INDEX),
                              ('http://192.168.1.2:8080/video',
                               'http://192.168.1.2:8080/video')):
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            capture = FakeCapture(frame)
            with self.subTest(url=url), patch.object(config, 'CAMERA_URL', url), \
                 patch('perception.camera.cv2.VideoCapture', return_value=capture) as open_capture, \
                 patch('perception.camera.WorldFrame', return_value=FakeWorld()):
                camera = Camera()
                ok, decoded = camera.read()
                self.assertTrue(ok)
                self.assertIs(decoded, frame)
                open_capture.assert_called_once_with(expected)
                self.assertEqual(camera.world.shape, frame.shape)

    def test_failed_network_read_invalidates_world(self):
        with patch.object(config, 'CAMERA_URL', 'https://example.com/mjpeg'), \
             patch('perception.camera.cv2.VideoCapture', return_value=FakeCapture()), \
             patch('perception.camera.WorldFrame', return_value=FakeWorld()):
            camera = Camera()
            self.assertEqual(camera.read(), (False, None))
            self.assertEqual(camera.world.error, 'camera read failed')

    def test_unsupported_nonempty_url_rejected(self):
        with patch.object(config, 'CAMERA_URL', 'rtsp://example.com/live'):
            with self.assertRaises(ValueError):
                Camera()


if __name__ == '__main__':
    unittest.main()
