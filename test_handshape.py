"""Run with: python -m unittest test_handshape -v."""
import tempfile
from pathlib import Path
import unittest
import sys

import numpy as np
import torch

import handshape


class HandshapeTests(unittest.TestCase):
    def test_selection_and_clip_normalization(self):
        points = np.arange(2 * 75 * 3, dtype=np.float32).reshape(2, 75, 3) / 100
        points[:, 11, :2] = [[0.2, 0.4], [0.3, 0.5]]
        points[:, 12, :2] = [[0.6, 0.4], [0.9, 0.5]]
        original = points.copy()
        tensor = handshape.prepare_input(points)
        selected = [0, 2, 5, 11, 12, 13, 14, 33, 37, 38, 41, 42, 45,
                    46, 49, 50, 53, 54, 58, 59, 62, 63, 66, 67, 70, 71, 74]
        center = np.array([0.5, 0.45])
        expected = ((points[:, selected, :2] - center) / 0.5).transpose(2, 0, 1)[None]
        np.testing.assert_allclose(tensor.numpy(), expected, atol=2e-6)
        np.testing.assert_array_equal(points, original)
        self.assertEqual(tuple(tensor.shape), (1, 2, 2, 27))

    def test_invalid_input(self):
        for points in [np.zeros((0, 75, 3)), np.zeros((16, 225)),
                       np.full((3, 75, 3), np.nan), np.zeros((3, 75, 3))]:
            with self.subTest(shape=points.shape), self.assertRaises(ValueError):
                handshape.prepare_input(points)
        points = np.zeros((3, 75, 3), dtype=np.float32)
        points[:, 11, 0] = 0.2
        points[:, 12, 0] = 0.6
        with self.assertRaisesRegex(ValueError, 'No hand'):
            handshape.prepare_input(points)

    def test_region_does_not_import_api_or_cl0(self):
        points = np.zeros((8, 75, 3), dtype=np.float32)
        self.assertEqual(handshape.signing_region(points), (0, 7))
        self.assertNotIn('server.main', sys.modules)
        self.assertNotIn('predict', sys.modules)

    def test_real_checkpoint_and_deterministic_forward(self):
        model, labels = handshape.load_model()
        self.assertEqual(len(labels), 50)
        self.assertIn('-1', labels)
        self.assertIn('open_b', labels)
        torch.manual_seed(12)
        tensor = torch.randn(1, 2, 19, 27)
        with torch.inference_mode():
            first = model(tensor)
            second = model(tensor)
        self.assertEqual(tuple(first.shape), (1, 50))
        self.assertTrue(torch.isfinite(first).all())
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        self.assertAlmostEqual(first.softmax(-1).sum().item(), 1, places=6)

    def test_missing_encoder_weight_rejected(self):
        checkpoint = handshape.read_checkpoint(handshape.DEFAULT_CHECKPOINT)
        checkpoint = {'state_dict': checkpoint['state_dict']}
        del checkpoint['state_dict']['model.encoder.data_bn.weight']
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'incomplete.ckpt'
            torch.save(checkpoint, path)
            with self.assertRaisesRegex(RuntimeError, 'Missing key'):
                handshape.load_model(path)


if __name__ == '__main__':
    unittest.main()
