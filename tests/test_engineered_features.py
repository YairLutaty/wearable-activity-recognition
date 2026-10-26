from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "modeling"))

from engineered_features import TOTAL_FEATURES, extract_engineered_features


class EngineeredFeatureTests(unittest.TestCase):
    def test_feature_contract_is_finite_unique_and_has_expected_shape(self) -> None:
        time = np.arange(100, dtype=np.float64) / 20.0
        two_hz = np.sin(2.0 * np.pi * 2.0 * time)
        accel = np.zeros((2, 100, 3), dtype=np.float32)
        gyro = np.zeros_like(accel)
        accel[0, :, 0] = two_hz
        gyro[1, :, 1] = two_hz

        matrix, names = extract_engineered_features(accel, gyro, 20.0)

        self.assertEqual(matrix.shape, (2, TOTAL_FEATURES))
        self.assertEqual(len(names), TOTAL_FEATURES)
        self.assertEqual(len(set(names)), TOTAL_FEATURES)
        self.assertTrue(np.isfinite(matrix).all())
        dominant_index = names.index("accel_x_dominant_frequency_hz")
        self.assertAlmostEqual(matrix[0, dominant_index], 2.0)

    def test_invalid_shape_is_rejected(self) -> None:
        invalid = np.zeros((1, 99, 3), dtype=np.float32)
        with self.assertRaises(ValueError):
            extract_engineered_features(invalid, invalid, 20.0)

    def test_noncanonical_sampling_rate_is_rejected(self) -> None:
        sequences = np.zeros((1, 100, 3), dtype=np.float32)
        with self.assertRaises(ValueError):
            extract_engineered_features(sequences, sequences, 25.0)


if __name__ == "__main__":
    unittest.main()
