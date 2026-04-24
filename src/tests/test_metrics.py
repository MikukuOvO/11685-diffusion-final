import unittest

import numpy as np

from fid_utils import compute_inception_score_from_probs


class MetricUtilsTests(unittest.TestCase):
    def test_inception_score_from_probs_rewards_diverse_confident_predictions(self):
        probs = np.eye(3, dtype=np.float64)[[0, 1, 2, 0, 1, 2]]

        mean, std = compute_inception_score_from_probs(probs, splits=3)

        self.assertAlmostEqual(mean, 2.0, places=6)
        self.assertAlmostEqual(std, 0.0, places=6)

    def test_inception_score_from_probs_rejects_too_few_samples(self):
        probs = np.array([[0.5, 0.5], [0.5, 0.5]])

        with self.assertRaises(ValueError):
            compute_inception_score_from_probs(probs, splits=3)


if __name__ == "__main__":
    unittest.main()
