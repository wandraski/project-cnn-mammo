import unittest

from project_cnn_mammo.sprint08 import calibration_metrics


class Sprint08Tests(unittest.TestCase):
    def test_reports_brier_ece_and_mce(self):
        metrics = calibration_metrics([0, 1, 1, 0], [0.1, 0.9, 0.6, 0.4], bins=2)
        self.assertAlmostEqual(metrics["brier"], 0.085)
        self.assertAlmostEqual(metrics["ece"], 0.25)
        self.assertAlmostEqual(metrics["mce"], 0.25)


if __name__ == "__main__":
    unittest.main()
