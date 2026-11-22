import unittest

from project_cnn_mammo.sprint09 import correct_prevalence, rescue_false_negatives


class Sprint09Tests(unittest.TestCase):
    def test_corrects_probability_for_target_prevalence(self):
        corrected = correct_prevalence(0.5, source_prevalence=0.1, target_prevalence=0.2)
        self.assertAlmostEqual(corrected, 9 / 13)
    def test_counts_false_negatives_rescued_by_anomaly_score(self):
        result = rescue_false_negatives(
            labels=[1, 1, 0], cnn_scores=[0.1, 0.3, 0.1], anomaly_scores=[0.8, 0.9, 0.7]
        )
        self.assertEqual(result, {"false_negatives": 1, "rescued": 1, "extra_false_positives": 1})


if __name__ == "__main__":
    unittest.main()
