import unittest

from project_cnn_mammo.sprint07 import (
    classifier_architectures,
    threshold_for_specificity,
    validate_patient_splits,
)


class Sprint07Tests(unittest.TestCase):
    def test_rejects_patient_leakage_between_splits(self):
        rows = [
            {"patient_key": "p1", "split": "train"},
            {"patient_key": "p1", "split": "test"},
        ]
        with self.assertRaisesRegex(ValueError, "p1"):
            validate_patient_splits(rows)

    def test_exposes_only_cnn_architectures_from_the_professor_scope(self):
        self.assertEqual(
            classifier_architectures(),
            ("resnet50", "efficientnet_b0", "efficientnet_b3", "convnext_tiny"),
        )
    def test_selects_threshold_only_from_validation_scores(self):
        threshold = threshold_for_specificity([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8], 0.5)
        self.assertEqual(threshold, 0.35)


if __name__ == "__main__":
    unittest.main()
