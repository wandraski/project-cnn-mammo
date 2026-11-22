import unittest

from project_cnn_mammo.sprint11 import aggregate_tta, refined_route


class Sprint11Tests(unittest.TestCase):
    def test_aggregates_weighted_tta_and_reports_dispersion(self):
        probability, dispersion = aggregate_tta(
            [0.2, 0.4, 0.6, 0.8], [0.40, 0.25, 0.175, 0.175]
        )
        self.assertAlmostEqual(probability, 0.425)
        self.assertGreater(dispersion, 0)
    def test_routes_refined_decisions_in_safety_order(self):
        self.assertEqual(refined_route(0.1, 0.8, True, 0.1), "RESCUE_ANOMALY")
        self.assertEqual(refined_route(0.8, 0.1, False, 0.1), "AUDIT_SALIENCY_OFF_LESION")
        self.assertEqual(refined_route(0.3, 0.1, True, 0.4), "ABSTAIN_SECOND_READ")
        self.assertEqual(refined_route(0.8, 0.1, True, 0.1), "AUTO_REFINED_CONFIDENT")


if __name__ == "__main__":
    unittest.main()
