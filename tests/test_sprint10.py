import unittest

from project_cnn_mammo.sprint10 import hybrid_gate


class Sprint10Tests(unittest.TestCase):
    def test_routes_rescue_abstention_and_confident_cases(self):
        self.assertEqual(hybrid_gate(0.10, 0.80, 0.90).route, "RESCUE_ANOMALY")
        self.assertEqual(hybrid_gate(0.25, 0.20, 0.90).route, "ABSTAIN_REVIEW")
        self.assertEqual(hybrid_gate(0.80, 0.10, 0.90).route, "AUTO_CONFIDENT")


if __name__ == "__main__":
    unittest.main()
