"""Testes do aprendizado não supervisionado pre-CNN."""

import math
import unittest

from project_cnn_mammo.unsupervised import (
    GaussianOneClass,
    KnnOneClass,
    autoencoder_scores,
    empirical_threshold,
    pre_cnn_flags,
    select_scorer,
)


class TestEmpiricalThreshold(unittest.TestCase):
    def test_quantiles(self):
        scores = [float(index) for index in range(100)]
        self.assertEqual(empirical_threshold(scores, 0.95), 94.0)

    def test_empty_fails(self):
        with self.assertRaises(ValueError):
            empirical_threshold([], 0.95)

    def test_invalid_quantile(self):
        with self.assertRaises(ValueError):
            empirical_threshold([1.0], 1.5)


class TestGaussianOneClass(unittest.TestCase):
    def setUp(self):
        # referencia concentrada em (0,0); ponto distante em (5,5)
        self.reference = [[0.0, 0.0], [0.1, -0.1], [-0.1, 0.1], [0.0, 0.0]]
        self.outlier = [[5.0, 5.0]]

    def test_outlier_scores_higher_than_reference(self):
        detector = GaussianOneClass().fit(self.reference)
        reference_max = max(detector.score(self.reference))
        outlier_score = detector.score(self.outlier)[0]
        self.assertGreater(outlier_score, reference_max)

    def test_threshold_is_empirical(self):
        detector = GaussianOneClass().fit(self.reference)
        limit = detector.threshold(0.95)
        self.assertGreaterEqual(limit, min(detector.score(self.reference)))

    def test_fit_rejects_inconsistent_dims(self):
        with self.assertRaises(ValueError):
            GaussianOneClass().fit([[1.0], [1.0, 2.0]])

    def test_score_before_fit_fails(self):
        with self.assertRaises(RuntimeError):
            GaussianOneClass().score([[1.0, 2.0]])


class TestKnnOneClass(unittest.TestCase):
    def test_distance_grows_with_isolation(self):
        reference = [[float(index)] for index in range(10)]
        detector = KnnOneClass(k=2).fit(reference)
        self.assertGreater(detector.score([[50.0]])[0], detector.score([[4.0]])[0])

    def test_k_must_be_positive(self):
        with self.assertRaises(ValueError):
            KnnOneClass(k=0)


class TestAutoencoderGate(unittest.TestCase):
    def test_returns_none_without_torch(self):
        try:
            import torch  # noqa: F401

            self.skipTest("torch instalado")
        except ImportError:
            pass
        self.assertIsNone(autoencoder_scores([[0.0, 1.0], [1.0, 0.0]], epochs=1))


class TestPreCnnFlags(unittest.TestCase):
    def test_flags_outliers_before_cnn(self):
        reference = [[0.0, 0.0]] * 20
        samples = [[0.0, 0.0]] * 5 + [[9.0, 9.0]]
        result = pre_cnn_flags(samples, reference=reference, method="knn", quantile=0.90)
        anomalies = result["anomalies"]
        assert isinstance(anomalies, list)
        self.assertIn(5, anomalies)
        self.assertEqual(result["method"], "knn")

    def test_unknown_method_fails(self):
        with self.assertRaises(ValueError):
            select_scorer("isolation_forest")


if __name__ == "__main__":
    unittest.main()
