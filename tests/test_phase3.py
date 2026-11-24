"""Testes da Fase 3: imagem, DICOM, datasets e matriz experimental."""

from pathlib import Path
import tempfile
import unittest

from project_cnn_mammo.datasets import (
    DATASETS,
    INBREAST_NOTE,
    dataset,
    resolve_labels,
    scan_directory,
)
from project_cnn_mammo.experiment import (
    batch_ablation,
    expand_grid,
    normalization_ablation,
    pareto_front,
    resolution_ablation,
)
from project_cnn_mammo.imaging import clahe, clahe_fast, normalize_zscore, preprocess_array
from project_cnn_mammo.io_dicom import apply_windowing, invert_monochrome1


class TestImaging(unittest.TestCase):
    def setUp(self):
        # 8x8 com metade escura e metade clara
        self.width = 8
        self.height = 8
        self.pixels = [20] * 32 + [220] * 32

    def test_clahe_preserves_shape_and_is_deterministic(self):
        first = clahe(self.pixels, self.width, self.height)
        second = clahe(self.pixels, self.width, self.height)
        self.assertEqual(first, second)
        self.assertEqual(len(first), len(self.pixels))
        self.assertTrue(all(0 <= value <= 255 for value in first))

    def test_clahe_rejects_bad_geometry(self):
        with self.assertRaises(ValueError):
            clahe([1, 2, 3], 2, 2)

    def test_clahe_fast_without_cv2_returns_none(self):
        try:
            import cv2  # noqa: F401

            self.skipTest("cv2 instalado")
        except ImportError:
            pass
        self.assertIsNone(clahe_fast(self.pixels, self.width, self.height))

    def test_preprocess_array_fallback_path(self):
        result = preprocess_array(self.pixels, self.width, self.height)
        values = result["values"]
        assert isinstance(values, list)
        self.assertIn(result["source"], {"stdlib", "cv2"})
        self.assertEqual(len(values), len(self.pixels))

    def test_zscore_is_deterministic(self):
        values = normalize_zscore([1.0, 2.0, 3.0])
        self.assertAlmostEqual(values[-1], -values[0], places=9)


class TestDicom(unittest.TestCase):
    def test_read_dicom_requires_pydicom(self):
        from project_cnn_mammo.io_dicom import read_dicom

        try:
            import pydicom  # noqa: F401

            self.skipTest("pydicom instalado")
        except ImportError:
            pass
        with self.assertRaises(RuntimeError):
            read_dicom("inexistente.dcm")

    def test_invert_monochrome1(self):
        payload = {"pixels": [10.0, 20.0, 30.0], "monochrome1": True}
        self.assertEqual(invert_monochrome1(payload), [20.0, 10.0, 0.0])

    def test_windowing_saturates(self):
        payload = {"pixels": [0.0, 1000.0, 2000.0], "monochrome1": False,
                   "window_center": 1000.0, "window_width": 1000.0}
        self.assertEqual(apply_windowing(payload), [0, 128, 255])


class TestDatasets(unittest.TestCase):
    def test_scope_datasets_registered(self):
        self.assertEqual(
            set(DATASETS),
            {"cbisddsm", "vindr_mammo", "cmmd", "rsna_breast_cancer", "multibase_subset"},
        )
        self.assertTrue(dataset("vindr_mammo").requires_agreement)
        self.assertIn("INbreast", INBREAST_NOTE)

    def test_multibase_subset_is_development_only(self):
        source = dataset("multibase_subset")
        self.assertIn("desenvolvimento", source.role)
        self.assertTrue(source.url.endswith("subset-multibase-cancer-mama"))

    def test_scan_and_resolve(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "benigno").mkdir()
            (root / "maligno").mkdir()
            (root / "benigno" / "a.dcm").write_bytes(b"x")
            (root / "maligno" / "b.dcm").write_bytes(b"y")
            rows = scan_directory(root, source="cbisddsm")
            self.assertEqual(len(rows), 2)
            resolved = resolve_labels(rows, {"benigno": 0, "maligno": 1})
            self.assertEqual({row["label"] for row in resolved}, {0, 1})

    def test_unknown_dataset_fails(self):
        with self.assertRaises(ValueError):
            dataset("inbreast")


class TestExperiment(unittest.TestCase):
    def test_expand_grid_cartesian(self):
        specs = expand_grid({"epochs": 5}, {"batch_size": [8, 16], "resolution": [224, 512]})
        self.assertEqual(len(specs), 4)
        self.assertEqual(specs[0].tag, "run-000")

    def test_empty_grid_returns_base(self):
        specs = expand_grid({"epochs": 5}, {})
        self.assertEqual(len(specs), 1)
        self.assertEqual(specs[0].params, {"epochs": 5})

    def test_pareto_front_minimizes_cost_maximizes_metric(self):
        costs = [3.0, 1.0, 2.0, 2.5]
        metrics = [0.8, 0.7, 0.9, 0.6]
        self.assertEqual(pareto_front(costs, metrics), [1, 2])

    def test_ablations_match_protocol(self):
        self.assertEqual([spec.params["batch_size"] for spec in batch_ablation()], [8, 16, 32, 64])
        self.assertEqual(
            [spec.params["resolution"] for spec in resolution_ablation()], [224, 384, 512, 1024]
        )
        self.assertEqual(len(normalization_ablation()), 2)


class TestTorchGate(unittest.TestCase):
    def test_require_torch_message_is_actionable(self):
        from project_cnn_mammo.training import require_torch

        try:
            import torch  # noqa: F401

            self.skipTest("torch instalado")
        except ImportError:
            pass
        with self.assertRaises(RuntimeError) as ctx:
            require_torch()
        self.assertIn("extra `ml`", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
