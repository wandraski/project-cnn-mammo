"""Testes do núcleo da Fase 2: contratos, dados, pré-processamento e metrics."""

import json
from pathlib import Path
import unittest

from project_cnn_mammo.config import load_config
from project_cnn_mammo.contracts import ManifestEntry, ModelCard, SplitPlan
from project_cnn_mammo.data import assign_split, build_manifest, split_plan
from project_cnn_mammo.evaluation import auprc, auroc, bootstrap_ci, evaluate, threshold_at_sensitivity
from project_cnn_mammo.models import architectures, classifier, segmenter, supports_pretraining
from project_cnn_mammo.preprocessing import TransformPlan, apply_windowing, describe, zscore


class TestConfig(unittest.TestCase):
    def test_default_config(self):
        config = load_config()
        self.assertEqual(config.common.seeds, (42, 123, 999))
        self.assertTrue(config.common.patient_split)
        self.assertEqual(
            config.architectures, ("resnet50", "efficientnet_b0", "efficientnet_b3", "convnext_tiny")
        )

    def test_missing_config_fails_loudly(self):
        with self.assertRaises(FileNotFoundError):
            load_config("/tmp/inexistente-ao-xyz.toml")


class TestContractsAndData(unittest.TestCase):
    def _rows(self):
        return [
            {
                "sample_key": f"s{index}",
                "patient_key": f"p{index}",
                "path": f"/tmp/s{index}.dcm",
                "source": "cbisddsm",
                "label": index % 2,
                "split": "train" if index < 6 else "val",
            }
            for index in range(8)
        ]

    def test_manifest_and_split_plan(self):
        entries = build_manifest(self._rows())
        plan = split_plan(entries)
        self.assertEqual(plan.counts["train"], 6)
        self.assertTrue(plan.leakage_free)

    def test_leakage_is_refused(self):
        rows = self._rows()
        rows[0]["patient_key"] = rows[6]["patient_key"]
        with self.assertRaises(ValueError):
            build_manifest(rows)

    def test_split_assignment_is_deterministic(self):
        self.assertEqual(assign_split("paciente-a", {"train": 0.7, "val": 0.15, "test": 0.15}, 42),
                         assign_split("paciente-a", {"train": 0.7, "val": 0.15, "test": 0.15}, 42))
        self.assertNotEqual(
            assign_split("paciente-a", {"train": 0.7, "val": 0.15, "test": 0.15}, 42),
            assign_split("paciente-b", {"train": 0.7, "val": 0.15, "test": 0.15}, 42),
        )

    def test_bad_label_fails(self):
        with self.assertRaises(ValueError):
            ManifestEntry("s", "p", Path("x"), "cbisddsm", 2)


class TestPreprocessing(unittest.TestCase):
    def test_plan_rejects_out_of_protocol_resolution(self):
        with self.assertRaises(ValueError):
            TransformPlan(resolution=300)

    def test_windowing_and_inversion(self):
        windowed = apply_windowing([0.0, 1000.0, 2000.0], center=1000.0, width=1000.0)
        self.assertEqual(windowed, [0.0, 127.5, 255.0])
        self.assertEqual(describe([10.0, 20.0, 30.0], TransformPlan())["plan"]["resolution"], 512)

    def test_zscore_is_centered(self):
        values = zscore([0.0, 1.0, 2.0])
        self.assertAlmostEqual(sum(values) / len(values), 0.0, places=9)


class TestModels(unittest.TestCase):
    def test_registry_lists_cnn_scope(self):
        names = {item.name for item in architectures("cnn")}
        self.assertEqual(names, {"resnet50", "efficientnet_b0", "efficientnet_b3", "convnext_tiny"})

    def test_transformers_are_out_of_scope(self):
        with self.assertRaises(ValueError):
            classifier("vit_b16")

    def test_segmenters_and_pretraining(self):
        self.assertEqual(segmenter("unet_resnet34").name, "unet_resnet34")
        self.assertTrue(supports_pretraining("resnet50", "imagenet"))
        self.assertFalse(supports_pretraining("efficientnet_b0", "radimagenet"))


class TestEvaluation(unittest.TestCase):
    def setUp(self):
        self.labels = [0, 0, 1, 1]
        self.scores = [0.1, 0.2, 0.8, 0.9]

    def test_perfect_ranking(self):
        self.assertAlmostEqual(auroc(self.labels, self.scores), 1.0)
        self.assertAlmostEqual(auprc(self.labels, self.scores), 1.0)

    def test_threshold_at_sensitivity(self):
        threshold = threshold_at_sensitivity(self.labels, self.scores, target=0.5)
        self.assertGreaterEqual(threshold, 0.2)

    def test_bootstrap_reproducible(self):
        first = bootstrap_ci(self.labels, self.scores, samples=50, seed=7, statistic=auroc)
        second = bootstrap_ci(self.labels, self.scores, samples=50, seed=7, statistic=auroc)
        self.assertEqual(first, second)

    def test_evaluate_returns_primary_metric(self):
        metrics = evaluate(self.labels, self.scores, samples=50, seed=1)
        self.assertEqual(metrics.n, 4)
        self.assertGreater(metrics.ci95_high, metrics.ci95_low)


class TestPipelineAndServing(unittest.TestCase):
    def test_smoke_pipeline_writes_tracker(self):
        from project_cnn_mammo.pipeline import Pipeline, ingest, preprocess, validate
        from project_cnn_mammo.preprocessing import TransformPlan

        output = Path(self.id().split(".")[-1] + "-output")
        context_pipeline = Pipeline(
            [ingest([]), validate(), preprocess(TransformPlan())],
            output_dir=output,
            dry_run=True,
        )
        results = context_pipeline.run()
        self.assertEqual([result.status for result in results], ["ok", "ok", "ok"])
        runs = list((output / "mlruns").glob("*.json"))
        self.assertEqual(len(runs), 1)
        payload = json.loads(runs[0].read_text(encoding="utf-8"))
        self.assertEqual(payload["status"], "FINISHED")
        shutil_rmtree(output)

    def test_handle_uses_same_transform_as_training(self):
        from project_cnn_mammo.serving import handle

        card = ModelCard(
            name="resnet50",
            version="0.1.0",
            architecture="resnet50",
            weights_sha256="0" * 64,
            data_version="unset",
            threshold=0.22,
            metrics={"tau_clinical": 0.22},
        )
        response = handle({"pixels": [10.0, 20.0, 30.0], "probability": 0.9}, card=card, plan=TransformPlan())
        self.assertIn("route", response)
        self.assertEqual(response["transform"]["normalization"], "clahe_zscore")


def shutil_rmtree(path: Path) -> None:
    import shutil

    shutil.rmtree(path, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
