"""Orquestração por etapas, no espírito Lambda: batch e serving compartilham o núcleo.

As etapas são funções puras sobre contratos; `dry_run` permite validar o pipeline
sem dataset nem GPU.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import SprintConfig, load_config
from .contracts import ManifestEntry, RunContext
from .data import build_manifest, split_plan
from .evaluation import evaluate
from .preprocessing import TransformPlan
from .registry import ModelRegistry
from .sprint10 import hybrid_gate
from .sprint11 import aggregate_tta, refined_route
from .tracking import JsonTracker, context_as_params
from .unsupervised import pre_cnn_flags

# A etapa de aprendizado não supervisionado roda ANTES da CNN: ela só pontua
# anomalias; nunca decide o rótulo.


@dataclass(frozen=True)
class StageResult:
    name: str
    status: str
    detail: dict[str, Any]


Stage = Callable[[list[ManifestEntry], RunContext], StageResult]


def ingest(rows: Sequence[dict[str, object]], root: Path | None = None) -> Stage:
    def run(entries: list[ManifestEntry], context: RunContext) -> StageResult:
        manifest = build_manifest(rows, root=root)
        return StageResult(
            "ingest",
            "ok",
            {"samples": len(manifest), "output_dir": str(context.output_dir)},
        )

    return run


def validate() -> Stage:
    def run(entries: list[ManifestEntry], context: RunContext) -> StageResult:
        plan = split_plan(entries)
        if not plan.leakage_free:
            raise ValueError("vazamento de paciente entre splits")
        return StageResult("validate", "ok", {"counts": plan.counts})

    return run


def preprocess(plan: TransformPlan) -> Stage:
    def run(entries: list[ManifestEntry], context: RunContext) -> StageResult:
        return StageResult("preprocess", "ok", {"transform": plan.as_dict(), "samples": len(entries)})

    return run


def evaluate_stage(labels: Sequence[int], scores: Sequence[float]) -> Stage:
    def run(entries: list[ManifestEntry], context: RunContext) -> StageResult:
        metrics = evaluate(labels, scores, samples=context.as_dict().get("bootstrap_samples", 1000) or 1000)  # type: ignore[arg-type]
        return StageResult(
            "evaluate",
            "ok",
            {
                "auroc": round(metrics.auroc, 4),
                "auprc": round(metrics.auprc, 4),
                "sens_at_90_spec": round(metrics.sensitivity_at_90_specificity, 4),
                "ci95": [round(metrics.ci95_low, 4), round(metrics.ci95_high, 4)],
            },
        )

    return run


class Pipeline:
    """Sequência nomeada de etapas com tracker e falha rápida."""

    def __init__(
        self,
        stages: Sequence[Stage],
        *,
        config_path: Path | None = None,
        output_dir: Path | None = None,
        dry_run: bool = False,
    ):
        self.stages = list(stages)
        self.config: SprintConfig = load_config(config_path)
        self.dry_run = dry_run
        resolved_output = Path(output_dir) if output_dir else Path("outputs")
        self.context = RunContext(
            seed=self.config.common.seeds[0],
            commit="local",
            data_version="unset",
            config_path=config_path or Path("configs/sprints.toml"),
            output_dir=resolved_output,
        )
        self.tracker = JsonTracker(self.context.output_dir / "mlruns")

    def run(self, entries: list[ManifestEntry] | None = None) -> list[StageResult]:
        data = entries or []
        if self.dry_run and not data:
            data = [
                ManifestEntry(
                    sample_key=f"synthetic-{index}",
                    patient_key=f"patient-{index}",
                    path=Path(f"synthetic/{index}.dcm"),
                    source="synthetic",
                    label=index % 2,
                    split="train",
                )
                for index in range(8)
            ]
        results: list[StageResult] = []
        self.tracker.start_run("pipeline")
        self.tracker.log_params(context_as_params(self.context))
        try:
            for stage in self.stages:
                results.append(stage(data, self.context))
        except Exception as exc:  # falha rápida com causa registrada
            self.tracker.log_metrics({"failed_stage": float(len(results))})
            self.tracker.end_run("FAILED")
            raise
        self.tracker.log_metrics({"stages_ok": float(len(results))})
        self.tracker.end_run("FINISHED")
        return results


def refine_decision(
    probabilities: Sequence[float],
    weights: Sequence[float],
    *,
    anomaly_score: float = 0.0,
    pointing_hit: bool = True,
    view_gap: float = 0.0,
) -> dict[str, object]:
    """Adapter da Sprint 11 para o vocabulário de rota do gateway."""
    score, dispersion = aggregate_tta(probabilities, weights)
    route = refined_route(
        score,
        anomaly_score,
        pointing_hit,
        view_gap,
    )
    return {"route": route, "score": score, "dispersion": dispersion}


def gate_decision(p_cnn: float, s_anom: float, unet_conf: float) -> object:
    """Adapter do gate híbrido (Sprint 10)."""
    return hybrid_gate(p_cnn, s_anom, unet_conf)


def registry(root: Path | str = "artifacts/registry") -> ModelRegistry:
    return ModelRegistry(root)
