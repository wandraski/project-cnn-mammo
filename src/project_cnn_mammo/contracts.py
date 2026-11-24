"""Contratos de dados do pipeline (POO por responsabilidade).

Cada módulo do pacote consome e produz estes tipos; nada de dicionários soltos
entre camadas.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

SplitName = Literal["train", "val", "test"]
Route = Literal["RESCUE_ANOMALY", "ABSTAIN_REVIEW", "AUTO_CONFIDENT"]


@dataclass(frozen=True)
class ManifestEntry:
    """Uma amostra do manifesto, identificada por paciente."""

    sample_key: str
    patient_key: str
    path: Path
    source: str
    label: int
    view: str = ""
    laterality: str = ""
    split: SplitName = "train"
    sha256: str = ""

    def __post_init__(self) -> None:
        if self.label not in {0, 1}:
            raise ValueError("label deve ser 0 ou 1")
        if self.split not in {"train", "val", "test"}:
            raise ValueError(f"split inválido: {self.split}")


@dataclass(frozen=True)
class SplitPlan:
    """Distribuição por split, sempre por paciente."""

    counts: dict[str, int] = field(default_factory=dict)
    patients: dict[str, set[str]] = field(default_factory=dict)

    @property
    def leakage_free(self) -> bool:
        return all(
            not (self.patients.get(left, set()) & self.patients.get(right, set()))
            for left, right in (("train", "val"), ("train", "test"), ("val", "test"))
        )


@dataclass(frozen=True)
class RunContext:
    """Contexto imutável de uma execução: é o que garante rastreabilidade."""

    seed: int
    commit: str
    data_version: str
    config_path: Path
    output_dir: Path
    environment: str = ""

    def as_dict(self) -> dict[str, object]:
        return {
            "seed": self.seed,
            "commit": self.commit,
            "data_version": self.data_version,
            "config": str(self.config_path),
            "output_dir": str(self.output_dir),
            "environment": self.environment,
        }


@dataclass(frozen=True)
class MetricSet:
    """Métricas com intervalo de confiança por bootstrap."""

    auroc: float
    auprc: float
    sensitivity_at_90_specificity: float
    specificity_at_90_sensitivity: float
    f1: float
    threshold: float
    ci95_low: float
    ci95_high: float
    n: int


@dataclass(frozen=True)
class GateDecision:
    """Decisão do gate híbrido (espelha `sprint10.GateDecision`)."""

    route: Route
    score: float
    uncertainty: float
    abstain: bool
    reason: str


@dataclass(frozen=True)
class ModelCard:
    """Registro mínimo de um modelo promovido (Encontro 18 do plano)."""

    name: str
    version: str
    architecture: str
    weights_sha256: str
    data_version: str
    threshold: float
    metrics: dict[str, float]
    intended_use: str = "pesquisa acadêmica; não é dispositivo médico"
    limitations: tuple[str, ...] = ()
