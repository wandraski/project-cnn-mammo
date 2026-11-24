"""Portao de qualidade: recusa metrica primaria abaixo do minimo normativo.

Uso: python scripts/quality_gate.py <diretorio-de-runs> <metricas.json>
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

MINIMUM_SENSITIVITY_AT_90_SPECIFICITY = 0.80
MINIMUM_AUROC = 0.80


def gate(metrics: dict[str, float]) -> tuple[bool, list[str]]:
    failures: list[str] = []
    sensitivity = float(metrics.get("sens_at_90_spec", 0.0))
    auroc = float(metrics.get("auroc", 0.0))
    if sensitivity < MINIMUM_SENSITIVITY_AT_90_SPECIFICITY:
        failures.append(f"sensibilidade@90esp {sensitivity:.4f} < {MINIMUM_SENSITIVITY_AT_90_SPECIFICITY}")
    if auroc < MINIMUM_AUROC:
        failures.append(f"AUROC {auroc:.4f} < {MINIMUM_AUROC}")
    return not failures, failures


def main(argv: list[str]) -> int:
    metrics_path = Path(argv[2] if len(argv) > 2 else "outputs/metrics.json")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    passed, failures = gate(metrics)
    for failure in failures:
        print(f"[reprovado] {failure}")
    print("portao: aprovado" if passed else "portao: reprovado")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
