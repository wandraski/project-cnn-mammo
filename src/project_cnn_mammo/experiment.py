"""Matriz experimental e frente de Pareto (stdlib, testável sem GPU).

A matriz é declarativa: mudamos um parâmetro por vez em torno da configuração
base, como exige o protocolo docente.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import product


@dataclass(frozen=True)
class RunSpec:
    tag: str
    params: dict[str, object]

    def to_row(self) -> dict[str, object]:
        return {"tag": self.tag, **self.params}


def expand_grid(
    base: Mapping[str, object],
    grid: Mapping[str, Sequence[object]],
    *,
    prefix: str = "run",
) -> list[RunSpec]:
    """Produto cartesiano dos eixos; inclui a configuração base quando grade vazia."""
    keys = list(grid)
    if not keys:
        return [RunSpec(f"{prefix}-base", dict(base))]
    specs: list[RunSpec] = []
    for index, combination in enumerate(product(*(grid[key] for key in keys))):
        params = dict(base)
        overrides = {key: value for key, value in zip(keys, combination, strict=True)}
        params.update(overrides)
        specs.append(RunSpec(f"{prefix}-{index:03d}", params))
    return specs


def pareto_front(
    costs: Sequence[float], metrics: Sequence[float]
) -> list[int]:
    """Índices não dominados: menor custo e maior métrica.

    Custo e métrica devem ter o mesmo tamanho.
    """
    if len(costs) != len(metrics) or not costs:
        raise ValueError("custos e métricas devem ter o mesmo tamanho não nulo")
    front: list[int] = []
    for index, (cost, metric) in enumerate(zip(costs, metrics, strict=True)):
        dominated = any(
            other_cost <= cost and other_metric >= metric
            for other_index, (other_cost, other_metric) in enumerate(zip(costs, metrics, strict=True))
            if other_index != index
            and (other_cost < cost or other_metric > metric)
        )
        if not dominated:
            front.append(index)
    return sorted(front, key=lambda index: costs[index])


def matrix_rows(specs: Sequence[RunSpec]) -> list[dict[str, object]]:
    return [spec.to_row() for spec in specs]


def batch_ablation(batches: Sequence[int] = (8, 16, 32, 64)) -> list[RunSpec]:
    """Ablação obrigatória do plano: batch sizes com todo o resto constante."""
    return [
        RunSpec(f"batch-{size}", {"batch_size": size, "lr_rule": "linear" if size > 8 else "fixed"})
        for size in batches
    ]


def resolution_ablation(resolutions: Sequence[int] = (224, 384, 512, 1024)) -> list[RunSpec]:
    return [RunSpec(f"res-{value}", {"resolution": value}) for value in resolutions]


def normalization_ablation() -> list[RunSpec]:
    return [RunSpec("norm-zscore", {"normalization": "zscore"}), RunSpec("norm-clahe", {"normalization": "clahe_zscore"})]
