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


def batch_ablation(batches: Sequence[int] | None = None) -> list[RunSpec]:
    """S9 solo: dois braços por batch; no batch de referência são idênticos."""
    from .config import load_config

    s9 = load_config().section("sprint09")
    sizes = s9["batches"] if batches is None else batches
    lr = float(str(s9["learning_rate"]))
    reference = int(str(s9["linear_reference_batch"]))
    if not isinstance(sizes, (tuple, list)) or any(type(b) is not int or b <= 0 for b in sizes):
        raise ValueError("batches deve listar inteiros positivos")
    if reference <= 0 or not 0 < lr < 1:
        raise ValueError("LR fixo ou batch de referência inválido")
    return [
        RunSpec(f"batch-{size}-{rule}", {
            "batch_size": size, "lr_rule": rule, "learning_rate": rate,
            "epochs": s9["epochs"], "resolution": s9["resolution"], "optimizer": s9["optimizer"],
        })
        for size in sizes
        for rule, rate in (("fixed", lr), ("linear", lr * (size / reference)))
    ]


def resolution_ablation(resolutions: Sequence[int] = (224, 384, 512, 1024)) -> list[RunSpec]:
    return [RunSpec(f"res-{value}", {"resolution": value}) for value in resolutions]


def normalization_ablation() -> list[RunSpec]:
    return [RunSpec("norm-zscore", {"normalization": "zscore"}), RunSpec("norm-clahe", {"normalization": "clahe_zscore"})]
