"""Aprendizado não supervisionado **antes** da CNN: detecção de anomalia auxiliar.

Papel no pipeline (nunca substitui a avaliação supervisionada):
1. etapa pré-CNN que pontua amostras fora da distribuição;
2. sinal auxiliar para resgate de falsos negativos e para revisão humana;
3. base para ponderação de amostras raras.

Implementações stdlib (testáveis sem GPU) + autoencoder opcional (extra `ml`).
"""

from __future__ import annotations

from collections.abc import Sequence
import math
from typing import Protocol


class AnomalyScorer(Protocol):
    def fit(self, samples: Sequence[Sequence[float]]) -> "AnomalyScorer": ...
    def score(self, samples: Sequence[Sequence[float]]) -> list[float]: ...
    def threshold(self, quantile: float = 0.95) -> float: ...


def _mean_vector(samples: Sequence[Sequence[float]], dim: int) -> list[float]:
    return [sum(sample[index] for sample in samples) / len(samples) for index in range(dim)]


def _covariance_inverse(
    samples: Sequence[Sequence[float]], mean: Sequence[float], *, ridge: float = 1e-6
) -> list[list[float]]:
    """Matriz de covariância regularizada e invertida (Gauss-Jordan)."""
    dim = len(mean)
    matrix = [[0.0] * dim for _ in range(dim)]
    for sample in samples:
        delta = [sample[index] - mean[index] for index in range(dim)]
        for row in range(dim):
            for column in range(dim):
                matrix[row][column] += delta[row] * delta[column]
    scale = max(len(samples) - 1, 1)
    for row in range(dim):
        for column in range(dim):
            matrix[row][column] /= scale
        matrix[row][row] += ridge

    inverse = [[1.0 if row == column else 0.0 for column in range(dim)] for row in range(dim)]
    for pivot in range(dim):
        candidate = max(range(pivot, dim), key=lambda row: abs(matrix[row][pivot]))
        if abs(matrix[candidate][pivot]) < 1e-12:
            raise ValueError("covariância singular; reduza a dimensionalidade ou aumente a amostra")
        matrix[pivot], matrix[candidate] = matrix[candidate], matrix[pivot]
        inverse[pivot], inverse[candidate] = inverse[candidate], inverse[pivot]
        factor = matrix[pivot][pivot]
        matrix[pivot] = [value / factor for value in matrix[pivot]]
        inverse[pivot] = [value / factor for value in inverse[pivot]]
        for row in range(dim):
            if row == pivot or matrix[row][pivot] == 0.0:
                continue
            ratio = matrix[row][pivot]
            matrix[row] = [left - ratio * right for left, right in zip(matrix[row], matrix[pivot])]
            inverse[row] = [left - ratio * right for left, right in zip(inverse[row], inverse[pivot])]
    return inverse


def _mahalanobis(sample: Sequence[float], mean: Sequence[float], inverse: Sequence[Sequence[float]]) -> float:
    delta = [sample[index] - mean[index] for index in range(len(mean))]
    return math.sqrt(
        sum(delta[row] * inverse[row][column] * delta[column] for row in range(len(delta)) for column in range(len(delta)))
    )


class GaussianOneClass:
    """One-class por distância de Mahalanobis (referência: população esperada)."""

    def __init__(self, ridge: float = 1e-6):
        self.ridge = ridge
        self.mean: list[float] | None = None
        self.inverse: list[list[float]] | None = None
        self.fitted: list[list[float]] = []

    def fit(self, samples: Sequence[Sequence[float]]) -> "GaussianOneClass":
        if not samples:
            raise ValueError("amostra de referência vazia")
        width = len(samples[0])
        if any(len(sample) != width for sample in samples):
            raise ValueError("dimensões inconsistentes")
        self.fitted = [list(sample) for sample in samples]
        self.mean = _mean_vector(samples, width)
        self.inverse = _covariance_inverse(samples, self.mean, ridge=self.ridge)
        return self

    def score(self, samples: Sequence[Sequence[float]]) -> list[float]:
        if self.mean is None or self.inverse is None:
            raise RuntimeError("detector não ajustado")
        return [_mahalanobis(sample, self.mean, self.inverse) for sample in samples]

    def threshold(self, quantile: float = 0.95) -> float:
        if not self.fitted:
            raise RuntimeError("detector não ajustado")
        return empirical_threshold(self.score(self.fitted), quantile)


def empirical_threshold(scores: Sequence[float], quantile: float = 0.95) -> float:
    """Limiar empírico por quantil dos scores de referência (sem suposição gaussiana)."""
    if not scores or not 0.0 <= quantile <= 1.0:
        raise ValueError("scores não vazios e quantile em [0,1]")
    ordered = sorted(scores)
    index = min(len(ordered) - 1, max(0, int(round(quantile * (len(ordered) - 1)))))
    return ordered[index]


class KnnOneClass:
    """One-class por distância média aos k vizinhos na referência."""

    def __init__(self, k: int = 5):
        if k < 1:
            raise ValueError("k deve ser positivo")
        self.k = k
        self.reference: list[list[float]] = []

    def fit(self, samples: Sequence[Sequence[float]]) -> "KnnOneClass":
        if not samples:
            raise ValueError("amostra de referência vazia")
        self.reference = [list(sample) for sample in samples]
        return self

    def score(self, samples: Sequence[Sequence[float]]) -> list[float]:
        if not self.reference:
            raise RuntimeError("detector não ajustado")
        values: list[float] = []
        for sample in samples:
            distances = sorted(_euclidean(sample, other) for other in self.reference)
            neighbours = distances[: min(self.k, len(distances))]
            values.append(sum(neighbours) / len(neighbours) if neighbours else 0.0)
        return values

    def threshold(self, quantile: float = 0.95) -> float:
        return empirical_threshold(self.score(self.reference), quantile)


def _euclidean(left: Sequence[float], right: Sequence[float]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right, strict=True)))


def autoencoder_scores(samples: Sequence[Sequence[float]], *, epochs: int = 30, seed: int = 42):
    """Erro de reconstrução de autoencoder (torch). Retorna `None` sem o extra `ml`."""
    try:
        import torch  # type: ignore
    except ImportError:
        return None
    from .training import set_determinism

    set_determinism(seed)
    width = len(samples[0])
    tensor = torch.tensor([list(sample) for sample in samples], dtype=torch.float32)
    model = torch.nn.Sequential(
        torch.nn.Linear(width, max(2, width // 2)),
        torch.nn.ReLU(),
        torch.nn.Linear(max(2, width // 2), width),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = torch.nn.functional.mse_loss(model(tensor), tensor)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        errors = torch.mean((model(tensor) - tensor) ** 2, dim=1)
    return errors.cpu().tolist()


def select_scorer(method: str) -> AnomalyScorer:
    if method == "gaussian":
        return GaussianOneClass()
    if method == "knn":
        return KnnOneClass()
    raise ValueError(f"método desconhecido: {method}")


def pre_cnn_flags(
    samples: Sequence[Sequence[float]],
    *,
    reference: Sequence[Sequence[float]] | None = None,
    method: str = "gaussian",
    quantile: float = 0.95,
) -> dict[str, object]:
    """Etapa pré-CNN: ajusta o detector e devolve scores/limiar/anomalias."""
    scorer = select_scorer(method)
    base = list(reference) if reference is not None else list(samples)
    scorer.fit(base)
    scores = scorer.score(samples)
    limit = scorer.threshold(quantile)
    anomalies = [index for index, value in enumerate(scores) if value >= limit]
    return {
        "method": method,
        "n": len(samples),
        "threshold": limit,
        "anomalies": anomalies,
        "scores": [round(value, 6) for value in scores],
    }
