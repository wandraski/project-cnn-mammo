"""Avaliação sem dependências: AUROC/AUPRC por ranking, pontos clínicos e bootstrap.

Implementação deliberadamente stdlib-first para que o CI rode sem PyTorch/numpy.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
import random

from .contracts import MetricSet


def confusion_matrix(
    labels: Sequence[int], scores: Sequence[float], threshold: float
) -> dict[str, int]:
    tp = tn = fp = fn = 0
    for label, score in zip(labels, scores, strict=True):
        positive = score >= threshold
        tp += int(label == 1 and positive)
        fn += int(label == 1 and not positive)
        tn += int(label == 0 and not positive)
        fp += int(label == 0 and positive)
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def _rates(labels: Sequence[int], scores: Sequence[float], threshold: float) -> tuple[float, float]:
    matrix = confusion_matrix(labels, scores, threshold)
    sensitivity = matrix["tp"] / (matrix["tp"] + matrix["fn"]) if matrix["tp"] + matrix["fn"] else 0.0
    specificity = matrix["tn"] / (matrix["tn"] + matrix["fp"]) if matrix["tn"] + matrix["fp"] else 0.0
    return sensitivity, specificity


def auroc(labels: Sequence[int], scores: Sequence[float]) -> float:
    """AUROC pelo estatístico de Mann-Whitney (empates contados como 0,5)."""
    if len(labels) != len(scores) or not labels:
        raise ValueError("labels e scores devem ter o mesmo tamanho não nulo")
    pairs = sorted(zip(scores, labels, strict=True), key=lambda pair: pair[0])
    rank_sum = 0.0
    index = 0
    while index < len(pairs):
        end = index + 1
        while end < len(pairs) and pairs[end][0] == pairs[index][0]:
            end += 1
        average_rank = (index + 1 + end) / 2
        for position in range(index, end):
            if pairs[position][1] == 1:
                rank_sum += average_rank
        index = end
    positives = sum(1 for label in labels if label == 1)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        raise ValueError("AUROC exige ambas as classes")
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def auprc(labels: Sequence[int], scores: Sequence[float]) -> float:
    """AUPRC por integração dos limiares distintos (curva precisão-recall)."""
    if len(labels) != len(scores) or not labels:
        raise ValueError("labels e scores devem ter o mesmo tamanho não nulo")
    positives_total = sum(1 for label in labels if label == 1)
    if positives_total == 0:
        raise ValueError("AUPRC exige ao menos um positivo")
    ordered = sorted(zip(scores, labels, strict=True), key=lambda pair: -pair[0])
    true_positive = 0
    area = 0.0
    previous_recall = 0.0
    for index, (_, label) in enumerate(ordered, start=1):
        true_positive += int(label == 1)
        precision = true_positive / index
        recall = true_positive / positives_total
        area += precision * (recall - previous_recall)
        previous_recall = recall
    return area


def f1(labels: Sequence[int], scores: Sequence[float], threshold: float) -> float:
    matrix = confusion_matrix(labels, scores, threshold)
    precision = matrix["tp"] / (matrix["tp"] + matrix["fp"]) if matrix["tp"] + matrix["fp"] else 0.0
    recall = matrix["tp"] / (matrix["tp"] + matrix["fn"]) if matrix["tp"] + matrix["fn"] else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def threshold_at_specificity(
    labels: Sequence[int], scores: Sequence[float], target: float = 0.90
) -> float:
    """Limiar que maximiza sensibilidade com a especificidade alvo (validação)."""
    from .sprint07 import threshold_for_specificity

    return threshold_for_specificity(labels, scores, target)


def threshold_at_sensitivity(
    labels: Sequence[int], scores: Sequence[float], target: float = 0.90
) -> float:
    """Limiar que maximiza especificidade com a sensibilidade alvo."""
    candidates: list[tuple[float, float]] = []
    for threshold in sorted(set(scores)):
        sensitivity, specificity = _rates(labels, scores, threshold)
        if sensitivity >= target:
            candidates.append((specificity, threshold))
    if not candidates:
        raise ValueError("nenhum limiar atende à sensibilidade solicitada")
    best = max(specificity for specificity, _ in candidates)
    return min(threshold for specificity, threshold in candidates if specificity == best)


def bootstrap_ci(
    labels: Sequence[int],
    scores: Sequence[float],
    *,
    samples: int = 1000,
    seed: int = 42,
    statistic: Callable[[Sequence[int], Sequence[float]], float],
) -> tuple[float, float]:
    """IC 95% por bootstrap com reprodução determinística."""
    if len(labels) != len(scores) or not labels:
        raise ValueError("labels e scores devem ter o mesmo tamanho não nulo")
    rng = random.Random(seed)
    values = []
    size = len(labels)
    for _ in range(samples):
        draw_labels = [labels[rng.randrange(size)] for _ in range(size)]
        draw_scores = [scores[rng.randrange(size)] for _ in range(size)]
        try:
            values.append(statistic(draw_labels, draw_scores))
        except ValueError:
            continue
    if not values:
        raise ValueError("bootstrap sem amostras válidas")
    values.sort()
    return values[int(0.025 * len(values))], values[min(len(values) - 1, int(0.975 * len(values)))]


def evaluate(
    labels: Sequence[int],
    scores: Sequence[float],
    *,
    samples: int = 1000,
    seed: int = 42,
) -> MetricSet:
    """Métrica primária do protocolo: sensibilidade com especificidade 0,90."""
    threshold = threshold_at_specificity(labels, scores)
    sensitivity, specificity = _rates(labels, scores, threshold)
    low, high = bootstrap_ci(
        labels, scores, samples=samples, seed=seed, statistic=lambda y, s: _rates(y, s, threshold)[0]
    )
    return MetricSet(
        auroc=auroc(labels, scores),
        auprc=auprc(labels, scores),
        sensitivity_at_90_specificity=sensitivity,
        specificity_at_90_sensitivity=_rates(
            labels, scores, threshold_at_sensitivity(labels, scores)
        )[1],
        f1=f1(labels, scores, threshold),
        threshold=threshold,
        ci95_low=low,
        ci95_high=high,
        n=len(labels),
    )
