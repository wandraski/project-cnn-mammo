"""Sprint 8: avaliação de calibração probabilística."""

from collections.abc import Sequence


def calibration_metrics(
    labels: Sequence[int], probabilities: Sequence[float], bins: int = 10
) -> dict[str, float]:
    if len(labels) != len(probabilities) or not labels or bins < 1:
        raise ValueError("labels, probabilidades e bins inválidos")
    if any(label not in {0, 1} for label in labels):
        raise ValueError("labels devem ser binários")
    if any(not 0 <= probability <= 1 for probability in probabilities):
        raise ValueError("probabilidades devem estar entre 0 e 1")

    brier = sum((probability - label) ** 2 for label, probability in zip(labels, probabilities, strict=True)) / len(labels)
    ece = mce = 0.0
    for index in range(bins):
        low, high = index / bins, (index + 1) / bins
        members = [i for i, p in enumerate(probabilities) if low <= p < high or index == bins - 1 and p == 1]
        if not members:
            continue
        accuracy = sum(labels[i] for i in members) / len(members)
        confidence = sum(probabilities[i] for i in members) / len(members)
        gap = abs(accuracy - confidence)
        ece += len(members) / len(labels) * gap
        mce = max(mce, gap)
    return {"brier": brier, "ece": ece, "mce": mce}
