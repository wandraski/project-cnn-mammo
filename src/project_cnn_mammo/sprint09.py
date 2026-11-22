"""Sprint 9: desbalanceamento, prevalência e resgate auxiliar."""

from collections.abc import Sequence


def correct_prevalence(
    probability: float, source_prevalence: float, target_prevalence: float
) -> float:
    """Corrige odds sob mudança de prevalência sem alterar o ranking do modelo."""
    values = (probability, source_prevalence, target_prevalence)
    if any(not 0 < value < 1 for value in values):
        raise ValueError("probabilidade e prevalências devem estar entre 0 e 1")
    odds = probability / (1 - probability)
    prevalence_ratio = (target_prevalence / (1 - target_prevalence)) / (
        source_prevalence / (1 - source_prevalence)
    )
    corrected_odds = odds * prevalence_ratio
    return corrected_odds / (1 + corrected_odds)


def rescue_false_negatives(
    labels: Sequence[int],
    cnn_scores: Sequence[float],
    anomaly_scores: Sequence[float],
    *,
    cnn_threshold: float = 0.22,
    anomaly_threshold: float = 0.65,
) -> dict[str, int]:
    if not (len(labels) == len(cnn_scores) == len(anomaly_scores)):
        raise ValueError("entradas devem ter o mesmo tamanho")
    false_negatives = rescued = extra_false_positives = 0
    for label, cnn_score, anomaly_score in zip(labels, cnn_scores, anomaly_scores, strict=True):
        rescue = cnn_score < cnn_threshold and anomaly_score >= anomaly_threshold
        false_negatives += int(label == 1 and cnn_score < cnn_threshold)
        rescued += int(label == 1 and rescue)
        extra_false_positives += int(label == 0 and rescue)
    return {
        "false_negatives": false_negatives,
        "rescued": rescued,
        "extra_false_positives": extra_false_positives,
    }
