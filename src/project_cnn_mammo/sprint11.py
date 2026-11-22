"""Sprint 11: refinamentos de inferência sobre checkpoints congelados."""

from collections.abc import Sequence
import math


def aggregate_tta(
    probabilities: Sequence[float], weights: Sequence[float]
) -> tuple[float, float]:
    if len(probabilities) != len(weights) or not probabilities:
        raise ValueError("probabilidades e pesos devem ter o mesmo tamanho")
    if any(not 0 <= value <= 1 for value in probabilities) or any(weight < 0 for weight in weights):
        raise ValueError("probabilidades ou pesos inválidos")
    total = sum(weights)
    if total <= 0:
        raise ValueError("a soma dos pesos deve ser positiva")
    normalized = [weight / total for weight in weights]
    mean = sum(weight * probability for weight, probability in zip(normalized, probabilities, strict=True))
    variance = sum(weight * (probability - mean) ** 2 for weight, probability in zip(normalized, probabilities, strict=True))
    return mean, math.sqrt(variance)


def refined_route(
    probability: float,
    anomaly_score: float,
    pointing_hit: bool,
    view_gap: float,
    *,
    clinical_threshold: float = 0.22,
    anomaly_threshold: float = 0.68,
    abstention_band: float = 0.065,
    max_view_gap: float = 0.35,
) -> str:
    """Aplica as quatro rotas da Sprint 11 em ordem conservadora."""
    if probability < clinical_threshold and anomaly_score >= anomaly_threshold:
        return "RESCUE_ANOMALY"
    if probability >= clinical_threshold and not pointing_hit:
        return "AUDIT_SALIENCY_OFF_LESION"
    if abs(probability - clinical_threshold) <= abstention_band or view_gap >= max_view_gap:
        return "ABSTAIN_SECOND_READ"
    return "AUTO_REFINED_CONFIDENT"
