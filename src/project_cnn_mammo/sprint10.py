"""Sprint 10: gate híbrido e predição seletiva."""

from dataclasses import dataclass


@dataclass(frozen=True)
class GateDecision:
    route: str
    score: float
    uncertainty: float
    abstain: bool
    reason: str


def hybrid_gate(
    p_cnn: float,
    s_anom: float,
    unet_conf: float,
    *,
    tau_clinical: float = 0.22,
    tau_anomaly: float = 0.68,
    alpha: float = 0.78,
    band: float = 0.08,
) -> GateDecision:
    if any(not 0 <= value <= 1 for value in (p_cnn, s_anom, unet_conf, alpha)):
        raise ValueError("probabilidades e alpha devem estar entre 0 e 1")
    score = alpha * p_cnn + (1 - alpha) * s_anom
    margin = abs(p_cnn - tau_clinical)
    uncertainty = 0.55 * (1 - min(margin / 0.30, 1)) + 0.45 * abs(p_cnn - s_anom)
    if p_cnn < tau_clinical and s_anom >= tau_anomaly:
        return GateDecision("RESCUE_ANOMALY", score, uncertainty, False, "anomalia alta")
    if margin <= band or unet_conf < 0.50:
        return GateDecision("ABSTAIN_REVIEW", score, uncertainty, True, "caso incerto")
    return GateDecision("AUTO_CONFIDENT", score, uncertainty, False, "sinais convergentes")
