"""Sprint 7: contrato de dados e prevenção de leakage."""

from collections.abc import Iterable, Mapping, Sequence


_CNN_ARCHITECTURES = ("resnet50", "efficientnet_b0", "efficientnet_b3", "convnext_tiny")


def classifier_architectures() -> tuple[str, ...]:
    """CNNs do escopo normativo; Transformers ficam fora da implementação principal."""
    return _CNN_ARCHITECTURES


def validate_patient_splits(rows: Iterable[Mapping[str, object]]) -> None:
    """Falha se o mesmo paciente aparecer em mais de um split."""
    seen: dict[str, str] = {}
    for row in rows:
        patient = str(row.get("patient_key", "")).strip()
        split = str(row.get("split", "")).strip()
        if not patient or split not in {"train", "val", "test"}:
            raise ValueError("patient_key e split válido são obrigatórios")
        previous = seen.setdefault(patient, split)
        if previous != split:
            raise ValueError(f"leakage do paciente {patient}: {previous} e {split}")


def threshold_for_specificity(
    labels: Sequence[int], scores: Sequence[float], target: float = 0.90
) -> float:
    """Maximiza sensibilidade mantendo a especificidade alvo na validação."""
    if len(labels) != len(scores) or not labels or any(label not in {0, 1} for label in labels):
        raise ValueError("labels e scores binários devem ter o mesmo tamanho")
    candidates: list[tuple[float, float]] = []
    for threshold in sorted(set(scores)):
        tp = tn = fp = fn = 0
        for label, score in zip(labels, scores, strict=True):
            positive = score >= threshold
            tp += int(label == 1 and positive)
            fn += int(label == 1 and not positive)
            tn += int(label == 0 and not positive)
            fp += int(label == 0 and positive)
        sensitivity = tp / (tp + fn) if tp + fn else 0.0
        specificity = tn / (tn + fp) if tn + fp else 0.0
        if specificity >= target:
            candidates.append((sensitivity, threshold))
    if not candidates:
        raise ValueError("nenhum limiar atende à especificidade solicitada")
    best_sensitivity = max(sensitivity for sensitivity, _ in candidates)
    return min(threshold for sensitivity, threshold in candidates if sensitivity == best_sensitivity)
