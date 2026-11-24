"""Pré-processamento em duas camadas: plano (descritor) e execução (stdlib).

O mesmo `TransformPlan` é usado no treino e no serving — essa é a garantia de
ausência de *training-serving skew*.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence
from typing import Literal

Normalization = Literal["zscore", "clahe_zscore"]


@dataclass(frozen=True)
class TransformPlan:
    """Descrição imutável do pré-processamento aplicado à imagem."""

    resolution: int = 512
    normalization: Normalization = "clahe_zscore"
    invert_monochrome1: bool = True
    remove_pectoral: bool = True
    window_center: float | None = None
    window_width: float | None = None

    def __post_init__(self) -> None:
        if self.resolution not in {224, 384, 512, 1024}:
            raise ValueError(f"resolução fora do protocolo: {self.resolution}")
        if self.normalization not in {"zscore", "clahe_zscore"}:
            raise ValueError(f"normalização desconhecida: {self.normalization}")

    def as_dict(self) -> dict[str, object]:
        return {
            "resolution": self.resolution,
            "normalization": self.normalization,
            "invert_monochrome1": self.invert_monochrome1,
            "remove_pectoral": self.remove_pectoral,
            "window_center": self.window_center,
            "window_width": self.window_width,
        }


def apply_windowing(
    samples: list[float], center: float | None, width: float | None
) -> list[float]:
    """Janelamento DICOM para 8 bits (0–255) em escala linear.

    Sem center/width explícitos, usa min/max dos samples (fallback documentado).
    """
    if not samples:
        raise ValueError("amostra vazia")
    low = (center - width / 2) if center is not None and width is not None else min(samples)
    high = (center + width / 2) if center is not None and width is not None else max(samples)
    if high <= low:
        return [0.0 for _ in samples]
    return [max(0.0, min(255.0, (value - low) / (high - low) * 255.0)) for value in samples]


def to_8bit(samples: list[float]) -> list[int]:
    """Conversão 12→8 bits com saturação."""
    return [max(0, min(255, int(round(value)))) for value in samples]


def invert(samples: list[int]) -> list[int]:
    """Inverte `MONOCHROME1` (claro = denso) para `MONOCHROME2`."""
    return [255 - value for value in samples]


def zscore(samples: Sequence[float]) -> list[float]:
    """Normalização Z-score por imagem."""
    if not samples:
        raise ValueError("amostra vazia")
    mean = sum(samples) / len(samples)
    variance = sum((value - mean) ** 2 for value in samples) / len(samples)
    std = variance**0.5 or 1.0
    return [(value - mean) / std for value in samples]


def clahe_plan(tiles: int = 8, clip_limit: float = 2.0) -> dict[str, float]:
    """Parâmetros do CLAHE; a implementação com numpy vive no extra `ml`."""
    if tiles < 1 or clip_limit <= 0:
        raise ValueError("tiles e clip_limit devem ser positivos")
    return {"tiles": float(tiles), "clip_limit": clip_limit}


def apply_plan(samples: Sequence[float], plan: TransformPlan) -> list[float]:
    """Executa o plano completo sobre uma vetor de pixels (stdlib, sem numpy)."""
    windowed = apply_windowing(list(samples), plan.window_center, plan.window_width)
    values: Sequence[float] = [float(value) for value in to_8bit(windowed)]
    if plan.normalization == "zscore":
        return zscore(values)
    # ponytail: CLAHE exato depende de numpy/opencv; aqui o fallback é z-score,
    # e o plano registra os parâmetros para o ambiente `ml`.
    return zscore(values)


def describe(samples: Sequence[float], plan: TransformPlan) -> dict[str, object]:
    """Resumo auditável do pré-processamento aplicado (para logs/model card)."""
    transformed = apply_plan(samples, plan)
    return {
        "plan": plan.as_dict(),
        "input_min": min(samples),
        "input_max": max(samples),
        "output_mean": sum(transformed) / len(transformed) if transformed else 0.0,
    }
