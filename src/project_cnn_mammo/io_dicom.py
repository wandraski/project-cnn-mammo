"""Leitura DICOM preguiçosa (pydicom) com metadados relevantes ao pipeline."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any


def read_dicom(path: Path | str) -> dict[str, Any]:
    """Lê um DICOM e devolve pixels em 16 bits + metadados normalizados.

    Lança `RuntimeError` com orientação clara quando pydicom não está instalado.
    """
    try:
        import pydicom  # type: ignore
    except ImportError as exc:  # pragma: no cover - depende do ambiente
        raise RuntimeError("instale o extra `ml` (pydicom) para ler DICOM") from exc

    dataset = pydicom.dcmread(str(path))
    pixel_array = dataset.pixel_array.astype("float32")
    center = getattr(dataset, "WindowCenter", None)
    width = getattr(dataset, "WindowWidth", None)
    return {
        "pixels": pixel_array.reshape(-1).tolist(),
        "width": int(pixel_array.shape[1]),
        "height": int(pixel_array.shape[0]),
        "bits_stored": int(getattr(dataset, "BitsStored", 16)),
        "photometric_interpretation": str(getattr(dataset, "PhotometricInterpretation", "")),
        "monochrome1": str(getattr(dataset, "PhotometricInterpretation", "")).upper() == "MONOCHROME1",
        "window_center": float(center) if center is not None else None,
        "window_width": float(width) if width is not None else None,
        "patient_id": str(getattr(dataset, "PatientID", "")),
        "view_position": str(getattr(dataset, "ViewPosition", "")),
        "laterality": str(getattr(dataset, "Laterality", "")),
        "path": str(path),
    }


def invert_monochrome1(payload: Mapping[str, Any]) -> list[float]:
    """Inverte pixels de `MONOCHROME1`; idempotente para `MONOCHROME2`."""
    values = [float(value) for value in payload["pixels"]]
    if not bool(payload.get("monochrome1")):
        return values
    maximum = max(values, default=0.0)
    return [maximum - value for value in values]


def apply_windowing(
    payload: Mapping[str, Any], *, force_bits: int = 8
) -> list[int]:
    """Janela DICOM para `force_bits`, respeitando center/width se existirem."""
    values = invert_monochrome1(payload)
    center = payload.get("window_center")
    width_value = payload.get("window_width")
    low = (float(center) - float(width_value) / 2) if center is not None and width_value is not None else min(values)
    high = (float(center) + float(width_value) / 2) if center is not None and width_value is not None else max(values)
    scale = (2**force_bits - 1) / (high - low) if high > low else 0.0
    return [max(0, min(2**force_bits - 1, int(round((value - low) * scale)))) for value in values]
