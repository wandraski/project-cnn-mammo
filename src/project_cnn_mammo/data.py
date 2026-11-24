"""Camada de dados: manifesto, hashing e split por paciente.

Nenhuma imagem entra no Git; o manifesto carrega apenas caminhos, rótulos,
metadados e hash do artefato.
"""

from __future__ import annotations

import csv
import hashlib
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import cast

from .contracts import ManifestEntry, SplitPlan, SplitName
from .sprint07 import threshold_for_specificity, validate_patient_splits


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Hash SHA-256 de um arquivo, em streaming."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def build_manifest(
    rows: Iterable[Mapping[str, object]], root: Path | None = None
) -> list[ManifestEntry]:
    """Converte linhas brutas em `ManifestEntry`, validando campos obrigatórios."""
    entries: list[ManifestEntry] = []
    base = Path(root) if root else None
    for row in rows:
        missing = {"sample_key", "patient_key", "path", "source", "label"} - set(row)
        if missing:
            raise ValueError(f"campos obrigatórios ausentes: {sorted(missing)}")
        raw_path = Path(str(row["path"]))
        path = (base / raw_path) if base and not raw_path.is_absolute() else raw_path
        entries.append(
            ManifestEntry(
                sample_key=str(row["sample_key"]),
                patient_key=str(row["patient_key"]),
                path=path,
                source=str(row["source"]),
                label=int(str(row["label"])),
                view=str(row.get("view", "")),
                laterality=str(row.get("laterality", "")),
                split=cast("SplitName", str(row.get("split", "train"))),
                sha256=str(row.get("sha256", "")),
            )
        )
    validate_patient_splits(
        [{"patient_key": entry.patient_key, "split": entry.split} for entry in entries]
    )
    return entries


def read_manifest(path: Path) -> list[ManifestEntry]:
    """Lê manifesto CSV; recusa leakage antes de qualquer uso."""
    with Path(path).open(newline="", encoding="utf-8") as stream:
        entries = build_manifest(csv.DictReader(stream))
    return entries


def split_plan(entries: Sequence[ManifestEntry]) -> SplitPlan:
    """Resume contagem por split e conjunto de pacientes por split."""
    counts: dict[str, int] = {}
    patients: dict[str, set[str]] = {}
    for entry in entries:
        counts[entry.split] = counts.get(entry.split, 0) + 1
        patients.setdefault(entry.split, set()).add(entry.patient_key)
    return SplitPlan(counts=counts, patients=patients)


def assign_split(
    patient_key: str, ratios: Mapping[str, float], seed: int
) -> str:
    """Split determinístico por paciente (sem estado global).

    O hash do paciente com a seed decide o split, garantindo que o mesmo
    paciente caia sempre no mesmo split entre execuções.
    """
    if not patient_key:
        raise ValueError("patient_key não pode ser vazio")
    total = sum(ratios.values())
    if total <= 0:
        raise ValueError("razões de split devem somar valor positivo")
    digest = hashlib.sha256(f"{seed}:{patient_key}".encode()).hexdigest()
    draw = (int(digest[:8], 16) % 10_000) / 10_000
    cumulative = 0.0
    for name in ("train", "val", "test"):
        cumulative += ratios[name] / total
        if draw < cumulative:
            return name
    return "test"


__all__ = [
    "assign_split",
    "build_manifest",
    "read_manifest",
    "sha256_bytes",
    "sha256_file",
    "split_plan",
    "threshold_for_specificity",
]
