"""Registro de datasets e varredura de diretórios para manifesto.

Sem download implícito: cada fonte declara licença e procedência, e o manifesto
só é montado sobre dados já presentes no ambiente (Kaggle/HF/disco local).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatasetSource:
    name: str
    role: str
    license_note: str
    url: str
    requires_agreement: bool = False
    recommended_split: str = "train"


DATASETS: dict[str, DatasetSource] = {
    "cbisddsm": DatasetSource(
        "CBIS-DDSM",
        "referência (treino/validação)",
        "TCIA/NBIA; uso público com citação",
        "https://www.cancerimagingarchive.net/collection/cbisddsm/",
    ),
    "vindr_mammo": DatasetSource(
        "VinDr-Mammo",
        "validação externa 1",
        "PhysioNet: credencial + treinamento CITI + acordo de uso",
        "https://physionet.org/content/vindrmammo/1.0.0/",
        requires_agreement=True,
        recommended_split="test",
    ),
    "cmmd": DatasetSource(
        "CMMD",
        "validação externa 2",
        "TCIA; aberto com citação",
        "https://doi.org/10.7937/tcia.eqde4b16",
        recommended_split="test",
    ),
    "rsna_breast_cancer": DatasetSource(
        "RSNA Screening (Kaggle 2022)",
        "prova de realidade (prevalência ~2%)",
        "regras da competição Kaggle",
        "https://www.kaggle.com/competitions/rsnabreastcancerdetection/data",
        requires_agreement=True,
        recommended_split="test",
    ),
}

INBREAST_NOTE = (
    "INbreast não deve ser usado: sem link oficial; espelhos de procedência incerta. "
    "CMMD é o substituto normativo."
)


def sources() -> tuple[DatasetSource, ...]:
    return tuple(DATASETS.values())


def dataset(name: str) -> DatasetSource:
    try:
        return DATASETS[name]
    except KeyError as exc:
        raise ValueError(f"dataset desconhecido: {name}; opções: {sorted(DATASETS)}") from exc


def scan_directory(
    root: Path | str,
    *,
    source: str,
    pattern: str = "**/*",
    label_from: str = "parent",
    default_label: int | None = None,
) -> list[dict[str, object]]:
    """Varre um diretório e produz linhas de manifesto (sem ler pixels).

    `label_from="parent"` usa o nome da pasta pai como rótulo textual, que deve
    ser convertido por `labels_map` antes de `build_manifest`.
    """
    base = Path(root)
    if not base.exists():
        raise FileNotFoundError(f"diretório inexistente: {base}")
    rows: list[dict[str, object]] = []
    for path in sorted(base.glob(pattern)):
        if not path.is_file() or path.suffix.lower() in {".txt", ".csv", ".json"}:
            continue
        relative = path.relative_to(base)
        label_value = relative.parts[0] if label_from == "parent" and len(relative.parts) > 1 else None
        if default_label is not None:
            label_value = default_label
        rows.append(
            {
                "sample_key": str(relative.with_suffix("")),
                "patient_key": relative.parts[0] if len(relative.parts) > 1 else relative.stem,
                "path": str(path),
                "source": source,
                "label": label_value,
            }
        )
    return rows


def resolve_labels(rows: Iterable[Mapping[str, object]], mapping: Mapping[str, int]) -> list[dict[str, object]]:
    """Converte rótulos textuais em inteiros usando o mapa informado."""
    resolved: list[dict[str, object]] = []
    for row in rows:
        raw = row.get("label")
        if isinstance(raw, int):
            resolved.append(dict(row))
            continue
        key = str(raw).strip().lower()
        if key not in mapping:
            raise ValueError(f"rótulo fora do mapa: {raw!r}")
        resolved.append({**row, "label": int(mapping[key])})
    return resolved


def kaggle_setup_notes() -> dict[str, str]:
    """Orientações de ambiente Kaggle/Colab antes de processar."""
    return {
        "kaggle": "Settings > Accelerator > GPU T4 x2 ou P100; anexe o dataset ao notebook.",
        "colab": "Ambiente de execução > Alterar tipo > GPU T4; monte Drive se o dataset for grande.",
        "modal": "Rode `modal run modal_app.py::run_pipeline` com `gpu='T4'`.",
        "comum": "Nunca baixe o teste final para o mesmo volume de treino.",
    }
