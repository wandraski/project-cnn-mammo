"""Execução real de treino sobre um diretório de dados (Kaggle/Colab/Modal).

Uma única implementação para os três ambientes: o notebook só aponta o caminho
do dataset e as épocas fixas da configuração.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from .config import load_config
from .contracts import ManifestEntry
from .data import build_manifest, sha256_file
from .evaluation import auroc
from .training import build_classifier, fit, predict_proba, require_torch, set_determinism


def epoch_plan(config_path: Path | None = None) -> dict[str, int]:
    config = load_config(config_path)
    return {
        "benchmark": int(str(config.section("treino")["classifier_epochs"])),
        "linear_probe": int(str(config.section("sprint08")["linear_probe_epochs"])),
        "batch_ablation": int(str(config.section("sprint09")["epochs"])),
    }


def build_dataset(rows: list[ManifestEntry], resolution: int):
    """Dataset do benchmark CNN; mantém os valores normalizados em float."""
    import numpy as np
    from PIL import Image

    torch, _ = require_torch()
    from .imaging import preprocess_array
    from .io_dicom import apply_windowing, read_dicom

    class _MammoDataset(torch.utils.data.Dataset):
        def __len__(self) -> int:
            return len(rows)

        def __getitem__(self, index: int):
            row = rows[index]
            if row.path.suffix.lower() in {".dcm", ".dicom"}:
                payload = read_dicom(row.path)
                width, height = payload["width"], payload["height"]
                pixels = apply_windowing(payload)
            else:
                with Image.open(row.path) as source:
                    gray = source.convert("L")
                    width, height = gray.size
                    pixels = list(gray.tobytes())
            result = preprocess_array(pixels, width, height)
            array = np.asarray(result["values"], dtype="float32").reshape(height, width)
            image = Image.fromarray(array).resize((resolution, resolution), Image.Resampling.BILINEAR)
            # ponytail: replicação de cinza para a entrada RGB dos encoders ImageNet.
            tensor = torch.from_numpy(np.array(image)).unsqueeze(0).repeat(3, 1, 1)
            return tensor, torch.tensor(float(row.label))

    return _MammoDataset()


def kaggle_task_manifest(data_dir: Path, output: Path) -> None:
    """Converte o manifesto da tarefa sem tocar nas imagens do split test."""
    source = data_dir / 'metadata/tasks/manifesto_benigno_x_maligno.csv'
    columns = ('image_id', 'patient_id', 'relative_path', 'dataset', 'label_task', 'split')
    labels = {'benigno': 0, 'maligno': 1}
    rows = []
    with source.open(newline='', encoding='utf-8-sig') as stream:
        reader = csv.DictReader(stream)
        if not set(columns) <= set(reader.fieldnames or ()):
            raise ValueError(f'Colunas obrigatórias no manifesto Kaggle: {", ".join(columns)}')
        for line, row in enumerate(reader, start=2):
            split = (row['split'] or '').strip().lower()
            if split == 'test':
                continue
            if split not in {'train', 'val'} or any(not (row[key] or '').strip() for key in columns):
                raise ValueError(f'Linha {line}: split ou campo obrigatório inválido no manifesto Kaggle')
            image, patient, path, dataset, label, _ = (row[key].strip() for key in columns)
            if label.lower() not in labels:
                raise ValueError(f'Linha {line}: label_task precisa ser benigno ou maligno')
            rows.append((f'{dataset}:{image}', patient, path, dataset, labels[label.lower()], split))
    with output.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(('sample_key', 'patient_key', 'path', 'source', 'label', 'split'))
        writer.writerows(rows)


def development_manifest(data_dir: Path | str, manifest_path: Path | None = None) -> list[ManifestEntry]:
    """Valida os metadados sem abrir imagens nem consultar o teste final."""
    base = Path(data_dir).resolve()
    with (manifest_path or base / "manifest.csv").open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    required = {"sample_key", "patient_key", "path", "source", "label", "split"}
    if not rows or any(not required <= row.keys() or any(not (row[k] or "").strip() for k in required) for row in rows):
        raise ValueError(f"manifest.csv precisa de linhas preenchidas com: {', '.join(sorted(required))}")
    if any(row["split"] not in {"train", "val"} for row in rows):
        raise ValueError("manifesto de desenvolvimento só aceita train/val; mantenha o teste final separado")
    if len({row["sample_key"] for row in rows}) != len(rows):
        raise ValueError("sample_key duplicada no manifesto")
    entries = build_manifest(rows, root=base)
    paths = set()
    for entry in entries:
        path = entry.path.resolve()
        if not path.is_relative_to(base) or not path.is_file() or path in paths:
            raise ValueError("imagem ausente, duplicada ou fora da raiz do manifesto")
        paths.add(path)
    if any({row.label for row in entries if row.split == split} != {0, 1} for split in ("train", "val")):
        raise ValueError("treino e validação precisam conter ambas as classes")
    return entries


def train_on_directory(
    data_dir: Path | str,
    *,
    architecture: str = "resnet50",
    seed: int | None = None,
    config_path: Path | None = None,
    output_dir: Path | str = "outputs",
    batch_size: int | None = None,
    resolution: int | None = None,
    device: str | None = None,
    epochs: int | None = None,
    learning_rate: float | None = None,
    linear_probe: bool = False,
    pretrained: bool = True,
    manifest_path: Path | None = None,
    limit_per_class: int | None = None,
) -> dict[str, object]:
    """Treina uma configuração sobre o mesmo manifesto de desenvolvimento."""
    manifest_path = manifest_path or Path(data_dir) / "manifest.csv"
    entries = development_manifest(data_dir, manifest_path)
    manifest_sha = sha256_file(manifest_path)
    train_rows = [row for row in entries if row.split == "train"]
    val_rows = [row for row in entries if row.split == "val"]
    if limit_per_class is not None:
        if limit_per_class < 1:
            raise ValueError("limite do smoke deve ser positivo")
        train_rows, val_rows = (
            [row for label in (0, 1) for row in [entry for entry in split if entry.label == label][:limit_per_class]]
            for split in (train_rows, val_rows)
        )

    config = load_config(config_path)
    benchmark = config.section("treino")
    epochs = int(str(benchmark["classifier_epochs"])) if epochs is None else epochs
    seed = int(str(benchmark["reference_seed"])) if seed is None else seed
    batch_size = int(str(benchmark["batch_size"])) if batch_size is None else batch_size
    resolution = int(str(benchmark["resolution"])) if resolution is None else resolution
    learning_rate = float(str(benchmark["learning_rate"])) if learning_rate is None else learning_rate
    if min(epochs, batch_size, resolution) <= 0 or not 0 < learning_rate < 1:
        raise ValueError("épocas, batch, resolução ou learning rate inválidos")
    torch, _ = require_torch()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    set_determinism(seed)
    model = build_classifier(architecture, pretrained=pretrained)
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA indisponível: habilite GPU no ambiente antes do treino")
        torch.cuda.reset_peak_memory_stats()
    from time import perf_counter

    started = perf_counter()
    generator = torch.Generator().manual_seed(seed)
    train_loader = torch.utils.data.DataLoader(
        build_dataset(train_rows, resolution), batch_size=batch_size, shuffle=True,
        generator=generator, num_workers=0,
    )
    val_loader = torch.utils.data.DataLoader(
        build_dataset(val_rows, resolution), batch_size=batch_size, num_workers=0
    )
    history = fit(model, train_loader, val_loader, epochs=epochs, learning_rate=learning_rate,
                  seed=seed, device=device, linear_probe=linear_probe)
    elapsed = perf_counter() - started

    # ponytail: guarda predições de validação; não publica IC até validar o bootstrap por paciente.
    scores = predict_proba(model, val_loader, device=device)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / f"{architecture}-seed{seed}.pt")
    payload = {
        "architecture": architecture,
        "seed": seed,
        "epochs": epochs,
        "resolution": resolution,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "manifest_sha256": manifest_sha,
        "device": device,
        "pretrained": pretrained,
        "linear_probe": linear_probe,
        "training_seconds": elapsed,
        "peak_gpu_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else None,
        "validation_auroc": auroc([row.label for row in val_rows], scores),
        "history": history,
        "validation_predictions": [
            {"sample_key": row.sample_key, "label": row.label, "score": score}
            for row, score in zip(val_rows, scores, strict=True)
        ],
    }
    (out / f"{architecture}-seed{seed}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return payload


def run_matrix(
    data_dir: Path | str,
    *,
    seeds: tuple[int, ...] | None = None,
    architectures: tuple[str, ...] | None = None,
    config_path: Path | None = None,
    output_dir: Path | str = "outputs",
    batch_size: int | None = None,
    resolution: int | None = None,
    device: str | None = None,
) -> list[dict[str, object]]:
    """Executa a grade de referência, preservando o manifesto entre seeds."""
    config = load_config(config_path)
    if seeds is None:
        seeds = config.common.seeds
    if architectures is None:
        architectures = config.architectures
    results = []
    for architecture in architectures:
        for seed in seeds:
            results.append(
                train_on_directory(
                    data_dir,
                    architecture=architecture,
                    seed=seed,
                    config_path=config_path,
                    output_dir=output_dir,
                    batch_size=batch_size,
                    resolution=resolution,
                    device=device,
                )
            )
    return results
