#!/usr/bin/env python3
"""Gera os cadernos da Fase 3 em ordem de execução (reprodutiveis e versionados).

Uso: python scripts/make_notebooks.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks"

HEADER = [
    ("markdown", "# Project CNN Mammo — Fase 3\n"),
]

NOTEBOOKS: list[tuple[str, str, str, list[tuple[str, str]]]] = [
    (
        "00_ingestao",
        "00 · Ingestão e manifesto",
        "Baixa/valida os dados, registra licença e monta o manifesto com SHA-256.",
        [
            ("code", "import sys, pathlib\nsys.path.insert(0, str(pathlib.Path.cwd() / 'src'))\nfrom project_cnn_mammo.datasets import DATASETS, INBREAST_NOTE\nprint(INBREAST_NOTE)\nfor name, source in DATASETS.items():\n    print(f'{name:20s} {source.role:32s} acordo={source.requires_agreement}')"),
            ("code", "from project_cnn_mammo.data import scan_directory, resolve_labels\nrows = scan_directory('/data/cbisddsm', source='cbisddsm')\nprint(len(rows), 'arquivos encontrados')\nprint(rows[0] if rows else 'sem arquivos: monte o volume do dataset primeiro')"),
            ("code", "import csv, pathlib\npathlib.Path('data/manifests').mkdir(parents=True, exist_ok=True)\nwith open('data/manifests/cbisddsm.csv', 'w', newline='', encoding='utf-8') as stream:\n    writer = csv.DictWriter(stream, fieldnames=['sample_key','patient_key','path','source','label','view','laterality','split','sha256'])\n    writer.writeheader()\n    writer.writerows(rows)"),
        ],
    ),
    (
        "01_auditoria",
        "01 · Auditoria de dados",
        "Sanidade: pacientes duplicados, vazamento entre splits, distribuição por vista/lateralidade.",
        [
            ("code", "from project_cnn_mammo.data import read_manifest, split_plan\nentries = read_manifest('data/manifests/cbisddsm.csv')\nplan = split_plan(entries)\nprint(plan.counts)\nassert plan.leakage_free, 'vazamento detectado'"),
            ("code", "from collections import Counter\nprint(Counter((e.view, e.laterality) for e in entries))\nprint(Counter(e.label for e in entries))"),
        ],
    ),
    (
        "02_preprocessamento",
        "02 · Pré-processamento",
        "DICOM → 8 bits, MONOCHROME1, recorte; ablação CLAHE × Z-score.",
        [
            ("code", "from project_cnn_mammo.io_dicom import read_dicom, apply_windowing\npayload = read_dicom('data/exemplo.dcm')\nprint({k: payload[k] for k in ('width','height','photometric_interpretation','monochrome1')})"),
            ("code", "from project_cnn_mammo.imaging import preprocess_array\npixels = apply_windowing(payload)\nresult = preprocess_array(pixels, payload['width'], payload['height'], normalization='clahe_zscore')\nprint(result['source'], result['tiles'])"),
            ("code", "z = preprocess_array(pixels, payload['width'], payload['height'], normalization='zscore')\nprint('ablacao registrada:', {'clahe': result['source'], 'zscore': z['source']})"),
            ("markdown", "## Etapa pré-CNN: aprendizado não supervisionado\n"),
            ("code", "from project_cnn_mammo.unsupervised import pre_cnn_flags\nfeatures = [[float(v)] for v in pixels[:64]]\nflags = pre_cnn_flags(features, method='knn', quantile=0.90)\nprint('limiar:', round(flags['threshold'], 3), ' anomalias:', flags['anomalies'])"),
        ],
    ),
    (
        "03_baseline",
        "03 · Baselines CNN e U-Net",
        "ResNet50 (ImageNet), U-Net do zero e U-Net+ResNet34 sob o mesmo protocolo.",
        [
            ("code", "from project_cnn_mammo.training import build_classifier, build_segmenter, set_determinism\nset_determinism(42)\nmodel = build_classifier('resnet50')\nprint(type(model).__name__)"),
            ("code", "segmenter = build_segmenter('unet_resnet34')\nprint(type(segmenter).__name__)"),
            ("code", "from project_cnn_mammo.evaluation import evaluate\nmetrics = evaluate(labels=[0,0,1,1], scores=[0.1,0.2,0.8,0.9], samples=200, seed=42)\nprint(metrics)"),
        ],
    ),
    (
        "04_benchmark",
        "04 · Benchmark de encoders",
        "Todas as CNNs do escopo no mesmo split, seed e hiperparâmetros.",
        [
            ("code", "from project_cnn_mammo.experiment import expand_grid\nbase = {'epochs': 5, 'learning_rate': 1e-4, 'batch_size': 16}\nspecs = expand_grid(base, {'architecture': ['resnet50','efficientnet_b0','efficientnet_b3','convnext_tiny']})\nprint(len(specs), 'execuções')\nprint([s.tag for s in specs])"),
        ],
    ),
    (
        "05_calibracao",
        "05 · Calibração",
        "Linear probing/fine-tuning e calibração isotrópica só na validação.",
        [
            ("code", "from project_cnn_mammo.sprint08 import calibration_metrics\nm = calibration_metrics(labels=[0,0,1,1], probabilities=[0.1,0.2,0.7,0.9])\nprint(m)"),
            ("code", "from project_cnn_mammo.evaluation import threshold_at_specificity\nprint('limiar:', threshold_at_specificity([0,0,1,1],[0.1,0.2,0.8,0.9]))"),
        ],
    ),
    (
        "06_desbalanceamento",
        "06 · Desbalanceamento e prevalência",
        "BCE/pos_weight/sampler; pareamento CC×MLO; correção de prevalência.",
        [
            ("code", "from project_cnn_mammo.sprint09 import correct_prevalence, rescue_false_negatives\nprint('prevalencia corrigida:', round(correct_prevalence(0.5, 0.02, 0.10), 4))\nprint(rescue_false_negatives(labels=[1,1,0], cnn_scores=[0.1,0.1,0.1], anomaly_scores=[0.8,0.7,0.9]))"),
            ("code", "from project_cnn_mammo.experiment import batch_ablation\nprint(batch_ablation())"),
        ],
    ),
    (
        "07_gate",
        "07 · Gate híbrido e robustez",
        "CNN + OC-SVM, incerteza e rotas de resgate/abstenção/confiança.",
        [
            ("code", "from project_cnn_mammo.pipeline import gate_decision, refine_decision\ndecision = gate_decision(0.85, 0.20, 0.90)\nprint(decision.route, decision.score)"),
            ("code", "curva = [(taxa, 1 - r) for taxa, r in [(0.0,1.0),(0.1,0.95),(0.2,0.91),(0.3,0.88)]]\nprint(curva)"),
        ],
    ),
    (
        "08_refinamento",
        "08 · TTA e refinamento",
        "TTA ponderado, Temperature Scaling e rotas refinadas.",
        [
            ("code", "from project_cnn_mammo.pipeline import refine_decision\nprint(refine_decision([0.7,0.72,0.68,0.71], [0.40,0.25,0.175,0.175]))"),
        ],
    ),
    (
        "09_exportacao",
        "09 · Exportação e release",
        "Artefato, model card, digest e registro champion/challenger.",
        [
            ("code", "from project_cnn_mammo.registry import ModelRegistry, artifact_digest\nfrom project_cnn_mammo.contracts import ModelCard\ncard = ModelCard(name='resnet50', version='0.1.0', architecture='resnet50', weights_sha256='0'*64, data_version='cbisddsm-v1', threshold=0.22, metrics={'auroc': 0.86})\nregistry = ModelRegistry('artifacts/registry')\nprint(registry.register(card, __import__('pathlib').Path('artifacts/weights.bin')))"),
            ("code", "from project_cnn_mammo.cli import main\nmain(['--config','configs/sprints.toml','smoke'])"),
        ],
    ),
]


def notebook(cells: list[tuple[str, str]]) -> dict[str, object]:
    return {
        "cells": [
            {
                "cell_type": kind,
                "metadata": {},
                "source": [source],
                **({"outputs": [], "execution_count": None} if kind == "code" else {}),
            }
            for kind, source in cells
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    written = []
    for filename, title, objective, cells in NOTEBOOKS:
        body = [("markdown", f"## {title}\n"), ("markdown", f"{objective}\n")]
        body.extend(cells)
        path = OUT / f"{filename}.ipynb"
        path.write_text(json.dumps(notebook(body), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        written.append(path.name)
    readme = [
        "# Notebooks — Fase 3 (ordem de execução)\n",
        "Abra no Colab ou Kaggle com GPU (T4/P100). O pacote é importado de `../src`.\n",
        "| Ordem | Caderno | Objetivo |",
        "|---|---|---|",
    ]
    for index, (filename, title, objective, _) in enumerate(NOTEBOOKS):
        readme.append(f"| {index} | [{filename}]({filename}.ipynb) | {objective} |")
    readme += [
        "",
        "## Ambientes",
        "- **Kaggle**: Accelerator > GPU T4 x2; anexe o dataset.",
        "- **Colab**: Ambiente > Alterar tipo > GPU; monte o Drive se necessário.",
        "- **Modal**: `modal run modal_app.py`.",
        "",
        "> Nunca processe o teste final junto ao treino: ele permanece isolado até o congelamento.",
    ]
    (OUT / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    print(f"{len(written)} notebooks em {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
