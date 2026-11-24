"""Pontos de execução compartilhados pelos cadernos S7–S11."""
from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path

from .config import load_config
from .experiment import batch_ablation
from .sprint10 import hybrid_gate
from .sprint11 import aggregate_tta
from .training_run import train_on_directory


def _validation_csv(path: Path, required: set[str]) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"{path.name} ausente: forneça predições reais de validação na pasta de input")
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    required |= {"split", "sample_key"}
    if not rows or any(any(not (row.get(k) or "").strip() for k in required) for row in rows):
        raise ValueError(f"{path.name} precisa das colunas preenchidas: {', '.join(sorted(required))}")
    if any(row["split"] != "val" for row in rows):
        raise ValueError(f"{path.name}: apenas predições de validação; teste final isolado")
    if len({row["sample_key"] for row in rows}) != len(rows):
        raise ValueError(f"{path.name}: sample_key duplicada")
    return rows


def score_gate(path: Path) -> dict[str, object]:
    rows = _validation_csv(path, {"p_cnn", "s_anom", "unet_conf"})
    s10 = load_config().section("sprint10")
    decisions = [hybrid_gate(
        float(r["p_cnn"]), float(r["s_anom"]), float(r["unet_conf"]),
        tau_clinical=float(str(s10["tau_clinical"])), tau_anomaly=float(str(s10["tau_anomaly"])),
        alpha=float(str(s10["alpha"])), band=float(str(s10["abstention_band"])),
    ) for r in rows]
    counts = Counter(decision.route for decision in decisions)
    return {"n": len(rows), "routes": dict(counts),
            "automatic_coverage": counts["AUTO_CONFIDENT"] / len(rows)}


def calibrate_tta(path: Path) -> dict[str, object]:
    """S11 parcial: ajusta temperatura e agrega quatro vistas somente em validação."""
    columns = ("logit_original", "logit_hflip", "logit_plus5", "logit_minus5")
    rows = _validation_csv(path, {"label", *columns})
    logits = [[float(row[column]) for column in columns] for row in rows]
    labels = [int(row["label"]) for row in rows]
    if set(labels) != {0, 1} or any(not math.isfinite(x) for views in logits for x in views):
        raise ValueError("são necessários logits finitos e rótulos 0/1 das duas classes")

    def nll(temperature: float) -> float:
        losses = []
        for views, label in zip(logits, labels, strict=True):
            x = (1 - 2 * label) * views[0] / temperature
            losses.append(max(x, 0) + math.log1p(math.exp(-abs(x))))
        return sum(losses) / len(losses)

    # ponytail: grade logarítmica finita; só refine se a calibração exigir.
    temperature = min((2 ** (i / 20) for i in range(-60, 61)), key=nll)
    weights = load_config().section("sprint11")["tta_weights"]
    if not isinstance(weights, list):
        raise ValueError("sprint11.tta_weights precisa de quatro pesos")
    scores = [aggregate_tta([1 / (1 + math.exp(-max(-700, min(700, x / temperature))))
                            for x in views], weights) for views in logits]
    return {"n": len(rows), "temperature": temperature,
            "original_nll_before": nll(1.0), "original_nll_after_fit": nll(temperature),
            "mean_tta_uncertainty": sum(std for _, std in scores) / len(scores),
            "split": "val (ajuste; não é avaliação externa)"}


def run_sprint(sprint: int, data: Path, out: Path, *, smoke: bool = True,
               device: str = "cuda", arm: str = "batch-16-fixed",
               manifest_path: Path | None = None) -> dict[str, object]:
    if sprint == 10:
        result = score_gate(data / "sprint10_val.csv")
    elif sprint == 11:
        result = calibrate_tta(data / "sprint11_val.csv")
    else:
        config = load_config()
        params: dict[str, object] = {}
        if sprint == 8:
            s8 = config.section("sprint08")
            params = {name: float(str(s8[f"linear_probe_{name}"])) if name == "learning_rate"
                      else int(str(s8[f"linear_probe_{name}"]))
                      for name in ("epochs", "batch_size", "resolution", "learning_rate")}
        elif sprint == 9:
            spec = next((spec for spec in batch_ablation() if spec.tag == arm), None)
            if spec is None:
                raise ValueError("braço S9 desconhecido; use batch-{8,16,32,64,128}-{fixed,linear}")
            params = {name: spec.params[name] for name in ("epochs", "batch_size", "resolution", "learning_rate")}
        elif sprint != 7:
            raise ValueError(f"Sprint inválida: {sprint}")
        result = train_on_directory(
            data, output_dir=out, device=device, linear_probe=sprint == 8,
            pretrained=not smoke, manifest_path=manifest_path,
            epochs=1 if smoke else int(str(params['epochs'])) if params else None,
            batch_size=4 if smoke else int(str(params['batch_size'])) if params else None,
            resolution=int(str(params['resolution'])) if params else None,
            learning_rate=float(str(params['learning_rate'])) if params else None,
            limit_per_class=25 if smoke else None,
        )
        result.pop("validation_predictions")
        result.pop("history")
    result.update(sprint=sprint, mode=("smoke — não comparar aos slides" if smoke else "experimento"))
    if sprint == 9:
        result["arm"] = arm
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result
