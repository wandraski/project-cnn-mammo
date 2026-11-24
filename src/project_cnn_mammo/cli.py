"""CLI do pipeline: ponto de entrada único para treino, avaliação e serving."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .config import load_config
from .contracts import RunContext
from .data import read_manifest, sha256_file, split_plan
from .evaluation import evaluate
from .pipeline import Pipeline, ingest, preprocess, validate
from .preprocessing import TransformPlan, describe
from .registry import ModelRegistry
from .serving import load_card, serve


def _manifest(args: argparse.Namespace) -> int:
    entries = read_manifest(Path(args.manifest))
    plan = split_plan(entries)
    print(json.dumps({"samples": len(entries), "counts": plan.counts}, indent=2))
    return 0


def _smoke(args: argparse.Namespace) -> int:
    """Pipeline completo em dados sintéticos, sem GPU nem dataset real."""
    pipeline = Pipeline(
        [ingest([]), validate(), preprocess(TransformPlan())],
        config_path=args.config,
        dry_run=True,
    )
    results = pipeline.run()
    print(json.dumps([{"stage": r.name, "status": r.status, **r.detail} for r in results], indent=2))
    return 0


def _evaluate(args: argparse.Namespace) -> int:
    labels = [int(value) for value in args.labels.split(",")]
    scores = [float(value) for value in args.scores.split(",")]
    metrics = evaluate(labels, scores, samples=args.bootstrap, seed=args.seed)
    print(metrics)
    return 0


def _preprocess(args: argparse.Namespace) -> int:
    payload = json.loads(Path(args.image).read_text(encoding="utf-8"))
    pixels = [float(value) for value in payload["pixels"]]
    plan = TransformPlan(
        resolution=int(payload.get("resolution", 512)),
        normalization=payload.get("normalization", "clahe_zscore"),
    )
    print(json.dumps(describe(pixels, plan), indent=2))
    return 0


def _digest(args: argparse.Namespace) -> int:
    print(sha256_file(Path(args.path)))
    return 0


def _register(args: argparse.Namespace) -> int:
    registry = ModelRegistry(args.root)
    card = load_card(args.card)
    alias = registry.register(card, Path(args.weights), challenger=args.challenger)
    print(alias)
    return 0


def _serve(args: argparse.Namespace) -> int:
    card = load_card(args.card)
    plan = TransformPlan(
        resolution=int(args.resolution),
        normalization=args.normalization,
    )
    serve(card, plan, port=args.port)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cnn-mammo")
    parser.add_argument("--config", default=None, help="caminho do sprints.toml")
    sub = parser.add_subparsers(dest="command", required=True)

    manifest = sub.add_parser("manifest", help="resume manifesto e valida splits")
    manifest.add_argument("--manifest", required=True)
    manifest.set_defaults(func=_manifest)

    smoke = sub.add_parser("smoke", help="pipeline de fumaça sem dados reais")
    smoke.set_defaults(func=_smoke)

    evaluate_parser = sub.add_parser("evaluate", help="métricas com IC bootstrap")
    evaluate_parser.add_argument("--labels", required=True)
    evaluate_parser.add_argument("--scores", required=True)
    evaluate_parser.add_argument("--bootstrap", type=int, default=1000)
    evaluate_parser.add_argument("--seed", type=int, default=42)
    evaluate_parser.set_defaults(func=_evaluate)

    preprocess = sub.add_parser("preprocess", help="aplica o plano a um JSON de pixels")
    preprocess.add_argument("--image", required=True)
    preprocess.set_defaults(func=_preprocess)

    digest = sub.add_parser("digest", help="SHA-256 de um arquivo")
    digest.add_argument("--path", required=True)
    digest.set_defaults(func=_digest)

    register = sub.add_parser("register", help="registra modelo no registry local")
    register.add_argument("--root", default="artifacts/registry")
    register.add_argument("--card", required=True)
    register.add_argument("--weights", required=True)
    register.add_argument("--challenger", action="store_true")
    register.set_defaults(func=_register)

    serve_parser = sub.add_parser("serve", help="sobe o servidor de inferência")
    serve_parser.add_argument("--card", required=True)
    serve_parser.add_argument("--port", type=int, default=8080)
    serve_parser.add_argument("--resolution", type=int, default=512)
    serve_parser.add_argument("--normalization", default="clahe_zscore")
    serve_parser.set_defaults(func=_serve)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (ValueError, FileNotFoundError, KeyError) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    sys.exit(main())
