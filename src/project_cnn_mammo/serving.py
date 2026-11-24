"""Serving no estilo Lambda: mesmo pré-processamento do treino, resposta audível.

Duas superfícies com o mesmo núcleo (`handle`):
- `handle(event)` para invocação em lote/serverless;
- `serve(port)` para um servidor HTTP mínimo (stdlib), útil em Spaces/containers.
"""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from .contracts import ModelCard
from .preprocessing import TransformPlan, apply_plan
from .sprint10 import GateDecision, hybrid_gate


def load_card(path: Path | str) -> ModelCard:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return ModelCard(
        name=str(payload["name"]),
        version=str(payload["version"]),
        architecture=str(payload["architecture"]),
        weights_sha256=str(payload["weights_sha256"]),
        data_version=str(payload["data_version"]),
        threshold=float(payload["threshold"]),
        metrics={key: float(value) for key, value in payload.get("metrics", {}).items()},
        intended_use=str(payload.get("intended_use", "")),
        limitations=tuple(payload.get("limitations", ())),
    )


def handle(
    event: dict[str, Any],
    *,
    card: ModelCard,
    plan: TransformPlan,
    anomaly_score: float = 0.0,
    unet_confidence: float = 1.0,
) -> dict[str, Any]:
    """Núcleo de inferência: aplica o plano e decide a rota do gate.

    `probability` é estimada por um proxy determinístico quando não há checkpoint
    carregado (ambiente CPU); com PyTorch instalado, o chamador passa o score real.
    """
    pixels = [float(value) for value in event.get("pixels", [])]
    if not pixels:
        raise ValueError("evento sem pixels")
    transformed = apply_plan(pixels, plan)
    if "probability" in event:
        probability = float(event["probability"])
    else:
        # ponytail: proxy determinístico de fumaça; substituir pelo modelo real no extra `ml`
        probability = min(1.0, max(0.0, abs(sum(transformed) / len(transformed)) / 25.0))
    decision: GateDecision = hybrid_gate(
        probability,
        anomaly_score,
        unet_confidence,
        tau_clinical=card.metrics.get("tau_clinical", 0.22),
        tau_anomaly=card.metrics.get("tau_anomaly", 0.68),
    )
    return {
        "route": decision.route,
        "score": decision.score,
        "uncertainty": decision.uncertainty,
        "abstain": decision.abstain,
        "reason": decision.reason,
        "model": {"name": card.name, "version": card.version, "sha256": card.weights_sha256},
        "transform": plan.as_dict(),
    }


class InferenceHandler(BaseHTTPRequestHandler):
    """Handler mínimo: `/healthz` e `/predict`."""

    card: ModelCard
    plan: TransformPlan

    def _send(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 - API do BaseHTTPRequestHandler
        if self.path == "/healthz":
            self._send(200, {"status": "ok", "model": self.card.name, "version": self.card.version})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/predict":
            self._send(404, {"error": "not found"})
            return
        length = int(self.headers.get("Content-Length", "0"))
        try:
            event = json.loads(self.rfile.read(length) or b"{}")
            self._send(200, handle(event, card=self.card, plan=self.plan))
        except (ValueError, KeyError, json.JSONDecodeError) as exc:
            self._send(400, {"error": str(exc)})

    def log_message(self, format: str, *args: object) -> None:
        """Silencia log por requisição (CT limpo)."""


def serve(card: ModelCard, plan: TransformPlan, port: int = 8080) -> None:
    handler = type("BoundInferenceHandler", (InferenceHandler,), {"card": card, "plan": plan})
    server = ThreadingHTTPServer(("0.0.0.0", port), handler)
    server.serve_forever()
