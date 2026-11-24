"""Tracking de experimentos: MLflow quando disponível, JSON sempre.

A URI e o commit são gravados em todo run; nada de telemetria escondida.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

from .contracts import RunContext


@dataclass
class _RunRecord:
    run: str
    started_at: str
    status: str = "RUNNING"
    params: dict[str, object] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    artifacts: list[str] = field(default_factory=list)


class Tracker(Protocol):
    def start_run(self, name: str) -> None: ...
    def log_params(self, values: dict[str, object]) -> None: ...
    def log_metrics(self, values: dict[str, float]) -> None: ...
    def log_artifact(self, path: Path) -> None: ...
    def end_run(self, status: str = "FINISHED") -> None: ...
    @property
    def uri(self) -> str: ...


class JsonTracker:
    """Tracker determinístico em disco: cada run é um arquivo JSON versionável."""

    def __init__(self, root: Path | str = "mlruns"):
        self.root = Path(root)
        self.current: _RunRecord | None = None
        self._path: Path | None = None

    @property
    def uri(self) -> str:
        return f"json://{self.root.resolve()}"

    def start_run(self, name: str) -> None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self._path = self.root / f"{name}-{stamp}.json"
        self.current = _RunRecord(run=name, started_at=stamp)

    def log_params(self, values: dict[str, object]) -> None:
        assert self.current is not None
        self.current.params.update(values)

    def log_metrics(self, values: dict[str, float]) -> None:
        assert self.current is not None
        self.current.metrics.update(values)

    def log_artifact(self, path: Path) -> None:
        assert self.current is not None
        self.current.artifacts.append(str(Path(path)))

    def end_run(self, status: str = "FINISHED") -> None:
        assert self.current is not None and self._path is not None
        self.current.status = status
        payload = {
            "run": self.current.run,
            "started_at": self.current.started_at,
            "status": self.current.status,
            "params": self.current.params,
            "metrics": self.current.metrics,
            "artifacts": self.current.artifacts,
        }
        self.root.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


class MlflowTracker:
    """Adapter enxuto para MLflow; falha com orientação clara se faltar configuração."""

    def __init__(self, uri: str | None = None):
        try:
            import mlflow  # type: ignore
        except ImportError as exc:  # pragma: no cover - depende do ambiente
            raise RuntimeError("instale o extra `ml` ou use JsonTracker") from exc
        self._mlflow = mlflow
        self.uri = uri or "mlruns"
        mlflow.set_tracking_uri(self.uri)

    def start_run(self, name: str) -> None:
        self._mlflow.start_run(run_name=name)

    def log_params(self, values: dict[str, object]) -> None:
        for key, value in values.items():
            self._mlflow.log_param(key, value)

    def log_metrics(self, values: dict[str, float]) -> None:
        self._mlflow.log_metrics(values)

    def log_artifact(self, path: Path) -> None:
        self._mlflow.log_artifact(str(path))

    def end_run(self, status: str = "FINISHED") -> None:
        self._mlflow.end_run(status=status)


def make_tracker(context: RunContext, prefer_mlflow: bool = False) -> Tracker:
    """Escolhe tracker: MLflow se pedida e configurada; senão JSON local."""
    if prefer_mlflow:
        try:
            return MlflowTracker()
        except RuntimeError:
            pass
    return JsonTracker(context.output_dir / "mlruns")


def context_as_params(context: RunContext) -> dict[str, object]:
    return context.as_dict()
