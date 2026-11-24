"""Registro local de modelos: card, digest dos pesos e promoção champion/challenger.

O registro é um diretório versionável com `registry.json`; nada de serviço externo
obrigatório para rodar o protocolo.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from .contracts import ModelCard
from .data import sha256_file


def artifact_digest(path: Path) -> str:
    """Digest do artefato de pesos — é ele que identifica o modelo."""
    return sha256_file(Path(path))


class ModelRegistry:
    """Registro em arquivo único; promoção é explícita e auditável."""

    def __init__(self, root: Path | str):
        self.root = Path(root)
        self.index_path = self.root / "registry.json"

    def _load(self) -> dict[str, object]:
        if not self.index_path.exists():
            return {"models": {}, "champion": None}
        return json.loads(self.index_path.read_text(encoding="utf-8"))

    def _save(self, index: dict[str, object]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def register(
        self,
        card: ModelCard,
        weights: Path,
        *,
        challenger: bool = False,
    ) -> str:
        """Registra ou promove um modelo; retorna o alias (`champion`/`challenger`)."""
        index = self._load()
        models = index.setdefault("models", {})
        assert isinstance(models, dict)
        digest = artifact_digest(weights)
        record = {
            **card.__dict__,
            "limitations": list(card.limitations),
            "metrics": dict(card.metrics),
            "weights_sha256": digest,
            "registered_at": datetime.now(timezone.utc).isoformat(),
            "alias": "challenger" if challenger else "candidate",
        }
        previous_champion = index.get("champion")
        if not challenger:
            if isinstance(previous_champion, dict):
                previous_champion["alias"] = "archived"
            index["champion"] = record
            alias = "champion"
        else:
            index["challenger"] = record
            alias = "challenger"
        models[f"{card.name}:{card.version}"] = record
        self._save(index)
        return alias

    def promote(self, name: str, version: str) -> dict[str, object]:
        """Promove challenger a champion (porta do CD, com aprovação humana à parte)."""
        index = self._load()
        models = index.get("models", {})
        key = f"{name}:{version}"
        if not isinstance(models, dict) or key not in models:
            raise KeyError(f"modelo não registrado: {key}")
        assert isinstance(index.get("champion"), dict) or index.get("champion") is None
        current = index.get("champion")
        if isinstance(current, dict):
            current["alias"] = "archived"
        index["champion"] = models[key]
        index["champion"]["alias"] = "champion"
        if index.get("challenger") == models[key]:
            index["challenger"] = None
        self._save(index)
        return index["champion"]

    def rollback(self) -> dict[str, object] | None:
        """Volta o champion para o último arquivado (rollback explícito)."""
        index = self._load()
        models = index.get("models", {})
        if not isinstance(models, dict):
            return None
        archived = [record for record in models.values() if isinstance(record, dict) and record.get("alias") == "archived"]
        if not archived:
            return None
        latest = max(archived, key=lambda record: str(record.get("registered_at", "")))
        current = index.get("champion")
        if isinstance(current, dict):
            current["alias"] = "archived"
        latest["alias"] = "champion"
        index["champion"] = latest
        self._save(index)
        return latest

    def champion(self) -> dict[str, object] | None:
        current = self._load().get("champion")
        return current if isinstance(current, dict) else None
