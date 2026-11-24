"""Configuração via TOML (stdlib `tomllib`), com valores padrão auditáveis."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import tomllib

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "sprints.toml"


@dataclass(frozen=True)
class CommonConfig:
    seeds: tuple[int, ...] = (42, 123, 999)
    bootstrap_samples: int = 1000
    patient_split: bool = True
    final_test_locked: bool = True
    primary_metric: str = "sensitivity_at_90_specificity"


@dataclass(frozen=True)
class SprintConfig:
    """Seções nomeadas do `sprints.toml`; novas seções não quebram o carregamento."""

    common: CommonConfig = field(default_factory=CommonConfig)
    sections: dict[str, dict[str, object]] = field(default_factory=dict)

    def section(self, name: str, default: dict[str, object] | None = None) -> dict[str, object]:
        return self.sections.get(name, {} if default is None else default)

    @property
    def architectures(self) -> tuple[str, ...]:
        values = self.section("sprint07").get("architectures", ())
        return tuple(str(item) for item in values)


def load_config(path: Path | str | None = None) -> SprintConfig:
    """Carrega o TOML; falha com mensagem clara em vez de silently ignorar."""
    config_path = Path(path) if path else DEFAULT_CONFIG
    if not config_path.exists():
        raise FileNotFoundError(f"configuração não encontrada: {config_path}")
    raw = tomllib.loads(config_path.read_text(encoding="utf-8"))
    common_raw = raw.pop("common", {})
    seeds = common_raw.get("seeds", list(CommonConfig.seeds))
    if not isinstance(seeds, (list, tuple)):
        raise ValueError("common.seeds deve ser uma lista")
    common = CommonConfig(
        seeds=tuple(int(seed) for seed in seeds),
        bootstrap_samples=int(common_raw.get("bootstrap_samples", CommonConfig.bootstrap_samples)),
        patient_split=bool(common_raw.get("patient_split", CommonConfig.patient_split)),
        final_test_locked=bool(common_raw.get("final_test_locked", CommonConfig.final_test_locked)),
        primary_metric=str(common_raw.get("primary_metric", CommonConfig.primary_metric)),
    )
    return SprintConfig(common=common, sections=dict(raw))
