# ADR 0002 — Pacote modular, POO por responsabilidade e stdlib-first

**Status:** aceito (2026-11-23)

## Contexto

Os primeiros módulos (`sprint07..11`) são contratos testáveis de regras de negócio. A Fase 2 pede orientação a objetos, simplicidade de código e fronteiras claras para crescer sem virar script gigante.

## Decisão

Organizar o pacote por responsabilidade, cada módulo com um contrato (`contracts.py`):

| Módulo | Responsabilidade |
|---|---|
| `contracts.py` | tipos e dataclasses do pipeline |
| `config.py` | leitura do `sprints.toml` |
| `data.py` | manifesto, hash, split por paciente |
| `preprocessing.py` | `TransformPlan` e transforms |
| `models.py` | registry de CNNs/U-Net do escopo |
| `evaluation.py` | AUROC/AUPRC, pontos clínicos, bootstrap |
| `tracking.py` | MLflow opcional / JSON determinístico |
| `registry.py` | model card, digest, champion/challenger |
| `serving.py` | inferência (batch + HTTP) |
| `pipeline.py` | orquestração e adapters de gate/refinement |
| `cli.py` | ponto de entrada |

Regras: stdlib-first (sem numpy para o núcleo), uma responsabilidade por módulo, sem abstração de implementação única, configuração só para o que varia.

## Consequências

- CI roda sem instalar PyTorch.
- `sprint07..11` permanecem como especificação de referência e são reutilizados (não reimplementados).
- Módulos novos exigem teste de contrato correspondente.

## Alternativas descartadas

- Framework de pipeline (Airflow/Prefect): custo operacional desproporcional.
- Tudo em `pipeline.py`: volta ao script gigante.
