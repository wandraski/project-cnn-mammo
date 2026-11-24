# Changelog

Formato baseado em [Keep a Changelog](https://keepachangelog.com/).

## [0.2.0] — 2026-11-23

### Adicionado
- Núcleo modular da Fase 2: `contracts`, `config`, `data`, `preprocessing`, `models`, `evaluation`, `tracking`, `registry`, `serving`, `pipeline` e `cli`.
- Métricas stdlib: AUROC/AUPRC por ranking, pontos clínicos e IC 95% por bootstrap.
- Registry local de modelos com champion/challenger, digest dos pesos e rollback.
- Servidor HTTP mínimo de inferência com `/healthz` e `/predict`.
- Workflows de CI, CT e CD; varredura de segredos e portão de qualidade.
- `dvc.yaml` com ingest/preprocess/evaluate/export.
- Dockerfile multi-stage e Makefile.
- ADRs 0001–0003.

### Alterado
- `ci.yml` agora valida configuração, rodeia segredos e executa o smoke do pipeline.

## [0.1.0] — 2026-11-21

### Adicionado
- Contratos das Sprints 7–11 (splits, calibração, prevalência, gate, refinamento).
- Configuração `configs/sprints.toml` e workflow de CI inicial.
