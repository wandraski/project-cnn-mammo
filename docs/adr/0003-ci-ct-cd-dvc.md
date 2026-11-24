# ADR 0003 — CI/CT/CD e DVC com remote gratuito

**Status:** aceito (2026-11-23)

## Contexto

O plano docente exige reprodutibilidade e versionamento de dados/artefatos. É preciso separar validação de código (CI), treino rastreável (CT) e publicação (CD), sem custo nem serviço externo obrigatório.

## Decisão

- **CI** (`ci.yml`): testes, compilação, validação do TOML, varredura de segredos e smoke do pipeline — sem dependências pesadas.
- **CT** (`ct.yml`): manual ou agendado; restaura dados via DVC, executa o pipeline, aplica `scripts/quality_gate.py` (mínimos normativos: AUROC ≥ 0,80 e sensibilidade ≥ 0,80 com especificidade 0,90) e registra champion/challenger no `registry.py`.
- **CD** (`cd.yml`): tag `v*` → build → `SHA256SUMS` → release; ambiente `release` exige aprovação humana.
- **DVC**: `dvc.yaml` declara as etapas; `.dvc/config` (não versionado) define o remote — Google Drive/rclone, DAGsHub ou Hugging Face, todos gratuitos.
- **Tracking**: MLflow quando configurado; caso contrário, JSON determinístico em `outputs/mlruns`.

## Consequências

- CT nunca toca o conjunto final de teste; ele permanece travado até o congelamento.
- O portão usa metas do plano como piso; reprovação impede promoção.
- Sem remote configurado, o CT roda com dados locais e registra isso no log.

## Alternativas descartadas

- Remote pago/obrigatório: viola a restrição de custo zero.
- Portão só por acurácia: proibido pelo plano.
