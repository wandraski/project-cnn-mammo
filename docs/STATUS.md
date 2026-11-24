# Estado verificável do projeto

| Área | Estado | Evidência |
|---|---|---|
| Repositório novo | implementado | histórico Git deste repositório |
| Requisitos do PDF principal | extraído | `docs/requirements/professor.md` |
| Requisitos adjacentes (ES + MLOps) | extraídos e canônicos | `docs/requirements/professores-adjacentes.md` |
| Orientações definitivas dos professores | consolidadas | `docs/requirements/orientacoes-definitivas.md` |
| Plano da Sprint 7 | extraído | `docs/requirements/sprint-07.md` |
| Código de treino CNN/U-Net | planejado; dependências declaradas | `pyproject.toml`, `docs/IMPLEMENTATION.md` |
| Contrato de dados/leakage | implementado | `src/project_cnn_mammo/sprint07.py` |
| Limiar clínico na validação | implementado | `src/project_cnn_mammo/sprint07.py` |
| Métricas de calibração | implementado | `src/project_cnn_mammo/sprint08.py` |
| Prevalência e resgate de FNs | implementado | `src/project_cnn_mammo/sprint09.py` |
| Gate híbrido | implementado | `src/project_cnn_mammo/sprint10.py` |
| TTA e decisão refinada | implementado | `src/project_cnn_mammo/sprint11.py` |
| Núcleo modular (Fase 2) | implementado | `contracts/config/data/preprocessing/models/evaluation/tracking/registry/serving/pipeline/cli` |
| Métricas stdlib com IC bootstrap | implementado | `evaluation.py`, `tests/test_core.py` |
| CI/CT/CD e varredura de segredos | implementado | `.github/workflows/`, `scripts/` |
| DVC dag e tracking | implementado | `dvc.yaml`, `tracking.py` |
| Servidor de inferência (Lambda) | implementado | `serving.py`, `Dockerfile` |
| Champion/challenger e rollback | implementado | `registry.py` |
| Documentação viva (ADRs, changelog, commits) | implementado | `docs/adr/`, `CHANGELOG.md`, `docs/COMMITS.md` |
| Notebooks da Fase 3 (00–09) | implementado | `notebooks/`, gerados por `scripts/make_notebooks.py` |
| CLAHE/janelamento/DICOM e datasets | implementado | `imaging.py`, `io_dicom.py`, `datasets.py` |
| Treino PyTorch (CNN/U-Net) | implementado no pacote; não executado | `training.py`, extra `ml` |
| Matriz experimental e Pareto | implementado | `experiment.py` |
| Aprendizado não supervisionado pré-CNN | implementado | `unsupervised.py` (Gaussian/kNN one-class + autoencoder opcional) |
| Entrada Modal (GPU) | implementado | `modal_app.py` |
| Dataset de desenvolvimento (`subset-multibase-cancer-mama`) | registrado; estrutura a validar no Kaggle | `datasets.py` |
| Execução em GPU/datasets reais | congelada (decisão do usuário, 2026-11-23) | — |
| Parâmetros dos slides S7–S11 | reconciliados por experimento; conferir lacunas e diferenças de fonte | `configs/sprints.toml`, `docs/IMPLEMENTATION.md` |
| Entrada do benchmark CNN | exige manifesto de desenvolvimento com split por paciente; não validada em GPU | `notebooks/03`, `04`, `training_run.py` |
| Experimentos e métricas reais | não iniciados | — |
| Resultados dos slides | metas, não medidos | fontes externas ao repositório |
| Requisitos dos demais professores | pendentes | `docs/requirements/slide-intake.md` |

## Vocabulário obrigatório

- **meta:** resultado ou capacidade desejada;
- **simulação:** cenário ou número apresentado sem execução verificável neste repositório;
- **implementado:** código/configuração presente e testável;
- **medido:** resultado produzido por execução rastreável neste repositório.
