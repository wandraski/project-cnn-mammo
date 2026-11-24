# ADR 0001 — Arquitetura Lambda com núcleo de inferência compartilhado

**Status:** aceito (2026-11-23)

## Contexto

O projeto precisa de treino/avaliação em lote e de um caminho de serving que produza exatamente as mesmas decisões para a mesma imagem. Divergências de pré-processamento entre treino e serving (*training-serving skew*) invalidam a avaliação clínica.

## Decisão

Adotar arquitetura **Lambda** com um único núcleo:

- **Camada batch:** `pipeline.Pipeline` executa ingest → validate → preprocess → evaluate, com tracker por run.
- **Camada serving:** `serving.handle(event)` aplica o mesmo `TransformPlan` e o mesmo gate híbrido (`sprint10.hybrid_gate`).
- **`TransformPlan`** é o contrato imutável que liga as duas camadas; ele é registrado no model card e em todo log.

## Consequências

- Qualquer mudança de pré-processamento exige novo `TransformPlan` e nova versão de modelo.
- O proxy de probabilidade no serving é explicitamente temporário (extra `ml` substitui pelo checkpoint).
- Serve HTTP em container e `handle()` em lote compartilham `preprocessing.apply_plan`.

## Alternativas descartadas

- FastAPI dedicado: dependência extra sem ganho nesta fase; o handler stdlib cobre `/healthz` e `/predict`.
- Pipeline separado para API: risco de divergência de pré-processamento.
