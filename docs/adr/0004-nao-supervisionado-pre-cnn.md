# ADR 0004 — Aprendizado não supervisionado antes da CNN

**Status:** aceito (2026-11-23)

## Contexto

O plano prevê que o modelo use sinais além da CNN e que anomalias/fora de distribuição sejam tratadas sem supervisioná-las. A exigência do projeto é aplicar **aprendizado não supervisionado antes da CNN**, como apoio — nunca como substituto da avaliação supervisionada.

## Decisão

Criar `unsupervised.py` com detectores one-class em stdlib:

- `GaussianOneClass` — distância de Mahalanobis com covariância regularizada invertida por Gauss-Jordan;
- `KnnOneClass` — distância média aos k vizinhos da referência;
- `empirical_threshold` — limiar por quantil dos scores de referência (sem suposição distribucional);
- `autoencoder_scores` — erro de reconstrução (torch, opcional);
- `pre_cnn_flags(...)` — etapa pré-CNN que devolve scores, limiar e índices anômalos.

Regras: o detector **não decide rótulo**; ele alimenta (a) ponderação de amostras raras, (b) resgate auxiliar de falsos negativos (`sprint09`) e (c) rota de revisão/abstenção (`sprint10`). O treino do detector usa apenas dados de treino/validação; o teste final não participa.

## Consequências

- Adiciona uma etapa ao pipeline sem custo de GPU na implementação stdlib.
- Scores e limiar são registrados no tracking para auditoria.
- Um autoencoder real exige o extra `ml`; sem ele, o detector permanece Gaussian/kNN.

## Alternativas descartadas

- Isolation Forest / LOF: exigem dependência pesada e ganho não comprovado frente ao protocolo atual.
- Usar o score não supervisionado como preditor principal: viola o escopo supervisionado da tarefa.
