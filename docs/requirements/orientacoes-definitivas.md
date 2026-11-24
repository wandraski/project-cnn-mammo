# Orientações definitivas dos professores

**Autoria exclusiva do projeto:** Marco Antonio Wandraski (`@wandraski`).
**Fontes normativas:**
- `arquivos/github-organização-antiga/documentos-governanca/materiais/professor/projeto_integrador_cancer_mama.pdf` (plano geral, 13 p.)
- `arquivos/projeto-final-docs/projeto-integrador-grupo-01.pdf` (plano somente do Grupo 1, 11 p.; subconjunto do anterior)

Em divergência entre fontes, **prevalece a orientação definitiva do professor**; o PDF do Grupo 1 é a versão de escopo do nosso grupo. Este documento consolida `CONTE-9`.

## 1. Escopo definitivo — CNNs e pipeline base

- Implementar **somente a linha de CNNs e pipeline base** (Grupo 1).
- Arquiteturas: **ResNet50, EfficientNet-B0/B3 e ConvNeXt-Tiny**.
- Comparar pré-treino **ImageNet** com pré-treino médico (**RadImageNet**, quando disponível).
- O Grupo 1 é o **dono do pipeline de pré-processamento compartilhado** e da implementação de referência do `evaluate.py`.
- Baselines comuns obrigatórios: **ResNet50 (ImageNet)** para classificação e **U-Net do zero + U-Net com encoder ResNet34** para segmentação.

## 2. Tarefas e protocolo

1. Classificação binária maligno × não maligno em **patches** do CBIS-DDSM (tarefa primária).
2. Classificação por **imagem inteira / mama** combinando CC e MLO (tarefa secundária).
3. **Segmentação** com U-Net usando máscaras pixel a pixel.
4. Comparação **pipeline direto × pipeline em duas fases** (U-Net → classificador).

- Split **por paciente**; respeitar o split oficial do CBIS-DDSM.
- Ablação obrigatória de **batch size**: 8, 16, 32 e 64 (128 se couber), com e sem regra do **LR linear**.
- Resoluções: **224, 384, 512 e 1024** quando viável.
- CLAHE × normalização Z-score.
- Cinco artigos **Q1 (2020+)**, com DOI, periódico/ano, quartil SJR confirmado, método, bases e métricas reportadas.

## 3. Métricas obrigatórias (com IC 95% por bootstrap ≥ 1.000 reamostragens)

AUROC, AUPRC, sensibilidade fixando especificidade em 90%, especificidade fixando sensibilidade em 90%, F1, matriz de confusão. Segmentação: Dice, IoU, sensibilidade e precisão por pixel, além de Dice por tamanho de lesão. **Acurácia não pode ser a métrica principal.**

## 4. Bases de dados (seção 3 do plano)

| Base | Papel | Acesso |
|---|---|---|
| CBIS-DDSM | referência, treino/validação, split oficial | direto (NBIA) |
| VinDr-Mammo | validação externa 1 | PhysioNet + CITI + acordo |
| CMMD | validação externa 2 | direto (NBIA) |
| RSNA (Kaggle 2022) | "prova de realidade", prevalência ~2% | conta Kaggle |
| MIAS, DMID, BMCD, BCDR, CDDCESM, BCSDBT, EMBED, BreakHis | prototipagem/estudos específicos | variado |

Observação normativa: **INbreast não deve ser usado** (sem link oficial; espelhos de procedência incerta). CMMD o substitui.

## 5. Metas quantitativas — classificação: `meta` (não resultado)

**Mínimas:** AUROC ≥ 0,80 em patches; sensibilidade ≥ 0,80 com especificidade 0,90; Dice ≥ 0,60; validação externa em VinDr/CMMD sem retreino, com queda discutida; ≥ 4 batch sizes; 5 artigos Q1; IC 95% em todas as métricas.

**Excelência:** AUROC ≥ 0,90 em patches; sensibilidade ≥ 0,90 com especificidade ≥ 0,80 em imagem inteira; Dice ≥ 0,75; queda < 10 pontos de AUROC na validação externa; ablação de batch com discussão de BatchNorm/LR linear.

## 6. Pesos da avaliação

Reprodutibilidade do repositório **20%** · rigor metodológico **30%** · profundidade técnica **20%** · análise crítica **20%** · comunicação **10%**. Resultado negativo bem fundamentado vale nota máxima.

## 7. Cronograma dos 20 encontros → fases deste roadmap

| Bloco | Encontros | Fase do roadmap |
|---|---|---|
| Fundamentos | 1–3 | 1 (contextualização, EDA, splits, literatura) |
| Baseline | 4–7 | 2 e 3 (pipeline v1, baselines, evaluate.py) |
| Desenvolvimento | 8–13 | 3 (ablações, balanceamento, migração para imagem inteira) |
| Experimentos | 14–17 | 4 (validação externa, pipeline em duas fases, XAI, tabela comparativa) |
| Consolidação | 18–20 | 5 (redação, ensaio, seminário) |

## 8. Limites clínicos, éticos e legais

- Nenhum modelo deste projeto é dispositivo médico; **não usar em decisão clínica**.
- Não versionar imagens de pacientes, mesmo em notebooks demonstrativos.
- Respeitar LGPD, termos das bases e proibição de reidentificação.
- Uso clínico exigiria ANVISA (RDC 657/2022), validação prospectiva multicêntrica e monitoramento pós-mercado.

## 9. Ordem de autoridade e matriz de conflitos

**Ordem:** (1) orientação definitiva do professor (PDFs acima); (2) requisitos canônicos adjacentes (`ES-CAN`, `MLOPS-CAN`); (3) slides/labs como metas; (4) exemplos didáticos.

| Conflito | Decisão |
|---|---|
| Escopo de outro grupo (Transformers/híbridos/XAI) | fora do escopo; reference-only |
| Slides antigos do time vs. restrição individual | slides são `meta`; nada de coautoria ou histórico importado |
| Valores de slide vs. execução real | só conta o que este repositório medir |
| Acurácia como métrica principal | proibida pelo plano |
| Uso do INbreast | proibido; usar CMMD |
| Prazo acadêmico vs. granularidade técnica | prazo externo registrado à parte; fases sem prazo artificial |

## 10. Pendências

- Requisitos das demais disciplinas, quando enviados, entram no mesmo formato canônico.
- Formato final de entrega (artigo/relatório/slides) segue o plano: IMRaD + apresentação no seminário.
