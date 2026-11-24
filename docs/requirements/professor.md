# Requisitos iniciais — orientação do professor

Fontes: `projeto_integrador_cancer_mama.pdf` (plano geral) e `arquivos/projeto-final-docs/projeto-integrador-grupo-01.pdf` (plano somente do Grupo 1, subconjunto). Este documento resume os PDFs; em divergência, os PDFs originais prevalecem. Consolidação das orientações definitivas: [`orientacoes-definitivas.md`](orientacoes-definitivas.md).

## Escopo do Grupo 1

- Implementar somente a linha de **CNNs e pipeline base**.
- Comparar ResNet50, EfficientNet-B0/B3 e ConvNeXt-Tiny sob o mesmo protocolo.
- Comparar pré-treino ImageNet com pré-treino médico quando disponível.
- Manter pipeline reprodutível de pré-processamento e avaliação.

## Tarefas comuns obrigatórias

1. Classificação binária maligno × não maligno em patches do CBIS-DDSM.
2. Classificação por imagem inteira ou por mama combinando CC e MLO.
3. Segmentação com U-Net usando máscaras do CBIS-DDSM.
4. Comparação entre pipeline direto e pipeline U-Net → classificador CNN.

## Dados

- Split por paciente; nunca misturar imagens do mesmo paciente entre treino, validação e teste.
- Respeitar o split oficial do CBIS-DDSM.
- CBIS-DDSM como base de referência.
- VinDr-Mammo e CMMD para validação externa sem retreino.
- RSNA como prova de realidade, com prevalência próxima do rastreamento.
- Bases pequenas somente para prototipagem e depuração.

## Métricas obrigatórias

- AUROC e AUPRC;
- sensibilidade fixando especificidade em 90%;
- especificidade fixando sensibilidade em 90%;
- F1 e matriz de confusão no ponto escolhido;
- IC 95% por bootstrap com no mínimo 1.000 reamostragens;
- Dice, IoU, sensibilidade e precisão por pixel para segmentação;
- análise de Dice por tamanho da lesão.

Acurácia não pode ser a métrica principal.

## Baselines

- ResNet50 pré-treinada em ImageNet para classificação.
- U-Net do zero e U-Net com encoder ResNet34 para segmentação.

## Ablações obrigatórias

- Batch sizes 8, 16, 32 e 64; 128 somente se couber.
- Comparar learning rate fixo com regra linear proporcional ao batch.
- Registrar AUROC com IC 95%, sensibilidade @ 90% de especificidade, tempo por época e VRAM.
- Avaliar resolução 224, 384, 512 e 1024 quando viável.
- Comparar CLAHE e normalização Z-score.

## Literatura

Revisar ao menos cinco artigos de periódicos Q1 publicados desde 2020. Registrar DOI, autores, periódico/ano, quartil SJR, método, bases, AUROC/recall e relevância.

## Reprodutibilidade

- GitHub com README, dependências e instruções de reprodução.
- Kaggle como execução GPU padrão; Colab como alternativa.
- MLflow ou Weights & Biases para rastreamento.
- Sementes fixas.
- Configuração e avaliação comuns entre os modelos.

## Metas quantitativas do professor

Metas são objetivos, não resultados:

- AUROC ≥ 0,80 em patches do CBIS-DDSM;
- sensibilidade ≥ 0,80 com especificidade 0,90;
- Dice ≥ 0,60 na U-Net;
- validação externa em VinDr e CMMD;
- quatro batch sizes;
- cinco artigos Q1;
- IC 95% em todas as métricas.

## Avaliação

- reprodutibilidade: 20%;
- rigor metodológico: 30%;
- profundidade técnica: 20%;
- análise crítica: 20%;
- comunicação: 10%.

Resultados negativos bem fundamentados são válidos. Nenhum resultado deve ser maquiado.

## Limites clínicos e éticos

- Não usar para diagnóstico.
- Não versionar imagens de pacientes em repositório público ou privado.
- Respeitar LGPD, termos das bases e proibição de reidentificação.
- Uso clínico exigiria validação prospectiva, regulação e monitoramento fora do escopo.
