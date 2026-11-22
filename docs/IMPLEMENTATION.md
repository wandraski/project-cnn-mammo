# Implementação por Sprint

Todos os valores dos slides são metas a reproduzir. Somente resultados gerados por este repositório podem ser classificados como medidos.

## Sprint 7 — baseline e contrato de dados

**Implementado agora**

- lista canônica das CNNs do escopo;
- bloqueio de leakage de paciente entre `train`, `val` e `test`;
- seleção de limiar na validação para especificidade-alvo;
- configuração dos baselines CNN/U-Net e métricas normativas.

**Próxima vertical**

- loader DICOM e manifesto;
- treino ResNet50 e U-Net com PyTorch;
- AUROC/AUPRC, bootstrap por paciente e Grad-CAM;
- comparação direta versus U-Net → CNN.

## Sprint 8 — pré-treino, resolução e calibração

**Implementado agora**

- Brier Score, ECE e MCE sem dependências externas;
- configuração ImageNet/RadImageNet, resoluções, batches e calibração isotônica.

**Próxima vertical**

- linear probing e fine-tuning;
- ablação CLAHE × Z-score;
- calibração isotônica treinada somente na validação;
- preflight de fontes externas e OC-SVM por fonte.

## Sprint 9 — desbalanceamento e prevalência

**Implementado agora**

- correção bayesiana de probabilidade por prevalência local;
- contagem auditável de falsos negativos resgatados pelo score de anomalia e falsos positivos extras;
- configuração BCE, `pos_weight`, sampler e ablação batch × LR.

**Próxima vertical**

- runners PyTorch para as três estratégias de desbalanceamento;
- pareamento CC/MLO;
- OC-SVM/SimSiam e persistência incremental das execuções.

## Sprint 10 — gate híbrido e robustez

**Implementado agora**

- combinação reproduzível de CNN + OC-SVM;
- cálculo de incerteza;
- rotas `RESCUE_ANOMALY`, `ABSTAIN_REVIEW` e `AUTO_CONFIDENT`;
- CI sem dados clínicos.

**Próxima vertical**

- curva cobertura × risco;
- validação externa congelada;
- KS/Wasserstein por fonte;
- resume de treino, auditoria de artefatos e tracking MLflow.

## Sprint 11 — refinamento de inferência

**Implementado agora**

- agregação ponderada de quatro previsões TTA com dispersão;
- rotas refinadas para resgate, auditoria de saliência, segunda leitura e automação confiante;
- configuração para Temperature Scaling, TENT/EATA, CC+MLO e exportação ONNX.

**Próxima vertical**

- Temperature Scaling sobre logits de validação;
- TTA real sobre imagens;
- adaptação TENT/EATA somente em LayerNorm;
- fusão CC/MLO, Grad-CAM++, ONNX FP16 e benchmark.

## Comandos verificáveis

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
python -m compileall -q src tests
```

O extra `ml` do `pyproject.toml` será instalado no ambiente de treinamento, não neste checkout de documentação/CPU.
