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

## Parâmetros dos slides S7–S11

Os slides solo orientam o fork, dentro do escopo do professor. Os slides da equipe são referências separadas, não evidência do fork. Não há um número único de épocas; os valores anteriores de 12/8 não tinham fonte.

| Fonte (página do PDF) | Protocolo | Situação no fork |
|---|---|---|
| S7 solo, p. 1, 5, 8 | ImageNet/224²; seeds 42/123/999; smoke 100; 1.000 reamostragens. U-Net original vs. ResNet34: 512², perda 0,5 BCE + 0,5 Dice. OC-SVM ajustado só em benignos. | CNNs do núcleo, não todos os 10 encoders; segmentação/OC-SVM completos e IC por paciente pendentes. Épocas da U-Net solo não confirmadas. |
| S8 equipe, p. 2; S8 solo, p. 2–7 | Linear probe: 5 épocas, batch 16, 224², AdamW, LR 1e-4. Solo também prevê fine-tuning parcial RadImageNet (`layer3`/`layer4`), normalizações, calibração por fonte e resolução até 1024² (batch 8 e AMP na T4). | Configuração de referência e demos; fine-tuning, calibração por fonte e comparações reais pendentes. |
| S9 solo, p. 1, 3–4, 7 | 15 épocas, AdamW, 224², batch 8/16/32/64/128, LR fixo 1e-4 ou `1e-4 × batch/16`; limiares de estudo 0,22/0,65. Não confundir com `batch/32` da S7/S8 equipe ou as 5 épocas do braço de desbalanceamento da S9 equipe (p. 2). | Grade declarada; treino controlado da ablação, BatchNorm/VRAM e resgate real pendentes. |
| S10 equipe, p. 21–23 | Referência de CNN: 20 épocas, batch 32, 224², LR 1e-4, seed 42. Referência de segmentação distinta: 15 épocas, batch 4, 768×512, BCE+Dice, AMP, seeds 42/7/123. | Benchmark CNN lê os parâmetros do TOML; não aplica o perfil de segmentação da equipe à S7 solo. |
| S10 solo, p. 1–2, 8 | ConvNeXt-Tiny direto com CLAHE/batch 16; comparação com U-Net → CNN → OC-SVM. Gate: 0,22/0,68, α 0,78, banda 0,08 e confiança mínima da U-Net 0,50. | Gate unitário; comparação real e avaliação externa congelada pendentes. |
| S11 solo, p. 1–7 | Temperature Scaling por NLL em validação, TTA original/HFlip/±5°, TENT/EATA em LayerNorm como braço separado, fusão CC/MLO, revisão a 0,35 de divergência/banda 0,065, ONNX FP16 e Grad-CAM++. | TTA e rotas demonstrados com valores sintéticos; pesos TTA são configuração do fork, não valores confirmados do slide. Adaptação não substitui o teste final congelado. |

O notebook `03` só dispara treino mediante caminho explícito. O diretório de desenvolvimento deve conter `manifest.csv` com `sample_key,patient_key,path,source,label,split`, caminhos relativos ao diretório e treino/validação definidos por paciente. O notebook `04` exibe a grade, não inicia treino. Nenhum resultado de GPU ou número simulado dos slides foi validado como resultado do fork.

## Comandos verificáveis

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
python -m compileall -q src tests
```

O extra `ml` do `pyproject.toml` será instalado no ambiente de treinamento, não neste checkout de documentação/CPU.
