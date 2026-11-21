# Sprint 7 — ponto inicial

**Estado:** metas ainda não implementadas.

## Objetivo

Estabelecer os baselines comuns exigidos pelo professor e uma avaliação reproduzível antes de avançar às metas dos slides posteriores.

## Entregas

- [ ] manifesto de dados com chave pseudonimizada de paciente, fonte, rótulo, vista, lateralidade e split;
- [ ] validação automática contra vazamento de paciente entre splits;
- [ ] baseline ResNet50 pré-treinada em ImageNet;
- [ ] U-Net do zero;
- [ ] U-Net com encoder ResNet34 pré-treinado;
- [ ] avaliação de segmentação com Dice, IoU, sensibilidade e precisão por pixel;
- [ ] `evaluate.py` com AUROC, AUPRC, pontos clínicos, F1, matriz de confusão e IC 95% por bootstrap;
- [ ] comparação inicial do pipeline direto com U-Net → CNN;
- [ ] rastreamento de configuração, seed, commit, versão dos dados, ambiente, tempo e VRAM;
- [ ] smoke test sem dados reais e sem GPU;
- [ ] documentação das limitações e resultados negativos.

## Critérios de aceite

1. Todos os testes e o smoke test rodam sem dataset real.
2. Nenhum dado clínico, checkpoint ou segredo entra no Git.
3. O conjunto final de teste permanece inacessível ao treino e à seleção.
4. A mesma configuração pode ser reproduzida por comando documentado.
5. Métricas são calculadas por paciente quando aplicável e com bootstrap de 1.000 reamostragens.

## Não incluído nesta etapa

- resultados mencionados nos slides;
- CI/CD/CT completo;
- arquitetura Lambda e DVC;
- deploy ou API;
- execução do teste final.

Esses itens entram após o baseline da Sprint 7 e a extração sistemática dos requisitos dos demais slides.
