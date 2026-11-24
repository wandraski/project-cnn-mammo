# Notebooks — Fase 3 (ordem de execução)

Abra no Colab ou Kaggle com GPU (T4/P100). O pacote é importado de `../src`.

| Ordem | Caderno | Objetivo |
|---|---|---|
| 0 | [00_ingestao](00_ingestao.ipynb) | Baixa/valida os dados, registra licença e monta o manifesto com SHA-256. |
| 1 | [01_auditoria](01_auditoria.ipynb) | Sanidade: pacientes duplicados, vazamento entre splits, distribuição por vista/lateralidade. |
| 2 | [02_preprocessamento](02_preprocessamento.ipynb) | DICOM → 8 bits, MONOCHROME1, recorte; ablação CLAHE × Z-score. |
| 3 | [03_baseline](03_baseline.ipynb) | ResNet50 (ImageNet), U-Net do zero e U-Net+ResNet34 sob o mesmo protocolo. |
| 4 | [04_benchmark](04_benchmark.ipynb) | Todas as CNNs do escopo no mesmo split, seed e hiperparâmetros. |
| 5 | [05_calibracao](05_calibracao.ipynb) | Linear probing/fine-tuning e calibração isotrópica só na validação. |
| 6 | [06_desbalanceamento](06_desbalanceamento.ipynb) | BCE/pos_weight/sampler; pareamento CC×MLO; correção de prevalência. |
| 7 | [07_gate](07_gate.ipynb) | CNN + OC-SVM, incerteza e rotas de resgate/abstenção/confiança. |
| 8 | [08_refinamento](08_refinamento.ipynb) | TTA ponderado, Temperature Scaling e rotas refinadas. |
| 9 | [09_exportacao](09_exportacao.ipynb) | Artefato, model card, digest e registro champion/challenger. |

## Ambientes
- **Kaggle**: Accelerator > GPU T4 x2; anexe o dataset.
- **Colab**: Ambiente > Alterar tipo > GPU; monte o Drive se necessário.
- **Modal**: `modal run modal_app.py`.

> Nunca processe o teste final junto ao treino: ele permanece isolado até o congelamento.
