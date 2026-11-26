# Sprint 08 · Linear probing — tutorial

**O que é este caderno?** Ele faz uma tarefa do projeto de mamografia. Pense no
caderno como uma receita: cada quadro ("célula") é um passo. Você roda um passo,
olha o que aconteceu e passa para o próximo.

**Regras de ouro**

1. Rode as células **na ordem**, de cima para baixo.
2. Se aparecer um erro, leia a última linha. Ela quase sempre diz o que faltou.
3. Nunca coloque senhas, tokens ou chaves dentro do caderno.
4. Este caderno só usa dados de **treino e validação**. O teste final fica trancado.

**O que você vai ver no fim**

- Um arquivo `summary.json` com os resultados da Sprint.
- Nas S07–S09, também um checkpoint e as predições de validação.
- No Colab e no Kaggle, as saídas vão para um **ZIP**; no Modal, ficam no volume de resultados.


## O que este caderno faz

Treina **só a última camada** da ResNet50 (o "linear probing"), com o resto da
rede congelado. É rápido: 5 épocas, 224 px, batch 16, AdamW.

Serve para comparar com o treino completo da S07. Não substitui o fine-tuning do
RadImageNet nem a calibração por fonte.

## Antes de rodar

Nas S07–S09, use só imagens autorizadas de desenvolvimento e um `manifest.csv`:

```text
sample_key,patient_key,path,source,label,split
```

`path` é relativo à raiz das imagens; `patient_key` deve ser igual em todas as imagens do mesmo paciente. Use `label` 0 ou 1 e `split` `train`/`val`; cada divisão precisa das duas classes. Nunca use o teste final. Nas S10/S11, use o CSV de validação descrito acima, sem imagens.

Veja o [guia geral](../../README.md).

## Rodar no Kaggle

1. Importe o `.ipynb`. Escolha GPU nas S07–S09; nas S10/S11, CPU basta.
2. Use **Add Input** para anexar a entrada descrita acima. Se o subset não trouxer o manifesto, anexe o `manifest.csv` separado. Ajuste `DATA` na primeira célula se o caminho do subset for diferente.
3. Execute tudo. Não precisa anexar código; o ZIP aparece em `/kaggle/working/`.
