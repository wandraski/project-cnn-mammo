# Sprint 10 · Portão de decisão — tutorial

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

Aplica o **portão de decisão** (gate) sobre as notas que você já tem: nota da CNN,
nota de anomalia e confiança da U-Net. Para cada exame, ele escolhe entre
"resgatar", "segunda leitura" ou "decisão automática". **Não treina nada.**

Você precisa do arquivo `sprint10_val.csv` com as notas reais de validação:
`sample_key,split,p_cnn,s_anom,unet_conf`. Use `split=val` e notas entre 0 e 1.

## Antes de rodar

Nas S07–S09, use só imagens autorizadas de desenvolvimento e um `manifest.csv`:

```text
sample_key,patient_key,path,source,label,split
```

`path` é relativo à raiz das imagens; `patient_key` deve ser igual em todas as imagens do mesmo paciente. Use `label` 0 ou 1 e `split` `train`/`val`; cada divisão precisa das duas classes. Nunca use o teste final. Nas S10/S11, use o CSV de validação descrito acima, sem imagens.

Veja o [guia geral](../../README.md).

## Rodar no Kaggle

1. Importe o `.ipynb`. Escolha GPU nas S07–S09; nas S10/S11, CPU basta.
2. Nas S07–S09, anexe `marcowandraski/subset-multibase-cancer-mama` em **Add Input**. O notebook adapta o manifesto benigno × maligno automaticamente, usando apenas `train`/`val`. Nas S10/S11, anexe o CSV de predições indicado acima.
3. Execute tudo. Não precisa anexar código; o ZIP aparece em `/kaggle/working/`.
