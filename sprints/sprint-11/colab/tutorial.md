# Sprint 11 · Calibração e TTA (parte) — tutorial

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

Parte da Sprint 11: ajusta a **temperatura** (deixa as probabilidades mais
calibradas) e combina **quatro visões** da mesma imagem (original, espelhada,
girada para um lado e para o outro). **Não treina pesos novos.**

Você precisa do arquivo `sprint11_val.csv`, com quatro logits por exame:
`sample_key,split,label,logit_original,logit_hflip,logit_plus5,logit_minus5`.
Use `split=val`, rótulos 0 ou 1 e valores antes da sigmoide.

## Antes de rodar

Nas S07–S09, use só imagens autorizadas de desenvolvimento e um `manifest.csv`:

```text
sample_key,patient_key,path,source,label,split
```

`path` é relativo à raiz das imagens; `patient_key` deve ser igual em todas as imagens do mesmo paciente. Use `label` 0 ou 1 e `split` `train`/`val`; cada divisão precisa das duas classes. Nunca use o teste final. Nas S10/S11, use o CSV de validação descrito acima, sem imagens.

Veja o [guia geral](../../README.md).

## Rodar no Colab

1. Abra o `.ipynb`. Escolha GPU nas S07–S09; nas S10/S11, CPU basta.
2. Coloque os dados em `Meu Drive/mammo/desenvolvimento/`. Nas S07–S09, se for baixar do Kaggle, mude `BAIXAR_DO_KAGGLE = True` na primeira célula e habilite `KAGGLE_API_TOKEN` no painel **Segredos**. Se faltar o manifesto, coloque o seu em `Meu Drive/mammo/`. Nas S10/S11, você pode colocar o CSV diretamente nessa pasta.
3. Clique em **Executar tudo** e autorize o Drive. O código já está no notebook. O ZIP aparece em `Meu Drive/mammo/` e o Colab inicia o download.
