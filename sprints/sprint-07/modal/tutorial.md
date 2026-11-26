# Sprint 07 · Baseline CNN — tutorial

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

Treina a rede **ResNet50** para dizer se uma mamografia é benigna ou maligna.

- Com `SMOKE = True`: uma época rapidinha, poucas imagens, sem baixar pesos. Serve
  só para ver se tudo funciona. **Não compare essas notas com os slides.**
- Com `SMOKE = False`: o treino de verdade, com os tempos e tamanhos do `configs/sprints.toml`.

Este caderno **não** treina a U-Net nem os 10 encoders dos slides.

## Antes de rodar

Nas S07–S09, use só imagens autorizadas de desenvolvimento e um `manifest.csv`:

```text
sample_key,patient_key,path,source,label,split
```

`path` é relativo à raiz das imagens; `patient_key` deve ser igual em todas as imagens do mesmo paciente. Use `label` 0 ou 1 e `split` `train`/`val`; cada divisão precisa das duas classes. Nunca use o teste final. Nas S10/S11, use o CSV de validação descrito acima, sem imagens.

Veja o [guia geral](../../README.md).

## Passo a passo no Modal

Aqui o caderno é só um **lançador**: quem trabalha é o comando no terminal.

1. Instale e autentique (uma vez):

   ```
   python -m pip install modal==1.6.4
   modal setup
   ```

2. Crie dois volumes (uma vez):

   ```
   modal volume create mammo-input
   modal volume create mammo-resultados
   ```

3. Suba a pasta de desenvolvimento para o volume:

   ```
   modal volume put mammo-input ./desenvolvimento /desenvolvimento
   modal volume ls mammo-input /desenvolvimento
   ```

   Confirme que apareceu o `manifest.csv` (S07–S09) ou o CSV de validação (S10/S11).

4. Abra este caderno no Jupyter **a partir da pasta do projeto** e rode a única
   célula. Ou rode direto no terminal:

   ```
   modal run sprints/modal_app.py --sprint 7 --mode smoke
   ```

5. Para o experimento completo (S07–S09), troque `smoke` por `experiment`.
   Para outro braço da S09, use `--arm batch-16-fixed` (ou outro indicado nela).

6. Copie o comando `modal volume get` que aparece no fim para baixar a pasta de resultados.

**Erros comuns**

- "modal not found": o CLI não está instalado nesse Python.
- "volume not found": os dois volumes ainda não foram criados.
- Cobrança inesperada: confira a cota do workspace antes de usar `experiment`.
