"""Gera um tutorial.md por caderno (Sprint x plataforma), em linguagem simples."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPRINTS = ROOT / 'sprints'

O_QUE_E = '''# {titulo} — tutorial

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
'''

SPRINT_TEXTOS = {
    7: '''## O que este caderno faz

Treina a rede **ResNet50** para dizer se uma mamografia é benigna ou maligna.

- Com `SMOKE = True`: uma época rapidinha, poucas imagens, sem baixar pesos. Serve
  só para ver se tudo funciona. **Não compare essas notas com os slides.**
- Com `SMOKE = False`: o treino de verdade, com os tempos e tamanhos do `configs/sprints.toml`.

Este caderno **não** treina a U-Net nem os 10 encoders dos slides.''',
    8: '''## O que este caderno faz

Treina **só a última camada** da ResNet50 (o "linear probing"), com o resto da
rede congelado. É rápido: 5 épocas, 224 px, batch 16, AdamW.

Serve para comparar com o treino completo da S07. Não substitui o fine-tuning do
RadImageNet nem a calibração por fonte.''',
    9: '''## O que este caderno faz

Testa **tamanhos de lote** (batch): 8, 16, 32, 64 e 128. Para cada tamanho roda
dois braços — taxa de aprendizado fixa e taxa que cresce junto com o lote.

São 15 épocas por braço, então **roda um braço por vez**. Troque `ARM` na primeira
célula de código, por exemplo `batch-8-fixed` ou `batch-128-linear`.

Se faltar memória da GPU, pare e anote. Não troque o batch por um menor e finja
que era o original.''',
    10: '''## O que este caderno faz

Aplica o **portão de decisão** (gate) sobre as notas que você já tem: nota da CNN,
nota de anomalia e confiança da U-Net. Para cada exame, ele escolhe entre
"resgatar", "segunda leitura" ou "decisão automática". **Não treina nada.**

Você precisa do arquivo `sprint10_val.csv` com as notas reais de validação:
`sample_key,split,p_cnn,s_anom,unet_conf`. Use `split=val` e notas entre 0 e 1.''',
    11: '''## O que este caderno faz

Parte da Sprint 11: ajusta a **temperatura** (deixa as probabilidades mais
calibradas) e combina **quatro visões** da mesma imagem (original, espelhada,
girada para um lado e para o outro). **Não treina pesos novos.**

Você precisa do arquivo `sprint11_val.csv`, com quatro logits por exame:
`sample_key,split,label,logit_original,logit_hflip,logit_plus5,logit_minus5`.
Use `split=val`, rótulos 0 ou 1 e valores antes da sigmoide.''',
}

ANTES = """## Antes de rodar

Nas S07–S09, use só imagens autorizadas de desenvolvimento e um `manifest.csv`:

```text
sample_key,patient_key,path,source,label,split
```

`path` é relativo à raiz das imagens; `patient_key` deve ser igual em todas as imagens do mesmo paciente. Use `label` 0 ou 1 e `split` `train`/`val`; cada divisão precisa das duas classes. Nunca use o teste final. Nas S10/S11, use o CSV de validação descrito acima, sem imagens.

Veja o [guia geral](../../README.md)."""

PLATAFORMA = {
    'colab': """## Rodar no Colab

1. Abra o `.ipynb`. Escolha GPU nas S07–S09; nas S10/S11, CPU basta.
2. Coloque os dados em `Meu Drive/mammo/desenvolvimento/`. Nas S07–S09, se for baixar do Kaggle, mude `BAIXAR_DO_KAGGLE = True` na primeira célula e habilite `KAGGLE_API_TOKEN` no painel **Segredos**. Se faltar o manifesto, coloque o seu em `Meu Drive/mammo/`. Nas S10/S11, você pode colocar o CSV diretamente nessa pasta.
3. Clique em **Executar tudo** e autorize o Drive. O código já está no notebook. O ZIP aparece em `Meu Drive/mammo/` e o Colab inicia o download.""",
    'kaggle': """## Rodar no Kaggle

1. Importe o `.ipynb`. Escolha GPU nas S07–S09; nas S10/S11, CPU basta.
2. Nas S07–S09, anexe `marcowandraski/subset-multibase-cancer-mama` em **Add Input**. O notebook adapta o manifesto benigno × maligno automaticamente, usando apenas `train`/`val`. Nas S10/S11, anexe o CSV de predições indicado acima.
3. Execute tudo. Não precisa anexar código; o ZIP aparece em `/kaggle/working/`.""",
    'modal': '''## Passo a passo no Modal

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
   modal run sprints/modal_app.py --sprint {sprint} --mode {mode}
   ```

5. Para o experimento completo (S07–S09), troque `smoke` por `experiment`.
   Para outro braço da S09, use `--arm batch-16-fixed` (ou outro indicado nela).

6. Copie o comando `modal volume get` que aparece no fim para baixar a pasta de resultados.

**Erros comuns**

- "modal not found": o CLI não está instalado nesse Python.
- "volume not found": os dois volumes ainda não foram criados.
- Cobrança inesperada: confira a cota do workspace antes de usar `experiment`.
''',
}


def tutorial(sprint: int, platform: str) -> str:
    titulos = {
        7: 'Sprint 07 · Baseline CNN', 8: 'Sprint 08 · Linear probing',
        9: 'Sprint 09 · Ablação de batch e taxa', 10: 'Sprint 10 · Portão de decisão',
        11: 'Sprint 11 · Calibração e TTA (parte)',
    }
    return '\n\n'.join([
        O_QUE_E.format(titulo=titulos[sprint]),
        SPRINT_TEXTOS[sprint],
        ANTES,
        PLATAFORMA[platform].format(sprint=sprint, mode='smoke' if sprint < 10 else 'experiment').rstrip(),
    ]) + '\n'


if __name__ == '__main__':
    for sprint in range(7, 12):
        for platform in ('colab', 'kaggle', 'modal'):
            path = SPRINTS / f'sprint-{sprint:02}' / platform / 'tutorial.md'
            path.write_text(tutorial(sprint, platform), encoding='utf-8')
    print('15 tutoriais gerados')
