# project-cnn-mammo

Projeto individual de visão computacional para apoio à análise de mamografias, iniciado a partir das metas da **Sprint 7**.

## Estado atual

- repositório novo, sem reaproveitamento de histórico Git da antiga equipe;
- único autor e responsável: **Marco Antonio Wandraski (`@wandraski`)**;
- o PDF do professor é a fonte normativa inicial;
- todos os slides, antigos ou individuais, são **metas a implementar**, não evidências de trabalho concluído;
- nenhum resultado experimental está declarado como executado;
- escopo principal limitado a **CNNs**.

## Objetivo acadêmico

Desenvolver e avaliar CNNs para classificação de lesões mamárias sob protocolo reproduzível, priorizando sensibilidade/recall, custo computacional, generalização externa e interpretabilidade. O projeto não produz diagnóstico clínico.

## Início: Sprint 7

A Sprint 7 estabelece:

1. baseline de classificação com ResNet50 pré-treinada;
2. baseline de segmentação com U-Net do zero e U-Net com encoder ResNet34;
3. `evaluate.py` comum com métricas clínicas e IC 95% por bootstrap;
4. split por paciente e isolamento do conjunto final de teste;
5. rastreabilidade de configuração, seed, dados e ambiente;
6. preparação para as metas posteriores extraídas dos slides.

Veja [`docs/requirements/sprint-07.md`](docs/requirements/sprint-07.md).

## Fontes

- [`docs/requirements/professor.md`](docs/requirements/professor.md): requisitos extraídos do PDF de orientação.
- [`docs/requirements/professores-adjacentes.md`](docs/requirements/professores-adjacentes.md): 237 requisitos canônicos de Engenharia de Software e MLOps (99 + 138), com critério de aceite e proveniência.
- [`docs/requirements/sprint-07.md`](docs/requirements/sprint-07.md): ponto inicial do trabalho.
- [`docs/requirements/slide-intake.md`](docs/requirements/slide-intake.md): processo para incorporar os slides das disciplinas.
- [`docs/IMPLEMENTATION.md`](docs/IMPLEMENTATION.md): código e próximas verticais das Sprints 7–11.
- [`docs/STATUS.md`](docs/STATUS.md): distinção entre meta, implementado e medido.

## Verificação local

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
python -m compileall -q src tests
```

As dependências pesadas de treino estão no extra `ml` do `pyproject.toml`; o núcleo de regras e segurança usa somente a biblioteca padrão.

## Segurança e dados

Não versionar imagens, DICOMs, máscaras, checkpoints, credenciais, bancos de experimentos ou identificadores de pacientes. Os dados finais de teste não podem orientar decisões de desenvolvimento.

## Autoria

Este é um trabalho individual. A política está em [`CONTRIBUTORS.md`](CONTRIBUTORS.md) e é reforçada por [`.github/CODEOWNERS`](.github/CODEOWNERS).
