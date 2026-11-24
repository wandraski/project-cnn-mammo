# Convenções de commits e código

## Commits

- Mensagens **em português**, padrão **Conventional Commits**: `tipo(escopo): descrição imperativa`.
- Tipos usados: `feat`, `fix`, `docs`, `test`, `refactor`, `ci`, `cd`, `ct`, `build`, `chore`.
- Uma intenção por commit. Commits pequenos e reversíveis.
- **Autor único**: Marco Antonio Wandraski (`@wandraski`). Proibido `Co-authored-by` e qualquer coautoria.

Exemplos:

```
feat(dados): adiciona validação de manifesto por paciente
fix(avaliação): corrige IC bootstrap em amostra degenerada
docs(adr): registra decisão da arquitetura lambda
```

## Código

- Uma responsabilidade por módulo (ver `docs/adr/0002-poo-modular.md`).
- stdlib-first; dependências pesadas apenas no extra `ml`.
- Todo contrato novo tem teste em `tests/`.
- Nada de dados clínicos, checkpoints ou segredos no repositório (varredura em `scripts/check_secrets.py`).

## Verificação local

```bash
make test      # unittest
make compile   # compileall
make smoke     # pipeline de fumaça sem dados reais
make scan      # varredura de segredos
```
