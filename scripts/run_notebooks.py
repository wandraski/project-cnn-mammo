#!/usr/bin/env python3
"""Executa os notebooks localmente, celula a celula, sem Jupyter instalado.

Uso: python scripts/run_notebooks.py [notebook1.ipynb ...]
Somente codigo Python e executado; comandos de shell (`!...`) sao anotados.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]


def code_cells(path: Path) -> list[str]:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    return [
        "".join(cell.get("source", []))
        for cell in notebook.get("cells", [])
        if cell.get("cell_type") == "code"
    ]


def sanitize(source: str) -> tuple[str, list[str]]:
    lines = source.splitlines()
    kept: list[str] = []
    skipped: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("!"):
            skipped.append(stripped)
            continue
        kept.append(line)
    return "\n".join(kept), skipped


def run(path: Path) -> dict[str, object]:
    environment: dict[str, object] = {}
    results = []
    for index, source in enumerate(code_cells(path), start=1):
        code, skipped = sanitize(source)
        try:
            exec(compile(code, f"{path.name}#celula-{index}", "exec"), environment)  # noqa: S102
            results.append({"celula": index, "status": "ok", "skipped": skipped})
        except Exception as exc:  # registra e para, como o Jupyter faria
            results.append(
                {
                    "celula": index,
                    "status": "erro",
                    "erro": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(limit=2).splitlines()[-3:],
                }
            )
            break
    return {"notebook": path.name, "celulas": results}


def main(argv: list[str]) -> int:
    targets = [Path(item) for item in argv[1:]] or sorted((ROOT / "notebooks").glob("*.ipynb"))
    failures = 0
    for target in targets:
        report = run(target)
        print(f"\n=== {report['notebook']}")
        for cell in report["celulas"]:
            mark = "ok" if cell["status"] == "ok" else "FALHA"
            detail = cell.get("erro", "")
            print(f"  celula {cell['celula']}: {mark} {detail}")
            if cell.get("skipped"):
                print(f"    (ignorado: {'; '.join(cell['skipped'])})")
            if cell["status"] == "erro":
                failures += 1
    print(f"\n{failures} celula(s) com falha")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
