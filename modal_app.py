"""Entrada Modal (GPU) para o pipeline — `modal run modal_app.py`.

O módulo importa sem `modal` instalado; só a execução remota exige o pacote.
"""

from __future__ import annotations

from pathlib import Path

MODAL_GUIDANCE = (
    "Instale com `pip install modal` e autentique com `modal token new` antes de rodar."
)


def _modal():
    try:
        import modal  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(MODAL_GUIDANCE) from exc
    return modal


def build_image():
    modal = _modal()
    return (
        modal.Image.debian_slim(python_version="3.12")
        .pip_install("numpy", "pandas", "pillow", "pydicom", "torch", "torchvision", "scikit-learn")
        .copy_local_dir(Path(__file__).resolve().parents[1] / "src", "/root/src")
    )


def _modal_available() -> bool:
    try:
        import modal  # noqa: F401  # type: ignore
    except ImportError:
        return False
    return True


app = None
if _modal_available():
    modal = _modal()
    app = modal.App("project-cnn-mammo", image=build_image())

    @app.function(gpu="T4", timeout=3600)
    def run_pipeline(config: str = "configs/sprints.toml", smoke: bool = False) -> str:
        """Executa o pipeline dentro do container; retorna caminho do relatório."""
        import subprocess

        command = ["python", "-m", "project_cnn_mammo.cli", "--config", config, "smoke" if smoke else "manifest"]
        completed = subprocess.run(command, capture_output=True, text=True, cwd="/root")
        return completed.stdout or completed.stderr


def main() -> int:
    print(MODAL_GUIDANCE if not _modal_available() else "execute `modal run modal_app.py`")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
