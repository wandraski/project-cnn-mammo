"""Modal CLI para os cadernos das Sprints; não inicia jobs na importação."""
from datetime import datetime, timezone
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
app = modal.App("mammo-sprints")
inputs = modal.Volume.from_name("mammo-input")
outputs = modal.Volume.from_name("mammo-resultados")
source = (modal.Image.debian_slim(python_version="3.12")
          .env({"PYTHONPATH": "/project/src"}))


def with_code(image):
    return (image.add_local_dir(ROOT / "src", remote_path="/project/src")
            .add_local_file(ROOT / "configs" / "sprints.toml", remote_path="/project/configs/sprints.toml"))


mounts = {"/input": inputs, "/results": outputs}
gpu_image = (source.pip_install("torch==2.7.1", "torchvision==0.22.1",
                               index_url="https://download.pytorch.org/whl/cu126")
             .pip_install("numpy>=2,<3", "Pillow>=10", "pydicom>=3,<4", "pylibjpeg>=2,<3",
                          "pylibjpeg-libjpeg>=2,<3", "pylibjpeg-openjpeg>=2,<3"))


def execute(sprint, mode, arm):
    from project_cnn_mammo.protocols import run_sprint

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
    out = Path("/results") / f"sprint-{sprint:02}" / stamp
    try:
        run_sprint(sprint, Path("/input/desenvolvimento"), out, smoke=mode == "smoke", arm=arm)
    finally:
        outputs.commit()  # preserva artefatos já gravados se a execução falhar
    return str(out.relative_to("/results"))


@app.function(image=with_code(gpu_image), volumes=mounts, gpu="T4", timeout=8 * 3600)
def train(sprint: int, mode: str, arm: str) -> str:
    return execute(sprint, mode, arm)


@app.function(image=with_code(source), volumes=mounts)
def score(sprint: int) -> str:
    return execute(sprint, "experiment", "")


@app.local_entrypoint()
def main(sprint: int = 7, mode: str = "smoke", arm: str = "batch-16-fixed"):
    if sprint not in range(7, 12) or mode not in {"smoke", "experiment"}:
        raise ValueError("use --sprint 7..11 e --mode smoke ou experiment")
    path = train.remote(sprint, mode, arm) if sprint < 10 else score.remote(sprint)
    print(f"Resultados: modal volume get mammo-resultados {path} ./resultados")
