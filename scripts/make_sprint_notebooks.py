"""Gera os 15 pontos de entrada; nenhum notebook redefine protocolos."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TITLES = {7: "Baseline CNN", 8: "Linear probing ImageNet", 9: "Ablação batch × LR",
          10: "Gate sobre predições de validação", 11: "Temperature Scaling + TTA (parcial)"}

CLOUD = '''from pathlib import Path
import subprocess, sys
from zipfile import ZipFile

{{paths}}
# Altere DATA se o input estiver em outro caminho. Não aponte para o teste final.
SMOKE = True  # S7–S9: uma época, sem pesos externos. False usa o protocolo configurado.
ARM = 'batch-16-fixed'  # somente S9; um braço por execução

if not SOURCE_ZIP.is_file():
    raise FileNotFoundError(f'Anexe o ZIP de código em {SOURCE_ZIP}; veja sprints/README.md')
with ZipFile(SOURCE_ZIP) as z:
    for item in z.infolist():
        dest = (PROJECT / item.filename).resolve()
        if not dest.is_relative_to(PROJECT.resolve()):
            raise ValueError('Arquivo de código com caminho inválido')
    z.extractall(PROJECT)
sys.path.insert(0, str(PROJECT / 'src'))
if not (PROJECT / 'configs' / 'sprints.toml').is_file():
    raise RuntimeError('ZIP incorreto: gere com python scripts/package_sprints.py')
'''

COLAB = '''from google.colab import drive
drive.mount('/content/drive')
base = Path('/content/drive/MyDrive/mammo')
SOURCE_ZIP = base / 'mammo-codigo.zip'
PROJECT = Path('/content/mammo-codigo')
DATA = base / 'desenvolvimento'
OUTPUT = base / 'resultados' '''
KAGGLE = '''mounted = Path('/kaggle/input')
slug = 'subset-multibase-cancer-mama'
roots = [p for p in (mounted / slug, mounted / 'datasets' / 'marcowandraski' / slug) if p.is_dir()]
DATA = roots[0] if len(roots) == 1 else mounted / slug
zips = sorted(mounted.rglob('mammo-codigo.zip'))
if len(zips) != 1:
    raise RuntimeError('Anexe exatamente um mammo-codigo.zip como input de código')
SOURCE_ZIP = zips[0]
PROJECT = Path('/kaggle/working/mammo-codigo')
OUTPUT = Path('/kaggle/working/resultados')'''

TRAIN = '''from project_cnn_mammo.training_run import development_manifest
MANIFEST = DATA / 'manifest.csv'
if not MANIFEST.is_file():
    MANIFEST = SOURCE_ZIP.parent / 'manifest.csv'  # sidecar, sem recarregar as imagens
if not MANIFEST.is_file():
    raise FileNotFoundError('Anexe manifest.csv com IDs e split reais; veja sprints/README.md')
entries = development_manifest(DATA, MANIFEST)
print(f'Manifesto válido: {len(entries)} imagens de desenvolvimento')
import torch, torchvision  # use o par já instalado no runtime; não o atualize aqui
if not torch.cuda.is_available():
    raise RuntimeError('Selecione um runtime GPU, reinicie a sessão e execute de novo')
if any(entry.path.suffix.lower() in {'.dcm', '.dicom'} for entry in entries):
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'pydicom>=3,<4',
                    'pylibjpeg>=2,<3', 'pylibjpeg-libjpeg>=2,<3', 'pylibjpeg-openjpeg>=2,<3'], check=True)
print('GPU:', torch.cuda.get_device_name())'''

RUN = '''from datetime import datetime, timezone
from zipfile import ZIP_DEFLATED, ZipFile


def pack_results():
    """Compacta toda a saída; roda mesmo se o experimento falhar."""
    archive = OUTPUT / 'sprint-{n:02}-resultados.zip'
    with ZipFile(archive, 'w', ZIP_DEFLATED) as stream:
        for path in sorted(OUTPUT.rglob('*')):
            if path.is_file() and path != archive:
                stream.write(path, path.relative_to(OUTPUT))
    assert ZipFile(archive).testzip() is None, 'ZIP corrompido'
    return archive


try:
    from project_cnn_mammo.protocols import run_sprint
    out = OUTPUT / 'sprint-{n:02}' / datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-%f')
    print(run_sprint({n}, DATA, out, smoke={smoke}, arm=ARM, manifest_path=globals().get('MANIFEST')))
    print('Saída:', out)
finally:
    archive = pack_results()
    print('ZIP:', archive, archive.stat().st_size, 'bytes')
{download}'''

DOWNLOAD_COLAB = """    from google.colab import files
    files.download(archive)  # baixa o ZIP automaticamente no navegador"""

DOWNLOAD_KAGGLE = """    print('Baixe o ZIP em Save Version > Output ou na aba Data do notebook.')"""

MODAL = '''# Jupyter local, iniciado a partir da raiz deste checkout. Setup e upload: sprints/README.md.
from pathlib import Path
import shutil, subprocess
root = Path.cwd()
if not (root / 'sprints/modal_app.py').is_file() or not shutil.which('modal'):
    raise RuntimeError('Abra pela raiz do checkout e instale/autentique o Modal CLI')
MODE = '{mode}'  # S7–S9: smoke ou experiment
ARM = 'batch-16-fixed'
subprocess.run(['modal', 'run', 'sprints/modal_app.py', '--sprint', '{n}',
                '--mode', MODE, '--arm', ARM], cwd=root, check=True)'''


def make_notebook(n: int, platform: str) -> dict[str, object]:
    required = ('manifest.csv e as imagens de desenvolvimento' if n < 10 else
                f'sprint{n}_val.csv, com predições reais de validação (veja o schema no README)')
    body = [('markdown', f'# Sprint {n:02} · {TITLES[n]} · {platform}\n\n'
             f'Antes de executar: leia `sprints/README.md`. Input obrigatório: {required}. '
             'Não usa o teste final. Não cole tokens no notebook.')]
    if platform == 'modal':
        body.append(('code', MODAL.format(n=n, mode='smoke' if n < 10 else 'experiment')))
    else:
        body.append(('code', CLOUD.replace('{{paths}}', COLAB if platform == 'colab' else KAGGLE)))
        if n < 10:
            body.append(('code', TRAIN))
        body.append(('code', RUN.format(
            n=n, smoke='SMOKE' if n < 10 else 'False',
            download=DOWNLOAD_COLAB if platform == 'colab' else DOWNLOAD_KAGGLE,
        )))
    return {
        'cells': [{'cell_type': kind, 'metadata': {}, 'source': text,
                   **({'execution_count': None, 'outputs': []} if kind == 'code' else {})}
                  for kind, text in body],
        'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}},
        'nbformat': 4, 'nbformat_minor': 4,
    }


if __name__ == '__main__':
    for n in TITLES:
        for platform in ('colab', 'kaggle', 'modal'):
            path = ROOT / 'sprints' / f'sprint-{n:02}' / platform / f'sprint_{n:02}_{platform}.ipynb'
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(make_notebook(n, platform), ensure_ascii=False, indent=1) + '\n')
    print('15 notebooks gerados')
