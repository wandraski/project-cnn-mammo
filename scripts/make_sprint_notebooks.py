"""Gera os 15 pontos de entrada; nenhum notebook redefine protocolos."""
from __future__ import annotations

from base64 import b64encode
from io import BytesIO
import json
from pathlib import Path
from textwrap import wrap

from package_sprints import package_code

ROOT = Path(__file__).resolve().parents[1]
TITLES = {7: 'Baseline CNN', 8: 'Linear probing ImageNet', 9: 'Ablação batch × LR',
          10: 'Gate sobre predições de validação', 11: 'Temperature Scaling + TTA (parcial)'}

SETUP = '''from pathlib import Path

SMOKE = True  # S07–S09: teste curto; False usa o protocolo experimental
ARM = 'batch-16-fixed'  # usado apenas na S09
'''
COLAB = '''from google.colab import drive

drive.mount('/content/drive')
OUTPUT = Path('/content/drive/MyDrive/mammo')
DATA = OUTPUT / 'desenvolvimento'  # também aceita os CSVs diretamente em mammo/
'''
COLAB_TRAIN = '''BAIXAR_DO_KAGGLE = False  # True se preferir baixar o subset com KAGGLE_API_TOKEN
if BAIXAR_DO_KAGGLE:
    import os, subprocess, sys
    from google.colab import userdata

    DATA = Path('/content/desenvolvimento')
    if not (DATA / '.download-ok').is_file():
        token = userdata.get('KAGGLE_API_TOKEN')
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-U', 'kaggle'], check=True)
        try:
            subprocess.run(['kaggle', 'datasets', 'download',
                            'marcowandraski/subset-multibase-cancer-mama',
                            '-p', str(DATA), '--unzip', '--force'],
                           env={**os.environ, 'KAGGLE_API_TOKEN': token}, check=True)
        finally:
            del token
        (DATA / '.download-ok').touch()
'''
KAGGLE = '''mounted = Path('/kaggle/input')
OUTPUT = Path('/kaggle/working')
'''
KAGGLE_TRAIN = '''slug = 'subset-multibase-cancer-mama'
roots = [p for p in (mounted / slug, mounted / 'datasets' / 'marcowandraski' / slug) if p.is_dir()]
DATA = roots[0] if len(roots) == 1 else mounted / slug  # ajuste se o input tiver outro nome
'''

BOOTSTRAP = '''# ponytail: código e TOML embutidos; extração temporária, sem ZIP de código separado.
import subprocess, sys
from base64 import b64decode
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

CODE_DIR = TemporaryDirectory(prefix='mammo-codigo-')
PROJECT = Path(CODE_DIR.name).resolve()
with ZipFile(BytesIO(b64decode(
{payload}
))) as source:
    for name in source.namelist():
        if not (PROJECT / name).resolve().is_relative_to(PROJECT):
            raise ValueError('Caminho inválido no código embutido')
    source.extractall(PROJECT)
sys.path.insert(0, str(PROJECT / 'src'))
'''

TRAIN = '''from project_cnn_mammo.training_run import development_manifest

if not DATA.is_dir():
    raise FileNotFoundError(f'Dados não encontrados em {DATA}. Confira DATA na primeira célula.')
MANIFEST = DATA / 'manifest.csv'
if not MANIFEST.is_file():
    candidates = {sidecars}
    if len(candidates) != 1:
        raise FileNotFoundError('Forneça um único manifest.csv real na raiz do dataset ou como arquivo separado.')
    MANIFEST = candidates[0]
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
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

from project_cnn_mammo.protocols import run_sprint

stamp = datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-%f')
archive = OUTPUT / f'sprint-{n:02}-{{stamp}}-resultados.zip'
with TemporaryDirectory(prefix='mammo-run-') as temp:
    out = Path(temp)
    try:
        print(run_sprint({n}, DATA, out, smoke={smoke}, arm=ARM, manifest_path=globals().get('MANIFEST')))
    finally:
        OUTPUT.mkdir(parents=True, exist_ok=True)
        pending = archive.with_suffix('.zip.part')
        with ZipFile(pending, 'w', ZIP_DEFLATED) as stream:
            for path in sorted(out.rglob('*')):
                if path.is_file():
                    stream.write(path, path.relative_to(out))
        with ZipFile(pending) as stream:
            if stream.testzip() is not None:
                raise RuntimeError(f'ZIP corrompido: {{pending}}')
        pending.replace(archive)
        print('ZIP:', archive)
'''

DOWNLOAD_COLAB = '''        from google.colab import files
        files.download(str(archive))'''
DOWNLOAD_KAGGLE = '''        print('Baixe o ZIP na saída do notebook.')'''

MODAL = '''# Requer o checkout local e os volumes do guia sprints/README.md.
from pathlib import Path
import shutil, subprocess
cwd = Path.cwd()
root = next((p for p in (cwd, *cwd.parents) if (p / 'sprints/modal_app.py').is_file()), None)
if root is None or not shutil.which('modal'):
    raise RuntimeError('Abra pelo checkout do projeto e instale/autentique o Modal CLI')
MODE = '{mode}'  # S07–S09: smoke ou experiment
ARM = 'batch-16-fixed'
subprocess.run(['modal', 'run', 'sprints/modal_app.py', '--sprint', '{n}',
                '--mode', MODE, '--arm', ARM], cwd=root, check=True)'''


def code_payload():
    stream = BytesIO()
    package_code(stream)
    return '\n'.join(f'    {part!r}' for part in wrap(b64encode(stream.getvalue()).decode('ascii'), 96))


def make_notebook(n: int, platform: str, payload: str) -> dict[str, object]:
    required = ('manifest.csv e imagens autorizadas de desenvolvimento' if n < 10 else
                f'sprint{n}_val.csv, com predições reais de validação')
    body = [('markdown', f'# Sprint {n:02} · {TITLES[n]} · {platform}\n\n'
             f'Leia o `tutorial.md` da pasta. Input: {required}. '
             'Não use o teste final nem cole tokens no notebook. ' +
             ('O código já está embutido.' if platform != 'modal' else 'Execute a partir do checkout local.'))]
    if platform == 'modal':
        body.append(('code', MODAL.format(n=n, mode='smoke' if n < 10 else 'experiment')))
    else:
        source = SETUP + (COLAB if platform == 'colab' else KAGGLE)
        if n < 10:
            source += COLAB_TRAIN if platform == 'colab' else KAGGLE_TRAIN
        elif platform == 'colab':
            source += f"if (OUTPUT / 'sprint{n}_val.csv').is_file():\n    DATA = OUTPUT\n"
        else:
            source += (f"inputs = sorted(mounted.rglob('sprint{n}_val.csv'))\n"
                       f"if len(inputs) != 1:\n    raise FileNotFoundError('Anexe um único sprint{n}_val.csv.')\n"
                       "DATA = inputs[0].parent\n")
        body.append(('code', source))
        body.append(('code', BOOTSTRAP.format(payload=payload)))
        if n < 10:
            sidecars = ("[p for p in (OUTPUT / 'manifest.csv',) if p.is_file()]" if platform == 'colab'
                        else "sorted(mounted.rglob('manifest.csv'))")
            body.append(('code', TRAIN.replace('{sidecars}', sidecars)))
        body.append(('code', RUN.format(n=n, smoke='SMOKE' if n < 10 else 'False') +
                     (DOWNLOAD_COLAB if platform == 'colab' else DOWNLOAD_KAGGLE)))
    return {
        'cells': [{'cell_type': kind, 'metadata': {}, 'source': text,
                   **({'execution_count': None, 'outputs': []} if kind == 'code' else {})}
                  for kind, text in body],
        'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}},
        'nbformat': 4, 'nbformat_minor': 4,
    }


if __name__ == '__main__':
    payload = code_payload()
    for n in TITLES:
        for platform in ('colab', 'kaggle', 'modal'):
            path = ROOT / 'sprints' / f'sprint-{n:02}' / platform / f'sprint_{n:02}_{platform}.ipynb'
            path.write_text(json.dumps(make_notebook(n, platform, payload), ensure_ascii=False, indent=1) + '\n')
    print('15 notebooks gerados')