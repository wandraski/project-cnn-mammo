"""Empacota somente código e configuração para upload; nunca dados ou credenciais."""
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]

if __name__ == '__main__':
    target = ROOT / 'outputs' / 'mammo-codigo.zip'
    target.parent.mkdir(exist_ok=True)
    with ZipFile(target, 'w', ZIP_DEFLATED) as archive:
        for path in sorted((ROOT / 'src' / 'project_cnn_mammo').glob('*.py')):
            archive.write(path, path.relative_to(ROOT).as_posix())
        archive.write(ROOT / 'configs/sprints.toml', 'configs/sprints.toml')
    print(target)
