"""Empacota somente código e configuração para upload; nunca dados ou credenciais."""
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

ROOT = Path(__file__).resolve().parents[1]

def package_code(target):
    with ZipFile(target, 'w', ZIP_DEFLATED) as archive:
        paths = sorted((ROOT / 'src' / 'project_cnn_mammo').glob('*.py')) + [ROOT / 'configs/sprints.toml']
        for path in paths:
            info = ZipInfo(path.relative_to(ROOT).as_posix())  # sem timestamps do checkout
            archive.writestr(info, path.read_bytes(), compress_type=ZIP_DEFLATED)


if __name__ == '__main__':
    target = ROOT / 'outputs' / 'mammo-codigo.zip'
    target.parent.mkdir(exist_ok=True)
    package_code(target)
    print(target)
