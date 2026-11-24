"""Varredura pre-commit: impede dados clinicos, checkpoints e segredos no Git.

Uso: python scripts/check_secrets.py .
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

FORBIDDEN_PATTERNS = {
    "credencial-aws": re.compile(r"AKIA[0-9A-Z]{16}"),
    "chave-privada": re.compile(r"-----BEGIN (RSA|OPENSSH|EC) PRIVATE KEY-----"),
    "token-generico": re.compile(r"(?i)(api[_-]?key|secret|token)\s*[=:]\s*['\"][A-Za-z0-9_\-]{16,}"),
}
FORBIDDEN_SUFFIXES = {".dcm", ".dicom", ".nii", ".nii.gz", ".pth", ".pt", ".ckpt", ".onnx"}
SKIP_PARTS = {".git", "__pycache__", ".venv", "node_modules"}


def scan(root: Path) -> list[str]:
    findings: list[str] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or any(part in SKIP_PARTS for part in path.parts):
            continue
        if path.suffix.lower() in FORBIDDEN_SUFFIXES or path.name.endswith(".dcm"):
            findings.append(f"arquivo proibido: {path}")
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for label, pattern in FORBIDDEN_PATTERNS.items():
            if pattern.search(text):
                findings.append(f"{label}: {path}")
    return findings


def main(argv: list[str]) -> int:
    root = Path(argv[1] if len(argv) > 1 else ".").resolve()
    findings = scan(root)
    for finding in findings:
        print(f"[bloqueado] {finding}")
    print(f"{len(findings)} achados em {root}")
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
