"""Digest de artefatos de release: SHA256SUMS por artefato."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: list[str]) -> int:
    root = Path(argv[1] if len(argv) > 1 else "dist")
    lines = [f"{sha256(item)}  {item.name}" for item in sorted(root.glob("*")) if item.is_file()]
    (root / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
