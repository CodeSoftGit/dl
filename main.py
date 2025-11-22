"""Compatibility shim for running dl directly from the repository."""

from __future__ import annotations

from pathlib import Path
import sys

# Ensure ``src`` is importable when running via ``python main.py`` without installing.
_ROOT = Path(__file__).resolve().parent
_SRC = _ROOT / "src"
if _SRC.exists():
    sys.path.insert(0, str(_SRC))

from dl.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
