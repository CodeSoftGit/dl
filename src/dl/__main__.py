"""Module executed when running ``python -m dl``."""

from __future__ import annotations

from .cli import main


def _main() -> int:
    return main()


if __name__ == "__main__":
    raise SystemExit(_main())

