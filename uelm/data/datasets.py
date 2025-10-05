"""Dataset utilities."""

from __future__ import annotations

from pathlib import Path


def load_tiny_shakespeare(path: str | Path) -> str:
    return Path(path).read_text()
