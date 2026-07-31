"""Utility helpers for dataset generator directory formatting and status checks."""

from pathlib import Path
from typing import Union


def ensure_directory(path: Union[Path, str]) -> Path:
    """Ensure a directory exists, creating parent directories if needed.

    Args:
        path: Path object or string.

    Returns:
        Path object to existing directory.
    """
    dir_path = Path(path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path
