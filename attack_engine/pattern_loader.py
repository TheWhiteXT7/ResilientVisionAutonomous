"""Optional RGB PNG loading helper; not needed when patterns are NumPy arrays."""

from pathlib import Path

import numpy as np


def load_rgb_pattern(path: str | Path) -> np.ndarray:
    """Load a PNG/JPEG pattern as an RGB uint8 NumPy array.

    Pillow is intentionally imported only here so the core attack API requires
    NumPy alone.
    """
    try:
        from PIL import Image
    except ImportError as error:
        raise ImportError(
            "Loading image files requires Pillow. Install it with: pip install Pillow"
        ) from error

    with Image.open(path) as source:
        return np.asarray(source.convert("RGB"), dtype=np.uint8)
