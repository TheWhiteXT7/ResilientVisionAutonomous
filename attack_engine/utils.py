"""Utility functions for array/PIL conversions and helper operations."""

from typing import Tuple, Union
from PIL import Image
import numpy as np


def pil_to_numpy(image: Image.Image) -> np.ndarray:
    """Convert a PIL Image to a uint8 NumPy array.

    Args:
        image: Source PIL Image.

    Returns:
        RGB or grayscale uint8 NumPy array.
    """
    if not isinstance(image, Image.Image):
        raise TypeError("image must be a PIL.Image.Image instance.")
    return np.asarray(image, dtype=np.uint8)


def numpy_to_pil(array: np.ndarray) -> Image.Image:
    """Convert a uint8 NumPy array to a PIL Image.

    Args:
        array: Source NumPy array.

    Returns:
        PIL Image instance.
    """
    if not isinstance(array, np.ndarray):
        raise TypeError("array must be a NumPy ndarray.")
    if array.dtype != np.uint8:
        raise ValueError("array dtype must be np.uint8.")
    return Image.fromarray(array)


def apply_pattern(base: np.ndarray, pattern: np.ndarray) -> np.ndarray:
    """Overlay an additive pattern onto a NumPy array (legacy additive overlay).

    Args:
        base: Source image as uint8 NumPy array.
        pattern: Laser pattern as uint8 NumPy array.

    Returns:
        New uint8 NumPy array with pattern added and clipped to [0, 255].
    """
    if not isinstance(base, np.ndarray) or not isinstance(pattern, np.ndarray):
        raise TypeError("base and pattern must be NumPy arrays.")
    if base.dtype != np.uint8 or pattern.dtype != np.uint8:
        raise ValueError("Inputs must have dtype np.uint8.")
    if base.shape != pattern.shape:
        raise ValueError("base and pattern shapes must match.")

    b_float = base.astype(float)
    p_float = pattern.astype(float)
    return np.clip(b_float + p_float, 0, 255).astype(np.uint8)
