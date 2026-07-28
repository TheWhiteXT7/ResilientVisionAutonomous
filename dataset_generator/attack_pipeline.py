"""In-memory bridge between RGB images and the verified attack engine.

This module validates already loaded image arrays and applies an already sized
laser pattern. It performs no filesystem, resizing, discovery, or metadata
operations.
"""

from __future__ import annotations

import numpy as np

from attack_engine import apply_attack


def attack_image(image: np.ndarray, pattern: np.ndarray) -> np.ndarray:
    """Apply an RGB laser pattern to an RGB image.

    Both inputs must be NumPy arrays with ``dtype=np.uint8`` and identical
    ``(height, width, 3)`` shapes. Once validated, the function delegates the
    additive, saturating pixel operation to :func:`attack_engine.apply_attack`.
    Neither input array is modified.

    Args:
        image: Source RGB image as a ``uint8`` array of shape ``(H, W, 3)``.
        pattern: RGB laser pattern as a ``uint8`` array of shape ``(H, W, 3)``.

    Returns:
        A new attacked RGB ``uint8`` NumPy array with the same shape as
        ``image``.

    Raises:
        TypeError: If either input is not a NumPy array.
        ValueError: If an input is not a non-empty uint8 RGB array or the
            image and pattern dimensions do not match.
    """
    _validate_rgb_array(image, "image")
    _validate_rgb_array(pattern, "pattern")

    if image.shape != pattern.shape:
        raise ValueError(
            "Image and pattern must have identical (height, width, 3) shapes."
        )

    attacked_image = apply_attack(image, pattern)
    _validate_rgb_array(attacked_image, "attacked image")
    return attacked_image


def _validate_rgb_array(array: np.ndarray, name: str) -> None:
    """Raise a descriptive error unless ``array`` is a non-empty RGB image."""
    if not isinstance(array, np.ndarray):
        raise TypeError(f"{name.capitalize()} must be a NumPy array.")
    if array.dtype != np.uint8:
        raise ValueError(f"{name.capitalize()} dtype must be np.uint8.")
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(
            f"{name.capitalize()} must have shape (height, width, 3)."
        )
    if array.shape[0] <= 0 or array.shape[1] <= 0:
        raise ValueError(
            f"{name.capitalize()} height and width must be positive."
        )
