"""Public API for generating a laser-pattern-corrupted RGB image."""

import numpy as np

from .apply_pattern import apply_pattern


def apply_attack(image: np.ndarray, pattern: np.ndarray) -> np.ndarray:
    """Return ``image`` with the additive rolling-shutter ``pattern`` applied.

    Args:
        image: RGB uint8 NumPy array with shape ``(H, W, 3)``.
        pattern: RGB uint8 NumPy array with shape ``(H, W, 3)``.

    Returns:
        A new RGB uint8 NumPy array of shape ``(H, W, 3)``.

    Raises:
        AssertionError: If inputs are not matching uint8 RGB image arrays.
    """
    assert image.ndim == 3 and image.shape[2] == 3
    assert pattern.ndim == 3 and pattern.shape[2] == 3
    return apply_pattern(image, pattern)
