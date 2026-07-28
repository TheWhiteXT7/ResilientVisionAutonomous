"""Pattern-image loading and preparation utilities.

This module loads RGB attack-pattern assets and resizes them to a target image
shape. It does not apply patterns to images, write files, or create metadata.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError


SUPPORTED_PATTERN_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg"})
"""File extensions accepted for attack-pattern images."""


def load_pattern(path: str | Path) -> np.ndarray:
    """Load an attack pattern as an RGB ``uint8`` NumPy array.

    The input must be a PNG, JPG, or JPEG file. Pillow decodes the image and
    converts it to RGB, ensuring that the returned array has shape
    ``(height, width, 3)`` and values in the inclusive range ``0`` to ``255``.
    The pattern is not resized or otherwise altered by this function.

    Args:
        path: Path to a readable PNG, JPG, or JPEG pattern image.

    Returns:
        A new RGB NumPy array with ``dtype=np.uint8`` and shape ``(H, W, 3)``.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the file extension is unsupported or the decoded result
            is not a three-channel RGB array.
        OSError: If the file cannot be opened or decoded as an image.
    """
    pattern_path = Path(path)

    if not pattern_path.exists():
        raise FileNotFoundError(
            f"Pattern image does not exist: {pattern_path}"
        )
    if not pattern_path.is_file():
        raise ValueError(f"Pattern path must be a file: {pattern_path}")
    if pattern_path.suffix.lower() not in SUPPORTED_PATTERN_EXTENSIONS:
        raise ValueError(
            "Unsupported pattern format. Expected PNG, JPG, or JPEG: "
            f"{pattern_path}"
        )

    try:
        with Image.open(pattern_path) as image:
            pattern = np.asarray(image.convert("RGB"), dtype=np.uint8)
    except (UnidentifiedImageError, OSError) as error:
        raise OSError(
            f"Unable to load pattern image: {pattern_path}"
        ) from error

    _validate_rgb_pattern(pattern)
    return pattern.copy()


def resize_pattern(
    pattern: np.ndarray,
    target_shape: tuple[int, int, int],
) -> np.ndarray:
    """Resize an RGB pattern to the height and width of a target image.

    Nearest-neighbour resampling preserves existing pattern pixel values. As a
    result, the returned pattern remains ``uint8`` and within the ``0`` to
    ``255`` pixel range. The target shape must be a three-channel image shape
    in ``(height, width, channels)`` form; only a channel count of three is
    accepted.

    Args:
        pattern: RGB NumPy array with `dtype=np.uint8` and shape
            `(H, W, 3)`.
        target_shape: Target image shape as ``(height, width, 3)``.

    Returns:
        A new RGB ``uint8`` NumPy array with shape ``target_shape``.

    Raises:
        TypeError: If ``pattern`` is not a NumPy array.
        ValueError: If the pattern or target shape is not a valid RGB image
            shape, the pattern has the wrong dtype, or target dimensions are
            not positive.
    """
    _validate_rgb_pattern(pattern)
    target_height, target_width = _validate_target_shape(target_shape)

    if pattern.shape[:2] == (target_height, target_width):
        return pattern.copy()

    pattern_image = Image.fromarray(pattern, mode="RGB")
    resized = pattern_image.resize(
        (target_width, target_height),
        resample=Image.Resampling.NEAREST,
    )
    resized_pattern = np.asarray(resized, dtype=np.uint8)
    _validate_rgb_pattern(resized_pattern)
    return resized_pattern.copy()


def get_pattern(
    path: str | Path,
    target_shape: tuple[int, int, int],
) -> np.ndarray:
    """Load a pattern and resize it to a target image shape.

    This convenience function composes :func:`load_pattern` and
    :func:`resize_pattern`. It performs no caching, attack application, file
    output, or metadata creation.

    Args:
        path: Path to a PNG, JPG, or JPEG pattern image.
        target_shape: Target image shape as ``(height, width, 3)``.

    Returns:
        An RGB ``uint8`` NumPy array with shape ``target_shape``.

    Raises:
        FileNotFoundError: If the pattern image does not exist.
        OSError: If the pattern image cannot be decoded.
        TypeError: If the loaded pattern is not a NumPy array.
        ValueError: If the file format, pattern, or target shape is invalid.
    """
    return resize_pattern(load_pattern(path), target_shape)


def _validate_rgb_pattern(pattern: np.ndarray) -> None:
    """Raise a descriptive error unless ``pattern`` is a uint8 RGB array."""
    if not isinstance(pattern, np.ndarray):
        raise TypeError("Pattern must be a NumPy array.")
    if pattern.dtype != np.uint8:
        raise ValueError("Pattern dtype must be np.uint8.")
    if pattern.ndim != 3 or pattern.shape[2] != 3:
        raise ValueError("Pattern must have shape (height, width, 3).")
    if pattern.shape[0] <= 0 or pattern.shape[1] <= 0:
        raise ValueError("Pattern height and width must be positive.")


def _validate_target_shape(
    target_shape: tuple[int, int, int],
) -> tuple[int, int]:
    """Validate a target shape and return its height and width."""
    if len(target_shape) != 3:
        raise ValueError("Target shape must be a three-element tuple.")

    target_height, target_width, target_channels = target_shape
    if not all(isinstance(value, int) for value in target_shape):
        raise ValueError("Target shape values must be integers.")
    if target_height <= 0 or target_width <= 0:
        raise ValueError("Target height and width must be positive.")
    if target_channels != 3:
        raise ValueError(
            "Target shape must specify exactly three RGB channels."
        )

    return target_height, target_width
