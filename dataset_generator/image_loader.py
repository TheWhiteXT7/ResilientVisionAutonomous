"""Image discovery and RGB loading utilities for source datasets.

This module is deliberately limited to locating supported image files and
loading individual files as RGB NumPy arrays. It does not resize, modify,
attack, save, or otherwise transform source images.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError


SUPPORTED_IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".bmp"})
"""File extensions accepted during recursive image discovery."""


def discover_images(root_directory: str | Path) -> list[Path]:
    """Recursively discover valid image files beneath a dataset directory.

    The returned paths are relative to ``root_directory``. This preserves the
    source dataset hierarchy so callers can construct output paths without
    losing split or nested-directory information. For example, an image at
    ``<root>/training/image_2/000123.png`` is returned as
    ``Path("training/image_2/000123.png")``.

    Only PNG, JPG, JPEG, and BMP files are considered. Every candidate is
    opened through :func:`load_image`; unreadable or corrupted files are
    silently skipped. The returned list is sorted by normalized relative path,
    making discovery deterministic across file-system enumeration orders.

    Args:
        root_directory: Existing directory at the root of the source dataset.

    Returns:
        A sorted list of readable image paths relative to ``root_directory``.
        To load a returned path, pass ``Path(root_directory) / relative_path``
        to :func:`load_image`.

    Raises:
        FileNotFoundError: If ``root_directory`` does not exist.
        NotADirectoryError: If ``root_directory`` is not a directory.
    """
    root_path = Path(root_directory)

    if not root_path.exists():
        raise FileNotFoundError(
            f"Dataset directory does not exist: {root_path}"
        )
    if not root_path.is_dir():
        raise NotADirectoryError(
            f"Dataset root must be a directory: {root_path}"
        )

    discovered: list[Path] = []
    for path in root_path.rglob("*"):
        if (
            not path.is_file()
            or path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS
        ):
            continue

        if load_image(path) is not None:
            discovered.append(path.relative_to(root_path))

    return sorted(discovered, key=lambda path: path.as_posix())


def load_image(path: str | Path) -> np.ndarray | None:
    """Load one image as an RGB ``uint8`` NumPy array.

    The image is decoded with Pillow and converted to RGB. The returned array
    therefore always has shape ``(height, width, 3)`` and ``dtype=np.uint8``.
    No resizing, cropping, attack application, file writing, or metadata work
    occurs in this function.

    A corrupt, unreadable, missing, or unsupported image returns ``None`` so a
    directory scan can continue safely. Callers that need a strict policy can
    treat a `None` result as an image-load error. Valid images are returned
    as independent arrays and remain unchanged on disk.

    Args:
        path: Path to an image file. PNG, JPG, JPEG, and BMP are supported by
            discovery; Pillow determines whether a directly supplied path can
            be decoded.

    Returns:
        An RGB NumPy array with three channels, or `None` when the file
        cannot be decoded or does not produce a three-channel RGB array.
    """
    try:
        with Image.open(path) as image:
            rgb_image = image.convert("RGB")
            image_array = np.asarray(rgb_image, dtype=np.uint8)
    except (FileNotFoundError, OSError, UnidentifiedImageError, ValueError):
        return None

    if image_array.ndim != 3 or image_array.shape[2] != 3:
        return None

    return image_array.copy()
