"""Safe persistence for generated attacked RGB images."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image


_FORMAT_DETAILS = {
    "png": (".png", "PNG"),
    "jpeg": (".jpg", "JPEG"),
    "jpg": (".jpg", "JPEG"),
}


def save_attacked_image(
    image: np.ndarray,
    original_relative_path: str | Path,
    output_directory: str | Path,
    output_format: str = "png",
    overwrite: bool = False,
) -> Path:
    """Save an attacked RGB image beneath an output root.

    The input's relative parent directories are recreated below
    ``output_directory``. The output suffix is selected from ``output_format``;
    a PNG source can therefore be written as JPEG without changing its source
    location. This function writes only beneath the explicit output root and
    never changes the raw input dataset.

    Args:
        image: RGB ``uint8`` array with shape ``(height, width, 3)``.
        original_relative_path: Source image path relative to its dataset root.
        output_directory: Root directory for generated images.
        output_format: Output image format: ``"png"``, ``"jpeg"``, or
            ``"jpg"``.
        overwrite: Whether an existing generated output may be replaced.

    Returns:
        The path of the committed generated image.

    Raises:
        TypeError: If ``image`` is not a NumPy array.
        ValueError: If the image, relative path, or output format is invalid.
        FileExistsError: If the output exists and ``overwrite`` is false.
        OSError: If the image cannot be written.
    """
    _validate_rgb_image(image)
    relative_path = _validate_relative_path(original_relative_path)
    suffix, pillow_format = _get_format_details(output_format)

    output_root = Path(output_directory)
    destination = output_root / relative_path.with_suffix(suffix)
    _ensure_within_output_root(destination, output_root)

    if destination.exists() and not overwrite:
        raise FileExistsError(f"Generated image already exists: {destination}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = _create_temporary_path(destination)

    try:
        Image.fromarray(image, mode="RGB").save(temporary_path, pillow_format)
        os.replace(temporary_path, destination)
    except OSError:
        temporary_path.unlink(missing_ok=True)
        raise

    return destination


def _get_format_details(output_format: str) -> tuple[str, str]:
    """Return the suffix and Pillow name for a supported output format."""
    normalized_format = output_format.lower().lstrip(".")
    try:
        return _FORMAT_DETAILS[normalized_format]
    except KeyError as error:
        raise ValueError(
            "Unsupported output format. Expected PNG or JPEG."
        ) from error


def _validate_rgb_image(image: np.ndarray) -> None:
    """Validate the array contract required by Pillow RGB encoding."""
    if not isinstance(image, np.ndarray):
        raise TypeError("Image must be a NumPy array.")
    if image.dtype != np.uint8:
        raise ValueError("Image dtype must be np.uint8.")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Image must have shape (height, width, 3).")
    if image.shape[0] <= 0 or image.shape[1] <= 0:
        raise ValueError("Image height and width must be positive.")


def _validate_relative_path(path: str | Path) -> Path:
    """Return a safe relative path or reject absolute and traversal paths."""
    relative_path = Path(path)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError(
            "Original path must be relative and cannot traverse up."
        )
    if not relative_path.parts or relative_path.name in {"", "."}:
        raise ValueError("Original relative path must name an image file.")
    return relative_path


def _ensure_within_output_root(destination: Path, output_root: Path) -> None:
    """Guard against path traversal before a generated file is created."""
    resolved_root = output_root.resolve()
    resolved_destination = destination.resolve()
    if resolved_destination != resolved_root and resolved_root not in (
        resolved_destination,
        *resolved_destination.parents,
    ):
        raise ValueError("Generated image path escapes the output directory.")


def _create_temporary_path(destination: Path) -> Path:
    """Create a closed temporary path beside its eventual destination."""
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.stem}_",
        suffix=destination.suffix,
        dir=destination.parent,
    )
    os.close(descriptor)
    return Path(temporary_name)
