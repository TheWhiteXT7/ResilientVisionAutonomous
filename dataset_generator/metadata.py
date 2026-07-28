"""CSV provenance records for generated dataset images."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


METADATA_FIELDS = (
    "image_id",
    "original_relative_path",
    "attacked_relative_path",
    "pattern_used",
    "image_width",
    "image_height",
    "processing_status",
    "error_message",
    "timestamp",
)
"""Column order for the generated ``metadata.csv`` file."""


@dataclass(frozen=True)
class MetadataRecord:
    """One success or failure entry in a generated dataset metadata file."""

    image_id: str
    original_relative_path: str
    attacked_relative_path: str
    pattern_used: str
    image_width: int
    image_height: int
    processing_status: str
    error_message: str
    timestamp: str


def create_metadata_record(
    image_id: str,
    original_relative_path: str | Path,
    attacked_relative_path: str | Path = "",
    pattern_used: str | Path = "",
    image_width: int = 0,
    image_height: int = 0,
    processing_status: str = "success",
    error_message: str = "",
) -> MetadataRecord:
    """Create a timestamped metadata record.

    Args:
        image_id: Stable identifier for the source image.
        original_relative_path: Source path relative to the input dataset root.
        attacked_relative_path: Generated path relative to the output root.
        pattern_used: Pattern path relative to the configured pattern
            directory.
        image_width: Source image width in pixels, or zero when unavailable.
        image_height: Source image height in pixels, or zero when unavailable.
        processing_status: ``"success"`` or ``"failed"``.
        error_message: Empty for successful records; failure context otherwise.

    Returns:
        A validated, UTC timestamped :class:`MetadataRecord`.

    Raises:
        ValueError: If required identifiers, dimensions, or status are invalid.
    """
    if not image_id:
        raise ValueError("image_id must not be empty.")
    if not str(original_relative_path):
        raise ValueError("original_relative_path must not be empty.")
    if processing_status not in {"success", "failed"}:
        raise ValueError("processing_status must be 'success' or 'failed'.")
    if image_width < 0 or image_height < 0:
        raise ValueError("Image dimensions cannot be negative.")

    return MetadataRecord(
        image_id=image_id,
        original_relative_path=Path(original_relative_path).as_posix(),
        attacked_relative_path=_as_posix_or_empty(attacked_relative_path),
        pattern_used=_as_posix_or_empty(pattern_used),
        image_width=image_width,
        image_height=image_height,
        processing_status=processing_status,
        error_message=error_message,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


def write_metadata_record(
    metadata_path: str | Path,
    record: MetadataRecord,
) -> Path:
    """Append one metadata record, creating its CSV file and header as needed.

    Args:
        metadata_path: Destination CSV file path.
        record: Complete record to append.

    Returns:
        The metadata CSV path.

    Raises:
        OSError: If the metadata location cannot be created or written.
    """
    destination = Path(metadata_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not destination.exists() or destination.stat().st_size == 0

    with destination.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=METADATA_FIELDS)
        if needs_header:
            writer.writeheader()
        writer.writerow(asdict(record))

    return destination


def _as_posix_or_empty(value: str | Path) -> str:
    """Normalize an optional path-like value for a portable CSV record."""
    return Path(value).as_posix() if str(value) else ""
