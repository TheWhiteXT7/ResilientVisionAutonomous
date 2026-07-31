"""Configuration dataclass for dataset generator options."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union


@dataclass(frozen=True)
class GeneratorConfig:
    """Immutable configuration for DatasetGenerator execution.

    Attributes:
        output_directory: Directory path where generated dataset will be saved.
        overwrite_existing: If True, overwrites existing generated files.
        save_metadata: If True, saves JSON metadata per sample and summary.
        save_original_copy: If True, copies original images alongside attacked images.
        copy_labels: If True, copies ground-truth label text files to output.
        copy_calibration: If True, copies calibration text files to output.
        workers: Number of worker processes/threads for execution (>= 1).
        batch_size: Number of samples per batch (>= 1).
        image_format: Image file extension format ('png', 'jpg', 'jpeg', 'bmp').
        metadata_format: Metadata serialization format ('json').
        logging_level: Logging severity level ('DEBUG', 'INFO', 'WARNING', 'ERROR').
    """

    output_directory: Union[Path, str] = Path("outputs/attacked_dataset")
    overwrite_existing: bool = False
    save_metadata: bool = True
    save_original_copy: bool = False
    copy_labels: bool = True
    copy_calibration: bool = True
    workers: int = 1
    batch_size: int = 32
    image_format: str = "png"
    metadata_format: str = "json"
    logging_level: str = "INFO"

    def __post_init__(self) -> None:
        """Validate configuration parameters upon dataclass initialization.

        Raises:
            TypeError: If an attribute has an incorrect type.
            ValueError: If an attribute has an out-of-range or invalid value.
        """
        # Validate output_directory
        if not isinstance(self.output_directory, (Path, str)):
            raise TypeError("output_directory must be a Path or str.")
        path_obj = Path(self.output_directory)
        if not str(path_obj).strip():
            raise ValueError("output_directory path cannot be empty.")
        object.__setattr__(self, "output_directory", path_obj)

        # Validate boolean flags
        bool_fields = {
            "overwrite_existing": self.overwrite_existing,
            "save_metadata": self.save_metadata,
            "save_original_copy": self.save_original_copy,
            "copy_labels": self.copy_labels,
            "copy_calibration": self.copy_calibration,
        }
        for name, value in bool_fields.items():
            if not isinstance(value, bool):
                raise TypeError(f"{name} must be a boolean, got {type(value).__name__}.")

        # Validate workers
        if isinstance(self.workers, bool) or not isinstance(self.workers, int):
            raise TypeError("workers must be an integer.")
        if self.workers <= 0:
            raise ValueError(f"workers must be greater than 0, got {self.workers}.")

        # Validate batch_size
        if isinstance(self.batch_size, bool) or not isinstance(self.batch_size, int):
            raise TypeError("batch_size must be an integer.")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be greater than 0, got {self.batch_size}.")

        # Validate image_format
        if not isinstance(self.image_format, str):
            raise TypeError("image_format must be a string.")
        fmt = self.image_format.strip().lower()
        if fmt not in ("png", "jpg", "jpeg", "bmp"):
            raise ValueError(f"Unsupported image_format '{self.image_format}'. Must be one of ('png', 'jpg', 'jpeg', 'bmp').")
        object.__setattr__(self, "image_format", fmt)

        # Validate metadata_format
        if not isinstance(self.metadata_format, str):
            raise TypeError("metadata_format must be a string.")
        meta_fmt = self.metadata_format.strip().lower()
        if meta_fmt not in ("json",):
            raise ValueError(f"Unsupported metadata_format '{self.metadata_format}'. Must be 'json'.")
        object.__setattr__(self, "metadata_format", meta_fmt)

        # Validate logging_level
        if not isinstance(self.logging_level, str):
            raise TypeError("logging_level must be a string.")
        lvl = self.logging_level.strip().upper()
        if lvl not in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
            raise ValueError(f"Invalid logging_level '{self.logging_level}'. Must be one of ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL').")
        object.__setattr__(self, "logging_level", lvl)
