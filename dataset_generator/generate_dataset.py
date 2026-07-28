"""Command-line orchestration for attacked-image dataset generation."""

from __future__ import annotations

import argparse
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import yaml
from tqdm import tqdm

from .attack_pipeline import attack_image
from .image_loader import discover_images, load_image
from .metadata import create_metadata_record, write_metadata_record
from .output_writer import save_attacked_image
from .pattern_manager import get_pattern


LOGGER = logging.getLogger(__name__)
_PATTERN_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg"})


@dataclass(frozen=True)
class GeneratorConfig:
    """Validated configuration for one dataset-generation run."""

    input_dataset: Path
    output_dataset: Path
    metadata_location: Path
    pattern_directory: Path
    output_image_format: str
    overwrite: bool
    recursive_scanning: bool
    random_pattern_selection: bool
    random_seed: int | None


@dataclass(frozen=True)
class ProcessingSummary:
    """Final counts emitted by :func:`run_generation`."""

    discovered: int
    successful: int
    failed: int


def load_config(path: str | Path) -> GeneratorConfig:
    """Read and validate a dataset-generator YAML configuration file.

    Relative paths in the YAML file are resolved relative to the configuration
    file itself, which allows the command to run from any working directory.
    """
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Configuration file does not exist: {config_path}"
        )

    with config_path.open("r", encoding="utf-8") as config_file:
        config_data = yaml.safe_load(config_file) or {}

    if not isinstance(config_data, dict):
        raise ValueError("Generator configuration must be a YAML mapping.")

    config_directory = config_path.parent
    input_dataset = _resolve_config_path(
        config_data, "input_dataset", config_directory
    )
    output_dataset = _resolve_config_path(
        config_data, "output_dataset", config_directory
    )
    metadata_location = _resolve_config_path(
        config_data, "metadata_location", config_directory
    )
    pattern_directory = _resolve_config_path(
        config_data, "pattern_directory", config_directory
    )
    output_image_format = str(
        config_data.get("output_image_format", "png")
    ).lower()
    if output_image_format not in {"png", "jpeg", "jpg"}:
        raise ValueError("output_image_format must be png, jpeg, or jpg.")

    config = GeneratorConfig(
        input_dataset=input_dataset,
        output_dataset=output_dataset,
        metadata_location=metadata_location,
        pattern_directory=pattern_directory,
        output_image_format=output_image_format,
        overwrite=_read_bool(config_data, "overwrite", False),
        recursive_scanning=_read_bool(config_data, "recursive_scanning", True),
        random_pattern_selection=_read_bool(
            config_data, "random_pattern_selection", False
        ),
        random_seed=_read_optional_seed(config_data),
    )
    _validate_config_paths(config)
    return config


def run_generation(config: GeneratorConfig) -> ProcessingSummary:
    """Generate attacked images and append one CSV metadata record per image.

    A failure while processing one image is logged and recorded in metadata;
    remaining images continue processing.
    """
    image_paths = discover_images(config.input_dataset)
    if not config.recursive_scanning:
        image_paths = [
            path for path in image_paths if path.parent == Path(".")
        ]

    pattern_paths = _discover_pattern_paths(config.pattern_directory)
    if not pattern_paths:
        raise ValueError(
            "No PNG, JPG, or JPEG pattern files were found in "
            f"{config.pattern_directory}."
        )

    random_source = random.Random(config.random_seed)
    successful = 0
    failed = 0

    for index, relative_path in enumerate(
        tqdm(image_paths, desc="Generating attacked images", unit="image")
    ):
        pattern_path = _select_pattern(
            pattern_paths,
            index,
            config.random_pattern_selection,
            random_source,
        )
        source_path = config.input_dataset / relative_path
        image_width = 0
        image_height = 0

        try:
            image = load_image(source_path)
            if image is None:
                raise ValueError("Image could not be decoded as RGB.")

            image_height, image_width = image.shape[:2]
            pattern = get_pattern(pattern_path, image.shape)
            attacked_image = attack_image(image, pattern)
            saved_path = save_attacked_image(
                attacked_image,
                relative_path,
                config.output_dataset,
                config.output_image_format,
                config.overwrite,
            )
            write_metadata_record(
                config.metadata_location,
                create_metadata_record(
                    image_id=_image_id(relative_path),
                    original_relative_path=relative_path,
                    attacked_relative_path=saved_path.relative_to(
                        config.output_dataset
                    ),
                    pattern_used=pattern_path.relative_to(
                        config.pattern_directory
                    ),
                    image_width=image_width,
                    image_height=image_height,
                ),
            )
            successful += 1
        except (OSError, TypeError, ValueError) as error:
            failed += 1
            LOGGER.error("Failed to process %s: %s", relative_path, error)
            _write_failure_record(
                config,
                relative_path,
                pattern_path,
                image_width,
                image_height,
                error,
            )

    summary = ProcessingSummary(
        discovered=len(image_paths),
        successful=successful,
        failed=failed,
    )
    print(
        "Processing summary: "
        f"discovered={summary.discovered}, "
        f"successful={summary.successful}, failed={summary.failed}"
    )
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    """Run dataset generation from ``config/generator.yaml`` by default."""
    parser = argparse.ArgumentParser(
        description="Generate an attacked RGB image dataset."
    )
    parser.add_argument(
        "--config",
        default="config/generator.yaml",
        help="Path to the generator YAML configuration file.",
    )
    arguments = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    try:
        summary = run_generation(load_config(arguments.config))
    except (FileNotFoundError, OSError, ValueError, yaml.YAMLError) as error:
        LOGGER.error("Dataset generation could not start: %s", error)
        return 1

    return 0 if summary.failed == 0 else 2


def _resolve_config_path(
    config_data: dict[str, Any],
    field_name: str,
    config_directory: Path,
) -> Path:
    """Read and resolve a required path setting from the YAML file."""
    value = config_data.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty path string.")

    path = Path(value)
    return path if path.is_absolute() else (config_directory / path).resolve()


def _read_bool(
    config_data: dict[str, Any],
    field_name: str,
    default: bool,
) -> bool:
    """Read a boolean setting without accepting ambiguous truthy values."""
    value = config_data.get(field_name, default)
    if not isinstance(value, bool):
        raise ValueError(f"{field_name} must be true or false.")
    return value


def _read_optional_seed(config_data: dict[str, Any]) -> int | None:
    """Read an optional integer seed used for random pattern selection."""
    value = config_data.get("random_seed")
    if value is None:
        return None
    if not isinstance(value, int):
        raise ValueError("random_seed must be an integer or null.")
    return value


def _validate_config_paths(config: GeneratorConfig) -> None:
    """Validate roots and prevent generated output from entering raw data."""
    if not config.input_dataset.is_dir():
        raise FileNotFoundError(
            f"Input dataset directory does not exist: {config.input_dataset}"
        )
    if not config.pattern_directory.is_dir():
        raise FileNotFoundError(
            f"Pattern directory does not exist: {config.pattern_directory}"
        )

    input_root = config.input_dataset.resolve()
    output_root = config.output_dataset.resolve()
    if output_root == input_root or input_root in output_root.parents:
        raise ValueError(
            "output_dataset must not be the input dataset or one of its "
            "children."
        )


def _discover_pattern_paths(pattern_directory: Path) -> list[Path]:
    """Return supported pattern files in deterministic relative-path order."""
    return sorted(
        (
            path
            for path in pattern_directory.rglob("*")
            if path.is_file() and path.suffix.lower() in _PATTERN_EXTENSIONS
        ),
        key=lambda path: path.relative_to(pattern_directory).as_posix(),
    )


def _select_pattern(
    pattern_paths: list[Path],
    image_index: int,
    random_selection: bool,
    random_source: random.Random,
) -> Path:
    """Select a random or deterministic round-robin pattern for one image."""
    if random_selection:
        return random_source.choice(pattern_paths)
    return pattern_paths[image_index % len(pattern_paths)]


def _image_id(relative_path: Path) -> str:
    """Return a stable, portable source-image identifier."""
    return relative_path.with_suffix("").as_posix()


def _write_failure_record(
    config: GeneratorConfig,
    relative_path: Path,
    pattern_path: Path,
    image_width: int,
    image_height: int,
    error: Exception,
) -> None:
    """Record a failure while allowing the rest of the run to continue."""
    try:
        write_metadata_record(
            config.metadata_location,
            create_metadata_record(
                image_id=_image_id(relative_path),
                original_relative_path=relative_path,
                pattern_used=pattern_path.relative_to(
                    config.pattern_directory
                ),
                image_width=image_width,
                image_height=image_height,
                processing_status="failed",
                error_message=str(error),
            ),
        )
    except OSError as metadata_error:
        LOGGER.error(
            "Could not record failure metadata for %s: %s",
            relative_path,
            metadata_error,
        )


if __name__ == "__main__":
    raise SystemExit(main())
