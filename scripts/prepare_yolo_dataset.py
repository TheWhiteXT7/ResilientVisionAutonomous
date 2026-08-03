"""KITTI → Ultralytics YOLO dataset preparation integration script.

Validates the KITTI dataset, generates reproducible train/val splits via
SplitManager, converts KITTI labels to YOLO format, and writes the complete
YOLO directory structure (images/, labels/, data.yaml) into
``outputs/yolo_dataset/``.

Usage::

    python -m scripts.prepare_yolo_dataset [OPTIONS]

    Options:
        --overwrite         Remove existing output directory and regenerate.
        --seed INT          Random seed for reproducible split generation.
                            (default: 42)
        --train-ratio FLOAT Fraction of samples assigned to training split.
                            (default: 0.8)
        --kitti-dir PATH    Override path to KITTI dataset root directory.
        --output-dir PATH   Override path to YOLO output directory.
                            (default: outputs/yolo_dataset)
        --verbose           Set log level to DEBUG.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Project-root discovery (allows running as `python -m scripts.xxx` from
# any working directory).
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config.paths import KITTI_DIR, OUTPUTS_DIR
from dataset_loader import DatasetValidator, KittiLoader, SplitManager
from models.utils import get_default_class_mapping, prepare_yolo_dataset

# ---------------------------------------------------------------------------
# Module-level logger — handlers attached by _configure_logging().
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

_SEPARATOR = "=" * 50


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def _configure_logging(verbose: bool) -> None:
    """Configure root logger with a timestamped console handler.

    Args:
        verbose: When True, set level to DEBUG; otherwise INFO.
    """
    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s [%(levelname)-8s] %(name)s: %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    logging.basicConfig(level=level, format=fmt, datefmt=datefmt)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument list. Defaults to ``sys.argv[1:]``.

    Returns:
        Populated :class:`argparse.Namespace` object.
    """
    parser = argparse.ArgumentParser(
        prog="prepare_yolo_dataset",
        description=(
            "Validate KITTI and prepare an Ultralytics-compatible YOLO "
            "dataset under outputs/yolo_dataset/."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Delete the existing output directory before regenerating.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        metavar="INT",
        help="Random seed for reproducible train/val split generation.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        dest="train_ratio",
        metavar="FLOAT",
        help="Fraction of samples assigned to the training split (0.0–1.0).",
    )
    parser.add_argument(
        "--kitti-dir",
        type=Path,
        default=None,
        dest="kitti_dir",
        metavar="PATH",
        help="Override default KITTI dataset root directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        dest="output_dir",
        metavar="PATH",
        help="Override default YOLO output directory.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Enable DEBUG-level logging.",
    )

    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _validate_args(args: argparse.Namespace) -> None:
    """Validate parsed argument values.

    Args:
        args: Parsed arguments.

    Raises:
        SystemExit: If any argument value is out of range.
    """
    if not 0.0 < args.train_ratio < 1.0:
        logger.error(
            "--train-ratio must be strictly between 0.0 and 1.0, "
            f"got {args.train_ratio}"
        )
        sys.exit(1)


def _validate_kitti(kitti_dir: Path) -> dict:
    """Run DatasetValidator and log the validation report.

    Args:
        kitti_dir: Root directory of the KITTI dataset.

    Returns:
        Validation report dictionary produced by
        :meth:`DatasetValidator.validate`.

    Raises:
        SystemExit: If the dataset root directory does not exist.
    """
    if not kitti_dir.exists():
        logger.error(
            f"KITTI dataset directory does not exist: {kitti_dir}\n"
            "  → Download KITTI from http://www.cvlibs.net/datasets/kitti/ "
            "and place it at the path above, or supply --kitti-dir."
        )
        sys.exit(1)

    logger.info(f"Validating KITTI dataset at: {kitti_dir}")
    validator = DatasetValidator(kitti_dir=kitti_dir)
    report = validator.validate()

    if report["errors"]:
        for err in report["errors"]:
            logger.error(f"  [VALIDATION ERROR] {err}")

    if report["warnings"]:
        for warn in report["warnings"]:
            logger.warning(f"  [VALIDATION WARNING] {warn}")

    if not report["structure_valid"]:
        logger.error(
            "KITTI structure validation failed. "
            "Resolve the errors above before re-running."
        )
        sys.exit(1)

    if report["missing_labels_count"] > 0:
        logger.warning(
            f"{report['missing_labels_count']} image(s) are missing "
            "corresponding label files and will be skipped."
        )

    logger.info(
        f"KITTI validation passed — "
        f"train images: {report['num_train_images']}, "
        f"train labels: {report['num_train_labels']}, "
        f"test images:  {report['num_test_images']}"
    )
    return report


# ---------------------------------------------------------------------------
# Output directory management
# ---------------------------------------------------------------------------


def _manage_output_directory(output_dir: Path, overwrite: bool) -> None:
    """Ensure the output directory is ready for population.

    If the directory already exists and ``--overwrite`` is set, it is
    removed entirely.  Without ``--overwrite`` the script aborts so
    the user can make an explicit decision.

    Args:
        output_dir: Target YOLO output directory.
        overwrite: Whether to remove an existing directory automatically.

    Raises:
        SystemExit: If the directory exists and ``overwrite`` is False,
            or if removal fails due to a permission error.
    """
    if not output_dir.exists():
        return

    if not overwrite:
        logger.error(
            f"Output directory already exists: {output_dir}\n"
            "  → Pass --overwrite to replace it automatically, or remove it "
            "manually."
        )
        sys.exit(1)

    logger.info(f"--overwrite supplied — removing existing directory: {output_dir}")
    try:
        shutil.rmtree(output_dir)
        logger.debug(f"Removed: {output_dir}")
    except PermissionError as exc:
        logger.error(
            f"Permission denied while trying to remove {output_dir}: {exc}"
        )
        sys.exit(1)


# ---------------------------------------------------------------------------
# Split counters (read back from generated ImageSets for the summary)
# ---------------------------------------------------------------------------


def _count_split(output_dir: Path, split_name: str) -> int:
    """Count files in a generated images/<split> subdirectory.

    Args:
        output_dir: Root of the generated YOLO dataset.
        split_name: Split name (``'train'``, ``'val'``, ``'test'``).

    Returns:
        Number of image files found, or 0 if the directory does not exist.
    """
    split_dir = output_dir / "images" / split_name
    if not split_dir.exists():
        return 0
    return sum(
        1
        for f in split_dir.iterdir()
        if f.is_file() and f.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}
    )


# ---------------------------------------------------------------------------
# Output verification
# ---------------------------------------------------------------------------


def _verify_outputs(output_dir: Path, yaml_path: Path) -> bool:
    """Verify that all required YOLO output files and directories exist.

    Args:
        output_dir: Root of the generated YOLO dataset.
        yaml_path: Path to the generated ``data.yaml`` file.

    Returns:
        True if every expected artefact is present; False otherwise.
    """
    required: list[Path] = [
        output_dir / "images" / "train",
        output_dir / "images" / "val",
        output_dir / "labels" / "train",
        output_dir / "labels" / "val",
        yaml_path,
    ]

    all_ok = True
    for path in required:
        if path.exists():
            logger.info(f"  [OK] {path.relative_to(output_dir.parent)}")
        else:
            logger.error(f"  [MISSING] {path.relative_to(output_dir.parent)}")
            all_ok = False

    return all_ok


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------


def _print_summary(
    kitti_dir: Path,
    total_samples: int,
    train_count: int,
    val_count: int,
    test_count: int,
    output_dir: Path,
    yaml_path: Path,
) -> None:
    """Print a structured dataset preparation summary to stdout.

    Args:
        kitti_dir: KITTI root directory that was used.
        total_samples: Total number of training samples discovered.
        train_count: Number of samples assigned to training split.
        val_count: Number of samples assigned to validation split.
        test_count: Number of samples assigned to test split (0 if absent).
        output_dir: Root of the generated YOLO dataset.
        yaml_path: Path to the generated ``data.yaml`` file.
    """
    lines = [
        "",
        _SEPARATOR,
        "YOLO DATASET PREPARATION SUMMARY",
        _SEPARATOR,
        "",
        f"  Dataset Root:           {kitti_dir}",
        f"  Total Samples:          {total_samples}",
        f"  Training Samples:       {train_count}",
        f"  Validation Samples:     {val_count}",
        f"  Test Samples:           {test_count}",
        "",
        f"  YOLO Dataset Directory: {output_dir}",
        f"  Generated data.yaml:    {yaml_path}",
        "",
        _SEPARATOR,
        "",
    ]
    print("\n".join(lines))


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run(argv: Optional[list[str]] = None) -> None:
    """Execute the full KITTI → YOLO dataset preparation pipeline.

    Steps:
        1. Parse and validate CLI arguments.
        2. Validate the KITTI dataset structure.
        3. Load all train-split samples via KittiLoader.
        4. Manage the output directory (create or overwrite).
        5. Call ``prepare_yolo_dataset()`` which invokes SplitManager
           internally to generate or load train/val splits.
        6. Verify generated artefacts.
        7. Print summary.

    Args:
        argv: Optional argument list override (useful in tests).

    Raises:
        SystemExit: On any unrecoverable error.
    """
    args = _parse_args(argv)
    _configure_logging(args.verbose)

    # -------------------------------------------------------------------
    # Resolve paths
    # -------------------------------------------------------------------
    kitti_dir: Path = args.kitti_dir if args.kitti_dir else KITTI_DIR
    output_dir: Path = (
        args.output_dir if args.output_dir else OUTPUTS_DIR / "yolo_dataset"
    )
    val_ratio: float = round(1.0 - args.train_ratio, 10)

    _validate_args(args)

    logger.info(_SEPARATOR)
    logger.info("ResilientVisionAutonomous — YOLO Dataset Preparation")
    logger.info(_SEPARATOR)
    logger.info(f"KITTI directory  : {kitti_dir}")
    logger.info(f"Output directory : {output_dir}")
    logger.info(f"Train ratio      : {args.train_ratio}")
    logger.info(f"Val ratio        : {val_ratio}")
    logger.info(f"Random seed      : {args.seed}")

    # -------------------------------------------------------------------
    # Step 1: Validate KITTI
    # -------------------------------------------------------------------
    validation_report = _validate_kitti(kitti_dir)

    # -------------------------------------------------------------------
    # Step 2: Load dataset via KittiLoader (trainval split to get all
    #         annotated samples; KittiLoader handles split file lookup).
    # -------------------------------------------------------------------
    logger.info("Loading KITTI training samples via KittiLoader…")
    try:
        loader = KittiLoader(
            kitti_dir=kitti_dir,
            split="trainval",
            validate=False,   # Already validated above
        )
    except Exception:
        # 'trainval' may not exist if no split files are present; fall back
        # to discovering all images in the training directory.
        logger.debug(
            "KittiLoader 'trainval' split unavailable; falling back to 'train'."
        )
        try:
            loader = KittiLoader(
                kitti_dir=kitti_dir,
                split="train",
                val_ratio=val_ratio,
                seed=args.seed,
                validate=False,
            )
        except Exception as exc:
            logger.error(f"Failed to initialise KittiLoader: {exc}")
            sys.exit(1)

    total_samples = len(loader)
    if total_samples == 0:
        logger.error(
            "KittiLoader returned 0 samples. "
            "Check that the KITTI image directory is non-empty."
        )
        sys.exit(1)

    logger.info(f"Loaded {total_samples} samples from KittiLoader.")

    # -------------------------------------------------------------------
    # Step 3: Manage output directory
    # -------------------------------------------------------------------
    _manage_output_directory(output_dir, overwrite=args.overwrite)

    # -------------------------------------------------------------------
    # Step 4: Prepare YOLO dataset
    #         prepare_yolo_dataset() integrates SplitManager internally.
    # -------------------------------------------------------------------
    logger.info("Generating YOLO dataset structure…")
    class_mapping = get_default_class_mapping()
    try:
        yaml_path = prepare_yolo_dataset(
            dataset=loader,
            output_dir=output_dir,
            class_mapping=class_mapping,
            val_ratio=val_ratio,
            seed=args.seed,
        )
    except PermissionError as exc:
        logger.error(f"Permission denied while writing YOLO dataset: {exc}")
        sys.exit(1)
    except Exception as exc:
        logger.error(f"Dataset preparation failed unexpectedly: {exc}", exc_info=True)
        sys.exit(1)

    logger.info(f"data.yaml written to: {yaml_path}")

    # -------------------------------------------------------------------
    # Step 5: Count per-split samples from generated directories
    # -------------------------------------------------------------------
    train_count = _count_split(output_dir, "train")
    val_count = _count_split(output_dir, "val")
    test_count = _count_split(output_dir, "test")

    # -------------------------------------------------------------------
    # Step 6: Verify outputs
    # -------------------------------------------------------------------
    logger.info("Verifying generated output structure…")
    success = _verify_outputs(output_dir, yaml_path)

    # -------------------------------------------------------------------
    # Step 7: Print summary
    # -------------------------------------------------------------------
    _print_summary(
        kitti_dir=kitti_dir,
        total_samples=total_samples,
        train_count=train_count,
        val_count=val_count,
        test_count=test_count,
        output_dir=output_dir,
        yaml_path=yaml_path,
    )

    if success:
        logger.info(
            "Dataset preparation complete. "
            "Run YOLO training with:\n"
            f"  yolo detect train data={yaml_path} model=yolov8n.pt"
        )
    else:
        logger.error(
            "One or more expected output artefacts are missing. "
            "Check the errors logged above."
        )
        sys.exit(1)


if __name__ == "__main__":
    run()
