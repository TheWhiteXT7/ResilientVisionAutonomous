"""YOLO Training CLI for ResilientVisionAutonomous.

Trains a YOLO model on a prepared dataset, saves checkpoints, config, and logs.

Usage::

    python -m scripts.train_yolo \\
        --data outputs/yolo_dataset/data.yaml \\
        --model yolov8n.pt \\
        --epochs 50 \\
        --batch-size 16 \\
        --name baseline \\
        --project outputs/experiments

    # Resume from last checkpoint:
    python -m scripts.train_yolo --data data.yaml --resume \\
        --resume-weights outputs/experiments/baseline/weights/last.pt
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

# ---------------------------------------------------------------------------
# Project root discovery — allows running as `python -m scripts.train_yolo`
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config.paths import OUTPUTS_DIR
from models.trainer import YoloTrainer
from models.yolo_config import YoloConfig
from models.yolo_wrapper import YoloWrapper

logger = logging.getLogger(__name__)

_SEPARATOR = "=" * 60


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def _configure_logging(verbose: bool) -> None:
    """Configure root logger with timestamped console handler.

    Args:
        verbose: If True, set log level to DEBUG; otherwise INFO.
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# ---------------------------------------------------------------------------
# Reproducibility helpers
# ---------------------------------------------------------------------------


def _set_random_seeds(seed: int) -> None:
    """Set random seeds for Python, NumPy, and PyTorch for reproducibility.

    Args:
        seed: Integer seed value to apply across all RNG sources.
    """
    random.seed(seed)
    logger.debug("Python random seed set to %d", seed)

    try:
        import numpy as np  # type: ignore

        np.random.seed(seed)
        logger.debug("NumPy random seed set to %d", seed)
    except ImportError:
        logger.debug("NumPy not available; skipping NumPy seed.")

    try:
        import torch  # type: ignore

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        logger.debug("PyTorch random seed set to %d", seed)
    except ImportError:
        logger.debug("PyTorch not available; skipping torch seed.")


def _collect_system_info(seed: int) -> Dict[str, Any]:
    """Collect environment metadata for reproducibility bookkeeping.

    Args:
        seed: Random seed used for this run.

    Returns:
        Dictionary containing versions, hardware info, git hash, and seed.
    """
    info: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "python_version": sys.version,
        "random_seed": seed,
    }

    # PyTorch / CUDA
    try:
        import torch  # type: ignore

        info["torch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["cuda_version"] = torch.version.cuda  # type: ignore[attr-defined]
            info["gpu_name"] = torch.cuda.get_device_name(0)
        else:
            info["cuda_version"] = "N/A"
            info["gpu_name"] = "N/A"
    except ImportError:
        info["torch_version"] = "not installed"
        info["cuda_available"] = False

    # Ultralytics version
    try:
        import ultralytics  # type: ignore

        info["ultralytics_version"] = ultralytics.__version__
    except ImportError:
        info["ultralytics_version"] = "not installed"

    # Git commit hash
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(_PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=5,
        )
        info["git_commit"] = result.stdout.strip() if result.returncode == 0 else "N/A"
    except Exception:
        info["git_commit"] = "N/A"

    return info


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    """Build and return the argument parser for the training CLI.

    Returns:
        Configured ArgumentParser instance.
    """
    parser = argparse.ArgumentParser(
        prog="train_yolo",
        description="Train a YOLO model for ResilientVisionAutonomous.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to data.yaml dataset configuration file.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="YOLO model name or path to pretrained weights (e.g. 'yolov8n.pt').",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        dest="batch_size",
        help="Training batch size.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Training image size (square, in pixels).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Compute device: 'cpu', 'cuda', or GPU index (e.g. '0').",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of data loader worker processes.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        help="Initial learning rate.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from a previous checkpoint.",
    )
    parser.add_argument(
        "--resume-weights",
        type=str,
        default=None,
        dest="resume_weights",
        help="Path to checkpoint weights for resuming (defaults to last.pt).",
    )
    parser.add_argument(
        "--project",
        type=str,
        default=str(OUTPUTS_DIR / "experiments"),
        help="Parent directory for experiment output.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="baseline",
        help="Experiment name / sub-directory under --project.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--save-json",
        action="store_true",
        dest="save_json",
        help="Save training configuration and results as JSON files.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )
    return parser


# ---------------------------------------------------------------------------
# Core training logic
# ---------------------------------------------------------------------------


def _build_config(args: argparse.Namespace) -> YoloConfig:
    """Construct a YoloConfig from parsed CLI arguments.

    Args:
        args: Parsed argument namespace from argparse.

    Returns:
        Immutable YoloConfig instance.
    """
    return YoloConfig(
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        image_size=args.imgsz,
        learning_rate=args.lr,
        device=args.device,
        project_directory=Path(args.project),
        experiment_name=args.name,
        verbose=args.verbose,
    )


def _save_json(path: Path, data: Dict[str, Any]) -> None:
    """Serialise *data* to *path* as indented JSON.

    Args:
        path: Destination file path.
        data: JSON-serialisable dictionary to write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=4, default=str)
    logger.info("Saved JSON → %s", path)


def _print_summary(summary: Dict[str, Any], config: YoloConfig) -> None:
    """Print a human-readable training summary to stdout.

    Args:
        summary: Training result dictionary returned by YoloTrainer.train().
        config: YoloConfig used for this run.
    """
    print(f"\n{_SEPARATOR}")
    print("  TRAINING SUMMARY")
    print(_SEPARATOR)
    print(f"  Status        : {summary.get('status', 'unknown')}")
    print(f"  Experiment    : {config.experiment_name}")
    print(f"  Epochs        : {config.epochs}")
    print(f"  Batch size    : {config.batch_size}")
    print(f"  Image size    : {config.image_size}")
    print(f"  Device        : {config.device}")
    print(f"  Experiment dir: {summary.get('experiment_dir', 'N/A')}")
    print(f"  Best weights  : {summary.get('best_weights', 'N/A')}")
    if "metrics" in summary:
        print("  Metrics:")
        for k, v in summary["metrics"].items():
            print(f"    {k}: {v}")
    print(_SEPARATOR)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Main entry point for the YOLO training CLI."""
    parser = _build_parser()
    args = parser.parse_args()

    _configure_logging(args.verbose)
    logger.info("%s", _SEPARATOR)
    logger.info("  ResilientVisionAutonomous — YOLO Training")
    logger.info("%s", _SEPARATOR)

    # Validate data.yaml
    data_path = Path(args.data)
    if not data_path.exists():
        logger.error("data.yaml not found: %s", data_path)
        sys.exit(1)

    # Reproducibility
    _set_random_seeds(args.seed)
    system_info = _collect_system_info(args.seed)
    logger.info("System info collected: torch=%s, YOLO=%s, git=%s",
                system_info.get("torch_version"),
                system_info.get("ultralytics_version"),
                system_info.get("git_commit"))

    # Build config and trainer
    config = _build_config(args)
    logger.info("YoloConfig: model=%s, epochs=%d, device=%s",
                config.model_name, config.epochs, config.device)

    try:
        wrapper = YoloWrapper(config=config)
        trainer = YoloTrainer(wrapper=wrapper, config=config)
    except RuntimeError as exc:
        logger.error("Failed to initialise model: %s", exc)
        sys.exit(1)

    # Output directories
    exp_dir = Path(args.project) / args.name
    exp_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = exp_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Optionally persist config JSON
    config_payload: Dict[str, Any] = {
        "yolo_config": config.to_dict(),
        "system_info": system_info,
        "cli_args": vars(args),
    }
    if args.save_json:
        _save_json(exp_dir / "config.json", config_payload)

    # Train or resume
    try:
        if args.resume:
            checkpoint: Optional[str] = args.resume_weights
            logger.info("Resuming training from checkpoint: %s",
                        checkpoint or trainer.last_weights_path)
            summary = trainer.resume(checkpoint_path=checkpoint)
        else:
            logger.info("Starting training on data: %s", data_path)
            summary = trainer.train(dataset=None, data_yaml_path=data_path)
    except (RuntimeError, FileNotFoundError) as exc:
        logger.error("Training failed: %s", exc)
        sys.exit(1)

    # Save summary / logs
    if args.save_json:
        _save_json(logs_dir / "training_log.json", summary)

    # Copy best weights to experiment root
    try:
        trainer.save_best(target_dir=exp_dir)
        logger.info("Best checkpoint copied to %s/best.pt", exp_dir)
    except FileNotFoundError:
        logger.warning("best.pt not found; skipping checkpoint copy.")

    _print_summary(summary, config)
    logger.info("Training complete.")


if __name__ == "__main__":
    main()
