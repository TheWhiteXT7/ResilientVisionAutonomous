"""Evaluation-only attack comparison CLI.

Evaluates a SINGLE frozen clean baseline YOLO model on the clean, random-attacked
and targeted-attacked validation populations and writes one JSON report with
explicit degradation comparisons. No training is performed and existing datasets
are never regenerated.

Example:
    python -m scripts.evaluate_attack_comparison \
        --weights outputs/experiments/baseline/<run>/checkpoints/best.pt \
        --clean-data outputs/yolo_dataset/data.yaml \
        --random-data outputs/experiments/random_attack/<run>/dataset_prepared/data.yaml \
        --target-data outputs/experiments/target_attack/<run>/dataset_prepared/data.yaml \
        --target-metadata outputs/experiments/target_attack/<run>/dataset_attacked/training/metadata
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Any, Optional

from config.paths import OUTPUTS_DIR
from evaluation.attack_evaluator import AttackComparisonConfig, AttackComparisonEvaluator

logger = logging.getLogger(__name__)

DEFAULT_EXPECTED_SAMPLES = 1497


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Evaluate a frozen clean baseline YOLO model on clean, "
        "random-attacked and targeted-attacked validation populations"
    )
    p.add_argument("--weights", required=True, help="Path to clean baseline weights (.pt)")
    p.add_argument("--clean-data", required=True, dest="clean_data", help="Clean validation data.yaml")
    p.add_argument("--random-data", required=True, dest="random_data", help="Random-attacked validation data.yaml")
    p.add_argument("--target-data", required=True, dest="target_data", help="Targeted-attacked validation data.yaml")
    p.add_argument(
        "--target-metadata",
        default=None,
        dest="target_metadata",
        help="Optional dir of targeted-attack per-sample metadata JSONs (target_found/preserved)",
    )
    p.add_argument(
        "--expected-samples",
        type=int,
        default=DEFAULT_EXPECTED_SAMPLES,
        help="Required sample count for every validation population (default %(default)s)",
    )
    p.add_argument("--device", default="cpu", help="Compute device: cpu, cuda, or a GPU index (default cpu)")
    p.add_argument(
        "--output",
        default=None,
        help="Output JSON report path (default outputs/experiments/attack_comparison/<timestamp>.json)",
    )
    return p


def main(argv: Any = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)

    weights = Path(args.weights)
    if not weights.exists():
        logger.error("Weights not found: %s", weights)
        return 2

    config = AttackComparisonConfig(
        weights=weights,
        clean_data_yaml=args.clean_data,
        random_data_yaml=args.random_data,
        target_data_yaml=args.target_data,
        expected_samples=args.expected_samples,
        device=args.device,
        target_metadata_dir=args.target_metadata,
    )

    output = (
        Path(args.output)
        if args.output
        else OUTPUTS_DIR
        / "experiments"
        / "attack_comparison"
        / f"{time.strftime('%Y%m%d_%H%M%S')}.json"
    )

    evaluator = AttackComparisonEvaluator(config)
    try:
        report = evaluator.evaluate()
        path = evaluator.save_report(report, output)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        logger.error("Attack comparison evaluation failed: %s", exc)
        return 2

    logger.info("Wrote attack comparison report to: %s", path)
    print(f"Wrote attack comparison report to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
