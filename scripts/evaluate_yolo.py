"""Evaluate YOLO weights using the project's YoloEvaluator service."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from models.yolo_wrapper import YoloWrapper
from models.predictor import YoloPredictor
from models.evaluator import YoloEvaluator
from dataset_loader.kitti_loader import KittiLoader
from models.utils import prepare_yolo_dataset

logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate YOLO weights on dataset")
    p.add_argument("--weights", required=True, help="Path to model weights (.pt)")
    p.add_argument("--data", default=None, help="Path to data.yaml or dataset directory")
    p.add_argument("--save-json", action="store_true", dest="save_json", help="Save evaluation report as JSON")
    p.add_argument("--plots", action="store_true", help="Produce plots (optional, not implemented)" )
    return p


def main(argv: Any = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO)

    weights = Path(args.weights)
    if not weights.exists():
        logger.error("Weights not found: %s", weights)
        return 2

    # Determine dataset source
    data_arg = args.data
    dataset = None
    if data_arg:
        data_p = Path(data_arg)
        if data_p.is_file() and data_p.suffix in (".yaml", ".yml"):
            # data.yaml contains path to dataset
            import yaml

            with open(data_p, "r", encoding="utf-8") as f:
                y = yaml.safe_load(f)
            base_path = Path(y.get("path", "."))
            dataset = KittiLoader(kitti_dir=base_path, split="val", load_images=False)
        elif data_p.is_dir():
            dataset = KittiLoader(kitti_dir=data_p, split="val", load_images=False)
        else:
            logger.error("Unsupported --data argument: %s", data_arg)
            return 2
    else:
        logger.info("No --data provided; using default KITTI val split")
        dataset = KittiLoader(split="val", load_images=False)

    # Load model wrapper and predictor
    try:
        wrapper = YoloWrapper(model_path=str(weights))
        predictor = YoloPredictor(wrapper=wrapper)
    except RuntimeError as exc:
        logger.error("Failed to load model: %s", exc)
        return 2

    evaluator = YoloEvaluator()
    report = evaluator.evaluate_dataset(dataset=dataset, predictor=predictor, dataset_name="val")

    if args.save_json:
        out = Path(weights).resolve().parent / f"evaluation_{Path(weights).stem}.json"
        report.save_json(out)
        logger.info("Saved evaluation report to %s", out)
    else:
        print(json.dumps(report.to_dict(), indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
