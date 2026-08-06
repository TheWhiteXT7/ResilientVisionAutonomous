"""Master orchestrator to run full experiments: baseline, attacks, defenses.

This script composes dataset preparation, dataset generation (attacks), training
and evaluation, and writes structured outputs under outputs/experiments/<stage>.
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

from config.paths import OUTPUTS_DIR
from dataset_loader.kitti_loader import KittiLoader
from models.utils import prepare_yolo_dataset
from models.yolo_config import YoloConfig
from models.trainer import YoloTrainer
from models.evaluator import YoloEvaluator
from dataset_generator.dataset_generator import DatasetGenerator
from dataset_generator.generator_config import GeneratorConfig

logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run a set of YOLO experiments end-to-end")
    p.add_argument("--stages", nargs="*", choices=["baseline", "random_attack", "target_attack", "defense", "custom"], default=["baseline"], help="Which experiment stages to run")
    p.add_argument("--project", default=str(OUTPUTS_DIR / "experiments"), help="Base project experiments output dir")
    p.add_argument("--name", default=None, help="Optional run name; if omitted timestamp used")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=16)
    return p


def _make_exp_dir(base: Path, stage: str, run_name: str) -> Path:
    d = base / stage / run_name
    d.mkdir(parents=True, exist_ok=True)
    return d


def _save_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=str)


def run_baseline(exp_dir: Path, args: argparse.Namespace) -> Dict[str, Any]:
    logger.info("Running baseline: prepare dataset, train, evaluate")
    # Prepare dataset from KITTI
    loader = KittiLoader(split="train", load_images=False)
    data_yaml = prepare_yolo_dataset(loader, output_dir=exp_dir / "dataset")

    # Configure training
    cfg = YoloConfig.from_dict({
        "model_name": "yolov8n.pt",
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "device": "cpu",
        "project_directory": exp_dir,
        "experiment_name": "baseline",
    })

    trainer = YoloTrainer(config=cfg)
    trainer.train(dataset=None, data_yaml_path=data_yaml)

    # Evaluate on validation split
    val_loader = KittiLoader(kitti_dir=str(exp_dir / "dataset"), split="val", load_images=False)
    evaluator = YoloEvaluator()
    wrapper = trainer.wrapper
    predictor = wrapper and None
    # Use predictor wrapper to evaluate
    from models.predictor import YoloPredictor

    predictor = YoloPredictor(wrapper=wrapper)
    report = evaluator.evaluate_dataset(dataset=val_loader, predictor=predictor, dataset_name="baseline_val")

    # Persist artifacts
    _save_json(exp_dir / "config.json", cfg.to_dict())
    report.save_json(exp_dir / "metrics.json")
    # Copy weights if available
    try:
        if trainer.best_weights_path.exists():
            shutil.copy2(trainer.best_weights_path, exp_dir / "best.pt")
        if trainer.last_weights_path.exists():
            shutil.copy2(trainer.last_weights_path, exp_dir / "last.pt")
    except Exception:
        logger.debug("No weights to copy for baseline")

    return report.to_dict()


def run_attack_stage(stage: str, exp_dir: Path, args: argparse.Namespace, pattern_type: str = "random") -> Dict[str, Any]:
    logger.info("Running attack stage '%s' pattern=%s", stage, pattern_type)
    # Generate attacked dataset into a subfolder of exp_dir
    out_ds_dir = exp_dir / "dataset_attacked"
    gen_cfg = GeneratorConfig(output_directory=str(out_ds_dir))
    generator = DatasetGenerator(config=gen_cfg)
    generator.generate_dataset(pattern_type=pattern_type)

    # Prepare YOLO dataset
    data_yaml = prepare_yolo_dataset(out_ds_dir, output_dir=exp_dir / "dataset_prepared")

    cfg = YoloConfig.from_dict({
        "model_name": "yolov8n.pt",
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "device": "cpu",
        "project_directory": exp_dir,
        "experiment_name": stage,
    })
    trainer = YoloTrainer(config=cfg)
    trainer.train(dataset=None, data_yaml_path=data_yaml)

    # Evaluate
    val_loader = KittiLoader(kitti_dir=str(exp_dir / "dataset_prepared"), split="val", load_images=False)
    evaluator = YoloEvaluator()
    predictor = __import__("models.predictor", fromlist=["YoloPredictor"]).YoloPredictor(wrapper=trainer.wrapper)
    report = evaluator.evaluate_dataset(dataset=val_loader, predictor=predictor, dataset_name=stage)
    report.save_json(exp_dir / "metrics.json")

    # copy weights
    try:
        if trainer.best_weights_path.exists():
            shutil.copy2(trainer.best_weights_path, exp_dir / "best.pt")
    except Exception:
        logger.debug("No best weights to copy")

    return report.to_dict()


def main(argv: Any = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    base = Path(args.project)
    run_name = args.name or time.strftime("%Y%m%d_%H%M%S")

    overall: Dict[str, Any] = {"run_name": run_name, "stages": {}}

    for stage in args.stages:
        exp_dir = _make_exp_dir(base, stage, run_name)
        try:
            if stage == "baseline":
                metrics = run_baseline(exp_dir, args)
            elif stage == "random_attack":
                metrics = run_attack_stage(stage, exp_dir, args, pattern_type="random")
            elif stage == "target_attack":
                metrics = run_attack_stage(stage, exp_dir, args, pattern_type="single")
            elif stage == "defense":
                metrics = run_attack_stage(stage, exp_dir, args, pattern_type="random")
            else:
                # custom: user may add custom generation outside this script
                metrics = {"note": "custom stage - no-op"}

            overall["stages"][stage] = metrics
        except Exception as exc:
            logger.exception("Stage failed: %s", exc)
            overall["stages"][stage] = {"error": str(exc)}

    # Persist overall run summary
    _save_json(base / "_last_run_summary.json", overall)
    print(json.dumps(overall, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
