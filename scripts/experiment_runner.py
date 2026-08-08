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
from models.yolo_dataset import YoloDataset
from dataset_generator.dataset_generator import DatasetGenerator
from dataset_generator.generator_config import GeneratorConfig

logger = logging.getLogger(__name__)


def _parse_device(value: str) -> str:
    """Accept CPU, CUDA, or a non-negative CUDA device index."""
    device = value.lower()
    if device in {"cpu", "cuda"} or device.isdigit():
        return device
    raise argparse.ArgumentTypeError("device must be 'cpu', 'cuda', or a non-negative GPU index")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run a set of YOLO experiments end-to-end")
    p.add_argument("--stages", nargs="*", choices=["baseline", "random_attack", "target_attack", "defense", "custom"], default=["baseline"], help="Which experiment stages to run")
    p.add_argument("--project", default=str(OUTPUTS_DIR / "experiments"), help="Base project experiments output dir")
    p.add_argument("--name", default=None, help="Optional run name; if omitted timestamp used")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--device", type=_parse_device, default="cpu", help="Compute device: cpu, cuda, or a GPU index such as 0")
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
    """Run the baseline experiment reusing the prepared YOLO dataset.

    Behavior changes:
    - Reuse outputs/yolo_dataset/data.yaml when available.
    - Do NOT create or copy image/label files into the experiment directory.
    - If the dataset YAML is missing, call prepare_yolo_dataset() once to create it
      under OUTPUTS_DIR / 'yolo_dataset'.
    """
    logger.info("Running baseline: using shared YOLO dataset (no duplication)")

    shared_data_yaml = OUTPUTS_DIR / "yolo_dataset" / "data.yaml"

    # Ensure a prepared YOLO dataset exists at outputs/yolo_dataset
    if not shared_data_yaml.exists():
        logger.info("Shared YOLO dataset not found at %s. Preparing...", shared_data_yaml)
        loader = KittiLoader(split="train", load_images=False)
        prepare_yolo_dataset(loader, output_dir=shared_data_yaml.parent)

    # Load YAML to validate the prepared YOLO dataset; it is not a KITTI root.
    try:
        import yaml

        with open(shared_data_yaml, "r", encoding="utf-8") as fh:
            y = yaml.safe_load(fh)
        ds_root = Path(y.get("path", str(shared_data_yaml.parent)))
    except Exception:
        # Fallback to parent folder if YAML parsing fails
        ds_root = shared_data_yaml.parent

    # Validate dataset contents before training
    try:
        from dataset_loader.kitti_loader import IMAGE_EXTENSIONS
    except Exception:
        IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}

    train_rel = y.get("train")
    val_rel = y.get("val")

    if not train_rel or not val_rel:
        raise RuntimeError(f"data.yaml must specify 'train' and 'val' entries: {shared_data_yaml}")

    train_images_dir = ds_root / Path(train_rel)
    val_images_dir = ds_root / Path(val_rel)

    train_label_dir = ds_root / "labels" / Path(train_rel).name
    val_label_dir = ds_root / "labels" / Path(val_rel).name

    def _count_images(p: Path) -> int:
        if not p.exists():
            return 0
        return sum(1 for f in p.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS)

    def _count_labels(p: Path) -> int:
        if not p.exists():
            return 0
        return sum(1 for f in p.iterdir() if f.is_file() and f.suffix.lower() == ".txt")

    n_train_imgs = _count_images(train_images_dir)
    n_val_imgs = _count_images(val_images_dir)
    n_train_lbls = _count_labels(train_label_dir)
    n_val_lbls = _count_labels(val_label_dir)

    # Log dataset statistics
    logger.info("Validated YOLO dataset: %s", shared_data_yaml)
    logger.info("  Train Images : %d", n_train_imgs)
    logger.info("  Val Images   : %d", n_val_imgs)
    logger.info("  Train Labels : %d", n_train_lbls)
    logger.info("  Val Labels   : %d", n_val_lbls)

    missing = []
    if not shared_data_yaml.exists():
        missing.append(f"data.yaml missing: {shared_data_yaml}")
    if n_train_imgs == 0:
        missing.append(f"train images empty: {train_images_dir}")
    if n_val_imgs == 0:
        missing.append(f"val images empty: {val_images_dir}")
    if n_train_lbls == 0:
        missing.append(f"train labels empty: {train_label_dir}")
    if n_val_lbls == 0:
        missing.append(f"val labels empty: {val_label_dir}")

    if missing:
        msg = "YOLO dataset validation failed:\n" + "\n".join(missing)
        logger.error(msg)
        raise RuntimeError(msg)

    # Configure training to write artifacts into the experiment directory only
    cfg = YoloConfig.from_dict({
        "model_name": "yolov8n.pt",
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "device": args.device,
        "project_directory": exp_dir,
        "experiment_name": "baseline",
    })

    trainer = YoloTrainer(config=cfg)
    # NOTE: pass the shared data.yaml path so Ultralytics reads images/labels in-place
    trainer.train(dataset=None, data_yaml_path=shared_data_yaml)

    # Evaluate the validation split referenced by the shared YOLO data.yaml.
    val_dataset = YoloDataset.from_yaml(shared_data_yaml, split="val")
    evaluator = YoloEvaluator()
    from models.predictor import YoloPredictor

    predictor = YoloPredictor(wrapper=trainer.wrapper)
    report = evaluator.evaluate_dataset(dataset=val_dataset, predictor=predictor, dataset_name="baseline_val")

    # Persist lightweight experiment artifacts only
    _save_json(exp_dir / "config.json", cfg.to_dict())
    report.save_json(exp_dir / "metrics.json")

    # Copy only checkpoints into exp_dir/checkpoints for convenience (do not duplicate dataset)
    try:
        checkpoints_dir = exp_dir / "checkpoints"
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        if trainer.best_weights_path.exists():
            shutil.copy2(trainer.best_weights_path, checkpoints_dir / "best.pt")
        if trainer.last_weights_path.exists():
            shutil.copy2(trainer.last_weights_path, checkpoints_dir / "last.pt")
    except Exception:
        logger.debug("No weights to copy for baseline")

    return report.to_dict()


def run_attack_stage(stage: str, exp_dir: Path, args: argparse.Namespace, pattern_type: str = "random") -> Dict[str, Any]:
    logger.info("Running attack stage '%s' pattern=%s", stage, pattern_type)
    # Generate attacked dataset into a subfolder of exp_dir
    out_ds_dir = exp_dir / "dataset_attacked"
    gen_cfg = GeneratorConfig(output_directory=str(out_ds_dir))
    attack_loader = KittiLoader(split="trainval", load_images=False)
    generator = DatasetGenerator(loader=attack_loader, config=gen_cfg)
    generator.generate_dataset(pattern_type=pattern_type)

    # Convert the generated KITTI-format attack output through its loader.
    attacked_loader = KittiLoader(kitti_dir=out_ds_dir, split="trainval", load_images=False)
    data_yaml = prepare_yolo_dataset(attacked_loader, output_dir=exp_dir / "dataset_prepared")

    cfg = YoloConfig.from_dict({
        "model_name": "yolov8n.pt",
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "device": args.device,
        "project_directory": exp_dir,
        "experiment_name": stage,
    })
    trainer = YoloTrainer(config=cfg)
    trainer.train(dataset=None, data_yaml_path=data_yaml)

    # Evaluate
    val_dataset = YoloDataset.from_yaml(data_yaml, split="val")
    evaluator = YoloEvaluator()
    predictor = __import__("models.predictor", fromlist=["YoloPredictor"]).YoloPredictor(wrapper=trainer.wrapper)
    report = evaluator.evaluate_dataset(dataset=val_dataset, predictor=predictor, dataset_name=stage)
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
                metrics = run_attack_stage(stage, exp_dir, args, pattern_type="targeted")
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






