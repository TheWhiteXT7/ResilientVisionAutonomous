"""Evaluation-only attack comparison workflow.

Measures degradation of a SINGLE FROZEN clean baseline YOLO detector across three
existing validation populations:

* clean          -- the clean baseline validation split
* random_attack  -- the random-attacked validation split
* target_attack  -- the targeted-attacked validation split

The detector weights are loaded exactly once and every population is evaluated
with the same predictor. No training is performed and no datasets are
regenerated: this module consumes existing YOLO-format ``data.yaml`` outputs
produced by the existing experiment pipeline.

The final report preserves the existing metric set (mAP50, mAP50-95, precision,
recall, F1, TP, FP, FN, num_samples) and adds explicit degradation comparisons
relative to the clean population. When a targeted-attack metadata directory is
provided, per-sample ``target_found``/``preserved`` information is summarized so
the final analysis can report how many targeted samples were actually attacked
versus preserved.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from models.evaluator import YoloEvaluator
from models.metrics import compare_metrics
from models.predictor import YoloPredictor
from models.yolo_config import YoloConfig
from models.yolo_dataset import YoloDataset
from models.yolo_wrapper import YoloWrapper

logger = logging.getLogger(__name__)

POPULATION_NAMES = ("clean", "random_attack", "target_attack")
COMPARED_POPULATIONS = ("random_attack", "target_attack")


@dataclass
class AttackComparisonConfig:
    """Configuration for an evaluation-only attack comparison run."""

    weights: Union[str, Path]
    clean_data_yaml: Union[str, Path]
    random_data_yaml: Union[str, Path]
    target_data_yaml: Union[str, Path]
    expected_samples: int = 1497
    device: str = "cpu"
    target_metadata_dir: Optional[Union[str, Path]] = None

    def data_yaml_paths(self) -> Dict[str, Path]:
        """Return the per-population data.yaml paths keyed by population name."""
        return {
            "clean": Path(self.clean_data_yaml),
            "random_attack": Path(self.random_data_yaml),
            "target_attack": Path(self.target_data_yaml),
        }


def count_validation_samples(data_yaml: Union[str, Path]) -> int:
    """Return the number of samples in the ``val`` split of a data.yaml.

    Raises:
        FileNotFoundError: If the data.yaml does not exist.
        RuntimeError: If the split cannot be loaded as a YOLO dataset.
    """
    data_yaml_path = Path(data_yaml)
    if not data_yaml_path.exists():
        raise FileNotFoundError(f"data.yaml not found: {data_yaml_path}")
    try:
        dataset = YoloDataset.from_yaml(data_yaml_path, split="val")
    except Exception as err:
        raise RuntimeError(
            f"Failed to load validation split from {data_yaml_path}: {err}"
        ) from err
    return len(dataset)


def validate_population_sizes(
    data_yamls: Dict[str, Path], expected_samples: int
) -> Dict[str, int]:
    """Verify every population contains exactly ``expected_samples`` samples.

    Raises:
        ValueError: If any population's sample count differs from the expected
            count, so mismatched populations are never silently compared.
    """
    counts: Dict[str, int] = {}
    for name, data_yaml in data_yamls.items():
        count = count_validation_samples(data_yaml)
        counts[name] = count
        if count != expected_samples:
            raise ValueError(
                f"Validation population '{name}' at {data_yaml} contains {count} "
                f"samples; expected {expected_samples}. Refusing to compare "
                f"different populations."
            )
    return counts


def compare_to_clean(
    clean_metrics: Dict[str, Any], attacked_metrics: Dict[str, Any]
) -> Dict[str, Any]:
    """Compute degradation of an attacked population relative to the clean one.

    Returns absolute changes (attacked - clean) and relative percentage changes
    for mAP50 and mAP50-95, plus the standard deltas/percentage_drops produced
    by ``models.metrics.compare_metrics``.
    """
    base = compare_metrics(
        clean_metrics=clean_metrics, attacked_metrics=attacked_metrics
    )

    clean_map50 = float(clean_metrics.get("mAP50", 0.0))
    clean_map50_95 = float(clean_metrics.get("mAP50-95", 0.0))
    attacked_map50 = float(attacked_metrics.get("mAP50", 0.0))
    attacked_map50_95 = float(attacked_metrics.get("mAP50-95", 0.0))

    absolute_map50 = attacked_map50 - clean_map50
    absolute_map50_95 = attacked_map50_95 - clean_map50_95

    def _relative_percentage(base_value: float, absolute_change: float) -> float:
        return (
            absolute_change / base_value * 100.0
            if base_value
            else 0.0
        )

    return {
        "absolute_mAP50_change": float(absolute_map50),
        "absolute_mAP50-95_change": float(absolute_map50_95),
        "relative_mAP50_change_pct": float(
            _relative_percentage(clean_map50, absolute_map50)
        ),
        "relative_mAP50-95_change_pct": float(
            _relative_percentage(clean_map50_95, absolute_map50_95)
        ),
        "deltas": base["deltas"],
        "percentage_drops": base["percentage_drops"],
    }


def summarize_target_metadata(
    metadata_dir: Union[str, Path], sample_ids: List[str]
) -> Dict[str, Any]:
    """Aggregate per-sample targeted-attack metadata for the evaluated population.

    Each metadata JSON (written by the existing DatasetGenerator) carries a
    ``target_found`` and a ``preserved`` flag. A sample whose target was found
    was actually attacked; a preserved sample kept its original image because no
    valid target was found.
    """
    metadata_path = Path(metadata_dir)
    if not metadata_path.is_dir():
        raise FileNotFoundError(
            f"Targeted-attack metadata directory not found: {metadata_path}"
        )

    attacked = 0
    preserved = 0
    samples_with_metadata = 0

    for sample_id in sample_ids:
        # YoloDataset sample ids carry a split prefix (e.g. "val/000000");
        # metadata files written by the generator are keyed by the bare
        # sample id (e.g. "000000"), so try both naming conventions.
        candidates = {sample_id, sample_id.rsplit("/", 1)[-1]}
        meta_file = next(
            (
                metadata_path / f"{candidate}.json"
                for candidate in candidates
                if (metadata_path / f"{candidate}.json").is_file()
            ),
            None,
        )
        if meta_file is None:
            continue
        with meta_file.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        samples_with_metadata += 1
        if data.get("target_found"):
            attacked += 1
        if data.get("preserved"):
            preserved += 1

    return {
        "metadata_dir": str(metadata_path.resolve()),
        "samples_with_metadata": samples_with_metadata,
        "attacked": attacked,
        "preserved": preserved,
        "unknown": max(0, len(sample_ids) - samples_with_metadata),
    }


class AttackComparisonEvaluator:
    """Evaluate one frozen model on all three validation populations.

    The wrapper/predictor pair is created exactly once and reused across every
    population, guaranteeing the same frozen weights are measured everywhere.
    """

    def __init__(self, config: AttackComparisonConfig) -> None:
        self.config = config
        self.wrapper: Optional[YoloWrapper] = None
        self.predictor: Optional[YoloPredictor] = None

    def load_model_once(self) -> None:
        """Load the frozen clean baseline weights exactly once."""
        if self.predictor is not None:
            return
        cfg = YoloConfig(
            device=self.config.device,
            extra_args={"save": False, "save_txt": False},
        )
        self.wrapper = YoloWrapper(
            config=cfg, model_path=str(Path(self.config.weights))
        )
        self.predictor = YoloPredictor(wrapper=self.wrapper)
        logger.info("Loaded frozen baseline weights: %s", self.config.weights)

    def _build_datasets(self) -> Dict[str, YoloDataset]:
        data_yamls = self.config.data_yaml_paths()
        counts = validate_population_sizes(data_yamls, self.config.expected_samples)
        logger.info("Validation population sizes: %s", counts)
        return {
            name: YoloDataset.from_yaml(data_yaml, split="val")
            for name, data_yaml in data_yamls.items()
        }

    def evaluate(self) -> Dict[str, Any]:
        """Run inference-only evaluation and return the full comparison report."""
        datasets = self._build_datasets()
        self.load_model_once()

        evaluator = YoloEvaluator()
        reports: Dict[str, Any] = {}
        for name in POPULATION_NAMES:
            logger.info("Evaluating frozen detector on '%s'...", name)
            reports[name] = evaluator.evaluate_dataset(
                dataset=datasets[name],
                predictor=self.predictor,  # type: ignore[arg-type]
                dataset_name=name,
            )

        comparisons: Dict[str, Any] = {}
        clean_metrics = reports["clean"].metrics.to_dict()
        for name in COMPARED_POPULATIONS:
            comparisons[name] = compare_to_clean(
                clean_metrics=clean_metrics,
                attacked_metrics=reports[name].metrics.to_dict(),
            )

        report: Dict[str, Any] = {
            "model_weights": str(Path(self.config.weights).resolve()),
            "datasets": {
                name: str(data_yaml.resolve())
                for name, data_yaml in self.config.data_yaml_paths().items()
            },
            "datasets_evaluated": list(POPULATION_NAMES),
            "expected_samples": self.config.expected_samples,
            "sample_counts": {
                name: reports[name].num_samples for name in POPULATION_NAMES
            },
            "clean": reports["clean"].to_dict(),
            "random_attack": reports["random_attack"].to_dict(),
            "target_attack": reports["target_attack"].to_dict(),
            "comparisons": comparisons,
        }

        if self.config.target_metadata_dir is not None:
            target_sample_ids = [
                sample.sample_id for sample in datasets["target_attack"]
            ]
            report["target_metadata_summary"] = summarize_target_metadata(
                metadata_dir=self.config.target_metadata_dir,
                sample_ids=target_sample_ids,
            )

        return report

    @staticmethod
    def save_report(report: Dict[str, Any], output_path: Union[str, Path]) -> Path:
        """Persist the comparison report as a single JSON file."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, default=str)
        logger.info("Saved attack comparison report to: %s", path)
        return path
