"""YOLO Evaluator module for clean and attacked KITTI dataset evaluation."""

from dataclasses import dataclass, field
import json
logging_logger = None  # to avoid shadowing name
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from models.metrics import compare_metrics, compute_detection_metrics
from models.predictor import DetectionResult, YoloPredictor

logger = logging.getLogger(__name__)


@dataclass
class EvaluationMetrics:
    """Dataclass representing computed object detection evaluation metrics.

    Attributes:
        mAP50: Mean Average Precision at IoU=0.50.
        mAP50_95: Mean Average Precision across IoU=0.50:0.95.
        precision: Overall detection precision.
        recall: Overall detection recall.
        f1_score: Overall F1 score.
        tp: Total True Positives count.
        fp: Total False Positives count.
        fn: Total False Negatives count.
    """

    mAP50: float
    mAP50_95: float
    precision: float
    recall: float
    f1_score: float
    tp: int = 0
    fp: int = 0
    fn: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert EvaluationMetrics to a dictionary.

        Returns:
            Dictionary of metrics.
        """
        return {
            "mAP50": self.mAP50,
            "mAP50-95": self.mAP50_95,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1_score,
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
        }


@dataclass
class EvaluationReport:
    """Dataclass representing an evaluation report for a dataset run.

    Attributes:
        dataset_name: Name of the evaluated dataset (e.g. 'Clean KITTI', 'Attacked KITTI').
        num_samples: Total number of samples evaluated.
        metrics: EvaluationMetrics instance.
        raw_metrics: Optional dictionary containing additional raw metric outputs.
    """

    dataset_name: str
    num_samples: int
    metrics: EvaluationMetrics
    raw_metrics: Optional[Dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert EvaluationReport to a dictionary.

        Returns:
            Dictionary of report data.
        """
        return {
            "dataset_name": self.dataset_name,
            "num_samples": self.num_samples,
            "metrics": self.metrics.to_dict(),
            "raw_metrics": self.raw_metrics if self.raw_metrics else {},
        }

    def save_json(self, output_path: Union[str, Path]) -> Path:
        """Save evaluation report to a JSON file.

        Args:
            output_path: Target JSON file path.

        Returns:
            Path object pointing to saved JSON file.
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=4)

        logger.info(f"Saved evaluation report JSON to: {path}")
        return path


@dataclass
class ComparisonReport:
    """Dataclass representing a comparative evaluation between clean and attacked runs.

    Attributes:
        clean_report: EvaluationReport for clean dataset.
        attacked_report: EvaluationReport for attacked dataset.
        comparison_metrics: Dictionary containing metric deltas and percentage drops.
    """

    clean_report: EvaluationReport
    attacked_report: EvaluationReport
    comparison_metrics: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert ComparisonReport to a dictionary.

        Returns:
            Dictionary of comparison report data.
        """
        return {
            "clean_report": self.clean_report.to_dict(),
            "attacked_report": self.attacked_report.to_dict(),
            "comparison": self.comparison_metrics,
        }

    def save_json(self, output_path: Union[str, Path]) -> Path:
        """Save comparison report to a JSON file.

        Args:
            output_path: Target JSON file path.

        Returns:
            Path object pointing to saved JSON file.
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=4)

        logger.info(f"Saved comparison report JSON to: {path}")
        return path


class YoloEvaluator:
    """Evaluator for evaluating YOLO models on clean and attacked datasets."""

    def evaluate_predictions(
        self,
        predictions: List[DetectionResult],
        ground_truths: Any,
        dataset_name: str = "dataset",
        iou_threshold: float = 0.5,
    ) -> EvaluationReport:
        """Evaluate a set of prediction objects against ground truth annotations.

        Args:
            predictions: List of DetectionResult objects.
            ground_truths: List of Annotation lists or ground truth dictionary.
            dataset_name: Descriptor name for dataset.
            iou_threshold: IoU threshold for evaluation.

        Returns:
            EvaluationReport instance.
        """
        metrics_dict = compute_detection_metrics(
            predictions=predictions,
            ground_truths=ground_truths,
            iou_threshold=iou_threshold,
        )

        metrics_obj = EvaluationMetrics(
            mAP50=metrics_dict["mAP50"],
            mAP50_95=metrics_dict["mAP50-95"],
            precision=metrics_dict["precision"],
            recall=metrics_dict["recall"],
            f1_score=metrics_dict["f1"],
            tp=metrics_dict.get("tp", 0),
            fp=metrics_dict.get("fp", 0),
            fn=metrics_dict.get("fn", 0),
        )

        return EvaluationReport(
            dataset_name=dataset_name,
            num_samples=len(predictions),
            metrics=metrics_obj,
            raw_metrics=metrics_dict,
        )

    def evaluate_dataset(
        self,
        dataset: Any,
        predictor: YoloPredictor,
        dataset_name: str = "dataset",
        iou_threshold: float = 0.5,
        **kwargs: Any,
    ) -> EvaluationReport:
        """Run predictor inference on a dataset and evaluate against ground truth.

        Args:
            dataset: Sequence dataset object (e.g. KittiLoader).
            predictor: YoloPredictor instance to generate predictions.
            dataset_name: Descriptor name for dataset.
            iou_threshold: IoU threshold for evaluation.
            **kwargs: Extra arguments passed to predictor.

        Returns:
            EvaluationReport instance.
        """
        logger.info(f"Evaluating model on dataset '{dataset_name}'...")
        predictions = predictor.predict_dataset(dataset, **kwargs)

        # Extract ground truth annotations from dataset samples
        ground_truths: List[List[Any]] = []
        if hasattr(dataset, "__iter__"):
            for sample in dataset:
                anns = getattr(sample, "annotations", [])
                ground_truths.append(anns)

        return self.evaluate_predictions(
            predictions=predictions,
            ground_truths=ground_truths,
            dataset_name=dataset_name,
            iou_threshold=iou_threshold,
        )

    def compare(
        self,
        clean_results: Union[EvaluationReport, List[DetectionResult]],
        attacked_results: Union[EvaluationReport, List[DetectionResult]],
        ground_truths: Optional[Any] = None,
        output_json_path: Optional[Union[str, Path]] = None,
    ) -> ComparisonReport:
        """Compare clean evaluation results with attacked evaluation results.

        Args:
            clean_results: EvaluationReport or List of DetectionResult for clean dataset.
            attacked_results: EvaluationReport or List of DetectionResult for attacked dataset.
            ground_truths: Optional ground truth annotations if results are lists of DetectionResult.
            output_json_path: Optional output path to save JSON comparison report.

        Returns:
            ComparisonReport instance.
        """
        if isinstance(clean_results, EvaluationReport):
            clean_report = clean_results
        else:
            clean_report = self.evaluate_predictions(
                clean_results, ground_truths or [], dataset_name="Clean KITTI"
            )

        if isinstance(attacked_results, EvaluationReport):
            attacked_report = attacked_results
        else:
            attacked_report = self.evaluate_predictions(
                attacked_results, ground_truths or [], dataset_name="Attacked KITTI"
            )

        comp_metrics = compare_metrics(
            clean_metrics=clean_report.metrics.to_dict(),
            attacked_metrics=attacked_report.metrics.to_dict(),
        )

        comp_report = ComparisonReport(
            clean_report=clean_report,
            attacked_report=attacked_report,
            comparison_metrics=comp_metrics,
        )

        if output_json_path is not None:
            comp_report.save_json(output_json_path)

        return comp_report
