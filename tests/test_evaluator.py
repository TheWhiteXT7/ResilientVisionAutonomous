"""Unit tests for YoloEvaluator module."""

import json
from pathlib import Path
from unittest.mock import MagicMock
import pytest

from models.evaluator import EvaluationReport, YoloEvaluator
from models.predictor import DetectionBox, DetectionResult, YoloPredictor


@pytest.fixture
def mock_predictions_and_gts() -> tuple:
    """Fixture providing sample detection predictions and ground truth annotations."""
    pred = DetectionResult(
        sample_id="000001",
        image_path=Path("000001.png"),
        boxes=[
            DetectionBox(
                class_id=0,
                class_name="Car",
                confidence=0.9,
                bbox=(10.0, 10.0, 100.0, 100.0),
            )
        ],
    )

    gt_ann = MagicMock()
    gt_ann.class_name = "Car"
    gt_ann.bbox = (10.0, 10.0, 100.0, 100.0)

    return ([pred], [[gt_ann]])


def test_evaluator_evaluate_predictions(mock_predictions_and_gts: tuple) -> None:
    """Test YoloEvaluator evaluate_predictions method."""
    preds, gts = mock_predictions_and_gts
    evaluator = YoloEvaluator()

    report = evaluator.evaluate_predictions(preds, gts, dataset_name="Test DS")

    assert isinstance(report, EvaluationReport)
    assert report.dataset_name == "Test DS"
    assert report.num_samples == 1
    assert report.metrics.mAP50 == pytest.approx(1.0)
    assert report.metrics.precision == pytest.approx(1.0)
    assert report.metrics.recall == pytest.approx(1.0)
    assert report.metrics.f1_score == pytest.approx(1.0)


def test_evaluator_evaluate_dataset(mock_predictions_and_gts: tuple) -> None:
    """Test YoloEvaluator evaluate_dataset method."""
    preds, gts = mock_predictions_and_gts
    mock_predictor = MagicMock(spec=YoloPredictor)
    mock_predictor.predict_dataset.return_value = preds

    sample = MagicMock()
    sample.annotations = gts[0]
    dataset = [sample]

    evaluator = YoloEvaluator()
    report = evaluator.evaluate_dataset(dataset, mock_predictor, dataset_name="Mock DS")

    assert report.dataset_name == "Mock DS"
    assert report.metrics.mAP50 == pytest.approx(1.0)


def test_evaluator_compare_and_save_json(
    mock_predictions_and_gts: tuple, tmp_path: Path
) -> None:
    """Test YoloEvaluator compare method and JSON serialization."""
    preds, gts = mock_predictions_and_gts

    # Clean predictions match GT perfectly
    clean_preds = preds

    # Attacked predictions miss GT (False Positive elsewhere)
    attacked_pred = DetectionResult(
        sample_id="000001",
        image_path=Path("000001.png"),
        boxes=[
            DetectionBox(
                class_id=0,
                class_name="Car",
                confidence=0.8,
                bbox=(500.0, 500.0, 600.0, 600.0),
            )
        ],
    )

    evaluator = YoloEvaluator()
    clean_report = evaluator.evaluate_predictions(clean_preds, gts, dataset_name="Clean")
    attacked_report = evaluator.evaluate_predictions([attacked_pred], gts, dataset_name="Attacked")

    output_json = tmp_path / "comparison.json"
    comp_report = evaluator.compare(
        clean_results=clean_report,
        attacked_results=attacked_report,
        output_json_path=output_json,
    )

    assert output_json.exists()
    with open(output_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "clean_report" in data
    assert "attacked_report" in data
    assert "comparison" in data
    assert comp_report.comparison_metrics["deltas"]["mAP50"] == pytest.approx(1.0)
