"""Unit tests for metrics calculation utilities."""

from unittest.mock import MagicMock
import numpy as np
import pytest

from models.metrics import (
    aggregate_metrics,
    compare_metrics,
    compute_confusion_matrix,
    compute_detection_metrics,
    compute_iou,
    compute_precision_recall_f1,
)
from models.predictor import DetectionBox, DetectionResult


def test_compute_iou() -> None:
    """Test IoU calculation for overlapping, non-overlapping, and identical boxes."""
    box1 = (0.0, 0.0, 10.0, 10.0)
    box2 = (0.0, 0.0, 10.0, 10.0)
    assert compute_iou(box1, box2) == pytest.approx(1.0)

    box3 = (10.0, 10.0, 20.0, 20.0)
    assert compute_iou(box1, box3) == pytest.approx(0.0)

    box4 = (5.0, 0.0, 15.0, 10.0)  # 50% overlap area
    # Inter: 5x10 = 50, Union: 100 + 100 - 50 = 150 -> 50/150 = 1/3
    assert compute_iou(box1, box4) == pytest.approx(1 / 3)


def test_compute_precision_recall_f1() -> None:
    """Test Precision, Recall, F1 score calculations."""
    p, r, f1 = compute_precision_recall_f1(tp=10, fp=2, fn=3)
    assert p == pytest.approx(10 / 12)
    assert r == pytest.approx(10 / 13)
    assert f1 == pytest.approx(2 * p * r / (p + r))

    # Test edge case zero counts
    p0, r0, f1_0 = compute_precision_recall_f1(tp=0, fp=0, fn=0)
    assert p0 == 0.0
    assert r0 == 0.0
    assert f1_0 == 0.0


def test_compute_detection_metrics() -> None:
    """Test end-to-end detection metrics computation."""
    pred = DetectionResult(
        sample_id="001",
        image_path=None,
        boxes=[
            DetectionBox(class_id=0, class_name="Car", confidence=0.9, bbox=(0.0, 0.0, 10.0, 10.0))
        ],
    )
    ann = MagicMock()
    ann.class_name = "Car"
    ann.bbox = (0.0, 0.0, 10.0, 10.0)

    metrics = compute_detection_metrics([pred], [[ann]], iou_threshold=0.5)

    assert metrics["mAP50"] == pytest.approx(1.0)
    assert metrics["precision"] == pytest.approx(1.0)
    assert metrics["recall"] == pytest.approx(1.0)
    assert metrics["f1"] == pytest.approx(1.0)


def test_compute_confusion_matrix() -> None:
    """Test confusion matrix generation."""
    pred = DetectionResult(
        sample_id="001",
        image_path=None,
        boxes=[
            DetectionBox(class_id=0, class_name="Car", confidence=0.9, bbox=(0.0, 0.0, 10.0, 10.0))
        ],
    )
    ann = MagicMock()
    ann.class_name = "Car"
    ann.class_id = 0
    ann.bbox = (0.0, 0.0, 10.0, 10.0)

    cm = compute_confusion_matrix([pred], [[ann]], num_classes=2, iou_threshold=0.5)

    assert isinstance(cm, np.ndarray)
    assert cm.shape == (3, 3)
    assert cm[0, 0] == 1  # 1 TP for class 0


def test_compare_metrics() -> None:
    """Test metric comparison between clean and attacked metrics."""
    clean = {"mAP50": 0.90, "precision": 0.85, "recall": 0.80, "f1": 0.824}
    attacked = {"mAP50": 0.45, "precision": 0.50, "recall": 0.40, "f1": 0.444}

    comp = compare_metrics(clean, attacked)

    assert comp["deltas"]["mAP50"] == pytest.approx(0.45)
    assert comp["percentage_drops"]["mAP50"] == pytest.approx(50.0)


def test_aggregate_metrics() -> None:
    """Test metrics aggregation across multiple runs."""
    m1 = {"mAP50": 0.80, "f1": 0.75}
    m2 = {"mAP50": 0.90, "f1": 0.85}

    agg = aggregate_metrics([m1, m2])

    assert agg["mAP50"]["mean"] == pytest.approx(0.85)
    assert agg["mAP50"]["min"] == pytest.approx(0.80)
    assert agg["mAP50"]["max"] == pytest.approx(0.90)
