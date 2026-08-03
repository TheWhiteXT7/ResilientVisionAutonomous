"""Unit tests for YoloPredictor module."""

from pathlib import Path
from unittest.mock import MagicMock
import numpy as np
from PIL import Image
import pytest

from models.predictor import DetectionBox, DetectionResult, YoloPredictor
from models.yolo_wrapper import YoloWrapper


def _create_mock_ultralytics_result() -> MagicMock:
    """Helper to create a mock Ultralytics Result object."""
    mock_res = MagicMock()
    mock_res.orig_shape = (375, 1242)
    mock_res.speed = {"preprocess": 1.5, "inference": 10.2, "postprocess": 0.8}
    mock_res.names = {0: "Car", 1: "Pedestrian"}

    # Mock boxes
    mock_boxes = MagicMock()
    mock_boxes.xyxy = np.array([[10.0, 20.0, 100.0, 200.0], [50.0, 60.0, 150.0, 250.0]])
    mock_boxes.conf = np.array([0.9, 0.75])
    mock_boxes.cls = np.array([0, 1])

    mock_res.boxes = mock_boxes
    return mock_res


def test_predict_image_single(tmp_path: Path) -> None:
    """Test YoloPredictor predict_image method."""
    mock_wrapper = MagicMock(spec=YoloWrapper)
    mock_wrapper.predict.return_value = [_create_mock_ultralytics_result()]

    predictor = YoloPredictor(wrapper=mock_wrapper)

    # Create dummy image file
    img_path = tmp_path / "test.png"
    img = Image.new("RGB", (100, 100), color="white")
    img.save(img_path)

    result = predictor.predict_image(img_path)

    assert isinstance(result, DetectionResult)
    assert result.sample_id == "test"
    assert len(result.boxes) == 2
    assert result.boxes[0].class_name == "Car"
    assert result.boxes[0].confidence == pytest.approx(0.9)
    assert result.boxes[0].bbox == (10.0, 20.0, 100.0, 200.0)


def test_predict_directory(tmp_path: Path) -> None:
    """Test YoloPredictor predict_directory method."""
    mock_wrapper = MagicMock(spec=YoloWrapper)
    mock_wrapper.predict.return_value = [_create_mock_ultralytics_result()]

    predictor = YoloPredictor(wrapper=mock_wrapper)

    # Create dummy images in directory
    (tmp_path / "img1.jpg").write_bytes(b"dummy")
    (tmp_path / "img2.png").write_bytes(b"dummy")

    results = predictor.predict_directory(tmp_path)

    assert len(results) == 2
    assert isinstance(results[0], DetectionResult)


def test_predict_dataset() -> None:
    """Test YoloPredictor predict_dataset method."""
    mock_wrapper = MagicMock(spec=YoloWrapper)
    mock_wrapper.predict.return_value = [_create_mock_ultralytics_result()]

    predictor = YoloPredictor(wrapper=mock_wrapper)

    # Mock dataset sample objects
    sample1 = MagicMock()
    sample1.sample_id = "000001"
    sample1.image_path = Path("path/to/000001.png")
    sample1.image = Image.new("RGB", (50, 50))

    sample2 = MagicMock()
    sample2.sample_id = "000002"
    sample2.image_path = Path("path/to/000002.png")
    sample2.image = Image.new("RGB", (50, 50))

    dataset = [sample1, sample2]
    results = predictor.predict_dataset(dataset)

    assert len(results) == 2
    assert results[0].sample_id == "000001"
    assert results[1].sample_id == "000002"
