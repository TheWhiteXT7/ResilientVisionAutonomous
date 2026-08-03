"""Unit tests for YoloConfig module."""

from pathlib import Path
import pytest

from models.yolo_config import YoloConfig


def test_yolo_config_defaults() -> None:
    """Test YoloConfig default initialization values."""
    config = YoloConfig()
    assert config.model_name == "yolov8n.pt"
    assert config.epochs == 50
    assert config.batch_size == 16
    assert config.image_size == 640
    assert config.learning_rate == 0.01
    assert config.confidence_threshold == 0.25
    assert config.iou_threshold == 0.45
    assert config.device == "cpu"
    assert isinstance(config.project_directory, Path)


def test_yolo_config_immutability() -> None:
    """Test YoloConfig frozen dataclass immutability."""
    config = YoloConfig()
    with pytest.raises(AttributeError):
        config.epochs = 100  # type: ignore[misc]


def test_yolo_config_validation_epochs() -> None:
    """Test validation of invalid epochs."""
    with pytest.raises(ValueError, match="epochs must be a positive integer"):
        YoloConfig(epochs=0)
    with pytest.raises(ValueError, match="epochs must be a positive integer"):
        YoloConfig(epochs=-5)


def test_yolo_config_validation_batch_size() -> None:
    """Test validation of invalid batch size."""
    with pytest.raises(ValueError, match="batch_size must be a positive integer"):
        YoloConfig(batch_size=0)


def test_yolo_config_validation_confidence_threshold() -> None:
    """Test validation of invalid confidence threshold."""
    with pytest.raises(ValueError, match="confidence_threshold must be between"):
        YoloConfig(confidence_threshold=1.5)
    with pytest.raises(ValueError, match="confidence_threshold must be between"):
        YoloConfig(confidence_threshold=-0.1)


def test_yolo_config_validation_iou_threshold() -> None:
    """Test validation of invalid IoU threshold."""
    with pytest.raises(ValueError, match="iou_threshold must be between"):
        YoloConfig(iou_threshold=2.0)


def test_yolo_config_to_dict_and_from_dict() -> None:
    """Test dictionary serialization and deserialization."""
    config = YoloConfig(epochs=30, batch_size=8, experiment_name="test_exp")
    data_dict = config.to_dict()

    assert data_dict["epochs"] == 30
    assert data_dict["batch_size"] == 8
    assert data_dict["experiment_name"] == "test_exp"

    recreated = YoloConfig.from_dict(data_dict)
    assert recreated.epochs == config.epochs
    assert recreated.batch_size == config.batch_size
    assert recreated.experiment_name == config.experiment_name


def test_to_ultralytics_args() -> None:
    """Test generation of Ultralytics API arguments."""
    config = YoloConfig(epochs=25, batch_size=4, image_size=512)

    train_args = config.to_ultralytics_args(stage="train")
    assert train_args["epochs"] == 25
    assert train_args["batch"] == 4
    assert train_args["imgsz"] == 512

    predict_args = config.to_ultralytics_args(stage="predict")
    assert predict_args["conf"] == 0.25
    assert predict_args["imgsz"] == 512
