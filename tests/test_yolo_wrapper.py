"""Unit tests for YoloWrapper module."""

from unittest.mock import MagicMock, patch
import pytest

from models.yolo_config import YoloConfig
from models.yolo_wrapper import YoloWrapper


@patch("models.yolo_wrapper.YOLO")
def test_yolo_wrapper_init_default(mock_yolo: MagicMock) -> None:
    """Test YoloWrapper default initialization with mocked YOLO."""
    config = YoloConfig(model_name="yolov8s.pt")
    wrapper = YoloWrapper(config=config)

    mock_yolo.assert_called_once_with("yolov8s.pt")
    assert wrapper.model is mock_yolo.return_value


@patch("models.yolo_wrapper.YOLO")
def test_yolo_wrapper_init_custom_path(mock_yolo: MagicMock) -> None:
    """Test YoloWrapper initialization with custom model path."""
    wrapper = YoloWrapper(model_path="custom_weights.pt")
    mock_yolo.assert_called_once_with("custom_weights.pt")
    assert wrapper.model is not None


@patch("models.yolo_wrapper.YOLO", side_effect=Exception("File read error"))
def test_yolo_wrapper_init_failure(mock_yolo: MagicMock) -> None:
    """Test exception handling during model loading failure."""
    with pytest.raises(RuntimeError, match="Failed to load YOLO model"):
        YoloWrapper(model_path="invalid.pt")


@patch("models.yolo_wrapper.YOLO")
def test_yolo_wrapper_train(mock_yolo: MagicMock) -> None:
    """Test YoloWrapper train delegation."""
    mock_model_instance = MagicMock()
    mock_yolo.return_value = mock_model_instance

    wrapper = YoloWrapper()
    wrapper.train(data="data.yaml", epochs=10)

    mock_model_instance.train.assert_called_once()
    call_kwargs = mock_model_instance.train.call_args.kwargs
    assert call_kwargs["data"] == "data.yaml"
    assert call_kwargs["epochs"] == 10


@patch("models.yolo_wrapper.YOLO")
def test_yolo_wrapper_predict(mock_yolo: MagicMock) -> None:
    """Test YoloWrapper predict delegation."""
    mock_model_instance = MagicMock()
    mock_yolo.return_value = mock_model_instance

    wrapper = YoloWrapper()
    wrapper.predict(source="image.png", conf=0.5)

    mock_model_instance.predict.assert_called_once()
    call_kwargs = mock_model_instance.predict.call_args.kwargs
    assert call_kwargs["source"] == "image.png"
    assert call_kwargs["conf"] == 0.5


@patch("models.yolo_wrapper.YOLO")
def test_yolo_wrapper_validate(mock_yolo: MagicMock) -> None:
    """Test YoloWrapper validate delegation."""
    mock_model_instance = MagicMock()
    mock_yolo.return_value = mock_model_instance

    wrapper = YoloWrapper()
    wrapper.validate(data="data.yaml")

    mock_model_instance.val.assert_called_once()


@patch("models.yolo_wrapper.YOLO")
def test_yolo_wrapper_export(mock_yolo: MagicMock) -> None:
    """Test YoloWrapper export delegation."""
    mock_model_instance = MagicMock()
    mock_yolo.return_value = mock_model_instance

    wrapper = YoloWrapper()
    wrapper.export(format="onnx")

    mock_model_instance.export.assert_called_once_with(format="onnx")
