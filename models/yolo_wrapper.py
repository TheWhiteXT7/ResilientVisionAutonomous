"""YOLO model wrapper abstraction around Ultralytics YOLO."""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

from ultralytics import YOLO

from models.yolo_config import YoloConfig

logger = logging.getLogger(__name__)


class YoloWrapper:
    """Wrapper abstraction around Ultralytics YOLO models.

    Provides a clean object-oriented interface for loading, training, predicting,
    evaluating, and exporting YOLO object detection models without embedding training
    or evaluation logic within the wrapper itself.
    """

    def __init__(
        self,
        config: Optional[YoloConfig] = None,
        model_path: Optional[Union[str, Path]] = None,
    ) -> None:
        """Initialize YoloWrapper and load the YOLO model.

        Args:
            config: Optional YoloConfig object. Defaults to default YoloConfig.
            model_path: Optional explicit model path or weights name to load.
                If provided, overrides model specified in config.

        Raises:
            RuntimeError: If model fails to load.
        """
        self.config = config if config is not None else YoloConfig()

        weights_to_load = (
            str(model_path)
            if model_path is not None
            else (
                self.config.pretrained_weights
                if self.config.pretrained_weights is not None
                else self.config.model_name
            )
        )

        try:
            logger.info(f"Loading YOLO model from: {weights_to_load}")
            self._model = YOLO(weights_to_load)
        except Exception as err:
            msg = f"Failed to load YOLO model '{weights_to_load}': {err}"
            logger.error(msg)
            raise RuntimeError(msg) from err

    @property
    def model(self) -> YOLO:
        """Access underlying Ultralytics YOLO model instance.

        Returns:
            Ultralytics YOLO instance.
        """
        return self._model

    def train(self, data: Union[str, Path, Dict[str, Any]], **kwargs: Any) -> Any:
        """Delegate training execution to the underlying Ultralytics YOLO model.

        Args:
            data: Path to data config YAML file or dataset configuration dict.
            **kwargs: Additional training arguments overriding defaults in config.

        Returns:
            Ultralytics training results object.
        """
        train_args = self.config.to_ultralytics_args(stage="train")
        train_args["data"] = str(data) if isinstance(data, (str, Path)) else data
        train_args.update(kwargs)

        logger.info(
            f"Starting YOLO training with data: {data}, epochs: {train_args.get('epochs')}"
        )
        return self._model.train(**train_args)

    def predict(self, source: Any, **kwargs: Any) -> Any:
        """Delegate prediction execution to the underlying Ultralytics YOLO model.

        Args:
            source: Input image source (file path, directory path, PIL Image, ndarray).
            **kwargs: Additional prediction arguments overriding defaults in config.

        Returns:
            List of Ultralytics Results objects.
        """
        predict_args = self.config.to_ultralytics_args(stage="predict")
        predict_args["source"] = source
        predict_args.update(kwargs)

        logger.debug(f"Running YOLO inference on source: {source}")
        return self._model.predict(**predict_args)

    def validate(self, data: Union[str, Path, Dict[str, Any]], **kwargs: Any) -> Any:
        """Delegate validation execution to the underlying Ultralytics YOLO model.

        Args:
            data: Path to data config YAML file or dataset configuration dict.
            **kwargs: Additional validation arguments overriding defaults in config.

        Returns:
            Ultralytics validation metrics object.
        """
        val_args = self.config.to_ultralytics_args(stage="val")
        val_args["data"] = str(data) if isinstance(data, (str, Path)) else data
        val_args.update(kwargs)

        logger.info(f"Running YOLO validation on data: {data}")
        return self._model.val(**val_args)

    def export(self, format: str = "onnx", **kwargs: Any) -> Any:
        """Delegate model export execution to the underlying Ultralytics YOLO model.

        Args:
            format: Target export format (e.g. 'onnx', 'torchscript', 'engine').
            **kwargs: Additional export arguments.

        Returns:
            Path or metadata of exported model file.
        """
        logger.info(f"Exporting YOLO model to format: {format}")
        return self._model.export(format=format, **kwargs)
