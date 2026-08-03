"""Configuration module for YOLO integration in ResilientVisionAutonomous."""

from dataclasses import dataclass, field
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from config.paths import OUTPUTS_DIR

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class YoloConfig:
    """Immutable configuration dataclass for YOLO models.

    Attributes:
        model_name: Name or path of the model architecture (e.g. 'yolov8n.pt').
        pretrained_weights: Optional path or name for pretrained weights.
        epochs: Number of training epochs.
        batch_size: Batch size for training/inference.
        image_size: Target square image size (pixels).
        learning_rate: Initial learning rate.
        optimizer: Optimizer name (e.g. 'SGD', 'Adam', 'auto').
        confidence_threshold: Confidence threshold for predictions [0.0..1.0].
        iou_threshold: NMS IoU threshold for predictions [0.0..1.0].
        device: Target computation device (e.g. 'cpu', 'cuda', '0').
        project_directory: Parent directory for training/eval outputs.
        experiment_name: Name of experiment run subdirectory.
        save_predictions: Whether to save prediction output files.
        save_visualizations: Whether to save annotated visual images.
        verbose: Enable verbose logging during training/inference.
        extra_args: Additional keyword arguments for future compatibility.
    """

    model_name: str = "yolov8n.pt"
    pretrained_weights: Optional[str] = None
    epochs: int = 50
    batch_size: int = 16
    image_size: int = 640
    learning_rate: float = 0.01
    optimizer: str = "auto"
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.45
    device: str = "cpu"
    project_directory: Path = field(
        default_factory=lambda: OUTPUTS_DIR / "yolo"
    )
    experiment_name: str = "exp"
    save_predictions: bool = True
    save_visualizations: bool = True
    verbose: bool = True
    extra_args: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        """Validate all parameters and enforce immutability rules.

        Raises:
            ValueError: If numerical values or parameters fall outside valid ranges.
            TypeError: If parameter types are invalid.
        """
        # Ensure project_directory is a Path object
        if not isinstance(self.project_directory, Path):
            object.__setattr__(
                self, "project_directory", Path(self.project_directory)
            )

        # Validate epochs
        if not isinstance(self.epochs, int) or self.epochs <= 0:
            raise ValueError(f"epochs must be a positive integer, got {self.epochs}")

        # Validate batch_size
        if not isinstance(self.batch_size, int) or self.batch_size <= 0:
            raise ValueError(
                f"batch_size must be a positive integer, got {self.batch_size}"
            )

        # Validate image_size
        if not isinstance(self.image_size, int) or self.image_size <= 0:
            raise ValueError(
                f"image_size must be a positive integer, got {self.image_size}"
            )

        # Validate learning_rate
        if not isinstance(self.learning_rate, (int, float)) or self.learning_rate <= 0:
            raise ValueError(
                f"learning_rate must be a positive float, got {self.learning_rate}"
            )

        # Validate confidence_threshold
        if (
            not isinstance(self.confidence_threshold, (int, float))
            or not (0.0 <= self.confidence_threshold <= 1.0)
        ):
            raise ValueError(
                f"confidence_threshold must be between 0.0 and 1.0, got {self.confidence_threshold}"
            )

        # Validate iou_threshold
        if (
            not isinstance(self.iou_threshold, (int, float))
            or not (0.0 <= self.iou_threshold <= 1.0)
        ):
            raise ValueError(
                f"iou_threshold must be between 0.0 and 1.0, got {self.iou_threshold}"
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to a dictionary representation.

        Returns:
            Dictionary containing all configuration values.
        """
        return {
            "model_name": self.model_name,
            "pretrained_weights": self.pretrained_weights,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "image_size": self.image_size,
            "learning_rate": self.learning_rate,
            "optimizer": self.optimizer,
            "confidence_threshold": self.confidence_threshold,
            "iou_threshold": self.iou_threshold,
            "device": self.device,
            "project_directory": str(self.project_directory),
            "experiment_name": self.experiment_name,
            "save_predictions": self.save_predictions,
            "save_visualizations": self.save_visualizations,
            "verbose": self.verbose,
            "extra_args": self.extra_args.copy() if self.extra_args else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "YoloConfig":
        """Create a YoloConfig instance from a dictionary.

        Args:
            data: Dictionary containing configuration key-value pairs.

        Returns:
            YoloConfig instance initialized with dictionary data.
        """
        config_data = data.copy()
        if "project_directory" in config_data and config_data["project_directory"]:
            config_data["project_directory"] = Path(config_data["project_directory"])
        return cls(**config_data)

    def to_ultralytics_args(self, stage: str = "train") -> Dict[str, Any]:
        """Format configuration into kwargs suitable for Ultralytics API calls.

        Args:
            stage: API stage ('train', 'predict', or 'val').

        Returns:
            Keyword arguments dictionary for Ultralytics YOLO methods.
        """
        base_args: Dict[str, Any] = {
            "device": self.device,
            "verbose": self.verbose,
        }

        if stage == "train":
            base_args.update(
                {
                    "epochs": self.epochs,
                    "batch": self.batch_size,
                    "imgsz": self.image_size,
                    "lr0": self.learning_rate,
                    "optimizer": self.optimizer,
                    "project": str(self.project_directory),
                    "name": self.experiment_name,
                }
            )
        elif stage == "predict":
            base_args.update(
                {
                    "conf": self.confidence_threshold,
                    "iou": self.iou_threshold,
                    "imgsz": self.image_size,
                    "project": str(self.project_directory),
                    "name": self.experiment_name,
                    "save": self.save_visualizations,
                    "save_txt": self.save_predictions,
                }
            )
        elif stage == "val":
            base_args.update(
                {
                    "conf": self.confidence_threshold,
                    "iou": self.iou_threshold,
                    "imgsz": self.image_size,
                    "batch": self.batch_size,
                    "project": str(self.project_directory),
                    "name": self.experiment_name,
                }
            )

        if self.extra_args:
            base_args.update(self.extra_args)

        return base_args
