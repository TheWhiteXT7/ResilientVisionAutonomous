"""YOLO Model Trainer module for ResilientVisionAutonomous."""

import logging
from pathlib import Path
import shutil
from typing import Any, Dict, Optional, Union

from models.utils import prepare_yolo_dataset
from models.yolo_config import YoloConfig
from models.yolo_wrapper import YoloWrapper

logger = logging.getLogger(__name__)


class YoloTrainer:
    """Trainer orchestrator for training YOLO models.

    Handles training execution, checkpoint management, training resumption, logging,
    and best weights saving.
    """

    def __init__(
        self,
        wrapper: Optional[YoloWrapper] = None,
        config: Optional[YoloConfig] = None,
    ) -> None:
        """Initialize YoloTrainer.

        Args:
            wrapper: Optional YoloWrapper instance. Created from config if None.
            config: Optional YoloConfig instance. Uses wrapper config or default if None.
        """
        if config is not None:
            self.config = config
        elif wrapper is not None and getattr(wrapper, "config", None) is not None:
            self.config = wrapper.config
        else:
            self.config = YoloConfig()

        if wrapper is not None:
            self.wrapper = wrapper
        else:
            self.wrapper = YoloWrapper(config=self.config)

        self._last_results: Optional[Any] = None
        self._experiment_dir: Path = (
            self.config.project_directory / self.config.experiment_name
        )

    @property
    def experiment_dir(self) -> Path:
        """Get path to the active experiment output directory.

        Returns:
            Path object to experiment directory.
        """
        return self._experiment_dir

    @property
    def best_weights_path(self) -> Path:
        """Get path to best model checkpoint weights (best.pt).

        Returns:
            Path object to best.pt weights file.
        """
        return self._experiment_dir / "weights" / "best.pt"

    @property
    def last_weights_path(self) -> Path:
        """Get path to last model checkpoint weights (last.pt).

        Returns:
            Path object to last.pt weights file.
        """
        return self._experiment_dir / "weights" / "last.pt"

    def train(
        self,
        dataset: Any,
        data_yaml_path: Optional[Union[str, Path]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Execute model training on the provided dataset.

        Args:
            dataset: KittiLoader instance, Path to data.yaml, or dataset object.
            data_yaml_path: Explicit path to pre-existing data.yaml config.
            **kwargs: Additional training parameter overrides.

        Returns:
            Dictionary summary of training results and metrics.

        Raises:
            RuntimeError: If training fails to execute.
        """
        logger.info("Preparing dataset for YOLO training...")
        if data_yaml_path is not None:
            resolved_yaml = Path(data_yaml_path)
        else:
            prepared_dir = self.config.project_directory / "dataset_prepared"
            resolved_yaml = prepare_yolo_dataset(dataset, output_dir=prepared_dir)

        try:
            results = self.wrapper.train(data=resolved_yaml, **kwargs)
            self._last_results = results

            summary = {
                "status": "success",
                "experiment_dir": str(self.experiment_dir),
                "best_weights": str(self.best_weights_path),
                "last_weights": str(self.last_weights_path),
                "epochs": self.config.epochs,
                "data_yaml": str(resolved_yaml),
            }

            if hasattr(results, "results_dict") and isinstance(results.results_dict, dict):
                summary["metrics"] = results.results_dict

            logger.info(
                f"Training completed successfully. Artifacts saved in: {self.experiment_dir}"
            )
            return summary

        except Exception as err:
            msg = f"YOLO training failed: {err}"
            logger.error(msg)
            raise RuntimeError(msg) from err

    def resume(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Resume training from a previous checkpoint.

        Args:
            checkpoint_path: Explicit checkpoint path. Defaults to last_weights_path if None.
            **kwargs: Additional parameter overrides.

        Returns:
            Dictionary summary of resumed training results.

        Raises:
            FileNotFoundError: If checkpoint file does not exist.
        """
        ckpt = (
            Path(checkpoint_path)
            if checkpoint_path is not None
            else self.last_weights_path
        )
        if not ckpt.exists():
            msg = f"Checkpoint for resuming not found: {ckpt}"
            logger.error(msg)
            raise FileNotFoundError(msg)

        logger.info(f"Resuming YOLO training from checkpoint: {ckpt}")
        return self.wrapper.train(resume=True, model=str(ckpt), **kwargs)

    def save_best(self, target_dir: Union[str, Path]) -> Path:
        """Save the best model checkpoint weights to a target directory.

        Args:
            target_dir: Target directory path where best.pt will be saved.

        Returns:
            Path object to destination saved best checkpoint file.

        Raises:
            FileNotFoundError: If best.pt checkpoint file does not exist yet.
        """
        dest_dir = Path(target_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_file = dest_dir / "best.pt"

        if not self.best_weights_path.exists():
            msg = f"Best weights file not found at: {self.best_weights_path}"
            logger.error(msg)
            raise FileNotFoundError(msg)

        shutil.copy2(self.best_weights_path, dest_file)
        logger.info(f"Saved best weights checkpoint to: {dest_file}")
        return dest_file
