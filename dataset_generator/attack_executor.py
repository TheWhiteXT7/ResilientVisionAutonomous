"""In-memory attack executor connecting KittiSample to AttackPipeline."""

from datetime import datetime, timezone
import time
from typing import Any, Dict, Optional, Tuple
from PIL import Image

from attack_engine.attack_config import AttackConfig
from attack_engine.attack_pipeline import AttackPipeline
from attack_engine.laser_pattern import LaserPattern
from dataset_loader.kitti_loader import KittiSample

TARGETED_PATTERN_TYPES = ("targeted", "targeted_spots")


class AttackExecutor:
    """Executes laser pattern attacks on KittiSample instances in memory without writing files."""

    def __init__(
        self,
        pipeline: Optional[AttackPipeline] = None,
        attack_config: Optional[AttackConfig] = None,
    ) -> None:
        """Initialize AttackExecutor.

        Args:
            pipeline: Optional AttackPipeline instance.
            attack_config: Optional AttackConfig instance.

        Raises:
            TypeError: If pipeline or attack_config are invalid types.
        """
        if pipeline is not None and not isinstance(pipeline, AttackPipeline):
            raise TypeError(f"pipeline must be an AttackPipeline instance, got {type(pipeline).__name__}.")
        if attack_config is not None and not isinstance(attack_config, AttackConfig):
            raise TypeError(f"attack_config must be an AttackConfig instance, got {type(attack_config).__name__}.")

        if pipeline is not None:
            self.pipeline = pipeline
        elif attack_config is not None:
            self.pipeline = AttackPipeline(config=attack_config)
        else:
            self.pipeline = AttackPipeline()

    def execute(
        self,
        sample: KittiSample,
        pattern_type: str = "random",
        target_class: str = "Car",
        **kwargs: Any,
    ) -> Tuple[Image.Image, LaserPattern, Dict[str, Any]]:
        """Apply laser pattern attack to a KittiSample in memory.

        Args:
            sample: KittiSample object containing image/path data.
            pattern_type: Identifier for pattern generation algorithm.
            target_class: Target class used by 'targeted' attacks (default 'Car').
            **kwargs: Additional parameter overrides for AttackConfig.

        Returns:
            Tuple of (attacked_image, laser_pattern, execution_metadata_dict).

        Raises:
            TypeError: If sample is not an instance of KittiSample.
            ValueError: If a targeted attack has no valid target annotation.
        """
        if not isinstance(sample, KittiSample):
            raise TypeError(f"sample must be a KittiSample instance, got {type(sample).__name__}.")

        start_time = time.perf_counter()

        # Load PIL Image from KittiSample
        image = sample.image if sample.image is not None else sample.load_image()

        # Thread parsed KittiSample annotations into targeted attacks only, so
        # random attacks receive no target information and behave as before.
        pipeline_kwargs = dict(kwargs)
        if str(pattern_type).strip().lower() in TARGETED_PATTERN_TYPES:
            pipeline_kwargs["annotations"] = list(sample.annotations)
            pipeline_kwargs["target_class"] = target_class

        # Delegate attack execution to AttackPipeline
        attacked_image, laser_pattern = self.pipeline.execute(
            image=image,
            pattern_type=pattern_type,
            **pipeline_kwargs,
        )

        end_time = time.perf_counter()
        elapsed_ms = (end_time - start_time) * 1000.0

        target_info: Optional[Dict[str, Any]] = None
        last_target = getattr(self.pipeline, "last_target", None)
        if last_target is not None:
            target_info = {
                "class_name": last_target.class_name,
                "bbox": list(last_target.bbox),
            }

        execution_metadata = {
            "sample_id": sample.sample_id,
            "pattern_type": pattern_type,
            "spots_count": len(laser_pattern),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "processing_time_ms": round(elapsed_ms, 3),
            "image_size": image.size,
            "target": target_info,
        }

        return attacked_image, laser_pattern, execution_metadata
