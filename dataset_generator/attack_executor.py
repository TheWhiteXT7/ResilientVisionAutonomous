"""In-memory attack executor connecting KittiSample to AttackPipeline."""

from datetime import datetime, timezone
import time
from typing import Any, Dict, Optional, Tuple
from PIL import Image

from attack_engine.attack_config import AttackConfig
from attack_engine.attack_pipeline import AttackPipeline
from attack_engine.laser_pattern import LaserPattern
from attack_engine.target_selection import TargetSelectionError
from dataset_loader.kitti_loader import KittiSample

TARGETED_PATTERN_TYPES = ("targeted", "targeted_spots")
MISSING_TARGET_POLICIES = ("preserve", "fail")


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
        missing_target_policy: Optional[str] = None,
        **kwargs: Any,
    ) -> Tuple[Image.Image, LaserPattern, Dict[str, Any]]:
        """Apply laser pattern attack to a KittiSample in memory.

        For 'targeted' patterns, a missing valid target is handled according to
        ``missing_target_policy``:
        - 'preserve' (default): returns the original image unchanged with an
          empty LaserPattern so downstream processing (label/calib/metadata)
          continues normally; no fallback to random or single attack.
        - 'fail': re-raises the TargetSelectionError from the pipeline.

        Args:
            sample: KittiSample object containing image/path data.
            pattern_type: Identifier for pattern generation algorithm.
            target_class: Target class used by 'targeted' attacks (default 'Car').
            missing_target_policy: Optional policy for missing targets; defaults
                to pipeline.config.missing_target_policy ('preserve').
            **kwargs: Additional parameter overrides for AttackConfig.

        Returns:
            Tuple of (attacked_image, laser_pattern, execution_metadata_dict).
            For preserved samples, attacked_image is the original image and the
            metadata records target_found=False / preserved=True.

        Raises:
            TypeError: If sample is not an instance of KittiSample.
            ValueError: If missing_target_policy is unsupported.
            TargetSelectionError: If a targeted attack finds no valid target and
                the missing_target_policy is 'fail'.
        """
        if not isinstance(sample, KittiSample):
            raise TypeError(f"sample must be a KittiSample instance, got {type(sample).__name__}.")

        if missing_target_policy is None:
            missing_target_policy = getattr(self.pipeline.config, "missing_target_policy", "preserve")
        missing_target_policy = str(missing_target_policy).strip().lower()
        if missing_target_policy not in MISSING_TARGET_POLICIES:
            raise ValueError(
                f"missing_target_policy must be one of {MISSING_TARGET_POLICIES}, "
                f"got '{missing_target_policy}'."
            )

        is_targeted = str(pattern_type).strip().lower() in TARGETED_PATTERN_TYPES

        start_time = time.perf_counter()

        # Load PIL Image from KittiSample
        image = sample.image if sample.image is not None else sample.load_image()

        # Thread parsed KittiSample annotations into targeted attacks only, so
        # random attacks receive no target information and behave as before.
        pipeline_kwargs = dict(kwargs)
        if is_targeted:
            pipeline_kwargs["annotations"] = list(sample.annotations)
            pipeline_kwargs["target_class"] = target_class

        target_found: Optional[bool] = None
        preserved = False
        try:
            # Delegate attack execution to AttackPipeline
            attacked_image, laser_pattern = self.pipeline.execute(
                image=image,
                pattern_type=pattern_type,
                **pipeline_kwargs,
            )
        except TargetSelectionError:
            if missing_target_policy != "preserve":
                raise
            attacked_image = image
            laser_pattern = LaserPattern()
            target_found = False
            preserved = True
        else:
            if is_targeted:
                target_found = True

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
            "target_found": target_found,
            "preserved": preserved,
        }

        return attacked_image, laser_pattern, execution_metadata
