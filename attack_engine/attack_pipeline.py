"""High-level attack pipeline orchestrating pattern generation and projection."""

from typing import Any, Optional, Sequence, Tuple
from PIL import Image

from .attack_config import AttackConfig
from .laser_pattern import LaserPattern
from .pattern_generator import PatternGenerator
from .projection_engine import ProjectionEngine
from .target_selection import TargetRegion, select_target

TARGETED_PATTERN_TYPES = ("targeted", "targeted_spots")


class AttackPipeline:
    """Orchestrates pattern generation and image rendering into a single workflow."""

    def __init__(self, config: Optional[AttackConfig] = None) -> None:
        """Initialize AttackPipeline.

        Args:
            config: Optional AttackConfig instance. If None, default config is used.
        """
        self.config = config or AttackConfig()
        self.projection_engine = ProjectionEngine()
        self.last_target: Optional[TargetRegion] = None

    def execute(
        self,
        image: Image.Image,
        pattern_type: str = "random",
        annotations: Optional[Sequence[Any]] = None,
        target_class: Optional[str] = None,
        target_region: Optional[TargetRegion] = None,
        **kwargs: Any,
    ) -> Tuple[Image.Image, LaserPattern]:
        """Generate pattern and render it onto the input image.

        Args:
            image: Source PIL Image.
            pattern_type: Type of pattern ("random", "single", "horizontal_line",
                "targeted", etc.).
            annotations: Optional sequence of candidate objects (each exposing
                ``class_name`` and ``bbox``) used to select a target for
                'targeted' patterns.
            target_class: Optional target class override; defaults to
                config.target_class.
            target_region: Optional pre-selected TargetRegion for 'targeted'
                patterns (skips selection).
            **kwargs: Overrides for AttackConfig parameters.

        Returns:
            Tuple of (attacked_image, laser_pattern).

        Raises:
            TypeError: If image is not a PIL Image.
            ValueError: If a targeted pattern is requested without annotations
                or a target region, or if no valid target can be selected.
        """
        if not isinstance(image, Image.Image):
            raise TypeError(f"image must be a PIL.Image.Image instance, got {type(image).__name__}.")

        self.last_target = None
        current_config = self.config
        if (
            kwargs
            or pattern_type != current_config.pattern_type
            or (target_class is not None and target_class != current_config.target_class)
        ):
            config_dict = {
                "laser_color": current_config.laser_color,
                "intensity": current_config.intensity,
                "alpha": current_config.alpha,
                "blur_radius": current_config.blur_radius,
                "spot_radius": current_config.spot_radius,
                "max_spots": current_config.max_spots,
                "random_seed": current_config.random_seed,
                "pattern_type": pattern_type,
                "target_class": current_config.target_class if target_class is None else target_class,
                "output_dtype": current_config.output_dtype,
            }
            config_dict.update(kwargs)
            current_config = AttackConfig(**config_dict)

        width, height = image.size
        generator = PatternGenerator(width, height, current_config)

        if pattern_type.strip().lower() in TARGETED_PATTERN_TYPES:
            if target_region is None:
                if not annotations:
                    raise ValueError(
                        "Targeted attack requires 'annotations' or a 'target_region'; got neither."
                    )
                target_region = select_target(
                    annotations,
                    image_size=(width, height),
                    target_class=current_config.target_class,
                    random_seed=current_config.random_seed,
                )
            self.last_target = target_region
            pattern = generator.generate(pattern_type, target=target_region)
        else:
            pattern = generator.generate(pattern_type)

        attacked_image = self.projection_engine.render(image, pattern, current_config)

        return attacked_image, pattern


def apply_attack(
    image: Image.Image,
    pattern_type: str = "random",
    config: Optional[AttackConfig] = None,
    annotations: Optional[Sequence[Any]] = None,
    target_class: Optional[str] = None,
    target_region: Optional[TargetRegion] = None,
    **kwargs: Any,
) -> Tuple[Image.Image, LaserPattern]:
    """Public top-level API function for applying laser pattern attacks.

    Pipeline:
        Generate Pattern -> Render Pattern -> Return (attacked_image, laser_pattern)

    Args:
        image: Source PIL Image.
        pattern_type: Type of pattern to generate and apply.
        config: Optional AttackConfig instance.
        annotations: Optional candidate objects for 'targeted' attacks.
        target_class: Optional target class override for 'targeted' attacks.
        target_region: Optional pre-selected TargetRegion for 'targeted' attacks.
        **kwargs: Optional AttackConfig parameter overrides.

    Returns:
        Tuple of (attacked_image, laser_pattern).
    """
    pipeline = AttackPipeline(config=config)
    return pipeline.execute(
        image,
        pattern_type=pattern_type,
        annotations=annotations,
        target_class=target_class,
        target_region=target_region,
        **kwargs,
    )
