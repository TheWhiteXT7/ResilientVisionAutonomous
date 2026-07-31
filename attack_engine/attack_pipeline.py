"""High-level attack pipeline orchestrating pattern generation and projection."""

from typing import Any, Optional, Tuple
from PIL import Image

from .attack_config import AttackConfig
from .laser_pattern import LaserPattern
from .pattern_generator import PatternGenerator
from .projection_engine import ProjectionEngine


class AttackPipeline:
    """Orchestrates pattern generation and image rendering into a single workflow."""

    def __init__(self, config: Optional[AttackConfig] = None) -> None:
        """Initialize AttackPipeline.

        Args:
            config: Optional AttackConfig instance. If None, default config is used.
        """
        self.config = config or AttackConfig()
        self.projection_engine = ProjectionEngine()

    def execute(
        self,
        image: Image.Image,
        pattern_type: str = "random",
        **kwargs: Any,
    ) -> Tuple[Image.Image, LaserPattern]:
        """Generate pattern and render it onto the input image.

        Args:
            image: Source PIL Image.
            pattern_type: Type of pattern ("random", "single", "horizontal_line", etc.).
            **kwargs: Overrides for AttackConfig parameters.

        Returns:
            Tuple of (attacked_image, laser_pattern).

        Raises:
            TypeError: If image is not a PIL Image.
        """
        if not isinstance(image, Image.Image):
            raise TypeError(f"image must be a PIL.Image.Image instance, got {type(image).__name__}.")

        current_config = self.config
        if kwargs or pattern_type != current_config.pattern_type:
            config_dict = {
                "laser_color": current_config.laser_color,
                "intensity": current_config.intensity,
                "alpha": current_config.alpha,
                "blur_radius": current_config.blur_radius,
                "spot_radius": current_config.spot_radius,
                "max_spots": current_config.max_spots,
                "random_seed": current_config.random_seed,
                "pattern_type": pattern_type,
                "output_dtype": current_config.output_dtype,
            }
            config_dict.update(kwargs)
            current_config = AttackConfig(**config_dict)

        width, height = image.size
        generator = PatternGenerator(width, height, current_config)
        pattern = generator.generate(pattern_type)
        attacked_image = self.projection_engine.render(image, pattern, current_config)

        return attacked_image, pattern


def apply_attack(
    image: Image.Image,
    pattern_type: str = "random",
    config: Optional[AttackConfig] = None,
    **kwargs: Any,
) -> Tuple[Image.Image, LaserPattern]:
    """Public top-level API function for applying laser pattern attacks.

    Pipeline:
        Generate Pattern -> Render Pattern -> Return (attacked_image, laser_pattern)

    Args:
        image: Source PIL Image.
        pattern_type: Type of pattern to generate and apply.
        config: Optional AttackConfig instance.
        **kwargs: Optional AttackConfig parameter overrides.

    Returns:
        Tuple of (attacked_image, laser_pattern).
    """
    pipeline = AttackPipeline(config=config)
    return pipeline.execute(image, pattern_type=pattern_type, **kwargs)
