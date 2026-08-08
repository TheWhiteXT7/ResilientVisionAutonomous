"""Configuration dataclass for laser pattern attack parameters."""

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class AttackConfig:
    """Immutable configuration for laser pattern generation and projection.

    Attributes:
        laser_color: RGB color tuple for the laser (0-255 per channel).
        intensity: Intensity factor of the laser in range [0.0, 1.0].
        alpha: Opacity alpha blending factor in range [0.0, 1.0].
        blur_radius: Radius for Gaussian blur effect (>= 0.0).
        spot_radius: Radius of individual laser spots in pixels (> 0.0).
        max_spots: Maximum number of spots to generate (> 0).
        random_seed: Optional seed for reproducible random spot placement.
        pattern_type: Pattern generation algorithm identifier.
        output_dtype: Data type string for output representation.
        target_class: Object class targeted by 'targeted' attacks (e.g., 'Car').
        missing_target_policy: Behavior when a 'targeted' attack finds no valid
            target: 'preserve' keeps the original image unchanged and continues
            processing, 'fail' raises TargetSelectionError.
    """

    laser_color: Tuple[int, int, int] = (255, 0, 0)
    intensity: float = 1.0
    alpha: float = 0.8
    blur_radius: float = 5.0
    spot_radius: float = 15.0
    max_spots: int = 5
    random_seed: Optional[int] = None
    pattern_type: str = "random"
    output_dtype: str = "uint8"
    target_class: str = "Car"
    missing_target_policy: str = "preserve"

    def __post_init__(self) -> None:
        """Validate all parameters upon dataclass initialization.

        Raises:
            TypeError: If an attribute has an incorrect type.
            ValueError: If an attribute has an out-of-range or invalid value.
        """
        # Validate laser_color
        if not isinstance(self.laser_color, (tuple, list)) or len(self.laser_color) != 3:
            raise TypeError("laser_color must be a tuple or list of 3 integers (RGB).")
        for idx, channel in enumerate(self.laser_color):
            if isinstance(channel, bool) or not isinstance(channel, int):
                raise TypeError(
                    f"laser_color channel {idx} must be an integer, got {type(channel).__name__}."
                )
            if not (0 <= channel <= 255):
                raise ValueError(
                    f"laser_color channel {idx} must be in range [0, 255], got {channel}."
                )

        if isinstance(self.laser_color, list):
            object.__setattr__(self, "laser_color", tuple(self.laser_color))

        # Validate intensity
        if isinstance(self.intensity, bool) or not isinstance(self.intensity, (int, float)):
            raise TypeError("intensity must be a float or int.")
        if not (0.0 <= float(self.intensity) <= 1.0):
            raise ValueError(f"intensity must be between 0.0 and 1.0, got {self.intensity}.")

        # Validate alpha
        if isinstance(self.alpha, bool) or not isinstance(self.alpha, (int, float)):
            raise TypeError("alpha must be a float or int.")
        if not (0.0 <= float(self.alpha) <= 1.0):
            raise ValueError(f"alpha must be between 0.0 and 1.0, got {self.alpha}.")

        # Validate blur_radius
        if isinstance(self.blur_radius, bool) or not isinstance(self.blur_radius, (int, float)):
            raise TypeError("blur_radius must be a float or int.")
        if float(self.blur_radius) < 0.0:
            raise ValueError(f"blur_radius must be non-negative, got {self.blur_radius}.")

        # Validate spot_radius
        if isinstance(self.spot_radius, bool) or not isinstance(self.spot_radius, (int, float)):
            raise TypeError("spot_radius must be a float or int.")
        if float(self.spot_radius) <= 0.0:
            raise ValueError(f"spot_radius must be greater than 0, got {self.spot_radius}.")

        # Validate max_spots
        if isinstance(self.max_spots, bool) or not isinstance(self.max_spots, int):
            raise TypeError("max_spots must be an integer.")
        if self.max_spots <= 0:
            raise ValueError(f"max_spots must be greater than 0, got {self.max_spots}.")

        # Validate random_seed
        if self.random_seed is not None:
            if isinstance(self.random_seed, bool) or not isinstance(self.random_seed, int):
                raise TypeError("random_seed must be an integer or None.")

        # Validate pattern_type
        if not isinstance(self.pattern_type, str):
            raise TypeError("pattern_type must be a string.")
        if not self.pattern_type.strip():
            raise ValueError("pattern_type cannot be empty.")

        # Validate output_dtype
        if not isinstance(self.output_dtype, str):
            raise TypeError("output_dtype must be a string.")
        if not self.output_dtype.strip():
            raise ValueError("output_dtype cannot be empty.")

        # Validate target_class
        if isinstance(self.target_class, bool) or not isinstance(self.target_class, str):
            raise TypeError("target_class must be a string.")
        if not self.target_class.strip():
            raise ValueError("target_class cannot be empty.")

        # Validate missing_target_policy
        if isinstance(self.missing_target_policy, bool) or not isinstance(self.missing_target_policy, str):
            raise TypeError("missing_target_policy must be a string.")
        policy = self.missing_target_policy.strip().lower()
        if policy not in ("preserve", "fail"):
            raise ValueError(
                f"missing_target_policy must be one of ('preserve', 'fail'), "
                f"got '{self.missing_target_policy}'."
            )
        object.__setattr__(self, "missing_target_policy", policy)
