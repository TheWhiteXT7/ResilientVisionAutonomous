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
        rolling_shutter_start: Laser position (x, y) in pixels at the start of
            the frame readout.
        rolling_shutter_velocity: Laser velocity (x, y) in pixels per timing
            unit during the frame readout.
        row_readout_time: Delay between the start of adjacent row exposures.
        row_exposure_time: Duration of each row exposure, in the same timing
            units as row_readout_time.
        beam_width: Gaussian beam spatial standard deviation in pixels.
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
    rolling_shutter_start: Tuple[float, float] = (0.0, 0.0)
    rolling_shutter_velocity: Tuple[float, float] = (0.0, 0.0)
    row_readout_time: float = 1.0
    row_exposure_time: float = 1.0
    beam_width: float = 5.0
    rolling_shutter_artifact_start: Tuple[float, float] = (620.0, 180.0)
    rolling_shutter_artifact_velocity: Tuple[float, float] = (6.0, 0.0)
    rolling_shutter_artifact_power: float = 1.25
    rolling_shutter_artifact_beam_sigma: float = 10.0
    rolling_shutter_artifact_row_readout_time: float = 0.02
    rolling_shutter_artifact_row_exposure_time: float = 0.08
    rolling_shutter_artifact_temporal_samples: int = 8
    rolling_shutter_artifact_power_profile: str = "constant"
    rolling_shutter_artifact_saturation_level: float = 0.75
    rolling_shutter_artifact_bloom_strength: float = 0.35
    rolling_shutter_artifact_bloom_sigma: float = 8.0
    rolling_shutter_artifact_smear_strength: float = 0.8
    rolling_shutter_artifact_smear_length: int = 72
    rolling_shutter_artifact_spectral_response: Tuple[float, float, float] = (0.72, 1.0, 0.38)
    rolling_shutter_artifact_noise_strength: float = 0.0
    rolling_shutter_artifact_strength: float = 1.0
    rolling_shutter_artifact_frequency_regime: str = "freq_mid_narrow"
    rolling_shutter_artifact_beam_profile: str = "full_frame_gaussian"
    rolling_shutter_artifact_temporal_frequency: float = 3.0
    rolling_shutter_artifact_temporal_modulation_amplitude: float = 0.55
    rolling_shutter_artifact_aliasing_frequency: float = 0.0
    rolling_shutter_artifact_aliasing_amplitude: float = 0.0
    rolling_shutter_artifact_spatial_modulation_frequency: float = 1.0
    rolling_shutter_artifact_random_disturbance_amplitude: float = 0.0
    rolling_shutter_artifact_nonlinearity: float = 1.0

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

        for name in ("rolling_shutter_start", "rolling_shutter_velocity"):
            value = getattr(self, name)
            if not isinstance(value, (tuple, list)) or len(value) != 2:
                raise TypeError(f"{name} must be a tuple or list of 2 numbers.")
            if any(isinstance(component, bool) or not isinstance(component, (int, float)) for component in value):
                raise TypeError(f"{name} components must be numbers.")
            object.__setattr__(self, name, (float(value[0]), float(value[1])))

        for name in ("row_readout_time", "row_exposure_time", "beam_width"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a float or int.")
            if float(value) <= 0.0:
                raise ValueError(f"{name} must be greater than 0, got {value}.")

        for name in ("rolling_shutter_artifact_start", "rolling_shutter_artifact_velocity"):
            value = getattr(self, name)
            if not isinstance(value, (tuple, list)) or len(value) != 2:
                raise TypeError(f"{name} must be a tuple or list of 2 numbers.")
            if any(isinstance(component, bool) or not isinstance(component, (int, float)) for component in value):
                raise TypeError(f"{name} components must be numbers.")
            object.__setattr__(self, name, (float(value[0]), float(value[1])))
        spectrum = self.rolling_shutter_artifact_spectral_response
        if not isinstance(spectrum, (tuple, list)) or len(spectrum) != 3:
            raise TypeError("rolling_shutter_artifact_spectral_response must be a tuple or list of 3 numbers.")
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) or float(value) < 0.0 for value in spectrum):
            raise ValueError("rolling_shutter_artifact_spectral_response values must be non-negative numbers.")
        if not any(float(value) > 0.0 for value in spectrum):
            raise ValueError("rolling_shutter_artifact_spectral_response must contain a positive value.")
        object.__setattr__(self, "rolling_shutter_artifact_spectral_response", tuple(float(value) for value in spectrum))
        for name in (
            "rolling_shutter_artifact_power", "rolling_shutter_artifact_beam_sigma",
            "rolling_shutter_artifact_row_readout_time", "rolling_shutter_artifact_row_exposure_time",
            "rolling_shutter_artifact_saturation_level", "rolling_shutter_artifact_bloom_strength",
            "rolling_shutter_artifact_bloom_sigma", "rolling_shutter_artifact_smear_strength",
            "rolling_shutter_artifact_noise_strength",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a float or int.")
            if float(value) < 0.0:
                raise ValueError(f"{name} must be non-negative, got {value}.")
        if self.rolling_shutter_artifact_beam_sigma == 0 or self.rolling_shutter_artifact_row_readout_time == 0 or self.rolling_shutter_artifact_row_exposure_time == 0 or self.rolling_shutter_artifact_saturation_level == 0:
            raise ValueError("artifact beam sigma, timing, and saturation level must be greater than 0.")
        if isinstance(self.rolling_shutter_artifact_temporal_samples, bool) or not isinstance(self.rolling_shutter_artifact_temporal_samples, int):
            raise TypeError("rolling_shutter_artifact_temporal_samples must be an integer.")
        if self.rolling_shutter_artifact_temporal_samples <= 0:
            raise ValueError("rolling_shutter_artifact_temporal_samples must be greater than 0.")
        if isinstance(self.rolling_shutter_artifact_smear_length, bool) or not isinstance(self.rolling_shutter_artifact_smear_length, int):
            raise TypeError("rolling_shutter_artifact_smear_length must be an integer.")
        if self.rolling_shutter_artifact_smear_length <= 0:
            raise ValueError("rolling_shutter_artifact_smear_length must be greater than 0.")
        profile = self.rolling_shutter_artifact_power_profile
        if not isinstance(profile, str):
            raise TypeError("rolling_shutter_artifact_power_profile must be a string.")
        profile = profile.strip().lower()
        if profile not in ("constant", "gaussian_pulse"):
            raise ValueError("rolling_shutter_artifact_power_profile must be 'constant' or 'gaussian_pulse'.")
        object.__setattr__(self, "rolling_shutter_artifact_power_profile", profile)
        for name in (
            "rolling_shutter_artifact_strength", "rolling_shutter_artifact_temporal_frequency",
            "rolling_shutter_artifact_temporal_modulation_amplitude", "rolling_shutter_artifact_aliasing_frequency",
            "rolling_shutter_artifact_aliasing_amplitude", "rolling_shutter_artifact_spatial_modulation_frequency",
            "rolling_shutter_artifact_random_disturbance_amplitude", "rolling_shutter_artifact_nonlinearity",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a float or int.")
            if float(value) < 0.0:
                raise ValueError(f"{name} must be non-negative, got {value}.")
        if not isinstance(self.rolling_shutter_artifact_frequency_regime, str) or self.rolling_shutter_artifact_frequency_regime.strip() not in (
            "freq_low_wide", "freq_mid_narrow", "freq_high_fine", "freq_ultra_aliasing", "freq_random_full"
        ):
            raise ValueError("rolling_shutter_artifact_frequency_regime is not supported.")
        if not isinstance(self.rolling_shutter_artifact_beam_profile, str) or self.rolling_shutter_artifact_beam_profile.strip() not in ("full_frame_gaussian", "full_frame_flat"):
            raise ValueError("rolling_shutter_artifact_beam_profile must be 'full_frame_gaussian' or 'full_frame_flat'.")
