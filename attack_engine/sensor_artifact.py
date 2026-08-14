"""Rolling-shutter laser irradiance integration and sensor response models."""

import numpy as np
from PIL import Image

from .attack_config import AttackConfig


def integrate_rolling_shutter_irradiance(width: int, height: int, config: AttackConfig) -> np.ndarray:
    """Integrate a moving Gaussian irradiance source over every row exposure."""
    samples = config.rolling_shutter_artifact_temporal_samples
    y_grid = np.arange(height, dtype=np.float32)
    x_grid = np.arange(width, dtype=np.float32)
    row_start = y_grid * float(config.rolling_shutter_artifact_row_readout_time)
    offsets = (np.arange(samples, dtype=np.float32) + 0.5) / samples
    irradiance = np.zeros((height, width), dtype=np.float32)
    start_x, start_y = config.rolling_shutter_artifact_start
    velocity_x, velocity_y = config.rolling_shutter_artifact_velocity
    sigma2 = float(config.rolling_shutter_artifact_beam_sigma) ** 2
    frame_end = row_start[-1] + float(config.rolling_shutter_artifact_row_exposure_time)
    for offset in offsets:
        times = row_start + offset * float(config.rolling_shutter_artifact_row_exposure_time)
        profile = np.ones_like(times)
        if config.rolling_shutter_artifact_power_profile == "gaussian_pulse":
            midpoint = frame_end / 2.0
            profile = np.exp(-0.5 * ((times - midpoint) / max(frame_end / 5.0, 1e-6)) ** 2)
        center_x = float(start_x) + float(velocity_x) * times
        center_y = float(start_y) + float(velocity_y) * times
        squared_distance = (x_grid[None, :] - center_x[:, None]) ** 2 + (y_grid - center_y)[:, None] ** 2
        irradiance += profile[:, None] * np.exp(-0.5 * squared_distance / sigma2)
    return irradiance / float(samples)


def _horizontal_box_blur(field: np.ndarray, length: int) -> np.ndarray:
    """Energy-preserving horizontal charge-spread approximation."""
    left = length // 2
    right = length - 1 - left
    padded = np.pad(field, ((0, 0), (left, right)), mode="edge")
    cumulative = np.pad(np.cumsum(padded, axis=1), ((0, 0), (0, 1)))
    return (cumulative[:, length:] - cumulative[:, :-length]) / float(length)


def _gaussian_blur(field: np.ndarray, sigma: float) -> np.ndarray:
    """Small separable float PSF convolution without RGB image operations."""
    if sigma <= 0:
        return field.copy()
    radius = max(1, int(np.ceil(3.0 * sigma)))
    coordinate = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-0.5 * (coordinate / sigma) ** 2)
    kernel /= kernel.sum()
    horizontal = np.apply_along_axis(lambda row: np.convolve(np.pad(row, (radius, radius), mode="edge"), kernel, mode="valid"), 1, field)
    return np.apply_along_axis(lambda column: np.convolve(np.pad(column, (radius, radius), mode="edge"), kernel, mode="valid"), 0, horizontal).astype(np.float32)


class SensorResponseModel:
    """Applies nonlinear full-well response, bloom, and row charge smear."""

    def render(self, image: Image.Image, irradiance: np.ndarray, config: AttackConfig) -> Image.Image:
        source_mode = image.mode
        rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        energy = irradiance * float(config.rolling_shutter_artifact_power)
        saturation = float(config.rolling_shutter_artifact_saturation_level)
        # Compress optical energy before full-well clipping: normal energy is
        # near linear; large energy asymptotically fills the sensor well.
        direct = saturation * (1.0 - np.exp(-energy / saturation))
        overflow = np.maximum(energy - saturation, 0.0)
        bloom = _gaussian_blur(overflow, float(config.rolling_shutter_artifact_bloom_sigma)) * float(config.rolling_shutter_artifact_bloom_strength)
        smear_seed = direct + bloom
        smear = _horizontal_box_blur(smear_seed, int(config.rolling_shutter_artifact_smear_length))
        sensor_energy = direct + bloom + float(config.rolling_shutter_artifact_smear_strength) * smear
        spectral = np.asarray(config.rolling_shutter_artifact_spectral_response, dtype=np.float32)
        corrupted = np.clip(rgb + sensor_energy[:, :, None] * spectral[None, None, :], 0.0, 1.0)
        if config.rolling_shutter_artifact_noise_strength:
            rng = np.random.default_rng(config.random_seed)
            noise = rng.normal(0.0, float(config.rolling_shutter_artifact_noise_strength), corrupted.shape)
            corrupted = np.clip(corrupted + noise.astype(np.float32), 0.0, 1.0)
        output = Image.fromarray(np.rint(corrupted * 255.0).astype(np.uint8), mode="RGB")
        return output.convert(source_mode) if source_mode in ("RGB", "L") else output.convert("RGBA")
