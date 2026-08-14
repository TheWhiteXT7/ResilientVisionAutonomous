"""Full-frame rolling-shutter optical integration and CMOS response simulation."""

from typing import Dict, Tuple

import numpy as np
from PIL import Image

from .attack_config import AttackConfig


def _gaussian_blur(field: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return field.copy()
    radius = max(1, int(np.ceil(3 * sigma)))
    coordinate = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-0.5 * (coordinate / sigma) ** 2)
    kernel /= kernel.sum()
    horizontal = np.apply_along_axis(lambda row: np.convolve(np.pad(row, (radius, radius), mode="edge"), kernel, mode="valid"), 1, field)
    return np.apply_along_axis(lambda col: np.convolve(np.pad(col, (radius, radius), mode="edge"), kernel, mode="valid"), 0, horizontal).astype(np.float32)


def _horizontal_box_blur(field: np.ndarray, length: int) -> np.ndarray:
    left, right = length // 2, length - 1 - length // 2
    padded = np.pad(field, ((0, 0), (left, right)), mode="edge")
    cumulative = np.pad(np.cumsum(padded, axis=1), ((0, 0), (0, 1)))
    return (cumulative[:, length:] - cumulative[:, :-length]) / float(length)


def _full_frame_profile(width: int, height: int, config: AttackConfig) -> np.ndarray:
    """Wide optical beam and moving spatial phase; never a localized spot."""
    x = np.linspace(-1.0, 1.0, width, dtype=np.float32)
    y = np.linspace(-1.0, 1.0, height, dtype=np.float32)
    start_x, start_y = config.rolling_shutter_artifact_start
    direction_x, direction_y = config.rolling_shutter_artifact_velocity
    centre_x = np.clip((start_x / max(width - 1, 1)) * 2 - 1 + direction_x * 0.001, -1.5, 1.5)
    centre_y = np.clip((start_y / max(height - 1, 1)) * 2 - 1 + direction_y * 0.001, -1.5, 1.5)
    sigma = max(float(config.rolling_shutter_artifact_beam_sigma) / max(width, height), 0.05)
    if config.rolling_shutter_artifact_beam_profile == "full_frame_flat":
        envelope = np.ones((height, width), dtype=np.float32)
    else:
        envelope = 0.45 + 0.55 * np.exp(-((x[None, :] - centre_x) ** 2 + (y[:, None] - centre_y) ** 2) / (2 * sigma ** 2))
    phase = 2 * np.pi * float(config.rolling_shutter_artifact_spatial_modulation_frequency) * (x[None, :] + 0.20 * y[:, None])
    return (envelope * (0.72 + 0.28 * np.sin(phase) ** 2)).astype(np.float32)


def integrate_rolling_shutter_irradiance(width: int, height: int, config: AttackConfig) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Integrate time-varying full-frame irradiance separately for every row."""
    samples = config.rolling_shutter_artifact_temporal_samples
    rows = np.arange(height, dtype=np.float32)
    row_start = rows * float(config.rolling_shutter_artifact_row_readout_time)
    offsets = (np.arange(samples, dtype=np.float32) + 0.5) / samples
    profile = _full_frame_profile(width, height, config)
    integrated = np.zeros((height, width), dtype=np.float32)
    row_signal = np.zeros(height, dtype=np.float32)
    trajectory_x, trajectory_y = config.rolling_shutter_artifact_velocity
    rng = np.random.default_rng(config.random_seed)
    random_phase = rng.uniform(0, 2 * np.pi, 4)
    for offset in offsets:
        times = row_start + offset * float(config.rolling_shutter_artifact_row_exposure_time)
        temporal = 1.0 + float(config.rolling_shutter_artifact_temporal_modulation_amplitude) * np.sin(2 * np.pi * float(config.rolling_shutter_artifact_temporal_frequency) * times + random_phase[0])
        alias = float(config.rolling_shutter_artifact_aliasing_amplitude) * np.sin(2 * np.pi * float(config.rolling_shutter_artifact_aliasing_frequency) * times + random_phase[1])
        trajectory_phase = 0.04 * np.sin((trajectory_x + trajectory_y) * times + random_phase[2])
        signal = np.maximum(temporal + alias + trajectory_phase, 0.0)
        integrated += signal[:, None] * profile
        row_signal += signal
    integrated /= samples
    row_signal /= samples
    if config.rolling_shutter_artifact_random_disturbance_amplitude:
        # Low-dimensional random temporal/spatial modes: structured, not pixel noise.
        x = np.linspace(0, 1, width, dtype=np.float32)
        random_rows = sum(rng.uniform(-1, 1) * np.sin(2 * np.pi * frequency * rows / height + phase) for frequency, phase in ((3, random_phase[0]), (11, random_phase[1]), (29, random_phase[2])))
        random_cols = sum(rng.uniform(-1, 1) * np.sin(2 * np.pi * frequency * x + phase) for frequency, phase in ((1, random_phase[1]), (5, random_phase[3])))
        structured = random_rows[:, None] * (0.6 + 0.4 * random_cols[None, :])
        integrated *= np.clip(1 + float(config.rolling_shutter_artifact_random_disturbance_amplitude) * structured, 0.05, 2.0)
    frequency_artifact = integrated - _gaussian_blur(integrated, max(height / 20.0, 1.0))
    return np.maximum(integrated, 0.0).astype(np.float32), {"temporal_integration": np.repeat(row_signal[:, None], width, axis=1), "frequency_artifact": frequency_artifact}


class SensorResponseModel:
    """CMOS response with full-well clipping, bloom, and row charge smear."""

    def render(self, image: Image.Image, irradiance: np.ndarray, config: AttackConfig) -> Tuple[Image.Image, Dict[str, np.ndarray]]:
        source_mode = image.mode
        rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        energy = irradiance * float(config.rolling_shutter_artifact_power) * float(config.rolling_shutter_artifact_strength)
        saturation_level = float(config.rolling_shutter_artifact_saturation_level)
        response = saturation_level * (1 - np.exp(-float(config.rolling_shutter_artifact_nonlinearity) * energy / saturation_level))
        saturation = np.clip((energy - 0.85 * saturation_level) / saturation_level, 0.0, 1.0)
        bloom = _gaussian_blur(saturation, float(config.rolling_shutter_artifact_bloom_sigma)) * float(config.rolling_shutter_artifact_bloom_strength)
        smear = _horizontal_box_blur(response + bloom, int(config.rolling_shutter_artifact_smear_length)) * float(config.rolling_shutter_artifact_smear_strength)
        sensor_energy = response + bloom + smear
        spectrum = np.asarray(config.rolling_shutter_artifact_spectral_response, dtype=np.float32)
        # Sensor energy is additive photocurrent, with a weak multiplicative
        # response that retains scene detail under broad full-frame exposure.
        corrupted = np.clip(
            rgb * (1 + 0.18 * sensor_energy[:, :, None])
            + sensor_energy[:, :, None] * spectrum[None, None, :] * 0.45
            + saturation[:, :, None] * spectrum[None, None, :] * 0.35,
            0, 1,
        )
        if config.rolling_shutter_artifact_noise_strength:
            rng = np.random.default_rng(config.random_seed)
            corrupted = np.clip(corrupted + rng.normal(0, float(config.rolling_shutter_artifact_noise_strength), corrupted.shape), 0, 1)
        output = Image.fromarray(np.rint(corrupted * 255).astype(np.uint8), "RGB")
        return (output.convert(source_mode) if source_mode in ("RGB", "L") else output.convert("RGBA"), {
            "sensor_response": response, "saturation": saturation, "bloom": bloom, "charge_smear": smear,
        })
