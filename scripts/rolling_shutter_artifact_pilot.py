"""Generate a diagnostic full-frame rolling-shutter sensor-artifact pilot."""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from attack_engine import AttackConfig, AttackPipeline  # noqa: E402


def _visualize(field: np.ndarray) -> Image.Image:
    field = np.asarray(field, dtype=np.float32)
    maximum = float(field.max())
    normalized = field / maximum if maximum > 1e-8 else field
    return Image.fromarray(np.rint(np.clip(normalized, 0, 1) * 255).astype(np.uint8), "L").convert("RGB")


def _save_stages(directory: Path, clean: Image.Image, attacked: Image.Image, pattern) -> None:
    diagnostics = pattern.diagnostics
    stages = [
        ("01_clean.png", clean), ("02_irradiance.png", _visualize(pattern.irradiance)),
        ("03_temporal_integration.png", _visualize(diagnostics["temporal_integration"])),
        ("04_sensor_response.png", _visualize(diagnostics["sensor_response"])),
        ("05_saturation.png", _visualize(diagnostics["saturation"])),
        ("06_bloom.png", _visualize(diagnostics["bloom"])),
        ("07_charge_smear.png", _visualize(diagnostics["charge_smear"])),
        ("08_frequency_artifact.png", _visualize(np.abs(diagnostics["frequency_artifact"]))),
        ("09_final_attacked.png", attacked),
    ]
    for name, image in stages:
        image.save(directory / name)
    difference = np.abs(np.asarray(attacked, dtype=np.int16) - np.asarray(clean, dtype=np.int16)).astype(np.uint8)
    Image.fromarray(difference, "RGB").save(directory / "10_difference.png")


def _stats(clean: Image.Image, attacked: Image.Image, pattern) -> dict:
    delta = np.abs(np.asarray(attacked, dtype=np.float32) - np.asarray(clean, dtype=np.float32))
    luminance = delta.mean(axis=2)
    frequency = np.abs(np.fft.rfft(luminance - luminance.mean(), axis=0)).mean(axis=1)
    return {
        "mean_absolute_difference": float(delta.mean()), "affected_pixel_percent": float((luminance > 2).mean() * 100),
        "row_variance": float(luminance.mean(axis=1).var()), "column_variance": float(luminance.mean(axis=0).var()),
        "high_frequency_energy": float(frequency[len(frequency) // 4:].mean()),
        "saturation_percent": float((np.asarray(attacked) >= 250).any(axis=2).mean() * 100),
        "metadata": pattern.metadata,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output-dir", default="outputs/rolling_shutter_fullframe_pilot")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    clean = Image.open(args.image).convert("RGB")
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    base = {
        "pattern_type": "rolling_shutter_artifact", "random_seed": args.seed,
        "rolling_shutter_artifact_start": (clean.width * 0.35, clean.height * 0.45),
        "rolling_shutter_artifact_velocity": (4.0, 0.7), "rolling_shutter_artifact_power": 0.6,
        "rolling_shutter_artifact_beam_profile": "full_frame_gaussian", "rolling_shutter_artifact_beam_sigma": max(clean.width, clean.height) * 0.75,
        "rolling_shutter_artifact_row_readout_time": 0.010, "rolling_shutter_artifact_row_exposure_time": 0.035,
        "rolling_shutter_artifact_temporal_samples": 16, "rolling_shutter_artifact_saturation_level": 0.72,
        "rolling_shutter_artifact_bloom_strength": 0.30, "rolling_shutter_artifact_bloom_sigma": 10.0,
        "rolling_shutter_artifact_smear_strength": 0.9, "rolling_shutter_artifact_smear_length": 80,
        "rolling_shutter_artifact_spectral_response": (0.92, 0.94, 1.0), "rolling_shutter_artifact_noise_strength": 0.0,
        "rolling_shutter_artifact_nonlinearity": 1.0,
    }
    categories = {
        "freq_low_wide": {"rolling_shutter_artifact_temporal_frequency": 0.35, "rolling_shutter_artifact_temporal_modulation_amplitude": 0.62, "rolling_shutter_artifact_spatial_modulation_frequency": 0.35, "rolling_shutter_artifact_aliasing_frequency": 0, "rolling_shutter_artifact_aliasing_amplitude": 0},
        "freq_mid_narrow": {"rolling_shutter_artifact_temporal_frequency": 2.2, "rolling_shutter_artifact_temporal_modulation_amplitude": 0.72, "rolling_shutter_artifact_spatial_modulation_frequency": 1.5, "rolling_shutter_artifact_beam_sigma": max(clean.width, clean.height) * 0.52, "rolling_shutter_artifact_aliasing_frequency": 0, "rolling_shutter_artifact_aliasing_amplitude": 0},
        "freq_high_fine": {"rolling_shutter_artifact_temporal_frequency": 12.0, "rolling_shutter_artifact_temporal_modulation_amplitude": 0.82, "rolling_shutter_artifact_spatial_modulation_frequency": 3.5, "rolling_shutter_artifact_row_exposure_time": 0.014, "rolling_shutter_artifact_aliasing_frequency": 0, "rolling_shutter_artifact_aliasing_amplitude": 0},
        "freq_ultra_aliasing": {"rolling_shutter_artifact_temporal_frequency": 34.0, "rolling_shutter_artifact_temporal_modulation_amplitude": 0.72, "rolling_shutter_artifact_spatial_modulation_frequency": 5.0, "rolling_shutter_artifact_row_exposure_time": 0.006, "rolling_shutter_artifact_aliasing_frequency": 71.0, "rolling_shutter_artifact_aliasing_amplitude": 0.50},
        "freq_random_full": {"rolling_shutter_artifact_temporal_frequency": 8.0, "rolling_shutter_artifact_temporal_modulation_amplitude": 0.55, "rolling_shutter_artifact_spatial_modulation_frequency": 2.5, "rolling_shutter_artifact_random_disturbance_amplitude": 0.55, "rolling_shutter_artifact_aliasing_frequency": 17.0, "rolling_shutter_artifact_aliasing_amplitude": 0.24},
    }
    strengths = {"low": 0.45, "medium": 0.85, "strong": 1.30}
    report, montage_images = {}, [("clean", clean)]
    clean_dir = root / "clean"
    clean_dir.mkdir(exist_ok=True)
    clean.save(clean_dir / "01_clean.png")
    for category, settings in categories.items():
        report[category] = {}
        for label, strength in strengths.items():
            directory = root / category / label
            directory.mkdir(parents=True, exist_ok=True)
            config = AttackConfig(**(base | settings | {"rolling_shutter_artifact_frequency_regime": category, "rolling_shutter_artifact_strength": strength}))
            attacked, pattern = AttackPipeline(config).execute(clean, "rolling_shutter_artifact")
            _save_stages(directory, clean, attacked, pattern)
            report[category][label] = {"config": asdict(config), "statistics": _stats(clean, attacked, pattern)}
            if label == "medium": montage_images.append((category, attacked))
    width, height = clean.size
    montage = Image.new("RGB", (width * 6, height + 28), "white")
    draw = ImageDraw.Draw(montage)
    for index, (name, image) in enumerate(montage_images):
        montage.paste(image, (index * width, 28))
        draw.text((index * width + 4, 5), name, fill="black")
    montage_dir = root / "montages"
    montage_dir.mkdir(exist_ok=True)
    montage.save(montage_dir / "comparison_montage.png")
    (root / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: {level: values["statistics"] for level, values in data.items()} for key, data in report.items()}, indent=2))


if __name__ == "__main__":
    main()
