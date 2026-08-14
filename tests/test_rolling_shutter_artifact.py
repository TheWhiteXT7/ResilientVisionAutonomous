"""Tests for the rolling-shutter irradiance-to-sensor-artifact attack."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from attack_engine import AttackConfig, AttackPipeline, RollingShutterArtifactPattern
from attack_engine.target_selection import TargetRegion
from dataset_generator.metadata_writer import MetadataWriter


class TestRollingShutterArtifact(unittest.TestCase):
    def setUp(self) -> None:
        self.image = Image.new("RGB", (128, 80), (48, 68, 92))
        self.config = AttackConfig(
            pattern_type="rolling_shutter_artifact", random_seed=3,
            rolling_shutter_artifact_start=(32.0, 26.0),
            rolling_shutter_artifact_velocity=(1.3, 0.08),
            rolling_shutter_artifact_power=1.5, rolling_shutter_artifact_beam_sigma=7.0,
            rolling_shutter_artifact_row_readout_time=0.08,
            rolling_shutter_artifact_row_exposure_time=0.2,
            rolling_shutter_artifact_temporal_samples=8, rolling_shutter_artifact_bloom_sigma=4.0,
        )

    def attack(self, **overrides):
        return AttackPipeline(self.config).execute(self.image, "rolling_shutter_artifact", **overrides)

    def test_deterministic_dimensions_dtype_and_non_destructive(self):
        original = self.image.tobytes()
        first, pattern = self.attack()
        second, second_pattern = self.attack()
        self.assertIsInstance(pattern, RollingShutterArtifactPattern)
        self.assertEqual(pattern.irradiance.shape, (80, 128))
        self.assertEqual(first.size, self.image.size)
        self.assertEqual(first.mode, "RGB")
        self.assertEqual(first.tobytes(), second.tobytes())
        np.testing.assert_array_equal(pattern.irradiance, second_pattern.irradiance)
        self.assertEqual(self.image.tobytes(), original)
        self.assertGreater(np.mean(np.abs(np.asarray(first, dtype=float) - np.asarray(self.image, dtype=float))), 0.0)

    def test_physical_parameters_change_output(self):
        baseline, _ = self.attack()
        for name, value in (
            ("rolling_shutter_artifact_row_readout_time", 0.03),
            ("rolling_shutter_artifact_velocity", (-0.5, 0.25)),
            ("rolling_shutter_artifact_beam_sigma", 14.0),
            ("rolling_shutter_artifact_power", 3.0),
            ("rolling_shutter_artifact_bloom_strength", 1.0),
            ("rolling_shutter_artifact_smear_length", 12),
            ("rolling_shutter_artifact_temporal_samples", 1),
        ):
            with self.subTest(parameter=name):
                changed, _ = self.attack(**{name: value})
                self.assertNotEqual(baseline.tobytes(), changed.tobytes())

    def test_zero_power_is_clean_and_high_power_saturates(self):
        clean, _ = self.attack(rolling_shutter_artifact_power=0.0)
        strong, _ = self.attack(rolling_shutter_artifact_power=20.0)
        self.assertEqual(clean.tobytes(), self.image.tobytes())
        self.assertGreater((np.asarray(strong) >= 254).sum(), 0)

    def test_validation_dispatch_and_legacy_patterns(self):
        with self.assertRaises(ValueError):
            AttackConfig(rolling_shutter_artifact_temporal_samples=0)
        with self.assertRaises(ValueError):
            AttackConfig(rolling_shutter_artifact_spectral_response=(0, 0, 0))
        pipeline = AttackPipeline(AttackConfig(max_spots=2, random_seed=4))
        _, random_pattern = pipeline.execute(self.image, "random")
        _, targeted_pattern = pipeline.execute(self.image, "targeted_spots", target_region=TargetRegion("Car", (20, 20, 80, 60)))
        self.assertEqual(len(random_pattern), 2)
        self.assertEqual(len(targeted_pattern), 2)

    def test_metadata_has_parameters_not_spots(self):
        _, pattern = self.attack()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "metadata.json"
            MetadataWriter().write_sample_metadata(output, "x", pattern, self.config, {
                "pattern_type": "rolling_shutter_artifact", "image_size": self.image.size,
                "processing_time_ms": 1.0,
            })
            payload = json.loads(output.read_text())
        self.assertEqual(payload["pattern_type"], "rolling_shutter_artifact")
        self.assertNotIn("spots", payload)
        self.assertNotIn("spots_count", payload)
        self.assertEqual(payload["attack_config"]["rolling_shutter_artifact_temporal_samples"], 8)
        self.assertEqual(payload["pattern_metadata"]["representation"], "rolling_shutter_sensor_artifact")

    def test_full_frame_frequency_regimes_and_seeded_randomness(self):
        """Full-frame coverage and physically predicted row-frequency per regime.

        A temporal modulation at frequency f is sampled once per row (row start
        spaced by row_readout_time), so its dominant row-spatial frequency must be
        the aliased fold of f * row_readout_time cycles per row, wrapping at the
        0.5 cycles-per-row Nyquist. Above-Nyquist configurations must therefore
        reappear at high aliased row-frequencies rather than washing out.
        """
        configurations = {
            "freq_low_wide": {"rolling_shutter_artifact_temporal_frequency": 0.4, "rolling_shutter_artifact_spatial_modulation_frequency": 0.4},
            "freq_mid_narrow": {"rolling_shutter_artifact_temporal_frequency": 3.0},
            "freq_high_fine": {"rolling_shutter_artifact_temporal_frequency": 6.25},
            "freq_ultra_aliasing": {"rolling_shutter_artifact_temporal_frequency": 31.25, "rolling_shutter_artifact_aliasing_frequency": 42.5, "rolling_shutter_artifact_aliasing_amplitude": 0.5},
            "freq_random_full": {"rolling_shutter_artifact_temporal_frequency": 8.0, "rolling_shutter_artifact_random_disturbance_amplitude": 0.6},
        }
        row_readout = self.config.rolling_shutter_artifact_row_readout_time
        nyquist_hz = 1.0 / (2.0 * row_readout)
        self.assertGreater(
            configurations["freq_ultra_aliasing"]["rolling_shutter_artifact_temporal_frequency"], nyquist_hz,
        )
        outputs, dominant_bins = {}, {}
        clean = np.asarray(self.image, dtype=np.float32)
        for regime, values in configurations.items():
            attacked, pattern = self.attack(rolling_shutter_artifact_frequency_regime=regime, **values)
            delta = np.abs(np.asarray(attacked, dtype=np.float32) - clean).mean(axis=2)
            outputs[regime] = attacked
            self.assertTrue(np.isfinite(delta).all())
            self.assertGreater((delta > 2).mean(), 0.70)
            self.assertGreater(delta.mean(), 5.0)
            row_profile = pattern.irradiance.mean(axis=1)
            spectrum = np.abs(np.fft.rfft(row_profile - row_profile.mean()))
            dominant_bins[regime] = int(np.argmax(spectrum[1:]) + 1)
            if regime != "freq_random_full":
                frequency = values["rolling_shutter_artifact_aliasing_frequency"] if regime == "freq_ultra_aliasing" else values["rolling_shutter_artifact_temporal_frequency"]
                folded = abs((frequency * row_readout) % 1.0)
                folded = min(folded, 1.0 - folded)
                predicted_bin = round(folded * self.image.height)
                self.assertLessEqual(
                    abs(dominant_bins[regime] - predicted_bin), 8,
                    msg=f"{regime} dominant row bin {dominant_bins[regime]} must match the aliased fold of "
                        f"{frequency} Hz at {predicted_bin} (row bins of {self.image.height}).",
                )
        self.assertLess(dominant_bins["freq_low_wide"], dominant_bins["freq_mid_narrow"])
        self.assertLess(dominant_bins["freq_mid_narrow"], dominant_bins["freq_high_fine"])
        self.assertGreaterEqual(dominant_bins["freq_ultra_aliasing"], dominant_bins["freq_mid_narrow"])
        for left, right in (
            ("freq_low_wide", "freq_mid_narrow"), ("freq_low_wide", "freq_high_fine"),
            ("freq_low_wide", "freq_ultra_aliasing"), ("freq_mid_narrow", "freq_high_fine"),
            ("freq_mid_narrow", "freq_ultra_aliasing"), ("freq_high_fine", "freq_ultra_aliasing"),
        ):
            self.assertNotEqual(outputs[left].tobytes(), outputs[right].tobytes())
        first, _ = self.attack(rolling_shutter_artifact_frequency_regime="freq_random_full", rolling_shutter_artifact_random_disturbance_amplitude=0.6, random_seed=44)
        repeated, _ = self.attack(rolling_shutter_artifact_frequency_regime="freq_random_full", rolling_shutter_artifact_random_disturbance_amplitude=0.6, random_seed=44)
        changed_seed, _ = self.attack(rolling_shutter_artifact_frequency_regime="freq_random_full", rolling_shutter_artifact_random_disturbance_amplitude=0.6, random_seed=45)
        self.assertEqual(first.tobytes(), repeated.tobytes())
        self.assertNotEqual(first.tobytes(), changed_seed.tobytes())
