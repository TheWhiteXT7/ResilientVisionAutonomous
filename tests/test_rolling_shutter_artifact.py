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
