"""Focused tests for the continuous rolling-shutter laser attack."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from attack_engine import AttackConfig, AttackPipeline, RollingShutterPattern
from attack_engine.target_selection import TargetRegion
from dataset_generator.metadata_writer import MetadataWriter


class TestRollingShutterAttack(unittest.TestCase):
    def setUp(self) -> None:
        self.image = Image.new("RGB", (96, 72), (30, 30, 30))
        self.config = AttackConfig(
            pattern_type="rolling_shutter",
            random_seed=17,
            rolling_shutter_start=(12.0, 8.0),
            rolling_shutter_velocity=(0.9, 1.0),
            row_readout_time=0.6,
            row_exposure_time=1.2,
            beam_width=3.5,
            blur_radius=0.0,
        )

    def test_fixed_seed_is_deterministic_and_does_not_mutate_config(self) -> None:
        pipeline = AttackPipeline(self.config)
        first_image, first_pattern = pipeline.execute(self.image, pattern_type="rolling_shutter")
        second_image, second_pattern = pipeline.execute(self.image, pattern_type="rolling_shutter")

        self.assertEqual(first_image.tobytes(), second_image.tobytes())
        np.testing.assert_array_equal(first_pattern.exposure, second_pattern.exposure)
        self.assertEqual(pipeline.config, self.config)

    def test_output_has_valid_dimensions_and_continuous_exposure(self) -> None:
        attacked, pattern = AttackPipeline(self.config).execute(self.image, pattern_type="rolling_shutter")

        self.assertIsInstance(attacked, Image.Image)
        self.assertEqual(attacked.mode, "RGB")
        self.assertEqual(attacked.size, self.image.size)
        self.assertIsInstance(pattern, RollingShutterPattern)
        self.assertEqual(len(pattern), 0)  # The pattern is not represented as synthetic spots.
        self.assertEqual(pattern.exposure.shape, (72, 96))
        self.assertEqual(pattern.exposure.dtype, np.float32)
        active = pattern.exposure > 0.1
        self.assertGreater(active.sum(), 50)
        self.assertTrue(np.any(active[:, :-1] & active[:, 1:]))
        self.assertTrue(np.any(active[:-1, :] & active[1:, :]))

    def test_row_timing_changes_exposure_geometry(self) -> None:
        _, slow_readout = AttackPipeline(self.config).execute(self.image, pattern_type="rolling_shutter")
        _, fast_readout = AttackPipeline(self.config).execute(
            self.image, pattern_type="rolling_shutter", row_readout_time=0.2
        )
        self.assertFalse(np.array_equal(slow_readout.exposure, fast_readout.exposure))

    def test_trajectory_motion_changes_exposure_geometry(self) -> None:
        _, moving = AttackPipeline(self.config).execute(self.image, pattern_type="rolling_shutter")
        _, stationary = AttackPipeline(self.config).execute(
            self.image, pattern_type="rolling_shutter", rolling_shutter_velocity=(0.0, 0.0)
        )
        self.assertFalse(np.array_equal(moving.exposure, stationary.exposure))

    def test_configuration_validation(self) -> None:
        invalid = (
            {"rolling_shutter_start": (1.0,)},
            {"rolling_shutter_velocity": (1.0, True)},
            {"row_readout_time": 0.0},
            {"row_exposure_time": -1.0},
            {"beam_width": False},
        )
        for kwargs in invalid:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises((TypeError, ValueError)):
                    AttackConfig(**kwargs)

    def test_metadata_records_pattern_type_and_parameters(self) -> None:
        _, pattern = AttackPipeline(self.config).execute(self.image, pattern_type="rolling_shutter")
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "sample.json"
            MetadataWriter().write_sample_metadata(
                destination,
                "sample",
                pattern,
                self.config,
                {"pattern_type": "rolling_shutter", "spots_count": 0},
            )
            payload = json.loads(destination.read_text(encoding="utf-8"))

        self.assertEqual(payload["pattern_type"], "rolling_shutter")
        self.assertEqual(payload["attack_config"]["beam_width"], 3.5)
        self.assertEqual(payload["pattern_metadata"]["representation"], "rolling_shutter_exposure")

    def test_existing_random_and_targeted_spot_patterns_are_unchanged(self) -> None:
        pipeline = AttackPipeline(AttackConfig(max_spots=3, random_seed=9, spot_radius=4.0))
        _, random_pattern = pipeline.execute(self.image, pattern_type="random")
        self.assertEqual(len(random_pattern), 3)
        self.assertFalse(isinstance(random_pattern, RollingShutterPattern))
        _, targeted_pattern = pipeline.execute(
            self.image,
            pattern_type="targeted_spots",
            target_region=TargetRegion(class_name="Car", bbox=(20.0, 20.0, 70.0, 60.0)),
        )
        self.assertEqual(len(targeted_pattern), 3)
        self.assertFalse(isinstance(targeted_pattern, RollingShutterPattern))
