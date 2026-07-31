"""Unit tests for AttackPipeline orchestration and public API."""

import unittest
from PIL import Image
from attack_engine.attack_config import AttackConfig
from attack_engine.attack_pipeline import AttackPipeline, apply_attack
from attack_engine.laser_pattern import LaserPattern


class TestAttackPipeline(unittest.TestCase):
    """Test suite for AttackPipeline and apply_attack function."""

    def setUp(self) -> None:
        """Set up test image."""
        self.image = Image.new("RGB", (200, 200), (100, 100, 100))

    def test_pipeline_execute_default(self) -> None:
        """Test pipeline execution returns attacked image and pattern."""
        pipeline = AttackPipeline()
        attacked, pattern = pipeline.execute(self.image, pattern_type="random")

        self.assertIsInstance(attacked, Image.Image)
        self.assertIsInstance(pattern, LaserPattern)
        self.assertEqual(attacked.size, self.image.size)
        self.assertEqual(len(pattern), pipeline.config.max_spots)

    def test_apply_attack_functional_api(self) -> None:
        """Test top-level apply_attack function with kwargs overrides."""
        attacked, pattern = apply_attack(
            self.image,
            pattern_type="horizontal_line",
            laser_color=(0, 255, 0),
            max_spots=4,
            random_seed=123,
        )

        self.assertIsInstance(attacked, Image.Image)
        self.assertIsInstance(pattern, LaserPattern)
        self.assertEqual(len(pattern), 4)
        self.assertEqual(pattern[0].color, (0, 255, 0))

    def test_pipeline_custom_config(self) -> None:
        """Test AttackPipeline initialized with custom AttackConfig."""
        custom_config = AttackConfig(
            laser_color=(0, 0, 255),
            spot_radius=25.0,
            max_spots=2,
        )
        pipeline = AttackPipeline(config=custom_config)
        attacked, pattern = pipeline.execute(self.image, pattern_type="single")

        self.assertEqual(len(pattern), 1)
        self.assertEqual(pattern[0].color, (0, 0, 255))
        self.assertEqual(pattern[0].radius, 25.0)

    def test_invalid_image_type(self) -> None:
        """Test passing non-PIL image raises TypeError."""
        pipeline = AttackPipeline()
        with self.assertRaises(TypeError):
            pipeline.execute("invalid_image")  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            apply_attack("invalid_image")  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
