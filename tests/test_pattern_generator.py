"""Unit tests for PatternGenerator and pattern generation algorithms."""

import sys
import unittest
from attack_engine.attack_config import AttackConfig
from attack_engine.laser_pattern import LaserPattern, LaserSpot
from attack_engine.pattern_generator import PatternGenerator


class TestPatternGenerator(unittest.TestCase):
    """Test suite for PatternGenerator algorithm strategies."""

    def setUp(self) -> None:
        """Set up standard canvas dimensions and AttackConfig."""
        self.width = 640
        self.height = 480
        self.config = AttackConfig(
            laser_color=(255, 0, 0),
            spot_radius=10.0,
            intensity=0.8,
            max_spots=5,
            random_seed=42,
        )
        self.generator = PatternGenerator(self.width, self.height, self.config)

    def test_initialization_validation(self) -> None:
        """Test invalid image dimensions and config types."""
        with self.assertRaises(ValueError):
            PatternGenerator(0, 480, self.config)
        with self.assertRaises(ValueError):
            PatternGenerator(640, -10, self.config)
        with self.assertRaises(TypeError):
            PatternGenerator(640, 480, "invalid_config")  # type: ignore[arg-type]

    def test_single_spot_default_and_custom(self) -> None:
        """Test single spot placement at canvas center and custom coordinates."""
        # Default center
        pattern = self.generator.single_spot()
        self.assertEqual(len(pattern), 1)
        self.assertEqual(pattern[0].x, 320.0)
        self.assertEqual(pattern[0].y, 240.0)

        # Custom location
        custom_pattern = self.generator.single_spot(x=100.0, y=150.0)
        self.assertEqual(len(custom_pattern), 1)
        self.assertEqual(custom_pattern[0].x, 100.0)
        self.assertEqual(custom_pattern[0].y, 150.0)

    def test_random_spots_reproducibility(self) -> None:
        """Test random spots generation and random seed reproducibility."""
        pattern1 = self.generator.random_spots()
        self.assertEqual(len(pattern1), self.config.max_spots)

        # Create identical generator with same seed
        gen2 = PatternGenerator(self.width, self.height, self.config)
        pattern2 = gen2.random_spots()

        self.assertEqual(len(pattern1), len(pattern2))
        for spot1, spot2 in zip(pattern1, pattern2):
            self.assertEqual(spot1.x, spot2.x)
            self.assertEqual(spot1.y, spot2.y)

    def test_horizontal_line(self) -> None:
        """Test horizontal line spot placement."""
        pattern = self.generator.horizontal_line(y=100.0, num_spots=3)
        self.assertEqual(len(pattern), 3)
        for spot in pattern:
            self.assertEqual(spot.y, 100.0)

        # Spacing option
        spaced_pattern = self.generator.horizontal_line(num_spots=3, spacing=50.0)
        self.assertEqual(len(spaced_pattern), 3)
        self.assertEqual(spaced_pattern[1].x - spaced_pattern[0].x, 50.0)

    def test_vertical_line(self) -> None:
        """Test vertical line spot placement."""
        pattern = self.generator.vertical_line(x=200.0, num_spots=4)
        self.assertEqual(len(pattern), 4)
        for spot in pattern:
            self.assertEqual(spot.x, 200.0)

    def test_grid_pattern(self) -> None:
        """Test grid spot placement."""
        pattern = self.generator.grid(rows=2, cols=3)
        self.assertEqual(len(pattern), 6)

    def test_custom_spots(self) -> None:
        """Test custom spots generation."""
        spots = [
            LaserSpot(x=10, y=10, radius=5, intensity=0.5, color=(0, 255, 0)),
            LaserSpot(x=20, y=20, radius=5, intensity=0.5, color=(0, 255, 0)),
        ]
        pattern = self.generator.custom(spots)
        self.assertEqual(len(pattern), 2)
        self.assertEqual(pattern[0], spots[0])

    def test_generate_dispatch(self) -> None:
        """Test general generate method dispatching and error handling."""
        self.assertEqual(len(self.generator.generate("single")), 1)
        self.assertEqual(len(self.generator.generate("grid")), 9)  # Default 3x3

        with self.assertRaises(ValueError):
            self.generator.generate("unknown_pattern_type")

    def test_no_pillow_dependency(self) -> None:
        """Verify pattern_generator module does not import Pillow."""
        import attack_engine.pattern_generator as pg_mod
        for attr in dir(pg_mod):
            self.assertNotIn("PIL", attr)
            self.assertNotIn("Image", attr)


if __name__ == "__main__":
    unittest.main()
