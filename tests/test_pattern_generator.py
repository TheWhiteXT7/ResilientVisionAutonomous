"""Unit tests for PatternGenerator and pattern generation algorithms."""

import sys
import unittest
from attack_engine.attack_config import AttackConfig
from attack_engine.laser_pattern import LaserPattern, LaserSpot
from attack_engine.pattern_generator import PatternGenerator
from attack_engine.target_selection import TargetRegion


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


class TestTargetedSpots(unittest.TestCase):
    """Test suite for object-aware targeted spot generation."""

    def setUp(self) -> None:
        self.width = 640
        self.height = 480
        self.target = TargetRegion(class_name="Car", bbox=(100.0, 100.0, 300.0, 300.0))

    def test_targeted_pattern_type_recognized(self) -> None:
        """Verify generate() dispatches 'targeted' and 'targeted_spots'."""
        config = AttackConfig(max_spots=4, random_seed=42)
        generator = PatternGenerator(self.width, self.height, config)

        for ptype in ("targeted", "targeted_spots"):
            with self.subTest(pattern_type=ptype):
                pattern = generator.generate(ptype, target=self.target)
                self.assertEqual(len(pattern), 4)

    def test_targeted_requires_target_argument(self) -> None:
        """Verify generating a targeted pattern without a target fails clearly."""
        generator = PatternGenerator(self.width, self.height, AttackConfig())
        with self.assertRaises(ValueError):
            generator.generate("targeted")

    def test_targeted_requires_target_region_type(self) -> None:
        """Verify passing a non-TargetRegion target raises TypeError."""
        generator = PatternGenerator(self.width, self.height, AttackConfig())
        with self.assertRaises(TypeError):
            generator.targeted_spots("not_a_region")  # type: ignore[arg-type]

    def test_all_centers_inside_target_bbox(self) -> None:
        """Verify every spot center lies inside the selected target bounding box."""
        config = AttackConfig(spot_radius=10.0, max_spots=20, random_seed=42)
        generator = PatternGenerator(self.width, self.height, config)
        pattern = generator.targeted_spots(self.target)

        self.assertEqual(len(pattern), 20)
        for spot in pattern:
            self.assertGreaterEqual(spot.x, 100.0)
            self.assertLessEqual(spot.x, 300.0)
            self.assertGreaterEqual(spot.y, 100.0)
            self.assertLessEqual(spot.y, 300.0)

    def test_centers_clamped_by_spot_radius(self) -> None:
        """Verify centers keep the full disc inside the bbox when possible."""
        config = AttackConfig(spot_radius=10.0, max_spots=20, random_seed=42)
        generator = PatternGenerator(self.width, self.height, config)
        pattern = generator.targeted_spots(self.target)

        for spot in pattern:
            self.assertGreaterEqual(spot.x, 110.0)
            self.assertLessEqual(spot.x, 290.0)
            self.assertGreaterEqual(spot.y, 110.0)
            self.assertLessEqual(spot.y, 290.0)

    def test_small_bbox_falls_back_to_full_bbox(self) -> None:
        """Verify a bbox smaller than the spot diameter still yields valid centers."""
        target = TargetRegion(class_name="Car", bbox=(100.0, 100.0, 110.0, 110.0))
        config = AttackConfig(spot_radius=15.0, max_spots=5, random_seed=3)
        generator = PatternGenerator(self.width, self.height, config)
        pattern = generator.targeted_spots(target)

        self.assertEqual(len(pattern), 5)
        for spot in pattern:
            self.assertGreaterEqual(spot.x, 100.0)
            self.assertLessEqual(spot.x, 110.0)
            self.assertGreaterEqual(spot.y, 100.0)
            self.assertLessEqual(spot.y, 110.0)

    def test_spots_respect_image_boundaries(self) -> None:
        """Verify targeted spots stay within the image even near the border."""
        target = TargetRegion(class_name="Car", bbox=(600.0, 100.0, 640.0, 300.0))
        config = AttackConfig(spot_radius=10.0, max_spots=10, random_seed=1)
        generator = PatternGenerator(self.width, self.height, config)
        pattern = generator.targeted_spots(target)

        for spot in pattern:
            self.assertGreaterEqual(spot.x, 0.0)
            self.assertLessEqual(spot.x, 640.0)
            self.assertGreaterEqual(spot.y, 0.0)
            self.assertLessEqual(spot.y, 480.0)
            self.assertGreaterEqual(spot.x, 600.0)
            self.assertLessEqual(spot.x, 640.0)
            self.assertGreaterEqual(spot.y, 100.0)
            self.assertLessEqual(spot.y, 300.0)

    def test_targeted_reproducible_with_seed(self) -> None:
        """Verify targeted spots are reproducible with the same seed."""
        config = AttackConfig(spot_radius=10.0, max_spots=4, random_seed=7)
        g1 = PatternGenerator(self.width, self.height, config)
        g2 = PatternGenerator(self.width, self.height, config)

        p1 = g1.targeted_spots(self.target)
        p2 = g2.targeted_spots(self.target)

        self.assertEqual(len(p1), len(p2))
        for s1, s2 in zip(p1, p2):
            self.assertEqual(s1.x, s2.x)
            self.assertEqual(s1.y, s2.y)

    def test_preserves_visual_settings(self) -> None:
        """Verify laser_color/intensity/radius are preserved on targeted spots."""
        config = AttackConfig(
            laser_color=(0, 0, 255),
            intensity=0.6,
            spot_radius=12.0,
            max_spots=2,
            random_seed=5,
        )
        generator = PatternGenerator(self.width, self.height, config)
        pattern = generator.targeted_spots(self.target)

        self.assertEqual(len(pattern), 2)
        for spot in pattern:
            self.assertEqual(spot.color, (0, 0, 255))
            self.assertEqual(spot.intensity, 0.6)
            self.assertEqual(spot.radius, 12.0)

    def test_bbox_outside_image_rejected(self) -> None:
        """Verify a target bbox fully outside the canvas raises ValueError."""
        target = TargetRegion(class_name="Car", bbox=(700.0, 500.0, 800.0, 600.0))
        generator = PatternGenerator(self.width, self.height, AttackConfig())
        with self.assertRaises(ValueError):
            generator.targeted_spots(target)

    def test_num_spots_validation(self) -> None:
        """Verify non-positive num_spots override raises ValueError."""
        generator = PatternGenerator(self.width, self.height, AttackConfig())
        with self.assertRaises(ValueError):
            generator.targeted_spots(self.target, num_spots=0)


if __name__ == "__main__":
    unittest.main()
