"""Unit tests for ProjectionEngine Pillow rendering and blending."""

import unittest
from PIL import Image
from attack_engine.attack_config import AttackConfig
from attack_engine.laser_pattern import LaserPattern, LaserSpot
from attack_engine.projection_engine import ProjectionEngine


class TestProjectionEngine(unittest.TestCase):
    """Test suite for ProjectionEngine image rendering operations."""

    def setUp(self) -> None:
        """Set up test PIL images and default components."""
        self.engine = ProjectionEngine()
        self.image_rgb = Image.new("RGB", (100, 100), (50, 50, 50))
        self.config = AttackConfig(
            laser_color=(255, 0, 0),
            intensity=1.0,
            alpha=1.0,
            blur_radius=0.0,  # Sharp for exact pixel assertions
            spot_radius=10.0,
        )

    def test_original_image_not_modified(self) -> None:
        """Verify rendering returns a new image and leaves original untouched."""
        spot = LaserSpot(x=50, y=50, radius=10, intensity=1.0, color=(255, 0, 0))
        pattern = LaserPattern([spot])

        original_pixel_before = self.image_rgb.getpixel((50, 50))
        attacked = self.engine.render(self.image_rgb, pattern, self.config)
        original_pixel_after = self.image_rgb.getpixel((50, 50))

        self.assertEqual(original_pixel_before, original_pixel_after)
        self.assertIsNot(self.image_rgb, attacked)

    def test_alpha_blending_pixel_alteration(self) -> None:
        """Test pixel alteration at laser spot position."""
        spot = LaserSpot(x=50, y=50, radius=10, intensity=1.0, color=(255, 0, 0))
        pattern = LaserPattern([spot])

        attacked = self.engine.render(self.image_rgb, pattern, self.config)
        center_pixel = attacked.getpixel((50, 50))

        # Red channel should be higher than background 50
        self.assertGreater(center_pixel[0], 100)

    def test_blur_radius_effect(self) -> None:
        """Test Gaussian blur smooths spot edges."""
        spot = LaserSpot(x=50, y=50, radius=5, intensity=1.0, color=(0, 255, 0))
        pattern = LaserPattern([spot])

        config_no_blur = AttackConfig(blur_radius=0.0, spot_radius=5.0)
        config_with_blur = AttackConfig(blur_radius=5.0, spot_radius=5.0)

        img_sharp = self.engine.render(self.image_rgb, pattern, config_no_blur)
        img_blurred = self.engine.render(self.image_rgb, pattern, config_with_blur)

        # Pixel just outside radius should be background in sharp image, affected in blurred image
        sharp_outer = img_sharp.getpixel((50, 60))
        blurred_outer = img_blurred.getpixel((50, 60))

        self.assertEqual(sharp_outer, (50, 50, 50))
        self.assertNotEqual(blurred_outer, (50, 50, 50))

    def test_image_mode_preservation(self) -> None:
        """Test that original image mode is preserved."""
        image_l = Image.new("L", (100, 100), 128)
        spot = LaserSpot(x=50, y=50, radius=10, intensity=1.0, color=(255, 255, 255))
        pattern = LaserPattern([spot])

        attacked_l = self.engine.render(image_l, pattern, self.config)
        self.assertEqual(attacked_l.mode, "L")

        attacked_rgb = self.engine.render(self.image_rgb, pattern, self.config)
        self.assertEqual(attacked_rgb.mode, "RGB")

    def test_invalid_input_types(self) -> None:
        """Test exception raising for invalid input types."""
        pattern = LaserPattern()
        with self.assertRaises(TypeError):
            self.engine.render("not_an_image", pattern, self.config)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            self.engine.render(self.image_rgb, "not_a_pattern", self.config)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            self.engine.render(self.image_rgb, pattern, "not_a_config")  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
