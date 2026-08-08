"""Unit tests for AttackConfig dataclass and validation."""

import unittest
from dataclasses import FrozenInstanceError
from attack_engine.attack_config import AttackConfig


class TestAttackConfig(unittest.TestCase):
    """Test suite for AttackConfig instantiation and parameter validation."""

    def test_default_initialization(self) -> None:
        """Test default values are properly set and valid."""
        config = AttackConfig()
        self.assertEqual(config.laser_color, (255, 0, 0))
        self.assertEqual(config.intensity, 1.0)
        self.assertEqual(config.alpha, 0.8)
        self.assertEqual(config.blur_radius, 5.0)
        self.assertEqual(config.spot_radius, 15.0)
        self.assertEqual(config.max_spots, 5)
        self.assertIsNone(config.random_seed)
        self.assertEqual(config.pattern_type, "random")
        self.assertEqual(config.output_dtype, "uint8")

    def test_immutability(self) -> None:
        """Test that AttackConfig instance attributes cannot be modified."""
        config = AttackConfig()
        with self.assertRaises((FrozenInstanceError, AttributeError)):
            config.intensity = 0.5  # type: ignore[misc]

    def test_list_color_converted_to_tuple(self) -> None:
        """Test that passing laser_color as a list converts it to a tuple."""
        config = AttackConfig(laser_color=[0, 255, 128])  # type: ignore[arg-type]
        self.assertEqual(config.laser_color, (0, 255, 128))
        self.assertIsInstance(config.laser_color, tuple)

    def test_laser_color_validation(self) -> None:
        """Test invalid laser_color formats and out-of-range RGB values."""
        invalid_colors = [
            (255, 0),             # Not 3 channels
            (255, 0, 0, 255),     # 4 channels
            "red",                # Not a tuple/list
            (255, 0, 256),        # Channel > 255
            (-1, 0, 0),           # Channel < 0
            (255, 0, 1.5),        # Float channel
            (True, 0, 0),         # Boolean channel
        ]
        for color in invalid_colors:
            with self.subTest(color=color):
                with self.assertRaises((ValueError, TypeError)):
                    AttackConfig(laser_color=color)  # type: ignore[arg-type]

    def test_intensity_validation(self) -> None:
        """Test intensity range and type validations."""
        with self.assertRaises(ValueError):
            AttackConfig(intensity=-0.1)
        with self.assertRaises(ValueError):
            AttackConfig(intensity=1.1)
        with self.assertRaises(TypeError):
            AttackConfig(intensity=True)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            AttackConfig(intensity="high")  # type: ignore[arg-type]

    def test_alpha_validation(self) -> None:
        """Test alpha range and type validations."""
        with self.assertRaises(ValueError):
            AttackConfig(alpha=-0.1)
        with self.assertRaises(ValueError):
            AttackConfig(alpha=1.05)
        with self.assertRaises(TypeError):
            AttackConfig(alpha=False)  # type: ignore[arg-type]

    def test_blur_radius_validation(self) -> None:
        """Test blur_radius non-negative and type validations."""
        with self.assertRaises(ValueError):
            AttackConfig(blur_radius=-1.0)
        with self.assertRaises(TypeError):
            AttackConfig(blur_radius=True)  # type: ignore[arg-type]

    def test_spot_radius_validation(self) -> None:
        """Test spot_radius positive and type validations."""
        with self.assertRaises(ValueError):
            AttackConfig(spot_radius=0.0)
        with self.assertRaises(ValueError):
            AttackConfig(spot_radius=-5.0)
        with self.assertRaises(TypeError):
            AttackConfig(spot_radius=False)  # type: ignore[arg-type]

    def test_max_spots_validation(self) -> None:
        """Test max_spots positive integer and type validations."""
        with self.assertRaises(ValueError):
            AttackConfig(max_spots=0)
        with self.assertRaises(ValueError):
            AttackConfig(max_spots=-3)
        with self.assertRaises(TypeError):
            AttackConfig(max_spots=2.5)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            AttackConfig(max_spots=True)  # type: ignore[arg-type]

    def test_random_seed_validation(self) -> None:
        """Test random_seed integer and type validations."""
        valid_seed_config = AttackConfig(random_seed=42)
        self.assertEqual(valid_seed_config.random_seed, 42)

        with self.assertRaises(TypeError):
            AttackConfig(random_seed=True)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            AttackConfig(random_seed="seed_123")  # type: ignore[arg-type]

    def test_pattern_type_validation(self) -> None:
        """Test pattern_type string validation."""
        with self.assertRaises(TypeError):
            AttackConfig(pattern_type=123)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            AttackConfig(pattern_type="   ")

    def test_output_dtype_validation(self) -> None:
        """Test output_dtype string validation."""
        with self.assertRaises(TypeError):
            AttackConfig(output_dtype=None)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            AttackConfig(output_dtype="")

    def test_target_class_default(self) -> None:
        """Test target_class defaults to 'Car'."""
        config = AttackConfig()
        self.assertEqual(config.target_class, "Car")

    def test_target_class_validation(self) -> None:
        """Test target_class type and non-empty validation."""
        self.assertEqual(AttackConfig(target_class="Pedestrian").target_class, "Pedestrian")
        with self.assertRaises(TypeError):
            AttackConfig(target_class=123)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            AttackConfig(target_class="   ")


if __name__ == "__main__":
    unittest.main()
