"""Unit tests for GeneratorConfig dataclass and validation."""

import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path

from dataset_generator.generator_config import GeneratorConfig


class TestGeneratorConfig(unittest.TestCase):
    """Test suite for GeneratorConfig instantiation and parameter validation."""

    def test_default_initialization(self) -> None:
        """Test default values are properly initialized."""
        config = GeneratorConfig()
        self.assertEqual(config.output_directory, Path("outputs/attacked_dataset"))
        self.assertFalse(config.overwrite_existing)
        self.assertTrue(config.save_metadata)
        self.assertFalse(config.save_original_copy)
        self.assertTrue(config.copy_labels)
        self.assertTrue(config.copy_calibration)
        self.assertEqual(config.workers, 1)
        self.assertEqual(config.batch_size, 32)
        self.assertEqual(config.image_format, "png")
        self.assertEqual(config.metadata_format, "json")
        self.assertEqual(config.logging_level, "INFO")

    def test_string_path_conversion(self) -> None:
        """Test that string output_directory is converted to Path."""
        config = GeneratorConfig(output_directory="custom/output_path")
        self.assertIsInstance(config.output_directory, Path)
        self.assertEqual(config.output_directory, Path("custom/output_path"))

    def test_immutability(self) -> None:
        """Test that GeneratorConfig attributes cannot be modified."""
        config = GeneratorConfig()
        with self.assertRaises((FrozenInstanceError, AttributeError)):
            config.workers = 4  # type: ignore[misc]

    def test_invalid_output_directory(self) -> None:
        """Test invalid output_directory values."""
        with self.assertRaises(ValueError):
            GeneratorConfig(output_directory="   ")
        with self.assertRaises(TypeError):
            GeneratorConfig(output_directory=123)  # type: ignore[arg-type]

    def test_boolean_fields_validation(self) -> None:
        """Test type validation for boolean flags."""
        with self.assertRaises(TypeError):
            GeneratorConfig(overwrite_existing="true")  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            GeneratorConfig(save_metadata=1)  # type: ignore[arg-type]

    def test_workers_and_batch_size_validation(self) -> None:
        """Test workers and batch_size positive integer validations."""
        with self.assertRaises(ValueError):
            GeneratorConfig(workers=0)
        with self.assertRaises(ValueError):
            GeneratorConfig(batch_size=-5)
        with self.assertRaises(TypeError):
            GeneratorConfig(workers=2.5)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            GeneratorConfig(batch_size=True)  # type: ignore[arg-type]

    def test_image_format_validation(self) -> None:
        """Test image_format string validations."""
        valid_config = GeneratorConfig(image_format="JPG")
        self.assertEqual(valid_config.image_format, "jpg")

        with self.assertRaises(ValueError):
            GeneratorConfig(image_format="gif")
        with self.assertRaises(TypeError):
            GeneratorConfig(image_format=123)  # type: ignore[arg-type]

    def test_logging_level_validation(self) -> None:
        """Test logging_level uppercase conversion and validation."""
        valid_config = GeneratorConfig(logging_level="debug")
        self.assertEqual(valid_config.logging_level, "DEBUG")

        with self.assertRaises(ValueError):
            GeneratorConfig(logging_level="UNKNOWN")


if __name__ == "__main__":
    unittest.main()
