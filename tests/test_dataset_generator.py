"""Unit tests for the dataset-generator modules."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from dataset_generator.attack_pipeline import attack_image
from dataset_generator.image_loader import discover_images, load_image
from dataset_generator.metadata import (
    create_metadata_record,
    write_metadata_record,
)
from dataset_generator.output_writer import save_attacked_image
from dataset_generator.pattern_manager import get_pattern, load_pattern


class DatasetGeneratorTests(unittest.TestCase):
    """Exercise discovery, patterns, attack, output, and metadata."""

    def test_image_discovery_returns_relative_readable_paths(self) -> None:
        """Discovery preserves nested paths and excludes a corrupt image."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            _write_rgb_image(root / "nested" / "sample.png", 10)
            (root / "broken.jpg").write_text("not an image", encoding="utf-8")

            discovered = discover_images(root)

            self.assertEqual(discovered, [Path("nested/sample.png")])
            loaded = load_image(root / discovered[0])
            self.assertIsNotNone(loaded)
            assert loaded is not None
            self.assertEqual(loaded.shape, (2, 3, 3))

    def test_pattern_loading_and_resizing_preserves_rgb_uint8(self) -> None:
        """A loaded pattern can be resized to an image shape safely."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            pattern_path = Path(temporary_directory) / "laser.png"
            _write_rgb_image(pattern_path, 255)

            loaded = load_pattern(pattern_path)
            prepared = get_pattern(pattern_path, (4, 5, 3))

            self.assertEqual(loaded.dtype, np.uint8)
            self.assertEqual(loaded.shape, (2, 3, 3))
            self.assertEqual(prepared.shape, (4, 5, 3))
            self.assertTrue(np.all(prepared == 255))

    def test_attack_pipeline_applies_attack_engine_operation(self) -> None:
        """The pipeline delegates the expected saturating RGB addition."""
        image = np.full((2, 2, 3), 240, dtype=np.uint8)
        pattern = np.full((2, 2, 3), 30, dtype=np.uint8)

        attacked = attack_image(image, pattern)

        self.assertEqual(attacked.dtype, np.uint8)
        self.assertTrue(np.all(attacked == 255))
        self.assertTrue(np.all(image == 240))

    def test_output_writer_preserves_relative_parent_path(self) -> None:
        """Generated images are written beneath the output root only."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_root = Path(temporary_directory) / "generated"
            image = np.full((2, 3, 3), 42, dtype=np.uint8)

            saved_path = save_attacked_image(
                image,
                Path("training/image_2/000001.png"),
                output_root,
                "png",
            )

            self.assertEqual(
                saved_path,
                output_root / "training" / "image_2" / "000001.png",
            )
            self.assertTrue(saved_path.is_file())
            self.assertEqual(load_image(saved_path).shape, image.shape)

    def test_metadata_writer_creates_required_csv_columns(self) -> None:
        """CSV output contains a complete row with required metadata fields."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            metadata_path = Path(temporary_directory) / "metadata.csv"
            record = create_metadata_record(
                image_id="training/image_2/000001",
                original_relative_path="training/image_2/000001.png",
                attacked_relative_path="training/image_2/000001.png",
                pattern_used="laser.png",
                image_width=1242,
                image_height=375,
            )

            write_metadata_record(metadata_path, record)

            with metadata_path.open("r", newline="", encoding="utf-8") as file:
                rows = list(csv.DictReader(file))

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["image_id"], record.image_id)
            self.assertEqual(rows[0]["processing_status"], "success")
            self.assertTrue(rows[0]["timestamp"])



def _write_rgb_image(path: Path, value: int) -> None:
    """Create a small RGB fixture image at ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full((2, 3, 3), value, dtype=np.uint8)
    Image.fromarray(image, mode="RGB").save(path)


if __name__ == "__main__":
    unittest.main()
