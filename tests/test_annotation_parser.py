"""Unit tests for KittiAnnotationParser."""

import tempfile
import unittest
from pathlib import Path

from dataset_loader.annotation_parser import (
    Annotation,
    AnnotationParseError,
    KittiAnnotationParser,
)


class TestKittiAnnotationParser(unittest.TestCase):
    """Test suite for KittiAnnotationParser."""

    def setUp(self) -> None:
        """Set up temporary directory for test files."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.parser = KittiAnnotationParser()

    def tearDown(self) -> None:
        """Clean up temporary directory."""
        self.temp_dir.cleanup()

    def _create_temp_file(self, content: str) -> Path:
        """Helper to create a temporary text file with given content."""
        file_path = Path(self.temp_dir.name) / "test_label.txt"
        file_path.write_text(content, encoding="utf-8")
        return file_path

    def test_valid_annotation_file(self) -> None:
        """Test parsing a valid annotation file with a single object."""
        content = (
            "Pedestrian 0.00 0 -0.20 712.40 143.00 810.73 307.92 "
            "1.89 0.48 1.20 1.84 1.47 8.41 0.01\n"
        )
        file_path = self._create_temp_file(content)
        annotations = self.parser.parse(file_path)

        self.assertEqual(len(annotations), 1)
        ann = annotations[0]
        self.assertIsInstance(ann, Annotation)
        self.assertEqual(ann.class_name, "Pedestrian")
        self.assertAlmostEqual(ann.truncated, 0.0)
        self.assertEqual(ann.occluded, 0)
        self.assertAlmostEqual(ann.alpha, -0.20)
        self.assertEqual(ann.bbox, (712.40, 143.00, 810.73, 307.92))
        self.assertEqual(ann.dimensions, (1.89, 0.48, 1.20))
        self.assertEqual(ann.location, (1.84, 1.47, 8.41))
        self.assertAlmostEqual(ann.rotation_y, 0.01)
        self.assertIsNone(ann.score)

    def test_empty_annotation_file(self) -> None:
        """Test parsing an empty annotation file."""
        file_path = self._create_temp_file("\n\n   \n")
        annotations = self.parser.parse(file_path)
        self.assertEqual(len(annotations), 0)

    def test_malformed_annotation_line_token_count(self) -> None:
        """Test parsing a line with invalid token count raises exception."""
        content = "Car 0.00 0 -1.57 599.41 156.40\n"
        file_path = self._create_temp_file(content)

        with self.assertRaises(AnnotationParseError) as ctx:
            self.parser.parse(file_path)

        self.assertIn("Expected 15 or 16 tokens", str(ctx.exception))

    def test_malformed_annotation_line_numerical(self) -> None:
        """Test parsing a line with non-numeric fields raises exception."""
        content = (
            "Car NOT_A_FLOAT 0 -1.57 599.41 156.40 629.75 189.25 "
            "2.85 2.63 12.34 0.47 1.49 69.44 -1.56\n"
        )
        file_path = self._create_temp_file(content)

        with self.assertRaises(AnnotationParseError) as ctx:
            self.parser.parse(file_path)

        self.assertIn("Failed to parse numerical values", str(ctx.exception))

    def test_multiple_objects(self) -> None:
        """Test parsing a file containing multiple objects."""
        content = (
            "Truck 0.00 0 -1.57 599.41 156.40 629.75 189.25 "
            "2.85 2.63 12.34 0.47 1.49 69.44 -1.56\n"
            "Car 0.00 0 1.85 387.63 181.54 423.81 203.12 "
            "1.67 1.87 3.69 -16.53 2.39 58.49 1.57\n"
            "DontCare -1 -1 -10 503.89 169.71 590.61 190.13 "
            "-1 -1 -1 -1000 -1000 -1000 -10\n"
        )
        file_path = self._create_temp_file(content)
        annotations = self.parser.parse(file_path)

        self.assertEqual(len(annotations), 3)
        self.assertEqual(annotations[0].class_name, "Truck")
        self.assertEqual(annotations[1].class_name, "Car")
        self.assertEqual(annotations[2].class_name, "DontCare")

    def test_annotation_with_score(self) -> None:
        """Test parsing 16-token annotation line containing prediction score."""
        content = (
            "Car 0.00 0 -1.57 599.41 156.40 629.75 189.25 "
            "2.85 2.63 12.34 0.47 1.49 69.44 -1.56 0.98\n"
        )
        file_path = self._create_temp_file(content)
        annotations = self.parser.parse(file_path)

        self.assertEqual(len(annotations), 1)
        self.assertAlmostEqual(annotations[0].score, 0.98)

    def test_file_not_found(self) -> None:
        """Test FileNotFoundError is raised for non-existent path."""
        non_existent = Path(self.temp_dir.name) / "does_not_exist.txt"
        with self.assertRaises(FileNotFoundError):
            self.parser.parse(non_existent)


if __name__ == "__main__":
    unittest.main()
