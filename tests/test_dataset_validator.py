"""Unit tests for DatasetValidator."""

import tempfile
import unittest
from pathlib import Path

from dataset_loader.dataset_validator import DatasetValidator


class TestDatasetValidator(unittest.TestCase):
    """Test suite for DatasetValidator functionality."""

    def setUp(self) -> None:
        """Set up temporary directory structure for testing."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.kitti_dir = Path(self.temp_dir.name) / "KITTI"
        self.kitti_dir.mkdir(parents=True, exist_ok=True)

        self.train_img_dir = self.kitti_dir / "training" / "image_2"
        self.train_lbl_dir = self.kitti_dir / "training" / "label_2"
        self.test_img_dir = self.kitti_dir / "testing" / "image_2"

        self.train_img_dir.mkdir(parents=True, exist_ok=True)
        self.train_lbl_dir.mkdir(parents=True, exist_ok=True)
        self.test_img_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        """Clean up temporary directory."""
        self.temp_dir.cleanup()

    def test_init_default_path(self) -> None:
        """Test default initialization imports from config.paths."""
        validator = DatasetValidator()
        self.assertIsNotNone(validator.kitti_dir)

    def test_validate_structure_success(self) -> None:
        """Test structure validation when all directories exist."""
        validator = DatasetValidator(kitti_dir=self.kitti_dir)
        is_valid, errors, warnings = validator.validate_structure()
        self.assertTrue(is_valid)
        self.assertEqual(len(errors), 0)
        self.assertEqual(len(warnings), 0)

    def test_validate_structure_missing_root(self) -> None:
        """Test structure validation with non-existent root."""
        non_existent = self.kitti_dir / "non_existent"
        validator = DatasetValidator(kitti_dir=non_existent)
        is_valid, errors, _ = validator.validate_structure()
        self.assertFalse(is_valid)
        self.assertTrue(len(errors) > 0)

    def test_count_files(self) -> None:
        """Test counting image and label files."""
        # Create dummy training images and labels
        (self.train_img_dir / "000000.png").touch()
        (self.train_img_dir / "000001.png").touch()
        (self.train_lbl_dir / "000000.txt").touch()

        # Create dummy testing images
        (self.test_img_dir / "000002.png").touch()
        (self.test_img_dir / "000003.png").touch()
        (self.test_img_dir / "000004.png").touch()

        validator = DatasetValidator(kitti_dir=self.kitti_dir)
        counts = validator.count_files()

        self.assertEqual(counts["num_train_images"], 2)
        self.assertEqual(counts["num_train_labels"], 1)
        self.assertEqual(counts["num_test_images"], 3)

    def test_detect_missing_pairs(self) -> None:
        """Test detecting missing image-label pairs."""
        (self.train_img_dir / "000000.png").touch()
        (self.train_img_dir / "000001.png").touch()
        (self.train_lbl_dir / "000000.txt").touch()
        (self.train_lbl_dir / "000002.txt").touch()

        validator = DatasetValidator(kitti_dir=self.kitti_dir)
        missing_labels, missing_images = validator.detect_missing_pairs()

        self.assertEqual(missing_labels, ["000001"])
        self.assertEqual(missing_images, ["000002"])

    def test_validate_report(self) -> None:
        """Test full validation report dictionary structure."""
        (self.train_img_dir / "000000.png").touch()
        (self.train_lbl_dir / "000000.txt").touch()

        validator = DatasetValidator(kitti_dir=self.kitti_dir)
        report = validator.validate()

        self.assertIn("is_valid", report)
        self.assertIn("kitti_dir", report)
        self.assertIn("num_train_images", report)
        self.assertIn("num_test_images", report)
        self.assertIn("num_train_labels", report)
        self.assertIn("missing_labels", report)
        self.assertIn("missing_images", report)
        self.assertIn("errors", report)
        self.assertIn("warnings", report)
        self.assertTrue(report["is_valid"])
        self.assertEqual(report["missing_labels_count"], 0)


if __name__ == "__main__":
    unittest.main()
