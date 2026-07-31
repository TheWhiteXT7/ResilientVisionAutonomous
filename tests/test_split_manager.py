"""Unit tests for SplitManager."""

import tempfile
import unittest
from pathlib import Path

from dataset_loader.split_manager import SplitManager


class TestSplitManager(unittest.TestCase):
    """Test suite for SplitManager."""

    def setUp(self) -> None:
        """Set up temporary directory structure."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.kitti_dir = Path(self.temp_dir.name) / "KITTI"
        self.splits_dir = self.kitti_dir / "ImageSets"
        self.splits_dir.mkdir(parents=True, exist_ok=True)
        self.manager = SplitManager(
            kitti_dir=self.kitti_dir, splits_dir=self.splits_dir
        )

    def tearDown(self) -> None:
        """Clean up temporary directory."""
        self.temp_dir.cleanup()

    def test_save_and_load_split(self) -> None:
        """Test saving sample IDs to file and reading them back."""
        sample_ids = ["000000", "000001", "000002"]
        saved_path = self.manager.save_split("train", sample_ids)
        self.assertTrue(saved_path.exists())

        loaded_ids = self.manager.load_split("train")
        self.assertEqual(loaded_ids, sample_ids)

    def test_load_split_not_found(self) -> None:
        """Test loading non-existent split raises FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            self.manager.load_split("non_existent_split")

    def test_create_random_split(self) -> None:
        """Test splitting sample IDs into train and validation sets."""
        sample_ids = [f"{i:06d}" for i in range(10)]
        train_ids, val_ids = self.manager.create_random_split(
            sample_ids, train_ratio=0.8, seed=42
        )

        self.assertEqual(len(train_ids), 8)
        self.assertEqual(len(val_ids), 2)
        self.assertEqual(len(set(train_ids).intersection(set(val_ids))), 0)

    def test_create_random_split_reproducibility(self) -> None:
        """Test identical random seed produces identical split results."""
        sample_ids = [f"{i:06d}" for i in range(10)]
        train_ids_1, val_ids_1 = self.manager.create_random_split(
            sample_ids, train_ratio=0.7, seed=123
        )
        train_ids_2, val_ids_2 = self.manager.create_random_split(
            sample_ids, train_ratio=0.7, seed=123
        )

        self.assertEqual(train_ids_1, train_ids_2)
        self.assertEqual(val_ids_1, val_ids_2)

    def test_invalid_train_ratio(self) -> None:
        """Test invalid train_ratio raises ValueError."""
        with self.assertRaises(ValueError):
            self.manager.create_random_split(["000000"], train_ratio=1.5)


if __name__ == "__main__":
    unittest.main()
