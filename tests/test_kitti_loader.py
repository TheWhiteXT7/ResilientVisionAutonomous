"""Unit tests for KittiLoader and KittiSample."""

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from dataset_loader.kitti_loader import KittiLoader, KittiSample


class TestKittiLoader(unittest.TestCase):
    """Test suite for KittiLoader and KittiSample."""

    def setUp(self) -> None:
        """Set up mock KITTI dataset structure."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.kitti_dir = Path(self.temp_dir.name) / "KITTI"
        self.train_img_dir = self.kitti_dir / "training" / "image_2"
        self.train_lbl_dir = self.kitti_dir / "training" / "label_2"
        self.test_img_dir = self.kitti_dir / "testing" / "image_2"

        self.train_img_dir.mkdir(parents=True, exist_ok=True)
        self.train_lbl_dir.mkdir(parents=True, exist_ok=True)
        self.test_img_dir.mkdir(parents=True, exist_ok=True)

        # Create dummy training images and labels
        img_0 = Image.new("RGB", (100, 100), color="red")
        img_0.save(self.train_img_dir / "000000.png")
        lbl_0_content = (
            "Car 0.00 0 -1.57 599.41 156.40 629.75 189.25 "
            "2.85 2.63 12.34 0.47 1.49 69.44 -1.56\n"
        )
        (self.train_lbl_dir / "000000.txt").write_text(
            lbl_0_content, encoding="utf-8"
        )

        img_1 = Image.new("RGB", (100, 100), color="blue")
        img_1.save(self.train_img_dir / "000001.png")
        lbl_1_content = (
            "Pedestrian 0.00 0 -0.20 712.40 143.00 810.73 307.92 "
            "1.89 0.48 1.20 1.84 1.47 8.41 0.01\n"
        )
        (self.train_lbl_dir / "000001.txt").write_text(
            lbl_1_content, encoding="utf-8"
        )

        # Create dummy testing image
        img_test = Image.new("RGB", (100, 100), color="green")
        img_test.save(self.test_img_dir / "000002.png")

    def tearDown(self) -> None:
        """Clean up temporary directory."""
        self.temp_dir.cleanup()

    def test_kitti_sample_load_image(self) -> None:
        """Test KittiSample lazy image loading."""
        img_path = self.train_img_dir / "000000.png"
        sample = KittiSample(sample_id="000000", image_path=img_path)
        img = sample.load_image()

        self.assertIsInstance(img, Image.Image)
        self.assertEqual(img.size, (100, 100))

    def test_kitti_loader_training_split(self) -> None:
        """Test KittiLoader with training split."""
        loader = KittiLoader(kitti_dir=self.kitti_dir, split="train")

        self.assertEqual(len(loader), 2)
        self.assertEqual(loader.sample_ids, ["000000", "000001"])

        # Test index access
        sample_0 = loader[0]
        self.assertIsInstance(sample_0, KittiSample)
        self.assertEqual(sample_0.sample_id, "000000")
        self.assertEqual(len(sample_0.annotations), 1)
        self.assertEqual(sample_0.annotations[0].class_name, "Car")

    def test_kitti_loader_trainval_split(self) -> None:
        """trainval uses the union of KITTI train and validation split files."""
        split_dir = self.kitti_dir / "ImageSets"
        split_dir.mkdir()
        (split_dir / "train.txt").write_text("000000\n", encoding="utf-8")
        (split_dir / "val.txt").write_text("000001\n", encoding="utf-8")

        loader = KittiLoader(kitti_dir=self.kitti_dir, split="trainval")

        self.assertEqual(len(loader), 2)
        self.assertEqual(loader.sample_ids, ["000000", "000001"])
    def test_kitti_loader_string_id_access(self) -> None:
        """Test retrieving sample by string ID."""
        loader = KittiLoader(kitti_dir=self.kitti_dir, split="train")
        sample = loader["000001"]

        self.assertEqual(sample.sample_id, "000001")
        self.assertEqual(sample.annotations[0].class_name, "Pedestrian")

    def test_kitti_loader_slice_access(self) -> None:
        """Test retrieving slice of samples."""
        loader = KittiLoader(kitti_dir=self.kitti_dir, split="train")
        samples = loader[0:2]

        self.assertEqual(len(samples), 2)
        self.assertEqual(samples[0].sample_id, "000000")
        self.assertEqual(samples[1].sample_id, "000001")

    def test_kitti_loader_iteration(self) -> None:
        """Test iterating over KittiLoader."""
        loader = KittiLoader(kitti_dir=self.kitti_dir, split="train")
        samples = list(loader)

        self.assertEqual(len(samples), 2)
        self.assertEqual(samples[0].sample_id, "000000")
        self.assertEqual(samples[1].sample_id, "000001")

    def test_kitti_loader_testing_split(self) -> None:
        """Test KittiLoader with testing split."""
        loader = KittiLoader(kitti_dir=self.kitti_dir, split="test")

        self.assertEqual(len(loader), 1)
        self.assertEqual(loader.sample_ids, ["000002"])
        sample = loader[0]
        self.assertEqual(sample.sample_id, "000002")

    def test_kitti_loader_preloading_images(self) -> None:
        """Test preloading images with load_images=True."""
        loader = KittiLoader(
            kitti_dir=self.kitti_dir, split="train", load_images=True
        )
        sample = loader[0]

        self.assertIsNotNone(sample.image)
        self.assertIsInstance(sample.image, Image.Image)

    def test_kitti_loader_invalid_index(self) -> None:
        """Test IndexError raised for out-of-range index."""
        loader = KittiLoader(kitti_dir=self.kitti_dir, split="train")
        with self.assertRaises(IndexError):
            _ = loader[99]

    def test_kitti_loader_key_error(self) -> None:
        """Test KeyError raised for non-existent sample ID."""
        loader = KittiLoader(kitti_dir=self.kitti_dir, split="train")
        with self.assertRaises(KeyError):
            _ = loader["non_existent_id"]


if __name__ == "__main__":
    unittest.main()

