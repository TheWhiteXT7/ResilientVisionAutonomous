"""Comprehensive integration unit tests for DatasetGenerator."""

import tempfile
import unittest
from pathlib import Path
from PIL import Image

from attack_engine import AttackConfig, AttackPipeline
from dataset_generator.dataset_generator import DatasetGenerator
from dataset_generator.generator_config import GeneratorConfig
from dataset_loader import KittiLoader


class TestDatasetGenerator(unittest.TestCase):
    """Test suite for DatasetGenerator workflow and execution modes."""

    def setUp(self) -> None:
        """Set up mock KITTI dataset environment."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.kitti_dir = Path(self.temp_dir.name) / "KITTI"
        self.output_dir = Path(self.temp_dir.name) / "output"

        # Create KITTI folder structure
        self.img_dir = self.kitti_dir / "training" / "image_2"
        self.lbl_dir = self.kitti_dir / "training" / "label_2"
        self.cal_dir = self.kitti_dir / "training" / "calib"

        self.img_dir.mkdir(parents=True, exist_ok=True)
        self.lbl_dir.mkdir(parents=True, exist_ok=True)
        self.cal_dir.mkdir(parents=True, exist_ok=True)
        self.splits_dir = self.kitti_dir / "ImageSets"
        self.splits_dir.mkdir(parents=True, exist_ok=True)
        (self.splits_dir / "train.txt").write_text("000000\n000001\n000002\n", encoding="utf-8")
        (self.splits_dir / "val.txt").write_text("000002\n", encoding="utf-8")

        # Create 3 synthetic samples: 000000, 000001, 000002
        for sid in ("000000", "000001", "000002"):
            img_path = self.img_dir / f"{sid}.png"
            Image.new("RGB", (100, 100), (50, 50, 50)).save(img_path)

            lbl_path = self.lbl_dir / f"{sid}.txt"
            lbl_path.write_text("Car 0.0 0 0.0 50.0 50.0 100.0 100.0 1.5 1.5 4.0 0.0 1.0 10.0 0.0", encoding="utf-8")

            cal_path = self.cal_dir / f"{sid}.txt"
            cal_path.write_text("P2: 1 0 0 0 0 1 0 0 0 0 1 0", encoding="utf-8")

        self.loader = KittiLoader(kitti_dir=self.kitti_dir, split="train", validate=False)
        self.config = GeneratorConfig(
            output_directory=self.output_dir,
            overwrite_existing=False,
            save_metadata=True,
            copy_labels=True,
            copy_calibration=True,
        )
        self.attack_config = AttackConfig(
            laser_color=(255, 0, 0),
            spot_radius=10.0,
            intensity=1.0,
            max_spots=2,
            random_seed=42,
        )
        self.pipeline = AttackPipeline(config=self.attack_config)
        self.generator = DatasetGenerator(
            loader=self.loader,
            pipeline=self.pipeline,
            config=self.config,
        )

    def tearDown(self) -> None:
        """Clean up temporary directories."""
        self.temp_dir.cleanup()

    def test_generate_dataset_success(self) -> None:
        """Test generating complete dataset for all samples."""
        report = self.generator.generate_dataset(pattern_type="random")

        self.assertEqual(report["total_samples"], 3)
        self.assertEqual(report["processed_samples"], 3)
        self.assertEqual(report["successful_samples"], 3)
        self.assertEqual(report["failed_samples"], 0)
        self.assertEqual(report["status"], "completed")

        # Verify output files created
        out_img_dir = self.output_dir / "training" / "image_2"
        out_lbl_dir = self.output_dir / "training" / "label_2"
        out_cal_dir = self.output_dir / "training" / "calib"
        out_meta_dir = self.output_dir / "training" / "metadata"

        for sid in ("000000", "000001", "000002"):
            self.assertTrue((out_img_dir / f"{sid}.png").exists())
            self.assertTrue((out_lbl_dir / f"{sid}.txt").exists())
            self.assertTrue((out_cal_dir / f"{sid}.txt").exists())
            self.assertTrue((out_meta_dir / f"{sid}.json").exists())

        # Preserve source KITTI split metadata for downstream YOLO preparation
        self.assertEqual((self.output_dir / "ImageSets" / "train.txt").read_text(encoding="utf-8"), "000000\n000001\n000002\n")
        self.assertEqual((self.output_dir / "ImageSets" / "val.txt").read_text(encoding="utf-8"), "000002\n")

        # Verify generation summary file created
        self.assertTrue((self.output_dir / "generation_summary.json").exists())

    def test_generate_single(self) -> None:
        """Test generating single sample by sample ID."""
        report = self.generator.generate_single("000001", pattern_type="grid")

        self.assertEqual(report["total_samples"], 1)
        self.assertEqual(report["successful_samples"], 1)

        out_img = self.output_dir / "training" / "image_2" / "000001.png"
        self.assertTrue(out_img.exists())

    def test_generate_subset(self) -> None:
        """Test subset generation with count=2."""
        report = self.generator.generate_subset(count=2, pattern_type="horizontal_line")

        self.assertEqual(report["total_samples"], 2)
        self.assertEqual(report["successful_samples"], 2)

        out_img_dir = self.output_dir / "training" / "image_2"
        self.assertTrue((out_img_dir / "000000.png").exists())
        self.assertTrue((out_img_dir / "000001.png").exists())
        self.assertFalse((out_img_dir / "000002.png").exists())

    def test_resume_generation(self) -> None:
        """Test resuming dataset generation after partial run."""
        # Process sample 000000 first
        self.generator.generate_single("000000")

        # Resume generation should process remaining 2 samples
        report = self.generator.resume_generation(pattern_type="random")
        self.assertEqual(report["total_samples"], 2)
        self.assertEqual(report["successful_samples"], 2)

        # All 3 samples should now exist
        out_img_dir = self.output_dir / "training" / "image_2"
        for sid in ("000000", "000001", "000002"):
            self.assertTrue((out_img_dir / f"{sid}.png").exists())

    def test_failure_recovery_and_report(self) -> None:
        """Test graceful error handling when an image file is corrupted or missing."""
        # Delete image file for sample 000001 to simulate unreadable/corrupt image
        (self.img_dir / "000001.png").unlink()

        report = self.generator.generate_dataset(pattern_type="random")

        # Should continue processing remaining samples (000000 and 000002)
        self.assertEqual(report["total_samples"], 3)
        self.assertEqual(report["processed_samples"], 3)
        self.assertEqual(report["successful_samples"], 2)
        self.assertEqual(report["failed_samples"], 1)

        # Verify failure logged in report
        self.assertEqual(len(report["failures"]), 1)
        self.assertEqual(report["failures"][0]["sample_id"], "000001")


if __name__ == "__main__":
    unittest.main()


