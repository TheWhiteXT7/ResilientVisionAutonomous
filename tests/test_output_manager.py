"""Unit tests for OutputManager asset persistence and directory structure."""

import tempfile
import unittest
from pathlib import Path
from PIL import Image

from attack_engine import AttackConfig, LaserPattern
from dataset_generator.generator_config import GeneratorConfig
from dataset_generator.output_manager import OutputManager


class TestOutputManager(unittest.TestCase):
    """Test suite for OutputManager directory creation and persistence."""

    def setUp(self) -> None:
        """Set up temporary output directory and components."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.config = GeneratorConfig(
            output_directory=Path(self.temp_dir.name) / "output",
            overwrite_existing=False,
            save_metadata=True,
            copy_labels=True,
            copy_calibration=True,
        )
        self.manager = OutputManager(config=self.config)
        self.image = Image.new("RGB", (64, 64), (100, 100, 100))

    def tearDown(self) -> None:
        """Clean up temporary directory."""
        self.temp_dir.cleanup()

    def test_setup_structure(self) -> None:
        """Test setup_structure creates expected folders."""
        paths = self.manager.setup_structure(split="training")
        self.assertTrue(paths["image_2"].exists())
        self.assertTrue(paths["label_2"].exists())
        self.assertTrue(paths["calib"].exists())
        self.assertTrue(paths["metadata"].exists())

    def test_save_attacked_image(self) -> None:
        """Test saving attacked image."""
        img_path = self.manager.save_attacked_image(self.image, "000000", split="training")
        self.assertTrue(img_path.exists())
        self.assertEqual(img_path.suffix, ".png")

        # Verify saved content readable
        loaded = Image.open(img_path)
        self.assertEqual(loaded.size, (64, 64))

    def test_overwrite_handling(self) -> None:
        """Test overwrite_existing flag behavior."""
        # Initial save
        img_path1 = self.manager.save_attacked_image(self.image, "000000")
        mtime1 = img_path1.stat().st_mtime

        # Re-save with overwrite_existing=False should return existing file without overwrite
        img_path2 = self.manager.save_attacked_image(self.image, "000000")
        self.assertEqual(img_path1, img_path2)
        self.assertEqual(mtime1, img_path2.stat().st_mtime)

        # Re-save with overwrite_existing=True
        overwrite_config = GeneratorConfig(
            output_directory=Path(self.temp_dir.name) / "output",
            overwrite_existing=True,
        )
        overwrite_manager = OutputManager(config=overwrite_config)
        img_path3 = overwrite_manager.save_attacked_image(self.image, "000000")
        self.assertTrue(img_path3.exists())

    def test_copy_label_and_calib(self) -> None:
        """Test copying label and calibration text files."""
        lbl_source = Path(self.temp_dir.name) / "label.txt"
        lbl_source.write_text("Car 0.0 0 0.0 ...", encoding="utf-8")

        cal_source = Path(self.temp_dir.name) / "calib.txt"
        cal_source.write_text("P2: 1.0 0.0 ...", encoding="utf-8")

        copied_lbl = self.manager.copy_label(lbl_source, "000000")
        copied_cal = self.manager.copy_calib(cal_source, "000000")

        self.assertIsNotNone(copied_lbl)
        self.assertIsNotNone(copied_cal)
        assert copied_lbl is not None
        assert copied_cal is not None

        self.assertTrue(copied_lbl.exists())
        self.assertTrue(copied_cal.exists())
        self.assertEqual(copied_lbl.read_text(encoding="utf-8"), "Car 0.0 0 0.0 ...")

    def test_missing_source_files_handled_gracefully(self) -> None:
        """Test handling missing source label/calib files returns None."""
        missing_lbl = self.manager.copy_label(Path(self.temp_dir.name) / "does_not_exist.txt", "000000")
        self.assertIsNone(missing_lbl)

    def test_is_sample_processed(self) -> None:
        """Test checking if sample output files exist."""
        self.assertFalse(self.manager.is_sample_processed("000000"))

        # Save image and metadata
        self.manager.save_attacked_image(self.image, "000000")
        self.manager.save_metadata(
            sample_id="000000",
            pattern=LaserPattern(),
            attack_config=AttackConfig(),
            execution_metadata={},
        )
        self.assertTrue(self.manager.is_sample_processed("000000"))


if __name__ == "__main__":
    unittest.main()
