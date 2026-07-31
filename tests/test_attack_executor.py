"""Unit tests for AttackExecutor in-memory attack execution."""

import tempfile
import unittest
from pathlib import Path
from PIL import Image

from attack_engine import AttackConfig, AttackPipeline, LaserPattern
from dataset_generator.attack_executor import AttackExecutor
from dataset_loader import KittiSample


class TestAttackExecutor(unittest.TestCase):
    """Test suite for AttackExecutor in-memory execution."""

    def setUp(self) -> None:
        """Set up test environment and sample."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.img_path = Path(self.temp_dir.name) / "000000.png"
        img = Image.new("RGB", (100, 100), (128, 128, 128))
        img.save(self.img_path)

        self.sample = KittiSample(
            sample_id="000000",
            image_path=self.img_path,
        )

        self.attack_config = AttackConfig(
            laser_color=(255, 0, 0),
            spot_radius=10.0,
            intensity=1.0,
            max_spots=3,
        )
        self.pipeline = AttackPipeline(config=self.attack_config)
        self.executor = AttackExecutor(pipeline=self.pipeline)

    def tearDown(self) -> None:
        """Clean up temporary directory."""
        self.temp_dir.cleanup()

    def test_execute_returns_valid_outputs(self) -> None:
        """Test that execution returns attacked image, pattern, and metadata dict."""
        attacked_image, pattern, metadata = self.executor.execute(
            sample=self.sample,
            pattern_type="random",
        )

        self.assertIsInstance(attacked_image, Image.Image)
        self.assertEqual(attacked_image.size, (100, 100))
        self.assertIsInstance(pattern, LaserPattern)
        self.assertEqual(len(pattern), 3)

        self.assertEqual(metadata["sample_id"], "000000")
        self.assertEqual(metadata["pattern_type"], "random")
        self.assertEqual(metadata["spots_count"], 3)
        self.assertIn("timestamp", metadata)
        self.assertIn("processing_time_ms", metadata)

    def test_execute_does_not_write_files(self) -> None:
        """Verify AttackExecutor produces no files in working directory."""
        files_before = set(Path(self.temp_dir.name).iterdir())
        self.executor.execute(self.sample)
        files_after = set(Path(self.temp_dir.name).iterdir())

        self.assertEqual(files_before, files_after)

    def test_execute_with_preloaded_image(self) -> None:
        """Test execution with pre-loaded PIL Image in KittiSample."""
        self.sample.image = Image.new("RGB", (50, 50), (200, 200, 200))
        attacked_image, pattern, metadata = self.executor.execute(self.sample)

        self.assertEqual(attacked_image.size, (50, 50))
        self.assertEqual(len(pattern), 3)

    def test_invalid_sample_raises_type_error(self) -> None:
        """Test passing invalid sample object raises TypeError."""
        with self.assertRaises(TypeError):
            self.executor.execute("not_a_sample")  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
