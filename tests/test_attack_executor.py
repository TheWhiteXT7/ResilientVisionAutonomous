"""Unit tests for AttackExecutor in-memory attack execution."""

import tempfile
import unittest
from pathlib import Path
from PIL import Image

from attack_engine import AttackConfig, AttackPipeline, LaserPattern
from attack_engine.target_selection import TargetSelectionError
from dataset_generator.attack_executor import AttackExecutor
from dataset_loader import KittiSample
from dataset_loader.annotation_parser import Annotation


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

    def _make_car_annotation(self) -> Annotation:
        return Annotation(
            class_name="Car",
            truncated=0.0,
            occluded=0,
            alpha=0.0,
            bbox=(20.0, 20.0, 80.0, 80.0),
            dimensions=(1.5, 1.6, 3.5),
            location=(0.0, 0.0, 10.0),
            rotation_y=0.0,
        )

    def test_execute_targeted_passes_annotations(self) -> None:
        """Test executor threads KittiSample annotations into the pipeline."""
        captured = {}
        original = self.pipeline.execute

        def spy(image, pattern_type="random", **kwargs):
            captured["pattern_type"] = pattern_type
            captured["kwargs"] = kwargs
            return original(image=image, pattern_type=pattern_type, **kwargs)

        self.pipeline.execute = spy  # type: ignore[method-assign]
        self.sample.annotations = [self._make_car_annotation()]
        attacked_image, pattern, metadata = self.executor.execute(
            self.sample, pattern_type="targeted"
        )

        self.assertEqual(captured["pattern_type"], "targeted")
        self.assertIn("annotations", captured["kwargs"])
        self.assertEqual(captured["kwargs"]["annotations"], self.sample.annotations)
        self.assertEqual(captured["kwargs"]["target_class"], "Car")

        self.assertIsInstance(attacked_image, Image.Image)
        self.assertEqual(len(pattern), 3)
        for spot in pattern:
            self.assertGreaterEqual(spot.x, 20.0)
            self.assertLessEqual(spot.x, 80.0)
            self.assertGreaterEqual(spot.y, 20.0)
            self.assertLessEqual(spot.y, 80.0)

    def test_execute_targeted_metadata_includes_target(self) -> None:
        """Test targeted execution records the selected target in metadata."""
        self.sample.annotations = [self._make_car_annotation()]
        _, _, metadata = self.executor.execute(self.sample, pattern_type="targeted")

        self.assertEqual(metadata["pattern_type"], "targeted")
        self.assertEqual(metadata["target"]["class_name"], "Car")
        self.assertEqual(metadata["target"]["bbox"], [20.0, 20.0, 80.0, 80.0])
        self.assertIs(metadata["target_found"], True)
        self.assertIs(metadata["preserved"], False)

    def test_execute_random_receives_no_annotations(self) -> None:
        """Test random execution receives no target information."""
        captured = {}
        original = self.pipeline.execute

        def spy(image, pattern_type="random", **kwargs):
            captured["pattern_type"] = pattern_type
            captured["kwargs"] = kwargs
            return original(image=image, pattern_type=pattern_type, **kwargs)

        self.pipeline.execute = spy  # type: ignore[method-assign]
        self.sample.annotations = [self._make_car_annotation()]
        self.executor.execute(self.sample, pattern_type="random")

        self.assertEqual(captured["pattern_type"], "random")
        self.assertNotIn("annotations", captured["kwargs"])
        self.assertNotIn("target_class", captured["kwargs"])

    def test_execute_random_metadata_has_no_target(self) -> None:
        """Test random execution metadata carries no target information."""
        self.sample.annotations = [self._make_car_annotation()]
        _, _, metadata = self.executor.execute(self.sample, pattern_type="random")

        self.assertIsNone(metadata["target"])
        self.assertIsNone(metadata["target_found"])
        self.assertIs(metadata["preserved"], False)

    def test_execute_targeted_no_target_preserves_original(self) -> None:
        """Test default 'preserve' policy keeps the original image unchanged."""
        self.sample.annotations = [
            Annotation(
                class_name="Pedestrian",
                truncated=0.0,
                occluded=0,
                alpha=0.0,
                bbox=(20.0, 20.0, 80.0, 80.0),
                dimensions=(1.7, 0.6, 0.8),
                location=(0.0, 0.0, 5.0),
                rotation_y=0.0,
            )
        ]
        original_image = self.sample.load_image()

        attacked_image, pattern, metadata = self.executor.execute(
            self.sample, pattern_type="targeted"
        )

        self.assertEqual(attacked_image.size, original_image.size)
        self.assertEqual(attacked_image.tobytes(), original_image.tobytes())
        self.assertEqual(len(pattern), 0)
        self.assertEqual(metadata["pattern_type"], "targeted")
        self.assertIs(metadata["target_found"], False)
        self.assertIs(metadata["preserved"], True)
        self.assertIsNone(metadata["target"])
        self.assertEqual(metadata["spots_count"], 0)

    def test_execute_targeted_no_target_fail_policy_raises(self) -> None:
        """Test 'fail' policy re-raises TargetSelectionError when no target exists."""
        self.sample.annotations = [
            Annotation(
                class_name="Pedestrian",
                truncated=0.0,
                occluded=0,
                alpha=0.0,
                bbox=(20.0, 20.0, 80.0, 80.0),
                dimensions=(1.7, 0.6, 0.8),
                location=(0.0, 0.0, 5.0),
                rotation_y=0.0,
            )
        ]
        with self.assertRaises(TargetSelectionError):
            self.executor.execute(
                self.sample, pattern_type="targeted", missing_target_policy="fail"
            )

    def test_execute_targeted_policy_from_pipeline_config(self) -> None:
        """Test pipeline config missing_target_policy='fail' is honored."""
        self.pipeline.config = AttackConfig(
            laser_color=(255, 0, 0),
            spot_radius=10.0,
            intensity=1.0,
            max_spots=3,
            missing_target_policy="fail",
        )
        self.sample.annotations = [
            Annotation(
                class_name="Pedestrian",
                truncated=0.0,
                occluded=0,
                alpha=0.0,
                bbox=(20.0, 20.0, 80.0, 80.0),
                dimensions=(1.7, 0.6, 0.8),
                location=(0.0, 0.0, 5.0),
                rotation_y=0.0,
            )
        ]
        with self.assertRaises(TargetSelectionError):
            self.executor.execute(self.sample, pattern_type="targeted")

    def test_execute_targeted_invalid_policy_raises(self) -> None:
        """Test an unsupported missing_target_policy raises ValueError."""
        self.sample.annotations = [self._make_car_annotation()]
        with self.assertRaises(ValueError):
            self.executor.execute(
                self.sample, pattern_type="targeted", missing_target_policy="bogus"
            )


if __name__ == "__main__":
    unittest.main()
