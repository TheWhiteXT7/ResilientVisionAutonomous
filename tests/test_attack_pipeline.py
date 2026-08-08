"""Unit tests for AttackPipeline orchestration and public API."""

import unittest
from PIL import Image
from attack_engine.attack_config import AttackConfig
from attack_engine.attack_pipeline import AttackPipeline, apply_attack
from attack_engine.laser_pattern import LaserPattern
from attack_engine.target_selection import TargetRegion
from dataset_loader.annotation_parser import Annotation


def make_car_annotation(bbox):
    """Build a minimal Car annotation."""
    return Annotation(
        class_name="Car",
        truncated=0.0,
        occluded=0,
        alpha=0.0,
        bbox=bbox,
        dimensions=(1.5, 1.6, 3.5),
        location=(0.0, 0.0, 10.0),
        rotation_y=0.0,
    )


class TestAttackPipeline(unittest.TestCase):
    """Test suite for AttackPipeline and apply_attack function."""

    def setUp(self) -> None:
        """Set up test image."""
        self.image = Image.new("RGB", (200, 200), (100, 100, 100))

    def test_pipeline_execute_default(self) -> None:
        """Test pipeline execution returns attacked image and pattern."""
        pipeline = AttackPipeline()
        attacked, pattern = pipeline.execute(self.image, pattern_type="random")

        self.assertIsInstance(attacked, Image.Image)
        self.assertIsInstance(pattern, LaserPattern)
        self.assertEqual(attacked.size, self.image.size)
        self.assertEqual(len(pattern), pipeline.config.max_spots)

    def test_apply_attack_functional_api(self) -> None:
        """Test top-level apply_attack function with kwargs overrides."""
        attacked, pattern = apply_attack(
            self.image,
            pattern_type="horizontal_line",
            laser_color=(0, 255, 0),
            max_spots=4,
            random_seed=123,
        )

        self.assertIsInstance(attacked, Image.Image)
        self.assertIsInstance(pattern, LaserPattern)
        self.assertEqual(len(pattern), 4)
        self.assertEqual(pattern[0].color, (0, 255, 0))

    def test_pipeline_custom_config(self) -> None:
        """Test AttackPipeline initialized with custom AttackConfig."""
        custom_config = AttackConfig(
            laser_color=(0, 0, 255),
            spot_radius=25.0,
            max_spots=2,
        )
        pipeline = AttackPipeline(config=custom_config)
        attacked, pattern = pipeline.execute(self.image, pattern_type="single")

        self.assertEqual(len(pattern), 1)
        self.assertEqual(pattern[0].color, (0, 0, 255))
        self.assertEqual(pattern[0].radius, 25.0)

    def test_invalid_image_type(self) -> None:
        """Test passing non-PIL image raises TypeError."""
        pipeline = AttackPipeline()
        with self.assertRaises(TypeError):
            pipeline.execute("invalid_image")  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            apply_attack("invalid_image")  # type: ignore[arg-type]

    def test_execute_targeted_with_annotations(self) -> None:
        """Test targeted execution selects a target and confines spots to its bbox."""
        pipeline = AttackPipeline(config=AttackConfig(max_spots=4, random_seed=42, spot_radius=10.0))
        ann = make_car_annotation((50.0, 50.0, 150.0, 150.0))
        attacked, pattern = pipeline.execute(
            self.image, pattern_type="targeted", annotations=[ann]
        )

        self.assertIsInstance(attacked, Image.Image)
        self.assertEqual(len(pattern), 4)
        for spot in pattern:
            self.assertGreaterEqual(spot.x, 50.0)
            self.assertLessEqual(spot.x, 150.0)
            self.assertGreaterEqual(spot.y, 50.0)
            self.assertLessEqual(spot.y, 150.0)

    def test_execute_targeted_modified_image(self) -> None:
        """Test a targeted attack produces a modified image."""
        pipeline = AttackPipeline(config=AttackConfig(max_spots=3, random_seed=1, spot_radius=10.0))
        ann = make_car_annotation((50.0, 50.0, 150.0, 150.0))
        attacked, _ = pipeline.execute(self.image, pattern_type="targeted", annotations=[ann])

        self.assertNotEqual(attacked.tobytes(), self.image.tobytes())

    def test_execute_targeted_requires_annotations(self) -> None:
        """Test targeted execution without annotations or region fails clearly."""
        pipeline = AttackPipeline()
        with self.assertRaises(ValueError):
            pipeline.execute(self.image, pattern_type="targeted")

    def test_execute_targeted_with_explicit_region(self) -> None:
        """Test targeted execution accepts an explicit TargetRegion."""
        pipeline = AttackPipeline(config=AttackConfig(max_spots=2, random_seed=3, spot_radius=10.0))
        region = TargetRegion(class_name="Car", bbox=(50.0, 50.0, 150.0, 150.0))
        attacked, pattern = pipeline.execute(
            self.image, pattern_type="targeted", target_region=region
        )

        self.assertEqual(len(pattern), 2)
        self.assertEqual(pipeline.last_target, region)

    def test_execute_targeted_uses_config_target_class(self) -> None:
        """Test config target_class filters candidate annotations."""
        pipeline = AttackPipeline(config=AttackConfig(target_class="Truck"))
        ann = make_car_annotation((50.0, 50.0, 150.0, 150.0))
        with self.assertRaises(ValueError):
            pipeline.execute(self.image, pattern_type="targeted", annotations=[ann])

    def test_execute_target_class_override(self) -> None:
        """Test explicit target_class override wins over config."""
        pipeline = AttackPipeline(config=AttackConfig(target_class="Truck", max_spots=2, random_seed=5))
        ann = make_car_annotation((50.0, 50.0, 150.0, 150.0))
        attacked, pattern = pipeline.execute(
            self.image,
            pattern_type="targeted",
            annotations=[ann],
            target_class="Car",
        )

        self.assertEqual(len(pattern), 2)
        self.assertEqual(pipeline.last_target.class_name, "Car")

    def test_apply_attack_targeted_functional_api(self) -> None:
        """Test top-level apply_attack supports targeted attacks."""
        ann = make_car_annotation((50.0, 50.0, 150.0, 150.0))
        attacked, pattern = apply_attack(
            self.image,
            pattern_type="targeted",
            annotations=[ann],
            target_class="Car",
            max_spots=3,
            random_seed=9,
        )

        self.assertIsInstance(attacked, Image.Image)
        self.assertEqual(len(pattern), 3)
        for spot in pattern:
            self.assertGreaterEqual(spot.x, 50.0)
            self.assertLessEqual(spot.x, 150.0)

    def test_random_execute_unchanged(self) -> None:
        """Test random execution receives no target and still behaves as before."""
        pipeline = AttackPipeline(config=AttackConfig(max_spots=4, random_seed=42))
        attacked, pattern = pipeline.execute(self.image, pattern_type="random")

        self.assertEqual(len(pattern), 4)
        self.assertIsNone(pipeline.last_target)


if __name__ == "__main__":
    unittest.main()
