"""Unit tests for object-aware target selection logic."""

import unittest

from attack_engine.target_selection import (
    TargetRegion,
    TargetSelectionError,
    select_target,
)
from dataset_loader.annotation_parser import Annotation


def make_annotation(class_name: str, bbox) -> Annotation:
    """Build a minimal KITTI Annotation for testing."""
    return Annotation(
        class_name=class_name,
        truncated=0.0,
        occluded=0,
        alpha=0.0,
        bbox=bbox,
        dimensions=(1.5, 1.6, 3.5),
        location=(0.0, 0.0, 10.0),
        rotation_y=0.0,
    )


class TestTargetRegion(unittest.TestCase):
    """Test suite for the TargetRegion immutable data model."""

    def test_construction_and_fields(self) -> None:
        region = TargetRegion(class_name="Car", bbox=(10.0, 20.0, 50.0, 60.0))
        self.assertEqual(region.class_name, "Car")
        self.assertEqual(region.bbox, (10.0, 20.0, 50.0, 60.0))
        self.assertIsNone(region.metadata)

    def test_immutability(self) -> None:
        region = TargetRegion(class_name="Car", bbox=(1, 2, 3, 4))
        with self.assertRaises(AttributeError):
            region.class_name = "Truck"  # type: ignore[misc]
        with self.assertRaises(AttributeError):
            region.bbox = (5, 6, 7, 8)  # type: ignore[misc]

    def test_bbox_list_normalized_to_tuple(self) -> None:
        region = TargetRegion(class_name="Car", bbox=[10, 20, 50, 60])  # type: ignore[arg-type]
        self.assertIsInstance(region.bbox, tuple)
        self.assertEqual(region.bbox, (10.0, 20.0, 50.0, 60.0))

    def test_invalid_bbox_rejected(self) -> None:
        invalid_boxes = [
            (50.0, 20.0, 10.0, 60.0),   # x1 >= x2
            (10.0, 60.0, 50.0, 20.0),   # y1 >= y2
            (10.0, 20.0, 10.0, 60.0),   # zero-width
            (10.0, 20.0, 50.0, 20.0),   # zero-height
            (10, 20, 50),               # wrong length
            (10, 20, 50, "60"),         # non-number
        ]
        for box in invalid_boxes:
            with self.subTest(bbox=box):
                with self.assertRaises((ValueError, TypeError)):
                    TargetRegion(class_name="Car", bbox=box)  # type: ignore[arg-type]

    def test_empty_class_rejected(self) -> None:
        with self.assertRaises(ValueError):
            TargetRegion(class_name="", bbox=(1, 2, 3, 4))
        with self.assertRaises(ValueError):
            TargetRegion(class_name="   ", bbox=(1, 2, 3, 4))
        with self.assertRaises(TypeError):
            TargetRegion(class_name=123, bbox=(1, 2, 3, 4))  # type: ignore[arg-type]


class TestSelectTarget(unittest.TestCase):
    """Test suite for deterministic target selection."""

    def setUp(self) -> None:
        self.image_size = (100, 100)
        self.cars = [
            make_annotation("Car", (10.0, 10.0, 50.0, 50.0)),
            make_annotation("Car", (20.0, 20.0, 80.0, 80.0)),
        ]
        self.pedestrian = make_annotation("Pedestrian", (5.0, 5.0, 30.0, 60.0))

    def test_default_target_class_is_car(self) -> None:
        region = select_target([self.pedestrian] + self.cars, image_size=self.image_size)
        self.assertEqual(region.class_name, "Car")
        self.assertIn(region.bbox, [(10.0, 10.0, 50.0, 50.0), (20.0, 20.0, 80.0, 80.0)])

    def test_returns_target_region(self) -> None:
        region = select_target(self.cars, image_size=self.image_size)
        self.assertIsInstance(region, TargetRegion)

    def test_deterministic_with_same_seed(self) -> None:
        r1 = select_target(self.cars, image_size=self.image_size, random_seed=42)
        r2 = select_target(self.cars, image_size=self.image_size, random_seed=42)
        self.assertEqual(r1, r2)
        self.assertEqual(r1.bbox, r2.bbox)

    def test_no_matching_target_raises(self) -> None:
        with self.assertRaises(TargetSelectionError):
            select_target([self.pedestrian], image_size=self.image_size)

    def test_empty_annotations_raises(self) -> None:
        with self.assertRaises(TargetSelectionError):
            select_target([], image_size=self.image_size)

    def test_target_class_case_insensitive(self) -> None:
        region = select_target(self.cars, image_size=self.image_size, target_class="car")
        self.assertEqual(region.class_name, "Car")

    def test_invalid_bbox_excluded(self) -> None:
        invalid = make_annotation("Car", (50.0, 50.0, 20.0, 80.0))
        valid = make_annotation("Car", (30.0, 30.0, 70.0, 70.0))
        region = select_target([invalid, valid], image_size=self.image_size)
        self.assertEqual(region.bbox, (30.0, 30.0, 70.0, 70.0))

    def test_out_of_image_bbox_excluded(self) -> None:
        out_left = make_annotation("Car", (-10.0, 10.0, 50.0, 50.0))
        out_right = make_annotation("Car", (90.0, 10.0, 150.0, 50.0))
        out_top = make_annotation("Car", (10.0, -5.0, 50.0, 50.0))
        out_bottom = make_annotation("Car", (10.0, 90.0, 50.0, 120.0))
        valid = make_annotation("Car", (30.0, 30.0, 70.0, 70.0))
        region = select_target(
            [out_left, out_right, out_top, out_bottom, valid],
            image_size=self.image_size,
        )
        self.assertEqual(region.bbox, (30.0, 30.0, 70.0, 70.0))

    def test_all_invalid_raises(self) -> None:
        with self.assertRaises(TargetSelectionError):
            select_target(
                [make_annotation("Car", (50.0, 50.0, 20.0, 80.0))],
                image_size=self.image_size,
            )

    def test_all_out_of_image_raises(self) -> None:
        with self.assertRaises(TargetSelectionError):
            select_target(
                [make_annotation("Car", (90.0, 10.0, 150.0, 50.0))],
                image_size=self.image_size,
            )

    def test_target_region_objects_accepted(self) -> None:
        regions = [
            TargetRegion(class_name="Car", bbox=(10.0, 10.0, 40.0, 40.0)),
            TargetRegion(class_name="Car", bbox=(30.0, 30.0, 80.0, 80.0)),
        ]
        region = select_target(regions, image_size=self.image_size, random_seed=7)
        self.assertIsInstance(region, TargetRegion)
        self.assertEqual(region.class_name, "Car")

    def test_metadata_preserved(self) -> None:
        ann = make_annotation("Car", (10.0, 10.0, 50.0, 50.0))
        region = select_target([ann], image_size=self.image_size)
        self.assertIs(region.metadata, ann)

    def test_invalid_image_size_rejected(self) -> None:
        with self.assertRaises(ValueError):
            select_target(self.cars, image_size=(0, 100))


if __name__ == "__main__":
    unittest.main()
