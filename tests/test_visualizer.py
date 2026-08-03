"""Unit tests for YoloVisualizer module."""

from pathlib import Path
from PIL import Image
import pytest

from models.predictor import DetectionBox, DetectionResult
from models.visualizer import YoloVisualizer


def test_draw_bounding_boxes() -> None:
    """Test YoloVisualizer draw_bounding_boxes."""
    vis = YoloVisualizer()
    img = Image.new("RGB", (200, 200), color="white")
    boxes = [
        DetectionBox(class_id=0, class_name="Car", confidence=0.95, bbox=(10.0, 10.0, 100.0, 100.0)),
        DetectionBox(class_id=1, class_name="Pedestrian", confidence=0.8, bbox=(110.0, 50.0, 150.0, 150.0)),
    ]

    annotated = vis.draw_bounding_boxes(img, boxes)

    assert isinstance(annotated, Image.Image)
    assert annotated.size == (200, 200)


def test_visualize_prediction(tmp_path: Path) -> None:
    """Test YoloVisualizer visualize_prediction and save image output."""
    vis = YoloVisualizer()
    img = Image.new("RGB", (100, 100), color="blue")
    result = DetectionResult(
        sample_id="sample1",
        image_path=tmp_path / "img.png",
        boxes=[DetectionBox(class_id=0, class_name="Car", confidence=0.9, bbox=(5.0, 5.0, 50.0, 50.0))],
    )

    out_file = tmp_path / "annotated.png"
    annotated = vis.visualize_prediction(img, result, output_path=out_file)

    assert out_file.exists()
    assert isinstance(annotated, Image.Image)


def test_create_side_by_side_comparison(tmp_path: Path) -> None:
    """Test YoloVisualizer side-by-side comparative visualization generation."""
    vis = YoloVisualizer()
    img1 = Image.new("RGB", (100, 100), color="red")
    img2 = Image.new("RGB", (100, 100), color="green")

    out_file = tmp_path / "comparison.png"
    composite = vis.create_side_by_side_comparison(
        img1, img2, title1="Clean Input", title2="Laser Attacked", output_path=out_file
    )

    assert out_file.exists()
    assert isinstance(composite, Image.Image)
    assert composite.width > 200  # Combined width of both images + padding
