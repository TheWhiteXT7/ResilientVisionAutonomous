"""Unit tests for models/utils.py helper functions and dataset preparation pipeline."""

from pathlib import Path
from unittest.mock import MagicMock
from PIL import Image
import pytest
import yaml

from dataset_loader.annotation_parser import Annotation
from dataset_loader.kitti_loader import KittiSample
from models.utils import (
    get_default_class_mapping,
    kitti_bbox_to_yolo,
    prepare_yolo_dataset,
    yolo_bbox_to_kitti,
)


def test_kitti_bbox_to_yolo_conversion() -> None:
    """Test KITTI to YOLO bbox conversion coordinates."""
    # Box: left=100, top=100, right=300, bottom=300 -> w=200, h=200, cx=200, cy=200
    # Image: 1000x1000 -> norm: cx=0.2, cy=0.2, w=0.2, h=0.2
    bbox = (100.0, 100.0, 300.0, 300.0)
    xc, yc, w, h = kitti_bbox_to_yolo(bbox, img_width=1000, img_height=1000)

    assert xc == pytest.approx(0.2)
    assert yc == pytest.approx(0.2)
    assert w == pytest.approx(0.2)
    assert h == pytest.approx(0.2)


def test_yolo_bbox_to_kitti_conversion() -> None:
    """Test YOLO to KITTI bbox conversion coordinates."""
    bbox_norm = (0.2, 0.2, 0.2, 0.2)
    left, top, right, bottom = yolo_bbox_to_kitti(
        bbox_norm, img_width=1000, img_height=1000
    )

    assert left == pytest.approx(100.0)
    assert top == pytest.approx(100.0)
    assert right == pytest.approx(300.0)
    assert bottom == pytest.approx(300.0)


def test_prepare_yolo_dataset_automatic_split_and_imagesets(tmp_path: Path) -> None:
    """Test automatic split generation, ImageSets creation, train/val directory creation, and data.yaml content."""
    img_dir = tmp_path / "raw_images"
    img_dir.mkdir(parents=True, exist_ok=True)

    samples = []
    # Create 10 dummy samples
    for i in range(10):
        sid = f"00000{i}"
        ipath = img_dir / f"{sid}.png"
        img = Image.new("RGB", (1000, 500), color="white")
        img.save(ipath)

        sample = KittiSample(
            sample_id=sid,
            image_path=ipath,
            annotations=[
                Annotation(
                    class_name="Car",
                    truncated=0.0,
                    occluded=0,
                    alpha=0.0,
                    bbox=(100.0, 50.0, 300.0, 150.0),
                    dimensions=(1.5, 1.6, 3.5),
                    location=(0.0, 0.0, 10.0),
                    rotation_y=0.0,
                )
            ],
            image=img,
        )
        samples.append(sample)

    output_dir = tmp_path / "yolo_prepared"
    yaml_path = prepare_yolo_dataset(
        dataset=samples,
        output_dir=output_dir,
        val_ratio=0.20,
        seed=42,
    )

    # 1. Check data.yaml file existence and contents
    assert yaml_path.exists()
    assert yaml_path.name == "data.yaml"

    with open(yaml_path, "r", encoding="utf-8") as f:
        yaml_content = yaml.safe_load(f)

    assert yaml_content["train"] == "images/train"
    assert yaml_content["val"] == "images/val"
    assert "names" in yaml_content
    assert yaml_content["names"][0] == "Car"

    # 2. Check ImageSets train.txt and val.txt creation
    imagesets_dir = output_dir / "ImageSets"
    assert imagesets_dir.exists()
    assert (imagesets_dir / "train.txt").exists()
    assert (imagesets_dir / "val.txt").exists()

    train_sids = (imagesets_dir / "train.txt").read_text().splitlines()
    val_sids = (imagesets_dir / "val.txt").read_text().splitlines()

    assert len(train_sids) == 8
    assert len(val_sids) == 2

    # 3. Check train/val directory structure
    train_img_dir = output_dir / "images" / "train"
    val_img_dir = output_dir / "images" / "val"
    train_lbl_dir = output_dir / "labels" / "train"
    val_lbl_dir = output_dir / "labels" / "val"

    assert train_img_dir.exists()
    assert val_img_dir.exists()
    assert train_lbl_dir.exists()
    assert val_lbl_dir.exists()

    assert len(list(train_img_dir.glob("*.png"))) == 8
    assert len(list(val_img_dir.glob("*.png"))) == 2
    assert len(list(train_lbl_dir.glob("*.txt"))) == 8
    assert len(list(val_lbl_dir.glob("*.txt"))) == 2


def test_prepare_yolo_dataset_existing_imagesets(tmp_path: Path) -> None:
    """Test prepare_yolo_dataset respects pre-existing ImageSets split files."""
    imagesets_dir = tmp_path / "ImageSets"
    imagesets_dir.mkdir(parents=True, exist_ok=True)

    (imagesets_dir / "train.txt").write_text("000001\n000002\n")
    (imagesets_dir / "val.txt").write_text("000003\n")

    img_dir = tmp_path / "raw_images"
    img_dir.mkdir(parents=True, exist_ok=True)

    samples = []
    for sid in ["000001", "000002", "000003"]:
        ipath = img_dir / f"{sid}.png"
        img = Image.new("RGB", (500, 500), color="blue")
        img.save(ipath)
        samples.append(
            KittiSample(
                sample_id=sid,
                image_path=ipath,
                annotations=[],
                image=img,
            )
        )

    output_dir = tmp_path / "output_existing_splits"
    yaml_path = prepare_yolo_dataset(
        dataset=samples,
        output_dir=output_dir,
        splits_dir=imagesets_dir,
    )

    assert yaml_path.exists()
    assert len(list((output_dir / "images" / "train").glob("*.png"))) == 2
    assert len(list((output_dir / "images" / "val").glob("*.png"))) == 1
