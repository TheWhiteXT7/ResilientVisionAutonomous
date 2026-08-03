"""General helper functions for YOLO model integration."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import yaml

from dataset_loader.split_manager import SplitManager

logger = logging.getLogger(__name__)

DEFAULT_KITTI_CLASSES: Dict[str, int] = {
    "Car": 0,
    "Pedestrian": 1,
    "Cyclist": 2,
    "Van": 3,
    "Truck": 4,
    "Person_sitting": 5,
    "Tram": 6,
    "Misc": 7,
}


def get_default_class_mapping() -> Dict[str, int]:
    """Get default object class name to class index mapping for KITTI dataset.

    Returns:
        Dictionary mapping class string names to integer class IDs.
    """
    return DEFAULT_KITTI_CLASSES.copy()


def kitti_bbox_to_yolo(
    bbox: Tuple[float, float, float, float], img_width: int, img_height: int
) -> Tuple[float, float, float, float]:
    """Convert KITTI bounding box (xmin, ymin, xmax, ymax) to YOLO normalized (x_center, y_center, width, height).

    Args:
        bbox: Bounding box in pixel coordinates (left, top, right, bottom).
        img_width: Image width in pixels.
        img_height: Image height in pixels.

    Returns:
        Normalized bounding box coordinates (x_center, y_center, width, height) in range [0.0, 1.0].

    Raises:
        ValueError: If image dimensions are non-positive.
    """
    if img_width <= 0 or img_height <= 0:
        raise ValueError(
            f"Image dimensions must be positive, got width={img_width}, height={img_height}"
        )

    left, top, right, bottom = bbox
    box_w = max(0.0, right - left)
    box_h = max(0.0, bottom - top)
    x_center = left + box_w / 2.0
    y_center = top + box_h / 2.0

    return (
        x_center / img_width,
        y_center / img_height,
        box_w / img_width,
        box_h / img_height,
    )


def yolo_bbox_to_kitti(
    bbox: Tuple[float, float, float, float], img_width: int, img_height: int
) -> Tuple[float, float, float, float]:
    """Convert YOLO normalized bounding box (x_center, y_center, width, height) to KITTI pixel (xmin, ymin, xmax, ymax).

    Args:
        bbox: Normalized bounding box (x_center, y_center, width, height).
        img_width: Image width in pixels.
        img_height: Image height in pixels.

    Returns:
        Bounding box pixel coordinates (left, top, right, bottom).

    Raises:
        ValueError: If image dimensions are non-positive.
    """
    if img_width <= 0 or img_height <= 0:
        raise ValueError(
            f"Image dimensions must be positive, got width={img_width}, height={img_height}"
        )

    x_center_norm, y_center_norm, w_norm, h_norm = bbox
    x_center = x_center_norm * img_width
    y_center = y_center_norm * img_height
    box_w = w_norm * img_width
    box_h = h_norm * img_height

    left = x_center - box_w / 2.0
    top = y_center - box_h / 2.0
    right = x_center + box_w / 2.0
    bottom = y_center + box_h / 2.0

    return (left, top, right, bottom)


def prepare_yolo_dataset(
    dataset: Any,
    output_dir: Union[str, Path],
    class_mapping: Optional[Dict[str, int]] = None,
    val_ratio: float = 0.20,
    seed: Optional[int] = 42,
    splits_dir: Optional[Union[str, Path]] = None,
) -> Path:
    """Prepare YOLO-formatted dataset configuration and structure from a dataset loader or path.

    Integrates with SplitManager to maintain or generate reproducible train/val splits.

    Args:
        dataset: KittiLoader instance, Path to data.yaml, or dataset directory Path.
        output_dir: Destination directory for generating YOLO labels/images and data.yaml.
        class_mapping: Optional mapping from class names to integer IDs.
        val_ratio: Fraction assigned to validation split when auto-generating splits (default 0.20).
        seed: Random seed for reproducible split generation.
        splits_dir: Optional custom directory path for ImageSets split files.

    Returns:
        Path object pointing to the generated or existing data.yaml dataset config file.

    Raises:
        ValueError: If dataset type is invalid or cannot be processed.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Case 1: dataset is already a path to a data.yaml file
    if isinstance(dataset, (str, Path)):
        ds_path = Path(dataset)
        if ds_path.is_file() and ds_path.name.endswith(".yaml"):
            return ds_path
        elif ds_path.is_dir():
            yaml_candidate = ds_path / "data.yaml"
            if yaml_candidate.is_file():
                return yaml_candidate

    cls_map = class_mapping if class_mapping is not None else get_default_class_mapping()
    names = {idx: name for name, idx in cls_map.items()}

    # Resolve splits directory and SplitManager
    kitti_dir: Optional[Path] = getattr(dataset, "kitti_dir", None)
    if kitti_dir is None and isinstance(dataset, (str, Path)):
        kitti_dir = Path(dataset)

    if splits_dir is not None:
        target_splits_dir = Path(splits_dir)
    elif kitti_dir is not None and (kitti_dir / "ImageSets").exists():
        target_splits_dir = kitti_dir / "ImageSets"
    else:
        target_splits_dir = out_dir / "ImageSets"

    split_mgr = SplitManager(kitti_dir=kitti_dir, splits_dir=target_splits_dir)

    # Check if train.txt and val.txt exist in target_splits_dir
    train_txt = target_splits_dir / "train.txt"
    val_txt = target_splits_dir / "val.txt"
    test_txt = target_splits_dir / "test.txt"

    train_ids: Set[str] = set()
    val_ids: Set[str] = set()
    test_ids: Set[str] = set()

    if train_txt.exists() and val_txt.exists():
        train_ids = set(split_mgr.load_split("train"))
        val_ids = set(split_mgr.load_split("val"))
        if test_txt.exists():
            test_ids = set(split_mgr.load_split("test"))
    else:
        # Extract all sample IDs from dataset
        all_sids: List[str] = []
        if hasattr(dataset, "sample_ids") and dataset.sample_ids:
            all_sids = list(dataset.sample_ids)
        elif hasattr(dataset, "__iter__"):
            all_sids = [getattr(s, "sample_id", f"sample_{i}") for i, s in enumerate(dataset)]

        if all_sids:
            train_ratio = max(0.0, min(1.0, 1.0 - val_ratio))
            tr_list, val_list = split_mgr.create_random_split(
                all_sids, train_ratio=train_ratio, seed=seed
            )

            # Ensure train set is non-empty if samples exist
            if not tr_list and val_list:
                tr_list = val_list
            elif len(all_sids) > 1 and not val_list and val_ratio > 0:
                val_list = [tr_list.pop()]

            split_mgr.save_split("train", tr_list)
            split_mgr.save_split("val", val_list)

            train_ids = set(tr_list)
            val_ids = set(val_list)

    # Prepare directories
    split_names = ["train", "val"]
    if test_ids:
        split_names.append("test")

    for sname in split_names:
        (out_dir / "images" / sname).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / sname).mkdir(parents=True, exist_ok=True)

    # Process samples into appropriate splits
    if hasattr(dataset, "__iter__"):
        for sample in dataset:
            sample_id = getattr(sample, "sample_id", "sample")
            img_path = getattr(sample, "image_path", None)

            target_splits: List[str] = []
            if sample_id in train_ids:
                target_splits.append("train")
            if sample_id in val_ids and "val" not in target_splits:
                target_splits.append("val")
            if sample_id in test_ids and "test" not in target_splits:
                target_splits.append("test")

            # Fallback if sample not matched to any split
            if not target_splits:
                target_splits = ["train"]

            # Dynamic image size extraction
            img_w, img_h = None, None
            if hasattr(sample, "image") and sample.image is not None:
                img_w, img_h = sample.image.size
            elif img_path and Path(img_path).exists():
                try:
                    from PIL import Image

                    with Image.open(img_path) as img:
                        img_w, img_h = img.size
                except Exception:
                    pass

            if img_w is None or img_h is None or img_w <= 0 or img_h <= 0:
                img_w, img_h = 1242, 375  # Default KITTI camera resolution fallback

            # Convert annotations to YOLO txt format
            annotations = getattr(sample, "annotations", [])
            yolo_lines: List[str] = []
            for ann in annotations:
                cls_name = getattr(ann, "class_name", "DontCare")
                if cls_name in cls_map:
                    cls_id = cls_map[cls_name]
                    bbox = getattr(ann, "bbox", (0, 0, 0, 0))
                    xc, yc, bw, bh = kitti_bbox_to_yolo(bbox, img_w, img_h)
                    yolo_lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")

            for split_dest in target_splits:
                target_img_path = out_dir / "images" / split_dest / f"{sample_id}.png"
                if img_path and Path(img_path).exists():
                    if not target_img_path.exists():
                        try:
                            import shutil

                            shutil.copy2(Path(img_path), target_img_path)
                        except Exception as err:
                            logger.warning(
                                f"Failed to copy image {img_path} to {target_img_path}: {err}"
                            )

                label_file = out_dir / "labels" / split_dest / f"{sample_id}.txt"
                with open(label_file, "w", encoding="utf-8") as f:
                    f.writelines(yolo_lines)

    # Write data.yaml configuration file
    yaml_data: Dict[str, Any] = {
        "path": str(out_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": names,
    }

    test_img_dir = out_dir / "images" / "test"
    if test_img_dir.exists() and any(test_img_dir.iterdir()):
        yaml_data["test"] = "images/test"

    yaml_path = out_dir / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(yaml_data, f, default_flow_style=False)

    logger.info(f"YOLO dataset config prepared at: {yaml_path}")
    return yaml_path
