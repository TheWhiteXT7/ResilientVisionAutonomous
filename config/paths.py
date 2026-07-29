"""Path definitions for ResilientVisionAutonomous datasets and directories."""

from pathlib import Path

# Base project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Core subdirectories
CONFIG_DIR = PROJECT_ROOT / "config"
DATASETS_DIR = PROJECT_ROOT / "datasets"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = PROJECT_ROOT / "models"

# Default KITTI dataset directory
KITTI_DIR = DATASETS_DIR / "KITTI"

# Standard KITTI paths
KITTI_TRAIN_IMAGE_DIR = KITTI_DIR / "training" / "image_2"
KITTI_TRAIN_LABEL_DIR = KITTI_DIR / "training" / "label_2"
KITTI_TEST_IMAGE_DIR = KITTI_DIR / "testing" / "image_2"
KITTI_TEST_LABEL_DIR = KITTI_DIR / "testing" / "label_2"

# Alternative KITTI paths (data_object_* layout)
KITTI_ALT_TRAIN_IMAGE_DIR = (
    KITTI_DIR / "data_object_image_2" / "training" / "image_2"
)
KITTI_ALT_TEST_IMAGE_DIR = (
    KITTI_DIR / "data_object_image_2" / "testing" / "image_2"
)
KITTI_ALT_TRAIN_LABEL_DIR = (
    KITTI_DIR / "data_object_label_2" / "training" / "label_2"
)
