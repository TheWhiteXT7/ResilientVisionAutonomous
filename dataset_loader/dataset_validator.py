"""Dataset structure and integrity validator for KITTI dataset format."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from config.paths import (
    KITTI_ALT_TEST_IMAGE_DIR,
    KITTI_ALT_TRAIN_IMAGE_DIR,
    KITTI_ALT_TRAIN_LABEL_DIR,
    KITTI_DIR,
    KITTI_TEST_IMAGE_DIR,
    KITTI_TRAIN_IMAGE_DIR,
    KITTI_TRAIN_LABEL_DIR,
)

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}
LABEL_EXTENSIONS = {".txt"}


class DatasetValidator:
    """Validator for verifying KITTI dataset directory structures and pairs.

    This class checks the existence of required directories (training and
    testing images, training labels), counts image and label files, and
    identifies missing image-label pairs.
    """

    def __init__(self, kitti_dir: Optional[Path] = None) -> None:
        """Initialize DatasetValidator with target KITTI directory.

        Args:
            kitti_dir: Path to KITTI dataset root directory. If None, defaults
                to KITTI_DIR imported from config.paths.
        """
        self.kitti_dir = Path(kitti_dir) if kitti_dir else KITTI_DIR
        self._train_image_dir: Optional[Path] = None
        self._train_label_dir: Optional[Path] = None
        self._test_image_dir: Optional[Path] = None
        self._resolve_paths()

    def _resolve_paths(self) -> None:
        """Resolve actual training and testing directory paths.

        Checks both standard (training/image_2) and nested
        (data_object_image_2/training/image_2) directory layouts.
        """
        std_train_img = self.kitti_dir / "training" / "image_2"
        std_train_lbl = self.kitti_dir / "training" / "label_2"
        std_test_img = self.kitti_dir / "testing" / "image_2"

        alt_train_img = (
            self.kitti_dir / "data_object_image_2" / "training" / "image_2"
        )
        alt_train_lbl = (
            self.kitti_dir / "data_object_label_2" / "training" / "label_2"
        )
        alt_test_img = (
            self.kitti_dir / "data_object_image_2" / "testing" / "image_2"
        )

        if std_train_img.exists():
            self._train_image_dir = std_train_img
        elif alt_train_img.exists():
            self._train_image_dir = alt_train_img
        else:
            self._train_image_dir = std_train_img

        if std_train_lbl.exists():
            self._train_label_dir = std_train_lbl
        elif alt_train_lbl.exists():
            self._train_label_dir = alt_train_lbl
        else:
            self._train_label_dir = std_train_lbl

        if std_test_img.exists():
            self._test_image_dir = std_test_img
        elif alt_test_img.exists():
            self._test_image_dir = alt_test_img
        else:
            self._test_image_dir = std_test_img

    def validate_structure(self) -> Tuple[bool, List[str], List[str]]:
        """Validate existence of KITTI directory structure.

        Returns:
            Tuple of (is_valid, errors, warnings).
        """
        errors: List[str] = []
        warnings: List[str] = []

        if not self.kitti_dir.exists():
            msg = f"KITTI root directory does not exist: {self.kitti_dir}"
            logger.error(msg)
            errors.append(msg)
            return False, errors, warnings

        if not self.kitti_dir.is_dir():
            msg = f"KITTI root path is not a directory: {self.kitti_dir}"
            logger.error(msg)
            errors.append(msg)
            return False, errors, warnings

        if self._train_image_dir is None or not self._train_image_dir.exists():
            msg = (
                f"Training image directory missing: "
                f"{self.kitti_dir / 'training' / 'image_2'}"
            )
            logger.error(msg)
            errors.append(msg)

        if self._train_label_dir is None or not self._train_label_dir.exists():
            msg = (
                f"Training label directory missing: "
                f"{self.kitti_dir / 'training' / 'label_2'}"
            )
            logger.error(msg)
            errors.append(msg)

        if self._test_image_dir is None or not self._test_image_dir.exists():
            msg = (
                f"Testing image directory missing: "
                f"{self.kitti_dir / 'testing' / 'image_2'}"
            )
            logger.warning(msg)
            warnings.append(msg)

        is_valid = len(errors) == 0
        return is_valid, errors, warnings

    def count_files(self) -> Dict[str, int]:
        """Count images and label files across splits.

        Returns:
            Dictionary containing counts for train_images, test_images, and
            train_labels.
        """
        train_img_count = 0
        test_img_count = 0
        train_lbl_count = 0

        if self._train_image_dir and self._train_image_dir.exists():
            train_img_count = sum(
                1
                for f in self._train_image_dir.iterdir()
                if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
            )

        if self._test_image_dir and self._test_image_dir.exists():
            test_img_count = sum(
                1
                for f in self._test_image_dir.iterdir()
                if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
            )

        if self._train_label_dir and self._train_label_dir.exists():
            train_lbl_count = sum(
                1
                for f in self._train_label_dir.iterdir()
                if f.is_file() and f.suffix.lower() in LABEL_EXTENSIONS
            )

        logger.info(
            f"Counts - Train Images: {train_img_count}, "
            f"Test Images: {test_img_count}, "
            f"Train Labels: {train_lbl_count}"
        )

        return {
            "num_train_images": train_img_count,
            "num_test_images": test_img_count,
            "num_train_labels": train_lbl_count,
        }

    def detect_missing_pairs(self) -> Tuple[List[str], List[str]]:
        """Detect missing image-label pairs in training dataset split.

        Returns:
            Tuple of (missing_labels, missing_images):
            - missing_labels: Image stems that do not have matching labels.
            - missing_images: Label stems that do not have matching images.
        """
        img_stems: Set[str] = set()
        lbl_stems: Set[str] = set()

        if self._train_image_dir and self._train_image_dir.exists():
            img_stems = {
                f.stem
                for f in self._train_image_dir.iterdir()
                if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
            }

        if self._train_label_dir and self._train_label_dir.exists():
            lbl_stems = {
                f.stem
                for f in self._train_label_dir.iterdir()
                if f.is_file() and f.suffix.lower() in LABEL_EXTENSIONS
            }

        missing_labels = sorted(list(img_stems - lbl_stems))
        missing_images = sorted(list(lbl_stems - img_stems))

        if missing_labels:
            logger.warning(
                f"Found {len(missing_labels)} training images without labels."
            )
        if missing_images:
            logger.warning(
                f"Found {len(missing_images)} training labels without images."
            )

        return missing_labels, missing_images

    def validate(self) -> Dict[str, Any]:
        """Perform full validation of KITTI dataset structure and paired integrity.

        Returns:
            Structured validation report dictionary.
        """
        logger.info(f"Starting KITTI dataset validation at {self.kitti_dir}")

        struct_valid, errors, warnings = self.validate_structure()
        counts = self.count_files()
        missing_labels, missing_images = self.detect_missing_pairs()

        if missing_labels:
            warnings.append(
                f"{len(missing_labels)} images missing matching label files."
            )
        if missing_images:
            warnings.append(
                f"{len(missing_images)} labels missing matching image files."
            )

        is_valid = struct_valid and len(missing_labels) == 0

        report: Dict[str, Any] = {
            "is_valid": is_valid,
            "kitti_dir": str(self.kitti_dir),
            "structure_valid": struct_valid,
            "train_image_dir": (
                str(self._train_image_dir) if self._train_image_dir else None
            ),
            "train_label_dir": (
                str(self._train_label_dir) if self._train_label_dir else None
            ),
            "test_image_dir": (
                str(self._test_image_dir) if self._test_image_dir else None
            ),
            "num_train_images": counts["num_train_images"],
            "num_test_images": counts["num_test_images"],
            "num_train_labels": counts["num_train_labels"],
            "missing_labels_count": len(missing_labels),
            "missing_images_count": len(missing_images),
            "missing_labels": missing_labels,
            "missing_images": missing_images,
            "errors": errors,
            "warnings": warnings,
        }

        logger.info(
            f"Validation finished. Valid: {is_valid}, Errors: {len(errors)}, "
            f"Warnings: {len(warnings)}"
        )
        return report
