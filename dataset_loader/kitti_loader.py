"""KITTI dataset loader and sample representation."""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from PIL import Image

from config.paths import KITTI_DIR
from dataset_loader.annotation_parser import Annotation, KittiAnnotationParser
from dataset_loader.base_loader import BaseDatasetLoader
from dataset_loader.dataset_validator import DatasetValidator
from dataset_loader.split_manager import SplitManager

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}


@dataclass
class KittiSample:
    """Dataclass representing a single KITTI sample.

    Attributes:
        sample_id: Unique string identifier of the sample (e.g., '000000').
        image_path: Path to camera image file.
        label_path: Optional path to ground-truth label text file.
        calib_path: Optional path to calibration text file.
        annotations: List of parsed Annotation objects for this sample.
        image: Optional pre-loaded PIL Image object.
    """

    sample_id: str
    image_path: Path
    label_path: Optional[Path] = None
    calib_path: Optional[Path] = None
    annotations: List[Annotation] = field(default_factory=list)
    image: Optional[Image.Image] = None

    def load_image(self) -> Image.Image:
        """Load and return PIL RGB Image for this sample.

        Returns:
            PIL Image object in RGB mode.

        Raises:
            FileNotFoundError: If the image file does not exist.
        """
        if not self.image_path.exists():
            msg = f"Image file not found: {self.image_path}"
            logger.error(msg)
            raise FileNotFoundError(msg)
        return Image.open(self.image_path).convert("RGB")


class KittiLoader(BaseDatasetLoader[KittiSample]):
    """Dataset loader for KITTI object detection format.

    Supports train/val/test splits, sample lookup by index or ID, lazy image
    loading, and dataset structure validation.
    """

    def __init__(
        self,
        kitti_dir: Optional[Path] = None,
        split: str = "train",
        val_ratio: float = 0.0,
        seed: Optional[int] = 42,
        validate: bool = True,
        load_images: bool = False,
    ) -> None:
        """Initialize KittiLoader.

        Args:
            kitti_dir: Root KITTI directory. Defaults to KITTI_DIR.
            split: Target split name ('train', 'val', 'test', 'trainval').
            val_ratio: Ratio for dynamic validation set creation if no split
                file exists.
            seed: Random seed for dynamic split creation.
            validate: If True, executes DatasetValidator on init.
            load_images: If True, pre-loads PIL Image instances into samples.
        """
        self.kitti_dir = Path(kitti_dir) if kitti_dir else KITTI_DIR
        self.split = split.lower()
        self.load_images = load_images

        self.validator = DatasetValidator(kitti_dir=self.kitti_dir)
        if validate:
            report = self.validator.validate()
            if not report["structure_valid"]:
                logger.warning(
                    f"Dataset structure warnings for {self.kitti_dir}: "
                    f"{report['warnings']}"
                )

        self.annotation_parser = KittiAnnotationParser()
        self.split_manager = SplitManager(kitti_dir=self.kitti_dir)

        self._resolve_directories()
        self._sample_ids = self._load_split_ids(val_ratio=val_ratio, seed=seed)

        logger.info(
            f"Initialized KittiLoader (split='{self.split}') "
            f"with {len(self._sample_ids)} samples."
        )

    def _resolve_directories(self) -> None:
        """Resolve actual directory locations for images, labels, and calib."""
        std_train_img = self.kitti_dir / "training" / "image_2"
        std_train_lbl = self.kitti_dir / "training" / "label_2"
        std_test_img = self.kitti_dir / "testing" / "image_2"
        std_calib = self.kitti_dir / "training" / "calib"

        alt_train_img = (
            self.kitti_dir / "data_object_image_2" / "training" / "image_2"
        )
        alt_train_lbl = (
            self.kitti_dir / "data_object_label_2" / "training" / "label_2"
        )
        alt_test_img = (
            self.kitti_dir / "data_object_image_2" / "testing" / "image_2"
        )

        if self.split in ("test", "testing"):
            if std_test_img.exists():
                self.image_dir = std_test_img
            elif alt_test_img.exists():
                self.image_dir = alt_test_img
            else:
                self.image_dir = std_test_img
        else:
            if std_train_img.exists():
                self.image_dir = std_train_img
            elif alt_train_img.exists():
                self.image_dir = alt_train_img
            else:
                self.image_dir = std_train_img

        if std_train_lbl.exists():
            self.label_dir: Optional[Path] = std_train_lbl
        elif alt_train_lbl.exists():
            self.label_dir = alt_train_lbl
        else:
            self.label_dir = None

        self.calib_dir: Optional[Path] = (
            std_calib if std_calib.exists() else None
        )

    def _load_split_ids(
        self, val_ratio: float, seed: Optional[int]
    ) -> List[str]:
        """Load or discover sample ID strings for current split."""
        try:
            split_ids = self.split_manager.load_split(self.split)
            return sorted(split_ids)
        except FileNotFoundError:
            logger.debug(
                f"No split file found for '{self.split}'. "
                f"Discovering images directly."
            )

        if not self.image_dir.exists():
            return []

        all_stems = sorted(
            [
                f.stem
                for f in self.image_dir.iterdir()
                if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
            ]
        )

        if self.split in ("test", "testing") or val_ratio <= 0.0:
            return all_stems

        train_ids, val_ids = self.split_manager.create_random_split(
            all_stems, train_ratio=1.0 - val_ratio, seed=seed
        )

        if self.split == "val":
            return val_ids
        return train_ids

    @property
    def sample_ids(self) -> List[str]:
        """Return list of sample ID strings in loaded split."""
        return self._sample_ids

    def _find_image_file(self, sample_id: str) -> Path:
        """Find image file corresponding to sample ID."""
        for ext in (".png", ".jpg", ".jpeg", ".bmp"):
            candidate = self.image_dir / f"{sample_id}{ext}"
            if candidate.exists():
                return candidate
        return self.image_dir / f"{sample_id}.png"

    def get_sample_by_id(self, sample_id: str) -> KittiSample:
        """Fetch KittiSample by sample ID string.

        Args:
            sample_id: Sample identifier string (e.g. '000000').

        Returns:
            KittiSample object.

        Raises:
            KeyError: If sample_id is not found in the split or directory.
        """
        image_path = self._find_image_file(sample_id)
        if sample_id not in self._sample_ids and not image_path.exists():
            raise KeyError(
                f"Sample ID '{sample_id}' not found in split or directory."
            )

        label_path: Optional[Path] = None
        annotations: List[Annotation] = []
        if self.label_dir:
            lbl_candidate = self.label_dir / f"{sample_id}.txt"
            if lbl_candidate.exists():
                label_path = lbl_candidate
                annotations = self.annotation_parser.parse(label_path)

        calib_path: Optional[Path] = None
        if self.calib_dir:
            cal_candidate = self.calib_dir / f"{sample_id}.txt"
            if cal_candidate.exists():
                calib_path = cal_candidate

        sample = KittiSample(
            sample_id=sample_id,
            image_path=image_path,
            label_path=label_path,
            calib_path=calib_path,
            annotations=annotations,
        )

        if self.load_images and image_path.exists():
            sample.image = sample.load_image()

        return sample
