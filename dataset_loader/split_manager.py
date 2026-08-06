"""Split manager for KITTI dataset train/val/test splits."""

import logging
import random
from pathlib import Path
from typing import List, Optional, Tuple

from config.paths import KITTI_DIR

logger = logging.getLogger(__name__)


class SplitManager:
    """Manages dataset sample ID splits for training, validation, and testing."""

    def __init__(
        self,
        kitti_dir: Optional[Path] = None,
        splits_dir: Optional[Path] = None,
    ) -> None:
        """Initialize SplitManager.

        Args:
            kitti_dir: Root directory of KITTI dataset. Defaults to KITTI_DIR.
            splits_dir: Directory containing split text files (train.txt, etc.).
        """
        self.kitti_dir = Path(kitti_dir) if kitti_dir else KITTI_DIR
        if splits_dir:
            self.splits_dir = Path(splits_dir)
        else:
            img_sets = self.kitti_dir / "ImageSets"
            splits = self.kitti_dir / "splits"
            if img_sets.exists():
                self.splits_dir = img_sets
            elif splits.exists():
                self.splits_dir = splits
            else:
                self.splits_dir = img_sets

    def load_split(self, split_name: str) -> List[str]:
        """Load list of sample IDs from a split file.

        Args:
            split_name: Split identifier (e.g. 'train', 'val', 'test').

        Returns:
            List of sample ID strings.

        Raises:
            FileNotFoundError: If the split file does not exist.
        """
        if split_name == "trainval":
            train_ids = self.load_split("train")
            val_ids = self.load_split("val")
            sample_ids = sorted(set(train_ids).union(val_ids))
            logger.info(
                "Loaded %d unique sample IDs for combined split 'trainval'",
                len(sample_ids),
            )
            return sample_ids
        file_name = (
            f"{split_name}.txt"
            if not split_name.endswith(".txt")
            else split_name
        )
        split_path = self.splits_dir / file_name

        if not split_path.exists():
            msg = f"Split file not found: {split_path}"
            logger.error(msg)
            raise FileNotFoundError(msg)

        sample_ids: List[str] = []
        with open(split_path, "r", encoding="utf-8") as f:
            for line in f:
                sid = line.strip()
                if sid:
                    sample_ids.append(sid)

        logger.info(
            f"Loaded {len(sample_ids)} sample IDs for split '{split_name}' "
            f"from {split_path}"
        )
        return sample_ids

    def create_random_split(
        self,
        sample_ids: List[str],
        train_ratio: float = 0.8,
        seed: Optional[int] = 42,
    ) -> Tuple[List[str], List[str]]:
        """Split a list of sample IDs into training and validation sets.

        Args:
            sample_ids: List of sample IDs to partition.
            train_ratio: Fraction assigned to training set (0.0 to 1.0).
            seed: Random seed for reproducible splitting.

        Returns:
            Tuple of (train_ids, val_ids).

        Raises:
            ValueError: If train_ratio is outside [0.0, 1.0].
        """
        if not 0.0 <= train_ratio <= 1.0:
            raise ValueError(
                f"train_ratio must be between 0.0 and 1.0, got {train_ratio}"
            )

        unique_ids = sorted(list(set(sample_ids)))
        if seed is not None:
            rng = random.Random(seed)
            shuffled = unique_ids.copy()
            rng.shuffle(shuffled)
        else:
            shuffled = unique_ids.copy()
            random.shuffle(shuffled)

        train_count = int(len(shuffled) * train_ratio)
        train_ids = sorted(shuffled[:train_count])
        val_ids = sorted(shuffled[train_count:])

        logger.info(
            f"Created split (train_ratio={train_ratio}): "
            f"{len(train_ids)} train, {len(val_ids)} val samples"
        )
        return train_ids, val_ids

    def save_split(self, split_name: str, sample_ids: List[str]) -> Path:
        """Save a list of sample IDs to a split text file.

        Args:
            split_name: Name of split (e.g., 'train', 'val').
            sample_ids: List of sample ID strings to write.

        Returns:
            Path to the saved split text file.
        """
        self.splits_dir.mkdir(parents=True, exist_ok=True)
        if split_name == "trainval":
            train_ids = self.load_split("train")
            val_ids = self.load_split("val")
            sample_ids = sorted(set(train_ids).union(val_ids))
            logger.info(
                "Loaded %d unique sample IDs for combined split 'trainval'",
                len(sample_ids),
            )
            return sample_ids
        file_name = (
            f"{split_name}.txt"
            if not split_name.endswith(".txt")
            else split_name
        )
        split_path = self.splits_dir / file_name

        with open(split_path, "w", encoding="utf-8") as f:
            for sid in sample_ids:
                f.write(f"{sid}\n")

        logger.info(f"Saved {len(sample_ids)} sample IDs to {split_path}")
        return split_path


