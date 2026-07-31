"""Output directory manager for saving generated dataset assets."""

import shutil
from pathlib import Path
from typing import Any, Dict, Optional, Union
from PIL import Image

from attack_engine.attack_config import AttackConfig
from attack_engine.laser_pattern import LaserPattern
from .generator_config import GeneratorConfig
from .metadata_writer import MetadataWriter


class OutputManager:
    """Manages output directory structure, saving images, annotations, calib, and metadata."""

    def __init__(
        self,
        config: GeneratorConfig,
        metadata_writer: Optional[MetadataWriter] = None,
    ) -> None:
        """Initialize OutputManager.

        Args:
            config: GeneratorConfig instance.
            metadata_writer: Optional MetadataWriter instance.

        Raises:
            TypeError: If config is not a GeneratorConfig instance.
        """
        if not isinstance(config, GeneratorConfig):
            raise TypeError(f"config must be a GeneratorConfig instance, got {type(config).__name__}.")

        self.config = config
        self.output_dir = Path(config.output_directory)
        self.metadata_writer = metadata_writer or MetadataWriter()

    def setup_structure(self, split: str = "training") -> Dict[str, Path]:
        """Create output directory structure for a given dataset split.

        Args:
            split: Split folder name (e.g. 'training', 'testing').

        Returns:
            Dict mapping folder labels to Path objects.
        """
        split_dir = self.output_dir / split
        paths = {
            "image_2": split_dir / "image_2",
            "label_2": split_dir / "label_2",
            "calib": split_dir / "calib",
            "metadata": split_dir / "metadata",
        }

        if self.config.save_original_copy:
            paths["image_2_orig"] = split_dir / "image_2_orig"

        for p in paths.values():
            p.mkdir(parents=True, exist_ok=True)

        return paths

    def get_image_path(self, sample_id: str, split: str = "training") -> Path:
        """Get target output path for an attacked image."""
        ext = self.config.image_format.lower()
        return self.output_dir / split / "image_2" / f"{sample_id}.{ext}"

    def get_label_path(self, sample_id: str, split: str = "training") -> Path:
        """Get target output path for a label file."""
        return self.output_dir / split / "label_2" / f"{sample_id}.txt"

    def get_calib_path(self, sample_id: str, split: str = "training") -> Path:
        """Get target output path for a calibration file."""
        return self.output_dir / split / "calib" / f"{sample_id}.txt"

    def get_metadata_path(self, sample_id: str, split: str = "training") -> Path:
        """Get target output path for a metadata file."""
        return self.output_dir / split / "metadata" / f"{sample_id}.json"

    def is_sample_processed(self, sample_id: str, split: str = "training") -> bool:
        """Check if all expected output files already exist for sample_id."""
        image_path = self.get_image_path(sample_id, split)
        if not image_path.exists():
            return False
        if self.config.save_metadata:
            meta_path = self.get_metadata_path(sample_id, split)
            if not meta_path.exists():
                return False
        return True

    def save_attacked_image(
        self,
        image: Image.Image,
        sample_id: str,
        split: str = "training",
    ) -> Path:
        """Save attacked PIL image to output image_2 directory.

        Args:
            image: Attacked PIL Image.
            sample_id: Sample identifier string.
            split: Target split directory.

        Returns:
            Path to saved image.
        """
        dest_path = self.get_image_path(sample_id, split)
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        if dest_path.exists() and not self.config.overwrite_existing:
            return dest_path

        fmt = "JPEG" if self.config.image_format.lower() in ("jpg", "jpeg") else "PNG"
        image.save(dest_path, format=fmt)
        return dest_path

    def copy_label(
        self,
        label_path: Optional[Union[Path, str]],
        sample_id: str,
        split: str = "training",
    ) -> Optional[Path]:
        """Copy ground-truth label text file to output label_2 directory.

        Args:
            label_path: Source label text file path.
            sample_id: Sample identifier string.
            split: Target split directory.

        Returns:
            Path to copied label file or None.
        """
        if not self.config.copy_labels or label_path is None:
            return None

        source_file = Path(label_path)
        if not source_file.exists():
            return None

        dest_path = self.get_label_path(sample_id, split)
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        if dest_path.exists() and not self.config.overwrite_existing:
            return dest_path

        shutil.copy2(source_file, dest_path)
        return dest_path

    def copy_calib(
        self,
        calib_path: Optional[Union[Path, str]],
        sample_id: str,
        split: str = "training",
    ) -> Optional[Path]:
        """Copy calibration text file to output calib directory.

        Args:
            calib_path: Source calibration file path.
            sample_id: Sample identifier string.
            split: Target split directory.

        Returns:
            Path to copied calib file or None.
        """
        if not self.config.copy_calibration or calib_path is None:
            return None

        source_file = Path(calib_path)
        if not source_file.exists():
            return None

        dest_path = self.get_calib_path(sample_id, split)
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        if dest_path.exists() and not self.config.overwrite_existing:
            return dest_path

        shutil.copy2(source_file, dest_path)
        return dest_path

    def save_original_copy(
        self,
        original_path: Optional[Union[Path, str]],
        sample_id: str,
        split: str = "training",
    ) -> Optional[Path]:
        """Copy original image to image_2_orig directory if enabled in config."""
        if not self.config.save_original_copy or original_path is None:
            return None

        source_file = Path(original_path)
        if not source_file.exists():
            return None

        orig_dir = self.output_dir / split / "image_2_orig"
        orig_dir.mkdir(parents=True, exist_ok=True)
        dest_path = orig_dir / f"{sample_id}{source_file.suffix}"

        if dest_path.exists() and not self.config.overwrite_existing:
            return dest_path

        shutil.copy2(source_file, dest_path)
        return dest_path

    def save_metadata(
        self,
        sample_id: str,
        pattern: LaserPattern,
        attack_config: AttackConfig,
        execution_metadata: Dict[str, Any],
        split: str = "training",
    ) -> Optional[Path]:
        """Save sample JSON metadata if save_metadata is enabled."""
        if not self.config.save_metadata:
            return None

        dest_path = self.get_metadata_path(sample_id, split)
        return self.metadata_writer.write_sample_metadata(
            output_path=dest_path,
            sample_id=sample_id,
            pattern=pattern,
            config=attack_config,
            execution_metadata=execution_metadata,
        )
