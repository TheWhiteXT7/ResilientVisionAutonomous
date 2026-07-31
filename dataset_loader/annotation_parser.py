"""KITTI dataset annotation parser module."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


class AnnotationParseError(ValueError):
    """Exception raised when an annotation file or line is malformed."""

    pass


@dataclass
class Annotation:
    """Represents a single KITTI object annotation.

    Attributes:
        class_name: Object class type (e.g., 'Car', 'Pedestrian', 'DontCare').
        truncated: Truncation level from 0.0 (non-truncated) to 1.0 (truncated).
        occluded: Occlusion state (0=fully visible, 1=partly occluded,
            2=largely occluded, 3=unknown, -1=DontCare).
        alpha: Observation angle of the object [-pi..pi].
        bbox: 2D bounding box in image coordinates (left, top, right, bottom).
        dimensions: 3D object dimensions (height, width, length) in meters.
        location: 3D object center location (x, y, z) in camera coordinates.
        rotation_y: Rotation around Y-axis in camera coordinates [-pi..pi].
        score: Optional confidence score for model predictions.
    """

    class_name: str
    truncated: float
    occluded: int
    alpha: float
    bbox: Tuple[float, float, float, float]
    dimensions: Tuple[float, float, float]
    location: Tuple[float, float, float]
    rotation_y: float
    score: Optional[float] = None


class KittiAnnotationParser:
    """Parser for KITTI label text files into Annotation dataclass instances."""

    def parse(self, file_path: Path) -> List[Annotation]:
        """Parse a KITTI label text file into a list of Annotation objects.

        Args:
            file_path: Path to the annotation text file.

        Returns:
            List of Annotation objects parsed from the file.

        Raises:
            FileNotFoundError: If the specified file does not exist.
            AnnotationParseError: If any line in the file is malformed.
        """
        path = Path(file_path)
        if not path.exists():
            msg = f"Annotation file not found: {path}"
            logger.error(msg)
            raise FileNotFoundError(msg)

        annotations: List[Annotation] = []

        with open(path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, start=1):
                stripped_line = line.strip()
                if not stripped_line:
                    continue

                tokens = stripped_line.split()
                if len(tokens) < 15 or len(tokens) > 16:
                    msg = (
                        f"Malformed annotation line {line_idx} in {path}: "
                        f"Expected 15 or 16 tokens, got {len(tokens)}: "
                        f"'{stripped_line}'"
                    )
                    logger.error(msg)
                    raise AnnotationParseError(msg)

                try:
                    class_name = tokens[0]
                    truncated = float(tokens[1])
                    occluded = int(float(tokens[2]))
                    alpha = float(tokens[3])

                    bbox = (
                        float(tokens[4]),
                        float(tokens[5]),
                        float(tokens[6]),
                        float(tokens[7]),
                    )

                    dimensions = (
                        float(tokens[8]),
                        float(tokens[9]),
                        float(tokens[10]),
                    )

                    location = (
                        float(tokens[11]),
                        float(tokens[12]),
                        float(tokens[13]),
                    )

                    rotation_y = float(tokens[14])
                    score = float(tokens[15]) if len(tokens) == 16 else None

                    annotation = Annotation(
                        class_name=class_name,
                        truncated=truncated,
                        occluded=occluded,
                        alpha=alpha,
                        bbox=bbox,
                        dimensions=dimensions,
                        location=location,
                        rotation_y=rotation_y,
                        score=score,
                    )
                    annotations.append(annotation)
                except ValueError as exc:
                    msg = (
                        f"Failed to parse numerical values on line {line_idx} "
                        f"in {path}: '{stripped_line}'. Error: {exc}"
                    )
                    logger.error(msg)
                    raise AnnotationParseError(msg) from exc

        logger.debug(f"Parsed {len(annotations)} annotations from {path}")
        return annotations
