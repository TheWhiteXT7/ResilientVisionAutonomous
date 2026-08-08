"""Object-aware target selection for laser pattern attacks.

Provides the immutable TargetRegion data model and deterministic target
selection logic used to pick an object bounding box to attack.
"""

import random
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple


class TargetSelectionError(ValueError):
    """Raised when no valid target of the requested class can be selected."""


@dataclass(frozen=True)
class TargetRegion:
    """Immutable description of an image region selected as an attack target.

    Attributes:
        class_name: Target object class (e.g., 'Car').
        bbox: Bounding box in image coordinates (x1, y1, x2, y2) with
            x1 < x2 and y1 < y2.
        metadata: Optional originating object/sample metadata (e.g., the source
            KITTI Annotation the region was derived from).
    """

    class_name: str
    bbox: Tuple[float, float, float, float]
    metadata: Optional[Any] = None

    def __post_init__(self) -> None:
        """Validate the target region attributes.

        Raises:
            TypeError: If class_name or bbox have invalid types.
            ValueError: If class_name is empty or the bbox is degenerate.
        """
        if isinstance(self.class_name, bool) or not isinstance(self.class_name, str):
            raise TypeError(f"class_name must be a string, got {type(self.class_name).__name__}.")
        if not self.class_name.strip():
            raise ValueError("class_name cannot be empty.")

        raw = self.bbox
        if not isinstance(raw, (tuple, list)) or len(raw) != 4:
            raise TypeError("bbox must be a tuple/list of 4 numbers (x1, y1, x2, y2).")
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in raw):
            raise TypeError("bbox values must be numbers (x1, y1, x2, y2).")

        x1, y1, x2, y2 = (float(v) for v in raw)
        if x1 >= x2:
            raise ValueError(f"bbox x1 must be less than x2, got ({x1}, {x2}).")
        if y1 >= y2:
            raise ValueError(f"bbox y1 must be less than y2, got ({y1}, {y2}).")
        object.__setattr__(self, "bbox", (x1, y1, x2, y2))


def _is_valid_bbox(bbox: Any, width: int, height: int) -> bool:
    """Return True if bbox is a non-degenerate box fully inside the image."""
    if not isinstance(bbox, (tuple, list)) or len(bbox) != 4:
        return False
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in bbox):
        return False
    x1, y1, x2, y2 = (float(v) for v in bbox)
    if x1 >= x2 or y1 >= y2:
        return False
    return (
        x1 >= 0.0
        and y1 >= 0.0
        and x2 <= float(width)
        and y2 <= float(height)
    )


def select_target(
    annotations: Sequence[Any],
    image_size: Tuple[int, int],
    target_class: str = "Car",
    random_seed: Optional[int] = None,
) -> TargetRegion:
    """Select a deterministic target region of the requested object class.

    Candidate annotations are objects exposing ``class_name`` and ``bbox``
    attributes (e.g. KITTI ``Annotation`` or ``TargetRegion`` instances).
    Annotations whose bbox is degenerate (x1 >= x2 or y1 >= y2) or that extend
    outside the image canvas are excluded from selection, which safely rejects
    invalid/out-of-image boxes instead of crashing.

    Selection is deterministic for a fixed ``random_seed``. If no valid
    annotation of the requested class exists, a ``TargetSelectionError`` is
    raised; during dataset generation this surfaces as a per-sample failure and
    the remaining samples continue to be processed.

    Args:
        annotations: Sequence of candidate objects exposing ``class_name``
            and ``bbox``.
        image_size: (width, height) of the image in pixels.
        target_class: Object class to target (case-insensitive).
        random_seed: Optional seed for reproducible selection.

    Returns:
        A TargetRegion selected deterministically from the valid candidates.

    Raises:
        ValueError: If image_size is invalid or target_class is empty.
        TargetSelectionError: If no valid annotation of the target class exists.
    """
    if isinstance(target_class, bool) or not isinstance(target_class, str):
        raise TypeError(f"target_class must be a string, got {type(target_class).__name__}.")
    if not target_class.strip():
        raise ValueError("target_class cannot be empty.")

    if not isinstance(image_size, (tuple, list)) or len(image_size) != 2:
        raise TypeError(f"image_size must be a (width, height) pair, got {image_size!r}.")
    width, height = image_size
    if isinstance(width, bool) or not isinstance(width, int) or width <= 0:
        raise ValueError(f"image width must be a positive integer, got {width}.")
    if isinstance(height, bool) or not isinstance(height, int) or height <= 0:
        raise ValueError(f"image height must be a positive integer, got {height}.")

    want = target_class.strip().lower()
    candidates: List[TargetRegion] = []
    for annotation in annotations:
        class_name = getattr(annotation, "class_name", None)
        bbox = getattr(annotation, "bbox", None)
        if class_name is None or bbox is None:
            continue
        if str(class_name).strip().lower() != want:
            continue
        if not _is_valid_bbox(bbox, width, height):
            continue
        candidates.append(
            TargetRegion(
                class_name=str(class_name).strip(),
                bbox=bbox,
                metadata=annotation,
            )
        )

    if not candidates:
        raise TargetSelectionError(
            f"No valid '{target_class}' target found among {len(annotations)} annotation(s) "
            f"within image size {(width, height)}."
        )

    if len(candidates) == 1:
        return candidates[0]

    rng = random.Random(random_seed)
    return rng.choice(candidates)
