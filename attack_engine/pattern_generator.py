"""Dataset-agnostic laser pattern generator."""

import random
from typing import List, Optional

from .attack_config import AttackConfig
from .laser_pattern import LaserPattern, LaserSpot
from .target_selection import TargetRegion


class PatternGenerator:
    """Generates geometric LaserPattern instances without Pillow dependencies."""

    def __init__(
        self,
        image_width: int,
        image_height: int,
        config: AttackConfig,
    ) -> None:
        """Initialize the PatternGenerator.

        Args:
            image_width: Width of the target image canvas in pixels (> 0).
            image_height: Height of the target image canvas in pixels (> 0).
            config: AttackConfig instance containing pattern settings.

        Raises:
            ValueError: If image_width or image_height are non-positive.
            TypeError: If input parameters are of invalid types.
        """
        if isinstance(image_width, bool) or not isinstance(image_width, int) or image_width <= 0:
            raise ValueError(f"image_width must be a positive integer, got {image_width}.")
        if isinstance(image_height, bool) or not isinstance(image_height, int) or image_height <= 0:
            raise ValueError(f"image_height must be a positive integer, got {image_height}.")
        if not isinstance(config, AttackConfig):
            raise TypeError(f"config must be an AttackConfig instance, got {type(config).__name__}.")

        self.width = image_width
        self.height = image_height
        self.config = config
        self._rng = random.Random(config.random_seed)

    def single_spot(
        self,
        x: Optional[float] = None,
        y: Optional[float] = None,
    ) -> LaserPattern:
        """Generate a pattern containing a single laser spot.

        Args:
            x: Optional X coordinate (defaults to center of width).
            y: Optional Y coordinate (defaults to center of height).

        Returns:
            A LaserPattern containing 1 LaserSpot.
        """
        spot_x = float(self.width / 2.0) if x is None else float(x)
        spot_y = float(self.height / 2.0) if y is None else float(y)

        spot = LaserSpot(
            x=spot_x,
            y=spot_y,
            radius=float(self.config.spot_radius),
            intensity=float(self.config.intensity),
            color=self.config.laser_color,
        )
        return LaserPattern([spot])

    def random_spots(
        self,
        num_spots: Optional[int] = None,
    ) -> LaserPattern:
        """Generate a pattern containing uniformly distributed random spots.

        Args:
            num_spots: Optional spot count override (defaults to config.max_spots).

        Returns:
            A LaserPattern containing random LaserSpot instances.

        Raises:
            ValueError: If num_spots is non-positive.
        """
        count = self.config.max_spots if num_spots is None else num_spots
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError(f"num_spots must be a positive integer, got {count}.")

        pattern = LaserPattern()
        for _ in range(count):
            rx = self._rng.uniform(0.0, float(self.width))
            ry = self._rng.uniform(0.0, float(self.height))
            spot = LaserSpot(
                x=rx,
                y=ry,
                radius=float(self.config.spot_radius),
                intensity=float(self.config.intensity),
                color=self.config.laser_color,
            )
            pattern.add_spot(spot)
        return pattern

    def horizontal_line(
        self,
        y: Optional[float] = None,
        num_spots: Optional[int] = None,
        spacing: Optional[float] = None,
    ) -> LaserPattern:
        """Generate a line of laser spots arranged horizontally.

        Args:
            y: Optional Y coordinate for the line (defaults to center).
            num_spots: Optional spot count override (defaults to config.max_spots).
            spacing: Optional spacing between spot centers in pixels.

        Returns:
            A LaserPattern containing horizontally arranged spots.

        Raises:
            ValueError: If num_spots or spacing are invalid.
        """
        line_y = float(self.height / 2.0) if y is None else float(y)
        count = self.config.max_spots if num_spots is None else num_spots
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError(f"num_spots must be a positive integer, got {count}.")

        pattern = LaserPattern()
        if count == 1:
            x_coords = [float(self.width / 2.0)]
        elif spacing is not None:
            if isinstance(spacing, bool) or not isinstance(spacing, (int, float)) or float(spacing) <= 0:
                raise ValueError(f"spacing must be a positive number, got {spacing}.")
            start_x = (float(self.width) - (count - 1) * float(spacing)) / 2.0
            x_coords = [start_x + i * float(spacing) for i in range(count)]
        else:
            step = float(self.width) / float(count + 1)
            x_coords = [step * (i + 1) for i in range(count)]

        for x in x_coords:
            spot = LaserSpot(
                x=x,
                y=line_y,
                radius=float(self.config.spot_radius),
                intensity=float(self.config.intensity),
                color=self.config.laser_color,
            )
            pattern.add_spot(spot)
        return pattern

    def vertical_line(
        self,
        x: Optional[float] = None,
        num_spots: Optional[int] = None,
        spacing: Optional[float] = None,
    ) -> LaserPattern:
        """Generate a line of laser spots arranged vertically.

        Args:
            x: Optional X coordinate for the line (defaults to center).
            num_spots: Optional spot count override (defaults to config.max_spots).
            spacing: Optional spacing between spot centers in pixels.

        Returns:
            A LaserPattern containing vertically arranged spots.

        Raises:
            ValueError: If num_spots or spacing are invalid.
        """
        line_x = float(self.width / 2.0) if x is None else float(x)
        count = self.config.max_spots if num_spots is None else num_spots
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError(f"num_spots must be a positive integer, got {count}.")

        pattern = LaserPattern()
        if count == 1:
            y_coords = [float(self.height / 2.0)]
        elif spacing is not None:
            if isinstance(spacing, bool) or not isinstance(spacing, (int, float)) or float(spacing) <= 0:
                raise ValueError(f"spacing must be a positive number, got {spacing}.")
            start_y = (float(self.height) - (count - 1) * float(spacing)) / 2.0
            y_coords = [start_y + i * float(spacing) for i in range(count)]
        else:
            step = float(self.height) / float(count + 1)
            y_coords = [step * (i + 1) for i in range(count)]

        for y in y_coords:
            spot = LaserSpot(
                x=line_x,
                y=y,
                radius=float(self.config.spot_radius),
                intensity=float(self.config.intensity),
                color=self.config.laser_color,
            )
            pattern.add_spot(spot)
        return pattern

    def grid(
        self,
        rows: int = 3,
        cols: int = 3,
    ) -> LaserPattern:
        """Generate a grid layout of laser spots.

        Args:
            rows: Number of rows in grid (> 0).
            cols: Number of columns in grid (> 0).

        Returns:
            A LaserPattern containing grid-arranged spots.

        Raises:
            ValueError: If rows or cols are non-positive.
        """
        if isinstance(rows, bool) or not isinstance(rows, int) or rows <= 0:
            raise ValueError(f"rows must be a positive integer, got {rows}.")
        if isinstance(cols, bool) or not isinstance(cols, int) or cols <= 0:
            raise ValueError(f"cols must be a positive integer, got {cols}.")

        pattern = LaserPattern()
        x_step = float(self.width) / float(cols + 1)
        y_step = float(self.height) / float(rows + 1)

        for r in range(rows):
            cy = y_step * (r + 1)
            for c in range(cols):
                cx = x_step * (c + 1)
                spot = LaserSpot(
                    x=cx,
                    y=cy,
                    radius=float(self.config.spot_radius),
                    intensity=float(self.config.intensity),
                    color=self.config.laser_color,
                )
                pattern.add_spot(spot)
        return pattern

    def targeted_spots(
        self,
        target: TargetRegion,
        num_spots: Optional[int] = None,
    ) -> LaserPattern:
        """Generate a pattern whose spot centers are confined to a target bbox.

        Spot centers are sampled uniformly inside the target bounding box,
        intersected with the image canvas. When the bbox is large enough, the
        valid center region is shrunk by ``spot_radius`` so each laser disc
        stays inside the bbox; otherwise centers fall back to the full
        bbox/canvas intersection.

        Args:
            target: TargetRegion describing the object bounding box to attack.
            num_spots: Optional spot count override (defaults to config.max_spots).

        Returns:
            A LaserPattern containing spots inside the target bounding box.

        Raises:
            TypeError: If target is not a TargetRegion.
            ValueError: If num_spots is non-positive or the bbox does not
                intersect the image canvas.
        """
        if not isinstance(target, TargetRegion):
            raise TypeError(
                f"target must be a TargetRegion instance, got {type(target).__name__}."
            )

        count = self.config.max_spots if num_spots is None else num_spots
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError(f"num_spots must be a positive integer, got {count}.")

        x1, y1, x2, y2 = target.bbox
        radius = float(self.config.spot_radius)

        # Intersect the bbox with the image canvas.
        cx1 = max(x1, 0.0)
        cy1 = max(y1, 0.0)
        cx2 = min(x2, float(self.width))
        cy2 = min(y2, float(self.height))
        if cx1 >= cx2 or cy1 >= cy2:
            raise ValueError("target bbox does not intersect the image area.")

        # Clamp the spot-center region so the disc stays within the bbox when possible.
        sx1 = cx1 + radius
        sy1 = cy1 + radius
        sx2 = cx2 - radius
        sy2 = cy2 - radius
        if sx1 >= sx2 or sy1 >= sy2:
            sx1, sy1, sx2, sy2 = cx1, cy1, cx2, cy2

        pattern = LaserPattern()
        for _ in range(count):
            px = self._rng.uniform(sx1, sx2)
            py = self._rng.uniform(sy1, sy2)
            spot = LaserSpot(
                x=px,
                y=py,
                radius=radius,
                intensity=float(self.config.intensity),
                color=self.config.laser_color,
            )
            pattern.add_spot(spot)
        return pattern

    def custom(
        self,
        spots: List[LaserSpot],
    ) -> LaserPattern:
        """Generate a LaserPattern from a custom list of LaserSpot objects.

        Args:
            spots: List of LaserSpot instances.

        Returns:
            A new LaserPattern.

        Raises:
            TypeError: If spots is not a list or contains invalid objects.
        """
        if not isinstance(spots, list):
            raise TypeError("spots must be a list of LaserSpot instances.")
        pattern = LaserPattern()
        for spot in spots:
            if not isinstance(spot, LaserSpot):
                raise TypeError(f"All items in spots must be LaserSpot, got {type(spot).__name__}.")
            pattern.add_spot(spot)
        return pattern

    def generate(self, pattern_type: Optional[str] = None, target: Optional[TargetRegion] = None) -> LaserPattern:
        """Dispatch pattern generation according to pattern_type string.

        Args:
            pattern_type: Identifier string for pattern method.
            target: TargetRegion required by 'targeted'/'targeted_spots' patterns.

        Returns:
            LaserPattern generated by requested method.

        Raises:
            ValueError: If pattern_type is unrecognized or a required target is missing.
        """
        ptype = (pattern_type or self.config.pattern_type).strip().lower()
        if ptype in ("single", "single_spot"):
            return self.single_spot()
        elif ptype in ("random", "random_spots"):
            return self.random_spots()
        elif ptype in ("horizontal_line", "horizontal"):
            return self.horizontal_line()
        elif ptype in ("vertical_line", "vertical"):
            return self.vertical_line()
        elif ptype in ("grid", "grid_pattern"):
            return self.grid()
        elif ptype in ("targeted", "targeted_spots"):
            if target is None:
                raise ValueError(
                    "pattern_type 'targeted' requires a TargetRegion target argument."
                )
            return self.targeted_spots(target)
        elif ptype == "custom":
            return LaserPattern()
        else:
            raise ValueError(
                f"Unsupported pattern_type '{ptype}'. "
                "Supported types: 'single', 'random', 'horizontal_line', 'vertical_line', "
                "'grid', 'targeted', 'targeted_spots', 'custom'."
            )
