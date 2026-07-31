"""Data representation of laser spots and pattern layouts."""

from dataclasses import dataclass
from typing import Iterator, List, Optional, Tuple, Union, overload


@dataclass(frozen=True)
class LaserSpot:
    """Immutable representation of a single laser spot.

    Attributes:
        x: X-coordinate of the spot center in pixels.
        y: Y-coordinate of the spot center in pixels.
        radius: Spot radius in pixels (> 0.0).
        intensity: Spot intensity factor in range [0.0, 1.0].
        color: RGB color tuple (0-255 per channel).
    """

    x: float
    y: float
    radius: float
    intensity: float
    color: Tuple[int, int, int]

    def __post_init__(self) -> None:
        """Validate spot attributes upon initialization.

        Raises:
            TypeError: If an attribute has an incorrect type.
            ValueError: If an attribute has an out-of-range value.
        """
        if isinstance(self.x, bool) or not isinstance(self.x, (int, float)):
            raise TypeError("x must be a float or int.")
        if isinstance(self.y, bool) or not isinstance(self.y, (int, float)):
            raise TypeError("y must be a float or int.")

        if isinstance(self.radius, bool) or not isinstance(self.radius, (int, float)):
            raise TypeError("radius must be a float or int.")
        if float(self.radius) <= 0.0:
            raise ValueError(f"radius must be greater than 0, got {self.radius}.")

        if isinstance(self.intensity, bool) or not isinstance(self.intensity, (int, float)):
            raise TypeError("intensity must be a float or int.")
        if not (0.0 <= float(self.intensity) <= 1.0):
            raise ValueError(f"intensity must be between 0.0 and 1.0, got {self.intensity}.")

        if not isinstance(self.color, (tuple, list)) or len(self.color) != 3:
            raise TypeError("color must be a tuple or list of 3 integers (RGB).")
        for idx, channel in enumerate(self.color):
            if isinstance(channel, bool) or not isinstance(channel, int):
                raise TypeError(
                    f"color channel {idx} must be an integer, got {type(channel).__name__}."
                )
            if not (0 <= channel <= 255):
                raise ValueError(
                    f"color channel {idx} must be in range [0, 255], got {channel}."
                )

        if isinstance(self.color, list):
            object.__setattr__(self, "color", tuple(self.color))


class LaserPattern:
    """Container data structure for a collection of LaserSpot instances.

    LaserPattern represents purely geometric/structural data and performs no
    rendering or image manipulation.
    """

    def __init__(self, spots: Optional[List[LaserSpot]] = None) -> None:
        """Initialize a LaserPattern with optional initial spots.

        Args:
            spots: Optional list of LaserSpot objects.
        """
        self._spots: List[LaserSpot] = []
        if spots is not None:
            for spot in spots:
                self.add_spot(spot)

    @property
    def spots(self) -> List[LaserSpot]:
        """Return a shallow copy of the list of spots."""
        return list(self._spots)

    def add_spot(self, spot: LaserSpot) -> None:
        """Add a LaserSpot to the pattern.

        Args:
            spot: LaserSpot to append.

        Raises:
            TypeError: If spot is not an instance of LaserSpot.
        """
        if not isinstance(spot, LaserSpot):
            raise TypeError(f"Expected LaserSpot instance, got {type(spot).__name__}.")
        self._spots.append(spot)

    def remove_spot(self, target: Union[int, LaserSpot]) -> None:
        """Remove a spot by integer index or by LaserSpot instance reference.

        Args:
            target: Index of spot to remove (int) or LaserSpot instance to remove.

        Raises:
            TypeError: If target is neither int nor LaserSpot.
            IndexError: If int index is out of bounds.
            ValueError: If LaserSpot instance is not found.
        """
        if isinstance(target, bool):
            raise TypeError("target must be an integer index or LaserSpot instance, not bool.")
        if isinstance(target, int):
            if target < -len(self._spots) or target >= len(self._spots):
                raise IndexError(
                    f"Index {target} out of range for LaserPattern of size {len(self._spots)}."
                )
            self._spots.pop(target)
        elif isinstance(target, LaserSpot):
            self._spots.remove(target)
        else:
            raise TypeError(
                f"target must be an integer index or LaserSpot instance, got {type(target).__name__}."
            )

    def clear(self) -> None:
        """Remove all spots from the pattern."""
        self._spots.clear()

    def __len__(self) -> int:
        """Return the number of spots in the pattern."""
        return len(self._spots)

    def __iter__(self) -> Iterator[LaserSpot]:
        """Return an iterator over the pattern spots."""
        return iter(self._spots)

    @overload
    def __getitem__(self, index: int) -> LaserSpot: ...

    @overload
    def __getitem__(self, index: slice) -> List[LaserSpot]: ...

    def __getitem__(self, index: Union[int, slice]) -> Union[LaserSpot, List[LaserSpot]]:
        """Retrieve spot(s) by index or slice.

        Args:
            index: Integer index or slice.

        Returns:
            LaserSpot or list of LaserSpot objects.
        """
        return self._spots[index]

    def __repr__(self) -> str:
        """Return string representation of LaserPattern."""
        return f"LaserPattern(spots_count={len(self._spots)})"
