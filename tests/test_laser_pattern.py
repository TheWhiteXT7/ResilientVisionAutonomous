"""Unit tests for LaserSpot and LaserPattern data structures."""

import unittest
from attack_engine.laser_pattern import LaserPattern, LaserSpot


class TestLaserPattern(unittest.TestCase):
    """Test suite for LaserSpot and LaserPattern data containers."""

    def test_laser_spot_valid_initialization(self) -> None:
        """Test valid LaserSpot creation and attribute access."""
        spot = LaserSpot(x=10.5, y=20.0, radius=12.0, intensity=0.9, color=(255, 0, 0))
        self.assertEqual(spot.x, 10.5)
        self.assertEqual(spot.y, 20.0)
        self.assertEqual(spot.radius, 12.0)
        self.assertEqual(spot.intensity, 0.9)
        self.assertEqual(spot.color, (255, 0, 0))

    def test_laser_spot_validation(self) -> None:
        """Test LaserSpot attribute validations and edge cases."""
        # Invalid coordinates
        with self.assertRaises(TypeError):
            LaserSpot(x="10", y=20, radius=5, intensity=0.5, color=(0, 0, 0))  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            LaserSpot(x=10, y=True, radius=5, intensity=0.5, color=(0, 0, 0))  # type: ignore[arg-type]

        # Invalid radius
        with self.assertRaises(ValueError):
            LaserSpot(x=10, y=20, radius=0.0, intensity=0.5, color=(0, 0, 0))
        with self.assertRaises(ValueError):
            LaserSpot(x=10, y=20, radius=-2.0, intensity=0.5, color=(0, 0, 0))

        # Invalid intensity
        with self.assertRaises(ValueError):
            LaserSpot(x=10, y=20, radius=5, intensity=1.5, color=(0, 0, 0))
        with self.assertRaises(ValueError):
            LaserSpot(x=10, y=20, radius=5, intensity=-0.1, color=(0, 0, 0))

        # Invalid color
        with self.assertRaises(ValueError):
            LaserSpot(x=10, y=20, radius=5, intensity=0.5, color=(256, 0, 0))

    def test_laser_pattern_add_and_len(self) -> None:
        """Test adding spots and checking length of LaserPattern."""
        pattern = LaserPattern()
        self.assertEqual(len(pattern), 0)

        spot1 = LaserSpot(x=5, y=5, radius=10, intensity=1.0, color=(255, 0, 0))
        spot2 = LaserSpot(x=15, y=15, radius=10, intensity=0.8, color=(0, 255, 0))

        pattern.add_spot(spot1)
        pattern.add_spot(spot2)

        self.assertEqual(len(pattern), 2)
        with self.assertRaises(TypeError):
            pattern.add_spot("not_a_spot")  # type: ignore[arg-type]

    def test_laser_pattern_remove_by_index_and_spot(self) -> None:
        """Test removing spots by integer index and by LaserSpot instance."""
        spot1 = LaserSpot(x=5, y=5, radius=10, intensity=1.0, color=(255, 0, 0))
        spot2 = LaserSpot(x=15, y=15, radius=10, intensity=0.8, color=(0, 255, 0))
        spot3 = LaserSpot(x=25, y=25, radius=10, intensity=0.6, color=(0, 0, 255))

        pattern = LaserPattern([spot1, spot2, spot3])
        self.assertEqual(len(pattern), 3)

        # Remove by index
        pattern.remove_spot(0)
        self.assertEqual(len(pattern), 2)
        self.assertEqual(pattern[0], spot2)

        # Remove by instance reference
        pattern.remove_spot(spot3)
        self.assertEqual(len(pattern), 1)

        # Index out of bounds
        with self.assertRaises(IndexError):
            pattern.remove_spot(99)

        # Instance not in pattern
        with self.assertRaises(ValueError):
            pattern.remove_spot(spot1)

        # Invalid target type
        with self.assertRaises(TypeError):
            pattern.remove_spot(True)  # type: ignore[arg-type]

    def test_laser_pattern_clear(self) -> None:
        """Test clearing all spots from LaserPattern."""
        spot1 = LaserSpot(x=5, y=5, radius=10, intensity=1.0, color=(255, 0, 0))
        pattern = LaserPattern([spot1])
        self.assertEqual(len(pattern), 1)

        pattern.clear()
        self.assertEqual(len(pattern), 0)

    def test_laser_pattern_iteration_and_indexing(self) -> None:
        """Test iterating and slicing over LaserPattern."""
        spot1 = LaserSpot(x=5, y=5, radius=10, intensity=1.0, color=(255, 0, 0))
        spot2 = LaserSpot(x=15, y=15, radius=10, intensity=0.8, color=(0, 255, 0))
        pattern = LaserPattern([spot1, spot2])

        spots_list = list(pattern)
        self.assertEqual(spots_list, [spot1, spot2])

        self.assertEqual(pattern[0], spot1)
        self.assertEqual(pattern[1], spot2)
        self.assertEqual(pattern[0:2], [spot1, spot2])

    def test_pattern_has_no_draw_attributes(self) -> None:
        """Verify LaserPattern contains no rendering methods or PIL imports."""
        pattern = LaserPattern()
        self.assertFalse(hasattr(pattern, "draw"))
        self.assertFalse(hasattr(pattern, "render"))


if __name__ == "__main__":
    unittest.main()
