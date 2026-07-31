"""Unit tests for ProgressTracker execution monitoring."""

import unittest
from dataset_generator.progress_tracker import ProgressTracker


class TestProgressTracker(unittest.TestCase):
    """Test suite for ProgressTracker metrics and reporting."""

    def test_initial_state(self) -> None:
        """Test initial default values."""
        tracker = ProgressTracker(total_samples=10)
        stats = tracker.get_stats()

        self.assertEqual(stats["total_samples"], 10)
        self.assertEqual(stats["processed_samples"], 0)
        self.assertEqual(stats["successful_samples"], 0)
        self.assertEqual(stats["failed_samples"], 0)
        self.assertEqual(stats["percentage"], 0.0)
        self.assertEqual(stats["failures"], [])

    def test_update_success_and_failure(self) -> None:
        """Test recording successful and failed sample executions."""
        tracker = ProgressTracker(total_samples=5)
        tracker.start()

        tracker.update("000000", success=True)
        tracker.update("000001", success=False, error="File corrupted")
        tracker.update("000002", success=True)

        stats = tracker.get_stats()
        self.assertEqual(stats["processed_samples"], 3)
        self.assertEqual(stats["successful_samples"], 2)
        self.assertEqual(stats["failed_samples"], 1)
        self.assertEqual(stats["percentage"], 60.0)
        self.assertEqual(len(stats["failures"]), 1)
        self.assertEqual(stats["failures"][0]["sample_id"], "000001")

    def test_generate_report(self) -> None:
        """Test summary report generation."""
        tracker = ProgressTracker(total_samples=2)
        tracker.start()
        tracker.update("000000", success=True)
        tracker.update("000001", success=True)
        tracker.finish()

        report = tracker.generate_report()
        self.assertEqual(report["status"], "completed")
        self.assertEqual(report["processed_samples"], 2)
        self.assertEqual(report["percentage"], 100.0)


if __name__ == "__main__":
    unittest.main()
