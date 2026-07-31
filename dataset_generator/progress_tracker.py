"""Progress tracking and execution stats monitoring for dataset generation."""

import logging
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ProgressTracker:
    """Tracks sample generation progress, processing rate, ETA, and failures."""

    def __init__(self, total_samples: int = 0) -> None:
        """Initialize ProgressTracker.

        Args:
            total_samples: Total number of samples expected to be processed.
        """
        self.total_samples = max(0, total_samples)
        self.processed_samples = 0
        self.successful_samples = 0
        self.failed_samples = 0
        self.failures: List[Dict[str, str]] = []
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None

    def start(self, total_samples: Optional[int] = None) -> None:
        """Start tracking progress timer.

        Args:
            total_samples: Optional update to total sample count.
        """
        if total_samples is not None:
            self.total_samples = max(0, total_samples)
        self.processed_samples = 0
        self.successful_samples = 0
        self.failed_samples = 0
        self.failures.clear()
        self.start_time = time.perf_counter()
        self.end_time = None
        logger.info(f"ProgressTracker started for {self.total_samples} samples.")

    def update(
        self,
        sample_id: str,
        success: bool = True,
        error: Optional[str] = None,
    ) -> None:
        """Record the processing outcome for a single sample.

        Args:
            sample_id: Identifier string of processed sample.
            success: True if sample processed successfully.
            error: Optional error message string if failed.
        """
        if self.start_time is None:
            self.start_time = time.perf_counter()

        self.processed_samples += 1
        if success:
            self.successful_samples += 1
        else:
            self.failed_samples += 1
            err_msg = error or "Unknown error"
            self.failures.append({"sample_id": sample_id, "error": err_msg})
            logger.warning(f"Sample '{sample_id}' failed: {err_msg}")

        stats = self.get_stats()
        logger.debug(
            f"Progress: [{stats['processed_samples']}/{stats['total_samples']}] "
            f"({stats['percentage']:.1f}%) - ETA: {stats['eta']:.1f}s"
        )

    def finish(self) -> None:
        """Mark completion of progress tracking."""
        self.end_time = time.perf_counter()

    @property
    def elapsed_time(self) -> float:
        """Return elapsed time in seconds."""
        if self.start_time is None:
            return 0.0
        now = self.end_time if self.end_time is not None else time.perf_counter()
        return max(0.0, now - self.start_time)

    @property
    def percentage(self) -> float:
        """Return completion percentage in range [0.0, 100.0]."""
        if self.total_samples <= 0:
            return 0.0
        return min(100.0, (self.processed_samples / self.total_samples) * 100.0)

    @property
    def eta(self) -> float:
        """Estimate remaining time in seconds."""
        if self.processed_samples <= 0 or self.total_samples <= self.processed_samples:
            return 0.0
        avg_time = self.elapsed_time / self.processed_samples
        remaining = self.total_samples - self.processed_samples
        return max(0.0, avg_time * remaining)

    def get_stats(self) -> Dict[str, Any]:
        """Return a dictionary of current progress metrics."""
        return {
            "total_samples": self.total_samples,
            "processed_samples": self.processed_samples,
            "successful_samples": self.successful_samples,
            "failed_samples": self.failed_samples,
            "percentage": round(self.percentage, 2),
            "elapsed_time": round(self.elapsed_time, 2),
            "eta": round(self.eta, 2),
            "failures": list(self.failures),
        }

    def generate_report(self) -> Dict[str, Any]:
        """Generate a summary report dictionary."""
        stats = self.get_stats()
        stats["status"] = "completed" if self.processed_samples >= self.total_samples else "in_progress"
        return stats
