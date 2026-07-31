"""Unit tests for MetadataWriter JSON serialization."""

import json
import tempfile
import unittest
from pathlib import Path

from attack_engine import AttackConfig, LaserPattern, LaserSpot
from dataset_generator.metadata_writer import MetadataWriter


class TestMetadataWriter(unittest.TestCase):
    """Test suite for MetadataWriter JSON metadata writing."""

    def setUp(self) -> None:
        """Set up temporary directory and MetadataWriter instance."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.writer = MetadataWriter()
        self.config = AttackConfig(
            laser_color=(255, 0, 0),
            intensity=1.0,
            alpha=0.8,
            blur_radius=5.0,
            spot_radius=15.0,
            max_spots=2,
            random_seed=42,
        )
        self.pattern = LaserPattern([
            LaserSpot(x=10.0, y=20.0, radius=15.0, intensity=1.0, color=(255, 0, 0)),
            LaserSpot(x=30.0, y=40.0, radius=15.0, intensity=0.8, color=(255, 0, 0)),
        ])
        self.execution_metadata = {
            "pattern_type": "random",
            "timestamp": "2026-07-31T22:50:00+00:00",
            "processing_time_ms": 15.5,
        }

    def tearDown(self) -> None:
        """Clean up temporary directory."""
        self.temp_dir.cleanup()

    def test_write_sample_metadata(self) -> None:
        """Test writing per-sample JSON metadata file."""
        out_path = Path(self.temp_dir.name) / "metadata" / "000000.json"
        result_path = self.writer.write_sample_metadata(
            output_path=out_path,
            sample_id="000000",
            pattern=self.pattern,
            config=self.config,
            execution_metadata=self.execution_metadata,
        )

        self.assertTrue(result_path.exists())
        with result_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertEqual(data["sample_id"], "000000")
        self.assertEqual(data["pattern_type"], "random")
        self.assertEqual(len(data["spots"]), 2)
        self.assertEqual(data["spots"][0]["x"], 10.0)
        self.assertEqual(data["spots"][0]["color"], [255, 0, 0])
        self.assertEqual(data["seed"], 42)
        self.assertEqual(data["attack_config"]["laser_color"], [255, 0, 0])
        self.assertEqual(data["timestamp"], "2026-07-31T22:50:00+00:00")
        self.assertEqual(data["processing_time_ms"], 15.5)

    def test_write_dataset_summary(self) -> None:
        """Test writing overall dataset summary JSON file."""
        out_path = Path(self.temp_dir.name) / "generation_summary.json"
        summary_data = {
            "total_samples": 10,
            "processed_samples": 10,
            "successful_samples": 10,
            "failed_samples": 0,
            "elapsed_time": 1.25,
            "status": "completed",
        }
        result_path = self.writer.write_dataset_summary(out_path, summary_data)

        self.assertTrue(result_path.exists())
        with result_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertEqual(data, summary_data)


if __name__ == "__main__":
    unittest.main()
