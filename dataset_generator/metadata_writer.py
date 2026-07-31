"""JSON metadata serialization and writing for dataset generation."""

import json
from pathlib import Path
from typing import Any, Dict, Union

from attack_engine.attack_config import AttackConfig
from attack_engine.laser_pattern import LaserPattern


class MetadataWriter:
    """Handles serialization and writing of JSON metadata for dataset generation."""

    def write_sample_metadata(
        self,
        output_path: Union[Path, str],
        sample_id: str,
        pattern: LaserPattern,
        config: AttackConfig,
        execution_metadata: Dict[str, Any],
    ) -> Path:
        """Write JSON metadata for an individual attacked sample.

        Args:
            output_path: Target JSON file path.
            sample_id: Sample identifier string.
            pattern: LaserPattern containing spots.
            config: AttackConfig instance used.
            execution_metadata: Dict of execution stats (timestamp, time_ms, etc.).

        Returns:
            Path to the written JSON file.
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        spots_data = [
            {
                "x": float(spot.x),
                "y": float(spot.y),
                "radius": float(spot.radius),
                "intensity": float(spot.intensity),
                "color": list(spot.color),
            }
            for spot in pattern
        ]

        config_data = {
            "laser_color": list(config.laser_color),
            "intensity": float(config.intensity),
            "alpha": float(config.alpha),
            "blur_radius": float(config.blur_radius),
            "spot_radius": float(config.spot_radius),
            "max_spots": int(config.max_spots),
            "random_seed": config.random_seed,
            "pattern_type": str(config.pattern_type),
            "output_dtype": str(config.output_dtype),
        }

        payload = {
            "sample_id": sample_id,
            "pattern_type": execution_metadata.get("pattern_type", config.pattern_type),
            "spots": spots_data,
            "seed": config.random_seed,
            "attack_config": config_data,
            "timestamp": execution_metadata.get("timestamp"),
            "processing_time_ms": execution_metadata.get("processing_time_ms"),
        }

        with output_file.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=4)

        return output_file

    def write_dataset_summary(
        self,
        output_path: Union[Path, str],
        summary_data: Dict[str, Any],
    ) -> Path:
        """Write overall dataset generation summary JSON file.

        Args:
            output_path: Target JSON file path.
            summary_data: Dict containing dataset-level summary metrics.

        Returns:
            Path to the written JSON summary file.
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with output_file.open("w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=4)

        return output_file
