"""Create a six-image visual pilot for the rolling_shutter_artifact attack."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
from PIL import Image

from attack_engine import AttackConfig, AttackPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, help="KITTI PNG/JPEG input image")
    parser.add_argument("--output-dir", default="outputs/rolling_shutter_artifact_pilot")
    args = parser.parse_args()
    source = Image.open(args.image).convert("RGB")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    base = dict(
        pattern_type="rolling_shutter_artifact", random_seed=42,
        rolling_shutter_artifact_start=(source.width * 0.42, source.height * 0.34),
        rolling_shutter_artifact_velocity=(6.0, 0.03),
        rolling_shutter_artifact_beam_sigma=11.0,
        rolling_shutter_artifact_row_readout_time=0.025,
        rolling_shutter_artifact_row_exposure_time=0.10,
        rolling_shutter_artifact_temporal_samples=12,
        rolling_shutter_artifact_bloom_sigma=7.0,
        rolling_shutter_artifact_smear_strength=10.0,
        rolling_shutter_artifact_smear_length=96,
    )
    variants = {
        "clean": {"rolling_shutter_artifact_power": 0.0},
        "low": {"rolling_shutter_artifact_power": 0.45},
        "medium": {"rolling_shutter_artifact_power": 1.25},
        "strong": {"rolling_shutter_artifact_power": 3.5, "rolling_shutter_artifact_bloom_strength": 0.7},
        "different_trajectory": {"rolling_shutter_artifact_power": 1.25, "rolling_shutter_artifact_velocity": (-4.5, 0.12)},
        "different_timing": {"rolling_shutter_artifact_power": 1.25, "rolling_shutter_artifact_row_readout_time": 0.065},
    }
    clean = np.asarray(source, dtype=np.float32)
    report = {}
    for name, override in variants.items():
        config = AttackConfig(**(base | override))
        attacked, pattern = AttackPipeline(config).execute(source, "rolling_shutter_artifact")
        attacked.save(output_dir / f"{name}.png")
        delta = np.abs(np.asarray(attacked, dtype=np.float32) - clean)
        report[name] = {"mean_absolute_difference": float(delta.mean()), "max_absolute_difference": float(delta.max()), "config": asdict(config), "pattern": pattern.metadata}
    (output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
