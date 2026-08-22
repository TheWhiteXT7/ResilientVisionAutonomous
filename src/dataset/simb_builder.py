"""
Evaluation-set builder for Simulator B (cross-simulator robustness test).
=========================================================================
Generates an evaluation-only dataset with the INDEPENDENT renderer in
simulator_b.py, writing the same labels.csv schema as the v7 builder so
that dataloader/evaluate/audit_pixels work unchanged.

Cross-simulator semantics:
  - backgrounds come from the seed-consistent TEST split only (same scenes
    the model never trained on), and every row is labelled "test";
  - rendering is entirely simulator_b.py - different modulation physics,
    different parameter axes (spatial periods, not Hz+exposure), different
    sensor/post-processing chain. Any accuracy change vs the v7 OOD numbers
    therefore measures transfer across RENDERERS, not just parameters.

Usage:
    python src/dataset/simb_builder.py --config configs/simb_eval.yaml --seed 42
"""

import argparse
import csv
import hashlib
import random
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

try:
    from dataset_builder import assign_source_splits, load_clean_images, sample_range
    from simulator_b import SimBParams, render_clean, render_frame
except ImportError:  # pragma: no cover
    from src.dataset.dataset_builder import assign_source_splits, load_clean_images, sample_range
    from src.dataset.simulator_b import SimBParams, render_clean, render_frame


def _stable_rng(*parts) -> random.Random:
    """Deterministic RNG from string parts (hash() is salted per process)."""
    key = "-".join(str(p) for p in parts)
    return random.Random(int(hashlib.md5(key.encode()).hexdigest()[:8], 16))


def _stable_noise_seed(*parts) -> np.random.Generator:
    key = "-".join(str(p) for p in parts)
    return np.random.default_rng(int(hashlib.md5(key.encode()).hexdigest()[:8], 16))


def build_simb_params(rng: random.Random, var_cfg: dict) -> SimBParams:
    return SimBParams(
        band_cycles=sample_range(rng, var_cfg.get("band_cycles_range", [30, 60])),
        angle_deg=sample_range(rng, var_cfg.get("angle_range", [-30, 30])),
        amplitude=sample_range(rng, var_cfg.get("amplitude_range", [0.1, 0.3])),
        wavelength_nm=int(sample_range(rng, var_cfg.get("wavelength_choices", [532]))),
        envelope_cx=rng.uniform(0.25, 0.75),
        envelope_cy=rng.uniform(0.25, 0.75),
        envelope_sigma_x=sample_range(rng, var_cfg.get("envelope_sigma_x_range", [0.3, 0.6])),
        envelope_sigma_y=sample_range(rng, var_cfg.get("envelope_sigma_y_range", [0.3, 0.6])),
        phase=rng.uniform(0.0, 1.0),
        bloom_sigma=sample_range(rng, var_cfg.get("bloom_sigma_range", [1.0, 4.0])),
        noise_std=sample_range(rng, var_cfg.get("noise_std_range", [2.0, 6.0])),
        exposure_gain=sample_range(rng, var_cfg.get("exposure_gain_range", [0.9, 1.15])),
    )


def build_simb_dataset(config_path, seed: int = 42) -> Path:
    config_path = Path(config_path)
    if not config_path.is_absolute() and not config_path.exists():
        project_root = Path(__file__).resolve().parents[2]
        candidate = project_root / config_path
        if candidate.exists():
            config_path = candidate
    config_path = config_path.resolve()
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    project_root = config_path.parent.parent
    clean_base = project_root / cfg["paths"]["clean_base"]
    out_dir = project_root / cfg["paths"]["final_dataset"]
    out_dir.mkdir(parents=True, exist_ok=True)
    height = int(cfg.get("image_size", {}).get("height", 224))
    width = int(cfg.get("image_size", {}).get("width", 224))
    split_cfg = cfg.get("split", {"train": 0.7, "val": 0.15, "test": 0.15})

    clean_paths = load_clean_images(clean_base)
    print(f"Found {len(clean_paths)} clean background images in {clean_base}")

    # Same source-level split policy as training, restricted to test sources.
    split_of_source = assign_source_splits(clean_paths, split_cfg, seed)
    test_sources = [p for p, s in split_of_source.items() if s == "test"]
    if not test_sources:
        raise ValueError("no sources assigned to the test split for this seed")
    print(f"SimB eval set: restricting to {len(test_sources)} test-split background sources")

    labels_path = out_dir / "labels.csv"
    total_written = 0
    t0 = time.time()

    with open(labels_path, "w", newline="") as lf:
        writer = csv.writer(lf)
        writer.writerow(["path", "label", "split", "variation", "frequency", "wavelength",
                          "power_mw", "duty_cycle", "modulation", "coverage", "angle_deg",
                          "distance_m", "ellipticity", "exposure_time", "ae_gain",
                          "peak_saturation", "attack_area_fraction", "source_path"])

        for var_cfg in cfg["variations"]:
            name = var_cfg["name"]
            label = var_cfg.get("label", name)
            count = int(var_cfg["count"])
            is_clean = var_cfg.get("clean", False)

            var_dir = out_dir / name
            var_dir.mkdir(parents=True, exist_ok=True)

            print(f"[{name}] generating {count} images (clean={is_clean}) ...")
            for i in range(count):
                src_path = test_sources[_stable_rng(seed, name, i, "src").randrange(len(test_sources))]
                bg = cv2.imread(str(src_path))
                if bg is None:
                    continue
                if bg.shape[:2] != (height, width):
                    bg = cv2.resize(bg, (width, height))

                noise_rng = _stable_noise_seed(seed, name, i, "noise")
                if is_clean:
                    out, meta = render_clean(bg, noise_rng)
                else:
                    out, meta = render_frame(
                        bg,
                        build_simb_params(_stable_rng(seed, name, i, "params"), var_cfg),
                        noise_rng,
                    )

                fpath = var_dir / f"{name}_{i:05d}.png"
                cv2.imwrite(str(fpath), out)
                writer.writerow([str(fpath.relative_to(out_dir)), label, "test", name,
                                  meta["frequency"], meta["wavelength"], meta["power_mw"],
                                  meta["duty_cycle"], meta["modulation"], meta["coverage"],
                                  meta["angle_deg"], meta["distance_m"], meta["ellipticity"],
                                  meta["exposure_time"], meta["ae_gain"],
                                  meta["peak_saturation"], meta["attack_area_fraction"],
                                  str(Path(src_path).relative_to(clean_base))])
                total_written += 1
            print(f"[{name}] done.")

    elapsed = time.time() - t0
    print(f"\nWrote {total_written} images + labels to {out_dir} in {elapsed/60:.1f} minutes.")
    return out_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/simb_eval.yaml")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    build_simb_dataset(args.config, args.seed)


if __name__ == "__main__":
    main()
