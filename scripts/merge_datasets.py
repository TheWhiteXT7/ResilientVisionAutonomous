"""
Merge two v7-format datasets into one for multi-simulator training.
===================================================================
Reads two labels.csv files, copies images into a new merged directory,
and writes a combined labels.csv with an extra `simulator` column.

Usage:
    python scripts/merge_datasets.py \
        --src1 data/final_dataset/labels.csv \
        --src2 data/final_dataset_simb_train/labels.csv \
        --out data/final_dataset_multisim \
        --sim1 v7 --sim2 simb
"""

import argparse
import csv
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src1", required=True, help="First labels.csv (e.g., v7)")
    ap.add_argument("--src2", required=True, help="Second labels.csv (e.g., SimB)")
    ap.add_argument("--out", required=True, help="Output merged dataset directory")
    ap.add_argument("--sim1", default="v7", help="Simulator name for source 1")
    ap.add_argument("--sim2", default="simb", help="Simulator name for source 2")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Read both CSVs
    rows = []
    for src_path, sim_name in [(args.src1, args.sim1), (args.src2, args.sim2)]:
        src_root = Path(src_path).parent
        with open(src_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Prefix variation with simulator to avoid name collision,
                # EXCEPT for "clean" - keep it as "clean" so dataloader
                # recognizes it as the negative class.
                var = row["variation"]
                if var != "clean":
                    row["variation"] = f"{sim_name}_{var}"
                row["simulator"] = sim_name
                rows.append((src_root, row))

    # Shuffle deterministically so train/val/test batches are mixed
    import random
    rng = random.Random(42)
    rng.shuffle(rows)

    # Write merged CSV and copy images
    out_csv = out_dir / "labels.csv"
    total = 0
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "label", "split", "variation", "frequency", "wavelength",
                          "power_mw", "duty_cycle", "modulation", "coverage", "angle_deg",
                          "distance_m", "ellipticity", "exposure_time", "ae_gain",
                          "peak_saturation", "attack_area_fraction", "source_path", "simulator"])
        for src_root, row in rows:
            src_img = src_root / row["path"]
            # New path: simulator/variation/filename
            new_rel = Path(args.sim1 if row["simulator"] == args.sim1 else args.sim2) / row["variation"] / Path(row["path"]).name
            dst_img = out_dir / new_rel
            dst_img.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_img, dst_img)
            writer.writerow([str(new_rel), row["label"], row["split"], row["variation"],
                              row["frequency"], row["wavelength"], row["power_mw"],
                              row["duty_cycle"], row["modulation"], row["coverage"],
                              row["angle_deg"], row["distance_m"], row["ellipticity"],
                              row["exposure_time"], row["ae_gain"],
                              row["peak_saturation"], row["attack_area_fraction"],
                              row["source_path"], row["simulator"]])
            total += 1

    print(f"Merged {total} images into {out_dir}")


if __name__ == "__main__":
    main()