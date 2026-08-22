"""
Stripe-attribution ablation.
============================
Answers: do the detectors fire because of the STRIPES, or because of
secondary artifacts (color shift, flare, scene context)?

Method: blur sweep. Gaussian blur with growing sigma destroys fine periodic
stripe structure first, while preserving low-frequency content (scene layout,
global color casts, broad glows). We re-evaluate attacked recall and clean
specificity at each sigma:

    - recall collapsing toward 0 as sigma grows  -> detections ride on fine
      stripe texture (attribution confirmed)
    - clean images staying clean under blur     -> no false-alarm regression
    - recall surviving heavy blur               -> detector also uses coarse
      cues (envelope glow, color shift) - worth knowing, honestly reported

Usage (from project root):
    python scripts/attribution_ablation.py --config configs/config.yaml \
        --models single ensemble --sigmas 0,1,2,4,8
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml
from PIL import Image, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset.dataloader import LaserAttackDataset, get_dataloaders, get_ensemble_variation_names
from src.evaluation.evaluate import evaluate_model
from src.models.cnn import build_model
from src.models.ensemble import load_ensemble


class BlurredDataset(torch.utils.data.Dataset):
    """Wraps a dataset, applying PIL Gaussian blur before its transform."""

    def __init__(self, base, radius: float):
        self.base = base
        self.radius = radius

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        path, label, variation = self.base.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.radius > 0:
            img = img.filter(ImageFilter.GaussianBlur(radius=self.radius))
        if self.base.transform:
            img = self.base.transform(img)
        return img, label


def _load_models(cfg, which, device):
    ckpt_dir = cfg["paths"]["checkpoints"]
    models = {}
    if "single" in which:
        m = build_model(cfg)
        m.load_state_dict(torch.load(Path(ckpt_dir) / "single_cnn_best.pth", map_location=device))
        models["single"] = m.to(device).eval()
    if "ensemble" in which:
        models["ensemble"] = load_ensemble(
            get_ensemble_variation_names(cfg), ckpt_dir, cfg, device).to(device).eval()
    return models


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--models", default="single,ensemble")
    ap.add_argument("--sigmas", default="0,1,2,4,8", help="Gaussian blur sigmas to sweep")
    ap.add_argument("--split", default="test", choices=["test"])
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sigmas = [float(s) for s in args.sigmas.split(",")]

    loaders = get_dataloaders(cfg)
    base_loader = loaders[args.split]
    base_ds = base_loader.dataset
    train_cfg = cfg["training"]
    models = _load_models(cfg, args.models.split(","), device)

    results = {}
    for model_name, model in models.items():
        print(f"\n=== Stripe attribution: {model_name} on {args.split} ===")
        print(f"{'sigma':>6} {'recall_att':>11} {'precision':>10} {'F1':>8} {'FP':>5}")
        rows = {}
        for sigma in sigmas:
            ds = BlurredDataset(base_ds, radius=sigma)
            loader = torch.utils.data.DataLoader(
                ds, batch_size=train_cfg["batch_size"], shuffle=False,
                num_workers=train_cfg["num_workers"], pin_memory=False)
            m = evaluate_model(model, loader, device,
                               is_ensemble=(model_name == "ensemble"))
            cm = m["confusion_matrix"]
            fp = cm[0][1]
            rows[sigma] = {k: v for k, v in m.items() if k not in ("all_preds", "all_labels", "all_probs")}
            print(f"{sigma:>6.1f} {m['recall']:>11.4f} {m['precision']:>10.4f} "
                  f"{m['f1']:>8.4f} {fp:>5d}")
        results[model_name] = rows

    out_dir = Path(cfg["paths"]["results"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "attribution_ablation.json"
    with open(out_path, "w") as f:
        json.dump({"config": args.config, "split": args.split,
                   "sigmas": sigmas, "results": results}, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
