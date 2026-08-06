"""Compare two evaluation JSON reports and emit a comparison JSON."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict


def _load_report(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def compare(baseline: Dict[str, Any], experiment: Dict[str, Any]) -> Dict[str, Any]:
    keys = ["mAP50", "mAP50-95", "precision", "recall", "f1"]
    comp = []
    for k in keys:
        b = float(baseline.get("metrics", {}).get(k, 0.0))
        e = float(experiment.get("metrics", {}).get(k, 0.0))
        diff = e - b
        rel = (diff / b * 100.0) if b != 0 else None
        comp.append({"metric": k, "baseline": b, "experiment": e, "difference": diff, "relative_percent": rel})
    return {"comparison": comp}


def main(argv: Any = None) -> int:
    p = argparse.ArgumentParser(description="Compare two evaluation reports")
    p.add_argument("baseline_json", help="Baseline evaluation JSON")
    p.add_argument("experiment_json", help="Experiment evaluation JSON")
    p.add_argument("--output", help="Comparison output JSON", default="comparison.json")
    args = p.parse_args(argv)

    b = _load_report(Path(args.baseline_json))
    e = _load_report(Path(args.experiment_json))
    comp = compare(b, e)
    out_path = Path(args.output)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(comp, fh, indent=2)
    print(f"Wrote comparison to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())