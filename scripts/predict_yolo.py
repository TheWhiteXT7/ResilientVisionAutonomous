"""YOLO prediction CLI using YoloPredictor and YoloVisualizer."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, List, Optional

from models.yolo_wrapper import YoloWrapper
from models.predictor import YoloPredictor
from models.visualizer import YoloVisualizer
from dataset_loader.kitti_loader import KittiLoader

logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run YOLO inference on image(s) or dataset")
    p.add_argument("--weights", required=True, help="Path to model weights (.pt)")
    p.add_argument("--source", required=True, help="Image file, directory, or dataset directory")
    p.add_argument("--save-images", action="store_true", help="Save annotated images")
    p.add_argument("--conf", type=float, default=None, help="Confidence threshold override")
    p.add_argument("--iou", type=float, default=None, help="IoU threshold override")
    return p


def main(argv: Any = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)

    weights = Path(args.weights)
    if not weights.exists():
        logger.error("Weights not found: %s", weights)
        return 2

    wrapper = YoloWrapper(model_path=str(weights))
    predictor = YoloPredictor(wrapper=wrapper)
    visualizer = YoloVisualizer()

    src = Path(args.source)
    results: List = []

    try:
        if src.is_file():
            res = predictor.predict_image(str(src))
            results = [res]
        elif src.is_dir():
            # Decide whether this is a YOLO-prepared dataset (images/train) or plain image dir
            # If directory contains 'image_2' assume KITTI layout
            if (src / "image_2").exists():
                loader = KittiLoader(kitti_dir=src, split="test", load_images=False)
                results = predictor.predict_dataset(loader)
            else:
                results = predictor.predict_directory(src)
        else:
            logger.error("Source not found: %s", src)
            return 2

        # Save visualizations if requested
        if args.save_images:
            out_dir = Path("outputs") / "predictions"
            out_dir.mkdir(parents=True, exist_ok=True)
            for det in results:
                if det.image_path:
                    out_file = out_dir / f"{det.sample_id}_annotated.png"
                    try:
                        visualizer.visualize_prediction(det.image_path, det, output_path=out_file)
                    except Exception:
                        logger.exception("Failed to save visualization for %s", det.sample_id)

        # Print JSON summary
        def _serialisable(det):
            return {
                "sample_id": det.sample_id,
                "image_path": str(det.image_path) if det.image_path else None,
                "num_boxes": len(det.boxes),
                "boxes": [
                    {"class_id": b.class_id, "class_name": b.class_name, "confidence": b.confidence, "bbox": b.bbox}
                    for b in det.boxes
                ],
            }

        print(json.dumps([_serialisable(d) for d in results], indent=2))
        return 0
    except Exception:
        logger.exception("Inference failed")
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
