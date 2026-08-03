"""YOLO Predictor module for object detection inference."""

from dataclasses import dataclass, field
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image

from models.utils import get_default_class_mapping
from models.yolo_config import YoloConfig
from models.yolo_wrapper import YoloWrapper

logger = logging.getLogger(__name__)


@dataclass
class DetectionBox:
    """Dataclass representing a single detected bounding box.

    Attributes:
        class_id: Class integer ID.
        class_name: Object class string label (e.g. 'Car').
        confidence: Confidence score [0.0..1.0].
        bbox: Bounding box pixel coordinates (left, top, right, bottom).
    """

    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[float, float, float, float]


@dataclass
class DetectionResult:
    """Dataclass representing detection inference output for an image sample.

    Attributes:
        sample_id: Unique string identifier for sample.
        image_path: Optional path to source image.
        boxes: List of DetectionBox objects found in the image.
        orig_shape: Image original dimensions (height, width).
        speed: Optional dictionary containing inference timing breakdown (ms).
    """

    sample_id: str
    image_path: Optional[Path]
    boxes: List[DetectionBox] = field(default_factory=list)
    orig_shape: Tuple[int, int] = (0, 0)
    speed: Optional[Dict[str, float]] = field(default_factory=dict)


class YoloPredictor:
    """Predictor service for running object detection inference with YOLO models."""

    def __init__(
        self,
        wrapper: Optional[YoloWrapper] = None,
        config: Optional[YoloConfig] = None,
        class_names: Optional[List[str]] = None,
    ) -> None:
        """Initialize YoloPredictor.

        Args:
            wrapper: Optional YoloWrapper instance.
            config: Optional YoloConfig instance.
            class_names: Optional list of class names ordered by integer index.
        """
        if config is not None:
            self.config = config
        elif wrapper is not None and getattr(wrapper, "config", None) is not None:
            self.config = wrapper.config
        else:
            self.config = YoloConfig()

        if wrapper is not None:
            self.wrapper = wrapper
        else:
            self.wrapper = YoloWrapper(config=self.config)

        self._class_names = class_names

    def _parse_ultralytics_result(
        self, result: Any, sample_id: str = "", image_path: Optional[Path] = None
    ) -> DetectionResult:
        """Parse raw Ultralytics Result object into a structured DetectionResult.

        Args:
            result: Ultralytics Result object.
            sample_id: Identifier string for sample.
            image_path: Path to input image file if applicable.

        Returns:
            DetectionResult instance.
        """
        boxes: List[DetectionBox] = []
        orig_shape = (0, 0)
        speed_dict: Dict[str, float] = {}

        if hasattr(result, "orig_shape") and result.orig_shape is not None:
            orig_shape = (int(result.orig_shape[0]), int(result.orig_shape[1]))

        if hasattr(result, "speed") and isinstance(result.speed, dict):
            speed_dict = {k: float(v) for k, v in result.speed.items()}

        names_map = getattr(result, "names", {})
        if not names_map and self._class_names:
            names_map = {idx: name for idx, name in enumerate(self._class_names)}
        elif not names_map:
            names_map = {idx: name for name, idx in get_default_class_mapping().items()}

        if hasattr(result, "boxes") and result.boxes is not None:
            raw_boxes = result.boxes
            if hasattr(raw_boxes, "xyxy"):
                xyxy = raw_boxes.xyxy.cpu().numpy() if hasattr(raw_boxes.xyxy, "cpu") else np.array(raw_boxes.xyxy)
                confs = raw_boxes.conf.cpu().numpy() if hasattr(raw_boxes.conf, "cpu") else np.array(raw_boxes.conf)
                cls_ids = raw_boxes.cls.cpu().numpy() if hasattr(raw_boxes.cls, "cpu") else np.array(raw_boxes.cls)

                for box_arr, conf_val, cls_val in zip(xyxy, confs, cls_ids):
                    cid = int(cls_val)
                    cname = str(names_map.get(cid, f"class_{cid}"))
                    left, top, right, bottom = (
                        float(box_arr[0]),
                        float(box_arr[1]),
                        float(box_arr[2]),
                        float(box_arr[3]),
                    )
                    boxes.append(
                        DetectionBox(
                            class_id=cid,
                            class_name=cname,
                            confidence=float(conf_val),
                            bbox=(left, top, right, bottom),
                        )
                    )

        return DetectionResult(
            sample_id=sample_id,
            image_path=image_path,
            boxes=boxes,
            orig_shape=orig_shape,
            speed=speed_dict,
        )

    def predict_image(
        self,
        image_input: Union[str, Path, Image.Image, np.ndarray],
        sample_id: str = "single_image",
        **kwargs: Any,
    ) -> DetectionResult:
        """Run object detection prediction on a single image.

        Args:
            image_input: Image file path, PIL Image, or numpy array.
            sample_id: Optional custom identifier string.
            **kwargs: Inference argument overrides.

        Returns:
            DetectionResult instance containing predicted bounding boxes.
        """
        img_path: Optional[Path] = (
            Path(image_input) if isinstance(image_input, (str, Path)) else None
        )
        sid = sample_id if sample_id != "single_image" else (img_path.stem if img_path else "single_image")

        raw_results = self.wrapper.predict(image_input, **kwargs)
        raw_res = raw_results[0] if isinstance(raw_results, list) and raw_results else raw_results
        return self._parse_ultralytics_result(raw_res, sample_id=sid, image_path=img_path)

    def predict_directory(
        self, dir_path: Union[str, Path], **kwargs: Any
    ) -> List[DetectionResult]:
        """Run object detection prediction on all images inside a directory.

        Args:
            dir_path: Directory path containing image files.
            **kwargs: Inference argument overrides.

        Returns:
            List of DetectionResult instances for each image found.
        """
        dpath = Path(dir_path)
        valid_exts = {".png", ".jpg", ".jpeg", ".bmp"}
        img_files = sorted(
            [f for f in dpath.iterdir() if f.is_file() and f.suffix.lower() in valid_exts]
        )

        logger.info(f"Predicting on {len(img_files)} images from directory: {dpath}")
        results_list: List[DetectionResult] = []

        for img_file in img_files:
            res = self.predict_image(img_file, sample_id=img_file.stem, **kwargs)
            results_list.append(res)

        return results_list

    def predict_dataset(self, dataset: Any, **kwargs: Any) -> List[DetectionResult]:
        """Run object detection prediction across a dataset sequence (e.g. KittiLoader).

        Args:
            dataset: Sequence dataset object containing KittiSample objects or image paths.
            **kwargs: Inference argument overrides.

        Returns:
            List of DetectionResult instances for every sample in dataset.
        """
        results_list: List[DetectionResult] = []
        logger.info("Starting dataset prediction run...")

        if hasattr(dataset, "__iter__"):
            for sample in dataset:
                sid = getattr(sample, "sample_id", "sample")
                img_path = getattr(sample, "image_path", None)
                img_obj = getattr(sample, "image", None)

                source = img_obj if img_obj is not None else (img_path if img_path else None)
                if source is None:
                    continue

                res = self.predict_image(source, sample_id=sid, **kwargs)
                if img_path and res.image_path is None:
                    res.image_path = Path(img_path)
                results_list.append(res)

        logger.info(f"Dataset prediction finished. Total samples processed: {len(results_list)}")
        return results_list
