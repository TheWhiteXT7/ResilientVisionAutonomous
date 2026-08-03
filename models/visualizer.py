"""Visualization utilities for object detection predictions and comparison."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from models.predictor import DetectionBox, DetectionResult

logger = logging.getLogger(__name__)

DEFAULT_COLOR_PALETTE: List[Tuple[int, int, int]] = [
    (0, 255, 0),     # Bright Green
    (255, 0, 0),     # Bright Red
    (0, 0, 255),     # Bright Blue
    (255, 255, 0),   # Yellow
    (255, 0, 255),   # Magenta
    (0, 255, 255),   # Cyan
    (255, 128, 0),   # Orange
    (128, 0, 255),   # Purple
]


class YoloVisualizer:
    """Visualization service for rendering object detection bounding boxes and comparative figures."""

    def __init__(self, color_map: Optional[Dict[str, Tuple[int, int, int]]] = None) -> None:
        """Initialize YoloVisualizer.

        Args:
            color_map: Optional dictionary mapping class name string to RGB tuple.
        """
        self.color_map = color_map if color_map is not None else {}

    def _get_color(self, class_name: str, class_id: int) -> Tuple[int, int, int]:
        """Get RGB color for a given class label.

        Args:
            class_name: Object class label string.
            class_id: Class integer index.

        Returns:
            RGB tuple (r, g, b).
        """
        if class_name in self.color_map:
            return self.color_map[class_name]
        color_idx = class_id % len(DEFAULT_COLOR_PALETTE)
        return DEFAULT_COLOR_PALETTE[color_idx]

    def _load_pil_image(
        self, image_input: Union[Image.Image, np.ndarray, str, Path]
    ) -> Image.Image:
        """Helper to load PIL RGB Image from various input types.

        Args:
            image_input: PIL Image, numpy array, or file path.

        Returns:
            PIL Image in RGB mode.

        Raises:
            TypeError: If image_input is unsupported type.
        """
        if isinstance(image_input, Image.Image):
            return image_input.convert("RGB")
        elif isinstance(image_input, (str, Path)):
            path = Path(image_input)
            if not path.exists():
                raise FileNotFoundError(f"Image path not found: {path}")
            return Image.open(path).convert("RGB")
        elif isinstance(image_input, np.ndarray):
            if image_input.dtype != np.uint8:
                image_input = image_input.astype(np.uint8)
            if image_input.ndim == 2:
                return Image.fromarray(image_input).convert("RGB")
            elif image_input.ndim == 3:
                return Image.fromarray(image_input).convert("RGB")
        raise TypeError(f"Unsupported image input type: {type(image_input)}")

    def draw_bounding_boxes(
        self,
        image: Union[Image.Image, np.ndarray, str, Path],
        boxes: List[DetectionBox],
        line_thickness: int = 2,
    ) -> Image.Image:
        """Draw bounding boxes, labels, and confidence scores onto an image.

        Args:
            image: Source image (PIL Image, numpy array, or path).
            boxes: List of DetectionBox objects to draw.
            line_thickness: Bounding box line thickness in pixels.

        Returns:
            Annotated PIL Image in RGB format.
        """
        img = self._load_pil_image(image).copy()
        draw = ImageDraw.Draw(img)

        try:
            font = ImageFont.load_default()
        except Exception:
            font = None

        for box in boxes:
            color = self._get_color(box.class_name, box.class_id)
            left, top, right, bottom = box.bbox

            # Ensure coordinates are within image bounds
            left = max(0, left)
            top = max(0, top)
            right = min(img.width, right)
            bottom = min(img.height, bottom)

            # Draw bounding box rectangle
            for i in range(line_thickness):
                draw.rectangle(
                    [left - i, top - i, right + i, bottom + i],
                    outline=color,
                )

            # Format label string
            label = f"{box.class_name} {box.confidence:.2f}"

            # Calculate text background box size
            if hasattr(font, "getbbox"):
                bbox_text = font.getbbox(label)
                tw, th = bbox_text[2] - bbox_text[0], bbox_text[3] - bbox_text[1]
            else:
                tw, th = draw.textsize(label, font=font) if hasattr(draw, "textsize") else (8 * len(label), 12)

            text_top = max(0, top - th - 4)
            draw.rectangle(
                [left, text_top, left + tw + 6, text_top + th + 4],
                fill=color,
            )
            draw.text((left + 3, text_top + 2), label, fill=(255, 255, 255), font=font)

        return img

    def visualize_prediction(
        self,
        image: Union[Image.Image, np.ndarray, str, Path],
        detection_result: DetectionResult,
        output_path: Optional[Union[str, Path]] = None,
        line_thickness: int = 2,
    ) -> Image.Image:
        """Annotate an image with prediction results and optionally save to disk.

        Args:
            image: Source image.
            detection_result: DetectionResult containing predicted boxes.
            output_path: Optional output path to save annotated image.
            line_thickness: Bounding box line thickness.

        Returns:
            Annotated PIL Image.
        """
        annotated = self.draw_bounding_boxes(
            image=image,
            boxes=detection_result.boxes,
            line_thickness=line_thickness,
        )

        if output_path is not None:
            out_p = Path(output_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            annotated.save(out_p)
            logger.info(f"Saved annotated prediction image to: {out_p}")

        return annotated

    def create_side_by_side_comparison(
        self,
        image1: Union[Image.Image, np.ndarray, str, Path],
        image2: Union[Image.Image, np.ndarray, str, Path],
        title1: str = "Clean",
        title2: str = "Attacked",
        output_path: Optional[Union[str, Path]] = None,
    ) -> Image.Image:
        """Create a side-by-side comparative image (e.g., Clean vs Attacked).

        Args:
            image1: First PIL Image or path.
            image2: Second PIL Image or path.
            title1: Title banner for first image.
            title2: Title banner for second image.
            output_path: Optional path to save composite image.

        Returns:
            Combined PIL Image showing both inputs side-by-side.
        """
        img1 = self._load_pil_image(image1)
        img2 = self._load_pil_image(image2)

        # Target uniform height
        target_h = max(img1.height, img2.height)
        w1 = int(img1.width * (target_h / img1.height))
        w2 = int(img2.width * (target_h / img2.height))

        img1_res = img1.resize((w1, target_h), Image.Resampling.LANCZOS)
        img2_res = img2.resize((w2, target_h), Image.Resampling.LANCZOS)

        banner_h = 40
        canvas_w = w1 + w2 + 10  # 10px spacing
        canvas_h = target_h + banner_h

        combined = Image.new("RGB", (canvas_w, canvas_h), color=(30, 30, 30))

        # Paste images
        combined.paste(img1_res, (0, banner_h))
        combined.paste(img2_res, (w1 + 10, banner_h))

        draw = ImageDraw.Draw(combined)
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None

        # Draw title banners
        draw.text((15, 12), title1, fill=(255, 255, 255), font=font)
        draw.text((w1 + 25, 12), title2, fill=(255, 255, 255), font=font)

        if output_path is not None:
            out_p = Path(output_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            combined.save(out_p)
            logger.info(f"Saved side-by-side comparison image to: {out_p}")

        return combined
