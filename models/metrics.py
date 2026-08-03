"""Metrics utilities for evaluating object detection performance."""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


def compute_iou(
    box1: Tuple[float, float, float, float],
    box2: Tuple[float, float, float, float],
) -> float:
    """Compute Intersection over Union (IoU) between two bounding boxes.

    Args:
        box1: Bounding box pixel coords (xmin, ymin, xmax, ymax).
        box2: Bounding box pixel coords (xmin, ymin, xmax, ymax).

    Returns:
        IoU scalar value in range [0.0, 1.0].
    """
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    inter_x1 = max(x1_1, x1_2)
    inter_y1 = max(y1_1, y1_2)
    inter_x2 = min(x2_1, x2_2)
    inter_y2 = min(y2_1, y2_2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area1 = max(0.0, x2_1 - x1_1) * max(0.0, y2_1 - y1_1)
    area2 = max(0.0, x2_2 - x1_2) * max(0.0, y2_2 - y1_2)
    union_area = area1 + area2 - inter_area

    if union_area <= 0.0:
        return 0.0

    return float(inter_area / union_area)


def compute_precision_recall_f1(
    tp: int, fp: int, fn: int
) -> Tuple[float, float, float]:
    """Compute Precision, Recall, and F1 score from TP, FP, and FN counts.

    Args:
        tp: True Positive count.
        fp: False Positive count.
        fn: False Negative count.

    Returns:
        Tuple of (precision, recall, f1_score).
    """
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return (float(precision), float(recall), float(f1))


def _compute_ap_per_class(
    pred_boxes_class: List[Tuple[float, Tuple[float, float, float, float]]],
    gt_boxes_class: List[Tuple[float, float, float, float]],
    iou_thresh: float,
) -> Tuple[float, int, int, int]:
    """Helper to compute Average Precision (AP) for a single class at a given IoU threshold.

    Args:
        pred_boxes_class: List of (confidence, bbox) tuples.
        gt_boxes_class: List of ground truth bboxes.
        iou_thresh: IoU matching threshold.

    Returns:
        Tuple of (AP, tp_count, fp_count, fn_count).
    """
    num_gts = len(gt_boxes_class)
    if num_gts == 0 and len(pred_boxes_class) == 0:
        return (1.0, 0, 0, 0)
    elif num_gts == 0:
        return (0.0, 0, len(pred_boxes_class), 0)
    elif len(pred_boxes_class) == 0:
        return (0.0, 0, 0, num_gts)

    # Sort predictions by confidence descending
    sorted_preds = sorted(pred_boxes_class, key=lambda x: x[0], reverse=True)
    gt_matched = [False] * num_gts

    tp = np.zeros(len(sorted_preds))
    fp = np.zeros(len(sorted_preds))

    for idx, (_, p_box) in enumerate(sorted_preds):
        best_iou = 0.0
        best_gt_idx = -1
        for gt_idx, g_box in enumerate(gt_boxes_class):
            iou = compute_iou(p_box, g_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_iou >= iou_thresh and best_gt_idx >= 0 and not gt_matched[best_gt_idx]:
            tp[idx] = 1.0
            gt_matched[best_gt_idx] = True
        else:
            fp[idx] = 1.0

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recalls = tp_cum / num_gts
    precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1e-16)

    # Compute AP using 11-point interpolation or area under PR curve
    recalls_ext = np.concatenate(([0.0], recalls, [1.0]))
    precisions_ext = np.concatenate(([0.0], precisions, [0.0]))
    for i in range(len(precisions_ext) - 2, -1, -1):
        precisions_ext[i] = max(precisions_ext[i], precisions_ext[i + 1])

    i_indices = np.where(recalls_ext[1:] != recalls_ext[:-1])[0]
    ap = float(np.sum((recalls_ext[i_indices + 1] - recalls_ext[i_indices]) * precisions_ext[i_indices + 1]))

    tp_total = int(np.sum(tp))
    fp_total = int(np.sum(fp))
    fn_total = num_gts - tp_total

    return (ap, tp_total, fp_total, fn_total)


def compute_detection_metrics(
    predictions: List[Any],
    ground_truths: Union[List[List[Any]], Dict[str, List[Any]]],
    iou_threshold: float = 0.5,
    class_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Compute mAP50, mAP50-95, Precision, Recall, and F1 metrics.

    Args:
        predictions: List of DetectionResult objects.
        ground_truths: List of Annotation lists per image or sample_id to Annotation list dict.
        iou_threshold: IoU threshold for standard precision/recall/mAP50.
        class_names: Optional explicit list of class names to evaluate.

    Returns:
        Dictionary containing metric values (mAP50, mAP50-95, precision, recall, f1, etc.).
    """
    # Build dictionary of ground truths per sample
    gt_map: Dict[str, List[Any]] = {}
    if isinstance(ground_truths, dict):
        gt_map = ground_truths
    elif isinstance(ground_truths, list):
        for idx, res in enumerate(predictions):
            sid = getattr(res, "sample_id", str(idx))
            if idx < len(ground_truths):
                gt_map[sid] = ground_truths[idx]

    # Collect predictions and ground truths per class
    all_classes = set()
    if class_names:
        all_classes.update(class_names)

    pred_per_class: Dict[str, List[Tuple[float, Tuple[float, float, float, float]]]] = {}
    gt_per_class: Dict[str, List[Tuple[float, float, float, float]]] = {}

    for res in predictions:
        sid = getattr(res, "sample_id", "")
        boxes = getattr(res, "boxes", [])
        for box in boxes:
            cname = getattr(box, "class_name", str(getattr(box, "class_id", 0)))
            all_classes.add(cname)
            if cname not in pred_per_class:
                pred_per_class[cname] = []
            pred_per_class[cname].append((getattr(box, "confidence", 1.0), getattr(box, "bbox", (0, 0, 0, 0))))

        gts = gt_map.get(sid, [])
        for ann in gts:
            cname = getattr(ann, "class_name", "DontCare")
            if cname == "DontCare":
                continue
            all_classes.add(cname)
            if cname not in gt_per_class:
                gt_per_class[cname] = []
            gt_per_class[cname].append(getattr(ann, "bbox", (0, 0, 0, 0)))

    # Compute per-class mAP50 and mAP50-95
    ap50_list: List[float] = []
    ap_multi_list: List[float] = []
    total_tp, total_fp, total_fn = 0, 0, 0

    iou_thresholds = np.linspace(0.5, 0.95, 10)

    for cname in sorted(all_classes):
        preds = pred_per_class.get(cname, [])
        gts = gt_per_class.get(cname, [])

        ap50, tp, fp, fn = _compute_ap_per_class(preds, gts, iou_threshold)
        ap50_list.append(ap50)
        total_tp += tp
        total_fp += fp
        total_fn += fn

        # Multi IoU threshold APs for mAP50-95
        ap_thresholds = [
            _compute_ap_per_class(preds, gts, thresh)[0] for thresh in iou_thresholds
        ]
        ap_multi_list.append(float(np.mean(ap_thresholds)))

    mAP50 = float(np.mean(ap50_list)) if ap50_list else 0.0
    mAP50_95 = float(np.mean(ap_multi_list)) if ap_multi_list else 0.0
    precision, recall, f1 = compute_precision_recall_f1(total_tp, total_fp, total_fn)

    return {
        "mAP50": mAP50,
        "mAP50-95": mAP50_95,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
    }


def compute_confusion_matrix(
    predictions: List[Any],
    ground_truths: Union[List[List[Any]], Dict[str, List[Any]]],
    num_classes: int,
    iou_threshold: float = 0.5,
) -> np.ndarray:
    """Compute confusion matrix for object detection predictions vs ground truth.

    Size is (num_classes + 1, num_classes + 1) where last index is background.

    Args:
        predictions: List of DetectionResult objects.
        ground_truths: List of Annotation lists per sample or dict of sample_id to annotations.
        num_classes: Total number of target classes.
        iou_threshold: IoU threshold for matching.

    Returns:
        2D numpy array confusion matrix.
    """
    matrix = np.zeros((num_classes + 1, num_classes + 1), dtype=int)
    bg_idx = num_classes

    gt_map: Dict[str, List[Any]] = {}
    if isinstance(ground_truths, dict):
        gt_map = ground_truths
    elif isinstance(ground_truths, list):
        for idx, res in enumerate(predictions):
            sid = getattr(res, "sample_id", str(idx))
            if idx < len(ground_truths):
                gt_map[sid] = ground_truths[idx]

    for res in predictions:
        sid = getattr(res, "sample_id", "")
        p_boxes = getattr(res, "boxes", [])
        gts = gt_map.get(sid, [])

        gts_valid = [g for g in gts if getattr(g, "class_name", "") != "DontCare"]
        gt_matched = [False] * len(gts_valid)

        for p in p_boxes:
            p_cid = getattr(p, "class_id", 0)
            if p_cid >= num_classes:
                p_cid = bg_idx

            p_box = getattr(p, "bbox", (0, 0, 0, 0))

            best_iou = 0.0
            best_gt_idx = -1
            for g_idx, g in enumerate(gts_valid):
                g_box = getattr(g, "bbox", (0, 0, 0, 0))
                iou = compute_iou(p_box, g_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = g_idx

            if best_iou >= iou_threshold and best_gt_idx >= 0 and not gt_matched[best_gt_idx]:
                gt_cid = getattr(gts_valid[best_gt_idx], "class_id", 0)
                if gt_cid >= num_classes:
                    gt_cid = bg_idx
                matrix[gt_cid, p_cid] += 1
                gt_matched[best_gt_idx] = True
            else:
                matrix[bg_idx, p_cid] += 1

        for g_idx, g in enumerate(gts_valid):
            if not gt_matched[g_idx]:
                gt_cid = getattr(g, "class_id", 0)
                if gt_cid >= num_classes:
                    gt_cid = bg_idx
                matrix[gt_cid, bg_idx] += 1

    return matrix


def compare_metrics(
    clean_metrics: Dict[str, Any], attacked_metrics: Dict[str, Any]
) -> Dict[str, Any]:
    """Compute performance drops and deltas between clean and attacked metrics.

    Args:
        clean_metrics: Metric dictionary from clean dataset evaluation.
        attacked_metrics: Metric dictionary from attacked dataset evaluation.

    Returns:
        Dictionary containing metric deltas and percentage drops.
    """
    comparison: Dict[str, Any] = {
        "clean": clean_metrics,
        "attacked": attacked_metrics,
        "deltas": {},
        "percentage_drops": {},
    }

    metric_keys = ["mAP50", "mAP50-95", "precision", "recall", "f1"]
    for key in metric_keys:
        c_val = float(clean_metrics.get(key, 0.0))
        a_val = float(attacked_metrics.get(key, 0.0))
        delta = c_val - a_val
        pct_drop = (delta / c_val * 100.0) if c_val > 0 else 0.0

        comparison["deltas"][key] = float(delta)
        comparison["percentage_drops"][key] = float(pct_drop)

    return comparison


def aggregate_metrics(metrics_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate metrics across multiple evaluation runs.

    Args:
        metrics_list: List of metric dictionaries.

    Returns:
        Dictionary containing mean, std, min, and max for each metric.
    """
    if not metrics_list:
        return {}

    metric_keys = ["mAP50", "mAP50-95", "precision", "recall", "f1"]
    aggregated: Dict[str, Any] = {}

    for key in metric_keys:
        values = [float(m.get(key, 0.0)) for m in metrics_list if key in m]
        if values:
            aggregated[key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }

    return aggregated
