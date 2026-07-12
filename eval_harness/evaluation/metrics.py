from __future__ import annotations

import numpy as np


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    f1s = []
    for label in labels:
        tp = np.sum((y_true == label) & (y_pred == label))
        fp = np.sum((y_true != label) & (y_pred == label))
        fn = np.sum((y_true == label) & (y_pred != label))
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1s.append(0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall))
    return {
        "accuracy": float(np.mean(y_true == y_pred)),
        "macro_f1": float(np.mean(f1s)),
    }


def segmentation_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
    class_names: list[str] | None = None,
) -> dict[str, float]:
    ious = []
    result: dict[str, float] = {}
    for label in range(num_classes):
        intersection = np.sum((y_true == label) & (y_pred == label))
        union = np.sum((y_true == label) | (y_pred == label))
        if union > 0:
            iou = intersection / union
            ious.append(iou)
        else:
            iou = 0.0
        label_name = class_names[label] if class_names and label < len(class_names) else f"class_{label}"
        result[f"iou_{label_name}"] = float(iou)
    result["iou"] = float(np.mean(ious)) if ious else 0.0
    result["mean_iou"] = result["iou"]
    result["pixel_accuracy"] = float(np.mean(y_true == y_pred))
    if num_classes > 1:
        foreground = [
            value
            for key, value in result.items()
            if key.startswith("iou_") and key not in {"iou_background", "iou_not_flood"}
        ]
        if foreground:
            result["foreground_iou"] = float(np.mean(foreground))
    return result
