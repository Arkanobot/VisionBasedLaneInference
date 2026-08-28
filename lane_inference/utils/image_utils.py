"""Mask post-processing utilities."""
from __future__ import annotations

import cv2
import numpy as np


def keep_largest_blob(binary_mask: np.ndarray) -> np.ndarray:
    """Phase 3: retain only the single largest connected component."""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask)
    if num_labels <= 1:
        return binary_mask
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    cleaned = np.zeros_like(binary_mask)
    cleaned[labels == largest] = 255
    return cleaned


def apply_road_roi(mask: np.ndarray) -> np.ndarray:
    """Phase 3: fixed trapezoidal ROI (disabled in the Phase 3 pipeline)."""
    h, w = mask.shape
    roi = np.zeros_like(mask)
    polygon = np.array([[
        (int(0.1 * w), h),
        (int(0.9 * w), h),
        (int(0.65 * w), int(0.55 * h)),
        (int(0.35 * w), int(0.55 * h)),
    ]], dtype=np.int32)
    cv2.fillPoly(roi, polygon, 255)
    return cv2.bitwise_and(mask, roi)


def remove_small_components(mask: np.ndarray, min_area_ratio: float = 0.005) -> np.ndarray:
    """Drop connected components smaller than a fraction of the frame area."""
    h, w = mask.shape[:2]
    min_area = int(min_area_ratio * h * w)
    n, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
    out = np.zeros_like(mask)
    for lab in range(1, n):
        if stats[lab, cv2.CC_STAT_AREA] >= min_area:
            out[labels == lab] = 255
    return out


def fill_holes(mask: np.ndarray, max_hole_ratio: float = 0.05) -> np.ndarray:
    """Fill enclosed holes in the road mask (vehicles, potholes, shadows) whose"""
    h, w = mask.shape[:2]
    binary = (mask > 0).astype(np.uint8)
    inv = 1 - binary
    n, labels, stats, _ = cv2.connectedComponentsWithStats(inv, 4)
    out = binary.copy()
    max_hole = int(max_hole_ratio * h * w)
    border = set(np.unique(np.concatenate([
        labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]
    ])))
    for lab in range(1, n):
        if lab in border:
            continue
        if stats[lab, cv2.CC_STAT_AREA] <= max_hole:
            out[labels == lab] = 1
    return out * 255


def smooth_mask(mask: np.ndarray, kernel: int = 5) -> np.ndarray:
    """Close then open with an elliptical kernel to regularise the contour."""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel, kernel))
    m = cv2.morphologyEx((mask > 0).astype(np.uint8) * 255, cv2.MORPH_CLOSE, k)
    return cv2.morphologyEx(m, cv2.MORPH_OPEN, k)


def largest_contour(mask: np.ndarray):
    """Return the largest external contour of a binary mask, or None."""
    cnts, _ = cv2.findContours((mask > 0).astype(np.uint8),
                               cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return max(cnts, key=cv2.contourArea) if cnts else None


def overlay_mask(image: np.ndarray, mask: np.ndarray,
                 color=(60, 60, 220), alpha: float = 0.45) -> np.ndarray:
    """Alpha-blend a colour over the masked pixels only."""
    out = image.copy()
    sel = mask > 0
    if sel.any():
        tint = np.empty_like(image)
        tint[:] = color
        out[sel] = (image[sel] * (1.0 - alpha) + tint[sel] * alpha).astype(np.uint8)
    return out


def colorize_labels(label: np.ndarray, palette, ignore_index: int = 255) -> np.ndarray:
    """Map a class-index map to a BGR image using the given palette."""
    out = np.zeros((*label.shape, 3), dtype=np.uint8)
    for idx, color in enumerate(palette):
        out[label == idx] = color
    out[label == ignore_index] = (0, 0, 0)
    return out
