"""Reference implementations of the Phase 3 pipeline, used as ablation baselines."""
from __future__ import annotations

import cv2
import numpy as np

import config as cfg
from core.asphalt_detector import detect_asphalt, detect_asphalt_legacy
from utils.image_utils import keep_largest_blob


def phase3_cv_only(image: np.ndarray) -> np.ndarray:
    """Phase 3 with the DL branch disabled."""
    mask = detect_asphalt_legacy(image)
    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = (mask > 0).astype(np.uint8) * 255
    return keep_largest_blob(mask)


def phase3_hybrid(image: np.ndarray, dl_mask: np.ndarray) -> np.ndarray:
    """Phase 3 hybrid fusion, reproduced exactly."""
    cv_mask = detect_asphalt_legacy(image)
    dl_ratio = cv2.countNonZero(dl_mask) / (dl_mask.shape[0] * dl_mask.shape[1])
    mask = cv_mask if dl_ratio < 0.01 else cv2.bitwise_or(cv_mask, dl_mask)

    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = (mask > 0).astype(np.uint8) * 255
    return keep_largest_blob(mask)


def phase4_cv_only(image: np.ndarray) -> np.ndarray:
    """Phase 4 improved classical detector, no deep learning."""
    return detect_asphalt(image)


def const_all_road(image: np.ndarray) -> np.ndarray:
    """Predict every pixel as drivable - upper bound on recall, floor on IoU."""
    return np.full(image.shape[:2], 255, np.uint8)


def bottom_trapezoid_prior(image: np.ndarray) -> np.ndarray:
    """Fixed trapezoidal road prior with no image evidence at all."""
    h, w = image.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    poly = np.array([[
        (int(0.02 * w), h),
        (int(0.98 * w), h),
        (int(0.62 * w), int(0.52 * h)),
        (int(0.38 * w), int(0.52 * h)),
    ]], dtype=np.int32)
    cv2.fillPoly(mask, poly, 255)
    return mask
