"""Classical computer-vision road-surface detection."""
from __future__ import annotations

import cv2
import numpy as np

import config as cfg


def detect_asphalt_legacy(image: np.ndarray) -> np.ndarray:
    """Fixed-threshold HSV asphalt mask exactly as implemented in Phase 3."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(cfg.HSV_LOWER), np.array(cfg.HSV_UPPER))
    kernel = np.ones((cfg.MORPH_KERNEL_SIZE, cfg.MORPH_KERNEL_SIZE), np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def _seed_region(h: int, w: int) -> np.ndarray:
    """Trapezoid covering the road immediately in front of the camera."""
    seed = np.zeros((h, w), np.uint8)
    poly = np.array([[
        (int(0.30 * w), h - 1),
        (int(0.70 * w), h - 1),
        (int(0.60 * w), int(0.82 * h)),
        (int(0.40 * w), int(0.82 * h)),
    ]], dtype=np.int32)
    cv2.fillPoly(seed, poly, 255)
    return seed


def _local_variance(gray: np.ndarray, ksize: int = 7) -> np.ndarray:
    """Local intensity variance, used as a texture measure."""
    g = gray.astype(np.float32)
    mean = cv2.boxFilter(g, -1, (ksize, ksize), normalize=True)
    sq = cv2.boxFilter(g * g, -1, (ksize, ksize), normalize=True)
    return np.maximum(sq - mean * mean, 0.0)


def detect_asphalt(
    image: np.ndarray,
    return_confidence: bool = False,
    chroma_tol: float = 2.5,
    light_tol: float = 2.5,
    texture_percentile: float = 82.0,
):
    """Adaptive classical road-surface detector."""
    h, w = image.shape[:2]


    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    L = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(L)
    Lf, Af, Bf = L.astype(np.float32), A.astype(np.float32), B.astype(np.float32)


    seed = _seed_region(h, w) > 0
    mu_a, sd_a = float(Af[seed].mean()), float(Af[seed].std()) + 1.5
    mu_b, sd_b = float(Bf[seed].mean()), float(Bf[seed].std()) + 1.5
    mu_l, sd_l = float(Lf[seed].mean()), float(Lf[seed].std()) + 6.0

    d_chroma = np.sqrt(((Af - mu_a) / sd_a) ** 2 + ((Bf - mu_b) / sd_b) ** 2)
    d_light = np.abs(Lf - mu_l) / sd_l


    conf_chroma = np.clip(1.0 - d_chroma / chroma_tol, 0.0, 1.0)
    conf_light = np.clip(1.0 - d_light / light_tol, 0.0, 1.0)


    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    var = _local_variance(cv2.GaussianBlur(gray, (3, 3), 0))
    thr = float(np.percentile(var, texture_percentile)) + 1e-6
    conf_texture = np.clip(1.0 - var / (2.0 * thr), 0.0, 1.0)

    conf = (conf_chroma ** 0.5) * (conf_light ** 0.25) * (conf_texture ** 0.25)
    conf = cv2.GaussianBlur(conf.astype(np.float32), (0, 0), sigmaX=max(1.0, w / 160.0))

    mask = (conf > 0.35).astype(np.uint8) * 255
    k = np.ones((cfg.MORPH_KERNEL_SIZE, cfg.MORPH_KERNEL_SIZE), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = keep_bottom_connected(mask)

    return (mask, conf) if return_confidence else mask


def keep_bottom_connected(mask: np.ndarray, band_ratio: float = 0.06,
                          low_frac: float = 0.55, area_frac: float = 0.15) -> np.ndarray:
    """Keep the connected components that plausibly belong to the road ahead."""
    h, w = mask.shape[:2]
    n, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
    if n <= 1:
        return mask

    band_top = int(h * (1.0 - band_ratio))
    touching = set(np.unique(labels[band_top:, :])) - {0}


    low = []
    for lab in range(1, n):
        bottom = stats[lab, cv2.CC_STAT_TOP] + stats[lab, cv2.CC_STAT_HEIGHT]
        if bottom >= h * low_frac:
            low.append((lab, stats[lab, cv2.CC_STAT_AREA]))

    keep = set(touching)
    if low:
        biggest = max(a for _, a in low)
        keep |= {lab for lab, a in low if a >= area_frac * biggest}
    if not keep:
        keep = {1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))}

    out = np.zeros_like(mask)
    for lab in keep:
        out[labels == lab] = 255
    return out
