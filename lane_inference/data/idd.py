"""IDD-Lite dataset loader (Indian Driving Dataset, level-1 label hierarchy)."""
from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

import config as cfg

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def list_split(root: Path, split: str) -> list[tuple[Path, Path | None]]:
    """Return (image_path, label_path|None) pairs for a split, sorted."""
    img_dir = Path(root) / "leftImg8bit" / split
    lbl_dir = Path(root) / "gtFine" / split
    if not img_dir.is_dir():
        raise FileNotFoundError(f"No such split directory: {img_dir}")

    pairs: list[tuple[Path, Path | None]] = []
    for img_path in sorted(img_dir.glob("*/*_image.jpg")):
        drive = img_path.parent.name
        stem = img_path.name.replace("_image.jpg", "")
        lbl_path = lbl_dir / drive / f"{stem}_label.png"
        pairs.append((img_path, lbl_path if lbl_path.is_file() else None))
    return pairs


def _rand_scale_crop(img, masks, out_hw, rng):
    """Random scale in [0.85, 1.35] then crop back to out_hw."""
    H, W = out_hw
    s = rng.uniform(0.85, 1.35)
    nh, nw = max(H, int(round(H * s))), max(W, int(round(W * s)))
    img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    masks = [cv2.resize(m, (nw, nh), interpolation=cv2.INTER_NEAREST) for m in masks]
    y0 = rng.randint(0, nh - H) if nh > H else 0
    x0 = rng.randint(0, nw - W) if nw > W else 0
    return (img[y0:y0 + H, x0:x0 + W],
            [m[y0:y0 + H, x0:x0 + W] for m in masks])


def _color_jitter(img, rng):
    """Brightness / contrast / saturation / hue jitter in HSV space."""
    img = img.astype(np.float32)
    img *= rng.uniform(0.75, 1.25)
    mean = img.mean()
    img = (img - mean) * rng.uniform(0.75, 1.30) + mean
    img = np.clip(img, 0, 255).astype(np.uint8)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int16)
    hsv[..., 0] = (hsv[..., 0] + rng.randint(-6, 7)) % 180
    hsv[..., 1] = np.clip(hsv[..., 1] * rng.uniform(0.7, 1.3), 0, 255)
    return cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)


def _random_gamma(img, rng):
    """Simulate the over/under-exposure failure mode (Phase 3 Example 4)."""
    gamma = rng.uniform(0.55, 1.75)
    lut = np.clip(((np.arange(256) / 255.0) ** gamma) * 255.0, 0, 255).astype(np.uint8)
    return cv2.LUT(img, lut)


def _random_shadow(img, rng):
    """Cast 1-2 dark convex polygons (Phase 3 Example 5: shadow failures)."""
    h, w = img.shape[:2]
    overlay = np.ones((h, w), dtype=np.float32)
    for _ in range(rng.randint(1, 3)):
        n = rng.randint(3, 6)
        pts = np.array(
            [[rng.randint(-w // 4, w + w // 4), rng.randint(0, h)] for _ in range(n)],
            dtype=np.int32,
        )
        shade = np.ones((h, w), dtype=np.float32)
        cv2.fillConvexPoly(shade, cv2.convexHull(pts), float(rng.uniform(0.45, 0.80)))
        overlay = np.minimum(overlay, shade)
    overlay = cv2.GaussianBlur(overlay, (0, 0), sigmaX=max(3.0, w * 0.02))
    return np.clip(img.astype(np.float32) * overlay[..., None], 0, 255).astype(np.uint8)


def _vignette(img, rng):
    """Simulate the radial falloff of a wide-angle lens."""
    h, w = img.shape[:2]
    Y, X = np.ogrid[:h, :w]
    cx, cy = 0.5 + rng.uniform(-.05, .05), rng.uniform(.42, .58)
    rx, ry = rng.uniform(.52, .78), rng.uniform(.55, .85)
    r = np.sqrt(((X / w - cx) / rx) ** 2 + ((Y / h - cy) / ry) ** 2)
    strength = rng.uniform(1.1, 2.0)
    m = np.clip(1.0 - strength * np.clip(r - rng.uniform(.35, .6), 0, None), 0.0, 1.0)
    return np.clip(img.astype(np.float32) * m[..., None], 0, 255).astype(np.uint8)


def _low_light(img, rng):
    """Dusk and night. Not simply darker: the response is non-linear, colour"""
    x = img.astype(np.float32) / 255.0
    x = x ** rng.uniform(1.3, 2.4)
    x *= rng.uniform(0.45, 0.8)
    tint = np.array([rng.uniform(.9, 1.15), rng.uniform(.92, 1.06),
                     rng.uniform(.88, 1.18)], np.float32)
    x *= tint
    nrng = np.random.default_rng(rng.randrange(2 ** 32))
    x = x + nrng.normal(0, rng.uniform(.006, .022), x.shape)
    return np.clip(x * 255, 0, 255).astype(np.uint8)


def _barrel(img, masks, rng):
    """Barrel distortion, as a wide-angle lens produces. Applied to the image and"""
    h, w = img.shape[:2]
    k = rng.uniform(0.06, 0.28)
    cx, cy = w / 2, h / 2
    Y, X = np.indices((h, w), dtype=np.float32)
    nx, ny = (X - cx) / cx, (Y - cy) / cy
    r2 = nx * nx + ny * ny
    f = 1 + k * r2
    mx = (nx * f * cx + cx).astype(np.float32)
    my = (ny * f * cy + cy).astype(np.float32)
    img = cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    masks = [cv2.remap(m, mx, my, cv2.INTER_NEAREST,
                       borderMode=cv2.BORDER_CONSTANT,
                       borderValue=int(cfg.IGNORE_INDEX) if m.dtype == np.uint8 and m.max() > 1 else 0)
             for m in masks]
    return img, masks


def _quality_degrade(img, rng):
    """Downscale-upscale and/or motion blur (Phase 3 Example 3: low quality)."""
    h, w = img.shape[:2]
    if rng.random() < 0.6:
        f = rng.uniform(0.35, 0.75)
        small = cv2.resize(img, (max(8, int(w * f)), max(8, int(h * f))),
                           interpolation=cv2.INTER_AREA)
        img = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    if rng.random() < 0.5:
        k = rng.choice([3, 5, 7])
        kernel = np.zeros((k, k), np.float32)
        if rng.random() < 0.5:
            kernel[k // 2, :] = 1.0 / k
        else:
            kernel[:, k // 2] = 1.0 / k
        img = cv2.filter2D(img, -1, kernel)
    return img


class IDDLite(Dataset):
    """IDD-Lite semantic segmentation dataset."""

    def __init__(
        self,
        root: str | Path = cfg.DATA_ROOT,
        split: str = "train",
        input_size: tuple[int, int] = (224, 320),
        augment: bool | None = None,
        seed: int = 1337,
        load_lane: bool = False,
        dashcam_aug: bool = True,
    ):
        self.root = Path(root)
        self.split = split
        self.input_size = input_size
        self.augment = (split == "train") if augment is None else augment
        self.load_lane = load_lane
        self.dashcam_aug = dashcam_aug
        self.pairs = list_split(self.root, split)
        if not self.pairs:
            raise RuntimeError(f"No images found for split '{split}' under {self.root}")
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.pairs)

    def _lane_path(self, img_path: Path) -> Path:
        return (self.root / "laneFine" / self.split / img_path.parent.name /
                img_path.name.replace("_image.jpg", "_lane.png"))

    def _load_lane(self, img_path: Path, shape) -> np.ndarray:
        p = self._lane_path(img_path)
        if not p.is_file():
            return np.zeros(shape, dtype=np.uint8)
        m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        return np.zeros(shape, np.uint8) if m is None else (m > 127).astype(np.uint8)

    def _load(self, idx):
        img_path, lbl_path = self.pairs[idx]
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Failed to read image: {img_path}")
        if lbl_path is None:
            lbl = np.full(img.shape[:2], cfg.IGNORE_INDEX, dtype=np.uint8)
        else:
            lbl = cv2.imread(str(lbl_path), cv2.IMREAD_GRAYSCALE)
            if lbl is None:
                raise RuntimeError(f"Failed to read label: {lbl_path}")
        return img, lbl

    def __getitem__(self, idx):
        img, lbl = self._load(idx)
        masks = [lbl]
        if self.load_lane:
            masks.append(self._load_lane(self.pairs[idx][0], lbl.shape))

        H, W = self.input_size
        rng = random.Random(self.rng.random() * 1e9) if self.augment else None

        if self.augment:
            img, masks = _rand_scale_crop(img, masks, (H, W), rng)
            if rng.random() < 0.5:
                img = img[:, ::-1].copy()
                masks = [m[:, ::-1].copy() for m in masks]
            if rng.random() < 0.8:
                img = _color_jitter(img, rng)
            if rng.random() < 0.35:
                img = _random_gamma(img, rng)
            if rng.random() < 0.30:
                img = _random_shadow(img, rng)
            if rng.random() < 0.25:
                img = _quality_degrade(img, rng)


            if self.dashcam_aug:
                if rng.random() < 0.30:
                    img = _low_light(img, rng)
                if rng.random() < 0.30:
                    img = _vignette(img, rng)
                if rng.random() < 0.20:
                    img, masks = _barrel(img, masks, rng)
        else:
            img = cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)
            masks = [cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
                     for m in masks]

        x = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        x = (x - IMAGENET_MEAN) / IMAGENET_STD
        x = torch.from_numpy(x.transpose(2, 0, 1).copy())
        y = torch.from_numpy(masks[0].astype(np.int64).copy())
        if self.load_lane:
            lane = torch.from_numpy(masks[1].astype(np.float32).copy())
            return x, y, lane
        return x, y


    def raw(self, idx):
        """Return the un-normalised, resized (BGR image, label) pair."""
        img, lbl = self._load(idx)
        H, W = self.input_size
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)
        lbl = cv2.resize(lbl, (W, H), interpolation=cv2.INTER_NEAREST)
        return img, lbl


def preprocess_bgr(img_bgr: np.ndarray, input_size: tuple[int, int]) -> torch.Tensor:
    """Normalise an arbitrary BGR frame into a 1x3xHxW model input tensor."""
    H, W = input_size
    resized = cv2.resize(img_bgr, (W, H), interpolation=cv2.INTER_LINEAR)
    x = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(x.transpose(2, 0, 1).copy()).unsqueeze(0)


def class_frequencies(root=cfg.DATA_ROOT, split="train", limit=None) -> np.ndarray:
    """Pixel counts per class - used to derive inverse-frequency loss weights."""
    counts = np.zeros(cfg.NUM_CLASSES, dtype=np.int64)
    pairs = list_split(Path(root), split)
    if limit:
        pairs = pairs[:limit]
    for _, lbl_path in pairs:
        if lbl_path is None:
            continue
        lbl = cv2.imread(str(lbl_path), cv2.IMREAD_GRAYSCALE)
        valid = lbl[lbl != cfg.IGNORE_INDEX]
        counts += np.bincount(valid.ravel(), minlength=cfg.NUM_CLASSES)[:cfg.NUM_CLASSES]
    return counts
