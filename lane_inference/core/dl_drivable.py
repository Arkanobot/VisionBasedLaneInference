"""Deep-learning inference: drivable-area and semantic segmentation."""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

import config as cfg
from data.idd import IMAGENET_MEAN, IMAGENET_STD, preprocess_bgr


@dataclass
class SegmentationResult:
    """Output of a semantic segmentation backend, at the input image's size."""
    labels: np.ndarray
    probs: np.ndarray
    latency_ms: float
    lane_prob: np.ndarray | None = None

    def lane_mask(self, threshold: float = 0.5) -> np.ndarray | None:
        if self.lane_prob is None:
            return None
        return (self.lane_prob > threshold).astype(np.uint8) * 255

    def mask(self, class_id: int) -> np.ndarray:
        return (self.labels == class_id).astype(np.uint8) * 255

    def confidence(self, class_id: int) -> np.ndarray:
        return self.probs[class_id]

    @property
    def drivable(self) -> np.ndarray:
        return self.mask(cfg.DRIVABLE_ID)

    @property
    def drivable_confidence(self) -> np.ndarray:
        return self.probs[cfg.DRIVABLE_ID]


class SegmenterBackend:
    """Wraps the IDD-Lite-trained segmentation network."""

    def __init__(self, checkpoint: str | Path | None = None,
                 device: str | None = None, half: bool = False,
                 tta: bool = False):
        from models.segnet import build_model

        self.device = device or cfg.resolve_device()
        ckpt_path = Path(checkpoint) if checkpoint else cfg.default_checkpoint()
        if not ckpt_path.is_file():
            raise FileNotFoundError(
                f"No checkpoint at {ckpt_path}. Train one with "
                f"`python -m models.train`, or run the Colab notebook and copy "
                f"seg_mnv3/best.pt into outputs/checkpoints/.")

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        self.input_size = tuple(ckpt.get("input_size", cfg.DEFAULT.train.input_size))
        self.num_classes = int(ckpt.get("num_classes", cfg.NUM_CLASSES))
        self.has_lane_head = bool(ckpt.get("lane_head", False))
        self.tta = tta
        self.model = build_model(ckpt.get("arch", "mobilenetv3_unet"),
                                 num_classes=self.num_classes, pretrained=False,
                                 lane_head=self.has_lane_head)
        self.model.load_state_dict(ckpt["model"])
        self.model.to(self.device).eval()
        self.half = half and self.device == "cuda"
        if self.half:
            self.model.half()
        self.checkpoint_path = ckpt_path
        self.trained_miou = float(ckpt.get("miou", float("nan")))
        self._warmup()

    def _warmup(self, n: int = 3) -> None:
        H, W = self.input_size
        x = torch.zeros(1, 3, H, W, device=self.device,
                        dtype=torch.half if self.half else torch.float32)
        with torch.no_grad():
            for _ in range(n):
                self.model(x)
        self._sync()

    def _sync(self) -> None:
        if self.device == "mps":
            torch.mps.synchronize()
        elif self.device == "cuda":
            torch.cuda.synchronize()

    def _forward(self, x):
        """Run the model, optionally averaging over a horizontal flip."""
        out = self.model(x)
        seg, lane = out if isinstance(out, tuple) else (out, None)
        if not self.tta:
            return seg, lane
        out_f = self.model(torch.flip(x, dims=[3]))
        seg_f, lane_f = out_f if isinstance(out_f, tuple) else (out_f, None)
        seg = 0.5 * (seg.softmax(1) + torch.flip(seg_f, dims=[3]).softmax(1))
        if lane is not None and lane_f is not None:
            lane = 0.5 * (lane.sigmoid() + torch.flip(lane_f, dims=[3]).sigmoid())
        return seg, lane

    @torch.no_grad()
    def __call__(self, image_bgr: np.ndarray) -> SegmentationResult:
        h, w = image_bgr.shape[:2]
        x = preprocess_bgr(image_bgr, self.input_size).to(self.device)
        if self.half:
            x = x.half()

        t0 = time.perf_counter()
        seg, lane = self._forward(x)
        self._sync()
        latency = (time.perf_counter() - t0) * 1000.0


        probs = (seg if self.tta else seg.softmax(1)).float()[0].cpu().numpy()
        if (probs.shape[1], probs.shape[2]) != (h, w):
            probs = np.stack([cv2.resize(p, (w, h), interpolation=cv2.INTER_LINEAR)
                              for p in probs])
        labels = probs.argmax(0).astype(np.int64)

        lane_prob = None
        if lane is not None:
            lp = (lane if self.tta else lane.sigmoid()).float()[0, 0].cpu().numpy()
            if lp.shape != (h, w):
                lp = cv2.resize(lp, (w, h), interpolation=cv2.INTER_LINEAR)
            lane_prob = lp.astype(np.float32)

        return SegmentationResult(labels, probs.astype(np.float32), latency, lane_prob)


def letterbox(image: np.ndarray, new_shape=(384, 640), color=(114, 114, 114)):
    """Resize preserving aspect ratio and pad, as YOLOP expects."""
    h, w = image.shape[:2]
    nh, nw = new_shape
    r = min(nh / h, nw / w)
    rh, rw = int(round(h * r)), int(round(w * r))
    resized = cv2.resize(image, (rw, rh), interpolation=cv2.INTER_LINEAR)
    top = (nh - rh) // 2
    bottom = nh - rh - top
    left = (nw - rw) // 2
    right = nw - rw - left
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                cv2.BORDER_CONSTANT, value=color)
    return padded, r, (left, top)


class YolopBackend:
    """Pretrained YOLOP, wired correctly. Used as an ablation reference."""

    def __init__(self, device: str | None = None,
                 letterbox_input: bool = True, normalise: bool = True):
        self.device = device or cfg.resolve_device()

        if self.device == "mps":
            self.device = "cpu"
        self.letterbox_input = letterbox_input
        self.normalise = normalise
        self.model = torch.hub.load("hustvl/YOLOP", "yolop",
                                    pretrained=True, trust_repo=True)
        self.model.to(self.device).eval()
        self.input_shape = (384, 640)

    @torch.no_grad()
    def __call__(self, image_bgr: np.ndarray):
        """Returns (drivable_mask, lane_line_mask, latency_ms)."""
        h, w = image_bgr.shape[:2]
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        if self.letterbox_input:
            padded, ratio, (dx, dy) = letterbox(rgb, self.input_shape)
        else:
            padded = cv2.resize(rgb, (self.input_shape[1], self.input_shape[0]))
            ratio, dx, dy = None, 0, 0

        x = padded.astype(np.float32) / 255.0
        if self.normalise:
            x = (x - IMAGENET_MEAN) / IMAGENET_STD
        x = torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0).to(self.device)

        t0 = time.perf_counter()
        _, da_out, ll_out = self.model(x)
        latency = (time.perf_counter() - t0) * 1000.0

        def to_mask(out):
            m = out.softmax(1)[0, 1].cpu().numpy()
            m = (m > 0.5).astype(np.uint8) * 255
            if self.letterbox_input:
                nh, nw = self.input_shape
                m = m[dy:nh - dy if dy else nh, dx:nw - dx if dx else nw]
            return cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)

        return to_mask(da_out), to_mask(ll_out), latency


class LegacyYolopBackend(YolopBackend):
    """Exactly the Phase 3 wrapper: stretched resize, no normalisation, CPU."""

    def __init__(self):
        torch.set_num_threads(2)
        super().__init__(device="cpu", letterbox_input=False, normalise=False)


def fuse_masks(cv_confidence: np.ndarray, dl_confidence: np.ndarray,
               fusion: "cfg.FusionConfig" = None) -> tuple[np.ndarray, np.ndarray]:
    """Blend the classical and learned road estimates by confidence."""
    f = fusion or cfg.DEFAULT.fusion
    dl_trust = float(np.clip(dl_confidence.mean() / max(f.dl_trust_floor, 1e-6), 0.0, 1.0))
    w_dl = f.dl_weight * dl_trust
    w_cv = f.cv_weight + f.dl_weight * (1.0 - dl_trust)
    total = max(w_dl + w_cv, 1e-6)

    fused = (w_dl * dl_confidence + w_cv * cv_confidence) / total
    mask = (fused > f.decision_threshold).astype(np.uint8) * 255
    return mask, fused.astype(np.float32)


def gated_fallback(cv_confidence: np.ndarray, dl_confidence: np.ndarray,
                   fusion: "cfg.FusionConfig" = None,
                   min_coverage: float = 0.04) -> tuple[np.ndarray, np.ndarray, str]:
    """Use the learned branch, falling back to the classical branch only when the"""
    f = fusion or cfg.DEFAULT.fusion
    dl_mask = dl_confidence > f.decision_threshold
    coverage = float(dl_mask.mean())

    if coverage >= min_coverage:
        return (dl_mask.astype(np.uint8) * 255, dl_confidence.astype(np.float32),
                "learned")
    cv_mask = cv_confidence > 0.35
    return (cv_mask.astype(np.uint8) * 255, cv_confidence.astype(np.float32),
            "classical-fallback")
