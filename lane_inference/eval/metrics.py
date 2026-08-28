"""Segmentation metrics for the Phase 4 quantitative evaluation."""
from __future__ import annotations

import cv2
import numpy as np

import config as cfg


class ConfusionMatrix:
    """Streaming confusion matrix over NUM_CLASSES, ignoring IGNORE_INDEX."""

    def __init__(self, num_classes: int = cfg.NUM_CLASSES):
        self.n = num_classes
        self.mat = np.zeros((num_classes, num_classes), dtype=np.int64)

    def update(self, pred: np.ndarray, target: np.ndarray) -> None:
        pred = np.asarray(pred).ravel()
        target = np.asarray(target).ravel()
        keep = (target != cfg.IGNORE_INDEX) & (target < self.n)
        if not keep.any():
            return
        idx = target[keep].astype(np.int64) * self.n + pred[keep].astype(np.int64)
        self.mat += np.bincount(idx, minlength=self.n ** 2).reshape(self.n, self.n)

    def reset(self) -> None:
        self.mat[:] = 0


    @property
    def tp(self) -> np.ndarray:
        return np.diag(self.mat).astype(np.float64)

    @property
    def fp(self) -> np.ndarray:
        return self.mat.sum(axis=0).astype(np.float64) - self.tp

    @property
    def fn(self) -> np.ndarray:
        return self.mat.sum(axis=1).astype(np.float64) - self.tp

    def iou(self) -> np.ndarray:
        denom = self.tp + self.fp + self.fn
        return np.divide(self.tp, denom, out=np.full(self.n, np.nan), where=denom > 0)

    def dice(self) -> np.ndarray:
        denom = 2 * self.tp + self.fp + self.fn
        return np.divide(2 * self.tp, denom, out=np.full(self.n, np.nan), where=denom > 0)

    def precision(self) -> np.ndarray:
        denom = self.tp + self.fp
        return np.divide(self.tp, denom, out=np.full(self.n, np.nan), where=denom > 0)

    def recall(self) -> np.ndarray:
        denom = self.tp + self.fn
        return np.divide(self.tp, denom, out=np.full(self.n, np.nan), where=denom > 0)

    def pixel_accuracy(self) -> float:
        total = self.mat.sum()
        return float(self.tp.sum() / total) if total else float("nan")

    def summary(self) -> dict:
        iou, dice = self.iou(), self.dice()
        prec, rec = self.precision(), self.recall()
        return {
            "per_class_iou": iou,
            "per_class_dice": dice,
            "per_class_precision": prec,
            "per_class_recall": rec,
            "miou": float(np.nanmean(iou)),
            "mdice": float(np.nanmean(dice)),
            "pixel_accuracy": self.pixel_accuracy(),
            "mean_accuracy": float(np.nanmean(rec)),
            "drivable_iou": float(iou[cfg.DRIVABLE_ID]),
            "drivable_precision": float(prec[cfg.DRIVABLE_ID]),
            "drivable_recall": float(rec[cfg.DRIVABLE_ID]),
            "drivable_dice": float(dice[cfg.DRIVABLE_ID]),
        }


def _boundary(mask_bool: np.ndarray) -> np.ndarray:
    """1-pixel contour of a binary mask via morphological gradient."""
    m = mask_bool.astype(np.uint8)
    k = np.ones((3, 3), np.uint8)
    return (cv2.dilate(m, k) - cv2.erode(m, k)).astype(bool)


def boundary_f1(pred_bool: np.ndarray, gt_bool: np.ndarray, tolerance: int = 3) -> tuple:
    """Boundary F1 (Csurka et al.). A predicted boundary pixel counts as correct if"""
    pb, gb = _boundary(pred_bool), _boundary(gt_bool)
    n_pred, n_gt = int(pb.sum()), int(gb.sum())
    if n_pred == 0 or n_gt == 0:
        return 0, n_pred, 0, n_gt

    k = np.ones((2 * tolerance + 1, 2 * tolerance + 1), np.uint8)
    gb_dil = cv2.dilate(gb.astype(np.uint8), k).astype(bool)
    pb_dil = cv2.dilate(pb.astype(np.uint8), k).astype(bool)
    return int((pb & gb_dil).sum()), n_pred, int((gb & pb_dil).sum()), n_gt


class BoundaryAccumulator:
    """Accumulates boundary-F1 counts for one binary class across a dataset."""

    def __init__(self, tolerance: int = 3):
        self.tol = tolerance
        self.tp_pred = self.n_pred = self.tp_gt = self.n_gt = 0

    def update(self, pred_bool: np.ndarray, gt_bool: np.ndarray) -> None:
        a, b, c, d = boundary_f1(pred_bool, gt_bool, self.tol)
        self.tp_pred += a
        self.n_pred += b
        self.tp_gt += c
        self.n_gt += d

    def score(self) -> dict:
        p = self.tp_pred / self.n_pred if self.n_pred else float("nan")
        r = self.tp_gt / self.n_gt if self.n_gt else float("nan")
        f = 2 * p * r / (p + r) if (p + r) else float("nan")
        return {"boundary_precision": p, "boundary_recall": r, "boundary_f1": f}


def format_table(summary: dict, title: str = "") -> str:
    """Render a per-class metric table as fixed-width text for the report."""
    lines = []
    if title:
        lines += [title, "=" * len(title)]
    lines.append(f"{'class':<18}{'IoU':>9}{'Dice':>9}{'Prec':>9}{'Recall':>9}")
    lines.append("-" * 54)
    for i, name in enumerate(cfg.CLASS_NAMES):
        lines.append(
            f"{name:<18}"
            f"{summary['per_class_iou'][i]:>9.4f}"
            f"{summary['per_class_dice'][i]:>9.4f}"
            f"{summary['per_class_precision'][i]:>9.4f}"
            f"{summary['per_class_recall'][i]:>9.4f}"
        )
    lines.append("-" * 54)
    lines.append(f"{'mean':<18}{summary['miou']:>9.4f}{summary['mdice']:>9.4f}")
    lines.append(f"pixel accuracy   : {summary['pixel_accuracy']:.4f}")
    lines.append(f"mean class acc.  : {summary['mean_accuracy']:.4f}")
    return "\n".join(lines)


def markdown_row(name: str, summary: dict, extra: dict | None = None) -> str:
    """One row of the Phase 4 ablation table, in Markdown."""
    e = extra or {}
    return (
        f"| {name} | {summary['drivable_iou']:.4f} | {summary['drivable_precision']:.4f} "
        f"| {summary['drivable_recall']:.4f} | {summary['miou']:.4f} "
        f"| {e.get('boundary_f1', float('nan')):.4f} | {e.get('fps', float('nan')):.1f} |"
    )


MARKDOWN_HEADER = (
    "| Configuration | Drivable IoU | Precision | Recall | mIoU | Boundary F1 | FPS |\n"
    "|---|---|---|---|---|---|---|"
)
