"""
Rendering: lane overlays, rectified-view panels and the diagnostic dashboard.
"""
from __future__ import annotations

import cv2
import numpy as np

import config as cfg
from core.geometry import IPMTransform
from core.lane_tracker import LaneModel


LANE_COLORS = [
    (180, 119, 31), (14, 127, 255), (44, 160, 44), (40, 39, 214),
    (189, 103, 148), (75, 86, 140), (194, 119, 227), (207, 190, 23),
]
EGO_COLOR = (90, 220, 90)
BOUNDARY_COLOR = (255, 255, 255)


def legacy_build_lane_overlay(original: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Phase 3 visualisation: per row, take the widest run of road and paint the"""
    height, width = mask.shape
    overlay = original.copy()
    for y in range(int(height * 0.5), height):
        xs = np.where(mask[y] == 255)[0]
        if len(xs) < cfg.MIN_SEGMENT_WIDTH:
            continue
        segments, start, prev = [], xs[0], xs[0]
        for x in xs[1:]:
            if x - prev > 2:
                segments.append((start, prev))
                start = x
            prev = x
        segments.append((start, prev))
        left, right = max(segments, key=lambda s: s[1] - s[0])
        centre = int((left + right) / 2)
        cv2.line(overlay, (left, y), (centre, y), (0, 255, 0), 1)
        cv2.line(overlay, (centre, y), (right, y), (0, 0, 255), 1)
    return overlay


def lane_boundary_curves(model: LaneModel, ipm: IPMTransform,
                         n_samples: int = 40, y_max: float | None = None):
    """Sample every lane boundary and project it into the image."""
    if model.left_coeffs is None:
        return []
    y_far = min(y_max if y_max is not None else ipm.far_m, ipm.far_m)


    if model.support_far_m is not None:
        y_far = min(y_far, float(model.support_far_m))
    if y_far <= ipm.near_m:
        return []
    Ys = np.linspace(ipm.near_m, y_far, n_samples)

    n_bounds = len(model.boundary_offsets)
    curves = [[] for _ in range(n_bounds)]
    h, w = ipm.image_shape
    for Y in Ys:
        b = model.boundaries_at(float(Y))
        if len(b) != n_bounds:
            continue
        pts = ipm.ground_to_image(np.stack([b, np.full(n_bounds, Y)], axis=1))
        for i, (u, v) in enumerate(pts):
            if -2 * w < u < 3 * w and -h < v < 2 * h:
                curves[i].append((u, v))
    return [np.array(c, dtype=np.float32) for c in curves if len(c) >= 2]


def draw_lane_overlay(image: np.ndarray, model: LaneModel, ipm: IPMTransform,
                      alpha: float = 0.40, draw_boundaries: bool = True,
                      y_max: float | None = None) -> np.ndarray:
    """Fill each inferred lane and highlight the ego lane."""
    curves = lane_boundary_curves(model, ipm, y_max=y_max)
    if len(curves) < 2:
        return image.copy()

    layer = image.copy()
    for i in range(len(curves) - 1):
        left, right = curves[i], curves[i + 1]
        n = min(len(left), len(right))
        poly = np.concatenate([left[:n], right[:n][::-1]]).astype(np.int32)
        colour = EGO_COLOR if i == model.ego_lane else LANE_COLORS[i % len(LANE_COLORS)]
        cv2.fillPoly(layer, [poly], colour)

    out = cv2.addWeighted(image, 1.0 - alpha, layer, alpha, 0.0)

    if draw_boundaries:
        for i, c in enumerate(curves):
            edge = (i == 0) or (i == len(curves) - 1)
            cv2.polylines(out, [c.astype(np.int32)], False, BOUNDARY_COLOR,
                          2 if edge else 1, cv2.LINE_AA)
    return out


def draw_bev_panel(bev_image: np.ndarray, model: LaneModel | None,
                   ipm: IPMTransform, grid: bool = True) -> np.ndarray:
    """Rectified view annotated with a metre grid and the inferred lanes."""
    panel = bev_image.copy()
    if panel.ndim == 2:
        panel = cv2.cvtColor(panel, cv2.COLOR_GRAY2BGR)
    bh, bw = panel.shape[:2]

    if grid:
        step = 5.0
        Y = float(np.ceil(ipm.near_m / step) * step)
        while Y <= ipm.far_m:
            r = int(round(ipm.ground_to_bev([[0.0, Y]])[0][1]))
            if 0 <= r < bh:
                cv2.line(panel, (0, r), (bw - 1, r), (70, 70, 70), 1)
                cv2.putText(panel, f"{int(Y)}m", (4, r - 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.34, (170, 170, 170), 1)
            Y += step
        for X in np.arange(-6, 6.1, 3.0):
            c = int(round(ipm.ground_to_bev([[X, 0.0]])[0][0]))
            if 0 <= c < bw:
                cv2.line(panel, (c, 0), (c, bh - 1), (55, 55, 55), 1)

    if model is not None and model.left_coeffs is not None:

        y_far = ipm.far_m
        if model.support_far_m is not None:
            y_far = min(y_far, float(model.support_far_m))
        Ys = np.linspace(ipm.near_m, max(y_far, ipm.near_m + 1e-3), 60)
        n_bounds = len(model.boundary_offsets)
        cols = [[] for _ in range(n_bounds)]
        for Y in Ys:
            b = model.boundaries_at(float(Y))
            if len(b) != n_bounds:
                continue
            pts = ipm.ground_to_bev(np.stack([b, np.full(n_bounds, Y)], axis=1))
            for i, p in enumerate(pts):
                cols[i].append(p)
        for i, c in enumerate(cols):
            if len(c) < 2:
                continue
            arr = np.array(c, dtype=np.int32)
            edge = (i == 0) or (i == n_bounds - 1)
            cv2.polylines(panel, [arr], False,
                          (255, 255, 255) if edge else (0, 230, 255),
                          2 if edge else 1, cv2.LINE_AA)
        for X in model.marking_positions:
            c = int(round(ipm.ground_to_bev([[float(X), 0.0]])[0][0]))
            if 0 <= c < bw:
                cv2.line(panel, (c, bh - 30), (c, bh - 5), (0, 140, 255), 2)

    return panel


def draw_hud(image: np.ndarray, model: LaneModel | None, ipm: IPMTransform,
             extra: dict | None = None, top: int = 4) -> np.ndarray:
    """Compact text readout in the top-left corner."""
    out = image.copy()
    lines = []
    if model is None:
        lines.append("lane inference: no solution")
    else:
        ego = f"{model.ego_lane + 1}/{model.n_lanes}" if model.ego_lane >= 0 else "?"
        lines += [
            f"{model.road_type}  x{model.n_lanes} lanes",
            f"carriageway {model.carriageway_width_m:.1f} m",
            f"lane {model.lane_width_m:.2f} m   ego {ego}",
            f"conf {model.confidence:.2f}  [{model.source}]",
        ]
    lines.append(f"pitch {ipm.pitch_degrees:+.1f} deg")
    for k, v in (extra or {}).items():
        lines.append(f"{k} {v}")

    pad, lh = 6, 15
    box_w = max(int(6.6 * max(len(s) for s in lines)) + 2 * pad, 150)
    box_h = lh * len(lines) + 2 * pad
    box_w = min(box_w, out.shape[1] - 8)
    box_h = min(box_h, out.shape[0] - top - 2)
    panel = out[top:top + box_h, 4:4 + box_w].copy()
    panel = (panel * 0.35).astype(np.uint8)
    out[top:top + box_h, 4:4 + box_w] = panel
    for i, s in enumerate(lines):
        y = top + pad + lh * (i + 1) - 4
        if y >= out.shape[0]:
            break
        cv2.putText(out, s, (4 + pad, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def stack_panels(panels: list[np.ndarray], height: int = 360,
                 gap: int = 6, labels: list[str] | None = None) -> np.ndarray:
    """Resize panels to a common height and lay them out horizontally."""
    resized = []
    for i, p in enumerate(panels):
        if p.ndim == 2:
            p = cv2.cvtColor(p, cv2.COLOR_GRAY2BGR)
        scale = height / p.shape[0]
        r = cv2.resize(p, (max(1, int(round(p.shape[1] * scale))), height))
        if labels and i < len(labels):
            cv2.rectangle(r, (0, 0), (r.shape[1], 18), (0, 0, 0), -1)
            cv2.putText(r, labels[i], (5, 13), cv2.FONT_HERSHEY_SIMPLEX,
                        0.42, (255, 255, 255), 1, cv2.LINE_AA)
        resized.append(r)
        if i < len(panels) - 1:
            resized.append(np.zeros((height, gap, 3), np.uint8))
    return np.concatenate(resized, axis=1)
