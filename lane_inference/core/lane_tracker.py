"""Lane inference."""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import cv2
import numpy as np

import config as cfg


PROFILE_MODE = os.environ.get("PROFILE_MODE", "merge")
from core.geometry import IPMTransform, bridge_occlusions


def legacy_find_road_base(mask: np.ndarray):
    """Phase 3 road-base detection."""
    height, width = mask.shape
    bottom = mask[int(height * (1 - cfg.BOTTOM_REGION_RATIO)):height, :]
    histogram = np.sum(bottom, axis=0)
    road_columns = np.where(histogram > 255 * 5)[0]
    if len(road_columns) < cfg.MIN_ROAD_COLUMNS:
        return None, None
    return road_columns[0], road_columns[-1]


def legacy_lane_split(mask: np.ndarray):
    """Phase 3 lane estimate: per row, take the widest contiguous run of road and"""
    height, width = mask.shape
    left_pts, mid_pts, right_pts = [], [], []
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
        left_pts.append((int(left), y))
        right_pts.append((int(right), y))
        mid_pts.append((int((left + right) / 2), y))
    return left_pts, mid_pts, right_pts


@dataclass
class RoadProfile:
    """Per-forward-distance carriageway measurements, in metres."""
    Y: np.ndarray
    left: np.ndarray
    right: np.ndarray
    left_valid: np.ndarray
    right_valid: np.ndarray
    width: np.ndarray

    def __len__(self) -> int:
        return len(self.Y)

    @property
    def valid(self) -> np.ndarray:
        """Samples where the full width is measurable."""
        return self.left_valid & self.right_valid

    @property
    def n_valid(self) -> int:
        return int(self.valid.sum())


def bev_footprint(ipm: IPMTransform, erode_px: int = 2) -> np.ndarray:
    """The region of the rectified canvas that actually came from image pixels."""
    h, w = ipm.image_shape
    full = np.full((h, w), 255, np.uint8)
    fp = cv2.warpPerspective(full, ipm.H_bev_from_img,
                             (ipm.bev_size[1], ipm.bev_size[0]),
                             flags=cv2.INTER_NEAREST,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    if erode_px > 0:
        k = np.ones((erode_px * 2 + 1, erode_px * 2 + 1), np.uint8)
        fp = cv2.erode(fp, k)
    return fp


def extract_road_profile(bev_mask: np.ndarray, ipm: IPMTransform,
                         step_m: float = 0.5, min_run_px: int = 3,
                         footprint: np.ndarray | None = None) -> RoadProfile:
    """Measure the carriageway edges at regular forward distances."""
    bh, bw = bev_mask.shape[:2]
    binary = bev_mask > 0
    if footprint is None:
        footprint = bev_footprint(ipm)
    inside = footprint > 0

    Ys = np.arange(ipm.near_m + 1.0, ipm.far_m - 1.0 + 1e-9, step_m)
    left = np.full(len(Ys), np.nan)
    right = np.full(len(Ys), np.nan)
    lvalid = np.zeros(len(Ys), dtype=bool)
    rvalid = np.zeros(len(Ys), dtype=bool)

    for i, Y in enumerate(Ys):
        row = int(round(ipm.ground_to_bev([[0.0, Y]])[0][1]))
        if not (0 <= row < bh):
            continue
        xs = np.flatnonzero(binary[row])
        if xs.size < min_run_px:
            continue
        breaks = np.flatnonzero(np.diff(xs) > 1)
        starts = np.concatenate([[0], breaks + 1])
        ends = np.concatenate([breaks, [xs.size - 1]])
        k = int(np.argmax(ends - starts))
        a, b = int(xs[starts[k]]), int(xs[ends[k]])
        if (b - a) < min_run_px:
            continue

        gl, gr = ipm.bev_to_ground([[a, row], [b, row]])
        left[i], right[i] = gl[0], gr[0]


        lvalid[i] = bool(a > 1 and (a - 1) >= 0 and inside[row, a - 1])
        rvalid[i] = bool(b < bw - 2 and (b + 1) < bw and inside[row, b + 1])

    width = right - left
    return RoadProfile(Ys, left, right, lvalid, rvalid, width)


def merge_profiles(raw: "RoadProfile", bridged: "RoadProfile",
                   max_gain_m: float | None = None) -> "RoadProfile":
    """Take each row's measurement from the unbridged mask where it is available."""
    if max_gain_m is None:
        max_gain_m = cfg.DEFAULT.geometry.lane_width_urban_m
    out_left = raw.left.copy()
    out_right = raw.right.copy()
    lv = raw.left_valid.copy()
    rv = raw.right_valid.copy()

    both_raw = raw.valid
    both_br = bridged.valid
    for i in range(len(raw)):
        if both_raw[i] and both_br[i]:
            if (bridged.width[i] - raw.width[i]) <= max_gain_m:
                out_left[i], out_right[i] = bridged.left[i], bridged.right[i]
        elif both_br[i] and not both_raw[i]:
            out_left[i], out_right[i] = bridged.left[i], bridged.right[i]
            lv[i] = rv[i] = True
        else:
            if bridged.left_valid[i] and not lv[i]:
                out_left[i], lv[i] = bridged.left[i], True
            if bridged.right_valid[i] and not rv[i]:
                out_right[i], rv[i] = bridged.right[i], True

    return RoadProfile(Y=raw.Y, left=out_left, right=out_right,
                       left_valid=lv, right_valid=rv,
                       width=out_right - out_left)


def fit_edge_polynomial(Y: np.ndarray, X: np.ndarray, degree: int = 2,
                        iterations: int = 3, trim: float = 0.2):
    """Robustly fit X = f(Y) with iterative trimming. Returns numpy poly coeffs."""
    ok = np.isfinite(Y) & np.isfinite(X)
    if ok.sum() < degree + 2:
        return None
    y, x = Y[ok], X[ok]
    keep = np.ones(len(y), bool)
    coeffs = None
    for _ in range(iterations):
        if keep.sum() < degree + 2:
            break
        coeffs = np.polyfit(y[keep], x[keep], degree)
        resid = np.abs(np.polyval(coeffs, y) - x)
        cutoff = np.quantile(resid[keep], 1.0 - trim)
        nk = resid <= max(cutoff, 1e-6)
        if nk.sum() < degree + 2:
            break
        keep = nk
    return coeffs


def detect_lane_markings(bev_image: np.ndarray, bev_mask: np.ndarray,
                         ipm: IPMTransform, marking_width_m: float = 0.15,
                         min_peak_frac: float = 0.35):
    """Locate painted lane markings in the rectified view."""
    if bev_image.ndim == 3:
        gray = cv2.cvtColor(bev_image, cv2.COLOR_BGR2GRAY)
    else:
        gray = bev_image
    gray = gray.astype(np.int16)

    d = max(2, int(round(marking_width_m * ipm.px_per_m_lateral)))
    left_shift = np.roll(gray, d, axis=1)
    right_shift = np.roll(gray, -d, axis=1)
    tophat = np.minimum(gray - left_shift, gray - right_shift)
    tophat[:, :d] = 0
    tophat[:, -d:] = 0

    road = (bev_mask > 0)
    if road.sum() < 50:
        return np.empty(0), np.empty(0)

    resp = np.where(road, np.clip(tophat, 0, None), 0).astype(np.float32)
    scale = np.percentile(resp[road], 99.0)
    if scale < 4.0:
        return np.empty(0), np.empty(0)
    resp = np.clip(resp / scale, 0.0, 1.0)

    column = resp.sum(axis=0) / np.maximum(road.sum(axis=0), 1)
    column = cv2.GaussianBlur(column.reshape(1, -1).astype(np.float32),
                              (0, 0), sigmaX=1.2).ravel()
    if column.max() <= 1e-6:
        return np.empty(0), np.empty(0)

    norm = column / column.max()
    min_sep = max(3, int(round(1.8 * ipm.px_per_m_lateral)))
    peaks = []
    order = np.argsort(-norm)
    for c in order:
        if norm[c] < min_peak_frac:
            break
        if all(abs(c - p) >= min_sep for p in peaks):
            peaks.append(int(c))
    if not peaks:
        return np.empty(0), np.empty(0)

    peaks = np.array(sorted(peaks))
    rows = np.full(len(peaks), (bev_mask.shape[0] - 1) / 2.0)
    X = ipm.bev_to_ground(np.stack([peaks, rows], axis=1))[:, 0]
    return X, norm[peaks]


def lane_positions_from_mask(bev_lane: np.ndarray, bev_mask: np.ndarray,
                             ipm: IPMTransform, min_peak_frac: float = 0.30,
                             min_separation_m: float = 2.0):
    """Recover lane-marking positions from a learned lane-line mask."""
    road = bev_mask > 0
    lane = (bev_lane > 0) & road
    if lane.sum() < 20:
        return np.empty(0), np.empty(0)

    column = lane.sum(axis=0).astype(np.float32) / np.maximum(road.sum(axis=0), 1)
    column = cv2.GaussianBlur(column.reshape(1, -1), (0, 0), sigmaX=1.5).ravel()
    if column.max() <= 1e-6:
        return np.empty(0), np.empty(0)

    norm = column / column.max()
    min_sep = max(3, int(round(min_separation_m * ipm.px_per_m_lateral)))
    peaks = []
    for c in np.argsort(-norm):
        if norm[c] < min_peak_frac:
            break
        if all(abs(int(c) - p) >= min_sep for p in peaks):
            peaks.append(int(c))
    if not peaks:
        return np.empty(0), np.empty(0)

    peaks = np.array(sorted(peaks))
    rows = np.full(len(peaks), (bev_mask.shape[0] - 1) / 2.0)
    X = ipm.bev_to_ground(np.stack([peaks, rows], axis=1))[:, 0]
    return X, norm[peaks]


@dataclass
class LaneModel:
    n_lanes: int
    lane_width_m: float
    carriageway_width_m: float
    left_coeffs: np.ndarray | None
    right_coeffs: np.ndarray | None
    boundary_offsets: np.ndarray
    ego_lane: int
    road_type: str
    confidence: float
    source: str = "geometry"
    width_is_lower_bound: bool = False
    quality: dict = field(default_factory=dict)


    support_far_m: float | None = None
    marking_positions: np.ndarray = field(default_factory=lambda: np.empty(0))

    def boundaries_at(self, Y: float) -> np.ndarray:
        """Lateral positions (metres) of every lane boundary at distance Y."""
        if self.left_coeffs is None or self.right_coeffs is None:
            return np.empty(0)
        xl = float(np.polyval(self.left_coeffs, Y))
        xr = float(np.polyval(self.right_coeffs, Y))
        w = xr - xl
        if w <= 0 or self.carriageway_width_m <= 1e-6:
            return np.empty(0)

        return xl + self.boundary_offsets * (w / self.carriageway_width_m)

    def centre_at(self, Y: float) -> float:
        b = self.boundaries_at(Y)
        return float(0.5 * (b[0] + b[-1])) if len(b) >= 2 else float("nan")


def infer_lanes(bev_mask: np.ndarray, ipm: IPMTransform,
                bev_image: np.ndarray | None = None,
                geom: "cfg.GeometryConfig" = None,
                use_markings: bool = True,
                bev_occluders: np.ndarray | None = None,
                bev_lane: np.ndarray | None = None,
                vp_confidence: float = 0.5) -> LaneModel | None:
    """Infer the lane structure of the carriageway from its rectified mask."""
    geom = geom or cfg.DEFAULT.geometry
    profile = extract_road_profile(bev_mask, ipm)
    if bev_occluders is not None and PROFILE_MODE != "raw":
        bridged = bridge_occlusions(bev_mask, bev_occluders)
        bprof = extract_road_profile(bridged, ipm)
        profile = bprof if PROFILE_MODE == "bridge" else merge_profiles(profile, bprof)

    lv, rv = profile.left_valid, profile.right_valid
    if lv.sum() < 4 or rv.sum() < 4:
        return None


    left_c = fit_edge_polynomial(profile.Y[lv], profile.left[lv])
    right_c = fit_edge_polynomial(profile.Y[rv], profile.right[rv])
    if left_c is None or right_c is None:
        return None

    _sup = profile.Y[lv | rv]
    support_far_m = float(_sup.max()) if _sup.size else None

    both = profile.valid
    widths = profile.width[both]
    widths = widths[np.isfinite(widths)]


    width_is_lower_bound = len(widths) < 4
    if width_is_lower_bound:
        fallback = profile.width[np.isfinite(profile.width)]
        if len(fallback) < 4:
            return None
        widths = fallback
    W = float(np.median(widths))


    max_width = geom.max_lanes * geom.lane_width_urban_m + 4.0
    if not (geom.min_lane_width_m <= W <= max_width):
        return None


    def _fit_residual(Yv, Xv, coeffs):
        if coeffs is None or len(Yv) < 4:
            return 3.0
        return float(np.median(np.abs(np.polyval(coeffs, Yv) - Xv)))

    residual = max(_fit_residual(profile.Y[lv], profile.left[lv], left_c),
                   _fit_residual(profile.Y[rv], profile.right[rv], right_c))

    width_cv = float(np.std(widths) / max(np.mean(widths), 1e-6))
    both_valid_frac = profile.valid.sum() / max(len(profile), 1)

    c_res = float(np.clip(1.0 - residual / 1.2, 0.0, 1.0))
    c_vp = float(np.clip(vp_confidence, 0.0, 1.0))
    c_bvf = float(np.clip(both_valid_frac, 0.0, 1.0))
    c_cv = float(np.clip(1.0 - width_cv / 0.6, 0.0, 1.0))

    geom_conf = float(np.clip(0.40 * c_res + 0.25 * c_vp
                              + 0.20 * c_bvf + 0.15 * c_cv, 0.02, 1.0))
    quality = {"res": c_res, "vp": c_vp, "bvf": c_bvf, "cv": c_cv}
    coverage = max(lv.sum(), rv.sum()) / max(len(profile), 1)
    if width_is_lower_bound:
        geom_conf *= 0.5


    ref_Y = 0.5 * (ipm.near_m + min(ipm.far_m, 22.0))
    xl_ref = float(np.polyval(left_c, ref_Y))
    xr_ref = float(np.polyval(right_c, ref_Y))
    W_ref = xr_ref - xl_ref
    if not (geom.min_lane_width_m <= W_ref <= max_width):


        if not (geom.min_lane_width_m <= W <= max_width):
            return None
        centre = 0.5 * (xl_ref + xr_ref)
        xl_ref, xr_ref = centre - W / 2.0, centre + W / 2.0
        W_ref = W

    def dividers(marks: np.ndarray) -> np.ndarray:
        """Reduce raw marking detections to interior lane dividers."""
        if not len(marks):
            return np.empty(0)
        tol = max(1.0, 0.5 * geom.min_lane_width_m)
        interior = np.sort(marks[(marks > xl_ref + tol) & (marks < xr_ref - tol)])
        if not len(interior):
            return np.empty(0)
        merged = [interior[0]]
        for x in interior[1:]:
            if x - merged[-1] < 0.6 * geom.min_lane_width_m:
                merged[-1] = 0.5 * (merged[-1] + x)
            else:
                merged.append(x)
        return np.array(merged)

    def build(marks, strengths, source):
        """Return a structured LaneModel if these markings are plausible."""
        inside = dividers(marks)
        if len(inside) < 1:
            return None
        edges = np.concatenate([[xl_ref], inside, [xr_ref]])
        gaps = np.diff(edges)
        if gaps.min() < geom.min_lane_width_m or gaps.max() > 6.5:
            return None
        strength = float(np.mean(strengths)) if len(strengths) else 0.5
        model = LaneModel(
            n_lanes=len(gaps), lane_width_m=float(np.mean(gaps)),
            carriageway_width_m=W_ref, left_coeffs=left_c, right_coeffs=right_c,
            boundary_offsets=edges - xl_ref, ego_lane=-1, road_type="structured",
            confidence=float(np.clip(0.85 * geom_conf + 0.15 * strength, 0, 1)),
            source=f"markings:{source}", marking_positions=inside,
            width_is_lower_bound=width_is_lower_bound, quality=quality,
            support_far_m=support_far_m,
        )
        model.ego_lane = _ego_lane_index(model, ipm)
        return model


    marks = np.empty(0)
    if use_markings:
        candidates = []
        if bev_lane is not None:
            candidates.append(("learned", lane_positions_from_mask(bev_lane, bev_mask, ipm)))
        if bev_image is not None:
            candidates.append(("top-hat", detect_lane_markings(bev_image, bev_mask, ipm)))
        for source, (m, st) in candidates:
            if len(m) and marks.size == 0:
                marks = m
            model = build(m, st, source)
            if model is not None:
                return model


    nominal = geom.lane_width_urban_m if W_ref >= 6.0 else geom.lane_width_rural_m
    n = int(np.clip(round(W_ref / nominal), 1, geom.max_lanes))


    while n > 1 and (W_ref / n) < geom.min_lane_width_m:
        n -= 1

    offsets = np.linspace(0.0, W_ref, n + 1)

    model = LaneModel(
        n_lanes=n, lane_width_m=W_ref / n, carriageway_width_m=W_ref,
        left_coeffs=left_c, right_coeffs=right_c, boundary_offsets=offsets,
        ego_lane=-1, road_type="unstructured",
        confidence=geom_conf,
        source="width-partition", marking_positions=marks,
        width_is_lower_bound=width_is_lower_bound, quality=quality,
        support_far_m=support_far_m,
    )
    model.ego_lane = _ego_lane_index(model, ipm)
    return model


def _ego_lane_index(model: LaneModel, ipm: IPMTransform) -> int:
    """Which lane the camera is in."""
    Y = ipm.near_m + 2.0
    b = model.boundaries_at(Y)
    if len(b) < 2:
        return -1
    idx = int(np.searchsorted(b, 0.0) - 1)
    return idx if 0 <= idx < model.n_lanes else -1


class LaneKalmanFilter:
    """Kalman filter over the lane model's geometric parameters."""

    def __init__(self, cfg_tracker: "cfg.TrackerConfig" = None):
        t = cfg_tracker or cfg.DEFAULT.tracker
        self.dim = 4
        self.x = None
        self.P = np.eye(self.dim) * t.init_uncertainty


        q, r = t.process_noise, t.measurement_noise
        self.Q = np.diag([q, q, q * 10.0, q * 0.05])
        self.R = np.diag([r, r, r * 10.0, r * 20.0])
        self.max_missed = t.max_missed_frames
        self.missed = 0

    @staticmethod
    def _measure(model: LaneModel) -> np.ndarray:
        centre = 0.5 * (model.left_coeffs + model.right_coeffs)
        c2, c1, c0 = centre[0], centre[1], centre[2]
        return np.array([c0, c1, c2, model.carriageway_width_m], dtype=np.float64)

    def update(self, model: LaneModel | None) -> np.ndarray | None:
        if model is None or model.left_coeffs is None:
            self.missed += 1
            if self.missed > self.max_missed:
                self.reset()
                return None
            if self.x is not None:
                self.P = self.P + self.Q
            return self.x

        z = self._measure(model)
        if self.x is None:
            self.x = z.copy()
            self.missed = 0
            return self.x


        self.P = self.P + self.Q
        conf = max(model.confidence, 0.05)
        R = self.R / conf
        S = self.P + R
        K = self.P @ np.linalg.inv(S)
        self.x = self.x + K @ (z - self.x)
        self.P = (np.eye(self.dim) - K) @ self.P
        self.missed = 0
        return self.x

    def apply(self, model: LaneModel) -> LaneModel:
        """Rewrite a lane model's geometry from the current filter state."""
        if self.x is None:
            return model
        c0, c1, c2, w = self.x
        centre = np.array([c2, c1, c0])
        half = np.array([0.0, 0.0, w / 2.0])
        model.left_coeffs = centre - half
        model.right_coeffs = centre + half
        scale = w / max(model.carriageway_width_m, 1e-6)
        model.boundary_offsets = model.boundary_offsets * scale
        model.carriageway_width_m = float(w)
        model.lane_width_m = float(w / max(model.n_lanes, 1))
        model.source = model.source + "+kalman"
        return model

    def reset(self) -> None:
        self.x = None
        self.P = np.eye(self.dim) * cfg.DEFAULT.tracker.init_uncertainty
        self.missed = 0


class LaneCountStabiliser:
    """Majority vote over recent lane counts."""

    def __init__(self, window: int = 9):
        self.window = window
        self.history: list[int] = []

    def update(self, n: int | None) -> int | None:
        if n is not None:
            self.history.append(int(n))
            if len(self.history) > self.window:
                self.history.pop(0)
        if not self.history:
            return None
        vals, counts = np.unique(self.history, return_counts=True)
        return int(vals[int(np.argmax(counts))])

    def reset(self) -> None:
        self.history.clear()
