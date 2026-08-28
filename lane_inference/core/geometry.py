"""Scene geometry: vanishing-point estimation and inverse perspective mapping."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

import config as cfg


def robust_line_fit(y: np.ndarray, x: np.ndarray, iterations: int = 3,
                    trim: float = 0.25) -> tuple[float, float] | None:
    """Fit x = a*y + b with iterative trimming of the worst-fitting points."""
    if len(y) < 4:
        return None
    y = np.asarray(y, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)

    keep = np.ones(len(y), dtype=bool)
    a = b = 0.0
    for _ in range(iterations):
        if keep.sum() < 3:
            break
        a, b = np.polyfit(y[keep], x[keep], 1)
        resid = np.abs(a * y + b - x)
        cutoff = np.quantile(resid[keep], 1.0 - trim)
        new_keep = resid <= max(cutoff, 1e-6)
        if new_keep.sum() < 3:
            break
        keep = new_keep
    if keep.sum() < 3:
        return None
    a, b = np.polyfit(y[keep], x[keep], 1)
    return float(a), float(b)


def road_edge_points(mask: np.ndarray, row_step: int = 2, min_run: int = 4,
                     border_margin: int = 2):
    """Extract the left and right boundary of the road for each scanline."""
    h, w = mask.shape[:2]
    binary = mask > 0
    rows, lefts, rights = [], [], []

    for y in range(0, h, row_step):
        xs = np.flatnonzero(binary[y])
        if xs.size < min_run:
            continue
        breaks = np.flatnonzero(np.diff(xs) > 1)
        starts = np.concatenate([[0], breaks + 1])
        ends = np.concatenate([breaks, [xs.size - 1]])
        lengths = ends - starts
        k = int(np.argmax(lengths))
        if lengths[k] < min_run:
            continue
        rows.append(y)
        lefts.append(xs[starts[k]])
        rights.append(xs[ends[k]])

    rows = np.array(rows)
    lefts = np.array(lefts)
    rights = np.array(rights)
    left_valid = lefts > border_margin
    right_valid = rights < (w - 1 - border_margin)
    return rows, lefts, rights, left_valid, right_valid


def bridge_occlusions(drivable: np.ndarray, occluders: np.ndarray | None,
                      min_cover: float = 0.6, max_gap_ratio: float = 0.45) -> np.ndarray:
    """Reconnect road segments that a vehicle or pedestrian has split apart."""
    if occluders is None:
        return drivable.copy()

    h, w = drivable.shape[:2]
    road = drivable > 0
    occ = occluders > 0
    out = road.copy()
    max_gap = int(max_gap_ratio * w)

    for y in range(h):
        xs = np.flatnonzero(road[y])
        if xs.size < 2:
            continue
        breaks = np.flatnonzero(np.diff(xs) > 1)
        if breaks.size == 0:
            continue
        starts = np.concatenate([[0], breaks + 1])
        ends = np.concatenate([breaks, [xs.size - 1]])
        for i in range(len(starts) - 1):
            a = xs[ends[i]] + 1
            b = xs[starts[i + 1]]
            gap = b - a
            if gap <= 0 or gap > max_gap:
                continue
            if occ[y, a:b].mean() >= min_cover:
                out[y, a:b] = True

    return out.astype(np.uint8) * 255


@dataclass
class VanishingPoint:
    x: float
    y: float
    confidence: float
    source: str

    def as_tuple(self) -> tuple[float, float]:
        return (self.x, self.y)


def vp_from_road_mask(mask: np.ndarray,
                      max_pitch_deg: float = 15.0) -> VanishingPoint | None:
    """Estimate the vanishing point as the intersection of the two road edges."""
    h, w = mask.shape[:2]
    rows, lefts, rights, lv, rv = road_edge_points(mask)
    if len(rows) < cfg.DEFAULT.geometry.vp_min_road_rows:
        return None

    min_pts = max(4, cfg.DEFAULT.geometry.vp_min_road_rows // 2)
    if lv.sum() < min_pts or rv.sum() < min_pts:
        return None

    fit_l = robust_line_fit(rows[lv], lefts[lv])
    fit_r = robust_line_fit(rows[rv], rights[rv])
    if fit_l is None or fit_r is None:
        return None

    al, bl = fit_l
    ar, br = fit_r
    denom = al - ar
    if abs(denom) < 0.05:
        return None

    y_vp = (br - bl) / denom
    x_vp = al * y_vp + bl


    if not (-0.25 * h <= y_vp <= rows.min() + 0.05 * h):
        return None
    if not (-0.5 * w <= x_vp <= 1.5 * w):
        return None


    focal = (w / 2.0) / np.tan(np.radians(cfg.DEFAULT.geometry.fov_deg) / 2.0)
    pitch_deg = abs(np.degrees(np.arctan2(h / 2.0 - y_vp, focal)))
    if pitch_deg > max_pitch_deg:
        return None

    span = (rows.max() - rows.min()) / h
    convergence = min(1.0, abs(denom) / 0.35)
    coverage = min(1.0, (lv.sum() + rv.sum()) / (2.0 * len(rows)))
    conf = float(np.clip(0.4 * span + 0.4 * convergence + 0.2 * coverage, 0.0, 1.0))
    return VanishingPoint(float(x_vp), float(y_vp), conf, "road-mask")


def vp_from_line_segments(image: np.ndarray, mask: np.ndarray | None = None,
                          max_segments: int = 200) -> VanishingPoint | None:
    """Secondary estimator: RANSAC over long line segments (lane markings, kerbs,"""
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image

    detector = cv2.createLineSegmentDetector() if hasattr(cv2, "createLineSegmentDetector") else None
    if detector is not None:
        found = detector.detect(gray)[0]
        segs = found.reshape(-1, 4) if found is not None else np.empty((0, 4))
    else:  # pragma: no cover - OpenCV builds without LSD
        edges = cv2.Canny(gray, 60, 180)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 60, minLineLength=h * 0.12,
                                maxLineGap=12)
        segs = lines.reshape(-1, 4) if lines is not None else np.empty((0, 4))

    if len(segs) == 0:
        return None

    x1, y1, x2, y2 = segs[:, 0], segs[:, 1], segs[:, 2], segs[:, 3]
    length = np.hypot(x2 - x1, y2 - y1)
    angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))


    keep = (length > 0.06 * h) & (np.abs(angle) > 12) & (np.abs(angle) < 78)
    if mask is not None:

        near = cv2.dilate((mask > 0).astype(np.uint8),
                          np.ones((15, 15), np.uint8)) > 0
        mid_y = np.clip(((y1 + y2) / 2).astype(int), 0, h - 1)
        mid_x = np.clip(((x1 + x2) / 2).astype(int), 0, w - 1)
        keep &= near[mid_y, mid_x]

    segs, length = segs[keep], length[keep]
    if len(segs) < 4:
        return None
    if len(segs) > max_segments:
        idx = np.argsort(-length)[:max_segments]
        segs, length = segs[idx], length[idx]


    p1 = np.stack([segs[:, 0], segs[:, 1], np.ones(len(segs))], axis=1)
    p2 = np.stack([segs[:, 2], segs[:, 3], np.ones(len(segs))], axis=1)
    lines = np.cross(p1, p2)
    norm = np.linalg.norm(lines[:, :2], axis=1, keepdims=True)
    lines = lines / np.maximum(norm, 1e-9)

    rng = np.random.default_rng(0)
    best, best_score = None, -1.0
    n = len(lines)
    for _ in range(300):
        i, j = rng.choice(n, 2, replace=False)
        p = np.cross(lines[i], lines[j])
        if abs(p[2]) < 1e-9:
            continue
        p = p / p[2]
        if not (-1.0 * w <= p[0] <= 2.0 * w and -0.5 * h <= p[1] <= 0.95 * h):
            continue
        dist = np.abs(lines @ p)
        score = float((length * np.exp(-(dist ** 2) / (2 * (0.02 * h) ** 2))).sum())
        if score > best_score:
            best_score, best = score, p

    if best is None:
        return None
    conf = float(np.clip(best_score / max(length.sum(), 1e-6), 0.0, 1.0))
    return VanishingPoint(float(best[0]), float(best[1]), conf, "line-segments")


def fuse_vanishing_points(primary: VanishingPoint | None,
                          secondary: VanishingPoint | None) -> VanishingPoint | None:
    """Confidence-weighted average, but only when the two estimates agree."""
    if primary is None:
        return secondary
    if secondary is None:
        return primary
    if np.hypot(primary.x - secondary.x, primary.y - secondary.y) > 0.15 * 640:
        return primary
    wp, ws = primary.confidence, secondary.confidence
    total = max(wp + ws, 1e-6)
    return VanishingPoint(
        (wp * primary.x + ws * secondary.x) / total,
        (wp * primary.y + ws * secondary.y) / total,
        min(1.0, 0.5 * (wp + ws) + 0.15),
        "fused",
    )


class VanishingPointTracker:
    """Exponential-moving-average smoothing of the vanishing point over time."""

    def __init__(self, alpha: float | None = None):
        self.alpha = alpha if alpha is not None else cfg.DEFAULT.geometry.vp_smooth_alpha
        self.state: VanishingPoint | None = None
        self.misses = 0

    def update(self, vp: VanishingPoint | None) -> VanishingPoint | None:
        if vp is None:
            self.misses += 1
            return self.state
        if self.state is None:
            self.state = vp
        else:
            a = self.alpha * max(vp.confidence, 0.25)
            self.state = VanishingPoint(
                (1 - a) * self.state.x + a * vp.x,
                (1 - a) * self.state.y + a * vp.y,
                0.7 * self.state.confidence + 0.3 * vp.confidence,
                "smoothed",
            )
        self.misses = 0
        return self.state

    def reset(self) -> None:
        self.state = None
        self.misses = 0


def default_vanishing_point(shape: tuple[int, int]) -> VanishingPoint:
    """Fallback prior: frame centre, slightly above mid-height."""
    h, w = shape[:2]
    return VanishingPoint(w / 2.0, h * 0.48, 0.1, "prior")


@dataclass
class IPMTransform:
    """Ground-plane rectification derived from a vanishing point."""
    image_shape: tuple[int, int]
    focal: float
    principal_x: float
    principal_y: float
    pitch: float
    camera_height: float
    bev_size: tuple[int, int]
    near_m: float
    far_m: float
    lateral_m: float
    H_img_from_ground: np.ndarray
    H_img_from_bev: np.ndarray
    H_bev_from_img: np.ndarray


    @classmethod
    def from_vanishing_point(cls, vp: VanishingPoint, image_shape: tuple[int, int],
                             geom: "cfg.GeometryConfig" = None) -> "IPMTransform":
        geom = geom or cfg.DEFAULT.geometry
        h, w = image_shape[:2]

        focal = (w / 2.0) / np.tan(np.radians(geom.fov_deg) / 2.0)
        cy = h / 2.0
        cx = float(vp.x)


        pitch = float(np.arctan2(cy - vp.y, focal))
        pitch = float(np.clip(pitch, np.radians(-25.0), np.radians(35.0)))

        s, c = np.sin(pitch), np.cos(pitch)
        Hc = geom.camera_height_m


        M = np.array([
            [focal, cx * c,               cx * s * Hc],
            [0.0,   -focal * s + cy * c,  focal * c * Hc + cy * s * Hc],
            [0.0,   c,                    s * Hc],
        ], dtype=np.float64)

        bh, bw = geom.bev_height, geom.bev_width
        L, near, far = geom.bev_lateral_m, geom.bev_near_m, geom.bev_forward_m


        G = np.array([
            [L / (bw - 1), 0.0,                     -L / 2.0],
            [0.0,          -(far - near) / (bh - 1), far],
            [0.0,          0.0,                      1.0],
        ], dtype=np.float64)

        H_img_from_bev = M @ G
        H_bev_from_img = np.linalg.inv(H_img_from_bev)

        return cls(
            image_shape=(h, w), focal=float(focal), principal_x=cx, principal_y=cy,
            pitch=pitch, camera_height=Hc, bev_size=(bh, bw),
            near_m=near, far_m=far, lateral_m=L,
            H_img_from_ground=M, H_img_from_bev=H_img_from_bev,
            H_bev_from_img=H_bev_from_img,
        )


    @property
    def px_per_m_lateral(self) -> float:
        return (self.bev_size[1] - 1) / self.lateral_m

    @property
    def px_per_m_forward(self) -> float:
        return (self.bev_size[0] - 1) / (self.far_m - self.near_m)

    @property
    def pitch_degrees(self) -> float:
        return float(np.degrees(self.pitch))


    def warp_image(self, image: np.ndarray) -> np.ndarray:
        bh, bw = self.bev_size
        return cv2.warpPerspective(image, self.H_bev_from_img, (bw, bh),
                                   flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    def warp_mask(self, mask: np.ndarray) -> np.ndarray:
        bh, bw = self.bev_size
        out = cv2.warpPerspective((mask > 0).astype(np.uint8) * 255,
                                  self.H_bev_from_img, (bw, bh),
                                  flags=cv2.INTER_NEAREST,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        return out

    def unwarp_image(self, bev: np.ndarray) -> np.ndarray:
        h, w = self.image_shape
        return cv2.warpPerspective(bev, self.H_img_from_bev, (w, h),
                                   flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=0)


    @staticmethod
    def _apply(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
        homo = np.concatenate([pts, np.ones((len(pts), 1))], axis=1)
        out = homo @ H.T
        w = np.where(np.abs(out[:, 2:3]) < 1e-12, 1e-12, out[:, 2:3])
        return out[:, :2] / w

    def bev_to_ground(self, pts) -> np.ndarray:
        """(col, row) -> (X, Y) in metres."""
        pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
        X = -self.lateral_m / 2.0 + pts[:, 0] * self.lateral_m / (self.bev_size[1] - 1)
        Y = self.far_m - pts[:, 1] * (self.far_m - self.near_m) / (self.bev_size[0] - 1)
        return np.stack([X, Y], axis=1)

    def ground_to_bev(self, pts) -> np.ndarray:
        """(X, Y) in metres -> (col, row)."""
        pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
        col = (pts[:, 0] + self.lateral_m / 2.0) * (self.bev_size[1] - 1) / self.lateral_m
        row = (self.far_m - pts[:, 1]) * (self.bev_size[0] - 1) / (self.far_m - self.near_m)
        return np.stack([col, row], axis=1)

    def bev_to_image(self, pts) -> np.ndarray:
        return self._apply(self.H_img_from_bev, pts)

    def image_to_bev(self, pts) -> np.ndarray:
        return self._apply(self.H_bev_from_img, pts)

    def ground_to_image(self, pts) -> np.ndarray:
        return self._apply(self.H_img_from_ground, pts)


    def rescaled(self, factor: float) -> "IPMTransform":
        """Return an equivalent transform with the camera height scaled."""
        geom = cfg.GeometryConfig(
            bev_width=self.bev_size[1], bev_height=self.bev_size[0],
            bev_lateral_m=self.lateral_m, bev_forward_m=self.far_m,
            bev_near_m=self.near_m, camera_height_m=self.camera_height * factor,
            fov_deg=float(np.degrees(2 * np.arctan((self.image_shape[1] / 2) / self.focal))),
        )
        vp = VanishingPoint(self.principal_x,
                            self.principal_y - self.focal * np.tan(self.pitch),
                            1.0, "rescale")
        return IPMTransform.from_vanishing_point(vp, self.image_shape, geom)

    def calibrate_from_known_width(self, measured_m: float,
                                   true_m: float) -> "IPMTransform":
        """Re-derive the camera height from a carriageway of known true width."""
        if measured_m <= 1e-6:
            return self
        return self.rescaled(true_m / measured_m)


def build_ipm(mask: np.ndarray, image: np.ndarray | None = None,
              tracker: VanishingPointTracker | None = None,
              use_line_refinement: bool = True,
              geom: "cfg.GeometryConfig" = None) -> tuple[IPMTransform, VanishingPoint]:
    """Convenience constructor: estimate the vanishing point from a road mask"""
    vp = vp_from_road_mask(mask)
    if use_line_refinement and image is not None:
        vp = fuse_vanishing_points(vp, vp_from_line_segments(image, mask))
    if tracker is not None:
        vp = tracker.update(vp)
    if vp is None:
        vp = default_vanishing_point(mask.shape)
    return IPMTransform.from_vanishing_point(vp, mask.shape[:2], geom), vp
