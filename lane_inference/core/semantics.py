"""Scene analytics built on top of the inferred lane structure."""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

import config as cfg
from core.geometry import IPMTransform
from core.lane_tracker import LaneModel


@dataclass
class RoadUser:
    """A detected road user, positioned on the ground plane."""
    x_m: float
    y_m: float
    width_m: float
    lane: int
    pixel_area: int
    bbox: tuple
    class_id: int = cfg.VEHICLE_ID
    track_id: int = -1
    speed_y: float = float("nan")


def _ground_contact(comp: np.ndarray, band_frac: float = 0.15) -> tuple[int, int, int] | None:
    """Locate a road user's footprint: (row, first column, last column)."""
    rows = np.flatnonzero(comp.any(axis=1))
    if rows.size == 0:
        return None
    top, bottom = int(rows.min()), int(rows.max())
    height = bottom - top + 1
    band_start = max(top, bottom - max(1, int(round(band_frac * height))) + 1)

    best_row, best_span = None, -1
    for r in range(band_start, bottom + 1):
        cols = np.flatnonzero(comp[r])
        if cols.size < 2:
            continue
        span = int(cols[-1] - cols[0])
        if span > best_span:
            best_row, best_span = r, span
    if best_row is None:
        return None
    cols = np.flatnonzero(comp[best_row])
    return best_row, int(cols[0]), int(cols[-1])


def extract_road_users(labels: np.ndarray, ipm: IPMTransform,
                       lane_model: LaneModel | None,
                       class_ids=(cfg.VEHICLE_ID, cfg.LIVING_ID),
                       min_area: int = 60,
                       min_width_m: float = 0.30,
                       max_width_m: float = 4.5) -> list[RoadUser]:
    """Locate road users and place them on the ground plane."""
    users: list[RoadUser] = []
    max_range = min(ipm.far_m, 30.0)
    max_lateral = ipm.lateral_m / 2.0

    for cid in class_ids:
        binary = (labels == cid).astype(np.uint8)
        if binary.sum() < min_area:
            continue
        n, lab, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
        for k in range(1, n):
            x, y, bw, bh, area = stats[k]
            if area < min_area:
                continue
            contact = _ground_contact(lab[y:y + bh, x:x + bw] == k)
            if contact is None:
                continue
            row, c0, c1 = contact
            a_px, b_px, v = x + c0, x + c1, y + row

            ground = ipm.bev_to_ground(ipm.image_to_bev([[a_px, v], [b_px, v]]))
            y_m = float(0.5 * (ground[0][1] + ground[1][1]))
            x_m = float(0.5 * (ground[0][0] + ground[1][0]))
            width_m = float(ground[1][0] - ground[0][0])

            lo = min_width_m if cid == cfg.VEHICLE_ID else 0.15
            if not (ipm.near_m * 0.4 < y_m < max_range):
                continue
            if not (lo < width_m < max_width_m):
                continue
            if abs(x_m) > max_lateral:
                continue

            lane = -1
            if lane_model is not None:
                b = lane_model.boundaries_at(y_m)
                if len(b) >= 2 and b[0] <= x_m <= b[-1]:
                    lane = int(np.clip(np.searchsorted(b, x_m) - 1,
                                       0, lane_model.n_lanes - 1))
            users.append(RoadUser(x_m, y_m, width_m, lane, int(area),
                                  (int(x), int(y), int(bw), int(bh)), int(cid)))
    return users


@dataclass
class TrafficState:
    n_users: int
    per_lane: dict
    density_per_100m: float
    ego_lane_occupied: bool
    lead_distance_m: float
    road_area_m2: float
    occupancy_ratio: float

    def summary(self) -> str:
        lead = ("clear" if not np.isfinite(self.lead_distance_m)
                else f"{self.lead_distance_m:.1f} m")
        return (f"{self.n_users} road users, "
                f"{self.density_per_100m:.0f}/100m, lead {lead}")


def analyse_traffic(users: list[RoadUser], lane_model: LaneModel | None,
                    ipm: IPMTransform, range_m: float = 30.0) -> TrafficState:
    """Per-lane counts, density and lead-vehicle distance."""
    in_range = [u for u in users if u.y_m <= range_m]
    per_lane: dict[int, int] = {}
    for u in in_range:
        per_lane[u.lane] = per_lane.get(u.lane, 0) + 1

    on_road = [u for u in in_range if u.lane >= 0]
    span = max(range_m - ipm.near_m, 1e-6)
    density = 100.0 * len(on_road) / span

    lead = float("inf")
    ego = lane_model.ego_lane if lane_model else -1
    if ego >= 0:
        ahead = [u.y_m for u in on_road
                 if u.lane == ego and u.class_id == cfg.VEHICLE_ID]
        if ahead:
            lead = float(min(ahead))

    width = lane_model.carriageway_width_m if lane_model else 0.0
    area = width * span
    covered = sum(u.width_m * 2.5 for u in on_road)
    occupancy = float(np.clip(covered / max(area, 1e-6), 0.0, 1.0))

    return TrafficState(len(in_range), per_lane, density,
                        np.isfinite(lead), lead, area, occupancy)


class WrongSideDetector:
    """Flag road users travelling against the flow of their lane."""

    def __init__(self, oncoming_threshold: float = 0.45,
                 min_track_length: int = 4, max_association_m: float = 3.0,
                 keep_left: bool = True):
        self.oncoming_threshold = oncoming_threshold
        self.min_track_length = min_track_length
        self.max_association_m = max_association_m
        self.keep_left = keep_left
        self._tracks: dict[int, dict] = {}
        self._next_id = 0

    def reset(self) -> None:
        self._tracks.clear()
        self._next_id = 0

    def update(self, users: list[RoadUser],
               lane_model: LaneModel | None) -> list[RoadUser]:
        """Associate, estimate closing rate, and flag. Returns flagged users."""
        unmatched = dict(self._tracks)
        for u in users:
            best, best_d = None, self.max_association_m
            for tid, tr in unmatched.items():
                d = np.hypot(tr["x"] - u.x_m, tr["y"] - u.y_m)
                if d < best_d:
                    best, best_d = tid, d
            if best is None:
                tid = self._next_id
                self._next_id += 1
                self._tracks[tid] = {"x": u.x_m, "y": u.y_m, "n": 1,
                                     "dy": 0.0}
            else:
                tid = best
                tr = self._tracks[tid]
                dy = tr["y"] - u.y_m
                tr["dy"] = 0.6 * tr["dy"] + 0.4 * dy
                tr.update(x=u.x_m, y=u.y_m, n=tr["n"] + 1)
                unmatched.pop(tid, None)
            u.track_id = tid
            u.speed_y = self._tracks[tid]["dy"]

        for tid in unmatched:
            self._tracks[tid]["n"] = max(0, self._tracks[tid]["n"] - 1)
        self._tracks = {k: v for k, v in self._tracks.items() if v["n"] > 0}

        if lane_model is None or lane_model.n_lanes < 2:
            return []

        flagged = []
        for u in users:
            tr = self._tracks.get(u.track_id)
            if tr is None or tr["n"] < self.min_track_length:
                continue
            if u.lane < 0 or u.class_id != cfg.VEHICLE_ID:
                continue
            if u.speed_y < self.oncoming_threshold:
                continue
            b = lane_model.boundaries_at(u.y_m)
            if len(b) < 2:
                continue
            centre = 0.5 * (b[0] + b[-1])
            on_ego_side = (u.x_m < centre) if self.keep_left else (u.x_m > centre)
            if on_ego_side:
                flagged.append(u)
        return flagged


def draw_road_users(image: np.ndarray, users: list[RoadUser],
                    flagged: list[RoadUser] | None = None) -> np.ndarray:
    """Annotate road users with lane assignment and distance."""
    out = image.copy()
    bad = {id(u) for u in (flagged or [])}
    for u in users:
        x, y, w, h = u.bbox
        wrong = id(u) in bad
        colour = (0, 0, 255) if wrong else (
            (0, 200, 255) if u.class_id == cfg.VEHICLE_ID else (255, 100, 255))
        cv2.rectangle(out, (x, y), (x + w, y + h), colour, 2 if wrong else 1)
        tag = f"L{u.lane + 1}" if u.lane >= 0 else "off"
        label = f"{tag} {u.y_m:.0f}m" + ("  WRONG SIDE" if wrong else "")
        cv2.putText(out, label, (x, max(10, y - 3)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, colour, 1, cv2.LINE_AA)
    return out


@dataclass
class ScenePlausibility:
    """Whether the frame looks like a forward view of a road at all."""
    is_road_scene: bool
    reason: str
    n_classes: int
    road_above_horizon: float
    road_under_vehicle: float

    def __bool__(self) -> bool:
        return self.is_road_scene


def assess_scene(labels: np.ndarray,
                 min_classes: int = 5,
                 max_road_above: float = 0.05,
                 min_road_below: float = 0.15) -> ScenePlausibility:
    """Decide whether a semantic prediction describes a road scene."""
    h = labels.shape[0]
    n_classes = int(len(np.unique(labels[labels != cfg.IGNORE_INDEX])))
    road = labels == cfg.DRIVABLE_ID


    top, bot = int(0.25 * h), int(0.85 * h)
    if h < 8 or top < 1 or bot >= h:
        return ScenePlausibility(False, "frame is too small to assess",
                                 n_classes, 0.0, 0.0)
    above = float(road[:top].mean())
    below = float(road[bot:].mean())

    if n_classes < min_classes:
        return ScenePlausibility(False,
                                 f"only {n_classes} semantic classes present; a road "
                                 f"scene normally shows at least {min_classes}",
                                 n_classes, above, below)
    if above > max_road_above:
        return ScenePlausibility(False,
                                 f"{above*100:.0f}% of the upper frame is labelled "
                                 f"drivable; road cannot appear above the horizon",
                                 n_classes, above, below)
    if below < min_road_below:
        return ScenePlausibility(False,
                                 "no drivable surface directly ahead of the camera",
                                 n_classes, above, below)
    return ScenePlausibility(True, "road scene", n_classes, above, below)
