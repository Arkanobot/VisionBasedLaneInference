"""Empirical lane inference from observed traffic."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

import config as cfg
from core.geometry import IPMTransform
from core.lane_tracker import LaneModel


@dataclass
class Observation:
    offset_m: float
    distance_m: float
    weight: float


@dataclass
class EmpiricalLanes:
    centres_m: np.ndarray
    support: np.ndarray
    n_lanes: int
    spacing_m: float
    n_observations: int
    confidence: float

    def as_boundaries(self, width_m: float) -> np.ndarray:
        """Convert lane centres into boundaries by midpointing between them."""
        if len(self.centres_m) == 0:
            return np.empty(0)
        c = np.sort(self.centres_m)
        inner = 0.5 * (c[:-1] + c[1:])
        half = self.spacing_m / 2.0 if self.spacing_m > 0 else width_m / 2.0
        return np.concatenate([[c[0] - half], inner, [c[-1] + half]])


class TrajectoryLaneEstimator:
    """Accumulates road-user positions across frames and finds lane modes."""

    def __init__(self, bandwidth_m: float = 0.55, max_observations: int = 4000,
                 min_observations: int = 40, min_separation_m: float = 2.0,
                 half_extent_m: float = 12.0, resolution_m: float = 0.05,
                 decay: float = 0.995):
        self.bandwidth = bandwidth_m
        self.max_observations = max_observations
        self.min_observations = min_observations
        self.min_separation = min_separation_m
        self.half_extent = half_extent_m
        self.resolution = resolution_m
        self.decay = decay
        self._obs: list[Observation] = []

    def reset(self) -> None:
        self._obs.clear()

    @property
    def n_observations(self) -> int:
        return len(self._obs)


    def add_frame(self, users, lane_model: LaneModel | None,
                  ipm: IPMTransform) -> None:
        """Record one frame's road users, referenced to the carriageway centre."""
        if lane_model is None or lane_model.left_coeffs is None:
            return

        for o in self._obs:
            o.weight *= self.decay

        for u in users:
            if getattr(u, "class_id", cfg.VEHICLE_ID) != cfg.VEHICLE_ID:
                continue
            if not (ipm.near_m < u.y_m < min(ipm.far_m, 30.0)):
                continue
            centre = lane_model.centre_at(u.y_m)
            if not np.isfinite(centre):
                continue
            offset = u.x_m - centre
            if abs(offset) > self.half_extent:
                continue

            w = float(np.clip(10.0 / max(u.y_m, 1.0), 0.05, 1.0))
            self._obs.append(Observation(offset, u.y_m, w))

        if len(self._obs) > self.max_observations:
            self._obs = self._obs[-self.max_observations:]


    def estimate(self) -> EmpiricalLanes | None:
        if len(self._obs) < self.min_observations:
            return None

        offsets = np.array([o.offset_m for o in self._obs])
        weights = np.array([o.weight for o in self._obs])
        total_w = weights.sum()
        if total_w <= 1e-6:
            return None

        grid = np.arange(-self.half_extent, self.half_extent + 1e-9, self.resolution)

        z = (grid[None, :] - offsets[:, None]) / self.bandwidth
        density = (weights[:, None] * np.exp(-0.5 * z * z)).sum(axis=0)
        density /= (total_w * self.bandwidth * np.sqrt(2 * np.pi))
        if density.max() <= 1e-9:
            return None

        norm = density / density.max()
        min_sep_bins = max(1, int(round(self.min_separation / self.resolution)))


        cand = [i for i in range(1, len(norm) - 1)
                if norm[i] >= norm[i - 1] and norm[i] > norm[i + 1] and norm[i] > 0.25]
        peaks: list[int] = []
        for i in sorted(cand, key=lambda k: -norm[k]):
            if all(abs(i - p) >= min_sep_bins for p in peaks):
                peaks.append(i)
        if not peaks:
            return None
        peaks = np.array(sorted(peaks))

        centres = grid[peaks]

        support = np.array([
            weights[np.abs(offsets - c) <= self.min_separation / 2].sum()
            for c in centres])

        spacing = float(np.median(np.diff(centres))) if len(centres) > 1 else 0.0


        evidence = float(np.clip(total_w / 200.0, 0.0, 1.0))
        balance = float(support.min() / support.max()) if len(support) > 1 else 1.0
        conf = float(np.clip(0.6 * evidence + 0.4 * balance, 0.0, 1.0))

        return EmpiricalLanes(centres, support, len(centres), spacing,
                              len(self._obs), conf)


def combine(geometric: LaneModel | None, empirical: EmpiricalLanes | None,
            min_confidence: float = 0.45) -> tuple[LaneModel | None, str]:
    """Reconcile the geometric and empirical lane estimates."""
    if geometric is None:
        return None, "none"
    if empirical is None or empirical.confidence < min_confidence:
        return geometric, "geometric"
    if empirical.n_lanes < 1:
        return geometric, "geometric"
    if empirical.n_lanes > 1 and not (2.2 <= empirical.spacing_m <= 5.5):
        return geometric, "geometric"

    boundaries = empirical.as_boundaries(geometric.carriageway_width_m)
    if len(boundaries) < 2:
        return geometric, "geometric"


    left_edge = boundaries[0]
    geometric.boundary_offsets = boundaries - left_edge
    geometric.n_lanes = len(boundaries) - 1
    geometric.carriageway_width_m = float(boundaries[-1] - boundaries[0])
    geometric.lane_width_m = geometric.carriageway_width_m / max(geometric.n_lanes, 1)
    geometric.road_type = "unstructured-empirical"
    geometric.source = "traffic-trajectory"
    geometric.confidence = float(min(1.0, 0.5 * geometric.confidence
                                     + 0.5 * empirical.confidence))
    return geometric, "empirical"
