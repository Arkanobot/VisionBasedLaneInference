"""The overlay must never be drawn further than the evidence supports."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "lane_inference"))

from core.lane_tracker import LaneModel          # noqa: E402
from core.visualizer import lane_boundary_curves  # noqa: E402


class FakeIPM:
    """Flat-ground projection good enough to exercise the clamp."""
    near_m, far_m = 4.0, 40.0
    image_shape = (360, 640)

    def ground_to_image(self, pts):
        pts = np.asarray(pts, dtype=float)
        X, Y = pts[:, 0], pts[:, 1]
        v = 180.0 + 2000.0 / np.maximum(Y, 1e-6)
        u = 320.0 + X * 600.0 / np.maximum(Y, 1e-6)
        return np.stack([u, v], axis=1)


def _model(support):
    return LaneModel(
        n_lanes=2, lane_width_m=3.5, carriageway_width_m=7.0,
        left_coeffs=np.array([0.0, 0.0, -3.5]),
        right_coeffs=np.array([0.0, 0.0, 3.5]),
        boundary_offsets=np.array([0.0, 3.5, 7.0]),
        ego_lane=0, road_type="unstructured", confidence=0.8,
        source="test", marking_positions=np.empty(0),
        support_far_m=support,
    )


def _max_drawn_Y(curves, ipm):
    """Recover the furthest ground distance any drawn point corresponds to."""
    vs = np.concatenate([c[:, 1] for c in curves])
    return float(np.max(2000.0 / (np.min(vs) - 180.0)))


@pytest.mark.parametrize("support", [8.0, 12.0, 18.0, 25.0])
def test_overlay_stops_at_measured_support(support):
    ipm = FakeIPM()
    curves = lane_boundary_curves(_model(support), ipm, y_max=22.0)
    assert curves, "clamp should not remove the overlay entirely"
    assert _max_drawn_Y(curves, ipm) <= support + 1e-6


def test_support_never_extends_a_shorter_user_range():
    """A generous support must not override a tighter requested reach."""
    ipm = FakeIPM()
    curves = lane_boundary_curves(_model(35.0), ipm, y_max=10.0)
    assert _max_drawn_Y(curves, ipm) <= 10.0 + 1e-6


def test_no_support_recorded_falls_back_to_the_requested_range():
    ipm = FakeIPM()
    curves = lane_boundary_curves(_model(None), ipm, y_max=15.0)
    assert curves
    assert _max_drawn_Y(curves, ipm) <= 15.0 + 1e-6


def test_support_below_the_near_plane_draws_nothing():
    """Better an empty overlay than one drawn entirely past its evidence."""
    assert lane_boundary_curves(_model(2.0), FakeIPM(), y_max=22.0) == []
