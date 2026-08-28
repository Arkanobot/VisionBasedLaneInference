"""Synthetic validation of empirical lane inference."""
import numpy as np
import pytest

import config as cfg
from core.geometry import IPMTransform, VanishingPoint
from core.lane_tracker import LaneModel
from core.lane_trajectory import TrajectoryLaneEstimator, combine
from core.semantics import RoadUser

SHAPE = (227, 320)


def make_ipm():
    return IPMTransform.from_vanishing_point(
        VanishingPoint(160.0, 105.0, 1.0, "test"), SHAPE)


def make_lane_model(width=10.5, n=3):
    """A straight carriageway centred on the camera axis."""
    half = width / 2.0
    return LaneModel(
        n_lanes=n, lane_width_m=width / n, carriageway_width_m=width,
        left_coeffs=np.array([0.0, 0.0, -half]),
        right_coeffs=np.array([0.0, 0.0, half]),
        boundary_offsets=np.linspace(0.0, width, n + 1),
        ego_lane=1, road_type="unstructured", confidence=0.8)


def simulate(centres, n_frames=60, per_frame=4, jitter=0.45,
             ego_drift=0.8, noise_frac=0.0, seed=0):
    """Vehicles drawn from lanes at `centres`, with ego wander and jitter."""
    rng = np.random.default_rng(seed)
    ipm, model = make_ipm(), make_lane_model()
    est = TrajectoryLaneEstimator(min_observations=20)

    for f in range(n_frames):
        drift = ego_drift * np.sin(2 * np.pi * f / 25.0)
        users = []
        for _ in range(per_frame):
            y = float(rng.uniform(6.0, 28.0))
            if noise_frac and rng.random() < noise_frac:
                x = float(rng.uniform(-6.0, 6.0))
            else:
                x = float(rng.choice(centres) + rng.normal(0, jitter))
            users.append(RoadUser(x_m=x + drift, y_m=y, width_m=1.7, lane=0,
                                  pixel_area=400, bbox=(0, 0, 10, 10),
                                  class_id=cfg.VEHICLE_ID))


        m = make_lane_model()
        m.left_coeffs = np.array([0.0, 0.0, -5.25 + drift])
        m.right_coeffs = np.array([0.0, 0.0, 5.25 + drift])
        est.add_frame(users, m, ipm)
    return est


@pytest.mark.parametrize("centres", [
    [-3.5, 0.0, 3.5],
    [-1.75, 1.75],
    [-5.25, -1.75, 1.75, 5.25],
])
def test_recovers_known_lane_count(centres):
    est = simulate(centres, n_frames=80, per_frame=5)
    out = est.estimate()
    assert out is not None, "estimator returned nothing on clean synthetic traffic"
    assert out.n_lanes == len(centres),\
        f"expected {len(centres)} lanes, recovered {out.n_lanes} at {out.centres_m}"


def test_recovered_centres_are_accurate():
    truth = np.array([-3.5, 0.0, 3.5])
    out = simulate(truth, n_frames=100, per_frame=5).estimate()
    assert out is not None
    err = np.abs(np.sort(out.centres_m) - truth)
    assert err.max() < 0.5, f"centre error {err} exceeds 0.5 m"


def test_recovered_spacing_matches_truth():
    out = simulate([-3.5, 0.0, 3.5], n_frames=100, per_frame=5).estimate()
    assert out is not None
    assert abs(out.spacing_m - 3.5) < 0.4


def test_survives_off_lane_clutter():
    out = simulate([-3.5, 0.0, 3.5], n_frames=120, per_frame=6,
                   noise_frac=0.25, seed=3).estimate()
    assert out is not None
    assert out.n_lanes == 3, f"clutter broke the estimate: {out.centres_m}"


def test_returns_none_without_enough_traffic():
    """No traffic is not a reason to invent lanes."""
    est = simulate([-3.5, 0.0, 3.5], n_frames=3, per_frame=1)
    assert est.estimate() is None


def test_combine_prefers_geometry_when_unconfident():
    geo = make_lane_model(width=10.5, n=3)
    out = simulate([-3.5, 0.0, 3.5], n_frames=4, per_frame=2).estimate()
    model, source = combine(geo, out)
    assert source == "geometric"
    assert model.n_lanes == 3


def test_combine_rejects_implausible_spacing():
    """A confident estimate with 0.9 m lanes is wrong, not a discovery."""
    geo = make_lane_model()
    out = simulate([-0.9, 0.0, 0.9], n_frames=120, per_frame=6,
                   jitter=0.12).estimate()
    if out is not None and out.n_lanes > 1:
        _, source = combine(geo, out)
        assert source == "geometric"
