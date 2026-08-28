"""Geometry: the rectification must be metrically exact and fail safely."""
import numpy as np
import pytest

import config as cfg
from core.geometry import (IPMTransform, VanishingPoint, robust_line_fit,
                           road_edge_points, vp_from_road_mask)

SHAPE = (227, 320)


@pytest.fixture
def ipm():
    return IPMTransform.from_vanishing_point(
        VanishingPoint(160.0, 105.0, 1.0, "t"), SHAPE)


def test_bev_ground_round_trip(ipm):
    pts = np.array([[0, 0], [319, 479], [160, 240], [10, 400]], float)
    assert np.abs(ipm.ground_to_bev(ipm.bev_to_ground(pts)) - pts).max() < 1e-9


def test_ground_image_round_trip(ipm):
    g = np.array([[-3.5, 12.0], [0.0, 25.0], [5.0, 8.0]])
    back = ipm.bev_to_ground(ipm.image_to_bev(ipm.ground_to_image(g)))
    assert np.abs(back - g).max() < 1e-9


def test_vanishing_point_is_the_image_of_infinity(ipm):
    far = ipm.ground_to_image([[0.0, 1e7]])[0]
    assert abs(far[0] - 160.0) < 1e-3
    assert abs(far[1] - 105.0) < 1e-3


def test_constant_width_road_stays_constant(ipm):
    """The whole point of rectifying: 7 m is 7 m at every distance."""
    Y = np.linspace(5, 35, 13)
    left = ipm.ground_to_image(np.stack([np.full_like(Y, -3.5), Y], 1))
    right = ipm.ground_to_image(np.stack([np.full_like(Y, 3.5), Y], 1))
    px = right[:, 0] - left[:, 0]
    assert px.max() / px.min() > 5, "test is meaningless without real perspective"

    wl = ipm.bev_to_ground(ipm.image_to_bev(left))[:, 0]
    wr = ipm.bev_to_ground(ipm.image_to_bev(right))[:, 0]
    assert np.abs((wr - wl) - 7.0).max() < 1e-6


def test_lateral_scale_is_independent_of_focal_length():
    """Documented in outputs/results/scale_calibration.md; locked in here."""
    vp = VanishingPoint(160.0, 105.0, 1.0, "t")
    spans = []
    for fov in (40, 60, 90, 120, 140):
        g = cfg.GeometryConfig()
        g.fov_deg = float(fov)
        T = IPMTransform.from_vanishing_point(vp, SHAPE, g)
        gg = T.bev_to_ground(T.image_to_bev([[60, 180], [260, 180]]))
        spans.append(gg[1][0] - gg[0][0])
    spans = np.array(spans)
    assert (spans.max() - spans.min()) / spans.mean() < 0.02


def test_lateral_scale_is_proportional_to_camera_height():
    vp = VanishingPoint(160.0, 105.0, 1.0, "t")
    ratios = []
    for H in (1.0, 1.5, 2.0):
        g = cfg.GeometryConfig()
        g.camera_height_m = H
        T = IPMTransform.from_vanishing_point(vp, SHAPE, g)
        gg = T.bev_to_ground(T.image_to_bev([[60, 180], [260, 180]]))
        ratios.append((gg[1][0] - gg[0][0]) / H)
    assert np.std(ratios) / np.mean(ratios) < 1e-6


def test_border_clipped_edges_are_flagged():
    """A road filling the frame has no valid side boundary."""
    mask = np.zeros(SHAPE, np.uint8)
    mask[150:, :] = 255
    _, _, _, lv, rv = road_edge_points(mask)
    assert lv.sum() == 0 and rv.sum() == 0


def test_vp_rejected_when_road_touches_both_borders():
    """This is the frame-#101 failure: the border is not a road edge."""
    mask = np.zeros(SHAPE, np.uint8)
    mask[140:, :] = 255
    assert vp_from_road_mask(mask) is None


def test_vp_recovered_on_a_clean_trapezoid():
    mask = np.zeros(SHAPE, np.uint8)
    import cv2
    cv2.fillPoly(mask, [np.array([[40, 226], [280, 226], [175, 120], [145, 120]])], 255)
    vp = vp_from_road_mask(mask)
    assert vp is not None
    assert 120 <= vp.x <= 200
    assert vp.y < 130


def test_robust_line_fit_ignores_outliers():
    y = np.arange(0, 100, 2.0)
    x = 0.5 * y + 10.0
    x[[3, 9, 17]] += 60.0
    a, b = robust_line_fit(y, x)
    assert abs(a - 0.5) < 0.05 and abs(b - 10.0) < 2.0


def test_pitch_is_bounded():
    """An absurd vanishing point must not produce an absurd camera."""
    T = IPMTransform.from_vanishing_point(
        VanishingPoint(160.0, -400.0, 1.0, "t"), SHAPE)
    assert -25.1 <= T.pitch_degrees <= 35.1
