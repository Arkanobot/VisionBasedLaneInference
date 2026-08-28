"""Metrics must be exact on cases where the answer is known by hand."""
import numpy as np

import config as cfg
from eval.metrics import BoundaryAccumulator, ConfusionMatrix, boundary_f1


def test_perfect_prediction_scores_one():
    cm = ConfusionMatrix(2)
    gt = np.array([[0, 1], [1, 0]])
    cm.update(gt, gt)
    assert np.allclose(cm.iou(), 1.0)
    assert cm.pixel_accuracy() == 1.0


def test_known_iou():
    """8 pixels: 3 TP, 1 FP, 2 FN -> IoU 3/6."""
    gt = np.array([1, 1, 1, 1, 1, 0, 0, 0])
    pr = np.array([1, 1, 1, 0, 0, 1, 0, 0])
    cm = ConfusionMatrix(2)
    cm.update(pr, gt)
    assert abs(cm.iou()[1] - 0.5) < 1e-9
    assert abs(cm.precision()[1] - 0.75) < 1e-9
    assert abs(cm.recall()[1] - 0.6) < 1e-9


def test_ignore_index_excluded():
    gt = np.array([1, 1, cfg.IGNORE_INDEX, cfg.IGNORE_INDEX])
    pr = np.array([1, 1, 0, 0])
    cm = ConfusionMatrix(2)
    cm.update(pr, gt)
    assert cm.mat.sum() == 2, "ignored pixels leaked into the matrix"
    assert cm.pixel_accuracy() == 1.0


def test_accumulates_across_updates():
    cm = ConfusionMatrix(2)
    for _ in range(5):
        cm.update(np.array([1, 0]), np.array([1, 0]))
    assert cm.mat.sum() == 10


def test_absent_class_is_nan_not_zero():
    """A class with no pixels either way is undefined, not perfect and not failed."""
    cm = ConfusionMatrix(3)
    cm.update(np.array([0, 1]), np.array([0, 1]))
    assert np.isnan(cm.iou()[2])
    assert not np.isnan(np.nanmean(cm.iou()))


def test_boundary_f1_perfect_and_disjoint():
    a = np.zeros((40, 40), bool)
    a[10:30, 10:30] = True
    tp_p, n_p, tp_g, n_g = boundary_f1(a, a, tolerance=2)
    assert tp_p == n_p and tp_g == n_g

    b = np.zeros((40, 40), bool)
    b[0:5, 0:5] = True
    tp_p, n_p, tp_g, n_g = boundary_f1(b, a, tolerance=1)
    assert tp_p == 0 and tp_g == 0


def test_boundary_accumulator_scores():
    a = np.zeros((40, 40), bool)
    a[10:30, 10:30] = True
    acc = BoundaryAccumulator(tolerance=2)
    acc.update(a, a)
    assert abs(acc.score()["boundary_f1"] - 1.0) < 1e-9
