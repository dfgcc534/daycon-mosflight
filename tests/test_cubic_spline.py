"""Tests for src.baselines.cubic_spline (plan-002 §4.5)."""
from __future__ import annotations

import numpy as np
import pytest

from src.baselines.cubic_spline import (
    predict_cspline,
    predict_cspline_per_axis,
    predict_smoothing_spline,
    tune_per_axis_cspline,
    tune_per_axis_smoothing,
)
from src.io import TIMESTEPS_MS


@pytest.fixture
def t_target() -> int:
    return 80


def _build_X_from_axis(y: np.ndarray) -> np.ndarray:
    """y: (11,) → X: (1, 11, 3) with all 3 axes equal to y."""
    return np.broadcast_to(y[None, :, None], (1, 11, 3)).copy()


@pytest.mark.parametrize("bc_type", ["natural", "not-a-knot", "clamped"])
def test_cspline_linear_exact_all_bc(bc_type: str, t_target: int) -> None:
    a, b = 0.123, -0.456
    y = a * TIMESTEPS_MS.astype(np.float64) + b
    X = _build_X_from_axis(y)
    pred = predict_cspline(X, window=11, bc_type=bc_type, t_target=t_target)
    expected = a * float(t_target) + b
    assert pred.shape == (1, 3)
    assert np.allclose(pred, expected, atol=1e-9), (bc_type, pred, expected)


def test_cspline_quadratic_exact_notaknot(t_target: int) -> None:
    """Not-a-knot BC reproduces a quadratic exactly (3rd-derivative = 0).

    decision-note: spec-default — clamped is omitted from this test because
    predict_cspline uses chord-derivative end conditions, which differ from the
    true quadratic derivative by a*h (where h is the boundary step). Clamped is
    therefore not exact for quadratic data via predict_cspline; not-a-knot is.
    """
    a, b, c = 1e-4, -0.2, 0.5
    t = TIMESTEPS_MS.astype(np.float64)
    y = a * t * t + b * t + c
    X = _build_X_from_axis(y)
    pred = predict_cspline(X, window=11, bc_type="not-a-knot", t_target=t_target)
    expected = a * t_target * t_target + b * t_target + c
    assert np.allclose(pred, expected, atol=1e-9), (pred, expected)


def test_smoothing_spline_s0_interpolates_training_points() -> None:
    """UnivariateSpline(k=3, s=0) is interpolating; reproduces inputs exactly at knots."""
    rng = np.random.default_rng(42)
    n = 4
    X = rng.normal(scale=0.1, size=(n, 11, 3))
    s_per_axis = [0.0, 0.0, 0.0]
    for j, t_eval in enumerate(TIMESTEPS_MS.tolist()):
        pred = predict_smoothing_spline(X, s_per_axis, t_target=int(t_eval))
        assert np.allclose(pred, X[:, j, :], atol=1e-9), (t_eval, pred, X[:, j, :])


def test_finite_outputs_on_random_input() -> None:
    rng = np.random.default_rng(123)
    X = rng.normal(scale=0.5, size=(8, 11, 3))
    for bc in ["natural", "not-a-knot", "clamped"]:
        pred = predict_cspline(X, window=4, bc_type=bc, t_target=80)
        assert np.isfinite(pred).all(), bc
        pred = predict_cspline(X, window=11, bc_type=bc, t_target=80)
        assert np.isfinite(pred).all(), bc

    info: dict = {}
    pred = predict_smoothing_spline(
        X, [1e-4, 1e-4, 1e-4], t_target=80,
        s_grid=[0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1], info_out=info,
    )
    assert np.isfinite(pred).all()
    assert "smoothing_fallback_count" in info


def test_clamped_recovers_slope_on_linear() -> None:
    """Clamped BC end-derivatives equal the chord derivative; on linear data the
    chord derivative IS the true slope, so the spline = the line and t=+80
    equals true line value (also covered above, but this verifies clamped path)."""
    a, b = 0.05, 1.2
    t = TIMESTEPS_MS.astype(np.float64)
    y = a * t + b
    X = _build_X_from_axis(y)
    pred = predict_cspline(X, window=4, bc_type="clamped", t_target=80)
    expected = a * 80.0 + b
    assert np.allclose(pred, expected, atol=1e-9), (pred, expected)
    pred = predict_cspline(X, window=11, bc_type="clamped", t_target=80)
    assert np.allclose(pred, expected, atol=1e-9), (pred, expected)


def test_predict_cspline_per_axis_dispatches() -> None:
    rng = np.random.default_rng(7)
    X = rng.normal(scale=0.3, size=(4, 11, 3))
    configs = [(4, "natural"), (5, "not-a-knot"), (4, "clamped")]
    pred = predict_cspline_per_axis(X, configs, t_target=80)
    assert pred.shape == (4, 3)
    assert np.isfinite(pred).all()


def test_window_lt_4_rejected() -> None:
    rng = np.random.default_rng(7)
    X = rng.normal(scale=0.3, size=(2, 11, 3))
    with pytest.raises(ValueError):
        predict_cspline(X, window=3, bc_type="natural", t_target=80)


def test_tune_per_axis_cspline_returns_well_formed() -> None:
    rng = np.random.default_rng(1)
    n = 50
    X = rng.normal(scale=0.3, size=(n, 11, 3))
    y = X[:, -1, :] + rng.normal(scale=0.01, size=(n, 3))
    grid = [(4, "natural"), (4, "clamped"), (5, "not-a-knot")]
    chosen, errors = tune_per_axis_cspline(X, y, grid, t_target=80, k=5)
    assert len(chosen) == 3
    for cell in chosen:
        assert cell in errors
    for err in errors.values():
        assert err.shape == (3,)
        assert np.isfinite(err).all()


def test_tune_per_axis_smoothing_returns_well_formed() -> None:
    rng = np.random.default_rng(2)
    n = 50
    X = rng.normal(scale=0.3, size=(n, 11, 3))
    y = X[:, -1, :] + rng.normal(scale=0.01, size=(n, 3))
    s_grid = [0.0, 1e-4, 1e-2]
    chosen, errors = tune_per_axis_smoothing(X, y, s_grid, t_target=80, n_folds=5)
    assert len(chosen) == 3
    for s in chosen:
        assert s in errors
    for err in errors.values():
        assert err.shape == (3,)
        assert np.isfinite(err).all()
