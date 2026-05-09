"""Cubic spline baselines for muflight (plan-002).

Per plan-002 §4.1.

Functions:
- predict_cspline: per-axis interpolating cubic spline through last `window` points,
  evaluated at t_target. bc_type ∈ {"natural", "not-a-knot", "clamped"}.
- predict_cspline_per_axis: per-axis (window, bc_type) configs.
- tune_per_axis_cspline: inner k-fold CV grid search → axis-wise (window, bc_type).
- predict_smoothing_spline: per-axis UnivariateSpline (k=3, s>=0) on full 11 points.
- tune_per_axis_smoothing: inner k-fold CV grid search → axis-wise s.
"""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import scipy.interpolate

from src.io import T_TARGET_MS, TIMESTEPS_MS


def _inner_kfold_indices(n: int, k: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Deterministic stride-k split over indices 0..n-1.

    Mirrors src.io.kfold_split semantics (stride-k by sorted rank) without ids,
    so it stays stable across calls inside a single train fold. The runner
    `seed` argument is accepted for API stability but the split is deterministic.
    """
    fold_of = np.arange(n) % k
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for f in range(k):
        val_idx = np.where(fold_of == f)[0]
        train_idx = np.where(fold_of != f)[0]
        folds.append((train_idx, val_idx))
    return folds


def predict_cspline(
    X: np.ndarray,
    window: int,
    bc_type: str,
    t_target: int = T_TARGET_MS,
    timesteps: np.ndarray = TIMESTEPS_MS,
) -> np.ndarray:
    """Per-axis cubic spline through last `window` points; evaluate at t_target.

    Sample-loop / axis-loop. bc_type ∈ {"natural", "not-a-knot", "clamped"}.
    For clamped: end-derivatives are estimated by chord differences over the
    boundary step (left = (y[1]-y[0])/h, right = (y[-1]-y[-2])/h, h = uniform step).
    """
    if window < 4:
        raise ValueError(f"cubic spline 은 window >= 4 필요 (3점은 degenerate); got {window}")
    if window > timesteps.size:
        raise ValueError(f"window {window} > available timesteps {timesteps.size}")
    if bc_type not in ("natural", "not-a-knot", "clamped"):
        raise ValueError(f"unknown bc_type {bc_type!r}")

    n, _, n_axes = X.shape
    t = timesteps[-window:].astype(np.float64)
    h_left = float(t[1] - t[0])
    h_right = float(t[-1] - t[-2])
    out = np.zeros((n, n_axes), dtype=np.float64)

    for i in range(n):
        for ax in range(n_axes):
            y = X[i, -window:, ax]
            if bc_type == "clamped":
                left_d = float((y[1] - y[0]) / h_left)
                right_d = float((y[-1] - y[-2]) / h_right)
                bc: object = ((1, left_d), (1, right_d))
            else:
                bc = bc_type
            cs = scipy.interpolate.CubicSpline(t, y, bc_type=bc, extrapolate=True)
            out[i, ax] = float(cs(float(t_target)))
    return out


def predict_cspline_per_axis(
    X: np.ndarray,
    configs_per_axis: Sequence[tuple[int, str]],
    t_target: int = T_TARGET_MS,
    timesteps: np.ndarray = TIMESTEPS_MS,
) -> np.ndarray:
    """configs_per_axis: list of (window, bc_type) of length 3, one per axis."""
    if len(configs_per_axis) != 3:
        raise ValueError(f"need 3 configs (one per axis), got {len(configs_per_axis)}")
    out = np.empty((X.shape[0], 3), dtype=np.float64)
    for axis, cfg in enumerate(configs_per_axis):
        w, bc = int(cfg[0]), str(cfg[1])
        single = X[:, :, axis : axis + 1]
        pred = predict_cspline(single, w, bc, t_target=t_target, timesteps=timesteps)
        out[:, axis] = pred[:, 0]
    return out


def tune_per_axis_cspline(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    grid: Sequence[tuple[int, str]],
    t_target: int = T_TARGET_MS,
    k: int = 5,
    seed: int = 42,
    timesteps: np.ndarray = TIMESTEPS_MS,
) -> tuple[list[tuple[int, str]], dict[tuple[int, str], np.ndarray]]:
    """Inner k-fold CV grid search over (window, bc_type) cells; per-axis argmin.

    Returns (chosen, errors):
        chosen: list of (window, bc_type) of length 3 (axis 0/1/2).
        errors[(w, bc)]: np.ndarray shape (3,) — inner-CV mean per-axis MAE.
    """
    n = int(X_tr.shape[0])
    n_axes = int(y_tr.shape[1])
    folds = _inner_kfold_indices(n, k)

    errors: dict[tuple[int, str], np.ndarray] = {}
    for cell in grid:
        w, bc = int(cell[0]), str(cell[1])
        if w < 4 or w > timesteps.size:
            continue
        fold_axis_mae = np.zeros((k, n_axes), dtype=np.float64)
        for fi, (_tr_idx, va_idx) in enumerate(folds):
            pred = predict_cspline(
                X_tr[va_idx], w, bc, t_target=t_target, timesteps=timesteps
            )
            fold_axis_mae[fi] = np.abs(pred - y_tr[va_idx]).mean(axis=0)
        errors[(w, bc)] = fold_axis_mae.mean(axis=0)

    if not errors:
        raise ValueError("tune_per_axis_cspline: no valid grid entries")

    chosen: list[tuple[int, str]] = []
    for axis in range(n_axes):
        best_cell = min(errors.items(), key=lambda kv: float(kv[1][axis]))[0]
        chosen.append(best_cell)
    _ = seed
    return chosen, errors


def predict_smoothing_spline(
    X: np.ndarray,
    s_per_axis: Sequence[float],
    t_target: int = T_TARGET_MS,
    k: int = 3,
    timesteps: np.ndarray = TIMESTEPS_MS,
    s_grid: Iterable[float] | None = None,
    info_out: dict | None = None,
) -> np.ndarray:
    """Per-axis UnivariateSpline (degree=k, smoothing=s) through 11 points; eval at t_target.

    Fallback chain on fit failure / non-finite output:
      1. retry with next-larger s in s_grid (if any)
      2. CubicSpline (not-a-knot, extrapolate=True)
      3. last-resort: X[i, -1, ax] (last input value — always finite)

    info_out (optional dict): populated with `smoothing_fallback_count` after the call.
    """
    n, _, n_axes = X.shape
    out = np.zeros((n, n_axes), dtype=np.float64)
    fb_counts = {"step1_s_retry": 0, "step2_cubicspline": 0, "step3_last_input": 0}
    t = timesteps.astype(np.float64)
    s_grid_list: list[float] | None = (
        sorted(float(sx) for sx in s_grid) if s_grid is not None else None
    )

    for i in range(n):
        for ax in range(n_axes):
            y = X[i, :, ax]
            s_cur = float(s_per_axis[ax])
            value: float | None = None

            try:
                spl = scipy.interpolate.UnivariateSpline(t, y, k=k, s=s_cur)
                v = float(spl(float(t_target)))
                if np.isfinite(v):
                    value = v
            except Exception:
                value = None

            if value is None and s_grid_list is not None:
                next_s = [sx for sx in s_grid_list if sx > s_cur]
                for s_try in next_s:
                    try:
                        spl = scipy.interpolate.UnivariateSpline(t, y, k=k, s=s_try)
                        v = float(spl(float(t_target)))
                        if np.isfinite(v):
                            value = v
                            fb_counts["step1_s_retry"] += 1
                            break
                    except Exception:
                        continue

            if value is None:
                try:
                    cs = scipy.interpolate.CubicSpline(
                        t, y, bc_type="not-a-knot", extrapolate=True
                    )
                    v = float(cs(float(t_target)))
                    if np.isfinite(v):
                        value = v
                        fb_counts["step2_cubicspline"] += 1
                except Exception:
                    value = None

            if value is None or not np.isfinite(value):
                value = float(X[i, -1, ax])
                fb_counts["step3_last_input"] += 1

            out[i, ax] = value

    if info_out is not None:
        info_out["smoothing_fallback_count"] = fb_counts
    return out


def tune_per_axis_smoothing(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    s_grid: Sequence[float],
    t_target: int = T_TARGET_MS,
    k: int = 3,
    n_folds: int = 5,
    seed: int = 42,
    timesteps: np.ndarray = TIMESTEPS_MS,
) -> tuple[list[float], dict[float, np.ndarray]]:
    """Inner CV grid search over s; per-axis argmin.

    Returns (chosen_s_per_axis, errors):
        chosen_s_per_axis: list of float of length 3.
        errors[s]: np.ndarray shape (3,) — inner-CV mean per-axis MAE for that s.
    """
    n = int(X_tr.shape[0])
    n_axes = int(y_tr.shape[1])
    folds = _inner_kfold_indices(n, n_folds)

    errors: dict[float, np.ndarray] = {}
    for s_val in s_grid:
        s_val = float(s_val)
        s_per = [s_val, s_val, s_val]
        fold_axis_mae = np.zeros((n_folds, n_axes), dtype=np.float64)
        for fi, (_tr_idx, va_idx) in enumerate(folds):
            pred = predict_smoothing_spline(
                X_tr[va_idx], s_per, t_target=t_target, k=k,
                timesteps=timesteps, s_grid=s_grid,
            )
            fold_axis_mae[fi] = np.abs(pred - y_tr[va_idx]).mean(axis=0)
        errors[s_val] = fold_axis_mae.mean(axis=0)

    if not errors:
        raise ValueError("tune_per_axis_smoothing: empty s_grid")

    chosen: list[float] = []
    for axis in range(n_axes):
        best_s = min(errors.items(), key=lambda kv: float(kv[1][axis]))[0]
        chosen.append(float(best_s))
    _ = seed
    return chosen, errors
