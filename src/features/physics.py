"""Physics-derived per-timestep features.

Per plan-003 §4.3.

Units (40 ms uniform grid → dt_sec = 0.04):
  X        m
  velocity m/s
  accel    m/s^2
  jerk     m/s^3
  curvature 1/m  (κ = |v × a| / |v|^3)

Forward-fill at t=0 to keep shape (n, T, *).
NaN/Inf in curvature replaced by 0 (decision-note §0.5).
"""
from __future__ import annotations

import numpy as np


def velocity(X: np.ndarray, dt_sec: float = 0.04) -> np.ndarray:
    if X.ndim != 3:
        raise ValueError(f"X must be (n, T, d); got {X.shape}")
    v = np.zeros_like(X)
    v[:, 1:, :] = (X[:, 1:, :] - X[:, :-1, :]) / dt_sec
    v[:, 0, :] = v[:, 1, :]  # forward fill
    return v


def acceleration(X: np.ndarray, dt_sec: float = 0.04) -> np.ndarray:
    v = velocity(X, dt_sec)
    a = np.zeros_like(X)
    a[:, 1:, :] = (v[:, 1:, :] - v[:, :-1, :]) / dt_sec
    a[:, 0, :] = a[:, 1, :]
    return a


def jerk(X: np.ndarray, dt_sec: float = 0.04) -> np.ndarray:
    a = acceleration(X, dt_sec)
    j = np.zeros_like(X)
    j[:, 1:, :] = (a[:, 1:, :] - a[:, :-1, :]) / dt_sec
    j[:, 0, :] = j[:, 1, :]
    return j


def curvature(X: np.ndarray, dt_sec: float = 0.04, eps: float = 1e-9) -> np.ndarray:
    """κ per timestep, shape (n, T, 1). Returns nan_to_num(0)."""
    v = velocity(X, dt_sec)
    a = acceleration(X, dt_sec)
    cross = np.cross(v, a)  # (n, T, 3)
    num = np.linalg.norm(cross, axis=-1)  # (n, T)
    den = np.linalg.norm(v, axis=-1) ** 3 + eps
    kappa = (num / den)[:, :, None]
    return np.nan_to_num(kappa, nan=0.0, posinf=0.0, neginf=0.0)
