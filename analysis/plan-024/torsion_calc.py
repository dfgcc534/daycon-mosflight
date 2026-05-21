"""plan-024 c5 — Frenet torsion τ scalar per past step (§4.5).

식 (Frenet-Serret 3rd formula, numerical-safe form):
  v_t   = X[:, t]   - X[:, t-1]
  v_tm1 = X[:, t-1] - X[:, t-2]
  cross_t = v_tm1 × v_t
  b_hat_t = cross_t / max(‖cross_t‖, eps_collinear)            # eps-guarded normalize
  # consecutive sign-flip alignment + first-step init:
  if b_hat_prev is None: pass                                   # no flip
  else: b_hat_t = sign(b_hat_t · b_hat_prev) * b_hat_t
  db = b_hat_t - b_hat_prev (or 0 if first)
  ds = max(‖v_t‖, eps_speed)
  n_hat_t = perp(v_t, b_hat_t) = normalize(b_hat_t × v_t)
  tau_t = -dot(db / ds, n_hat_t)
  valid_mask_t = (‖cross‖ > eps_collinear) & (‖v_t‖ > eps_speed) & (b_hat_prev is not None)
  tau_t[~valid_mask_t] = 0.0
  seq_torsion_t = [tau_t, sign(tau_t) * log(1+|tau_t|), valid_mask_t.float()]

Thresholds: eps_collinear=1e-6, eps_speed=1e-4.
First valid step (t=4) → tau=0 + mask=0 (insufficient history for db).
"""
from __future__ import annotations

import numpy as np

EPS_COLLINEAR = 1e-6
EPS_SPEED = 1e-4


def build(X: np.ndarray, t_range: tuple[int, int] = (4, 11)) -> np.ndarray:
    """Frenet torsion τ scalar per step, with numerical-safe mask.

    Args:
        X: (N, 11, 3) float64 world coord.
        t_range: (start, stop) — half-open, default (4, 11) → t ∈ {4..10} = 7 step.

    Returns:
        seq_torsion: (N, 7, 3) float32 — [τ, sign·log(1+|τ|), valid_mask].
    """
    t_start, t_stop = t_range
    T_seq = t_stop - t_start
    N = X.shape[0]
    X = X.astype(np.float64)

    seq_torsion = np.zeros((N, T_seq, 3), dtype=np.float32)
    b_hat_prev: np.ndarray | None = None

    for i, t in enumerate(range(t_start, t_stop)):
        v_t = X[:, t] - X[:, t - 1]                 # (N, 3)
        v_tm1 = X[:, t - 1] - X[:, t - 2]           # (N, 3)
        cross_t = np.cross(v_tm1, v_t)              # (N, 3)
        cross_norm = np.linalg.norm(cross_t, axis=1)  # (N,)
        v_t_norm = np.linalg.norm(v_t, axis=1)      # (N,)

        # eps-guarded normalize (NaN guard *before* mask)
        denom = np.maximum(cross_norm, EPS_COLLINEAR)
        b_hat_t = cross_t / denom[:, None]          # (N, 3)

        if b_hat_prev is not None:
            # consecutive sign-flip alignment
            sign_align = np.sign(np.sum(b_hat_t * b_hat_prev, axis=1))
            sign_align = np.where(sign_align == 0, 1.0, sign_align)  # 0 → +1
            b_hat_t = b_hat_t * sign_align[:, None]
            db = b_hat_t - b_hat_prev               # (N, 3)
        else:
            db = np.zeros_like(b_hat_t)

        ds = np.maximum(v_t_norm, EPS_SPEED)        # (N,) eps guard
        # n_hat = normalize(b × v)
        n_raw = np.cross(b_hat_t, v_t)
        n_norm = np.linalg.norm(n_raw, axis=1)
        n_hat_t = n_raw / np.maximum(n_norm, EPS_COLLINEAR)[:, None]

        # torsion scalar
        tau_t = -np.sum((db / ds[:, None]) * n_hat_t, axis=1)   # (N,)

        # mask (post-hoc; b_hat_prev=None → mask 0 무조건)
        valid_mask = (
            (cross_norm > EPS_COLLINEAR)
            & (v_t_norm > EPS_SPEED)
            & (b_hat_prev is not None)
        )
        tau_t = np.where(valid_mask, tau_t, 0.0)

        # transform for seq input (3D)
        seq_torsion[:, i, 0] = tau_t.astype(np.float32)
        seq_torsion[:, i, 1] = (
            np.sign(tau_t) * np.log1p(np.abs(tau_t))
        ).astype(np.float32)
        seq_torsion[:, i, 2] = valid_mask.astype(np.float32)

        b_hat_prev = b_hat_t

    return seq_torsion


# ── smoke (__main__) ───────────────────────────────────────────────────


if __name__ == "__main__":
    rng = np.random.default_rng(20260521)
    N = 50

    # case 1: random curved trajectory — most valid
    X_curve = rng.standard_normal((N, 11, 3)).astype(np.float64) * 0.01
    torsion = build(X_curve)
    assert torsion.shape == (N, 7, 3)
    assert np.isfinite(torsion).all()
    valid_frac = torsion[:, 1:, 2].mean()           # exclude t=4 (no prev)
    print(f"[smoke] curve case: shape={torsion.shape} valid_frac (t≥5)={valid_frac:.3f}")

    # case 2: collinear (straight line) — all mask=0
    X_straight = np.zeros((N, 11, 3), dtype=np.float64)
    X_straight[:, :, 0] = np.arange(11) * 0.001     # +x direction only
    torsion_straight = build(X_straight)
    assert torsion_straight.shape == (N, 7, 3)
    assert np.isfinite(torsion_straight).all()
    assert torsion_straight[:, :, 2].sum() == 0     # no valid step
    print(f"[smoke] straight case: valid_frac={torsion_straight[:, :, 2].mean():.3f} (should be 0)")

    # case 3: t=4 first step always mask=0
    assert torsion[:, 0, 2].sum() == 0
    print(f"[smoke] first step t=4 mask=0 ✓")

    print("[smoke] torsion_calc build ✓")
