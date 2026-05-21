"""plan-024 c5.5 — train fold quantile carry (fold-leakage 차단).

spec @ plans/plan-024-cross-attention-anchor-vocab.md §3.6 / §4.4 (A10 Peak count
의 jerk_p90 threshold + S3 saccade 의 omega_p90 threshold).

train fold 위 quantile 계산 후 dict 박제. test fold 는 train fold quantile 그대로
사용 (fold-leakage 차단). plan-008/009 의 gap_ranking pattern carry.
"""
from __future__ import annotations

from typing import TypedDict

import numpy as np


class QuantileCarry(TypedDict):
    """train fold quantile 박제 schema (§3.6).

    Keys:
        omega_p90: ‖ω_Frenet‖ 의 90% quantile (saccade flag threshold)
        jerk_p90: ‖j_Frenet‖ 의 90% quantile (Peak count threshold)
        levy_tail_threshold: ‖Δp‖ 의 95% quantile (Lévy long-tail indicator,
            mosquito flight regime, 현 v1.1-rev2 미사용 — plan-025 후보)
    """
    omega_p90: float
    jerk_p90: float
    levy_tail_threshold: float


# ── helpers ────────────────────────────────────────────────────────────


def _compute_angular_velocity_frenet(X: np.ndarray, R_wfn: np.ndarray) -> np.ndarray:
    """sample-별 step t=2..10 의 angular velocity ω Frenet (§4.3 S2 식).

    ω_t = R_wfn^T · (v_{t-1} × v_t) / max(‖v_t‖², eps).
    Returns: (N, 9, 3) — t ∈ [2..10] = 9 step.
    """
    # v_t shape (N, T-1, 3), t=1..10 → 10 step
    v = np.diff(X, axis=1)                          # (N, 10, 3) world
    # cross_t = v_{t-1} × v_t, t-1∈[0..8], t∈[1..9] of v → 9 cross
    cross = np.cross(v[:, :-1, :], v[:, 1:, :], axis=2)   # (N, 9, 3) world
    v_t_norm_sq = (v[:, 1:, :] ** 2).sum(axis=2, keepdims=True)  # (N, 9, 1)
    omega_world = cross / np.maximum(v_t_norm_sq, 1e-12)         # (N, 9, 3)
    # to Frenet: R_wfn^T @ omega
    R_t = np.transpose(R_wfn, (0, 2, 1))                          # (N, 3, 3)
    omega_frenet = np.einsum("nij,ntj->nti", R_t, omega_world)    # (N, 9, 3)
    return omega_frenet.astype(np.float32)


def _compute_jerk_frenet(X: np.ndarray, R_wfn: np.ndarray, dt: float = 0.040) -> np.ndarray:
    """sample-별 step t=3..10 의 jerk Frenet (§4.3 S1 식).

    j_t = (a_t - a_{t-1}) / Δt, Frenet 분해.
    Returns: (N, 8, 3) — t ∈ [3..10] = 8 step.
    """
    v = np.diff(X, axis=1)                          # (N, 10, 3)
    a = np.diff(v, axis=1)                          # (N, 9, 3) world acc
    j_world = np.diff(a, axis=1) / dt               # (N, 8, 3) world jerk
    R_t = np.transpose(R_wfn, (0, 2, 1))            # (N, 3, 3)
    j_frenet = np.einsum("nij,ntj->nti", R_t, j_world)
    return j_frenet.astype(np.float32)


# ── public: build quantile_carry ───────────────────────────────────────


def build(X_train: np.ndarray, R_wfn_train: np.ndarray) -> QuantileCarry:
    """train fold quantile_carry 박제.

    Args:
        X_train: (N_tr, 11, 3) float, world coord (train fold sample only)
        R_wfn_train: (N_tr, 3, 3) float, per-sample Frenet basis (§4.0)

    Returns:
        QuantileCarry dict {omega_p90, jerk_p90, levy_tail_threshold}.
    """
    assert X_train.ndim == 3 and X_train.shape[1] == 11 and X_train.shape[2] == 3
    assert R_wfn_train.shape == (X_train.shape[0], 3, 3)

    # ω Frenet — t=2..10, 9 step × 3 axis → flatten N_tr × 9
    omega = _compute_angular_velocity_frenet(X_train.astype(np.float64),
                                              R_wfn_train.astype(np.float64))
    omega_norm = np.linalg.norm(omega, axis=2).flatten()             # (N_tr * 9,)
    omega_p90 = float(np.quantile(omega_norm, 0.90))

    # jerk Frenet — t=3..10, 8 step × 3 axis → flatten
    jerk = _compute_jerk_frenet(X_train.astype(np.float64),
                                 R_wfn_train.astype(np.float64))
    jerk_norm = np.linalg.norm(jerk, axis=2).flatten()               # (N_tr * 8,)
    jerk_p90 = float(np.quantile(jerk_norm, 0.90))

    # Lévy long-tail — ‖Δp‖ per step magnitude, 95% quantile
    dp_norm = np.linalg.norm(np.diff(X_train, axis=1), axis=2).flatten()  # (N_tr * 10,)
    levy_tail_threshold = float(np.quantile(dp_norm, 0.95))

    return {
        "omega_p90": omega_p90,
        "jerk_p90": jerk_p90,
        "levy_tail_threshold": levy_tail_threshold,
    }


# ── smoke (__main__) ───────────────────────────────────────────────────


if __name__ == "__main__":
    rng = np.random.default_rng(20260521)
    N = 100
    X = rng.standard_normal((N, 11, 3)).astype(np.float32) * 0.01
    R = np.tile(np.eye(3, dtype=np.float32)[None], (N, 1, 1))
    qc = build(X, R)
    print(f"[smoke] omega_p90={qc['omega_p90']:.6f}")
    print(f"[smoke] jerk_p90={qc['jerk_p90']:.6f}")
    print(f"[smoke] levy_tail_threshold={qc['levy_tail_threshold']:.6f}")
    assert qc["omega_p90"] > 0 and np.isfinite(qc["omega_p90"])
    assert qc["jerk_p90"] > 0 and np.isfinite(qc["jerk_p90"])
    assert qc["levy_tail_threshold"] > 0 and np.isfinite(qc["levy_tail_threshold"])
    print("[smoke] quantile_carry build ✓")
