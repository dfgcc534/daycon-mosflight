"""plan-024 c3 — seq 95D per step builder (§4.3).

per past step t (t ∈ {4..10}, length=7), 95 channel:

  0-2   A position Frenet                3   (R_wfn^T · (X[t] - origin))
  3-5   B velocity Frenet                3   (R_wfn^T · v[t])
  6-8   C acceleration Frenet            3   (R_wfn^T · a[t])
  9-11  S1 jerk Frenet                   3   (R_wfn^T · j[t] / Δt)
  12-14 S2 angular velocity ω Frenet     3   (R_wfn^T · (v[t-1] × v[t]) / ‖v[t]‖²)
  15    Vz_world                         1   ((X[t,2] - X[t-1,2]) / Δt)
  16-18 D F0 residual Frenet (sign 통일) 3   (R_wfn^T · (actual[t] - pred[t]))
  19-20 residual angle                   2   ([atan2(res_n,res_t), asin(clip(res_b/‖res‖))])
  21-23 pred_F0 Frenet                   3   (R_wfn^T · (pred[t] - origin))
  24-25 E soft hit                       2   (σ((R_HIT - d)/τ_loss), σ((R_HIT_LOOSE - d)/τ_loss))
  26-39 F anchor-vocab soft              14  (anchor_vocab.build output)
  40    G ‖residual_t‖                   1
  41-54 H top1 one-hot                   14
  55    F2 log-magnitude                 1
  56-58 I Δresidual                      3   (r[t] - r[t-1], t=4 zero)
  59-61 Δ²residual                       3   (r[t] - 2·r[t-1] + r[t-2], t<6 zero)
  62-70 J residual EWMA (3α)             9   ([α=0.1, α=0.3, α=0.5] × Frenet)
  71    K time offset                    1   (t/10)
  72-75 S5 sinusoidal time PE            4   ([sin(2πi/7), cos(2πi/7), sin(4πi/7), cos(4πi/7)])
  76    L F entropy                      1   (-Σ_k q log q of F[t])
  77    M F 2nd-best mass                1
  78    O speed magnitude                1   (‖v[t]‖)
  79    A9 anchor-saliency               1   (max_k <a_k/‖a_k‖, r_t>)
  80    A11 helicity                     1   (v[t] · ω[t])
  81-85 A5 WAP per-step                  5   (Frenet [‖v‖²·κ, ‖j‖/(‖a‖+ε), ½‖v‖², ‖v_perp‖·τ_F, dist·‖a_perp‖])
  86    A8 f0_conf per-step              1   (polyfit_residual_norm_t / step_spread_t)
  87-88 S3 saccade binary                2   ([1{‖ω‖>q90}, 1{turn_cos<cos(60°)}])
  89    turn_cos                         1
  90    curvature                        1   (perp_norm / speed)
  91    direction_flag                   1   (constant +1.0)
  92-94 torsion τ (carry torsion_calc)   3   ([τ, sign·log(1+|τ|), valid_mask])

total: 3+3+3+3+3+1+3+2+3+2+14+1+14+1+3+3+9+1+4+1+1+1+1+1+5+1+2+1+1+1+3 = 95.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Callable

import numpy as np

_THIS = Path(__file__).resolve().parent
_REPO = _THIS.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

_spec = importlib.util.spec_from_file_location("p024_anchor_vocab", _THIS / "anchor_vocab.py")
anchor_vocab_mod = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(anchor_vocab_mod)

_spec = importlib.util.spec_from_file_location("p024_torsion", _THIS / "torsion_calc.py")
torsion_mod = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(torsion_mod)

DT = 0.040
R_HIT = 0.01
R_HIT_LOOSE = 0.015
TAU_LOSS = 0.001
EPS_DIV = 1e-9


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return np.where(z >= 0, 1.0 / (1.0 + np.exp(-z)),
                    np.exp(z) / (1.0 + np.exp(z))).astype(np.float64)


def _ewma_last(seq: np.ndarray, alpha: float) -> np.ndarray:
    """EWMA last value, seq shape (N, T, C) → (N, C)."""
    s = seq[:, 0].astype(np.float64)
    for t in range(1, seq.shape[1]):
        s = alpha * seq[:, t].astype(np.float64) + (1 - alpha) * s
    return s


def build(
    X: np.ndarray,                # (N, 11, 3) float, world
    R_wfn: np.ndarray,            # (N, 3, 3) float, per-sample Frenet basis
    anchors: np.ndarray,          # (14, 3) float, Frenet
    f0_baseline_fn: Callable[[np.ndarray, int], np.ndarray],
    quantile_carry: dict,         # {'omega_p90': float, 'jerk_p90': float, ...}
    tau_past: float = 0.003,
    t_range: tuple[int, int] = (4, 11),
) -> np.ndarray:
    """seq 95D per step build.

    Returns: (N, T_seq=7, 95) float32.
    """
    t_start, t_stop = t_range
    T_seq = t_stop - t_start
    N = X.shape[0]
    K = anchors.shape[0]
    SEQ_DIM = 95

    X_f = X.astype(np.float64)
    R_t = np.transpose(R_wfn, (0, 2, 1)).astype(np.float64)
    origin = X_f[:, 10]                                     # (N, 3)
    anchors_f = anchors.astype(np.float64)
    anchor_unit = anchors_f / np.linalg.norm(anchors_f, axis=1, keepdims=True)  # (K, 3)

    # precompute world differences
    v_world = np.diff(X_f, axis=1)                          # (N, 10, 3) v[t]=X[t]-X[t-1] at idx t-1
    a_world = np.diff(v_world, axis=1)                      # (N, 9, 3) a[t] at idx t-2
    j_world = np.diff(a_world, axis=1)                      # (N, 8, 3) j[t] at idx t-3

    # anchor-vocab F/G/H/F2 (precompute)
    av = anchor_vocab_mod.build(X, R_wfn, anchors, f0_baseline_fn,
                                tau_past=tau_past, t_range=t_range)
    F = av["F"]; G = av["G"]; H = av["H"]; F2 = av["F2"]

    # torsion (precompute)
    seq_torsion = torsion_mod.build(X_f, t_range=t_range)   # (N, T_seq, 3)

    # residual Frenet recompute (for I/Δ²/J/saliency/helicity etc.)
    residual_frenet_seq = np.zeros((N, T_seq, 3), dtype=np.float64)
    pred_t_world_seq = np.zeros((N, T_seq, 3), dtype=np.float64)
    for i, t in enumerate(range(t_start, t_stop)):
        sub_x_t = X_f[:, t - 4:t - 1, :]
        pred_t = f0_baseline_fn(sub_x_t, end_idx=2).astype(np.float64)
        actual_t = X_f[:, t]
        residual_w_t = actual_t - pred_t                    # ★ sign 통일
        residual_frenet_seq[:, i] = np.einsum("nij,nj->ni", R_t, residual_w_t)
        pred_t_world_seq[:, i] = pred_t

    out = np.zeros((N, T_seq, SEQ_DIM), dtype=np.float32)

    for i, t in enumerate(range(t_start, t_stop)):
        # A position Frenet
        pos_world = X_f[:, t] - origin
        out[:, i, 0:3] = np.einsum("nij,nj->ni", R_t, pos_world).astype(np.float32)

        # B velocity Frenet (raw difference, plan-021 L1 carry)
        v_t = v_world[:, t - 1]                              # (N, 3)
        out[:, i, 3:6] = np.einsum("nij,nj->ni", R_t, v_t).astype(np.float32)

        # C acceleration Frenet
        a_t = a_world[:, t - 2]                              # (N, 3)
        out[:, i, 6:9] = np.einsum("nij,nj->ni", R_t, a_t).astype(np.float32)

        # S1 jerk Frenet
        j_t = j_world[:, t - 3] / DT                         # (N, 3)
        out[:, i, 9:12] = np.einsum("nij,nj->ni", R_t, j_t).astype(np.float32)

        # S2 angular velocity ω Frenet
        v_tm1 = v_world[:, t - 2]
        cross = np.cross(v_tm1, v_t)
        v_t_norm_sq = np.maximum(np.sum(v_t ** 2, axis=1, keepdims=True), EPS_DIV ** 2)
        omega_world = cross / v_t_norm_sq
        out[:, i, 12:15] = np.einsum("nij,nj->ni", R_t, omega_world).astype(np.float32)

        # Vz_world
        out[:, i, 15] = ((X_f[:, t, 2] - X_f[:, t - 1, 2]) / DT).astype(np.float32)

        # D F0 residual Frenet (precomputed)
        res_f = residual_frenet_seq[:, i]
        out[:, i, 16:19] = res_f.astype(np.float32)

        # residual angle (eps guard + clip)
        res_norm = np.linalg.norm(res_f, axis=1)
        azim = np.arctan2(res_f[:, 1], res_f[:, 0])
        elev = np.arcsin(np.clip(res_f[:, 2] / np.maximum(res_norm, EPS_DIV), -1.0, 1.0))
        out[:, i, 19] = azim.astype(np.float32)
        out[:, i, 20] = elev.astype(np.float32)

        # pred_F0 Frenet
        pred_pos = pred_t_world_seq[:, i] - origin
        out[:, i, 21:24] = np.einsum("nij,nj->ni", R_t, pred_pos).astype(np.float32)

        # E soft hit
        d = res_norm
        out[:, i, 24] = _sigmoid((R_HIT - d) / TAU_LOSS).astype(np.float32)
        out[:, i, 25] = _sigmoid((R_HIT_LOOSE - d) / TAU_LOSS).astype(np.float32)

        # F (anchor-vocab soft)
        out[:, i, 26:40] = F[:, i, :]

        # G (magnitude)
        out[:, i, 40] = G[:, i]

        # H (top1 one-hot)
        out[:, i, 41:55] = H[:, i, :]

        # F2 (log-magnitude)
        out[:, i, 55] = F2[:, i]

        # I Δresidual (r[t] - r[t-1], i=0 zero)
        if i == 0:
            dres = np.zeros_like(res_f)
        else:
            dres = residual_frenet_seq[:, i] - residual_frenet_seq[:, i - 1]
        out[:, i, 56:59] = dres.astype(np.float32)

        # Δ²residual (r[t] - 2·r[t-1] + r[t-2], i<2 zero)
        if i < 2:
            d2res = np.zeros_like(res_f)
        else:
            d2res = (residual_frenet_seq[:, i]
                     - 2 * residual_frenet_seq[:, i - 1]
                     + residual_frenet_seq[:, i - 2])
        out[:, i, 59:62] = d2res.astype(np.float32)

        # K time offset
        out[:, i, 71] = float(t) / 10.0

        # S5 sinusoidal time PE — use i (0..6) within T_seq
        phase1 = 2 * np.pi * i / T_seq
        phase2 = 4 * np.pi * i / T_seq
        out[:, i, 72] = np.float32(np.sin(phase1))
        out[:, i, 73] = np.float32(np.cos(phase1))
        out[:, i, 74] = np.float32(np.sin(phase2))
        out[:, i, 75] = np.float32(np.cos(phase2))

        # L F entropy
        F_t = F[:, i, :]                                     # (N, K)
        F_safe = np.maximum(F_t, 1e-12)
        out[:, i, 76] = (-(F_safe * np.log(F_safe)).sum(axis=1)).astype(np.float32)

        # M F 2nd-best mass
        F_sorted = np.sort(F_t, axis=1)
        out[:, i, 77] = F_sorted[:, -2].astype(np.float32)

        # O speed magnitude
        out[:, i, 78] = np.linalg.norm(v_t, axis=1).astype(np.float32)

        # A9 anchor-saliency (max projection)
        proj = anchor_unit @ res_f.T                        # (K, N)
        out[:, i, 79] = proj.max(axis=0).astype(np.float32)

        # A11 helicity (v · ω)  -- compute in world (invariant under rotation)
        helicity = (v_t * omega_world).sum(axis=1)
        out[:, i, 80] = helicity.astype(np.float32)

        # A5 WAP per-step (5 component, Frenet-based)
        v_f = out[:, i, 3:6].astype(np.float64)             # v Frenet
        a_f = out[:, i, 6:9].astype(np.float64)             # a Frenet
        j_f = out[:, i, 9:12].astype(np.float64)            # j Frenet
        torsion_t = seq_torsion[:, i, 0].astype(np.float64) # τ scalar
        v_norm = np.linalg.norm(v_f, axis=1)
        a_norm = np.linalg.norm(a_f, axis=1)
        j_norm = np.linalg.norm(j_f, axis=1)
        # curvature κ = ‖perp_a‖ / ‖v‖ — perp_a = a - (a·v̂)v̂
        v_unit = v_f / np.maximum(v_norm, EPS_DIV)[:, None]
        a_par_scalar = (a_f * v_unit).sum(axis=1, keepdims=True)
        a_perp = a_f - a_par_scalar * v_unit
        a_perp_norm = np.linalg.norm(a_perp, axis=1)
        kappa = a_perp_norm / np.maximum(v_norm, EPS_DIV)
        # WAP-5:
        out[:, i, 81] = (v_norm ** 2 * kappa).astype(np.float32)            # |v|²·κ
        out[:, i, 82] = (j_norm / np.maximum(a_norm, EPS_DIV)).astype(np.float32)  # |j|/|a|
        out[:, i, 83] = (0.5 * v_norm ** 2).astype(np.float32)               # ½|v|²
        # |v_perp| · τ_F — v_perp = perp component of v w.r.t. previous direction? use n̂ component (v_n)
        v_perp_norm = np.sqrt(np.maximum(v_norm ** 2 - (v_f[:, 0]) ** 2, 0.0))  # ‖[v_n, v_b]‖
        out[:, i, 84] = (v_perp_norm * torsion_t).astype(np.float32)         # |v_perp|·τ_F
        out[:, i, 85] = (np.linalg.norm(res_f, axis=1) * a_perp_norm).astype(np.float32)  # dist·|a_perp|

        # A8 f0_conf per-step — polyfit residual norm (linear extrap diff) / step_spread
        # linear extrap pred at step t = X[t-1] + (X[t-1] - X[t-2]) = 2*X[t-1] - X[t-2]
        # polyfit residual = ‖X[t] - linear_extrap‖
        lin_extrap = 2 * X_f[:, t - 1] - X_f[:, t - 2]
        polyfit_res = np.linalg.norm(X_f[:, t] - lin_extrap, axis=1)
        # step_spread = ‖v[t-1] - v[t-2]‖ / Δt = accel magnitude proxy
        step_spread = np.linalg.norm(v_world[:, t - 1] - v_world[:, t - 2], axis=1) / DT
        out[:, i, 86] = (polyfit_res / np.maximum(step_spread, EPS_DIV)).astype(np.float32)

        # S3 saccade binary (2)
        omega_norm_t = np.linalg.norm(out[:, i, 12:15], axis=1)
        out[:, i, 87] = (omega_norm_t > quantile_carry["omega_p90"]).astype(np.float32)
        # turn_cos
        v_t_norm = np.linalg.norm(v_t, axis=1)
        v_tm1_norm = np.linalg.norm(v_tm1, axis=1)
        turn_cos = (v_t * v_tm1).sum(axis=1) / (v_t_norm * v_tm1_norm + EPS_DIV)
        out[:, i, 88] = (turn_cos < np.cos(np.radians(60))).astype(np.float32)
        out[:, i, 89] = turn_cos.astype(np.float32)

        # curvature
        out[:, i, 90] = kappa.astype(np.float32)

        # direction_flag (constant +1)
        out[:, i, 91] = 1.0

        # torsion (3D from torsion_calc)
        out[:, i, 92:95] = seq_torsion[:, i, :]

    # J residual EWMA (3α) — apply along T_seq dim of residual_frenet_seq
    # Output is per-step (N, T_seq, 9) — incrementally compute EWMA at each step
    s_01 = residual_frenet_seq[:, 0].copy()
    s_03 = residual_frenet_seq[:, 0].copy()
    s_05 = residual_frenet_seq[:, 0].copy()
    for i in range(T_seq):
        if i > 0:
            s_01 = 0.1 * residual_frenet_seq[:, i] + 0.9 * s_01
            s_03 = 0.3 * residual_frenet_seq[:, i] + 0.7 * s_03
            s_05 = 0.5 * residual_frenet_seq[:, i] + 0.5 * s_05
        out[:, i, 62:65] = s_01.astype(np.float32)
        out[:, i, 65:68] = s_03.astype(np.float32)
        out[:, i, 68:71] = s_05.astype(np.float32)

    return out


# ── smoke (__main__) ───────────────────────────────────────────────────


if __name__ == "__main__":
    rng = np.random.default_rng(20260521)
    N = 30
    K = 14
    X = (rng.standard_normal((N, 11, 3)) * 0.005).astype(np.float64)
    R_wfn = np.tile(np.eye(3, dtype=np.float32)[None], (N, 1, 1))
    anchors = (rng.standard_normal((K, 3)) * 0.005).astype(np.float32)
    qc = {"omega_p90": 1.0, "jerk_p90": 1.0, "levy_tail_threshold": 0.05}

    def fake_f0(sub_x: np.ndarray, end_idx: int) -> np.ndarray:
        return sub_x[:, end_idx] + (sub_x[:, end_idx] - sub_x[:, end_idx - 1])

    seq = build(X, R_wfn, anchors, fake_f0, qc, tau_past=0.003)
    assert seq.shape == (N, 7, 95), f"shape {seq.shape}"
    assert np.isfinite(seq).all(), "NaN/Inf in seq"
    # F sum=1 invariance per step
    F_block = seq[:, :, 26:40]
    assert np.allclose(F_block.sum(axis=-1), 1.0, atol=1e-5)
    # H one-hot invariance
    H_block = seq[:, :, 41:55]
    assert np.allclose(H_block.sum(axis=-1), 1.0, atol=1e-5)
    # I=0 at i=0
    assert np.allclose(seq[:, 0, 56:59], 0.0)
    # Δ²=0 at i<2
    assert np.allclose(seq[:, :2, 59:62], 0.0)
    # K time offset increasing
    assert np.all(np.diff(seq[0, :, 71]) > 0)
    print(f"[smoke] seq_builder N={N} K={K} → shape {seq.shape} ✓")
    print(f"[smoke] F sum=1 ✓, H one-hot ✓, I/Δ² zero-init ✓, K monotonic ✓")
