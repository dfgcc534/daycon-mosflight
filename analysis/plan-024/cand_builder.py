"""plan-024 c4 — cand_feat 150D per anchor builder (§4.4, v1.1-rev2).

per (sample, anchor k=0..13), 150 channel = 4 묶음:

  묶음 ①  par/perp/dist (3D, sample × anchor)
        — anchor k 와 마지막 F0 residual r_last 의 Frenet 분해 (normalized by speed×horizon)

  묶음 ②  anchor spec (9D, anchor-static, sample 무관)
        — Frenet coord 3 + sign 3 + group 2 (axis vs corner) + idx scalar 1
        — (Fourier PE 12D 제거, v1.1-rev2)

  묶음 ③  ctx broadcast (128D, sample-conditional, 14 row 같은 값)
        — base 12 (last v/acc/res Frenet 9 + EWMA(α=0.3) res Frenet 3)
        — macro_stat 8 (plan-021 _macro_stat_9d − idx 1 straightness)
        — Bz/Tz 2 (R_wfn[:,2,2], R_wfn[:,2,0])
        — regime 18 (one-hot, plan-004 assign_regimes carry)
        — A1 STA/LTA ratio 3 (EWMA α=0.5/0.1 of F0 residual per Frenet axis)
        — A2 Multi-window 60 (144→60 trim, multiwindow_trim.json carry)
        — A5 WAP sample-level 5 (last-step WAP-5)
        — A6 wingbeat-jitter envelope 3 (std of EWMA-detrended Frenet pos per axis)
        — A8 f0_conf sample-level 2 (residual norm + step spread)
        — A10 Pct-rolling+Peak 12 (pct of rolling_std + count of jerk/sign-flip/sharp-turn)
        — A12 v_autocorr 3 (3-axis mean Pearson, lag k∈{1,2,3})

  묶음 ④  interactions (10D, sample × anchor)
        — base 8 scalar: anchor·res / anchor·v / anchor·acc / anchor·EWMA /
          corner×turn / sign-agreement / physics-extrap·anchor / anchor·Δz_world
        — A3 BCC adjacency 2 scalar: [mean_{j∈N(k)}<a_j, r_last>, std_{...}]
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Callable

import numpy as np

_THIS = Path(__file__).resolve().parent
_REPO = _THIS.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

DT = 0.040
HORIZON = 2  # 80ms = 2 step × 40ms (plan-020 carry)
EPS = 1e-9
N_ANCHORS = 14
CAND_DIM = 150


# ── helpers ────────────────────────────────────────────────────────────


def _bcc_adjacency(anchors: np.ndarray, k_neighbors: int = 3) -> np.ndarray:
    """anchor 별 k-nearest neighbor index (Euclidean, static precompute).

    Returns: (K, k_neighbors) int.
    """
    K = anchors.shape[0]
    pairwise = np.linalg.norm(anchors[:, None, :] - anchors[None, :, :], axis=2)
    np.fill_diagonal(pairwise, np.inf)                  # exclude self
    neighbors = np.argsort(pairwise, axis=1)[:, :k_neighbors]
    return neighbors                                     # (K, k_neighbors)


def _macro_stat_8(x: np.ndarray, end_idx: int) -> np.ndarray:
    """plan-021 _macro_stat_9d 의 idx 1 straightness 제외 → 8D (§4.4 ③ rev2).

    9-stat ordering: [path/cur_spd, *straightness*, speed_slope, speed_cv,
    turn_accum, accel_slope, turn_volatility, linear_resid, jerk_vol]
    rev2 keep idx [0, 2..8] = 8 stat.
    """
    start = max(0, end_idx - 5)
    pts = x[:, start:end_idx + 1, :]                    # (N, 6, 3)
    v = np.diff(pts, axis=1)                             # (N, 5, 3)
    speeds = np.linalg.norm(v, axis=2)                   # (N, 5)
    mean_speed = speeds.mean(axis=1, keepdims=True) + EPS
    path = speeds.sum(axis=1, keepdims=True)
    cur_speed = speeds[:, -1:] + EPS
    feat_0_path_speed = path / cur_speed
    speed_slope = (speeds[:, -1:] - speeds[:, 0:1]) / mean_speed
    speed_cv = speeds.std(axis=1, keepdims=True) / mean_speed
    v0 = v[:, :-1]
    v1 = v[:, 1:]
    turn_cos = (v0 * v1).sum(axis=2) / (
        np.linalg.norm(v0, axis=2) * np.linalg.norm(v1, axis=2) + EPS
    )
    turn_accum = (1.0 - np.clip(turn_cos, -1.0, 1.0)).mean(axis=1, keepdims=True)
    acc = np.diff(v, axis=1)                              # (N, 4, 3)
    acc_norm = np.linalg.norm(acc, axis=2)
    accel_slope = (acc_norm[:, -1:] - acc_norm[:, 0:1]) / (mean_speed + EPS)
    turn_volatility = np.std(
        1.0 - np.clip(turn_cos, -1.0, 1.0), axis=1, keepdims=True
    )
    linear_pred = pts[:, -2] + (pts[:, -2] - pts[:, -3])
    linear_resid = (
        np.linalg.norm(pts[:, -1] - linear_pred, axis=1, keepdims=True) / mean_speed
    )
    if acc.shape[1] < 2:
        jerk_vol = np.zeros_like(mean_speed)
    else:
        jerk = np.diff(acc, axis=1)
        jerk_vol = np.std(np.linalg.norm(jerk, axis=2), axis=1, keepdims=True) / mean_speed

    out = np.concatenate(
        [feat_0_path_speed, speed_slope, speed_cv, turn_accum, accel_slope,
         turn_volatility, linear_resid, jerk_vol], axis=1
    )                                                    # (N, 8)
    return out.astype(np.float32)


def _multiwindow_144(L1_frenet: np.ndarray) -> np.ndarray:
    """144D Multi-window stat — carry from multiwindow_trim_build._compute_144d_stat."""
    _spec = importlib.util.spec_from_file_location(
        "p024_mw_build", _THIS / "multiwindow_trim_build.py"
    )
    mw_build = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(mw_build)
    return mw_build._compute_144d_stat(L1_frenet)


def _build_L1_frenet(X: np.ndarray, R_wfn: np.ndarray) -> np.ndarray:
    """L1 (N, 11, 9) Frenet [p, v, a] — plan-021 _build_L1 carry simplified.

    p_t = R^T (X[t] - origin), v_t = R^T (X[t]-X[t-1]) for t≥1 else 0,
    a_t = R^T (v_t - v_{t-1}) for t≥2 else 0.
    """
    N, T, _ = X.shape
    R_t = np.transpose(R_wfn, (0, 2, 1)).astype(np.float64)
    origin = X[:, T - 1].astype(np.float64)
    L1 = np.zeros((N, T, 9), dtype=np.float32)

    for t in range(T):
        # position Frenet
        pos_w = X[:, t].astype(np.float64) - origin
        L1[:, t, 0:3] = np.einsum("nij,nj->ni", R_t, pos_w).astype(np.float32)
        # velocity Frenet
        if t >= 1:
            v_w = X[:, t].astype(np.float64) - X[:, t - 1].astype(np.float64)
            L1[:, t, 3:6] = np.einsum("nij,nj->ni", R_t, v_w).astype(np.float32)
        # accel Frenet
        if t >= 2:
            a_w = (X[:, t].astype(np.float64) - 2 * X[:, t - 1].astype(np.float64)
                   + X[:, t - 2].astype(np.float64))
            L1[:, t, 6:9] = np.einsum("nij,nj->ni", R_t, a_w).astype(np.float32)
    return L1


def _wingbeat_jitter(L1_frenet: np.ndarray, alpha: float = 0.6) -> np.ndarray:
    """A6 wingbeat-jitter envelope (3D, Frenet axis).

    std of (p_Frenet - EWMA_α(p_Frenet)) per Frenet axis (t̂/n̂/b̂).
    """
    p = L1_frenet[:, :, 0:3].astype(np.float64)         # (N, T, 3)
    s = p[:, 0].copy()
    detrended = np.zeros_like(p)
    detrended[:, 0] = 0.0
    for t in range(1, p.shape[1]):
        s = alpha * p[:, t] + (1 - alpha) * s
        detrended[:, t] = p[:, t] - s
    return detrended.std(axis=1).astype(np.float32)     # (N, 3)


def _pct_rolling_peak(
    L1_frenet: np.ndarray, X: np.ndarray, quantile_carry: dict
) -> np.ndarray:
    """A10 Pct-rolling+Peak 12D (§4.4):
    - pct_{20,50,80}(rolling_std(‖v_Frenet‖, w∈{3,5,7})) → 3×3=9
    - count_{t=4..10}(‖j_Frenet[t]‖ > quantile_carry.jerk_p90) → 1
    - count_{t=5..10}(sgn(v_t_Frenet[t,0])·sgn(v_t_Frenet[t-1,0]) < 0) → 1
    - count_{t=5..10}(turn_cos[t] < 0.5) → 1
    """
    N = L1_frenet.shape[0]
    v_frenet = L1_frenet[:, :, 3:6].astype(np.float64)  # (N, 11, 3)
    v_norm = np.linalg.norm(v_frenet, axis=2)            # (N, 11)
    a_frenet = L1_frenet[:, :, 6:9].astype(np.float64)
    # jerk Frenet per step (t=3..10, finite-diff of accel)
    j_frenet = np.diff(a_frenet, axis=1) / DT            # (N, 10, 3)
    j_norm = np.linalg.norm(j_frenet, axis=2)            # (N, 10)

    out = np.zeros((N, 12), dtype=np.float32)

    # rolling_std of ‖v‖ with w ∈ {3, 5, 7}
    idx = 0
    for w in (3, 5, 7):
        rolling_std = np.zeros((N, v_norm.shape[1] - w + 1))
        for i in range(v_norm.shape[1] - w + 1):
            rolling_std[:, i] = v_norm[:, i:i + w].std(axis=1)
        # pct 20/50/80
        for q in (0.20, 0.50, 0.80):
            out[:, idx] = np.quantile(rolling_std, q, axis=1).astype(np.float32)
            idx += 1
    assert idx == 9

    # jerk peak count (t=4..10 in original time → j_frenet idx t-1, i.e. idx 3..9)
    jerk_p90 = quantile_carry["jerk_p90"]
    j_slice = j_norm[:, 3:10]                             # (N, 7)
    out[:, 9] = (j_slice > jerk_p90).sum(axis=1).astype(np.float32)

    # sign-flip count on v_t̂ (Frenet t̂-axis = v_frenet[:, :, 0])
    v_t_axis = v_frenet[:, :, 0]                          # (N, 11)
    # t=5..10 → v_axis idx 4..9 vs 3..8
    sign_curr = np.sign(v_t_axis[:, 4:10])
    sign_prev = np.sign(v_t_axis[:, 3:9])
    flips = (sign_curr * sign_prev < 0).sum(axis=1).astype(np.float32)
    out[:, 10] = flips

    # sharp turn count: turn_cos[t] < 0.5, t=5..10 (v_t · v_{t-1} / (‖v_t‖‖v_{t-1}‖))
    # v_t at idx t-1 in v_frenet[:, 1:, :] effective indices 4..9
    v_a = v_frenet[:, 4:10, :]                            # (N, 6, 3) — v[t] for t=5..10 (idx 4..9 in 11-step)
    v_b = v_frenet[:, 3:9, :]                             # v[t-1]
    turn_cos = (v_a * v_b).sum(axis=2) / (
        np.linalg.norm(v_a, axis=2) * np.linalg.norm(v_b, axis=2) + EPS
    )
    out[:, 11] = (turn_cos < 0.5).sum(axis=1).astype(np.float32)

    return out


def _v_autocorr_3(L1_frenet: np.ndarray) -> np.ndarray:
    """A12 v_autocorr 3D — lag k∈{1,2,3}, 3축 Pearson 평균."""
    N = L1_frenet.shape[0]
    v_frenet = L1_frenet[:, 1:, 3:6].astype(np.float64)   # (N, 10, 3) — exclude v[0]=0
    out = np.zeros((N, 3), dtype=np.float32)
    for k_lag in (1, 2, 3):
        per_axis = np.zeros((N, 3))
        for c in range(3):
            a = v_frenet[:, k_lag:, c]
            b = v_frenet[:, :-k_lag, c]
            a_mean = a.mean(axis=1, keepdims=True)
            b_mean = b.mean(axis=1, keepdims=True)
            num = ((a - a_mean) * (b - b_mean)).sum(axis=1)
            den = np.sqrt(((a - a_mean) ** 2).sum(axis=1)
                          * ((b - b_mean) ** 2).sum(axis=1)) + EPS
            per_axis[:, c] = num / den
        out[:, k_lag - 1] = per_axis.mean(axis=1).astype(np.float32)
    return out


# ── public: build cand_feat 150D ───────────────────────────────────────


def build(
    X: np.ndarray,                # (N, 11, 3) float
    R_wfn: np.ndarray,            # (N, 3, 3) float
    pred_F0_world: np.ndarray,    # (N, 3) float — end_idx=10 의 80ms 미래 F0 예측
    anchors: np.ndarray,          # (14, 3) float, Frenet
    f0_baseline_fn: Callable[[np.ndarray, int], np.ndarray],
    regimes: np.ndarray,          # (N,) int64, regime 18-class (plan-004 carry)
    quantile_carry: dict,
    multiwindow_trim_path: str | Path = "analysis/plan-024/multiwindow_trim.json",
    regime_count: int = 18,
) -> np.ndarray:
    """cand_feat 150D per anchor build.

    Returns: (N, 14, 150) float32.
    """
    N = X.shape[0]
    K = anchors.shape[0]
    assert K == N_ANCHORS, f"K={K} != {N_ANCHORS}"
    X_f = X.astype(np.float64)
    R_t = np.transpose(R_wfn, (0, 2, 1)).astype(np.float64)
    anchors_f = anchors.astype(np.float64)

    out = np.zeros((N, K, CAND_DIM), dtype=np.float32)

    # ── pre-compute per-sample quantities ────────────────────────────
    # L1 Frenet (for macro/Multi-window/Pct-rolling/wingbeat/autocorr)
    L1_frenet = _build_L1_frenet(X, R_wfn)               # (N, 11, 9)

    # F0 residual at end_idx=10
    residual_w_last = X_f[:, 10] - pred_F0_world.astype(np.float64)   # (N, 3)
    residual_f_last = np.einsum("nij,nj->ni", R_t, residual_w_last)    # (N, 3) Frenet

    # last v/acc Frenet (= L1_frenet last step)
    v_last_f = L1_frenet[:, 10, 3:6].astype(np.float64)
    a_last_f = L1_frenet[:, 10, 6:9].astype(np.float64)

    # EWMA(α=0.3) of F0 residual Frenet per step (t=4..10, 7 step)
    # Need per-step residual: recompute
    residual_f_seq = np.zeros((N, 7, 3), dtype=np.float64)
    for i, t in enumerate(range(4, 11)):
        sub_x_t = X_f[:, t - 4:t - 1, :]
        pred_t = f0_baseline_fn(sub_x_t, end_idx=2).astype(np.float64)
        residual_w_t = X_f[:, t] - pred_t
        residual_f_seq[:, i] = np.einsum("nij,nj->ni", R_t, residual_w_t)
    s_03 = residual_f_seq[:, 0].copy()
    for t in range(1, 7):
        s_03 = 0.3 * residual_f_seq[:, t] + 0.7 * s_03
    ewma_03_res_f = s_03                                  # (N, 3)

    # STA/LTA ratio (EWMA α=0.5 / EWMA α=0.1)
    s_05 = residual_f_seq[:, 0].copy()
    s_01 = residual_f_seq[:, 0].copy()
    for t in range(1, 7):
        s_05 = 0.5 * residual_f_seq[:, t] + 0.5 * s_05
        s_01 = 0.1 * residual_f_seq[:, t] + 0.9 * s_01
    sta_lta = s_05 / (s_01 + EPS)                         # (N, 3) per Frenet axis

    # macro_stat 8 (plan-021 carry, straightness 제외)
    macro_8 = _macro_stat_8(X_f, end_idx=10)              # (N, 8)

    # Bz/Tz
    Bz = R_wfn[:, 2, 2].astype(np.float32)                # (N,) b̂_z
    Tz = R_wfn[:, 2, 0].astype(np.float32)                # (N,) t̂_z

    # regime one-hot (18)
    regime_idx = np.clip(regimes.astype(np.int64), 0, regime_count - 1)
    regime_oh = np.eye(regime_count, dtype=np.float32)[regime_idx]  # (N, 18)

    # Multi-window stat 144D + trim → 60D
    stat_144 = _multiwindow_144(L1_frenet)                # (N, 144)
    trim_path = Path(multiwindow_trim_path)
    with open(trim_path) as f:
        trim = json.load(f)
    kept_idx = np.asarray(trim["kept_indices"], dtype=np.int64)
    multiwindow_60 = stat_144[:, kept_idx]                # (N, 60)

    # WAP sample-level 5 (last-step Frenet, carry seq A5 식)
    v_norm = np.linalg.norm(v_last_f, axis=1)
    a_norm = np.linalg.norm(a_last_f, axis=1)
    j_last_f = L1_frenet[:, 10, 6:9].astype(np.float64) - L1_frenet[:, 9, 6:9].astype(np.float64)  # placeholder
    j_last_f = j_last_f / DT
    j_norm = np.linalg.norm(j_last_f, axis=1)
    v_unit = v_last_f / np.maximum(v_norm, EPS)[:, None]
    a_par = (a_last_f * v_unit).sum(axis=1, keepdims=True)
    a_perp = a_last_f - a_par * v_unit
    a_perp_norm = np.linalg.norm(a_perp, axis=1)
    kappa = a_perp_norm / np.maximum(v_norm, EPS)
    # WAP5
    wap5 = np.stack([
        v_norm ** 2 * kappa,
        j_norm / np.maximum(a_norm, EPS),
        0.5 * v_norm ** 2,
        np.sqrt(np.maximum(v_norm ** 2 - v_last_f[:, 0] ** 2, 0)) * 0.0,  # placeholder τ — sample-level τ approximation
        np.linalg.norm(residual_f_last, axis=1) * a_perp_norm,
    ], axis=1).astype(np.float32)                          # (N, 5)

    # wingbeat-jitter envelope 3
    wingbeat = _wingbeat_jitter(L1_frenet)                # (N, 3)

    # f0_conf sample-level 2
    f0_conf_norm = np.linalg.norm(residual_w_last, axis=1).astype(np.float32)
    v_world = np.diff(X_f, axis=1)
    step_speeds = np.linalg.norm(v_world, axis=2)         # (N, 10)
    f0_conf_spread = step_speeds.std(axis=1).astype(np.float32)
    f0_conf = np.stack([f0_conf_norm, f0_conf_spread], axis=1)   # (N, 2)

    # Pct-rolling + Peak 12
    pct_peak = _pct_rolling_peak(L1_frenet, X_f, quantile_carry)  # (N, 12)

    # v_autocorr 3
    v_autocorr = _v_autocorr_3(L1_frenet)                 # (N, 3)

    # ── pre-compute anchor-static (② spec) ───────────────────────────
    # ② anchor spec 9D — same for all sample (broadcast across N)
    coord = anchors_f.astype(np.float32)                  # (K, 3)
    sign_pat = np.sign(anchors_f).astype(np.float32)      # (K, 3)
    # group: axis (first 6) vs corner (last 8) — A6_bcc14 ordering
    is_axis = np.zeros(K, dtype=np.float32); is_axis[:6] = 1.0
    is_corner = np.zeros(K, dtype=np.float32); is_corner[6:] = 1.0
    idx_scalar = (np.arange(K) / K).astype(np.float32)
    spec_anchor_static = np.concatenate(
        [coord, sign_pat,
         np.stack([is_axis, is_corner], axis=1),
         idx_scalar[:, None]],
        axis=1,
    )                                                      # (K, 9)

    # BCC adjacency (static)
    adj_idx = _bcc_adjacency(anchors_f, k_neighbors=3)    # (K, 3)

    # ── per-sample × per-anchor compute ──────────────────────────────
    # ① par/perp/dist (3D)
    delta = anchors_f[None, :, :] - residual_f_last[:, None, :]   # (N, K, 3)
    speed_horizon = np.maximum(v_norm * HORIZON, EPS)[:, None]    # (N, 1)
    # par = (a_k - r_last) · t̂_Frenet — since residual already in Frenet, t̂ = unit_vector(1,0,0) in Frenet frame
    par = delta[:, :, 0:1]                                # (N, K, 1) — t̂ component
    perp_vec = delta - par * np.array([[[1, 0, 0]]])      # (N, K, 3)
    perp = np.linalg.norm(perp_vec, axis=2, keepdims=True)
    dist = np.linalg.norm(delta, axis=2, keepdims=True)
    out[:, :, 0:1] = (par / speed_horizon[:, :, None]).astype(np.float32)
    out[:, :, 1:2] = (perp / speed_horizon[:, :, None]).astype(np.float32)
    out[:, :, 2:3] = (dist / speed_horizon[:, :, None]).astype(np.float32)

    # ② anchor spec 9D — broadcast to (N, K, 9)
    out[:, :, 3:12] = np.broadcast_to(spec_anchor_static[None, :, :], (N, K, 9))

    # ③ ctx broadcast 128D — broadcast to (N, K, 128)
    # ordering: base 12 + macro 8 + Bz/Tz 2 + regime 18 + STA/LTA 3 + Multi 60 + WAP 5 + wingbeat 3 + f0_conf 2 + Pct 12 + autocorr 3
    base_12 = np.concatenate(
        [v_last_f.astype(np.float32), a_last_f.astype(np.float32),
         residual_f_last.astype(np.float32), ewma_03_res_f.astype(np.float32)],
        axis=1,
    )                                                      # (N, 12)
    Bz_Tz = np.stack([Bz, Tz], axis=1)                    # (N, 2)
    ctx_sample = np.concatenate(
        [base_12, macro_8, Bz_Tz, regime_oh,
         sta_lta.astype(np.float32),
         multiwindow_60,
         wap5,
         wingbeat,
         f0_conf,
         pct_peak,
         v_autocorr],
        axis=1,
    )                                                      # (N, 128)
    assert ctx_sample.shape[1] == 128, f"ctx_sample dim = {ctx_sample.shape[1]} != 128"
    out[:, :, 12:140] = np.broadcast_to(ctx_sample[:, None, :], (N, K, 128))

    # ④ interactions 10D — sample × anchor
    # (1) anchor·res
    int_1 = (anchors_f[None, :, :] * residual_f_last[:, None, :]).sum(axis=2)  # (N, K)
    # (2) anchor·v
    int_2 = (anchors_f[None, :, :] * v_last_f[:, None, :]).sum(axis=2)
    # (3) anchor·acc
    int_3 = (anchors_f[None, :, :] * a_last_f[:, None, :]).sum(axis=2)
    # (4) anchor·EWMA(res)
    int_4 = (anchors_f[None, :, :] * ewma_03_res_f[:, None, :]).sum(axis=2)
    # turn_cos last (between v[10] and v[9])
    v_prev = X_f[:, 9] - X_f[:, 8]
    v_curr = X_f[:, 10] - X_f[:, 9]
    turn_cos_last = (v_curr * v_prev).sum(axis=1) / (
        np.linalg.norm(v_curr, axis=1) * np.linalg.norm(v_prev, axis=1) + EPS
    )                                                      # (N,)
    # (5) corner×turn
    int_5 = is_corner[None, :] * turn_cos_last[:, None]   # (N, K)
    # (6) sign-agreement = Σ_c sgn(a_k,c) * sgn(res_c)
    res_sign = np.sign(residual_f_last)                   # (N, 3)
    int_6 = (sign_pat[None, :, :] * res_sign[:, None, :]).sum(axis=2)  # (N, K)
    # (7) physics-extrap·anchor = a_k · (v_last·Δt_horizon + ½·a_last·Δt_horizon²)
    dt_h = HORIZON * DT                                    # 0.080s = 80ms
    extrap = v_last_f * dt_h + 0.5 * a_last_f * dt_h ** 2  # (N, 3)
    int_7 = (anchors_f[None, :, :] * extrap[:, None, :]).sum(axis=2)
    # (8) anchor·Δz_world = (R_wfn @ a_k)[2]
    a_world = np.einsum("nij,kj->nki", R_wfn.astype(np.float64), anchors_f)  # (N, K, 3)
    int_8 = a_world[:, :, 2].astype(np.float32)
    # (A3) BCC adjacency neighbor pool 2D
    # for each anchor k, mean and std of <a_j, r_last> over j ∈ N(k)
    proj_anchor_res = (anchors_f * residual_f_last[:, None, :]).sum(axis=2)  # (N, K)
    adj_mean = np.zeros((N, K), dtype=np.float32)
    adj_std = np.zeros((N, K), dtype=np.float32)
    for k in range(K):
        proj_neigh = proj_anchor_res[:, adj_idx[k]]        # (N, k_neigh)
        adj_mean[:, k] = proj_neigh.mean(axis=1)
        adj_std[:, k] = proj_neigh.std(axis=1)
    interactions = np.stack(
        [int_1, int_2, int_3, int_4, int_5, int_6, int_7, int_8, adj_mean, adj_std],
        axis=2,
    ).astype(np.float32)                                   # (N, K, 10)
    out[:, :, 140:150] = interactions

    return out


# ── smoke (__main__) ───────────────────────────────────────────────────


if __name__ == "__main__":
    rng = np.random.default_rng(20260521)
    N = 30
    K = 14
    X = (rng.standard_normal((N, 11, 3)) * 0.005).astype(np.float64)
    R_wfn = np.tile(np.eye(3, dtype=np.float32)[None], (N, 1, 1))
    pred_F0 = X[:, 10] + (X[:, 10] - X[:, 9]).astype(np.float32)
    anchors = (rng.standard_normal((K, 3)) * 0.005).astype(np.float32)
    regimes = rng.integers(0, 18, size=N)

    def fake_f0(sub_x: np.ndarray, end_idx: int) -> np.ndarray:
        return sub_x[:, end_idx] + (sub_x[:, end_idx] - sub_x[:, end_idx - 1])

    qc = {"omega_p90": 0.5, "jerk_p90": 0.5, "levy_tail_threshold": 0.05}

    # build multiwindow_trim first
    import sys
    sys.path.insert(0, str(_THIS))
    import multiwindow_trim_build as mw_build
    L1 = _build_L1_frenet(X, R_wfn)
    mw_build.build_and_save(L1, output_path="/tmp/plan024_smoke_trim.json")

    cand = build(
        X, R_wfn, pred_F0, anchors, fake_f0,
        regimes=regimes, quantile_carry=qc,
        multiwindow_trim_path="/tmp/plan024_smoke_trim.json",
    )
    assert cand.shape == (N, K, 150), f"shape {cand.shape}"
    assert np.isfinite(cand).all()
    # ② spec invariance: same across N
    spec_block = cand[:, :, 3:12]
    assert np.allclose(spec_block[0], spec_block[15])
    # ③ ctx broadcast invariance: same across K
    ctx_block = cand[:, :, 12:140]
    assert np.allclose(ctx_block[:, 0], ctx_block[:, 7])
    # regime one-hot sum=1 (cand ③ 시작=12, base 12 + macro 8 + Bz/Tz 2 + regime 18 = 22~52)
    regime_block = cand[:, 0, 12 + 12 + 8 + 2:12 + 12 + 8 + 2 + 18]
    assert np.allclose(regime_block.sum(axis=1), 1.0), \
        f"regime block sum != 1: {regime_block.sum(axis=1)[:3]}"
    print(f"[smoke] cand_builder N={N} K={K} → shape {cand.shape} ✓")
    print(f"[smoke] ② spec invariant ✓, ③ broadcast invariant ✓, regime one-hot ✓")
