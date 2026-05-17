"""plan-020 §6 — 14 deterministic 후보 산식.

각 함수 signature 통일:
    C_i(x: np.ndarray, end_idx: int, fit_params: dict | None = None) -> np.ndarray
        x shape (N, T, 3), end_idx = T-1
        fit_params = CMA-ES 후 학습 param (0-param 후보는 None).
        returns pred shape (N, 3).

학습 (CMA-ES) 은 별도 `cma_es_fit.py` 책임. 본 module 은 *적용 only*.
"""
from __future__ import annotations

from typing import Any

import numpy as np

DT = 0.040
H = 0.080
F0_D1 = 1.98
F0_PAR = 1.20
F0_PERP = -0.20


# ── helpers ────────────────────────────────────────────────────────────


def _finite_diff_vaj(x: np.ndarray, end_idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """3D velocity / accel / jerk (m/s, m/s², m/s³) from last 4 points."""
    v = (x[:, end_idx] - x[:, end_idx - 1]) / DT
    a = (x[:, end_idx] - 2 * x[:, end_idx - 1] + x[:, end_idx - 2]) / DT**2
    j = (x[:, end_idx] - 3 * x[:, end_idx - 1] + 3 * x[:, end_idx - 2] - x[:, end_idx - 3]) / DT**3
    return v, a, j


def _f0_apply(x: np.ndarray, end_idx: int, d1: float | np.ndarray, par: float | np.ndarray, perp: float | np.ndarray) -> np.ndarray:
    """F0 산식 (vectorized) with per-sample coef if d1/par/perp are arrays shape (N,)."""
    p0 = x[:, end_idx]
    v_last = x[:, end_idx] - x[:, end_idx - 1]
    v_prev = x[:, end_idx - 1] - x[:, end_idx - 2]
    acc = v_last - v_prev
    speed = np.linalg.norm(v_last, axis=1, keepdims=True)
    tangent = v_last / (speed + 1e-9)
    acc_par_s = np.sum(acc * tangent, axis=1, keepdims=True)
    acc_par_vec = acc_par_s * tangent
    acc_perp_vec = acc - acc_par_vec

    def _broadcast(c):
        return np.asarray(c).reshape(-1, 1) if np.ndim(c) else float(c)

    d1_b, par_b, perp_b = _broadcast(d1), _broadcast(par), _broadcast(perp)
    return p0 + d1_b * v_last + par_b * acc_par_vec + perp_b * acc_perp_vec


def _rodrigues(axis_unit: np.ndarray, angle: np.ndarray) -> np.ndarray:
    """Batched Rodrigues. axis_unit shape (N, 3), angle shape (N,). returns (N, 3, 3)."""
    N = axis_unit.shape[0]
    K = np.zeros((N, 3, 3))
    K[:, 0, 1] = -axis_unit[:, 2]
    K[:, 0, 2] = axis_unit[:, 1]
    K[:, 1, 0] = axis_unit[:, 2]
    K[:, 1, 2] = -axis_unit[:, 0]
    K[:, 2, 0] = -axis_unit[:, 1]
    K[:, 2, 1] = axis_unit[:, 0]
    I = np.broadcast_to(np.eye(3), (N, 3, 3))
    s, c = np.sin(angle)[:, None, None], np.cos(angle)[:, None, None]
    return I + s * K + (1 - c) * np.einsum("nij,njk->nik", K, K)


def _v_last_frame(x: np.ndarray, end_idx: int) -> tuple[np.ndarray, np.ndarray]:
    """§C14 frame: R_s (N, 3, 3) world→frame rotation, origin_s (N, 3) = x[end_idx]."""
    v_last = x[:, end_idx] - x[:, end_idx - 1]
    T_hat = v_last / np.maximum(np.linalg.norm(v_last, axis=1, keepdims=True), 1e-9)
    z = np.broadcast_to(np.array([0.0, 0.0, 1.0]), T_hat.shape).copy()
    near_z = np.abs(np.sum(T_hat * z, axis=1)) > 0.99
    z[near_z] = np.array([1.0, 0.0, 0.0])
    N_hat = z - np.sum(z * T_hat, axis=1, keepdims=True) * T_hat
    N_hat = N_hat / np.maximum(np.linalg.norm(N_hat, axis=1, keepdims=True), 1e-9)
    B_hat = np.cross(T_hat, N_hat)
    R_s = np.stack([T_hat, N_hat, B_hat], axis=1)  # (N, 3, 3)
    return R_s, x[:, end_idx]


# ── C1. Local helix ────────────────────────────────────────────────────


def C01_helix(x: np.ndarray, end_idx: int, fit_params: dict | None = None) -> np.ndarray:
    fp = fit_params or {"alpha": 1.0, "beta": 1.0, "gamma": 1.0}
    a_coef, b_coef, g_coef = fp["alpha"], fp["beta"], fp["gamma"]
    v, a, j = _finite_diff_vaj(x, end_idx)
    v_norm = np.linalg.norm(v, axis=1, keepdims=True)
    tangent = v / (v_norm + 1e-9)
    acc_par = np.sum(a * tangent, axis=1, keepdims=True) * tangent
    acc_perp = a - acc_par
    perp_norm = np.linalg.norm(acc_perp, axis=1, keepdims=True)
    safe = (perp_norm > 1e-6) & (v_norm > 1e-6)
    normal = np.where(safe, acc_perp / (perp_norm + 1e-9), 0.0)
    binormal = np.cross(tangent, normal)
    kappa = (perp_norm / (v_norm**2 + 1e-9)).squeeze(-1)
    tau_num = np.sum(j * binormal, axis=1)
    tau = np.where(kappa > 1e-6, tau_num / (v_norm.squeeze(-1) ** 3 * np.maximum(kappa, 1e-9)), 0.0)
    tau = np.clip(tau, -10.0, 10.0)
    s = v_norm.squeeze(-1) * H

    ks = kappa * s
    sin_ks_kap = np.where(np.abs(kappa) > 1e-6, np.sin(ks) / np.maximum(kappa, 1e-9), s)
    one_cos_kap = np.where(np.abs(kappa) > 1e-6, (1 - np.cos(ks)) / np.maximum(kappa, 1e-9), 0.0)
    taus = tau * s

    pred = (
        x[:, end_idx]
        + a_coef * sin_ks_kap[:, None] * tangent
        + b_coef * one_cos_kap[:, None] * normal
        + g_coef * taus[:, None] * binormal
    )
    # κ < 1e-6 fallback → linear + accel
    fallback = ~safe.squeeze(-1)
    pred[fallback] = x[fallback, end_idx] + H * v[fallback] + 0.5 * H**2 * a[fallback]
    return pred


# ── C2. CTRA ───────────────────────────────────────────────────────────


def C02_ctra(x: np.ndarray, end_idx: int, fit_params: dict | None = None) -> np.ndarray:
    v_xy_t = (x[:, end_idx, :2] - x[:, end_idx - 1, :2]) / DT
    v_xy_p = (x[:, end_idx - 1, :2] - x[:, end_idx - 2, :2]) / DT
    v_xy = np.linalg.norm(v_xy_t, axis=1)
    v_xy_p_norm = np.linalg.norm(v_xy_p, axis=1)
    v_z = (x[:, end_idx, 2] - x[:, end_idx - 1, 2]) / DT
    v_z_p = (x[:, end_idx - 1, 2] - x[:, end_idx - 2, 2]) / DT
    theta_t = np.arctan2(v_xy_t[:, 1], v_xy_t[:, 0])
    theta_p = np.arctan2(v_xy_p[:, 1], v_xy_p[:, 0])
    omega = np.arctan2(np.sin(theta_t - theta_p), np.cos(theta_t - theta_p)) / DT
    a_xy_scalar = (v_xy - v_xy_p_norm) / DT
    a_z = (v_z - v_z_p) / DT
    omega = np.clip(omega, -30.0, 30.0)

    turn = np.abs(omega) > 1e-3
    h = H
    pred = np.empty((x.shape[0], 3))
    # xy
    sin_th = np.sin(theta_t)
    cos_th = np.cos(theta_t)
    sin_thh = np.sin(theta_t + omega * h)
    cos_thh = np.cos(theta_t + omega * h)
    om_safe = np.where(turn, omega, 1.0)
    dx_turn = (v_xy / om_safe) * (sin_thh - sin_th) + (a_xy_scalar / om_safe**2) * (cos_thh - cos_th + omega * h * sin_thh)
    dy_turn = -(v_xy / om_safe) * (cos_thh - cos_th) + (a_xy_scalar / om_safe**2) * (sin_thh - sin_th - omega * h * cos_thh)
    dx_lin = v_xy * cos_th * h + 0.5 * h**2 * a_xy_scalar * cos_th
    dy_lin = v_xy * sin_th * h + 0.5 * h**2 * a_xy_scalar * sin_th
    pred[:, 0] = x[:, end_idx, 0] + np.where(turn, dx_turn, dx_lin)
    pred[:, 1] = x[:, end_idx, 1] + np.where(turn, dy_turn, dy_lin)
    pred[:, 2] = x[:, end_idx, 2] + h * v_z + 0.5 * h**2 * a_z
    return pred


# ── C3. CTRV (CTRA with a=0) ──────────────────────────────────────────


def C03_ctrv(x: np.ndarray, end_idx: int, fit_params: dict | None = None) -> np.ndarray:
    v_xy_t = (x[:, end_idx, :2] - x[:, end_idx - 1, :2]) / DT
    v_xy_p = (x[:, end_idx - 1, :2] - x[:, end_idx - 2, :2]) / DT
    v_xy = np.linalg.norm(v_xy_t, axis=1)
    v_z = (x[:, end_idx, 2] - x[:, end_idx - 1, 2]) / DT
    theta_t = np.arctan2(v_xy_t[:, 1], v_xy_t[:, 0])
    theta_p = np.arctan2(v_xy_p[:, 1], v_xy_p[:, 0])
    omega = np.arctan2(np.sin(theta_t - theta_p), np.cos(theta_t - theta_p)) / DT
    omega = np.clip(omega, -30.0, 30.0)

    turn = np.abs(omega) > 1e-3
    h = H
    pred = np.empty((x.shape[0], 3))
    om_safe = np.where(turn, omega, 1.0)
    dx_turn = (v_xy / om_safe) * (np.sin(theta_t + omega * h) - np.sin(theta_t))
    dy_turn = -(v_xy / om_safe) * (np.cos(theta_t + omega * h) - np.cos(theta_t))
    dx_lin = v_xy * np.cos(theta_t) * h
    dy_lin = v_xy * np.sin(theta_t) * h
    pred[:, 0] = x[:, end_idx, 0] + np.where(turn, dx_turn, dx_lin)
    pred[:, 1] = x[:, end_idx, 1] + np.where(turn, dy_turn, dy_lin)
    pred[:, 2] = x[:, end_idx, 2] + h * v_z
    return pred


# ── C4. IMM (CV/CA/CT 3-mode mixture) ─────────────────────────────────


def C04_imm(x: np.ndarray, end_idx: int, fit_params: dict | None = None) -> np.ndarray:
    """fit_params keys: 'pi' (vec3) = (π_CV, π_CA, π_CT) softmax-normalized."""
    fp = fit_params or {"pi": np.array([1.0, 1.0, 1.0]) / 3.0}
    pi = np.asarray(fp["pi"], dtype=float)
    v_t = (x[:, end_idx] - x[:, end_idx - 1]) / DT
    v_p = (x[:, end_idx - 1] - x[:, end_idx - 2]) / DT
    a_t = (v_t - v_p) / DT
    p_CV = x[:, end_idx] + H * v_t
    p_CA = x[:, end_idx] + H * v_t + 0.5 * H**2 * a_t
    p_CT = C03_ctrv(x, end_idx)
    return pi[0] * p_CV + pi[1] * p_CA + pi[2] * p_CT


# ── C5. Per-regime F0 ──────────────────────────────────────────────────


def C05_per_regime_f0(x: np.ndarray, end_idx: int, fit_params: dict | None = None) -> np.ndarray:
    """fit_params keys: 'regime_params' = {r: (d1, par, perp)} for r ∈ 0..17,
       'regimes' = np.ndarray shape (N,) int — per-sample regime id (caller responsibility, fold-internal)."""
    if fit_params is None or "regime_params" not in fit_params or "regimes" not in fit_params:
        return _f0_apply(x, end_idx, F0_D1, F0_PAR, F0_PERP)
    regimes = np.asarray(fit_params["regimes"], dtype=int)
    d1 = np.full(x.shape[0], F0_D1, dtype=float)
    par = np.full(x.shape[0], F0_PAR, dtype=float)
    perp = np.full(x.shape[0], F0_PERP, dtype=float)
    for r, params in fit_params["regime_params"].items():
        mask = regimes == int(r)
        if not mask.any():
            continue
        d1[mask], par[mask], perp[mask] = params
    return _f0_apply(x, end_idx, d1, par, perp)


# ── C6. Quintic Hermite endpoint spline ───────────────────────────────


def C06_quintic_hermite(x: np.ndarray, end_idx: int, fit_params: dict | None = None) -> np.ndarray:
    """6 constraints: p(0..-120ms) positions (4) + p'(0), p''(0). axis-independent."""
    # times in seconds (relative to end_idx)
    t = np.array([0.0, -DT, -2 * DT, -3 * DT])
    # constraint matrix (6x6) for [a0, a1, a2, a3, a4, a5]
    # p(τ) = a0 + a1 τ + a2 τ² + a3 τ³ + a4 τ⁴ + a5 τ⁵
    # p'(τ) = a1 + 2 a2 τ + 3 a3 τ² + 4 a4 τ³ + 5 a5 τ⁴
    # p''(τ) = 2 a2 + 6 a3 τ + 12 a4 τ² + 20 a5 τ³
    rows = []
    for ti in t:
        rows.append([1, ti, ti**2, ti**3, ti**4, ti**5])
    rows.append([0, 1, 0, 0, 0, 0])  # p'(0)
    rows.append([0, 0, 2, 0, 0, 0])  # p''(0)
    M = np.asarray(rows, dtype=float)  # (6, 6)
    try:
        M_inv = np.linalg.inv(M)
    except np.linalg.LinAlgError:
        v = (x[:, end_idx] - x[:, end_idx - 1]) / DT
        a = (x[:, end_idx] - 2 * x[:, end_idx - 1] + x[:, end_idx - 2]) / DT**2
        return x[:, end_idx] + H * v + 0.5 * H**2 * a
    # build RHS per sample, per axis
    pred = np.empty((x.shape[0], 3))
    v_endpoint = (x[:, end_idx] - x[:, end_idx - 1]) / DT
    a_endpoint = (x[:, end_idx] - 2 * x[:, end_idx - 1] + x[:, end_idx - 2]) / DT**2
    for axis in range(3):
        rhs = np.stack(
            [
                x[:, end_idx, axis],
                x[:, end_idx - 1, axis],
                x[:, end_idx - 2, axis],
                x[:, end_idx - 3, axis],
                v_endpoint[:, axis],
                a_endpoint[:, axis],
            ],
            axis=1,
        )  # (N, 6)
        coefs = rhs @ M_inv.T  # (N, 6)
        powers = np.array([H**i for i in range(6)])
        pred[:, axis] = coefs @ powers
    return pred


# ── C7. Jerk-aware cubic ──────────────────────────────────────────────


def C07_jerk_quartic(x: np.ndarray, end_idx: int, fit_params: dict | None = None) -> np.ndarray:
    v, a, j = _finite_diff_vaj(x, end_idx)
    j_norm = np.linalg.norm(j, axis=1, keepdims=True)
    j_clipped = np.where(j_norm > 100.0, j * (100.0 / np.maximum(j_norm, 1e-9)), j)
    return x[:, end_idx] + H * v + 0.5 * H**2 * a + (H**3 / 6.0) * j_clipped


# ── C8. Singer maneuver ───────────────────────────────────────────────


def C08_singer(x: np.ndarray, end_idx: int, fit_params: dict | None = None) -> np.ndarray:
    fp = fit_params or {"tau_a": 0.100}
    tau_a = float(fp["tau_a"])
    v, a, _ = _finite_diff_vaj(x, end_idx)
    factor = tau_a**2 * (H / tau_a - 1.0 + np.exp(-H / tau_a))
    return x[:, end_idx] + H * v + a * factor


# ── C9. Adaptive Kalman smoother (per-axis) ───────────────────────────


def _kalman_per_axis(
    obs: np.ndarray,  # (N, T_obs) per-axis position obs
    q: float,
    r: float,
    forward_steps: int,
) -> np.ndarray:
    """KF forward filter on (p, v, a) const-accel model + forward propagate.
    (RTS smoother 후속 v1.4 박제 예정 — 본 v1.3 implementation 은 forward filter only.)"""
    N, T = obs.shape
    F = np.array([[1.0, DT, 0.5 * DT**2], [0.0, 1.0, DT], [0.0, 0.0, 1.0]])
    G = np.array([DT**3 / 6.0, DT**2 / 2.0, DT])
    Q_mat = q * np.outer(G, G)
    # init: pos = first obs, vel = first-diff, accel = 0 (finite-diff a is unstable on 3 points).
    # Initial P large enough for KF to self-correct within first few obs.
    x_state = np.zeros((N, 3))
    x_state[:, 0] = obs[:, 0]
    x_state[:, 1] = (obs[:, 1] - obs[:, 0]) / DT
    # x_state[:, 2] = 0  (already zero — let KF infer acceleration from successive obs)
    P_init_diag = np.array([1e-3, 1.0, 1e2])  # p tight, v loose, a very loose
    P = np.broadcast_to(np.diag(P_init_diag), (N, 3, 3)).copy()
    for t in range(1, T):
        x_pred = x_state @ F.T
        P_pred = F @ P @ F.T + Q_mat                # broadcast (3,3) @ (N,3,3) @ (3,3)
        y = obs[:, t] - x_pred[:, 0]                # H = [1,0,0]
        S_arr = P_pred[:, 0, 0] + r                 # (N,)
        K = P_pred[:, :, 0] / S_arr[:, None]        # (N, 3)
        x_state = x_pred + K * y[:, None]
        P = P_pred - K[:, :, None] * P_pred[:, 0:1, :]
    state = x_state
    for _ in range(forward_steps):
        state = state @ F.T
    return state[:, 0]


def C09_kalman_smoother(x: np.ndarray, end_idx: int, fit_params: dict | None = None) -> np.ndarray:
    """per-axis independent KF + forward propagate 2 steps (= 80 ms)."""
    fp = fit_params or {"log_q": -6.0, "log_r": -4.0}
    q = float(np.exp(fp["log_q"]))
    r = float(np.exp(fp["log_r"]))
    T_obs = end_idx + 1  # 11 points
    pred = np.empty((x.shape[0], 3))
    forward_steps = int(round(H / DT))
    for axis in range(3):
        pred[:, axis] = _kalman_per_axis(x[:, :T_obs, axis], q, r, forward_steps)
    return pred


# ── C10. Bishop rotation-minimizing frame ─────────────────────────────


def _bishop_frame(x: np.ndarray, end_idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sequential parallel transport. Returns (T_last, M1_last, M2_last) shape (N, 3)."""
    N = x.shape[0]
    # init at t=0 → 1
    T = np.zeros((N, 3))
    deltas = np.diff(x[:, : end_idx + 1, :], axis=1)  # (N, end_idx, 3)
    norms = np.linalg.norm(deltas, axis=2, keepdims=True)
    deltas_unit = deltas / np.maximum(norms, 1e-9)
    T = deltas_unit[:, 0]
    z = np.broadcast_to(np.array([0.0, 0.0, 1.0]), (N, 3)).copy()
    near_z = np.abs(np.sum(T * z, axis=1)) > 0.99
    z[near_z] = np.array([1.0, 0.0, 0.0])
    M1 = z - np.sum(z * T, axis=1, keepdims=True) * T
    M1 = M1 / np.maximum(np.linalg.norm(M1, axis=1, keepdims=True), 1e-9)
    M2 = np.cross(T, M1)
    for t in range(1, deltas_unit.shape[1]):
        T_next = deltas_unit[:, t]
        b = np.cross(T, T_next)
        b_norm = np.linalg.norm(b, axis=1, keepdims=True)
        theta = np.arctan2(b_norm.squeeze(-1), np.sum(T * T_next, axis=1))
        safe = b_norm.squeeze(-1) > 1e-9
        if safe.any():
            axis_unit = np.where(safe[:, None], b / np.maximum(b_norm, 1e-9), 0.0)
            R = _rodrigues(axis_unit, theta)
            M1_new = np.einsum("nij,nj->ni", R, M1)
            M2_new = np.einsum("nij,nj->ni", R, M2)
            M1 = np.where(safe[:, None], M1_new, M1)
            M2 = np.where(safe[:, None], M2_new, M2)
        T = T_next
    return T, M1, M2


def C10_bishop_frame(x: np.ndarray, end_idx: int, fit_params: dict | None = None) -> np.ndarray:
    fp = fit_params or {"lam": 1.0}
    lam = float(fp["lam"])
    if not np.isfinite(lam):
        lam = 1.0
    T_last, M1_last, M2_last = _bishop_frame(x, end_idx)
    v_last = x[:, end_idx] - x[:, end_idx - 1]
    v_prev = x[:, end_idx - 1] - x[:, end_idx - 2]
    acc = v_last - v_prev
    acc_par_vec = np.sum(acc * T_last, axis=1, keepdims=True) * T_last
    acc_perp_M1 = np.sum(acc * M1_last, axis=1, keepdims=True) * M1_last
    acc_perp_M2 = np.sum(acc * M2_last, axis=1, keepdims=True) * M2_last
    return x[:, end_idx] + F0_D1 * v_last + F0_PAR * acc_par_vec + F0_PERP * acc_perp_M1 + (F0_PERP * lam) * acc_perp_M2


# ── C11. SE(3) exponential twist (position-only approx) ───────────────


def C11_se3_twist(x: np.ndarray, end_idx: int, fit_params: dict | None = None) -> np.ndarray:
    v = (x[:, end_idx] - x[:, end_idx - 1]) / DT
    v_p = (x[:, end_idx - 1] - x[:, end_idx - 2]) / DT
    acc = (v - v_p) / DT
    T = v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)
    T_p = v_p / np.maximum(np.linalg.norm(v_p, axis=1, keepdims=True), 1e-9)
    b = np.cross(T_p, T)
    b_norm = np.linalg.norm(b, axis=1, keepdims=True)
    theta = np.arctan2(b_norm.squeeze(-1), np.sum(T_p * T, axis=1))
    safe = b_norm.squeeze(-1) > 1e-9
    omega = np.where(safe[:, None], (b / np.maximum(b_norm, 1e-9)) * (theta / DT)[:, None], 0.0)
    om_norm = np.linalg.norm(omega, axis=1)
    # clip scale
    over = om_norm > 10.0
    if over.any():
        omega[over] = omega[over] * (10.0 / np.maximum(om_norm[over, None], 1e-9))
        om_norm = np.linalg.norm(omega, axis=1)
    pred = np.empty_like(v)
    rotate = om_norm > 1e-6
    if rotate.any():
        axis_unit = omega[rotate] / np.maximum(om_norm[rotate, None], 1e-9)
        R_h = _rodrigues(axis_unit, om_norm[rotate] * H)
        dp = np.einsum("nij,nj->ni", R_h, v[rotate] * H)
        pred[rotate] = x[rotate, end_idx] + dp
    if (~rotate).any():
        idx = ~rotate
        pred[idx] = x[idx, end_idx] + H * v[idx] + 0.5 * H**2 * acc[idx]
    return pred


# ── C12. Wingbeat-corrected F0 (FFT pre-filter) ───────────────────────


def C12_wingbeat(x: np.ndarray, end_idx: int, fit_params: dict | None = None) -> np.ndarray:
    fp = fit_params or {"f_c": 8.0}
    f_c = float(fp["f_c"])
    T_obs = end_idx + 1  # 11
    bin_hz = 1.0 / (T_obs * DT)  # 2.27 Hz
    if f_c < bin_hz:
        # moving average fallback (window=11), apply to entire window
        x_clean = np.broadcast_to(x[:, :T_obs].mean(axis=1, keepdims=True), x[:, :T_obs].shape).copy()
    else:
        fft = np.fft.rfft(x[:, :T_obs], axis=1)
        freqs = np.fft.rfftfreq(T_obs, d=DT)
        mask = freqs <= f_c
        fft = fft * mask[None, :, None]
        x_clean = np.fft.irfft(fft, n=T_obs, axis=1)
    # apply F0 산식 on cleaned trajectory
    return _f0_apply(x_clean, end_idx, F0_D1, F0_PAR, F0_PERP)


# ── C13. Lévy-flight prior (deterministic mode = F0) ──────────────────


def C13_levy_prior(x: np.ndarray, end_idx: int, fit_params: dict | None = None) -> np.ndarray:
    # deterministic mode = F0 그대로 (§N+2 caveat #7). 측정 위해 포함.
    return _f0_apply(x, end_idx, F0_D1, F0_PAR, F0_PERP)


# ── C14. Trajectory KNN displacement ──────────────────────────────────


def C14_trajectory_knn(x: np.ndarray, end_idx: int, fit_params: dict | None = None) -> np.ndarray:
    """fit_params keys:
        'knn'  : sklearn KNeighborsRegressor (또는 fitted Faiss index wrapper)
        'k'    : int (already-set on knn)
        'train_R'      : (N_train, 3, 3) frame rotations (not strictly needed if knn handles)
        'train_origin' : (N_train, 3)
       Inference: build query 33D from x's v_last frame, knn.predict → disp_frame_avg, invert frame."""
    if fit_params is None or "knn" not in fit_params:
        # fallback: F0
        return _f0_apply(x, end_idx, F0_D1, F0_PAR, F0_PERP)
    knn = fit_params["knn"]
    R_s, origin_s = _v_last_frame(x, end_idx)
    # build query: traj in frame coords, flatten (T=11) × 3 = 33
    T_obs = end_idx + 1
    rel = x[:, :T_obs] - origin_s[:, None, :]  # (N, T, 3)
    traj_frame = np.einsum("nij,ntj->nti", R_s, rel)  # (N, T, 3)
    query = traj_frame.reshape(x.shape[0], -1)  # (N, 33)
    disp_frame_avg = knn.predict(query)  # (N, 3)
    pred = origin_s + np.einsum("nij,nj->ni", R_s.transpose(0, 2, 1), disp_frame_avg)
    return pred


# ── dispatch dict ──────────────────────────────────────────────────────


C01_TO_C14: dict[str, Any] = {
    "C01_helix": C01_helix,
    "C02_ctra": C02_ctra,
    "C03_ctrv": C03_ctrv,
    "C04_imm": C04_imm,
    "C05_per_regime_f0": C05_per_regime_f0,
    "C06_quintic_hermite": C06_quintic_hermite,
    "C07_jerk_quartic": C07_jerk_quartic,
    "C08_singer": C08_singer,
    "C09_kalman_smoother": C09_kalman_smoother,
    "C10_bishop_frame": C10_bishop_frame,
    "C11_se3_twist": C11_se3_twist,
    "C12_wingbeat_corrected": C12_wingbeat,
    "C13_levy_prior": C13_levy_prior,
    "C14_trajectory_knn": C14_trajectory_knn,
}
