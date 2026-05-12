"""plan-008 c3: Extended candidate families — 5 families × 15 templates.

spec @ plans/plan-008-candidate-redefine-corrector-redesign.md §5.2.1~§5.2.4.

Families (Family 4 per_regime drop, snap drop — v2.3):
  - 1 trig (4 templates):       rot_low_080, rot_mid_100, rot_mid_120, rot_high_150
  - 2 arc (3):                  arc_continue, arc_decel, arc_accel
  - 3 frenet_serret_3d (3):     fs_3d_planar, fs_3d_low_torsion, fs_3d_binormal
  - 5 higher_order (2):         jerk_acc_par_120, multi_step_rk2  (snap drop)
  - 6 cross_term (3):           speed_slope_d1_120, speed_norm_acc_par, omega_speed

API:
  * Per-family `make_*_candidates(x, end_idx, horizon=2) -> (N, K_fam, 3)` batch
  * Per-family `*_coord_func(x, end_idx, *, horizon, spec) -> (N, 1, 3)` per-spec
    — greedy set-cover §5.3 의 `template_pool` 호출 unit.
  * `build_template_pool() -> list[(name, spec, coord_func)]` — greedy 진입점.
  * `get_extended_candidates_list(kept_indices, kept_families) -> list[CandidateSpec]`
  * `make_candidates_extended(x, end_idx, horizon, kept_indices, kept_families) -> (N, K, 3)`
"""
from __future__ import annotations

import numpy as np

from src.pb_0_6822 import selector
from src.pb_0_6822.selector import CandidateSpec


# ── Family 1: Trig (rotation) — 4 templates ────────────────────────────────
TRIG_CANDIDATES = [
    CandidateSpec("rot_low_080",  d1=2.0, par=0.0, perp=0.0, omega_scale=0.8, family_id=1),
    CandidateSpec("rot_mid_100",  d1=2.0, par=0.0, perp=0.0, omega_scale=1.0, family_id=1),
    CandidateSpec("rot_mid_120",  d1=2.0, par=0.0, perp=0.0, omega_scale=1.2, family_id=1),
    CandidateSpec("rot_high_150", d1=2.0, par=0.0, perp=0.0, omega_scale=1.5, family_id=1),
]


def rot_coord_func(x, end_idx, *, horizon=2, spec):
    """Per-spec coord — (N, 1, 3). R(omega_z · omega_scale · horizon)·d1 + p0."""
    p0, d1, _ = selector.motion_terms(x, end_idx)
    d2 = x[:, end_idx - 1] - x[:, end_idx - 2]
    omega_z = np.arctan2(
        d2[:, 0] * d1[:, 1] - d2[:, 1] * d1[:, 0],
        d2[:, 0] * d1[:, 0] + d2[:, 1] * d1[:, 1],
    )
    theta = omega_z * spec.omega_scale * horizon
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    d1_rot = np.stack([
        cos_t * d1[:, 0] - sin_t * d1[:, 1],
        sin_t * d1[:, 0] + cos_t * d1[:, 1],
        spec.z_scale * d1[:, 2],
    ], axis=1)
    pred = p0 + d1_rot * horizon
    return pred[:, None, :].astype(np.float32)


def make_rot_candidates(x, end_idx, horizon=2):
    return np.concatenate(
        [rot_coord_func(x, end_idx, horizon=horizon, spec=s) for s in TRIG_CANDIDATES],
        axis=1,
    ).astype(np.float32)


# ── Family 2: Arc — 3 templates ─────────────────────────────────────────────
ARC_CANDIDATES = [
    CandidateSpec("arc_continue", d1=2.0, par=0.0, perp=0.0, arc_curvature=1.0, family_id=2),
    CandidateSpec("arc_decel",    d1=2.0, par=0.0, perp=0.0, arc_curvature=0.9, family_id=2),
    CandidateSpec("arc_accel",    d1=2.0, par=0.0, perp=0.0, arc_curvature=1.1, family_id=2),
]


def arc_coord_func(x, end_idx, *, horizon=2, spec):
    """Per-spec — 3-pt circular arc fit + arclength extrapolation. (N, 1, 3)."""
    p0, d1, _ = selector.motion_terms(x, end_idx)
    d2 = x[:, end_idx - 1] - x[:, end_idx - 2]
    omega_z = np.arctan2(
        d2[:, 0] * d1[:, 1] - d2[:, 1] * d1[:, 0],
        d2[:, 0] * d1[:, 0] + d2[:, 1] * d1[:, 1],
    )
    theta = omega_z * spec.arc_curvature * horizon
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    arc_step = np.stack([
        cos_t * d1[:, 0] - sin_t * d1[:, 1],
        sin_t * d1[:, 0] + cos_t * d1[:, 1],
        d1[:, 2],
    ], axis=1) * horizon
    pred = p0 + arc_step
    return pred[:, None, :].astype(np.float32)


def make_arc_candidates(x, end_idx, horizon=2):
    return np.concatenate(
        [arc_coord_func(x, end_idx, horizon=horizon, spec=s) for s in ARC_CANDIDATES],
        axis=1,
    ).astype(np.float32)


# ── Family 3: Frenet-Serret 3D (v2.3 binormal frame) — 3 templates ─────────
FS3D_CANDIDATES = [
    CandidateSpec("fs_3d_planar",      d1=2.0, par=0.0, perp=0.0, z_scale=0.0, family_id=3),
    CandidateSpec("fs_3d_low_torsion", d1=2.0, par=0.0, perp=0.0, z_scale=0.5, family_id=3),
    CandidateSpec("fs_3d_binormal",    d1=2.0, par=0.0, perp=0.0, z_scale=1.0, family_id=3),
]


def fs3d_coord_func(x, end_idx, *, horizon=2, spec):
    """Frenet-Serret 3D — T/N/B local frame, binormal 진폭. (N, 1, 3).

    T = unit(d1) ;  N = unit(d2 − (d2·T)T) ;  B = T × N
    K = ||d2_perp|| / ||d1||²  (kinematic curvature)
    pred = p0 + d1·horizon + spec.z_scale · ||d1|| · horizon · K · B
    """
    p0, d1, _ = selector.motion_terms(x, end_idx)
    d2 = x[:, end_idx - 1] - x[:, end_idx - 2]
    eps = 1e-9
    d1_norm = np.linalg.norm(d1, axis=1, keepdims=True) + eps
    T = d1 / d1_norm
    d2_par_scalar = np.einsum("ij,ij->i", d2, T)[:, None]
    d2_par = d2_par_scalar * T
    d2_perp = d2 - d2_par
    N_norm = np.linalg.norm(d2_perp, axis=1, keepdims=True) + eps
    N = d2_perp / N_norm
    B = np.cross(T, N)
    kappa = (N_norm[:, 0] / (d1_norm[:, 0] ** 2 + eps))[:, None]
    binormal_term = spec.z_scale * d1_norm * horizon * kappa * B
    pred = p0 + d1 * horizon + binormal_term
    return pred[:, None, :].astype(np.float32)


def make_fs3d_candidates(x, end_idx, horizon=2):
    return np.concatenate(
        [fs3d_coord_func(x, end_idx, horizon=horizon, spec=s) for s in FS3D_CANDIDATES],
        axis=1,
    ).astype(np.float32)


# ── Family 5: Higher-order (snap drop) — 2 templates ────────────────────────
HIGHER_ORDER_CANDIDATES = [
    CandidateSpec("jerk_acc_par_120", d1=2.0, par=1.2, perp=0.0, jerk=0.1, family_id=5),
    CandidateSpec("multi_step_rk2",   d1=2.0, par=0.0, perp=0.0, d2=1.0,   family_id=5),
]


def higher_order_coord_func(x, end_idx, *, horizon=2, spec):
    """Per-spec dispatch:
      - spec.jerk > 0  → jerk-augmented (3차 차분 jerk_vec)
      - 그 외          → 11-step constant-coef integration (spec d1/d2 배율)
    """
    p0, d1, acc = selector.motion_terms(x, end_idx)
    if spec.jerk > 0:
        d2 = x[:, end_idx - 1] - x[:, end_idx - 2]
        # 진짜 jerk = 3차 차분 (boundary: end_idx ≥ 4)
        jerk_vec = (
            x[:, end_idx - 1] - 3 * x[:, end_idx - 2]
            + 3 * x[:, end_idx - 3] - x[:, end_idx - 4]
        )
        pred = (
            p0 + spec.par * d1 * horizon
            + 0.5 * d2 * horizon ** 2
            + (spec.jerk / 6.0) * jerk_vec * horizon ** 3
        )
    else:
        d2 = x[:, end_idx - 1] - x[:, end_idx - 2]
        n_steps = 11
        dt = horizon / n_steps
        pred = p0.copy()
        d1_step = spec.d1 * d1 / max(horizon, 1e-9)
        d2_step = spec.d2 * d2 / max(horizon ** 2, 1e-9)
        for _ in range(n_steps):
            pred = pred + d1_step * dt + 0.5 * d2_step * dt ** 2
    return pred[:, None, :].astype(np.float32)


def make_higher_order_candidates(x, end_idx, horizon=2):
    return np.concatenate(
        [higher_order_coord_func(x, end_idx, horizon=horizon, spec=s) for s in HIGHER_ORDER_CANDIDATES],
        axis=1,
    ).astype(np.float32)


# ── Family 6: Cross-term — 3 templates ──────────────────────────────────────
CROSS_TERM_CANDIDATES = [
    CandidateSpec("speed_slope_d1_120", d1=2.0, par=1.2, perp=0.0, omega_scale=0.5, family_id=6),
    CandidateSpec("speed_norm_acc_par", d1=2.0, par=1.0, perp=0.0, jerk=0.3,        family_id=6),
    CandidateSpec("omega_speed",        d1=2.0, par=1.0, perp=0.0, omega_scale=1.0, family_id=6),
]


def cross_term_coord_func(x, end_idx, *, horizon=2, spec):
    """Feature × motion cross-term — per-sample 적응.
       speed_slope ≈ (||d1|| − ||d1_prev||) / ||d1_prev||
       cos_turn   = unit(d1) · unit(d1_prev)
       cross_term = spec.par · speed_slope · d1 + spec.omega_scale · cos_turn · d1
       pred = p0 + d1·horizon + cross_term · horizon
    """
    p0, d1, _ = selector.motion_terms(x, end_idx)
    eps = 1e-9
    d1_prev = x[:, end_idx - 1] - x[:, end_idx - 2]
    d1_norm = np.linalg.norm(d1, axis=1) + eps
    d1_prev_norm = np.linalg.norm(d1_prev, axis=1) + eps
    speed_slope = (d1_norm - d1_prev_norm) / d1_prev_norm
    cos_turn = np.einsum("ij,ij->i", d1, d1_prev) / (d1_norm * d1_prev_norm)
    cross_term = (
        spec.par * speed_slope[:, None] * d1
        + spec.omega_scale * cos_turn[:, None] * d1
    )
    pred = p0 + d1 * horizon + cross_term * horizon
    return pred[:, None, :].astype(np.float32)


def make_cross_term_candidates(x, end_idx, horizon=2):
    return np.concatenate(
        [cross_term_coord_func(x, end_idx, horizon=horizon, spec=s) for s in CROSS_TERM_CANDIDATES],
        axis=1,
    ).astype(np.float32)


# ── §5.2.4 build_template_pool — greedy set-cover 진입점 ─────────────────────
def build_template_pool() -> list:
    """Returns list[(name, spec, coord_func)] — len = 4+3+3+2+3 = 15."""
    return (
        [(s.name, s, rot_coord_func)         for s in TRIG_CANDIDATES]
        + [(s.name, s, arc_coord_func)        for s in ARC_CANDIDATES]
        + [(s.name, s, fs3d_coord_func)       for s in FS3D_CANDIDATES]
        + [(s.name, s, higher_order_coord_func) for s in HIGHER_ORDER_CANDIDATES]
        + [(s.name, s, cross_term_coord_func) for s in CROSS_TERM_CANDIDATES]
    )


# ── §5.2.2 integration ─────────────────────────────────────────────────────
FAMILY_TO_SPECS: dict[str, list[CandidateSpec]] = {
    "trig": TRIG_CANDIDATES,
    "arc": ARC_CANDIDATES,
    "frenet_serret_3d": FS3D_CANDIDATES,
    "higher_order": HIGHER_ORDER_CANDIDATES,
    "cross_term": CROSS_TERM_CANDIDATES,
}

FAMILY_TO_MAKE = {
    "trig": make_rot_candidates,
    "arc": make_arc_candidates,
    "frenet_serret_3d": make_fs3d_candidates,
    "higher_order": make_higher_order_candidates,
    "cross_term": make_cross_term_candidates,
}


def get_extended_candidates_list(
    kept_indices: list[int],
    kept_families: list[str] | set[str],
) -> list[CandidateSpec]:
    """기존 pruned base + family-on/off filter 적용 new specs → CandidateSpec list."""
    base_kept = [selector.CANDIDATES[i] for i in kept_indices]
    new_specs: list[CandidateSpec] = []
    for fam in ("trig", "arc", "frenet_serret_3d", "higher_order", "cross_term"):
        if fam in kept_families:
            new_specs.extend(FAMILY_TO_SPECS[fam])
    return base_kept + new_specs


def make_candidates_extended(
    x: np.ndarray,
    end_idx: int,
    horizon: int = 2,
    kept_indices: list[int] | None = None,
    kept_families: list[str] | set[str] | None = None,
    base_maker=None,
) -> np.ndarray:
    """기존 27 (pruned by `kept_indices`) + 새 family 후보 좌표 concat. (N, K, 3).

    `base_maker`: callable to produce base 27 candidates. Defaults to
    `selector.make_candidates` at *function call time*. **caller monkey-patches
    `selector.make_candidates` to point at THIS function 시에는 반드시 explicit
    base_maker (= 원본 selector.make_candidates) 를 전달 — recursion 회피.**
    """
    if kept_indices is None:
        kept_indices = list(range(len(selector.CANDIDATES)))
    if kept_families is None:
        kept_families = []
    if base_maker is None:
        base_maker = selector.make_candidates
    cands_base_27 = base_maker(x, end_idx, horizon)
    cands_base_kept = cands_base_27[:, kept_indices, :]
    new_cands_list = [cands_base_kept]
    for fam in ("trig", "arc", "frenet_serret_3d", "higher_order", "cross_term"):
        if fam in kept_families:
            new_cands_list.append(FAMILY_TO_MAKE[fam](x, end_idx, horizon))
    return np.concatenate(new_cands_list, axis=1).astype(np.float32)
