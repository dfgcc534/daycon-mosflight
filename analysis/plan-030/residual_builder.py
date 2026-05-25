"""plan-030 c1 — residual_builder.

3-pair 잔차 block 산출 (plan-030 spec §2.1 그대로):

  (a) raw(t+2) − F0_pred(t)             — sample-only (anchor-invariant), step 별
  (b) raw(t+2) − anchor_world_k(t)      — sample × anchor, step 별
                                          anchor_world_k(t) = F0_pred(t) + R_wfn @ anchors[k]

5 coord 분해 = [XY_norm, Z_signed, Frenet along/across/vert]
step align: seq 와 동일 t_wall = {-6, ..., 0} (i=0..6, length 7)
  - i=0..4 (t_wall=-6..-2) → raw(t+2) ∈ {raw(-4)..raw(0)} 관측됨, 값 채움
  - i=5,6 (t_wall=-1,0) → raw(t+2) ∈ {+1,+2} 미관측, zero-pad

R_wfn = end_idx=10 (t_wall=0) Frenet basis 1개, 모든 7 step 의 분해에 동일 사용 (step-invariant).
잔차 (c) F0 − anchor 는 anchor_spec + Bz/Tz 와 redundant 라 drop (plan-030 §1.2).
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


def build_residuals(
    X: np.ndarray,
    R_wfn: np.ndarray,
    anchors: np.ndarray,
    f0_baseline_fn: Callable[[np.ndarray, int], np.ndarray],
) -> dict[str, np.ndarray]:
    """plan-030 §2.1 residual builder.

    Args:
        X:               (N, 11, 3) float, world raw observations (t_wall = -10 ~ 0).
        R_wfn:           (N, 3, 3) float, end_idx=10 Frenet basis (step-invariant).
        anchors:         (K=14, 3) float, ANCHORS_A6 Frenet codebook.
        f0_baseline_fn:  analysis.plan_020.baseline_f0.f0_baseline (signature: (x: (N,T,3), end_idx: int) -> (N, 3)).

    Returns dict:
        "residual_a":     (N, 7, 5) float32 — raw(t+2) - F0_pred(t), 5 coord
        "residual_a_gru": (N, 7, 2) float32 — XY_norm + Z_signed only (GRU input concat)
        "residual_b":     (N, K=14, 7, 5) float32 — raw(t+2) - anchor_world_k(t)
    """
    X = np.asarray(X, dtype=np.float64)
    R_wfn_f = np.asarray(R_wfn, dtype=np.float64)
    R_wfn_T = R_wfn_f.transpose(0, 2, 1)              # (N, 3, 3)
    anchors_f = np.asarray(anchors, dtype=np.float64)  # (K, 3)

    N, T_obs, _ = X.shape
    K = anchors_f.shape[0]
    assert T_obs == 11, f"X must have T=11 (t_wall = -10..0), got {T_obs}"

    residual_a = np.zeros((N, 7, 5), dtype=np.float32)
    residual_b = np.zeros((N, K, 7, 5), dtype=np.float32)

    for i in range(7):
        t_wall = -6 + i                                # {-6, ..., 0}
        t_idx = t_wall + 10                            # absolute X index, {4, ..., 10}
        t_target_idx = t_idx + 2                       # raw(t+2) absolute index

        # F0_pred(t) — plan-020 baseline f0 호출 (sub_x = 3 step, end_idx=2)
        sub_x = X[:, t_idx - 2:t_idx + 1, :]           # (N, 3, 3)
        F0_pred_t = np.asarray(f0_baseline_fn(sub_x, 2), dtype=np.float64)  # (N, 3) world

        if t_target_idx <= 10:
            raw_t2 = X[:, t_target_idx, :]             # (N, 3) world
            delta_a = raw_t2 - F0_pred_t               # (N, 3)
            # anchor_world_k(t) = F0_pred(t) + R_wfn @ anchors[k]
            anchor_world_k = (
                F0_pred_t[:, None, :]
                + np.einsum("nij,kj->nki", R_wfn_f, anchors_f)
            )                                          # (N, K, 3)
            delta_b_per_k = raw_t2[:, None, :] - anchor_world_k  # (N, K, 3)
        else:
            delta_a = np.zeros((N, 3), dtype=np.float64)
            delta_b_per_k = np.zeros((N, K, 3), dtype=np.float64)

        # 5 coord 분해 (R_wfn = end_idx=10, step-invariant)
        residual_a[:, i, 0] = np.linalg.norm(delta_a[:, :2], axis=1).astype(np.float32)  # XY_norm
        residual_a[:, i, 1] = delta_a[:, 2].astype(np.float32)                            # Z_signed
        residual_a[:, i, 2:5] = np.einsum("nij,nj->ni", R_wfn_T, delta_a).astype(np.float32)  # Frenet

        residual_b[:, :, i, 0] = np.linalg.norm(delta_b_per_k[:, :, :2], axis=-1).astype(np.float32)
        residual_b[:, :, i, 1] = delta_b_per_k[:, :, 2].astype(np.float32)
        residual_b[:, :, i, 2:5] = np.einsum(
            "nij,nkj->nki", R_wfn_T, delta_b_per_k
        ).astype(np.float32)

    residual_a = np.nan_to_num(residual_a, nan=0.0, posinf=1e3, neginf=-1e3)
    residual_b = np.nan_to_num(residual_b, nan=0.0, posinf=1e3, neginf=-1e3)

    residual_a_gru = residual_a[:, :, :2].copy()       # XY_norm + Z_signed only

    return {
        "residual_a": residual_a,
        "residual_a_gru": residual_a_gru,
        "residual_b": residual_b,
    }


# ── smoke (__main__) ───────────────────────────────────────────────────


if __name__ == "__main__":
    _PLAN020 = _THIS.parent / "plan-020"
    _spec = importlib.util.spec_from_file_location("p020_baseline_f0", _PLAN020 / "baseline_f0.py")
    _bf = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_bf)
    # plan-020 entry: f0_baseline(x: (N, T, 3), end_idx: int) -> (N, 3)
    rng = np.random.default_rng(20260524)
    N = 8
    X = rng.standard_normal((N, 11, 3)).astype(np.float32) * 0.5
    R_wfn = np.tile(np.eye(3, dtype=np.float32)[None], (N, 1, 1))
    anchors = rng.standard_normal((14, 3)).astype(np.float32) * 0.01

    out = build_residuals(X, R_wfn, anchors, _bf.f0_baseline)

    assert out["residual_a"].shape == (N, 7, 5)
    assert out["residual_a_gru"].shape == (N, 7, 2)
    assert out["residual_b"].shape == (N, 14, 7, 5)

    # invalid steps i=5,6 zero check
    assert np.all(out["residual_a"][:, 5:7, :] == 0.0)
    assert np.all(out["residual_b"][:, :, 5:7, :] == 0.0)

    # finite
    for k, v in out.items():
        assert not np.isnan(v).any() and not np.isinf(v).any(), f"{k} has NaN/Inf"
        assert v.dtype == np.float32

    print(f"[smoke] residual_builder OK — N={N}, shapes "
          f"a={out['residual_a'].shape}, a_gru={out['residual_a_gru'].shape}, "
          f"b={out['residual_b'].shape}, zero-pad i=5,6 ✓")
