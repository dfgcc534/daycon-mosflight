"""plan-024 c2 — anchor-vocab 묶음 builder (§4.2).

per past step t (t ∈ {4..10}) 의 F0 residual 을 14-anchor codebook 으로 분해.
4 묶음 출력 (v1.1: F3 anchor-projection 제거 — redundancy A1+A2):

  F   (N, T, K)   : anchor-vocab soft assignment = softmax(-‖a_k - r_t‖ / τ_past)
  G   (N, T)      : magnitude = ‖r_t‖
  H   (N, T, K)   : top1 one-hot = argmin_k ‖a_k - r_t‖
  F2  (N, T)      : log-magnitude = log(1 + ‖r_t‖ / 0.005)

**sign convention (audit A 통일)**: residual = `actual - pred` (= "관측이 예측보다
어디에 떨어졌는지"). plan-021 L2 의 `pred - actual` 와 *opposite*, plan-022
build_soft_label_with_tau 의 `gt - pred` 와 정합.

**τ_past (default 0.003)**: output τ_cls (0.001) 와 *별도* hyperparam. past F0
residual magnitude 5~20mm 스케일 매칭 (anchor radius 0.005m 의 0.6×).
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


def build(
    X: np.ndarray,                # (N, 11, 3) float, world coord
    R_wfn: np.ndarray,            # (N, 3, 3) float, per-sample Frenet basis (§4.0)
    anchors: np.ndarray,          # (K, 3) float, ANCHORS_A6 Frenet (K=14)
    f0_baseline_fn: Callable[[np.ndarray, int], np.ndarray],
    tau_past: float = 0.003,
    t_range: tuple[int, int] = (4, 11),
) -> dict[str, np.ndarray]:
    """anchor-vocab 묶음 build (§4.2).

    Returns dict:
        F  : (N, T_seq, K) float32 — soft assignment
        G  : (N, T_seq)     float32 — magnitude (3D Frenet norm)
        H  : (N, T_seq, K) float32 — top1 one-hot (argmin distance)
        F2 : (N, T_seq)     float32 — log-magnitude
    """
    t_start, t_stop = t_range
    T_seq = t_stop - t_start
    N = X.shape[0]
    K = anchors.shape[0]

    X = X.astype(np.float64)
    R_t = np.transpose(R_wfn, (0, 2, 1)).astype(np.float64)   # (N, 3, 3) world→Frenet
    anchors_f = anchors.astype(np.float64)

    F = np.zeros((N, T_seq, K), dtype=np.float32)
    G = np.zeros((N, T_seq), dtype=np.float32)
    H = np.zeros((N, T_seq, K), dtype=np.float32)
    F2 = np.zeros((N, T_seq), dtype=np.float32)

    for i, t in enumerate(range(t_start, t_stop)):
        # sub_x_t = past 3 step (t-4, t-3, t-2), f0_baseline 의 80ms 미래 = X[:, t]
        sub_x_t = X[:, t - 4:t - 1, :]                        # (N, 3, 3)
        pred_t = f0_baseline_fn(sub_x_t, end_idx=2)           # (N, 3) world (plan-020 carry)
        actual_t = X[:, t]                                     # (N, 3) world
        residual_w_t = actual_t - pred_t.astype(np.float64)   # ★ sign 통일
        residual_t_frenet = np.einsum("nij,nj->ni", R_t, residual_w_t)  # (N, 3) Frenet

        # F: softmax(-‖a_k - r_t‖ / τ_past)
        diff = anchors_f[None, :, :] - residual_t_frenet[:, None, :]   # (N, K, 3)
        dist = np.linalg.norm(diff, axis=2)                            # (N, K)
        logits = -dist / tau_past
        logits = logits - logits.max(axis=1, keepdims=True)            # stability
        ex = np.exp(logits)
        F[:, i, :] = (ex / ex.sum(axis=1, keepdims=True)).astype(np.float32)

        # G: magnitude
        mag = np.linalg.norm(residual_t_frenet, axis=1)                # (N,)
        G[:, i] = mag.astype(np.float32)

        # H: top1 one-hot (argmin distance, = argmax soft)
        top1 = dist.argmin(axis=1)                                      # (N,)
        H_t = np.eye(K, dtype=np.float32)[top1]                         # (N, K)
        H[:, i, :] = H_t

        # F2: log-magnitude
        F2[:, i] = np.log1p(mag / 0.005).astype(np.float32)

    return {"F": F, "G": G, "H": H, "F2": F2}


# ── smoke (__main__) ───────────────────────────────────────────────────


if __name__ == "__main__":
    rng = np.random.default_rng(20260521)
    N = 50
    K = 14
    X = rng.standard_normal((N, 11, 3)).astype(np.float64) * 0.01
    R_wfn = np.tile(np.eye(3, dtype=np.float32)[None], (N, 1, 1))
    anchors = (rng.standard_normal((K, 3)).astype(np.float32) * 0.005)

    def fake_f0(sub_x: np.ndarray, end_idx: int) -> np.ndarray:
        # linear extrapolation: pred = x[end_idx] + (x[end_idx] - x[end_idx-1])
        return sub_x[:, end_idx] + (sub_x[:, end_idx] - sub_x[:, end_idx - 1])

    out = build(X, R_wfn, anchors, fake_f0, tau_past=0.003)
    assert out["F"].shape == (N, 7, K)
    assert out["G"].shape == (N, 7)
    assert out["H"].shape == (N, 7, K)
    assert out["F2"].shape == (N, 7)
    # sum=1 invariance for F
    assert np.allclose(out["F"].sum(axis=-1), 1.0, atol=1e-5)
    # one-hot invariance for H
    assert np.allclose(out["H"].sum(axis=-1), 1.0, atol=1e-5)
    assert np.isfinite(out["F"]).all()
    assert np.isfinite(out["G"]).all()
    assert np.isfinite(out["F2"]).all()
    print(f"[smoke] anchor_vocab build N={N} K={K} T=7 ✓")
    print(f"        F sum=1 invariance ✓, H one-hot ✓, F2 finite ✓")
