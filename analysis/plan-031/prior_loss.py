"""plan-031 c2 — regime + class prior loss.

KL-style cross-entropy: model log P(anchor) 를 (regime_prior + class_prior) weighted target 으로 align.

prior_target[b, k] = regime_strength * regime_anchor_prior[b, k]    # train-fold P(anchor=k | regime[b])
                   + class_strength * class_prior_global[k]          # train-fold P(anchor=k)
                  (normalize row sum = 1)

loss = -Σ_k prior_target[b, k] * log_softmax(score, dim=-1)[b, k]   # CE 형

plan-004 carry: regime_strength=0.65, class_strength=0.45.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def regime_class_prior_loss(
    score: torch.Tensor,                 # (B, K) raw logit
    regime_anchor_prior: torch.Tensor,   # (B, K) regime r 의 P(anchor=k)
    class_prior_global: torch.Tensor,    # (K,) train-fold 전체 P(anchor=k)
    regime_strength: float = 0.65,
    class_strength: float = 0.45,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Returns scalar loss."""
    log_probs = F.log_softmax(score, dim=-1)                                # (B, K)
    target = regime_strength * regime_anchor_prior + class_strength * class_prior_global.unsqueeze(0)
    target = target / target.sum(dim=-1, keepdim=True).clamp_min(eps)        # normalize row
    loss = -(target * log_probs).sum(dim=-1).mean()
    return loss


def build_class_prior_global(
    gt_anchor_idx_train: np.ndarray,    # (N_train,) int, train-fold gt nearest anchor
    K: int = 14,
    laplace: float = 1.0,
) -> np.ndarray:
    """train-fold 전체 P(anchor=k), Laplace smoothed. Returns (K,) float32."""
    counts = np.bincount(gt_anchor_idx_train, minlength=K).astype(np.float64)
    counts += laplace
    prior = counts / counts.sum()
    return prior.astype(np.float32)


def build_regime_anchor_prior(
    regime_anchor_table: np.ndarray,    # (regime_count, K) train-fold lookup (plan-029 carry)
    regimes: np.ndarray,                # (N,) int, sample regime
) -> np.ndarray:
    """sample 별 regime 의 P(anchor=k) 추출. Returns (N, K) float32."""
    return regime_anchor_table[regimes].astype(np.float32)


# ── smoke ────────────────────────────────────────────────────────────


def _smoke() -> None:
    torch.manual_seed(20260524)
    rng = np.random.default_rng(20260524)
    B, K, R = 8, 14, 18

    # class prior global (Laplace)
    gt_idx_tr = rng.integers(0, K, size=200)
    class_prior = build_class_prior_global(gt_idx_tr, K=K, laplace=1.0)
    assert class_prior.shape == (K,)
    assert abs(class_prior.sum() - 1.0) < 1e-5

    # regime anchor table
    regime_table = rng.uniform(0.0, 1.0, size=(R, K)).astype(np.float32)
    regime_table /= regime_table.sum(axis=-1, keepdims=True)
    regimes = rng.integers(0, R, size=B)
    regime_prior = build_regime_anchor_prior(regime_table, regimes)
    assert regime_prior.shape == (B, K)
    for b in range(B):
        np.testing.assert_allclose(regime_prior[b].sum(), 1.0, atol=1e-5)

    # loss
    score = torch.randn(B, K, requires_grad=True)
    loss = regime_class_prior_loss(
        score,
        torch.from_numpy(regime_prior),
        torch.from_numpy(class_prior),
        regime_strength=0.65, class_strength=0.45,
    )
    assert loss.dim() == 0
    assert torch.isfinite(loss).item()
    loss.backward()
    assert torch.isfinite(score.grad).all()

    # perfect alignment 시 loss = entropy(target) (positive)
    perfect_score = torch.log(torch.from_numpy(regime_prior) * 0.65 + torch.from_numpy(class_prior).unsqueeze(0) * 0.45)
    loss_perf = regime_class_prior_loss(
        perfect_score,
        torch.from_numpy(regime_prior),
        torch.from_numpy(class_prior),
    )
    # perfect loss = H(target) (not 0 — KL divergence 가 0 인 경우 loss = entropy)
    assert torch.isfinite(loss_perf).item()
    print(f"[smoke] prior_loss OK — loss={loss.item():.4f}, perfect_align_loss={loss_perf.item():.4f}, "
          f"class_prior sum={class_prior.sum():.6f}, regime_prior[0] sum={regime_prior[0].sum():.6f}")


if __name__ == "__main__":
    _smoke()
