"""plan-031 c1 — pairwise margin loss.

gt anchor 의 logit 이 other K-1 anchor 보다 최소 margin 만큼 크도록 강제:
    loss = mean_b [ mean_{j ≠ gt}( max(0, margin - (score[gt] - score[j])) ) ]

logit gap (score[gt] - score[j]) >= margin 이면 violation = 0,
gap < margin 이면 violation = margin - gap.

plan-030 G3 FAIL 의 root cause 3 (score_std 신장 실패) 직접 해결:
soft CE 만으로는 logit magnitude 못 키우는 문제를 hinge-type loss 로 직접 강제.
"""
from __future__ import annotations

import torch


def pairwise_margin_loss(
    score: torch.Tensor,           # (B, K) raw logit
    gt_anchor_idx: torch.Tensor,   # (B,) int64 — gt nearest anchor
    margin: float = 0.12,
) -> torch.Tensor:
    """Returns scalar loss.

    Per-sample loss = mean_{j ≠ gt}( clamp(margin - (score[gt] - score[j]), min=0) )
    """
    B, K = score.shape
    gt_score = score.gather(1, gt_anchor_idx.unsqueeze(1))    # (B, 1)
    gap = gt_score - score                                    # (B, K), gap[b, gt]=0
    violation = torch.clamp(margin - gap, min=0.0)            # (B, K)
    # mask gt index out (gt vs gt 자체 비교 = margin > 0 이라 잘못 박힘)
    mask = torch.ones_like(violation)
    mask.scatter_(1, gt_anchor_idx.unsqueeze(1), 0.0)
    loss_per_sample = (violation * mask).sum(dim=1) / (K - 1)  # (B,)
    return loss_per_sample.mean()


def compute_gt_anchor_idx(
    gt: torch.Tensor,            # (B, 3) world frame, float
    F0_pred: torch.Tensor,       # (B, 3) world frame
    R_wfn: torch.Tensor,         # (B, 3, 3) Frenet basis
    ANCHORS_A6: torch.Tensor,    # (K, 3) Frenet codebook
) -> torch.Tensor:
    """gt nearest anchor index = argmin_k ||R_wfn^T @ (gt - F0) - ANCHORS_A6[k]||.

    Returns: (B,) int64.
    """
    residual_world = gt - F0_pred                                              # (B, 3)
    R_t = R_wfn.transpose(-2, -1)                                              # (B, 3, 3)
    residual_frenet = torch.einsum("bij,bj->bi", R_t, residual_world)          # (B, 3)
    diff = ANCHORS_A6.unsqueeze(0) - residual_frenet.unsqueeze(1)              # (B, K, 3)
    dist = diff.norm(dim=-1)                                                   # (B, K)
    return dist.argmin(dim=1)                                                  # (B,)


# ── smoke ────────────────────────────────────────────────────────────


def _smoke() -> None:
    torch.manual_seed(20260524)
    B, K = 4, 14
    score = torch.randn(B, K, requires_grad=True)
    gt_idx = torch.randint(0, K, (B,))

    loss = pairwise_margin_loss(score, gt_idx, margin=0.12)
    assert loss.dim() == 0
    assert torch.isfinite(loss).item()
    loss.backward()
    assert score.grad is not None
    assert torch.isfinite(score.grad).all()

    # gt anchor 가 dominant 인 경우 loss → 0 확인
    perfect_score = torch.zeros(B, K)
    for b in range(B):
        perfect_score[b, gt_idx[b]] = 1.0
    loss_perfect = pairwise_margin_loss(perfect_score, gt_idx, margin=0.12)
    # gap[gt, j] = 1.0 - 0.0 = 1.0 >= 0.12 → violation 0
    assert loss_perfect.item() < 1e-6, f"loss_perfect={loss_perfect.item()}"

    # uniform score (logit gap 0) 인 경우 loss = margin
    uniform_score = torch.zeros(B, K)
    loss_uniform = pairwise_margin_loss(uniform_score, gt_idx, margin=0.12)
    assert abs(loss_uniform.item() - 0.12) < 1e-6, f"loss_uniform={loss_uniform.item()}"

    # compute_gt_anchor_idx smoke
    gt_world = torch.randn(B, 3) * 0.05
    F0 = torch.randn(B, 3) * 0.05
    R = torch.eye(3).unsqueeze(0).expand(B, -1, -1).contiguous()
    anchors = torch.randn(K, 3) * 0.01
    idx = compute_gt_anchor_idx(gt_world, F0, R, anchors)
    assert idx.shape == (B,)
    assert (idx >= 0).all() and (idx < K).all()
    assert idx.dtype == torch.int64

    print(f"[smoke] pairwise_loss OK — loss={loss.item():.4f}, "
          f"perfect={loss_perfect.item():.2e}, uniform={loss_uniform.item():.4f}, "
          f"gt_idx={idx.tolist()}")


if __name__ == "__main__":
    _smoke()
