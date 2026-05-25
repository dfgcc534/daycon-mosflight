"""plan-030 c4 — GRUNetX2.

Q = anchor channel (per K=14, 64D), K/V = GRU step (per T=7, H+5).
head = attention context + sample summary (broadcast) + slim7 per anchor.
F0 = numpy frozen (no gradient). final = F0 + R_wfn @ (probs @ ANCHORS_A6).

plan-030 spec §3 그대로:
  - GRU: seq 97 -> hidden 196, 2 layer dropout 0.10, no bidirection (plan-029 X1 carry)
  - K/V projection: cat(gru_out (B,T,H), residual_a_kv (B,T,5)) -> Linear(H+5, attn_dim=128); K = V tied
  - Q projection: Linear(64, 128)
  - attention: scaled dot-product single head, softmax over T -> (B, K=14, 128)
  - head MLP: Linear(128 + (H+51=247) + 7 = 382, 384) -> SiLU -> Dropout 0.08 -> Linear(384, 1) -> (B, K=14)
  - softmax over K (temp=1.0) -> probs -> world_pred

forward signature:
  forward(seq_97, residual_a_kv, query_64, head_summary_51, slim7, F0_pred, R_wfn) -> (world_pred, probs)
  ANCHORS_A6 는 register_buffer 로 model 안에 carry.
"""
from __future__ import annotations

from math import sqrt

import torch
import torch.nn as nn


class GRUNetX2(nn.Module):
    """plan-030 GRU-attention residual injection model."""

    def __init__(
        self,
        seq_dim: int = 97,             # seq 95 + residual_a_gru 2
        query_dim: int = 64,
        head_summary_dim: int = 51,
        slim7_dim: int = 7,
        hidden: int = 196,
        attn_dim: int = 128,
        head_hidden: int = 384,
        gru_dropout: float = 0.10,     # plan-029 X1 carry
        head_dropout: float = 0.08,
        K: int = 14,
        residual_a_kv_dim: int = 5,
        anchors: torch.Tensor | None = None,  # (K, 3) Frenet codebook
    ):
        super().__init__()
        self.K = K
        self.hidden = hidden
        self.attn_dim = attn_dim
        self.residual_a_kv_dim = residual_a_kv_dim

        # === GRU encoder (plan-029 X1 carry: 2 layer dropout 0.10) ===
        self.gru = nn.GRU(
            seq_dim, hidden, num_layers=2,
            dropout=gru_dropout, batch_first=True,
        )

        # === K/V projection (tied) ===
        self.kv_proj = nn.Linear(hidden + residual_a_kv_dim, attn_dim)

        # === Q projection ===
        self.q_proj = nn.Linear(query_dim, attn_dim)

        # === head MLP ===
        head_in_dim = attn_dim + (hidden + head_summary_dim) + slim7_dim  # 128 + 247 + 7 = 382
        self.head_in_dim = head_in_dim
        self.head_mlp = nn.Sequential(
            nn.Linear(head_in_dim, head_hidden),
            nn.SiLU(),
            nn.Dropout(head_dropout),
            nn.Linear(head_hidden, 1),
        )

        # === ANCHORS_A6 frozen buffer ===
        if anchors is None:
            anchors = torch.zeros(K, 3, dtype=torch.float32)
        else:
            anchors = anchors.detach().to(dtype=torch.float32)
            assert anchors.shape == (K, 3), f"anchors shape {anchors.shape} != ({K}, 3)"
        self.register_buffer("ANCHORS_A6", anchors)

    def forward(
        self,
        seq_97: torch.Tensor,             # (B, T=7, 97)
        residual_a_kv: torch.Tensor,      # (B, T=7, 5)
        query_64: torch.Tensor,           # (B, K=14, 64)
        head_summary_51: torch.Tensor,    # (B, 51)
        slim7: torch.Tensor,              # (B, K=14, 7)
        F0_pred: torch.Tensor,            # (B, 3) frozen, world XYZ at +80ms
        R_wfn: torch.Tensor,              # (B, 3, 3) frozen
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (world_pred (B, 3), probs (B, K=14))."""
        B = seq_97.shape[0]
        K = self.K
        H = self.hidden

        # === GRU ===
        gru_out, _ = self.gru(seq_97)                                    # (B, T, H)
        gru_hidden_last = gru_out[:, -1, :]                              # (B, H)

        # === K/V proj (tied) ===
        kv_raw = torch.cat([gru_out, residual_a_kv], dim=-1)             # (B, T, H+5)
        kv = self.kv_proj(kv_raw)                                        # (B, T, attn_dim)

        # === Q proj ===
        q = self.q_proj(query_64)                                        # (B, K, attn_dim)

        # === Attention (scaled dot-product, single head, K=V tied) ===
        attn_logits = torch.einsum("bka,bta->bkt", q, kv) / sqrt(self.attn_dim)  # (B, K, T)
        attn_w = torch.softmax(attn_logits, dim=-1)                      # (B, K, T)
        attn_context = torch.einsum("bkt,bta->bka", attn_w, kv)          # (B, K, attn_dim)

        # === head MLP input ===
        sample_summary_full = torch.cat([gru_hidden_last, head_summary_51], dim=-1)  # (B, H+51)
        sample_bias = sample_summary_full.unsqueeze(1).expand(-1, K, -1)             # (B, K, H+51)
        head_in = torch.cat([attn_context, sample_bias, slim7], dim=-1)              # (B, K, 382)

        score = self.head_mlp(head_in).squeeze(-1)                       # (B, K) logit-scale
        probs = torch.softmax(score, dim=-1)                             # (B, K), temperature=1.0

        # === final: F0 + R_wfn @ (probs @ ANCHORS_A6) ===
        residual_frenet = probs @ self.ANCHORS_A6                        # (B, K) @ (K, 3) = (B, 3)
        residual_world = torch.einsum("bij,bj->bi", R_wfn, residual_frenet)  # (B, 3)
        world_pred = F0_pred + residual_world                            # (B, 3)

        return world_pred, probs


# ── smoke ────────────────────────────────────────────────────────────


def _smoke() -> None:
    torch.manual_seed(20260524)
    B, T, K = 4, 7, 14

    anchors = torch.randn(K, 3) * 0.01
    model = GRUNetX2(anchors=anchors)
    model.train()

    seq_97 = torch.randn(B, T, 97)
    residual_a_kv = torch.randn(B, T, 5)
    query_64 = torch.randn(B, K, 64)
    head_summary_51 = torch.randn(B, 51)
    slim7 = torch.randn(B, K, 7)
    F0_pred = torch.randn(B, 3)
    R_wfn = torch.eye(3).unsqueeze(0).expand(B, -1, -1).contiguous()

    world_pred, probs = model(seq_97, residual_a_kv, query_64, head_summary_51, slim7, F0_pred, R_wfn)

    # shape
    assert world_pred.shape == (B, 3), f"world_pred {world_pred.shape}"
    assert probs.shape == (B, K), f"probs {probs.shape}"
    # finite
    assert not torch.isnan(world_pred).any() and not torch.isinf(world_pred).any()
    assert not torch.isnan(probs).any() and not torch.isinf(probs).any()
    # probs valid distribution
    row_sum = probs.sum(dim=-1)
    assert torch.allclose(row_sum, torch.ones(B), atol=1e-5), f"row_sum={row_sum}"
    assert (probs >= 0).all() and (probs <= 1).all()
    # head_in_dim
    assert model.head_in_dim == 382, f"head_in_dim={model.head_in_dim}"
    # ANCHORS_A6 frozen (buffer, requires_grad=False)
    assert not model.ANCHORS_A6.requires_grad

    # gradient flow
    loss = probs.sum() + world_pred.sum()
    loss.backward()
    grad_norms = [(n, p.grad.norm().item()) for n, p in model.named_parameters() if p.grad is not None]
    assert len(grad_norms) > 0, "no parameter has gradient"
    for n, g in grad_norms:
        assert g >= 0, f"{n} grad NaN"

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[smoke] GRUNetX2 OK — world_pred={world_pred.shape}, probs={probs.shape}, "
          f"head_in_dim={model.head_in_dim}, total_params={n_params}, "
          f"row_sum_mean={row_sum.mean():.6f}")


if __name__ == "__main__":
    _smoke()
