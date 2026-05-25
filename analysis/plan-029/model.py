"""plan-029 c4 — GRUNetX1.

4 lever (a)(b)(c)(d) 통합 attention model:
  (a) query enrichment    — cand_ext (B, K=14, 165) 외부 사전 산출 후 주입
  (b) anchor embedding 학습 — nn.Parameter(14, 8) init randn * 0.1
  (c) key anchor-conditional — key_anchor[b,k,t,:] = key[b,t,:] + anchor_key_proj(anchor_embed[k,:])
  (d) head raw skip 차단    — head_in = event_ctx (196) only, head = Linear(196, 1)

plan-024 backbone (GRU + query_mlp) 는 architecture template carry (design pattern only).
backbone.head + FWD wrapper class 모두 import X.

forward signature: forward(self, seq, cand_ext) -> score (B, K).
"""
from __future__ import annotations

from math import sqrt

import torch
import torch.nn as nn


class GRUNetX1(nn.Module):
    """plan-029 cross-attention model with head_skip_mode option.

    head_skip_mode:
      'none'           : head_in = event_ctx (H)                    — X1 cell (default, lever d 차단)
      'no_anchor_spec' : head_in = concat(event_ctx, cand_no_spec)  — X2 cell (anchor identity 9D 만 차단)
      'full'           : head_in = concat(event_ctx, cand_ext)      — X3 cell (PB framework 식 raw skip 부활)

    cand_ext layout (plan-024 cand_builder 150 + plan-029 extension 15 = 165):
      [:, :, 0:3]    par/perp/dist        (sample × anchor)
      [:, :, 3:12]   anchor spec          (anchor-static = "anchor data")
      [:, :, 12:140] ctx broadcast        (sample × broadcast, K 축 동일)
      [:, :, 140:150] interactions        (sample × anchor)
      [:, :, 150:165] anchor_query_extend (sample × anchor, 15 ch)

    forward signature: forward(self, seq, cand_ext) -> score (B, K) 단일 tensor
    """

    ANCHOR_SPEC_START = 3
    ANCHOR_SPEC_END = 12  # exclusive — anchor-static 9D index range

    def __init__(
        self,
        seq_dim: int = 95,
        cand_in_dim: int = 165,
        hidden: int = 196,
        anchor_embed_dim: int = 8,
        anchor_embed_init_scale: float = 0.1,
        gru_dropout: float = 0.10,
        K: int = 14,
        head_skip_mode: str = "none",
    ):
        super().__init__()
        assert head_skip_mode in ("none", "no_anchor_spec", "full")
        self.K = K
        self.hidden = hidden
        self.anchor_embed_dim = anchor_embed_dim
        self.cand_in_dim = cand_in_dim
        self.head_skip_mode = head_skip_mode

        # === lever (b) anchor embedding 학습 (query + key 양쪽) ===
        self.anchor_embed = nn.Parameter(torch.randn(K, anchor_embed_dim) * anchor_embed_init_scale)

        # === lever (c) anchor → key dim projection ===
        self.anchor_key_proj = nn.Linear(anchor_embed_dim, hidden)

        # === GRU encoder ===
        self.gru = nn.GRU(
            seq_dim, hidden, num_layers=2,
            dropout=gru_dropout, batch_first=True,
        )

        # === query_mlp ===
        self.query_mlp = nn.Sequential(
            nn.Linear(cand_in_dim + anchor_embed_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )

        # === head — input dim 은 head_skip_mode 에 따라 달라짐 ===
        if head_skip_mode == "none":
            head_in_dim = hidden
        elif head_skip_mode == "no_anchor_spec":
            head_in_dim = hidden + (cand_in_dim - (self.ANCHOR_SPEC_END - self.ANCHOR_SPEC_START))  # H + 156
        else:  # full
            head_in_dim = hidden + cand_in_dim                                       # H + 165
        self.head_in_dim = head_in_dim
        self.head = nn.Linear(head_in_dim, 1)

    def forward(self, seq: torch.Tensor, cand_ext: torch.Tensor) -> torch.Tensor:
        B = cand_ext.shape[0]
        K = self.K
        H = self.hidden

        # === lever (b) anchor_embed broadcast → query input ===
        anchor_embed_bc = self.anchor_embed.unsqueeze(0).expand(B, -1, -1)         # (B, K, D_EMBED)
        query_in = torch.cat([cand_ext, anchor_embed_bc], dim=-1)                  # (B, K, cand_in + D_EMBED)

        # === query projection ===
        query = self.query_mlp(query_in)                                           # (B, K, H)

        # === GRU encoder ===
        out, _ = self.gru(seq)                                                     # out (B, T, H)

        # === lever (c) key anchor-conditional broadcast add ===
        anchor_key_bias = self.anchor_key_proj(self.anchor_embed)                  # (K, H)
        key_anchor = out.unsqueeze(1) + anchor_key_bias.unsqueeze(0).unsqueeze(2)  # (B, K, T, H)

        # === Cross-attention (value = key_anchor 단순화) ===
        attn_logits = torch.einsum("bkh,bkth->bkt", query, key_anchor) / sqrt(H)   # (B, K, T)
        attn = torch.softmax(attn_logits, dim=-1)                                  # (B, K, T)
        event_ctx = torch.einsum("bkt,bkth->bkh", attn, key_anchor)                # (B, K, H)

        # === head input by head_skip_mode ===
        if self.head_skip_mode == "none":
            head_in = event_ctx                                                    # (B, K, H)
        elif self.head_skip_mode == "no_anchor_spec":
            cand_no_spec = torch.cat([
                cand_ext[:, :, : self.ANCHOR_SPEC_START],
                cand_ext[:, :, self.ANCHOR_SPEC_END :],
            ], dim=-1)                                                             # (B, K, cand_in - 9)
            head_in = torch.cat([event_ctx, cand_no_spec], dim=-1)                 # (B, K, H + cand-9)
        else:  # full
            head_in = torch.cat([event_ctx, cand_ext], dim=-1)                     # (B, K, H + cand_in)

        score = self.head(head_in).squeeze(-1)                                     # (B, K)
        return score


def _smoke() -> None:
    torch.manual_seed(20260522)
    B, T, K = 4, 7, 14
    seq_dim, cand_in_dim, hidden = 95, 165, 196
    model = GRUNetX1(seq_dim=seq_dim, cand_in_dim=cand_in_dim, hidden=hidden,
                     anchor_embed_dim=8, anchor_embed_init_scale=0.1, gru_dropout=0.10, K=K)
    seq = torch.randn(B, T, seq_dim)
    cand_ext = torch.randn(B, K, cand_in_dim)
    model.train()

    # forward
    score = model(seq, cand_ext)
    assert score.shape == (B, K), f"score shape={score.shape}"
    assert score.dtype == torch.float32
    assert not torch.isnan(score).any() and not torch.isinf(score).any()

    # anchor_embed shape + init scale + requires_grad
    assert model.anchor_embed.shape == (K, 8)
    assert 0.05 <= model.anchor_embed.std().item() <= 0.20, f"init std={model.anchor_embed.std().item()}"
    assert model.anchor_embed.requires_grad

    # gradient check
    loss = score.sum()
    loss.backward()
    assert model.anchor_embed.grad is not None
    grad_norm = model.anchor_embed.grad.norm().item()
    assert grad_norm > 0, f"anchor_embed grad norm = {grad_norm}"

    # head input dim 검증 (Linear(196, 1) — raw skip 차단)
    assert model.head.in_features == hidden, f"head in={model.head.in_features}"
    assert model.head.out_features == 1

    # param 합산
    n_params = sum(p.numel() for p in model.parameters())
    print(f"smoke OK: score={score.shape}, anchor_embed std={model.anchor_embed.std():.4f}, "
          f"grad_norm={grad_norm:.4f}, total_params={n_params}, head_in={model.head.in_features}")


if __name__ == "__main__":
    _smoke()
