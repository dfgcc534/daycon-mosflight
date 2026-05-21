"""plan-024 c6 — CrossAttentionAnchorSelector (§4.6, v1.1-rev2).

PB framework `CandidateAttentionGRUSelector` (src/pb_0_6822/selector.py:697-727)
1:1 carry + FeatureWeightedDropout input adaptor + softmax output head.

Hyperparam:
  GRU: hidden=384, num_layers=2, dropout=0.10 (v1.1: 0.08→0.10)
  Head MLP dropout=0.15 (PB carry 0.10 → v1.1 0.15)
  per-channel learnable scale + channel dropout (FeatureWeightedDropout)
  K=14, seq_dim=95, cand_dim=150
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

_THIS = Path(__file__).resolve().parent
_REPO = _THIS.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

_spec = importlib.util.spec_from_file_location(
    "p024_fwd", _THIS / "feature_weighted_dropout.py"
)
fwd_mod = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(fwd_mod)


class CandidateAttentionGRUSelectorCarry(nn.Module):
    """PB framework `CandidateAttentionGRUSelector` 의 self-contained carry
    (src/pb_0_6822/selector.py:697-727 의 식 정확 reproduce, plan-024 hidden=384).

    forward:
      seq: (b, T, seq_dim) → GRU → out (b, T, hidden) + h (num_layers, b, hidden).
      cand_feat: (b, K, cand_dim) → MLP query → (b, K, hidden).
      attn_logits = einsum(out, query) / sqrt(hidden) → (b, K, T).
      attn = softmax(attn_logits, dim=-1).
      event_ctx = einsum(attn, out) → (b, K, hidden).
      head_in = concat(h_final_broadcast, event_ctx, cand_feat) → (b, K, 2*hidden + cand_dim).
      score = MLP_head(head_in).squeeze(-1) → (b, K) logits.
    """

    def __init__(
        self,
        seq_dim: int = 95,
        cand_dim: int = 150,
        hidden: int = 384,
        cand_count: int = 14,
        gru_dropout: float = 0.10,
        head_dropout: float = 0.15,
    ):
        super().__init__()
        self.hidden = hidden
        self.cand_count = cand_count
        self.gru = nn.GRU(
            seq_dim, hidden, num_layers=2,
            dropout=gru_dropout, batch_first=True,
        )
        self.query_mlp = nn.Sequential(
            nn.Linear(cand_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        self.head = nn.Sequential(
            nn.Linear(2 * hidden + cand_dim, hidden),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, seq: torch.Tensor, cand_feat: torch.Tensor) -> torch.Tensor:
        # seq: (b, T, seq_dim). cand_feat: (b, K, cand_dim).
        out, h = self.gru(seq)                          # out: (b, T, hidden), h: (2, b, hidden)
        h_final = h[-1]                                  # (b, hidden) — last layer
        query = self.query_mlp(cand_feat)               # (b, K, hidden)
        attn_logits = torch.einsum("bth,bkh->bkt", out, query) / (self.hidden ** 0.5)  # (b, K, T)
        attn = torch.softmax(attn_logits, dim=-1)        # (b, K, T)
        event_ctx = torch.einsum("bkt,bth->bkh", attn, out)   # (b, K, hidden)
        # head input
        h_final_bc = h_final.unsqueeze(1).expand(-1, self.cand_count, -1)  # (b, K, hidden)
        head_in = torch.cat([h_final_bc, event_ctx, cand_feat], dim=-1)    # (b, K, 2H+cand_dim)
        score = self.head(head_in).squeeze(-1)           # (b, K)
        return score


class CrossAttentionAnchorSelector(nn.Module):
    """v1.1-rev2 plan-024 model: input adaptor + PB backbone + softmax.

    v5 patch (G2 diagnose finding): A7 Learnable anchor embedding 추가
    (14 × embed_dim, default 8). cross-attention 의 anchor-specific
    learnable capacity 보강 — fwd 후 cand_feat 에 broadcast concat.
    """

    def __init__(
        self,
        seq_dim: int = 95,
        cand_dim: int = 150,
        hidden: int = 384,
        cand_count: int = 14,
        cand_drop_p: float = 0.3,
        seq_drop_p: float = 0.2,
        anchor_embed_dim: int = 0,    # v5 patch: 0 = disable (default carry)
                                       # 8 = A7 learnable anchor embedding 활성
    ):
        super().__init__()
        self.fwd = fwd_mod.FeatureWeightedDropout(
            cand_dim=cand_dim,
            seq_dim=seq_dim,
            cand_drop_p=cand_drop_p,
            seq_drop_p=seq_drop_p,
        )
        self.anchor_embed_dim = anchor_embed_dim
        if anchor_embed_dim > 0:
            self.anchor_embed = nn.Parameter(
                torch.randn(cand_count, anchor_embed_dim) * 0.02
            )
            backbone_cand_dim = cand_dim + anchor_embed_dim
        else:
            self.register_parameter("anchor_embed", None)
            backbone_cand_dim = cand_dim
        self.backbone = CandidateAttentionGRUSelectorCarry(
            seq_dim=seq_dim,
            cand_dim=backbone_cand_dim,
            hidden=hidden,
            cand_count=cand_count,
        )

    def forward(self, seq: torch.Tensor, cand_feat: torch.Tensor):
        cand_feat, seq = self.fwd(cand_feat, seq, training=self.training)
        if self.anchor_embed is not None:
            b = cand_feat.shape[0]
            embed_bc = self.anchor_embed.unsqueeze(0).expand(b, -1, -1)
            cand_feat = torch.cat([cand_feat, embed_bc], dim=-1)
        score = self.backbone(seq, cand_feat)
        q_pred = F.softmax(score, dim=-1)
        return q_pred, score


# ── smoke (__main__) ───────────────────────────────────────────────────


if __name__ == "__main__":
    torch.manual_seed(20260521)
    model = CrossAttentionAnchorSelector()
    seq = torch.randn(4, 7, 95)
    cand = torch.randn(4, 14, 150)
    q_pred, score = model(seq, cand)
    assert q_pred.shape == (4, 14)
    assert score.shape == (4, 14)
    assert torch.allclose(q_pred.sum(-1), torch.ones(4), atol=1e-5)
    assert torch.isfinite(q_pred).all()
    assert torch.isfinite(score).all()
    print(f"[smoke] CrossAttentionAnchorSelector forward ✓")
    print(f"        q_pred shape={tuple(q_pred.shape)} sum=1 ✓, score shape={tuple(score.shape)}")
    # parameter count
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"        total trainable params: {n_params:,}")

    # 2-epoch fit smoke (test #10)
    optim = torch.optim.AdamW(model.parameters(), lr=7e-4, weight_decay=0.02)
    model.train()
    losses = []
    q_target = torch.softmax(torch.randn(4, 14), dim=-1)
    for epoch in range(2):
        epoch_losses = []
        for step in range(4):
            optim.zero_grad()
            q_pred, _ = model(seq, cand)
            loss = -(q_target * torch.log(q_pred + 1e-12)).sum(-1).mean()
            loss.backward()
            optim.step()
            epoch_losses.append(loss.item())
        losses.append(sum(epoch_losses) / len(epoch_losses))
        print(f"        epoch {epoch}: avg loss = {losses[-1]:.4f}")
    assert all(map(lambda x: x == x and abs(x) < 1e6, losses)), "NaN/Inf loss"
    print(f"[smoke] 2-epoch fit finite ✓")
