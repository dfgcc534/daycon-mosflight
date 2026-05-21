"""plan-024 c5.7 — per-channel learnable scale + channel dropout (§4.6, v1.1-rev2).

LGBM `feature_fraction` NN 등가 (Singer LANL 1st pattern carry).

- Learnable scale: `cand_scale ∈ R^150, seq_scale ∈ R^95, init=1.0, clamp(0.1, 10)`.
- Channel dropout:
  - cand ③ ctx broadcast 영역 [12, 140) 만 (p=0.3) — ①3 + ②9 = 12 까지 보호,
    ④10 = [140, 150) 보호.
  - seq redundant slice 만 (p=0.2): J EWMA [62..70] + A5 WAP per-step [81..85] = 14 channel.

식 (inverted dropout scaling):
  mask_cand[drop_region] = Bernoulli(1 - p)
  cand_out = cand_in * scale_cand * mask_cand[None, None, :] / (1 - p)
  (실제 scale = mask.numel / max(mask.sum, eps), inverted dropout standard form)
"""
from __future__ import annotations

import torch
import torch.nn as nn

CAND_DIM = 150
SEQ_DIM = 95
CAND_DROP_START = 12   # ①3 + ②9 = 12, drop 시작
CAND_DROP_END = 140    # ③ 끝 (12 + 128)
SEQ_DROP_INDICES = (
    list(range(62, 71))         # J EWMA 9D
    + list(range(81, 86))       # A5 WAP per-step 5D
)


class FeatureWeightedDropout(nn.Module):
    """per-channel learnable scale + channel-wise dropout."""

    def __init__(
        self,
        cand_dim: int = CAND_DIM,
        seq_dim: int = SEQ_DIM,
        cand_drop_p: float = 0.3,
        seq_drop_p: float = 0.2,
        cand_drop_start: int = CAND_DROP_START,
        cand_drop_end: int = CAND_DROP_END,
        seq_drop_indices: list[int] | None = None,
    ):
        super().__init__()
        self.cand_scale = nn.Parameter(torch.ones(cand_dim))
        self.seq_scale = nn.Parameter(torch.ones(seq_dim))
        self.cand_drop_p = float(cand_drop_p)
        self.seq_drop_p = float(seq_drop_p)
        self.cand_drop_start = int(cand_drop_start)
        self.cand_drop_end = int(cand_drop_end)
        seq_idx = seq_drop_indices if seq_drop_indices is not None else SEQ_DROP_INDICES
        self.register_buffer(
            "seq_drop_indices",
            torch.tensor(list(seq_idx), dtype=torch.long),
        )

    def _scale_clamped(self, scale: torch.Tensor) -> torch.Tensor:
        return torch.clamp(scale, 0.1, 10.0)

    def forward(self, cand: torch.Tensor, seq: torch.Tensor, training: bool | None = None):
        """cand: (b, K, cand_dim). seq: (b, T, seq_dim). returns same shape after weight+drop.

        training: if None, uses self.training.
        """
        if training is None:
            training = self.training
        # weighting (always)
        cand = cand * self._scale_clamped(self.cand_scale)[None, None, :]
        seq = seq * self._scale_clamped(self.seq_scale)[None, None, :]

        if training and self.cand_drop_p > 0:
            # cand: drop 영역만 (③ ctx broadcast)
            drop_n = self.cand_drop_end - self.cand_drop_start
            mask_full = torch.ones(cand.shape[-1], device=cand.device)
            keep = (torch.rand(drop_n, device=cand.device) > self.cand_drop_p).float()
            mask_full[self.cand_drop_start:self.cand_drop_end] = keep
            # inverted dropout: divide by keep ratio of drop region
            keep_ratio = keep.mean().clamp(min=1e-6)
            scale_factor = 1.0 / keep_ratio.where(keep_ratio > 0, torch.tensor(1.0, device=cand.device))
            # apply only to drop region
            drop_region_mask = torch.zeros_like(mask_full)
            drop_region_mask[self.cand_drop_start:self.cand_drop_end] = 1.0
            # element-wise: keep regions get 1.0, drop region get mask * scale_factor
            effective = mask_full + drop_region_mask * (mask_full * scale_factor - mask_full)
            # simpler: combine
            mask_eff = torch.where(
                drop_region_mask.bool(),
                mask_full * scale_factor,
                mask_full,
            )
            cand = cand * mask_eff[None, None, :]

        if training and self.seq_drop_p > 0:
            seq_mask = torch.ones(seq.shape[-1], device=seq.device)
            keep = (
                torch.rand(self.seq_drop_indices.numel(), device=seq.device)
                > self.seq_drop_p
            ).float()
            seq_mask[self.seq_drop_indices] = keep
            keep_ratio = keep.mean().clamp(min=1e-6)
            scale_factor = 1.0 / keep_ratio.where(keep_ratio > 0, torch.tensor(1.0, device=seq.device))
            drop_region_mask = torch.zeros_like(seq_mask)
            drop_region_mask[self.seq_drop_indices] = 1.0
            mask_eff = torch.where(
                drop_region_mask.bool(),
                seq_mask * scale_factor,
                seq_mask,
            )
            seq = seq * mask_eff[None, None, :]

        return cand, seq


# ── smoke (__main__) ───────────────────────────────────────────────────


if __name__ == "__main__":
    torch.manual_seed(20260521)
    fwd = FeatureWeightedDropout()
    cand = torch.randn(4, 14, 150)
    seq = torch.randn(4, 7, 95)

    # eval mode: weighting only, no mask
    fwd.eval()
    cand_e, seq_e = fwd(cand, seq)
    assert cand_e.shape == cand.shape
    assert seq_e.shape == seq.shape

    # training mode: mask applied
    fwd.train()
    cand_t, seq_t = fwd(cand, seq)
    # protected region (①+② = [0:12], ④ = [140:150]) should be untouched by mask
    # but they are still scaled by cand_scale (init=1.0 → identity in expectation)
    # check that some elements in [12:140] differ from cand_e (mask applied)
    diff_drop_region = (cand_t[0, 0, 12:140] - cand_e[0, 0, 12:140]).abs().sum()
    # diff_protected = (cand_t[0, 0, 0:12] - cand_e[0, 0, 0:12]).abs().sum()
    # protected region: cand_t == cand_e (same scale, no mask)
    diff_protected_low = (cand_t[0, 0, 0:12] - cand_e[0, 0, 0:12]).abs().sum()
    diff_protected_high = (cand_t[0, 0, 140:150] - cand_e[0, 0, 140:150]).abs().sum()
    assert diff_protected_low.item() < 1e-5, f"protected [0:12] modified: {diff_protected_low}"
    assert diff_protected_high.item() < 1e-5, f"protected [140:150] modified: {diff_protected_high}"
    print(f"[smoke] FeatureWeightedDropout forward ✓")
    print(f"        protected [0:12] untouched ({diff_protected_low.item():.2e})")
    print(f"        protected [140:150] untouched ({diff_protected_high.item():.2e})")
    print(f"        drop region [12:140] modified ({diff_drop_region.item():.4f})")
    # scale clamp test
    fwd.cand_scale.data.fill_(20.0)                 # > 10 max
    cand_c, _ = fwd(cand, seq, training=False)
    # expect clamp(20, 0.1, 10) = 10
    expected = cand * 10.0
    assert torch.allclose(cand_c, expected, atol=1e-5), "clamp upper bound 10 not enforced"
    print(f"[smoke] scale clamp upper bound ✓")
