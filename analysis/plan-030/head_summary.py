"""plan-030 c3 — head_summary builder.

head MLP sample summary (sample-only, anchor 차원 없음, head 에서 broadcast) 산출.
plan-030 spec §2.3 그대로.

51D 구성:
  Bz_Tz       = cand_feat_150[:, 0, 32:34]         # (N, 2)
  macro_8     = cand_feat_150[:, 0, 24:32]         # (N, 8)   plan-024 cand_builder 자체 macro_stat (plan-021 macro_9 와 별개)
  A1          = cand_feat_150[:, 0, 52:55]         # (N, 3)   STA/LTA
  A6          = cand_feat_150[:, 0, 120:123]       # (N, 3)   wingbeat-jitter std
  A10_pct     = cand_feat_150[:, 0, 125:134]       # (N, 9)   Pct-rolling only (Peak 3D drop)
  A12         = cand_feat_150[:, 0, 137:140]       # (N, 3)   v_autocorr lag {1,2,3}
  plan021_macro9                                    # (N, 9)   path/straight/slope/cv/turn/accel/turn_vol/linear_resid/jerk_vol
  soft_hit_L4                                       # (N, 14)  7 step × 2 (R_HIT, R_HIT_LOOSE) flatten

ordering: Bz/Tz + macro8 + A1 + A6 + A10_pct + A12 + plan021_macro9 + soft_hit_L4 = 51D

cand_feat_150 의 ctx [12:140] 는 모든 anchor 행이 같은 broadcast 값 (plan-024 cand_builder.py:409) →
anchor 차원 0 만 추출.
"""
from __future__ import annotations

import numpy as np


def build_head_summary(
    cand_feat_150: np.ndarray,
    plan021_macro9: np.ndarray,
    soft_hit_L4: np.ndarray,
) -> np.ndarray:
    """plan-030 §2.3 head MLP sample summary 51D 산출.

    Args:
        cand_feat_150:   (N, K=14, 150) plan-024 cand_builder 결과
        plan021_macro9:  (N, 9) plan-021 _macro_stat_9d
        soft_hit_L4:     (N, 14) plan-021 _build_L2_L4 의 L4 (N, 7, 2) flatten

    Returns:
        head_summary: (N, 51) float32
    """
    cand = np.asarray(cand_feat_150)
    m9 = np.asarray(plan021_macro9)
    L4 = np.asarray(soft_hit_L4)

    N = cand.shape[0]
    assert cand.shape[1:] == (14, 150), f"cand_feat_150 shape mismatch: {cand.shape}"
    assert m9.shape == (N, 9), f"plan021_macro9 shape mismatch: {m9.shape}"
    assert L4.shape == (N, 14), f"soft_hit_L4 shape mismatch: {L4.shape}"

    ctx0 = cand[:, 0, :]                              # (N, 150) anchor 0 row (broadcast 값)

    Bz_Tz = ctx0[:, 32:34]                            # 2
    macro_8 = ctx0[:, 24:32]                          # 8
    A1 = ctx0[:, 52:55]                               # 3
    A6 = ctx0[:, 120:123]                             # 3
    A10_pct = ctx0[:, 125:134]                        # 9
    A12 = ctx0[:, 137:140]                            # 3

    head_summary = np.concatenate(
        [Bz_Tz, macro_8, A1, A6, A10_pct, A12, m9, L4],
        axis=-1,
    ).astype(np.float32)
    assert head_summary.shape == (N, 51), f"head_summary dim mismatch: {head_summary.shape}"

    head_summary = np.nan_to_num(head_summary, nan=0.0, posinf=1e3, neginf=-1e3)
    return head_summary


# ── smoke ────────────────────────────────────────────────────────────


if __name__ == "__main__":
    rng = np.random.default_rng(20260524)
    N = 8
    cand_feat = rng.standard_normal((N, 14, 150)).astype(np.float32)
    m9 = rng.standard_normal((N, 9)).astype(np.float32)
    L4 = rng.uniform(0.0, 1.0, (N, 14)).astype(np.float32)
    hs = build_head_summary(cand_feat, m9, L4)
    assert hs.shape == (N, 51)
    assert not np.isnan(hs).any()
    print(f"[smoke] head_summary OK — hs.shape={hs.shape}")
