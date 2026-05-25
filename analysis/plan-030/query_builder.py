"""plan-030 c2 — query_builder.

attention-Q (per anchor, K=14, 64D) 산출 (plan-030 spec §2.2 그대로):

  query 64D 구성:
    anchor_spec      = cand_feat_150[:, :, 3:12]          # (N, K, 9)   plan-024 cand_builder
    par_perp_dist    = cand_feat_150[:, :, 0:3]           # (N, K, 3)
    interactions     = cand_feat_150[:, :, 140:150]       # (N, K, 10)
    residual_b_flat  = residual_b.reshape(N, K, 35)       # (N, K, 35)  7 step × 5 coord
    slim7            = extension_slim7                    # (N, K, 7)   plan-029 cand_ext col [159, 158, 160:165]
    query = concat([anchor_spec, par_perp_dist, interactions, residual_b_flat, slim7], -1)  # (N, 14, 64)
"""
from __future__ import annotations

import numpy as np


def build_query(
    cand_feat_150: np.ndarray,
    residual_b: np.ndarray,
    extension_slim7: np.ndarray,
) -> np.ndarray:
    """plan-030 §2.2 query builder.

    Args:
        cand_feat_150:    (N, K=14, 150) plan-024 cand_builder 결과
        residual_b:       (N, K=14, 7, 5) from residual_builder
        extension_slim7:  (N, K=14, 7) plan-029 cand_ext_165 col [159, 158, 160, 161, 162, 163, 164]
                                       = D.regime_anchor_prob 1 + B.cos 1 + F.2 5

    Returns:
        query: (N, K=14, 64) float32
    """
    cand = np.asarray(cand_feat_150)
    rb = np.asarray(residual_b)
    sl = np.asarray(extension_slim7)

    N, K, D = cand.shape
    assert D == 150, f"cand_feat_150 last dim must be 150, got {D}"
    assert rb.shape == (N, K, 7, 5), f"residual_b shape mismatch: {rb.shape}"
    assert sl.shape == (N, K, 7), f"extension_slim7 shape mismatch: {sl.shape}"

    anchor_spec = cand[:, :, 3:12]                  # (N, K, 9)
    par_perp_dist = cand[:, :, 0:3]                 # (N, K, 3)
    interactions = cand[:, :, 140:150]              # (N, K, 10)
    residual_b_flat = rb.reshape(N, K, 35)          # (N, K, 35)

    query = np.concatenate(
        [anchor_spec, par_perp_dist, interactions, residual_b_flat, sl],
        axis=-1,
    ).astype(np.float32)
    assert query.shape == (N, K, 64), f"query 64D mismatch: {query.shape}"

    query = np.nan_to_num(query, nan=0.0, posinf=1e3, neginf=-1e3)
    return query


def extract_slim7_from_cand_ext_165(cand_ext_165: np.ndarray) -> np.ndarray:
    """plan-030 §2.2 — plan-029 cand_ext_165 (N, K=14, 165) 에서 slim 7D 추출.

    Cols: [159] D.regime_anchor_prob + [158] B.cos + [160:165] F.2 anchor·v
    = 7D total.
    """
    arr = np.asarray(cand_ext_165)
    assert arr.shape[-1] == 165, f"cand_ext last dim must be 165, got {arr.shape[-1]}"
    cols = [159, 158, 160, 161, 162, 163, 164]
    return arr[:, :, cols].astype(np.float32)


# ── smoke ────────────────────────────────────────────────────────────


if __name__ == "__main__":
    rng = np.random.default_rng(20260524)
    N, K = 8, 14
    cand_feat = rng.standard_normal((N, K, 150)).astype(np.float32)
    residual_b = rng.standard_normal((N, K, 7, 5)).astype(np.float32)
    cand_ext = rng.standard_normal((N, K, 165)).astype(np.float32)
    slim7 = extract_slim7_from_cand_ext_165(cand_ext)

    assert slim7.shape == (N, K, 7)
    q = build_query(cand_feat, residual_b, slim7)
    assert q.shape == (N, K, 64)
    assert not np.isnan(q).any()
    print(f"[smoke] query_builder OK — q.shape={q.shape}, slim7.shape={slim7.shape}")
