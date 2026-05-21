"""plan-024 c5.8 — Multi-window stat grid 144 → 60 trim list 생성 (§4.4.1).

deterministic correlation-based greedy column drop:
  1. Full train (N=10000) 의 144D Multi-window stat 계산.
  2. 144×144 absolute Pearson correlation matrix.
  3. Greedy column drop: corr > 0.95 인 쌍 중 variance 작은 column 제거.
  4. drop 후 60 column 초과 시 variance 낮은 column 추가 drop.
  5. 결과 = 60 kept_indices → multiwindow_trim.json 박제.

식 (144D = 4 sub-window × 4 stat × 9 channel):
  sub_windows = [(0, 11), (4, 11), (6, 11), (8, 11)]  # 전체 / 뒤 7 / 뒤 5 / 뒤 3
  stats = [mean, std, slope, max]
  channels = Frenet [p_t, p_n, p_b, v_t, v_n, v_b, a_t, a_n, a_b] (9)

fold-leakage 결정: full train 위 단일 trim 채택 (label 안 사용 → leakage scale
미미, LANL Singer 1st pattern carry).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

CORR_THRESHOLD = 0.95
TARGET_KEPT = 60
SUB_WINDOWS = [(0, 11), (4, 11), (6, 11), (8, 11)]   # (start, end) inclusive of start, exclusive of end
N_STATS = 4                                            # mean, std, slope, max
N_CHANNELS = 9                                         # Frenet [p, v, a] each 3-axis


def _compute_144d_stat(L1_frenet: np.ndarray) -> np.ndarray:
    """L1 (N, 11, 9) Frenet [p, v, a] 9-channel × 11-step → 144D per sample.

    Args:
        L1_frenet: (N, 11, 9) float — plan-021 _build_L1 output (Frenet position +
            velocity + accel, 9-channel per step).

    Returns:
        stat_144: (N, 144) float — [sub-window × stat × channel] flatten.
    """
    N = L1_frenet.shape[0]
    out = np.zeros((N, len(SUB_WINDOWS) * N_STATS * N_CHANNELS), dtype=np.float32)

    idx = 0
    for (w_start, w_end) in SUB_WINDOWS:
        sub = L1_frenet[:, w_start:w_end, :]          # (N, w_len, 9)
        w_len = w_end - w_start
        # mean
        out[:, idx:idx + N_CHANNELS] = sub.mean(axis=1).astype(np.float32)
        idx += N_CHANNELS
        # std
        out[:, idx:idx + N_CHANNELS] = sub.std(axis=1).astype(np.float32)
        idx += N_CHANNELS
        # slope (last - first) / w_len
        if w_len > 1:
            slope = (sub[:, -1, :] - sub[:, 0, :]) / (w_len - 1)
        else:
            slope = np.zeros_like(sub[:, 0, :])
        out[:, idx:idx + N_CHANNELS] = slope.astype(np.float32)
        idx += N_CHANNELS
        # max (abs)
        out[:, idx:idx + N_CHANNELS] = np.abs(sub).max(axis=1).astype(np.float32)
        idx += N_CHANNELS

    assert idx == out.shape[1] == 144
    return out


def _greedy_trim_by_corr(stat_144: np.ndarray) -> tuple[list[int], list[int]]:
    """144 column 중 60 kept_indices 선정.

    1. abs Pearson corr matrix (144×144).
    2. corr > CORR_THRESHOLD 인 쌍 (i, j) 중 variance 작은 column 제거.
    3. drop 못 한 column 60 초과 시 variance 낮은 column 추가 drop.

    Returns:
        (kept_indices, drop_indices) — kept_indices sorted ascending, length = 60.
    """
    n_col = stat_144.shape[1]
    # variance per column (epsilon to avoid div-by-zero)
    col_std = stat_144.std(axis=0) + 1e-12          # (144,)

    # normalized
    centered = stat_144 - stat_144.mean(axis=0, keepdims=True)
    normed = centered / col_std[None, :]
    C = np.abs(normed.T @ normed) / stat_144.shape[0]  # absolute Pearson corr (144, 144)
    np.fill_diagonal(C, 0.0)                            # ignore self-corr

    kept = set(range(n_col))
    dropped: list[int] = []

    # phase 1: drop by corr > threshold
    # iterate over upper-triangle pairs sorted by corr desc
    iu = np.triu_indices(n_col, k=1)
    pair_corrs = C[iu]
    order = np.argsort(-pair_corrs)
    for k in order:
        i, j = int(iu[0][k]), int(iu[1][k])
        if pair_corrs[k] <= CORR_THRESHOLD:
            break
        if i not in kept or j not in kept:
            continue
        # drop the lower-variance column of the pair
        drop_col = i if col_std[i] < col_std[j] else j
        kept.discard(drop_col)
        dropped.append(drop_col)

    # phase 2: if remaining > target, drop low-variance columns
    while len(kept) > TARGET_KEPT:
        remaining = sorted(kept, key=lambda c: col_std[c])
        drop_col = remaining[0]                     # lowest variance
        kept.discard(drop_col)
        dropped.append(drop_col)

    kept_list = sorted(kept)
    drop_list = sorted(dropped)
    return kept_list, drop_list


def build_and_save(
    L1_frenet_train: np.ndarray,
    output_path: str | Path = "analysis/plan-024/multiwindow_trim.json",
) -> dict:
    """Multi-window trim build + JSON 박제.

    Args:
        L1_frenet_train: (N, 11, 9) Frenet L1 (plan-021 carry, full train OK
            since variance-only — label-free).
        output_path: JSON output path.

    Returns:
        {"kept_indices": [int × 60], "drop_indices": [int × 84],
         "corr_threshold": 0.95, "n_train": N, "n_total": 144}.
    """
    stat_144 = _compute_144d_stat(L1_frenet_train)
    kept, dropped = _greedy_trim_by_corr(stat_144)

    out = {
        "kept_indices": [int(i) for i in kept],
        "drop_indices": [int(i) for i in dropped],
        "corr_threshold": float(CORR_THRESHOLD),
        "n_train": int(L1_frenet_train.shape[0]),
        "n_total": 144,
        "n_kept": len(kept),
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(out, indent=2))
    return out


# ── smoke (__main__) ───────────────────────────────────────────────────


if __name__ == "__main__":
    rng = np.random.default_rng(20260521)
    N = 1000
    L1 = rng.standard_normal((N, 11, 9)).astype(np.float32) * 0.01
    out = build_and_save(L1, output_path="/tmp/plan024_smoke_trim.json")
    assert out["n_kept"] == 60
    assert len(out["kept_indices"]) == 60
    assert len(out["drop_indices"]) == 84
    assert set(out["kept_indices"]).isdisjoint(set(out["drop_indices"]))
    print(f"[smoke] trim build N={N} kept={out['n_kept']} drop={len(out['drop_indices'])} ✓")
