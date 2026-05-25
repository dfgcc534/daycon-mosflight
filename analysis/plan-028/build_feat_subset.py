"""plan-028 c3 — build_feat_1080 output 의 index slice + R1 raw seq wrapper.

§4.3 박제 따라 4 slice fn + build_R1_seq_raw + build_feat_1080_with_raw_seq wrapper.

NOTE: spec §4.3 의 slice fn signature 는 conceptual 표기 `X[N, 14, 1080]` 이지만,
plan-025 carry `build_feat_1080` 의 실제 output 은 **2D row-expanded** `(N*14, 1080)`
(plan-025 spec §4.2 + carry source line 175 ).
따라서 본 file 의 slice fn 도 2D row-expanded array 위에서 동작.
decision-note: spec-default — slice 는 row-expand 후 (N*K=14, dim) 위에서 진행.

block slice index (spec §4.3 정합):
- block ① indices [0:170]    plan-022 carry 170D
- block ② indices [170:298]  cand_builder ctx broadcast 128D
- block ③ indices [298:320]  cand_builder per-anchor 22D
- block ④ indices [320:1080] seq_builder 8-stat 760D
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Callable

import numpy as np

# ── importlib carry (plan-025 와 동일 패턴) ──────────────────────────────
_THIS = Path(__file__).resolve().parent              # analysis/plan-028/
_REPO = _THIS.parent.parent                           # repo root
_PLAN024 = _THIS.parent / "plan-024"
_PLAN025 = _THIS.parent / "plan-025"

for p in (_REPO, _THIS.parent / "plan-021", _PLAN024):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


_bf1080_mod = _load(_PLAN025 / "build_feat_1080.py", "p028_bf1080")
_seq_mod = _load(_PLAN024 / "seq_builder.py", "p028_seq")


# ── constants (spec §4.3 lock-in) ────────────────────────────────────────
BLOCK_SLICES: dict[str, slice] = {
    "block1_p022": slice(0, 170),
    "block2_ctx": slice(170, 298),
    "block3_per_anchor": slice(298, 320),
    "block4_seq_stat": slice(320, 1080),
}
TOTAL_DIM = 1080
K_ANCHORS = 14
T_SEQ = 7
N_SEQ_CHANNELS = 95


# ── slice fn 4개 (§4.3 박제) ─────────────────────────────────────────────
def slice_B1_anchor22(X_1080: np.ndarray) -> np.ndarray:
    """Block ③ only (per-anchor 22D). X_1080: (N*K, 1080) → (N*K, 22).

    (d) most aggressive — broadcast 완전 제거.
    """
    assert X_1080.ndim == 2 and X_1080.shape[1] == TOTAL_DIM
    return X_1080[:, 298:320].astype(np.float32)


def slice_B2_combo192(X_1080: np.ndarray) -> np.ndarray:
    """Block ①+③ (plan-022 base 170D + per-anchor 22D). (N*K, 192).

    (d) lift most likely.
    """
    assert X_1080.ndim == 2 and X_1080.shape[1] == TOTAL_DIM
    idx = np.r_[0:170, 298:320]
    return X_1080[:, idx].astype(np.float32)


def slice_B3_no_anchor1058(X_1080: np.ndarray) -> np.ndarray:
    """Block ①+②+④ (no ③). (N*K, 1058).

    (d) per-anchor 없으면 mode collapse 회복?
    """
    assert X_1080.ndim == 2 and X_1080.shape[1] == TOTAL_DIM
    idx = np.r_[0:298, 320:1080]
    return X_1080[:, idx].astype(np.float32)


def slice_B4_full1080(X_1080: np.ndarray) -> np.ndarray:
    """1080D full (baseline reference). (N*K, 1080) passthrough."""
    assert X_1080.ndim == 2 and X_1080.shape[1] == TOTAL_DIM
    return X_1080.astype(np.float32)


# ── R1 cell: block ④ 8-stat → raw seq flatten 665D 대체 ───────────────────
def build_R1_seq_raw(X_1080: np.ndarray, seq_raw: np.ndarray) -> np.ndarray:
    """Block ④ slice [320:1080] 제외 후 raw seq flatten concat → (N*K, 985).

    Args:
        X_1080: (N*K=14, 1080) row-expanded.
        seq_raw: (N, T_SEQ=7, N_SEQ_CHANNELS=95) — plan-024 seq_builder.build output.

    Returns:
        (N*K, 985) — block ①+②+③ (320D) + raw seq flatten broadcast (665D).

    (e) seq 압축 lossy 검증 — raw 95×7=665D vs 8-stat 760D 직접 비교.
    """
    assert X_1080.ndim == 2 and X_1080.shape[1] == TOTAL_DIM
    NK = X_1080.shape[0]
    N = NK // K_ANCHORS
    assert NK == N * K_ANCHORS, f"NK={NK} not divisible by K={K_ANCHORS}"
    assert seq_raw.shape == (N, T_SEQ, N_SEQ_CHANNELS), (
        f"seq_raw shape {seq_raw.shape} ≠ ({N}, {T_SEQ}, {N_SEQ_CHANNELS})"
    )

    # block ①+②+③ = X_1080[:, 0:320]
    block_123 = X_1080[:, 0:320]                                    # (N*K, 320)

    # raw seq flatten (N, 7*95=665) → broadcast 14 row → (N*K, 665)
    seq_flat = seq_raw.reshape(N, T_SEQ * N_SEQ_CHANNELS)           # (N, 665)
    seq_flat_exp = np.repeat(seq_flat, K_ANCHORS, axis=0)           # (N*K, 665)

    out = np.concatenate([block_123, seq_flat_exp], axis=1)         # (N*K, 985)
    assert out.shape == (NK, 985), f"R1 output shape mismatch: {out.shape}"
    return out.astype(np.float32)


# ── wrapper: build_feat_1080 + raw seq 동시 export (§4.3 R1 박제) ─────────
def build_feat_1080_with_raw_seq(
    X: np.ndarray,
    anchors: np.ndarray,
    f0_baseline_fn: Callable,
    quantiles: dict,
    regimes: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """plan-025 build_feat_1080 + plan-024 seq_builder.build 동시 호출.

    plan-025 file 직접 수정 X (= c2 read-only 박제 정합). plan-024 seq_builder 가
    deterministic carry symbol contract — 같은 (X, R_wfn, anchors, f0_baseline_fn,
    quantiles) 위에서 동일 output 보장 (random seed / fold leakage 없음).

    Args:
        X: (N, 11, 3) float32 world frame.
        anchors: (K=14, 3) Frenet coord.
        f0_baseline_fn: plan-020 carry.
        quantiles: plan-024 quantile_carry dict.
        regimes: optional (N,) int64.

    Returns:
        X_1080: (N*14, 1080) row-expanded LGBM input (plan-025 carry).
        seq_raw: (N, 7, 95) plan-024 seq_builder.build output (= R1 cell 용).
    """
    # plan-025 build_feat_1080: row-expanded (N*14, 1080)
    X_1080 = _bf1080_mod.build_feat_1080(X, anchors, f0_baseline_fn, quantiles, regimes)

    # plan-024 seq_builder.build: (N, 7, 95) — frenet basis 재계산 필요
    p021_build = _bf1080_mod._p021_build
    R_wfn = p021_build.build_frenet_basis_3d(X, end_idx=10)
    seq_raw = _seq_mod.build(X, R_wfn, anchors, f0_baseline_fn, quantiles)

    N = X.shape[0]
    assert X_1080.shape == (N * K_ANCHORS, TOTAL_DIM), (
        f"X_1080 shape {X_1080.shape} ≠ ({N * K_ANCHORS}, {TOTAL_DIM})"
    )
    assert seq_raw.shape == (N, T_SEQ, N_SEQ_CHANNELS), (
        f"seq_raw shape {seq_raw.shape} ≠ ({N}, {T_SEQ}, {N_SEQ_CHANNELS})"
    )
    return X_1080.astype(np.float32), seq_raw.astype(np.float32)


# ── weight flag helper (§4.3 박제) ───────────────────────────────────────
def weight_flag(on: bool, soft_label: np.ndarray) -> np.ndarray:
    """sample_weight 산식 (W1 cell 의 (b) 가설 isolation).

    Args:
        on: True (ON, soft_label-weighted, baseline) / False (OFF, 1.0 균등, W1).
        soft_label: (N, K=14) soft label per sample × anchor.

    Returns:
        weight: (N*K,) row-expanded sample_weight.
        - ON: weight[i*K + k] = soft_label[i, k]
        - OFF: weight[i*K + k] = 1.0 균등
    """
    N, K = soft_label.shape
    assert K == K_ANCHORS, f"K={K} ≠ {K_ANCHORS}"
    if on:
        return soft_label.reshape(N * K).astype(np.float32)
    return np.ones(N * K, dtype=np.float32)


# ── smoke ────────────────────────────────────────────────────────────────
def _smoke() -> None:
    """Smoke test — 4 slice fn dim 확인 + R1 dim + weight_flag shape."""
    rng = np.random.default_rng(20260522)
    N = 8
    X_1080 = rng.standard_normal((N * K_ANCHORS, TOTAL_DIM)).astype(np.float32)
    assert slice_B1_anchor22(X_1080).shape == (N * K_ANCHORS, 22)
    assert slice_B2_combo192(X_1080).shape == (N * K_ANCHORS, 192)
    assert slice_B3_no_anchor1058(X_1080).shape == (N * K_ANCHORS, 1058)
    assert slice_B4_full1080(X_1080).shape == (N * K_ANCHORS, TOTAL_DIM)

    seq_raw = rng.standard_normal((N, T_SEQ, N_SEQ_CHANNELS)).astype(np.float32)
    assert build_R1_seq_raw(X_1080, seq_raw).shape == (N * K_ANCHORS, 985)

    soft_label = rng.random((N, K_ANCHORS)).astype(np.float32)
    soft_label = soft_label / soft_label.sum(axis=1, keepdims=True)
    w_on = weight_flag(True, soft_label)
    w_off = weight_flag(False, soft_label)
    assert w_on.shape == (N * K_ANCHORS,) and w_off.shape == (N * K_ANCHORS,)
    assert np.allclose(w_off, 1.0)
    print("plan-028 build_feat_subset smoke OK")


if __name__ == "__main__":
    _smoke()
