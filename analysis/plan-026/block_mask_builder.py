"""plan-026 c2 — block mask helper (spec §3.1 + §4.2).

BLOCK_RANGES carry from plan-025 BLOCK_DIMS. build_feat_masked wrappers
plan-025 build_feat_1080 + column slice.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Callable

import numpy as np

# ── importlib carry ────────────────────────────────────────────────
_THIS = Path(__file__).resolve().parent              # analysis/plan-026/
_REPO = _THIS.parent.parent                           # repo root
_PLAN025 = _THIS.parent / "plan-025"

for p in (_REPO, _PLAN025):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


_p025_build = _load(_PLAN025 / "build_feat_1080.py", "p026_p025_build")
build_feat_1080 = _p025_build.build_feat_1080


# ── BLOCK_RANGES (spec §3.1) ────────────────────────────────────────
BLOCK_RANGES: dict[str, tuple[int, int]] = {
    "block1": (0, 170),       # plan-022 carry
    "block2": (170, 298),     # cand_builder ctx broadcast
    "block3": (298, 320),     # cand_builder per-anchor
    "block4": (320, 1080),    # seq_builder 8-stat
}

EXCLUSION_MAP: dict[str, str] = {
    "A1": "block2",
    "A2": "block3",
    "A3": "block4",
}

EXPECTED_D_MASKED: dict[str, int] = {
    "A1": 1080 - 128,   # 952
    "A2": 1080 - 22,    # 1058
    "A3": 1080 - 760,   # 320
}


def build_feat_masked(
    X: np.ndarray,
    anchors: np.ndarray,
    f0_baseline_fn: Callable,
    quantiles: dict,
    excluded_cell: str,
    regimes: np.ndarray | None = None,
) -> np.ndarray:
    """returns (N*K=14, D_masked) float32 — plan-025 build_feat_1080 의 column slice.

    Args:
        excluded_cell: "A1" / "A2" / "A3". 각 cell 에 대응하는 block 제거.
    """
    assert excluded_cell in EXCLUSION_MAP, f"unsupported cell: {excluded_cell}"
    feat_full = build_feat_1080(X, anchors, f0_baseline_fn, quantiles, regimes=regimes)
    excl_block = EXCLUSION_MAP[excluded_cell]
    start, end = BLOCK_RANGES[excl_block]
    keep_mask = np.ones(1080, dtype=bool)
    keep_mask[start:end] = False
    masked = feat_full[:, keep_mask].astype(np.float32)
    expected = EXPECTED_D_MASKED[excluded_cell]
    assert masked.shape[1] == expected, f"D_masked={masked.shape[1]} != expected {expected}"
    return masked


def _smoke() -> None:
    rng = np.random.default_rng(20260522)
    N = 5
    X = rng.standard_normal((N, 11, 3)).astype(np.float32)
    _p021_build = _load(_THIS.parent / "plan-021" / "build_input.py", "p026_p021_b")
    _qc_mod = _load(_THIS.parent / "plan-024" / "quantile_carry.py", "p026_qc")
    _p022_anchors = _load(_THIS.parent / "plan-022" / "anchors.py", "p026_p022_a")
    _bf = _load(_THIS.parent / "plan-020" / "baseline_f0.py", "p026_bf")
    R_wfn = _p021_build.build_frenet_basis_3d(X, end_idx=10)
    qc = _qc_mod.build(X, R_wfn)
    for cell, expected in EXPECTED_D_MASKED.items():
        out = build_feat_masked(X, _p022_anchors.ANCHORS_A6, _bf.f0_baseline, qc, cell)
        assert out.shape == (N * 14, expected), f"{cell}: {out.shape} != ({N*14}, {expected})"
        print(f"{cell}: shape={out.shape} ✓")


if __name__ == "__main__":
    _smoke()
