"""plan-029 c3 — anchor_query_extend.

plan-024 cand_builder.build (N, 14, 150) 호출 후 sample × anchor interaction channel 15개 추가.
출력 shape: (N, 14, 165).

5 group (N_new=15, 사용자 확정):
  A.dist           (5 ch) — past 5 step (t=5..9) anchor world distance norm
  A.tangent_proj   (3 ch) — past 3 step (t=8..10) Frenet t̂-axis projection (signed)
  B.cos            (1 ch) — cos(anchor_dir_w, vel_w), eps-safe
  D.regime_anchor_prob (1 ch) — train-fold P(gt=k | regime[b]) Laplace smoothed lookup
  F.2 multi-step anchor·v (5 ch) — t∈{5..9} ANCHORS_A6 · v_t_frenet

§3.4.1 식 그대로 구현. fold-leakage 차단: regime_anchor_table 은 train-fold 산출본을 inject.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Callable

import numpy as np

_THIS = Path(__file__).resolve().parent              # analysis/plan-029/
_REPO = _THIS.parent.parent                          # repo root
_PLAN024 = _THIS.parent / "plan-024"

for p in (_REPO, _PLAN024):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


_cand_mod = _load(_PLAN024 / "cand_builder.py", "p029_cand")


K_ANCHORS = 14
N_NEW = 15                                            # 5 + 3 + 1 + 1 + 5
CAND_BASE_DIM = 150
CAND_EXT_DIM = CAND_BASE_DIM + N_NEW                  # 165
MULTIWINDOW_TRIM_PATH_DEFAULT = str(_PLAN024 / "multiwindow_trim.json")


def build_regime_anchor_lookup(
    gt_train: np.ndarray,                            # (N_tr, 3) world
    regimes_train: np.ndarray,                       # (N_tr,) int ∈ [0, regime_count)
    ANCHORS_A6: np.ndarray,                          # (K=14, 3) Frenet
    R_wfn_train: np.ndarray,                         # (N_tr, 3, 3)
    F0_train: np.ndarray,                            # (N_tr, 3) world
    regime_count: int = 18,
    laplace: float = 1.0,
) -> np.ndarray:
    """train-fold only lookup table. shape (regime_count, K=14) float32, row-sum=1."""
    R_t = np.transpose(R_wfn_train, (0, 2, 1))
    gt_res_f = np.einsum("nij,nj->ni", R_t, gt_train - F0_train)                # (N_tr, 3) Frenet
    dist = np.linalg.norm(ANCHORS_A6[None, :, :] - gt_res_f[:, None, :], axis=-1)  # (N_tr, K)
    gt_anchor = dist.argmin(axis=1)                                              # (N_tr,)
    K = ANCHORS_A6.shape[0]
    table = np.full((regime_count, K), laplace, dtype=np.float32)                # Laplace init
    np.add.at(table, (regimes_train, gt_anchor), 1.0)
    table /= table.sum(axis=1, keepdims=True)                                    # row-sum=1
    return table


def build(
    X: np.ndarray,                                    # (N, 11, 3) world float
    R_wfn: np.ndarray,                                # (N, 3, 3) world↔Frenet basis (cols = [t̂,n̂,b̂])
    pred_F0_world: np.ndarray,                        # (N, 3) world
    anchors: np.ndarray,                              # (K=14, 3) Frenet (ANCHORS_A6)
    f0_baseline_fn: Callable[[np.ndarray, int], np.ndarray],
    regimes: np.ndarray,                              # (N,) int
    quantile_carry: dict,
    multiwindow_trim_path: str | Path = MULTIWINDOW_TRIM_PATH_DEFAULT,
    regime_count: int = 18,
    regime_anchor_table: np.ndarray | None = None,    # (regime_count, K) row-sum=1, train-fold only
) -> np.ndarray:
    """plan-024 cand_builder 150D + sample×anchor interaction 15ch → (N, K=14, 165) float32.

    fold-leakage 차단: regime_anchor_table 은 호출자 (train.py) 가 train-fold 산출 후 inject.
    None 인 경우 raise (D channel 산출 불가) — dead path 방지.
    """
    if regime_anchor_table is None:
        raise ValueError(
            "regime_anchor_table is None — D channel (lever a, 1/5) 산출 불가. "
            "호출자가 train-fold 산출본 (`build_regime_anchor_lookup(...)`) 을 inject 해야 함."
        )

    N = X.shape[0]
    K = anchors.shape[0]
    assert K == K_ANCHORS, f"K={K} != {K_ANCHORS}"
    X = X.astype(np.float32)

    # ── anchor_world (N, K, 3) — pred_F0_world + R_wfn @ anchors_frenet ──
    anchor_world = pred_F0_world[:, None, :] + np.einsum("nij,kj->nki", R_wfn, anchors)

    # ── A.dist (5 ch) — past 5 step (t=5..9) 의 ||anchor_world - X[:,t,:]|| ──
    diff = anchor_world[:, :, None, :] - X[:, None, 5:10, :]                   # (N, K, 5, 3)
    A_dist = np.linalg.norm(diff, axis=-1).astype(np.float32)                  # (N, K, 5)

    # ── A.tangent_proj (3 ch) — past 3 step (t=8..10) 의 Frenet t̂-axis projection ──
    # sign: past_disp_w = (관측점 - anchor). t̂ 성분 > 0 = anchor 진행 방향 앞쪽
    past_disp_w = X[:, None, 8:11, :] - anchor_world[:, :, None, :]            # (N, K, 3, 3)
    R_t = np.transpose(R_wfn, (0, 2, 1))                                       # world → Frenet
    past_disp_f = np.einsum("nij,nkpj->nkpi", R_t, past_disp_w)                # (N, K, 3, 3) Frenet
    A_tangent = past_disp_f[..., 0].astype(np.float32)                         # (N, K, 3)

    # ── B.cos (1 ch) — cos(anchor_dir_w, vel_w), eps-safe ──
    anchor_dir_w = np.einsum("nij,kj->nki", R_wfn, anchors)                    # (N, K, 3) = anchor_world - F0
    vel_w = (X[:, 10, :] - X[:, 9, :])[:, None, :]                             # (N, 1, 3)
    num = (anchor_dir_w * vel_w).sum(axis=-1)                                  # (N, K)
    den = np.linalg.norm(anchor_dir_w, axis=-1) * np.linalg.norm(vel_w, axis=-1) + 1e-9
    B_cos = (num / den).astype(np.float32)[:, :, None]                         # (N, K, 1)

    # ── D.regime_anchor_prob (1 ch) — train-fold lookup ──
    D = regime_anchor_table[regimes][:, :, None].astype(np.float32)            # (N, K, 1)

    # ── F.2 multi-step anchor·v (5 ch) — t∈{5..9} ANCHORS · v_t_frenet ──
    v_w_seq = X[:, 6:11, :] - X[:, 5:10, :]                                    # (N, 5, 3) world
    v_f_seq = np.einsum("nij,ntj->nti", R_t, v_w_seq)                          # (N, 5, 3) Frenet
    F2 = np.einsum("kj,ntj->nkt", anchors, v_f_seq).astype(np.float32)         # (N, K, 5)

    # ── concat 15 ch on top of plan-024 cand_builder 150D ──
    cand_base = _cand_mod.build(
        X, R_wfn, pred_F0_world, anchors, f0_baseline_fn,
        regimes, quantile_carry,
        multiwindow_trim_path=str(multiwindow_trim_path),
        regime_count=regime_count,
    )                                                                          # (N, K, 150)
    extra = np.concatenate([A_dist, A_tangent, B_cos, D, F2], axis=-1)         # (N, K, 15)
    cand_ext = np.concatenate([cand_base, extra], axis=-1)                     # (N, K, 165)
    cand_ext = np.nan_to_num(cand_ext, nan=0.0, posinf=1e3, neginf=-1e3)
    return cand_ext.astype(np.float32)


def _smoke() -> None:
    rng = np.random.default_rng(20260522)
    N = 8
    X = rng.standard_normal((N, 11, 3)).astype(np.float32)

    # baseline imports
    _plan020 = _THIS.parent / "plan-020"
    _plan021 = _THIS.parent / "plan-021"
    _plan022 = _THIS.parent / "plan-022"
    for p in (_plan020, _plan021, _plan022):
        sys.path.insert(0, str(p))
    _bf = _load(_plan020 / "baseline_f0.py", "p029_smoke_bf")
    _bi = _load(_plan021 / "build_input.py", "p029_smoke_bi")
    _av = _load(_plan022 / "anchors.py", "p029_smoke_av")
    _qc = _load(_PLAN024 / "quantile_carry.py", "p029_smoke_qc")

    R_wfn = _bi.build_frenet_basis_3d(X, end_idx=10)
    F0 = _bf.f0_baseline(X, end_idx=10).astype(np.float32)
    qc = _qc.build(X, R_wfn)
    regimes = rng.integers(0, 18, size=N)
    gt = X[:, 10, :] + rng.standard_normal((N, 3)).astype(np.float32) * 0.005

    table = build_regime_anchor_lookup(gt, regimes, _av.ANCHORS_A6, R_wfn, F0)
    assert table.shape == (18, 14)
    assert np.allclose(table.sum(axis=1), 1.0, atol=1e-5)

    cand_ext = build(X, R_wfn, F0, _av.ANCHORS_A6, _bf.f0_baseline,
                     regimes, qc, regime_anchor_table=table)
    assert cand_ext.shape == (N, 14, 165), f"shape={cand_ext.shape}"
    assert cand_ext.dtype == np.float32
    assert not np.isnan(cand_ext).any() and not np.isinf(cand_ext).any()
    print(f"smoke OK: cand_ext.shape={cand_ext.shape}, dtype={cand_ext.dtype}, "
          f"table.shape={table.shape}, table row-sum={table.sum(axis=1)[:3]}")


if __name__ == "__main__":
    _smoke()
