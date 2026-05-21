"""plan-025 c5 — smoke tests (≥ 10 pytest, spec §4.5).

G0 gate: 모두 green = STAGE 0 인프라 OK.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parent.parent       # repo root
_PLAN025 = _REPO / "analysis" / "plan-025"
_PLAN024 = _REPO / "analysis" / "plan-024"
_PLAN022 = _REPO / "analysis" / "plan-022"
_PLAN021 = _REPO / "analysis" / "plan-021"
_PLAN020 = _REPO / "analysis" / "plan-020"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def modules():
    """모든 carry module 한 번에 load."""
    if str(_REPO) not in sys.path:
        sys.path.insert(0, str(_REPO))
    return {
        "bf": _load(_PLAN020 / "baseline_f0.py", "t_bf"),
        "p021_build": _load(_PLAN021 / "build_input.py", "t_p021"),
        "p022_anchors": _load(_PLAN022 / "anchors.py", "t_p022_a"),
        "som": _load(_PLAN022 / "selector_only_model.py", "t_som"),
        "cand": _load(_PLAN024 / "cand_builder.py", "t_cand"),
        "seq": _load(_PLAN024 / "seq_builder.py", "t_seq"),
        "av": _load(_PLAN024 / "anchor_vocab.py", "t_av"),
        "qc": _load(_PLAN024 / "quantile_carry.py", "t_qc"),
        "p025_build": _load(_PLAN025 / "build_feat_1080.py", "t_p025_build"),
    }


def test_imports(modules):
    """plan-025 + plan-024 cherry-pick + plan-022/021/020 module 모두 import."""
    for key in ("bf", "p021_build", "p022_anchors", "som", "cand", "seq", "av", "qc", "p025_build"):
        assert key in modules, f"missing module: {key}"


def test_block_dims(modules):
    """BLOCK_DIMS sum = 1080."""
    bd = modules["p025_build"].BLOCK_DIMS
    assert bd["block1_p022"] + bd["block2_ctx"] + bd["block3_per_anchor"] + bd["block4_seq_stat"] == bd["total_per_row"]
    assert bd["total_per_row"] == 1080


def test_stat_names(modules):
    """STAT_NAMES = 8 stat (spec §6.1)."""
    sn = modules["p025_build"].STAT_NAMES
    assert sn == ["last", "first", "mean", "std", "slope", "max", "min", "range"]
    assert len(sn) == 8


def test_compress_seq_8stat_shape(modules):
    """compress_seq_8stat (N=5, 7, 95) → (5, 760)."""
    rng = np.random.default_rng(20260522)
    seq = rng.standard_normal((5, 7, 95)).astype(np.float32)
    out = modules["p025_build"].compress_seq_8stat(seq)
    assert out.shape == (5, 760), f"out shape: {out.shape}"
    assert out.dtype == np.float32


def test_compress_seq_8stat_invariants(modules):
    """last == seq[:, -1, :], first == seq[:, 0, :], range == max − min."""
    rng = np.random.default_rng(20260522)
    seq = rng.standard_normal((3, 7, 95)).astype(np.float32)
    out = modules["p025_build"].compress_seq_8stat(seq)
    # last (idx 0..94)
    assert np.allclose(out[:, 0:95], seq[:, -1, :], atol=1e-6)
    # first (idx 95..189)
    assert np.allclose(out[:, 95:190], seq[:, 0, :], atol=1e-6)
    # range (idx 665..760) == max (idx 475..570) - min (idx 570..665)
    max_v = out[:, 475:570]
    min_v = out[:, 570:665]
    range_v = out[:, 665:760]
    assert np.allclose(range_v, max_v - min_v, atol=1e-6)


def test_anchor_vocab_codebook_eq_A6(modules):
    """plan-024 anchor_vocab module 이 plan-022 ANCHORS_A6 와 호환 (= K=14, ‖a‖=0.005m)."""
    A6 = modules["p022_anchors"].ANCHORS_A6
    assert A6.shape == (14, 3), f"ANCHORS_A6 shape: {A6.shape}"
    norms = np.linalg.norm(A6, axis=1)
    assert np.allclose(norms, 0.005, atol=1e-6), f"norms: {norms}"


def test_quantile_carry_apply(modules):
    """build_train_quantiles roundtrip 정상."""
    rng = np.random.default_rng(20260522)
    X = rng.standard_normal((10, 11, 3)).astype(np.float32)
    R_wfn = modules["p021_build"].build_frenet_basis_3d(X, end_idx=10)
    qc = modules["qc"].build(X, R_wfn)
    assert "omega_p90" in qc
    assert "jerk_p90" in qc
    assert np.isfinite(qc["omega_p90"])
    assert np.isfinite(qc["jerk_p90"])


def test_build_feat_1080_shape(modules):
    """build_feat_1080 (N=5, 11, 3) → (5*14, 1080)."""
    rng = np.random.default_rng(20260522)
    N = 5
    X = rng.standard_normal((N, 11, 3)).astype(np.float32)
    R_wfn = modules["p021_build"].build_frenet_basis_3d(X, end_idx=10)
    qc = modules["qc"].build(X, R_wfn)
    feat = modules["p025_build"].build_feat_1080(
        X, modules["p022_anchors"].ANCHORS_A6, modules["bf"].f0_baseline, qc,
    )
    assert feat.shape == (N * 14, 1080), f"feat shape: {feat.shape}"
    assert feat.dtype == np.float32
    assert not np.isnan(feat).any()
    assert not np.isinf(feat).any()


def test_block2_broadcast_invariant(modules):
    """block ② = cand_feat[:, 0, 12:140] is broadcast across all 14 anchor rows."""
    rng = np.random.default_rng(20260522)
    N = 4
    X = rng.standard_normal((N, 11, 3)).astype(np.float32)
    R_wfn = modules["p021_build"].build_frenet_basis_3d(X, end_idx=10)
    qc = modules["qc"].build(X, R_wfn)
    anchors = modules["p022_anchors"].ANCHORS_A6
    from src.pb_0_6822.selector import fit_regime_bins, assign_regimes
    regime_bins = fit_regime_bins(X, end_idx=10)
    regimes = assign_regimes(X, end_idx=10, bins=regime_bins)
    pred_F0 = modules["bf"].f0_baseline(X, end_idx=10)
    cand_feat = modules["cand"].build(
        X, R_wfn, pred_F0, anchors, modules["bf"].f0_baseline, regimes, qc,
        multiwindow_trim_path=str(_PLAN024 / "multiwindow_trim.json"), regime_count=18,
    )
    assert cand_feat.shape == (N, 14, 150)
    # block ② = ctx broadcast → 14 row 모두 동일
    ctx_row0 = cand_feat[:, 0, 12:140]
    for k in range(1, 14):
        assert np.allclose(cand_feat[:, k, 12:140], ctx_row0, atol=1e-6), f"broadcast fail at anchor k={k}"


def test_lgbm_row_expanded_fit_predict_smoke(modules):
    """LgbmSelectorRowExpanded(K=14) + (N*K, 1080) row-expanded input 위 fit/predict 정상."""
    rng = np.random.default_rng(20260522)
    N = 8
    K = 14
    X_feat = rng.standard_normal((N * K, 1080)).astype(np.float32)
    q = rng.uniform(0, 1, size=(N, K)).astype(np.float32)
    q = q / q.sum(axis=1, keepdims=True)

    # plan-025 wrapper subclass (spec §4.4 선택 B carry)
    p025_run_oof = _load(_PLAN025 / "run_oof.py", "t_p025_run_oof")
    model = p025_run_oof.LgbmSelectorRowExpanded(K=K)
    model.clf.set_params(n_estimators=10, random_state=20260522)
    model.fit(X_feat, q)
    probs = model.clf.predict_proba(X_feat)
    assert probs.shape == (N * K, K), f"probs shape: {probs.shape}"
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)


def test_soft_label_sum_one(modules):
    """build_soft_label_with_tau output row-sum=1."""
    rng = np.random.default_rng(20260522)
    N = 5
    gt = rng.standard_normal((N, 3)).astype(np.float32) * 0.005
    R_wfn = rng.standard_normal((N, 3, 3)).astype(np.float32)
    # orthonormalize via QR
    for i in range(N):
        Q, _ = np.linalg.qr(R_wfn[i])
        R_wfn[i] = Q
    F0 = rng.standard_normal((N, 3)).astype(np.float32) * 0.005
    A6 = modules["p022_anchors"].ANCHORS_A6
    q = modules["som"].build_soft_label_with_tau(gt, R_wfn, F0, A6, tau_cls=0.001)
    assert q.shape == (N, 14)
    assert np.allclose(q.sum(axis=1), 1.0, atol=1e-5)


def test_f0_baseline_carry(modules):
    """plan-020 f0_baseline + plan-022 ANCHORS_A6 import 정상."""
    f = modules["bf"].f0_baseline
    A6 = modules["p022_anchors"].ANCHORS_A6
    assert callable(f)
    assert A6.shape == (14, 3)
    # smoke call
    rng = np.random.default_rng(20260522)
    X = rng.standard_normal((3, 11, 3)).astype(np.float32)
    pred = f(X, end_idx=10)
    assert pred.shape == (3, 3)
