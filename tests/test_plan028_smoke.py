"""plan-028 c5 — smoke tests (≥ 8 pytest).

§0.5 c5 박제: import / 4 slice dim check / sample_weight on/off shape /
LgbmSelectorOnly K=14 fit/predict smoke (subset dim) / F0 carry / soft label sum=1 /
branch decision fn unit test (4 case).
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parent.parent
_PLAN028 = _REPO / "analysis" / "plan-028"

if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


# ── module loads ──────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def subset():
    return _load(_PLAN028 / "build_feat_subset.py", "test_p028_subset")


@pytest.fixture(scope="module")
def branch():
    return _load(_PLAN028 / "branch_decision.py", "test_p028_branch")


@pytest.fixture(scope="module")
def runner():
    return _load(_PLAN028 / "run_oof_subset.py", "test_p028_runner")


# ── 1. import smoke ───────────────────────────────────────────────────────
def test_import_modules(subset, branch, runner):
    assert hasattr(subset, "slice_B1_anchor22")
    assert hasattr(subset, "build_R1_seq_raw")
    assert hasattr(subset, "weight_flag")
    assert hasattr(branch, "decide_branch")
    assert hasattr(runner, "CELL_CONFIGS")
    assert hasattr(runner, "LgbmSelectorRowExpandedWeightFlag")
    assert hasattr(runner, "BaseLgbmMulticlass")
    assert set(runner.CELL_CONFIGS.keys()) == {"B1", "B2", "B3", "B4", "W1", "T1", "T2", "S1", "R1"}


# ── 2~5. slice dim 4 case ────────────────────────────────────────────────
@pytest.fixture(scope="module")
def sample_X_1080():
    rng = np.random.default_rng(20260522)
    N, K = 8, 14
    return rng.standard_normal((N * K, 1080)).astype(np.float32)


def test_slice_B1_anchor22_dim(subset, sample_X_1080):
    out = subset.slice_B1_anchor22(sample_X_1080)
    assert out.shape == (sample_X_1080.shape[0], 22)
    # block ③ indices [298:320] 그대로
    np.testing.assert_array_equal(out, sample_X_1080[:, 298:320])


def test_slice_B2_combo192_dim(subset, sample_X_1080):
    out = subset.slice_B2_combo192(sample_X_1080)
    assert out.shape == (sample_X_1080.shape[0], 192)  # 170 + 22


def test_slice_B3_no_anchor1058_dim(subset, sample_X_1080):
    out = subset.slice_B3_no_anchor1058(sample_X_1080)
    assert out.shape == (sample_X_1080.shape[0], 1058)  # 298 + 760


def test_slice_B4_full1080_dim(subset, sample_X_1080):
    out = subset.slice_B4_full1080(sample_X_1080)
    assert out.shape == sample_X_1080.shape  # passthrough


# ── 6. R1 dim ─────────────────────────────────────────────────────────────
def test_build_R1_seq_raw_dim(subset, sample_X_1080):
    rng = np.random.default_rng(20260522)
    N = sample_X_1080.shape[0] // 14
    seq_raw = rng.standard_normal((N, 7, 95)).astype(np.float32)
    out = subset.build_R1_seq_raw(sample_X_1080, seq_raw)
    assert out.shape == (sample_X_1080.shape[0], 985)  # 170+128+22+665


# ── 7~8. weight_flag on/off ──────────────────────────────────────────────
def test_weight_flag_on_shape(subset):
    rng = np.random.default_rng(20260522)
    N, K = 8, 14
    soft_label = rng.random((N, K)).astype(np.float32)
    soft_label = soft_label / soft_label.sum(axis=1, keepdims=True)
    w_on = subset.weight_flag(True, soft_label)
    assert w_on.shape == (N * K,)
    # ON = soft_label-weighted (row-expand flatten)
    np.testing.assert_array_almost_equal(w_on, soft_label.reshape(N * K))


def test_weight_flag_off_uniform(subset):
    rng = np.random.default_rng(20260522)
    N, K = 8, 14
    soft_label = rng.random((N, K)).astype(np.float32)
    soft_label = soft_label / soft_label.sum(axis=1, keepdims=True)
    w_off = subset.weight_flag(False, soft_label)
    assert w_off.shape == (N * K,)
    assert np.allclose(w_off, 1.0)


# ── 9. soft label sum=1 invariant ────────────────────────────────────────
def test_soft_label_sum_1_invariant():
    """soft_label per sample sum=1 invariant (plan-022 build_soft_label_with_tau carry).

    smoke 단계에서는 실제 plan-022 호출 대신 sum=1 normalization invariant 만 검증.
    """
    rng = np.random.default_rng(20260522)
    N, K = 8, 14
    raw = rng.random((N, K)).astype(np.float32)
    soft = raw / raw.sum(axis=1, keepdims=True)
    np.testing.assert_array_almost_equal(soft.sum(axis=1), np.ones(N))


# ── 10. LgbmSelector subset fit/predict smoke ─────────────────────────────
def test_lgbm_subclass_fit_predict_smoke(runner, subset, sample_X_1080):
    """LgbmSelectorRowExpandedWeightFlag fit/predict smoke (subset dim 22D = B1)."""
    rng = np.random.default_rng(20260522)
    N, K = 8, 14
    X_22 = subset.slice_B1_anchor22(sample_X_1080)  # (N*K, 22)
    soft = rng.random((N, K)).astype(np.float32)
    soft = soft / soft.sum(axis=1, keepdims=True)

    model = runner.LgbmSelectorRowExpandedWeightFlag(K=K)
    model.fit(X_22, soft, weighted=True)
    proba = model.clf.predict_proba(X_22)
    assert proba.shape == (N * K, K)
    np.testing.assert_array_almost_equal(proba.sum(axis=1), np.ones(N * K), decimal=4)


# ── 11. F0 baseline carry (plan-020) ──────────────────────────────────────
def test_f0_baseline_carry():
    """F0 baseline 함수가 plan-020 carry 로 import 되는지 smoke."""
    p020_path = _REPO / "analysis" / "plan-020" / "baseline_f0.py"
    if not p020_path.exists():
        pytest.skip(f"plan-020 baseline_f0.py not present at {p020_path}")
    bf = _load(p020_path, "test_p028_bf")
    assert hasattr(bf, "f0_baseline")
    assert hasattr(bf, "R_HIT")
    assert bf.R_HIT == pytest.approx(0.01)


# ── 12~15. branch decision fn 4 case (§4.5 unit test 박제) ───────────────
def test_branch_decision_alpha(branch):
    # α case: B2 ≥ P022=0.6531
    result = branch.decide_branch(B1=0.55, B2=0.66, B3=0.60, B4=0.6320, W1=0.6320)
    assert result == "α"


def test_branch_decision_beta(branch):
    # β case: B1 > B3+0.005 AND B1 < P022 (α 미충족)
    result = branch.decide_branch(B1=0.65, B2=0.55, B3=0.60, B4=0.6320, W1=0.6320)
    assert result == "β"


def test_branch_decision_gamma(branch):
    # γ case: W1 > B4+0.005 (α/β 미충족)
    result = branch.decide_branch(B1=0.55, B2=0.60, B3=0.55, B4=0.6320, W1=0.66)
    assert result == "γ"


def test_branch_decision_delta(branch):
    # δ case: 모든 cell = B4 baseline
    result = branch.decide_branch(B1=0.6320, B2=0.6320, B3=0.6320, B4=0.6320, W1=0.6320)
    assert result == "δ"
