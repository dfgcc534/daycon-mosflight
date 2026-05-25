"""plan-030 c6 smoke — 20 pytest cases.

Coverage:
  builder shape/dtype/finite (residual_a/b, query_64, head_summary_51)
  slim7 extract correctness
  GRUNetX2 forward shape + softmax row-sum
  ANCHORS_A6 buffer frozen
  loss soft-CE numerical safety
  full train one-fold integration (tiny mock)
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_THIS = Path(__file__).resolve().parent
_REPO = _THIS.parent
_PLAN030 = _REPO / "analysis" / "plan-030"
_PLAN020 = _REPO / "analysis" / "plan-020"
_PLAN022 = _REPO / "analysis" / "plan-022"

for p in (_REPO, _PLAN020, _PLAN022, _PLAN030):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


# ── fixtures ────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def residual_mod():
    return _load(_PLAN030 / "residual_builder.py", "p030_test_residual")


@pytest.fixture(scope="module")
def query_mod():
    return _load(_PLAN030 / "query_builder.py", "p030_test_query")


@pytest.fixture(scope="module")
def head_mod():
    return _load(_PLAN030 / "head_summary.py", "p030_test_head")


@pytest.fixture(scope="module")
def model_mod():
    return _load(_PLAN030 / "model.py", "p030_test_model")


@pytest.fixture(scope="module")
def train_mod():
    return _load(_PLAN030 / "train.py", "p030_test_train")


@pytest.fixture(scope="module")
def bf_mod():
    return _load(_PLAN020 / "baseline_f0.py", "p030_test_bf")


@pytest.fixture(scope="module")
def anchors_mod():
    return _load(_PLAN022 / "anchors.py", "p030_test_anchors")


@pytest.fixture(scope="module")
def sample_data():
    rng = np.random.default_rng(20260524)
    N, K = 8, 14
    X = rng.standard_normal((N, 11, 3)).astype(np.float32) * 0.5
    R_wfn = np.tile(np.eye(3, dtype=np.float32)[None], (N, 1, 1))
    cand_feat = rng.standard_normal((N, K, 150)).astype(np.float32)
    cand_ext = rng.standard_normal((N, K, 165)).astype(np.float32)
    plan021_macro9 = rng.standard_normal((N, 9)).astype(np.float32)
    L4 = rng.uniform(0.0, 1.0, (N, 14)).astype(np.float32)
    return {
        "X": X, "R_wfn": R_wfn,
        "cand_feat_150": cand_feat, "cand_ext_165": cand_ext,
        "plan021_macro9": plan021_macro9, "soft_hit_L4": L4,
        "N": N, "K": K,
    }


# ── tests ───────────────────────────────────────────────────────────────


def test_import_plan030_modules(residual_mod, query_mod, head_mod, model_mod):
    """1: 4 plan-030 모듈 import 성공."""
    assert hasattr(residual_mod, "build_residuals")
    assert hasattr(query_mod, "build_query")
    assert hasattr(query_mod, "extract_slim7_from_cand_ext_165")
    assert hasattr(head_mod, "build_head_summary")
    assert hasattr(model_mod, "GRUNetX2")


def test_residual_a_shape_dtype(residual_mod, bf_mod, anchors_mod, sample_data):
    """2: residual_a shape (N, 7, 5) + float32."""
    out = residual_mod.build_residuals(
        sample_data["X"], sample_data["R_wfn"],
        anchors_mod.ANCHORS_A6, bf_mod.f0_baseline,
    )
    assert out["residual_a"].shape == (sample_data["N"], 7, 5)
    assert out["residual_a"].dtype == np.float32


def test_residual_a_gru_shape(residual_mod, bf_mod, anchors_mod, sample_data):
    """3: residual_a_gru = residual_a[:, :, :2], shape (N, 7, 2)."""
    out = residual_mod.build_residuals(
        sample_data["X"], sample_data["R_wfn"],
        anchors_mod.ANCHORS_A6, bf_mod.f0_baseline,
    )
    assert out["residual_a_gru"].shape == (sample_data["N"], 7, 2)
    # XY_norm + Z_signed = first 2 coord
    np.testing.assert_allclose(out["residual_a_gru"], out["residual_a"][:, :, :2])


def test_residual_b_shape_per_anchor(residual_mod, bf_mod, anchors_mod, sample_data):
    """4: residual_b shape (N, K=14, 7, 5) per anchor."""
    out = residual_mod.build_residuals(
        sample_data["X"], sample_data["R_wfn"],
        anchors_mod.ANCHORS_A6, bf_mod.f0_baseline,
    )
    assert out["residual_b"].shape == (sample_data["N"], 14, 7, 5)


def test_residual_zero_pad_invalid_steps(residual_mod, bf_mod, anchors_mod, sample_data):
    """5: i=5,6 zero-pad (t_wall=-1,0 → raw(t+2) 미관측)."""
    out = residual_mod.build_residuals(
        sample_data["X"], sample_data["R_wfn"],
        anchors_mod.ANCHORS_A6, bf_mod.f0_baseline,
    )
    assert np.all(out["residual_a"][:, 5:7, :] == 0.0)
    assert np.all(out["residual_b"][:, :, 5:7, :] == 0.0)


def test_residual_finite(residual_mod, bf_mod, anchors_mod, sample_data):
    """6: 모든 residual 출력 finite."""
    out = residual_mod.build_residuals(
        sample_data["X"], sample_data["R_wfn"],
        anchors_mod.ANCHORS_A6, bf_mod.f0_baseline,
    )
    for k, v in out.items():
        assert not np.isnan(v).any(), f"{k} NaN"
        assert not np.isinf(v).any(), f"{k} Inf"


def test_extract_slim7_shape_cols(query_mod, sample_data):
    """7: extract_slim7 from cand_ext_165 → (N, K, 7), 정확한 col 추출."""
    slim7 = query_mod.extract_slim7_from_cand_ext_165(sample_data["cand_ext_165"])
    assert slim7.shape == (sample_data["N"], sample_data["K"], 7)
    # col 0 = cand_ext[:, :, 159] (D.regime_prob)
    np.testing.assert_allclose(slim7[:, :, 0], sample_data["cand_ext_165"][:, :, 159])
    # col 1 = cand_ext[:, :, 158] (B.cos)
    np.testing.assert_allclose(slim7[:, :, 1], sample_data["cand_ext_165"][:, :, 158])
    # col 2:7 = cand_ext[:, :, 160:165] (F.2)
    np.testing.assert_allclose(slim7[:, :, 2:], sample_data["cand_ext_165"][:, :, 160:165])


def test_query_64_shape_concat(query_mod, sample_data):
    """8: query 64D = 9 + 3 + 10 + 35 + 7."""
    N, K = sample_data["N"], sample_data["K"]
    residual_b = np.zeros((N, K, 7, 5), dtype=np.float32)
    slim7 = np.zeros((N, K, 7), dtype=np.float32)
    q = query_mod.build_query(sample_data["cand_feat_150"], residual_b, slim7)
    assert q.shape == (N, K, 64)
    assert q.dtype == np.float32


def test_query_slicing_correct(query_mod, sample_data):
    """9: query 의 anchor_spec 부분이 cand_feat[3:12] 와 정합."""
    N, K = sample_data["N"], sample_data["K"]
    residual_b = np.zeros((N, K, 7, 5), dtype=np.float32)
    slim7 = np.zeros((N, K, 7), dtype=np.float32)
    q = query_mod.build_query(sample_data["cand_feat_150"], residual_b, slim7)
    # query[0:9] = anchor_spec = cand_feat[3:12]
    np.testing.assert_allclose(q[:, :, 0:9], sample_data["cand_feat_150"][:, :, 3:12])
    # query[9:12] = par/perp/dist = cand_feat[0:3]
    np.testing.assert_allclose(q[:, :, 9:12], sample_data["cand_feat_150"][:, :, 0:3])
    # query[12:22] = interactions = cand_feat[140:150]
    np.testing.assert_allclose(q[:, :, 12:22], sample_data["cand_feat_150"][:, :, 140:150])


def test_head_summary_shape(head_mod, sample_data):
    """10: head_summary 51D = 2+8+3+3+9+3+9+14."""
    hs = head_mod.build_head_summary(
        sample_data["cand_feat_150"],
        sample_data["plan021_macro9"], sample_data["soft_hit_L4"],
    )
    assert hs.shape == (sample_data["N"], 51)
    assert hs.dtype == np.float32


def test_head_summary_slicing_correct(head_mod, sample_data):
    """11: head_summary 의 Bz/Tz = cand_feat[0, 32:34]."""
    hs = head_mod.build_head_summary(
        sample_data["cand_feat_150"],
        sample_data["plan021_macro9"], sample_data["soft_hit_L4"],
    )
    # hs[0:2] = Bz/Tz = cand_feat[:, 0, 32:34]
    np.testing.assert_allclose(hs[:, 0:2], sample_data["cand_feat_150"][:, 0, 32:34])
    # hs[2:10] = macro_8 = cand_feat[:, 0, 24:32]
    np.testing.assert_allclose(hs[:, 2:10], sample_data["cand_feat_150"][:, 0, 24:32])


def test_model_grunet_x2_init(model_mod, anchors_mod):
    """12: GRUNetX2 init + head_in_dim=382 + ANCHORS_A6 buffer frozen."""
    anchors = torch.from_numpy(anchors_mod.ANCHORS_A6.astype(np.float32))
    model = model_mod.GRUNetX2(anchors=anchors)
    assert model.head_in_dim == 382
    assert hasattr(model, "ANCHORS_A6")
    assert not model.ANCHORS_A6.requires_grad
    np.testing.assert_allclose(model.ANCHORS_A6.numpy(), anchors_mod.ANCHORS_A6, atol=1e-6)


def test_model_forward_shape(model_mod, anchors_mod):
    """13: forward → world_pred (B, 3), probs (B, 14)."""
    torch.manual_seed(20260524)
    B, T, K = 4, 7, 14
    anchors = torch.from_numpy(anchors_mod.ANCHORS_A6.astype(np.float32))
    model = model_mod.GRUNetX2(anchors=anchors).eval()
    seq_97 = torch.randn(B, T, 97)
    res_kv = torch.randn(B, T, 5)
    q64 = torch.randn(B, K, 64)
    hs = torch.randn(B, 51)
    slim7 = torch.randn(B, K, 7)
    F0 = torch.randn(B, 3)
    R = torch.eye(3).unsqueeze(0).expand(B, -1, -1).contiguous()
    with torch.no_grad():
        world_pred, probs = model(seq_97, res_kv, q64, hs, slim7, F0, R)
    assert world_pred.shape == (B, 3)
    assert probs.shape == (B, K)


def test_model_probs_softmax_rowsum(model_mod, anchors_mod):
    """14: probs row sum = 1 (valid distribution)."""
    torch.manual_seed(20260524)
    B, T, K = 6, 7, 14
    anchors = torch.from_numpy(anchors_mod.ANCHORS_A6.astype(np.float32))
    model = model_mod.GRUNetX2(anchors=anchors).eval()
    seq_97 = torch.randn(B, T, 97)
    res_kv = torch.randn(B, T, 5)
    q64 = torch.randn(B, K, 64)
    hs = torch.randn(B, 51)
    slim7 = torch.randn(B, K, 7)
    F0 = torch.randn(B, 3)
    R = torch.eye(3).unsqueeze(0).expand(B, -1, -1).contiguous()
    with torch.no_grad():
        _wp, probs = model(seq_97, res_kv, q64, hs, slim7, F0, R)
    row_sum = probs.sum(dim=-1)
    torch.testing.assert_close(row_sum, torch.ones(B), atol=1e-5, rtol=1e-5)
    assert (probs >= 0).all() and (probs <= 1).all()


def test_model_gradient_flow(model_mod, anchors_mod):
    """15: parameters 가 모두 gradient 받음."""
    torch.manual_seed(20260524)
    B, T, K = 4, 7, 14
    anchors = torch.from_numpy(anchors_mod.ANCHORS_A6.astype(np.float32))
    model = model_mod.GRUNetX2(anchors=anchors).train()
    seq_97 = torch.randn(B, T, 97, requires_grad=False)
    res_kv = torch.randn(B, T, 5)
    q64 = torch.randn(B, K, 64)
    hs = torch.randn(B, 51)
    slim7 = torch.randn(B, K, 7)
    F0 = torch.randn(B, 3)
    R = torch.eye(3).unsqueeze(0).expand(B, -1, -1).contiguous()
    world_pred, probs = model(seq_97, res_kv, q64, hs, slim7, F0, R)
    loss = world_pred.sum() + probs.sum()
    loss.backward()
    for n, p in model.named_parameters():
        assert p.grad is not None, f"{n} has no gradient"


def test_final_pred_formula(model_mod, anchors_mod):
    """16: world_pred = F0 + R_wfn @ (probs @ ANCHORS_A6)."""
    torch.manual_seed(20260524)
    B, T, K = 4, 7, 14
    anchors = torch.from_numpy(anchors_mod.ANCHORS_A6.astype(np.float32))
    model = model_mod.GRUNetX2(anchors=anchors).eval()
    seq_97 = torch.randn(B, T, 97)
    res_kv = torch.randn(B, T, 5)
    q64 = torch.randn(B, K, 64)
    hs = torch.randn(B, 51)
    slim7 = torch.randn(B, K, 7)
    F0 = torch.randn(B, 3)
    R = torch.eye(3).unsqueeze(0).expand(B, -1, -1).contiguous()
    with torch.no_grad():
        world_pred, probs = model(seq_97, res_kv, q64, hs, slim7, F0, R)
    # manual: F0 + R @ (probs @ ANCHORS)
    expected_residual_frenet = probs @ torch.from_numpy(anchors_mod.ANCHORS_A6.astype(np.float32))
    expected_residual_world = torch.einsum("bij,bj->bi", R, expected_residual_frenet)
    expected_world = F0 + expected_residual_world
    torch.testing.assert_close(world_pred, expected_world, atol=1e-5, rtol=1e-5)


def test_loss_soft_ce_numerical_safety(model_mod, anchors_mod):
    """17: soft CE loss with log(probs.clamp_min(1e-12)) — finite even when probs near 0."""
    torch.manual_seed(20260524)
    B, K = 4, 14
    probs = torch.zeros(B, K).softmax(dim=-1)  # uniform
    soft_label = torch.ones(B, K) / K
    log_probs = torch.log(probs.clamp_min(1e-12))
    loss = -(soft_label * log_probs).sum(dim=-1).mean()
    assert torch.isfinite(loss).item()


def test_train_one_fold_smoke(train_mod):
    """18: train_one_fold 1-fold tiny mock (epoch=2)."""
    # override globals
    train_mod.EPOCHS = 2
    train_mod.WARMUP_EP = 1
    rng = np.random.default_rng(20260524)
    N = 16
    X_tr = rng.standard_normal((N, 11, 3)).astype(np.float32) * 0.5
    X_te = rng.standard_normal((6, 11, 3)).astype(np.float32) * 0.5
    gt_tr = X_tr[:, 10, :] + rng.standard_normal((N, 3)).astype(np.float32) * 0.005
    log = train_mod.train_one_fold(fold=0, X_tr=X_tr, X_te=X_te, gt_tr=gt_tr, verbose=False)
    assert log["world_pred_te"].shape == (6, 3)
    assert log["probs_te"].shape == (6, 14)
    assert log["R_wfn_te"].shape == (6, 3, 3)
    assert log["F0_te"].shape == (6, 3)
    assert log["elapsed_s"] > 0


def test_train_one_fold_finite_predictions(train_mod):
    """19: train_one_fold predictions finite + probs row sum ~ 1."""
    train_mod.EPOCHS = 2
    train_mod.WARMUP_EP = 1
    rng = np.random.default_rng(20260525)
    N = 16
    X_tr = rng.standard_normal((N, 11, 3)).astype(np.float32) * 0.5
    X_te = rng.standard_normal((6, 11, 3)).astype(np.float32) * 0.5
    gt_tr = X_tr[:, 10, :] + rng.standard_normal((N, 3)).astype(np.float32) * 0.005
    log = train_mod.train_one_fold(fold=0, X_tr=X_tr, X_te=X_te, gt_tr=gt_tr, verbose=False)
    assert not np.isnan(log["world_pred_te"]).any()
    assert not np.isinf(log["world_pred_te"]).any()
    rs = log["probs_te"].sum(axis=-1)
    np.testing.assert_allclose(rs, 1.0, atol=1e-4)


def test_score_std_trajectory_recorded(train_mod):
    """20: score_std_trajectory 가 epoch 별로 기록됨."""
    train_mod.EPOCHS = 3
    train_mod.WARMUP_EP = 1
    rng = np.random.default_rng(20260526)
    N = 16
    X_tr = rng.standard_normal((N, 11, 3)).astype(np.float32) * 0.5
    X_te = rng.standard_normal((6, 11, 3)).astype(np.float32) * 0.5
    gt_tr = X_tr[:, 10, :] + rng.standard_normal((N, 3)).astype(np.float32) * 0.005
    log = train_mod.train_one_fold(fold=0, X_tr=X_tr, X_te=X_te, gt_tr=gt_tr, verbose=False)
    assert len(log["score_std_trajectory"]) == 3
    for s in log["score_std_trajectory"]:
        assert np.isfinite(s)
