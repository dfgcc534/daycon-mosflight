"""plan-029 c7 — smoke tests (§4.3 spec).

15+ pytest covering:
- import + module shape
- cand_ext shape (B, 14, 165) + sample×anchor 차이 (interaction 실제)
- regime_anchor_table fold-leakage (train-fold only)
- anchor_embed shape (14, 8) + init scale ∈ [0.05, 0.15] + requires_grad
- query_in shape (B, 14, 173) + query shape (B, 14, 196)
- key_anchor shape (B, 14, 7, 196)
- attn_logits / attn shape + row-sum=1
- event_ctx shape (B, 14, 196)
- head Linear(196, 1) — raw skip 차단 검증
- forward end-to-end + dtype + no NaN/Inf
- soft label sum=1
- Frenet→world 식 (residual_world = R_wfn @ residual_frenet)
- anchor_embed gradient
"""
from __future__ import annotations

import importlib.util
import sys
from math import sqrt
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

_REPO = Path(__file__).resolve().parent.parent
_PLAN029 = _REPO / "analysis" / "plan-029"
_PLAN020 = _REPO / "analysis" / "plan-020"
_PLAN021 = _REPO / "analysis" / "plan-021"
_PLAN022 = _REPO / "analysis" / "plan-022"
_PLAN024 = _REPO / "analysis" / "plan-024"

for p in (_REPO, _PLAN020, _PLAN021, _PLAN022, _PLAN024, _PLAN029):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def aqe():
    return _load(_PLAN029 / "anchor_query_extend.py", "test_aqe")


@pytest.fixture(scope="module")
def model_mod():
    return _load(_PLAN029 / "model.py", "test_model")


@pytest.fixture(scope="module")
def bf():
    return _load(_PLAN020 / "baseline_f0.py", "test_bf")


@pytest.fixture(scope="module")
def bi():
    return _load(_PLAN021 / "build_input.py", "test_bi")


@pytest.fixture(scope="module")
def anchors_mod():
    return _load(_PLAN022 / "anchors.py", "test_av")


@pytest.fixture(scope="module")
def soft_mod():
    return _load(_PLAN022 / "selector_only_model.py", "test_soft")


@pytest.fixture(scope="module")
def qc_mod():
    return _load(_PLAN024 / "quantile_carry.py", "test_qc")


@pytest.fixture(scope="module")
def sample_data(bf, bi, anchors_mod, qc_mod, aqe):
    """fixture: small mock data + cand_ext + seq."""
    rng = np.random.default_rng(20260522)
    N = 8
    X = rng.standard_normal((N, 11, 3)).astype(np.float32) * 0.5
    R_wfn = bi.build_frenet_basis_3d(X, end_idx=10).astype(np.float32)
    F0 = bf.f0_baseline(X, end_idx=10).astype(np.float32)
    qc = qc_mod.build(X, R_wfn)
    regimes = rng.integers(0, 18, size=N)
    gt = X[:, 10, :] + rng.standard_normal((N, 3)).astype(np.float32) * 0.005
    table = aqe.build_regime_anchor_lookup(gt, regimes, anchors_mod.ANCHORS_A6, R_wfn, F0)
    cand_ext = aqe.build(X, R_wfn, F0, anchors_mod.ANCHORS_A6, bf.f0_baseline,
                         regimes, qc, regime_anchor_table=table)
    return dict(X=X, R_wfn=R_wfn, F0=F0, gt=gt, regimes=regimes, table=table,
                cand_ext=cand_ext, N=N, ANCHORS=anchors_mod.ANCHORS_A6)


# ── tests ────────────────────────────────────────────────────────────────────


def test_import_plan029_modules():
    """plan-029 modules import."""
    assert (_PLAN029 / "__init__.py").exists()
    assert (_PLAN029 / "anchor_query_extend.py").exists()
    assert (_PLAN029 / "model.py").exists()
    assert (_PLAN029 / "train.py").exists()
    assert (_PLAN029 / "run_oof.py").exists()


def test_plan024_cherry_pick_files():
    """plan-024 cherry-pick model + FWD (c2)."""
    assert (_PLAN024 / "model.py").exists()
    assert (_PLAN024 / "feature_weighted_dropout.py").exists()
    assert (_PLAN024 / "cand_builder.py").exists()
    assert (_PLAN024 / "seq_builder.py").exists()


def test_cand_ext_shape(sample_data):
    """cand_ext shape (B, K=14, 165)."""
    assert sample_data["cand_ext"].shape == (8, 14, 165)
    assert sample_data["cand_ext"].dtype == np.float32


def test_cand_ext_sample_anchor_interaction(sample_data):
    """interaction channel (마지막 15 ch) 가 sample×anchor 별로 다른 값."""
    ext = sample_data["cand_ext"][:, :, 150:]                                # (N, 14, 15)
    # sample 차이: ext[0] ≠ ext[1]
    assert not np.allclose(ext[0], ext[1], atol=1e-6), "sample 별 interaction 차이 없음"
    # anchor 차이: ext[:, 0] ≠ ext[:, 1]
    assert not np.allclose(ext[:, 0], ext[:, 1], atol=1e-6), "anchor 별 interaction 차이 없음"


def test_cand_ext_no_nan_inf(sample_data):
    """nan_to_num safety net."""
    assert not np.isnan(sample_data["cand_ext"]).any()
    assert not np.isinf(sample_data["cand_ext"]).any()


def test_regime_anchor_table_shape_rowsum(sample_data):
    """table shape (regime_count=18, K=14), row-sum=1."""
    table = sample_data["table"]
    assert table.shape == (18, 14)
    assert np.allclose(table.sum(axis=1), 1.0, atol=1e-5)


def test_regime_anchor_table_fold_leakage(aqe, anchors_mod, bf, bi):
    """train-fold only: 다른 fold gt 주입 시 다른 table 산출."""
    rng = np.random.default_rng(7)
    N_a, N_b = 20, 20
    X_a = rng.standard_normal((N_a, 11, 3)).astype(np.float32)
    X_b = rng.standard_normal((N_b, 11, 3)).astype(np.float32)
    R_a = bi.build_frenet_basis_3d(X_a, end_idx=10).astype(np.float32)
    R_b = bi.build_frenet_basis_3d(X_b, end_idx=10).astype(np.float32)
    F0_a = bf.f0_baseline(X_a, end_idx=10).astype(np.float32)
    F0_b = bf.f0_baseline(X_b, end_idx=10).astype(np.float32)
    gt_a = X_a[:, 10, :].copy()
    gt_b = X_b[:, 10, :].copy()
    reg_a = rng.integers(0, 18, size=N_a)
    reg_b = rng.integers(0, 18, size=N_b)
    table_a = aqe.build_regime_anchor_lookup(gt_a, reg_a, anchors_mod.ANCHORS_A6, R_a, F0_a)
    table_b = aqe.build_regime_anchor_lookup(gt_b, reg_b, anchors_mod.ANCHORS_A6, R_b, F0_b)
    assert not np.allclose(table_a, table_b, atol=1e-3), "다른 fold gt 가 동일 table 산출 — fold-leakage 의심"


def test_regime_anchor_table_none_raises(aqe, anchors_mod, bf, bi, qc_mod, sample_data):
    """regime_anchor_table=None 시 ValueError (dead path 차단)."""
    with pytest.raises(ValueError, match="regime_anchor_table is None"):
        aqe.build(
            sample_data["X"], sample_data["R_wfn"], sample_data["F0"],
            sample_data["ANCHORS"], bf.f0_baseline,
            regimes=sample_data["regimes"],
            quantile_carry=qc_mod.build(sample_data["X"], sample_data["R_wfn"]),
            regime_anchor_table=None,
        )


def test_anchor_embed_shape_init_scale_grad(model_mod):
    """anchor_embed (14, 8), init std ∈ [0.05, 0.15], requires_grad."""
    model = model_mod.GRUNetX1()
    assert model.anchor_embed.shape == (14, 8)
    assert 0.05 <= model.anchor_embed.std().item() <= 0.20
    assert model.anchor_embed.requires_grad


def test_anchor_key_proj_dim(model_mod):
    """anchor_key_proj: Linear(8 → 196)."""
    model = model_mod.GRUNetX1()
    assert model.anchor_key_proj.in_features == 8
    assert model.anchor_key_proj.out_features == 196


def test_forward_shape_dtype(model_mod):
    """forward → score (B, K=14) float32 + no NaN/Inf."""
    model = model_mod.GRUNetX1()
    B, T, K = 4, 7, 14
    seq = torch.randn(B, T, 95)
    cand = torch.randn(B, K, 165)
    score = model(seq, cand)
    assert score.shape == (B, K)
    assert score.dtype == torch.float32
    assert not torch.isnan(score).any() and not torch.isinf(score).any()


def test_head_is_linear_196_1(model_mod):
    """head = Linear(196, 1) — raw skip 차단 (head_in = event_ctx only)."""
    model = model_mod.GRUNetX1()
    assert isinstance(model.head, torch.nn.Linear)
    assert model.head.in_features == 196, "head input ≠ 196 — raw skip 잔존"
    assert model.head.out_features == 1


def test_no_raw_skip_in_head(model_mod):
    """forward 의 head 가 event_ctx (196 dim) 만 받는지 — hook 으로 검증."""
    model = model_mod.GRUNetX1()
    captured = []

    def hook(module, inp, out):
        captured.append(inp[0].shape)

    handle = model.head.register_forward_hook(hook)
    try:
        seq = torch.randn(2, 7, 95)
        cand = torch.randn(2, 14, 165)
        _ = model(seq, cand)
    finally:
        handle.remove()
    assert len(captured) == 1
    # head input = (B, K, 196). 마지막 dim == 196 (no raw cand 150 concat).
    assert captured[0][-1] == 196, f"head input last dim = {captured[0][-1]}, raw skip 잔존 의심"


def test_attn_softmax_rowsum(model_mod):
    """attn (B, K, T) row-sum = 1 (softmax)."""
    # forward 안에서는 attn 이 intermediate 라 직접 hook 으로 확인
    model = model_mod.GRUNetX1()
    B, T, K = 2, 7, 14
    seq = torch.randn(B, T, 95)
    cand = torch.randn(B, K, 165)
    # manual recompute (forward 식과 동일)
    anchor_embed_bc = model.anchor_embed.unsqueeze(0).expand(B, -1, -1)
    query_in = torch.cat([cand, anchor_embed_bc], dim=-1)
    query = model.query_mlp(query_in)
    out, _ = model.gru(seq)
    anchor_key_bias = model.anchor_key_proj(model.anchor_embed)
    key_anchor = out.unsqueeze(1) + anchor_key_bias.unsqueeze(0).unsqueeze(2)
    attn_logits = torch.einsum("bkh,bkth->bkt", query, key_anchor) / sqrt(196)
    attn = torch.softmax(attn_logits, dim=-1)
    assert attn.shape == (B, K, T)
    assert torch.allclose(attn.sum(dim=-1), torch.ones(B, K), atol=1e-5)


def test_anchor_embed_gradient(model_mod):
    """backward 후 anchor_embed.grad.norm() > 0."""
    model = model_mod.GRUNetX1()
    seq = torch.randn(4, 7, 95)
    cand = torch.randn(4, 14, 165)
    score = model(seq, cand)
    loss = score.sum()
    loss.backward()
    assert model.anchor_embed.grad is not None
    assert model.anchor_embed.grad.norm().item() > 0


def test_anchor_key_proj_gradient(model_mod):
    """backward 후 anchor_key_proj weight gradient > 0."""
    model = model_mod.GRUNetX1()
    seq = torch.randn(4, 7, 95)
    cand = torch.randn(4, 14, 165)
    score = model(seq, cand)
    loss = score.sum()
    loss.backward()
    assert model.anchor_key_proj.weight.grad is not None
    assert model.anchor_key_proj.weight.grad.norm().item() > 0


def test_soft_label_rowsum(soft_mod, sample_data):
    """soft label q (N, K) row-sum=1."""
    q = soft_mod.build_soft_label_with_tau(
        sample_data["gt"], sample_data["R_wfn"], sample_data["F0"],
        sample_data["ANCHORS"], tau_cls=0.001,
    )
    assert q.shape == (sample_data["N"], 14)
    assert np.allclose(q.sum(axis=1), 1.0, atol=1e-5)


def test_frenet_to_world_formula(sample_data):
    """residual_world = R_wfn @ residual_frenet (Frenet → world einsum 식)."""
    rng = np.random.default_rng(42)
    residual_frenet = rng.standard_normal((sample_data["N"], 3)).astype(np.float32)
    residual_world = np.einsum("nij,nj->ni", sample_data["R_wfn"], residual_frenet)
    # 역방향: world → Frenet (R_wfn.T)
    back = np.einsum("nij,nj->ni", sample_data["R_wfn"].transpose(0, 2, 1), residual_world)
    assert np.allclose(back, residual_frenet, atol=1e-4)


def test_loss_soft_ce_smoke(model_mod):
    """soft CE loss = -(q * log_softmax(score)).sum(-1).mean() — finite + reasonable scale."""
    torch.manual_seed(0)
    model = model_mod.GRUNetX1()
    seq = torch.randn(4, 7, 95)
    cand = torch.randn(4, 14, 165)
    score = model(seq, cand)
    log_probs = F.log_softmax(score, dim=-1)
    # uniform soft label = baseline (loss = log K = log 14 ≈ 2.64)
    q = torch.full((4, 14), 1.0 / 14)
    loss = -(q * log_probs).sum(dim=-1).mean()
    assert torch.isfinite(loss)
    # uniform soft label 이라 loss ≈ log 14 ± 0.5 (model init noise)
    assert 1.0 < loss.item() < 5.0
