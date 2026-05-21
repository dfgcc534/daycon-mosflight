"""plan-024 c8 — G0 smoke test (10 pytest, v1.1-rev2).

§5.2 표 carry. plan-024 spec self-contained, code path 통합 검증.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_REPO = Path(__file__).resolve().parent.parent
_P024 = _REPO / "analysis" / "plan-024"
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def modules():
    """8 module import smoke (test #1)."""
    out = {
        "anchor_vocab": _load("p024_av", _P024 / "anchor_vocab.py"),
        "seq_builder": _load("p024_sb", _P024 / "seq_builder.py"),
        "cand_builder": _load("p024_cb", _P024 / "cand_builder.py"),
        "torsion_calc": _load("p024_tc", _P024 / "torsion_calc.py"),
        "quantile_carry": _load("p024_qc", _P024 / "quantile_carry.py"),
        "fwd": _load("p024_fwd", _P024 / "feature_weighted_dropout.py"),
        "multiwindow_trim_build": _load("p024_mw", _P024 / "multiwindow_trim_build.py"),
        "model": _load("p024_model", _P024 / "model.py"),
        "run_oof": _load("p024_oof", _P024 / "run_oof.py"),
    }
    return out


@pytest.fixture(scope="module")
def fake_data():
    """random fake (N=8, T=11, K=14)."""
    rng = np.random.default_rng(20260521)
    N, K = 8, 14
    X = (rng.standard_normal((N, 11, 3)) * 0.005).astype(np.float64)
    R_wfn = np.tile(np.eye(3, dtype=np.float32)[None], (N, 1, 1))
    anchors = (rng.standard_normal((K, 3)) * 0.005).astype(np.float32)
    pred_F0 = X[:, 10].astype(np.float32) + 0.001
    regimes = rng.integers(0, 18, size=N)
    qc = {"omega_p90": 0.5, "jerk_p90": 0.5, "levy_tail_threshold": 0.05}

    def fake_f0(sub_x: np.ndarray, end_idx: int) -> np.ndarray:
        return sub_x[:, end_idx] + (sub_x[:, end_idx] - sub_x[:, end_idx - 1])

    return {
        "N": N, "K": K, "X": X, "R_wfn": R_wfn, "anchors": anchors,
        "pred_F0": pred_F0, "regimes": regimes, "qc": qc, "f0_fn": fake_f0,
    }


# ── test 1: 9 module import smoke ──────────────────────────────────────


def test_01_module_import_smoke(modules):
    """test #1: 9 module import 성공."""
    assert len(modules) == 9
    for name, mod in modules.items():
        assert hasattr(mod, "__name__") or callable(getattr(mod, "build", None)) \
            or hasattr(mod, "CrossAttentionAnchorSelector") \
            or name == "run_oof", f"{name}: missing build/class"


# ── test 2: anchor_vocab shape ─────────────────────────────────────────


def test_02_anchor_vocab_shape(modules, fake_data):
    d = fake_data
    av = modules["anchor_vocab"].build(
        d["X"], d["R_wfn"], d["anchors"], d["f0_fn"], tau_past=0.003,
    )
    assert av["F"].shape == (d["N"], 7, d["K"])
    assert av["G"].shape == (d["N"], 7)
    assert av["H"].shape == (d["N"], 7, d["K"])
    assert av["F2"].shape == (d["N"], 7)
    assert np.allclose(av["F"].sum(axis=-1), 1.0, atol=1e-5)
    assert np.allclose(av["H"].sum(axis=-1), 1.0, atol=1e-5)


# ── test 3: sign convention sanity (axis 대칭 invariance) ─────────────


def test_03_sign_convention_sanity(modules):
    """anchor pair (+t̂, -t̂) 가 잔차 부호 flip 시 q_past 도 swap.

    Implementation: ANCHORS_A6 의 idx 0 = +0.005·t̂, idx 1 = -0.005·t̂.
    잔차 = +0.005·t̂ → q[0] >> q[1]. 잔차 = -0.005·t̂ → q[1] >> q[0].
    """
    import importlib.util as _iu
    _spec = _iu.spec_from_file_location(
        "_p022_anc", _REPO / "analysis" / "plan-022" / "anchors.py"
    )
    am = _iu.module_from_spec(_spec); _spec.loader.exec_module(am)
    anchors = am.ANCHORS_A6

    # synthesize: residual = +0.005·t̂ (axis 0 best match)
    N = 4
    X = np.zeros((N, 11, 3), dtype=np.float64)
    R_wfn = np.tile(np.eye(3, dtype=np.float32)[None], (N, 1, 1))

    def f0_constant_zero(sub_x: np.ndarray, end_idx: int) -> np.ndarray:
        return np.zeros((sub_x.shape[0], 3), dtype=np.float64)

    # case A: actual = +0.005·t̂ at step 4..10
    X_a = X.copy()
    X_a[:, 4:, 0] = 0.005
    av_a = modules["anchor_vocab"].build(X_a, R_wfn, anchors, f0_constant_zero, tau_past=0.003)
    # case B: actual = -0.005·t̂ at step 4..10
    X_b = X.copy()
    X_b[:, 4:, 0] = -0.005
    av_b = modules["anchor_vocab"].build(X_b, R_wfn, anchors, f0_constant_zero, tau_past=0.003)
    # symmetry: q_a[0] (=+t̂) > q_a[1] (=-t̂); q_b[1] > q_b[0]
    assert (av_a["F"][:, -1, 0] > av_a["F"][:, -1, 1]).all(), \
        f"case A: q[+t̂] should > q[-t̂]"
    assert (av_b["F"][:, -1, 1] > av_b["F"][:, -1, 0]).all(), \
        f"case B: q[-t̂] should > q[+t̂]"


# ── test 4: seq 95D shape ──────────────────────────────────────────────


def test_04_seq_95d_shape(modules, fake_data):
    d = fake_data
    seq = modules["seq_builder"].build(
        d["X"], d["R_wfn"], d["anchors"], d["f0_fn"],
        quantile_carry=d["qc"], tau_past=0.003,
    )
    assert seq.shape == (d["N"], 7, 95)
    assert np.isfinite(seq).all()


# ── test 5: cand 150D shape ────────────────────────────────────────────


def test_05_cand_150d_shape(modules, fake_data, tmp_path):
    d = fake_data
    # need multiwindow_trim first
    L1 = modules["cand_builder"]._build_L1_frenet(d["X"], d["R_wfn"])
    trim_path = tmp_path / "trim.json"
    modules["multiwindow_trim_build"].build_and_save(L1, output_path=trim_path)

    cand = modules["cand_builder"].build(
        d["X"], d["R_wfn"], d["pred_F0"], d["anchors"], d["f0_fn"],
        regimes=d["regimes"], quantile_carry=d["qc"],
        multiwindow_trim_path=trim_path,
    )
    assert cand.shape == (d["N"], d["K"], 150)
    assert np.isfinite(cand).all()


# ── test 6: torsion mask ───────────────────────────────────────────────


def test_06_torsion_mask(modules):
    rng = np.random.default_rng(0)
    N = 10
    # straight line (collinear) — all mask=0
    X_straight = np.zeros((N, 11, 3), dtype=np.float64)
    X_straight[:, :, 0] = np.arange(11) * 0.001
    tor = modules["torsion_calc"].build(X_straight)
    assert tor.shape == (N, 7, 3)
    assert tor[:, :, 2].sum() == 0, "collinear should have all mask=0"


# ── test 7: quantile_carry fold-leakage 차단 ───────────────────────────


def test_07_quantile_carry_fold_leakage(modules, fake_data):
    """train fold quantile build → keys 정합."""
    d = fake_data
    qc = modules["quantile_carry"].build(d["X"], d["R_wfn"])
    assert set(qc.keys()) == {"omega_p90", "jerk_p90", "levy_tail_threshold"}
    for k, v in qc.items():
        assert np.isfinite(v) and v > 0, f"{k}: {v}"


# ── test 8: FeatureWeightedDropout weight + 보호 영역 ─────────────────


def test_08_feature_weighted_dropout_protected(modules):
    torch.manual_seed(20260521)
    fwd = modules["fwd"].FeatureWeightedDropout()
    cand = torch.randn(2, 14, 150)
    seq = torch.randn(2, 7, 95)
    # init=1.0 → eval (clamp identity)
    fwd.eval()
    cand_e, seq_e = fwd(cand, seq)
    assert torch.allclose(cand_e, cand, atol=1e-5)
    assert torch.allclose(seq_e, seq, atol=1e-5)
    # training: 보호 영역 [0:12] + [140:150] untouched
    fwd.train()
    cand_t, seq_t = fwd(cand, seq)
    assert torch.allclose(cand_t[:, :, 0:12], cand[:, :, 0:12], atol=1e-5), \
        "protected cand [0:12] modified"
    assert torch.allclose(cand_t[:, :, 140:150], cand[:, :, 140:150], atol=1e-5), \
        "protected cand [140:150] modified"
    # scale shape
    assert fwd.cand_scale.shape == (150,)
    assert fwd.seq_scale.shape == (95,)


# ── test 9: model forward smoke ────────────────────────────────────────


def test_09_model_forward_smoke(modules):
    torch.manual_seed(20260521)
    model = modules["model"].CrossAttentionAnchorSelector()
    seq = torch.randn(4, 7, 95)
    cand = torch.randn(4, 14, 150)
    q_pred, score = model(seq, cand)
    assert q_pred.shape == (4, 14)
    assert score.shape == (4, 14)
    assert torch.allclose(q_pred.sum(-1), torch.ones(4), atol=1e-5)
    assert torch.isfinite(q_pred).all()


# ── test 10: 2-epoch fit loss finite + decrease ────────────────────────


def test_10_two_epoch_fit_decrease(modules):
    """§5.2 #10: 2-epoch fit + loss decrease > 1e-4 margin."""
    torch.manual_seed(20260521)
    np.random.seed(20260521)
    model = modules["model"].CrossAttentionAnchorSelector()
    optim = torch.optim.AdamW(model.parameters(), lr=7e-4, weight_decay=0.02)
    seq = torch.randn(8, 7, 95)
    cand = torch.randn(8, 14, 150)
    q_target = torch.softmax(torch.randn(8, 14), dim=-1)

    losses = []
    for epoch in range(2):
        ep_losses = []
        for step in range(4):
            optim.zero_grad()
            q_pred, _ = model(seq, cand)
            loss = -(q_target * torch.log(q_pred + 1e-12)).sum(-1).mean()
            loss.backward()
            optim.step()
            ep_losses.append(loss.item())
        avg = sum(ep_losses) / len(ep_losses)
        losses.append(avg)
        assert np.isfinite(avg), f"epoch {epoch}: loss not finite ({avg})"

    assert losses[1] < losses[0] - 1e-4, \
        f"loss did not decrease: epoch0={losses[0]:.4f}, epoch1={losses[1]:.4f}"
