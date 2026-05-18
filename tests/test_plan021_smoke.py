"""plan-021 c5 — G0 smoke (4 module import + Frenet sanity + F0 reproduce)."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
P21 = REPO_ROOT / "analysis" / "plan-021"
P20 = REPO_ROOT / "analysis" / "plan-020"
sys.path.insert(0, str(REPO_ROOT))


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture(scope="module")
def modules():
    bf = _load("baseline_f0", P20 / "baseline_f0.py")
    bi = _load("build_input", P21 / "build_input.py")
    dh = _load("dual_head_model", P21 / "dual_head_model.py")
    return {"bf": bf, "bi": bi, "dh": dh}


@pytest.fixture(scope="module")
def synthetic():
    rng = np.random.default_rng(20260518)
    x = np.cumsum(rng.standard_normal((32, 11, 3)) * 0.01, axis=1).astype(np.float64)
    gt = x[:, -1] + rng.standard_normal((32, 3)) * 0.005
    return x, gt


def test_module_imports(modules):
    bi, dh, bf = modules["bi"], modules["dh"], modules["bf"]
    # build_input exports
    for sym in ("build_frenet_basis_3d", "to_frenet", "build_input_common",
                "build_input_lgbm_extra", "build_soft_label", "ANCHORS_FRENET",
                "DT", "H", "R_HIT", "R_HIT_LOOSE", "TAU_LOSS", "TAU_CLS"):
        assert hasattr(bi, sym), f"build_input.{sym} missing"
    assert bi.ANCHORS_FRENET.shape == (7, 3)
    # dual_head_model exports
    for sym in ("LgbmDualHead", "GRUDualHead", "soft_ce_loss", "smooth_hit_loss", "tau_for_epoch"):
        assert hasattr(dh, sym), f"dual_head_model.{sym} missing"
    # baseline_f0 carry
    assert callable(bf.f0_baseline)


def test_frenet_basis_orthonormality(modules, synthetic):
    bi = modules["bi"]
    x, _ = synthetic
    R = bi.build_frenet_basis_3d(x, end_idx=10)
    assert R.shape == (32, 3, 3)
    err = float(np.abs(R @ R.transpose(0, 2, 1) - np.eye(3)).max())
    assert err < 1e-5, f"orthonormality max err = {err}"
    assert bool((np.linalg.det(R) > 0).all()), "non right-handed basis"


def test_to_frenet_roundtrip(modules, synthetic):
    bi = modules["bi"]
    x, _ = synthetic
    R = bi.build_frenet_basis_3d(x, end_idx=10)
    v = (x[:, 10] - x[:, 9]).astype(np.float32)
    f = bi.to_frenet(v, R, origin=np.zeros((32, 3), dtype=np.float32))
    w = np.einsum("nij,nj->ni", R, f)
    err = float(np.abs(w - v).max())
    assert err < 1e-6, f"roundtrip max err = {err}"


def test_build_input_common(modules, synthetic):
    bi, bf = modules["bi"], modules["bf"]
    x, _ = synthetic
    common = bi.build_input_common(x, bf.f0_baseline)
    assert set(common.keys()) >= {"L1", "L2", "L4", "R_wfn", "origin"}
    assert common["L1"].shape == (32, 11, 9)
    assert common["L2"].shape == (32, 7, 3)
    assert common["L4"].shape == (32, 7, 2)
    assert common["R_wfn"].shape == (32, 3, 3)
    assert common["origin"].shape == (32, 3)
    for k in ("L1", "L2", "L4"):
        assert np.isfinite(common[k]).all(), f"{k} NaN/Inf"


def test_build_input_lgbm_extra(modules, synthetic):
    bi = modules["bi"]
    x, _ = synthetic
    extra = bi.build_input_lgbm_extra(x)
    assert extra.shape == (32, 36)
    assert np.isfinite(extra).all()


def test_build_soft_label(modules, synthetic):
    bi = modules["bi"]
    x, gt = synthetic
    R = bi.build_frenet_basis_3d(x, end_idx=10)
    origin = x[:, 10].astype(np.float32)
    q = bi.build_soft_label(gt, R, origin)
    assert q.shape == (32, 7)
    np.testing.assert_allclose(q.sum(axis=1), 1.0, atol=1e-5)


def test_dual_head_lgbm_forward(modules, synthetic):
    bi, dh, bf = modules["bi"], modules["dh"], modules["bf"]
    x, gt = synthetic
    common = bi.build_input_common(x, bf.f0_baseline)
    extra = bi.build_input_lgbm_extra(x, L1=common["L1"])
    N = x.shape[0]
    X_lgbm = np.concatenate([
        common["L1"].reshape(N, 99), common["L2"].reshape(N, 21),
        common["L4"].reshape(N, 14), extra,
    ], axis=1).astype(np.float32)
    q = bi.build_soft_label(gt, common["R_wfn"], common["origin"])
    res_true = bi.to_frenet(gt, common["R_wfn"], common["origin"])
    res_tg = res_true[:, None, :] - bi.ANCHORS_FRENET[None, :, :]

    model = dh.LgbmDualHead(n_estimators=20, lr=0.1)
    model.fit(X_lgbm, q, res_tg)
    probs, reg = model.predict(X_lgbm)
    assert probs.shape == (32, 7)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-5)
    assert reg.shape == (32, 7, 3)
    assert (np.abs(reg) <= 0.005 + 1e-6).all(), "reg_offset not bounded ±0.5cm"


def test_dual_head_gru_forward(modules, synthetic):
    import torch
    bi, dh, bf = modules["bi"], modules["dh"], modules["bf"]
    x, gt = synthetic
    common = bi.build_input_common(x, bf.f0_baseline)
    N = x.shape[0]
    seq = torch.from_numpy(common["L1"]).float()
    flat = torch.from_numpy(
        np.concatenate([common["L2"].reshape(N, 21), common["L4"].reshape(N, 14)], axis=1)
    ).float()
    R = torch.from_numpy(common["R_wfn"]).float()
    o = torch.from_numpy(common["origin"]).float()
    gru = dh.GRUDualHead()
    logits, reg = gru(seq, flat)
    assert logits.shape == (32, 7)
    assert reg.shape == (32, 7, 3)
    final = gru.predict_world(seq, flat, R, o)
    assert final.shape == (32, 3)
    assert torch.isfinite(final).all()


def test_losses_and_backward(modules, synthetic):
    import torch
    bi, dh, bf = modules["bi"], modules["dh"], modules["bf"]
    x, gt = synthetic
    common = bi.build_input_common(x, bf.f0_baseline)
    N = x.shape[0]
    seq = torch.from_numpy(common["L1"]).float()
    flat = torch.from_numpy(
        np.concatenate([common["L2"].reshape(N, 21), common["L4"].reshape(N, 14)], axis=1)
    ).float()
    R = torch.from_numpy(common["R_wfn"]).float()
    o = torch.from_numpy(common["origin"]).float()
    gru = dh.GRUDualHead()
    logits, _ = gru(seq, flat)
    final = gru.predict_world(seq, flat, R, o)
    q = torch.from_numpy(bi.build_soft_label(gt, common["R_wfn"], common["origin"])).float()
    gt_t = torch.from_numpy(gt.astype(np.float32))

    tau, ub = dh.tau_for_epoch(35)
    loss = dh.soft_ce_loss(logits, q) + dh.smooth_hit_loss(final, gt_t, tau=tau, use_boundary=ub)
    assert torch.isfinite(loss)
    loss.backward()


def test_tau_schedule(modules):
    dh = modules["dh"]
    assert dh.tau_for_epoch(0) == (0.003, False)
    assert dh.tau_for_epoch(14) == (0.003, False)
    assert dh.tau_for_epoch(15) == (0.001, False)
    assert dh.tau_for_epoch(29) == (0.001, False)
    assert dh.tau_for_epoch(30) == (0.0003, True)
    assert dh.tau_for_epoch(49) == (0.0003, True)


@pytest.mark.skipif(
    not (P21 / "baseline_carry.json").exists(),
    reason="run c6 G1 first to produce baseline_carry.json",
)
def test_g1_reproduce_carry():
    """G1 reproduce — baseline_carry.json 의 F0 metric 이 plan-020 spec 범위 안."""
    data = json.loads((P21 / "baseline_carry.json").read_text())
    f0 = data["f0_baseline"]
    assert 0.6315 <= f0["hit_1cm_5fold_concat"] <= 0.6325, f"G1 drift: {f0['hit_1cm_5fold_concat']}"
    assert 0.8028 <= f0["hit_1.5cm_5fold_concat"] <= 0.8038, f"G1 drift: {f0['hit_1.5cm_5fold_concat']}"
