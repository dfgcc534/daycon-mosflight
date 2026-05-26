"""plan-a-002 c4 — smoke tests (§5).

import + kalman internals shape/zero-pad + **leakage assert** (internals 가 t_pred 미참조)
+ cv_ca + features_ext dim/회전 + 15ch model finite/gradient.
무거운 학습은 run_oof --gate smoke 가 별도 담당 (CI 비포함).
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import torch

_PA1 = Path(__file__).resolve().parent.parent / "analysis" / "plan-a-001"
_PA2 = Path(__file__).resolve().parent.parent / "analysis" / "plan-a-002"


def _load(base, name):
    spec = importlib.util.spec_from_file_location(f"m_{name}", base / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_kf = _load(_PA2, "kalman_features")
_fe = _load(_PA2, "features_ext")
_model = _load(_PA1, "model")
_yaw = _load(_PA1, "yaw")

_rng = np.random.default_rng(20260526)


def _fake_X(n=200, t=11):
    """관측 시계열 모사 (등속 + 곡률 + noise)."""
    base = np.cumsum(_rng.standard_normal((n, t, 3)) * 0.01, axis=1)
    return base.astype(np.float64)


def test_import():
    assert hasattr(_kf, "kalman_with_internals")
    assert hasattr(_kf, "cv_ca_disagreement")
    assert hasattr(_fe, "build_seq_ext")
    assert hasattr(_fe, "build_scalar_ext")


def test_internals_shape_zeropad():
    X = _fake_X()
    pred, innov, fv = _kf.kalman_with_internals(X)
    assert pred.shape == (X.shape[0], 3)
    assert innov.shape == (X.shape[0], 11, 3)
    assert fv.shape == (X.shape[0], 11, 3)
    assert np.isfinite(pred).all() and np.isfinite(innov).all() and np.isfinite(fv).all()
    # t=0 zero-pad
    assert (innov[:, 0] == 0).all()
    assert (fv[:, 0] == 0).all()
    # pred == kalman_predict (canonical)
    assert np.allclose(pred, _kf.kalman_predict(X), atol=1e-12)


def test_leakage_internals_tpred_invariant():
    """leakage assert: innov_seq·filtered_v 는 관측창만의 함수 → t_pred 바꿔도 불변."""
    X = _fake_X()
    _, in1, fv1 = _kf.kalman_with_internals(X, t_pred=0.08)
    p2, in2, fv2 = _kf.kalman_with_internals(X, t_pred=0.40)
    assert np.allclose(in1, in2), "innov_seq 가 t_pred 에 의존 (leakage!)"
    assert np.allclose(fv1, fv2), "filtered_v 가 t_pred 에 의존 (leakage!)"
    # pred 는 t_pred 변하면 달라져야 (외삽 sanity)
    assert not np.allclose(_kf.kalman_predict(X, t_pred=0.08), p2)


def test_cv_ca():
    X = _fake_X()
    cc = _kf.cv_ca_disagreement(X)
    assert cc.shape == (X.shape[0], 3)
    assert np.isfinite(cc).all()
    assert np.abs(cc).mean() > 0  # CA≠CV (비자명 불일치)


def test_seq_ext_dims_and_z_preserve():
    X = _fake_X()
    _, innov, fv = _kf.kalman_with_internals(X)
    theta = _yaw.yaw_from_last_step(X)
    s9, n9 = _fe.build_seq_ext(X, input_yaw=False)
    s12, _ = _fe.build_seq_ext(X, innov_arr=innov, theta=theta, input_yaw=True)
    s15, n15 = _fe.build_seq_ext(X, innov_arr=innov, filtered_v_arr=fv, theta=theta, input_yaw=True)
    assert s9.shape[2] == 9 and s12.shape[2] == 12 and s15.shape[2] == 15
    assert len(n15) == 15
    # z 보존 (회전후 z == raw z)
    s15_raw, _ = _fe.build_seq_ext(X, innov_arr=innov, filtered_v_arr=fv, input_yaw=False)
    zcols = [2, 5, 8, 11, 14]
    assert np.allclose(s15[:, :, zcols], s15_raw[:, :, zcols])


def test_scalar_ext_norm_invariant():
    X = _fake_X()
    cc = _kf.cv_ca_disagreement(X)
    theta = _yaw.yaw_from_last_step(X)
    import importlib.util as _iu  # noqa: F401
    _feat = _fe._feat
    noise = _feat.compute_noise(X, with_loo=False)
    sc40, n40, _ = _fe.build_scalar_ext(X, noise["poly2"], noise["savgol"], noise["loo"])
    sc44_on, n44, rc = _fe.build_scalar_ext(
        X, noise["poly2"], noise["savgol"], noise["loo"], cv_ca_arr=cc, theta=theta, input_yaw=True)
    sc44_off, _, _ = _fe.build_scalar_ext(
        X, noise["poly2"], noise["savgol"], noise["loo"], cv_ca_arr=cc, theta=theta, input_yaw=False)
    assert sc40.shape[1] == 40 and sc44_on.shape[1] == 44
    # cvca_norm (마지막 컬럼) 회전불변
    assert np.allclose(sc44_on[:, -1], sc44_off[:, -1])
    assert sum(1 for v in rc.values() if v == "rotate") == 3


def test_model_15ch_finite_grad():
    X = _fake_X(64)
    _, innov, fv = _kf.kalman_with_internals(X)
    theta = _yaw.yaw_from_last_step(X)
    seq, _ = _fe.build_seq_ext(X, innov_arr=innov, filtered_v_arr=fv, theta=theta, input_yaw=True)
    net = _model.GRUModelMultiAux(n_channels=15, scal_dim=44, aux_dims=[3, 3])
    sq = torch.as_tensor(seq)
    sc = torch.zeros(64, 44)
    om, ax = net(sq, sc)
    assert om.shape == (64, 3) and torch.isfinite(om).all()
    om.abs().mean().backward()
    g = sum(p.grad.abs().sum().item() for p in net.parameters() if p.grad is not None)
    assert np.isfinite(g) and g > 0
