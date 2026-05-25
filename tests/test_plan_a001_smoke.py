"""plan-a-001 c5 — smoke tests (§5).

import + builder shape + model finite/gradient + yaw round-trip + KR002 회전 검증.
무거운 학습은 run_oof --gate smoke 가 별도 담당 (CI 비포함).
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import torch

_PA = Path(__file__).resolve().parent.parent / "analysis" / "plan-a-001"


def _load(name):
    spec = importlib.util.spec_from_file_location(f"pa_{name}", _PA / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_kalman = _load("kalman")
_yaw = _load("yaw")
_feat = _load("features")
_model = _load("model")
_losses = _load("losses")

_rng = np.random.default_rng(20260526)
_t = np.arange(-400, 1, 40) / 1000.0
_X = np.zeros((64, 11, 3))
for _i in range(64):
    _v = _rng.standard_normal(3) * 2
    _X[_i] = _rng.standard_normal(3)[None, :] + np.outer(_t, _v) + _rng.standard_normal((11, 3)) * 3e-4


def test_kalman_shape_finite():
    pred = _kalman.kalman_predict(_X)
    assert pred.shape == (64, 3)
    assert np.isfinite(pred).all()
    # main vs alt sigma 구분
    pa = _kalman.kalman_predict(_X, sigma_obs=_kalman.SIGMA_OBS_MAIN)
    pw = _kalman.kalman_predict(_X, sigma_obs=_kalman.SIGMA_OBS_ALT)
    assert not np.allclose(pa, pw)


def test_yaw_identity():
    _yaw.assert_rotation_identity()


def test_feature_shapes_finite():
    seq = _feat.build_seq_t3(_X)
    assert seq.shape == (64, 11, 9) and np.isfinite(seq).all()
    n = _feat.compute_noise(_X, with_loo=True)
    scal, names = _feat.build_scalar_40d(_X, n["poly2"], n["savgol"], n["loo"])
    assert scal.shape == (64, 40) and len(names) == 40 and np.isfinite(scal).all()


def test_model_forward_backward():
    net = _model.GRUModelMultiAux(aux_dims=[3, 3], aux_clips=[None, None])
    seq = torch.randn(8, 11, 9)
    scal = torch.randn(8, 40)
    om, ax = net(seq, scal)
    assert om.shape == (8, 3) and len(ax) == 2
    assert float(om.abs().max().detach()) <= 0.02 + 1e-6  # main clamp ±2cm
    true = torch.randn(8, 3) * 0.01
    loss = (_losses.loss_combo(om, true)
            + _losses.LAMBDA_AUX * _losses.loss_aux_euclid(ax[0], true)
            + _losses.LAMBDA_AUX * _losses.loss_aux_euclid(ax[1], true))
    assert torch.isfinite(loss)
    loss.backward()
    g = sum(float(p.grad.abs().sum()) for p in net.parameters() if p.grad is not None)
    assert np.isfinite(g) and g > 0


def test_softhit_bounded():
    pred = torch.randn(16, 3) * 0.01
    true = torch.randn(16, 3) * 0.01
    sh = float(_losses.loss_softhit(pred, true))
    assert 0.0 <= sh <= 1.0


def test_kr002_seq_rotation():
    """KR002: rel/v/a x,y 회전 + z 보존 + last-step v_y≈0 (yaw 정렬)."""
    run = _load("run_oof")
    theta = _yaw.yaw_from_last_step(_X)
    seq = _feat.build_seq_t3(_X)
    seq_rot = run.rotate_seq_input(seq, theta)
    assert not np.allclose(seq, seq_rot)                      # 입력 변함
    assert np.allclose(seq[:, :, [2, 5, 8]], seq_rot[:, :, [2, 5, 8]])  # z 보존
    assert np.abs(seq_rot[:, -1, 4]).max() < 1e-6             # last-step v_y ≈ 0
