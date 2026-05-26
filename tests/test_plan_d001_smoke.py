"""plan-d-001 c5 smoke — import + frame self-check + 1-batch forward/loss finite."""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis" / "plan-d-001"))


def test_frame_self_check():
    import frame
    frame.frame_self_check()  # diffs@R == rotate_xy(·,θ), R·Rᵀ=I


def test_extract_features_shapes():
    from features import extract_features
    X = torch.randn(8, 11, 3)
    ft, df, pl, th, _, _, _, R, sp, m, s = extract_features(X)
    assert ft.shape == (8, 24)
    assert df.shape == (8, 10, 3)
    assert pl.shape == (8, 3)
    assert th.shape == (8,)
    assert R.shape == (8, 3, 3)
    assert sp.shape == (8,)
    assert m.shape == (24,) and s.shape == (24,)
    assert torch.isfinite(ft).all()


def test_forward_loss_finite():
    from features import extract_features
    from losses import combined_loss
    from model import SimpleNeuralODEModel
    X = torch.randn(8, 11, 3)
    y = torch.randn(8, 3)
    ft, df, pl, th, _, _, _, R, sp, _, _ = extract_features(X)
    model = SimpleNeuralODEModel(input_dim=24, latent_dim=64)
    pred = model(ft, df, pl, th, sp, R)
    assert pred.shape == (8, 3)
    loss = combined_loss(pred, y, model._last_accels)
    assert torch.isfinite(loss)
    loss.backward()  # gradient path finite
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
