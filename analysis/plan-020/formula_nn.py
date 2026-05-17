"""plan-020 §7 — 3 NN coef regressors (N1 MLP / N2 TCN / N5 MoE) + 학습 loop.

Module signatures:
    N01_MLPCoef.forward(seq_feats_3)        : (B, 3, 9D) → coef (B, 3)
    N02_TCNCoef.forward(seq_feats_11)       : (B, 11, 9D) → coef (B, 3)
    N05_MoE.forward(seq_feats_11, expert_preds): (B, 11, 9D) + (B, K, 3) → pred (B, 3)

run_oof_nn dispatch:
    N1 : pred = f0_form_torch(seq_feats_3, model(seq_feats_3))
    N2 : pred = f0_form_torch(seq_feats_11[:, -3:, :], model(seq_feats_11))
    N5 : pred = model(seq_feats_11, expert_preds_batch)
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

R_HIT_MAIN = 0.01
R_HIT_LOOSE = 0.015
LOSS_WEIGHT_LOOSE = 0.5


# ── modules ────────────────────────────────────────────────────────────


class N01_MLPCoef(nn.Module):
    """§7.1 N1 — last 3 timesteps × 9D = 27D MLP → (d1, par, perp) residual."""

    def __init__(self, seq_dim: int = 9, hidden: int = 64, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3 * seq_dim, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 3),
        )

    def forward(self, seq_feats: torch.Tensor) -> torch.Tensor:
        # seq_feats: (B, 3, 9D)
        x = seq_feats.flatten(1)
        delta = self.net(x)
        return torch.stack(
            [1.98 + delta[:, 0], 1.20 + delta[:, 1], -0.20 + delta[:, 2]],
            dim=1,
        )


class N02_TCNCoef(nn.Module):
    """§7.1 N2 — dilated TCN [1, 2, 4] on (B, 11, 9D) → (d1, par, perp) residual."""

    def __init__(self, seq_dim: int = 9, hidden: int = 32, dropout: float = 0.1):
        super().__init__()
        self.tcn = nn.Sequential(
            nn.Conv1d(seq_dim, hidden, kernel_size=3, padding=1, dilation=1),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=2, dilation=2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=4, dilation=4),
            nn.SiLU(),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(hidden, 3),
        )

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        # seq: (B, 11, 9D) → (B, 9, 11) for Conv1d
        h = self.tcn(seq.transpose(1, 2))
        delta = self.head(h)
        return torch.stack(
            [1.98 + delta[:, 0], 1.20 + delta[:, 1], -0.20 + delta[:, 2]],
            dim=1,
        )


class N05_MoE(nn.Module):
    """§7.1 N5 — gating NN over K=4 frozen experts (F0, helix, hermite, ctra)."""

    K = 4

    def __init__(self, seq_dim: int = 9, hidden: int = 32, dropout: float = 0.1):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv1d(seq_dim, hidden, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(hidden, self.K),
        )

    def forward(self, seq: torch.Tensor, expert_preds: torch.Tensor) -> torch.Tensor:
        # seq: (B, 11, 9D), expert_preds: (B, K, 3)
        logits = self.gate(seq.transpose(1, 2))
        weights = torch.softmax(logits, dim=1)
        return (weights[:, :, None] * expert_preds).sum(dim=1)


# ── loss ───────────────────────────────────────────────────────────────


def smooth_hit_loss(
    pred: torch.Tensor,
    gt: torch.Tensor,
    tau: float,
    use_boundary: bool = False,
) -> torch.Tensor:
    """L = − mean_i [ w_i · (smooth_hit(R=0.01) + 0.5 · smooth_hit(R=0.015)) ].
    smooth_hit(pred, gt; R, τ) = sigmoid((R − ‖pred − gt‖₂) / τ).
    boundary weight (use_boundary=True): w_i = 1 + 5·exp(−((R − d_i.detach())/0.001)²) at R=0.01.
    """
    d = (pred - gt).norm(dim=1)
    sh_main = torch.sigmoid((R_HIT_MAIN - d) / tau)
    sh_loose = torch.sigmoid((R_HIT_LOOSE - d) / tau)
    per_sample = sh_main + LOSS_WEIGHT_LOOSE * sh_loose
    if use_boundary:
        d_detach = d.detach()
        w = 1.0 + 5.0 * torch.exp(-(((R_HIT_MAIN - d_detach) / 0.001) ** 2))
        per_sample = w * per_sample
    return -per_sample.mean()


def tau_for_epoch(epoch: int) -> tuple[float, bool]:
    """Returns (tau, use_boundary). §7.2 annealed schedule.
    0-15: τ=0.003, 16-30: τ=0.001, 31-50: τ=0.0003 + boundary weighting."""
    if epoch < 15:
        return 0.003, False
    if epoch < 30:
        return 0.001, False
    return 0.0003, True


# ── hit metric (eval) ──────────────────────────────────────────────────


def hit_at_R(pred: torch.Tensor | np.ndarray, gt: torch.Tensor | np.ndarray, R: float) -> float:
    if isinstance(pred, torch.Tensor):
        pred_np = pred.detach().cpu().numpy()
    else:
        pred_np = pred
    if isinstance(gt, torch.Tensor):
        gt_np = gt.detach().cpu().numpy()
    else:
        gt_np = gt
    d = np.linalg.norm(pred_np - gt_np, axis=1)
    return float((d <= R).mean())


# ── single-fold training ──────────────────────────────────────────────


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_nn_fold(
    model_factory: Callable[[], nn.Module],
    pred_fn: Callable[[nn.Module, torch.Tensor, torch.Tensor | None], torch.Tensor],
    train_seq: torch.Tensor,
    train_y: torch.Tensor,
    train_expert_preds: torch.Tensor | None = None,
    epochs: int = 50,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    seed: int = 0,
    device: str = "cuda:1",
    early_stop_patience: int = 10,
) -> tuple[nn.Module, float, list[float]]:
    """Train 1 fold. pred_fn(model, batch_seq, batch_expert_preds_or_None) → pred (B, 3).

    Early stop: train-hit plateau patience (val 의존 금지, multi-seed selection bias 회피).
    Returns: (trained_model, best_train_hit, train_hit_history).
    """
    _set_seed(seed)
    if torch.cuda.is_available() and device.startswith("cuda"):
        dev = torch.device(device)
    else:
        dev = torch.device("cpu")

    model = model_factory().to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_seq = train_seq.to(dev)
    train_y = train_y.to(dev)
    if train_expert_preds is not None:
        train_expert_preds = train_expert_preds.to(dev)

    N = train_seq.shape[0]
    train_hit_history: list[float] = []
    best_train_hit = -1.0
    plateau_counter = 0
    best_state = None

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(N, device=dev)
        for i in range(0, N, batch_size):
            idx = perm[i : i + batch_size]
            batch_seq = train_seq[idx]
            batch_y = train_y[idx]
            batch_expert = train_expert_preds[idx] if train_expert_preds is not None else None
            pred = pred_fn(model, batch_seq, batch_expert)
            tau, use_boundary = tau_for_epoch(epoch)
            loss = smooth_hit_loss(pred, batch_y, tau, use_boundary)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # eval on train (no val — early stop on train_hit plateau per §7.2)
        model.eval()
        with torch.no_grad():
            train_preds = []
            for i in range(0, N, batch_size):
                bs = train_seq[i : i + batch_size]
                be = train_expert_preds[i : i + batch_size] if train_expert_preds is not None else None
                train_preds.append(pred_fn(model, bs, be))
            train_pred = torch.cat(train_preds, dim=0)
        train_hit = hit_at_R(train_pred, train_y, R_HIT_MAIN)
        train_hit_history.append(train_hit)

        if train_hit > best_train_hit + 1e-5:
            best_train_hit = train_hit
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            plateau_counter = 0
        else:
            plateau_counter += 1
            if plateau_counter >= early_stop_patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_train_hit, train_hit_history


# ── pred_fn factories (run_oof_nn dispatch helpers) ────────────────────


def make_pred_fn_n1(f0_form_torch: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]) -> Callable:
    """N1: seq is (B, 3, 9D), apply f0_form_torch directly."""
    def _fn(model: nn.Module, seq: torch.Tensor, _expert_preds_unused: torch.Tensor | None) -> torch.Tensor:
        coef = model(seq)
        return f0_form_torch(seq, coef)
    return _fn


def make_pred_fn_n2(f0_form_torch: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]) -> Callable:
    """N2: seq is (B, 11, 9D), slice last 3 for f0_form_torch."""
    def _fn(model: nn.Module, seq: torch.Tensor, _expert_preds_unused: torch.Tensor | None) -> torch.Tensor:
        coef = model(seq)
        seq_3 = seq[:, -3:, :]
        return f0_form_torch(seq_3, coef)
    return _fn


def make_pred_fn_n5() -> Callable:
    """N5: pred = model(seq_11, expert_preds_batch). f0_form_torch 미경유."""
    def _fn(model: nn.Module, seq: torch.Tensor, expert_preds: torch.Tensor | None) -> torch.Tensor:
        assert expert_preds is not None, "N5 requires expert_preds"
        return model(seq, expert_preds)
    return _fn
