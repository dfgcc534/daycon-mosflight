"""plan-021 §6.2 + §7.2 — LGBM + GRU dual head + soft_ce + smooth_hit losses.

Exports (§4.2):
  LgbmDualHead, GRUDualHead, soft_ce_loss, smooth_hit_loss, tau_for_epoch
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightgbm import LGBMClassifier, LGBMRegressor


R_HIT_MAIN = 0.010
R_HIT_LOOSE = 0.015
ANCHOR_RADIUS = 0.005


# ── losses (§4.2 + §7.3.1) ─────────────────────────────────────────────


def soft_ce_loss(logits: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """CE with soft (continuous) target distribution q.
    logits (B, K), q (B, K) ∈ Δ^{K-1}. returns scalar."""
    log_p = F.log_softmax(logits, dim=1)
    return -(q * log_p).sum(dim=1).mean()


def smooth_hit_loss(
    pred: torch.Tensor,
    gt: torch.Tensor,
    tau: float,
    use_boundary: bool = False,
) -> torch.Tensor:
    """§7.3.1. pred (B, 3), gt (B, 3) world. tau scalar. returns scalar.
    L = -mean_i [ w_i · (sigmoid((R_main - d)/τ) + 0.5·sigmoid((R_loose - d)/τ)) ].
    """
    d = (pred - gt).norm(dim=1)
    sh_main = torch.sigmoid((R_HIT_MAIN - d) / tau)
    sh_loose = torch.sigmoid((R_HIT_LOOSE - d) / tau)
    per_sample = sh_main + 0.5 * sh_loose
    if use_boundary:
        d_detach = d.detach()
        w = 1.0 + 5.0 * torch.exp(-(((R_HIT_MAIN - d_detach) / 0.001) ** 2))
        per_sample = w * per_sample
    return -per_sample.mean()


def tau_for_epoch(epoch: int) -> tuple[float, bool]:
    """§7.3 schedule: epoch [0,15)→τ=0.003, [15,30)→0.001, [30,50)→0.0003 + boundary."""
    if epoch < 15:
        return 0.003, False
    if epoch < 30:
        return 0.001, False
    return 0.0003, True


# ── LGBM dual head (§6.2) ──────────────────────────────────────────────


class LgbmDualHead:
    """7-class classifier (soft prob via hard-target + sample_weight 우회) + 21 regressor."""

    def __init__(self, n_estimators: int = 500, lr: float = 0.05, num_leaves: int = 63,
                 random_state: int = 20260518):
        self.n_estimators = n_estimators
        self.lr = lr
        self.num_leaves = num_leaves
        self.random_state = random_state
        self._make_models()

    def _make_models(self):
        self.clf = LGBMClassifier(
            n_estimators=self.n_estimators, learning_rate=self.lr, num_leaves=self.num_leaves,
            objective="multiclass", num_class=7, verbose=-1, random_state=self.random_state,
        )
        self.reg = [LGBMRegressor(
            n_estimators=self.n_estimators, learning_rate=self.lr, num_leaves=self.num_leaves,
            objective="regression", verbose=-1, random_state=self.random_state,
        ) for _ in range(21)]

    def fit(self, X: np.ndarray, soft_label_q: np.ndarray, residual_targets: np.ndarray):
        """X (N, D), q (N, 7), residual_targets (N, 7, 3) — Frenet clip ±0.005m (§6.3)."""
        hard_target = soft_label_q.argmax(axis=1)
        weights = soft_label_q.max(axis=1)

        # single-class fallback (§6.2 v1.2 — dummy sample weight=0, classes_ 7-class 확보)
        unique = set(np.unique(hard_target).tolist())
        missing = sorted(set(range(7)) - unique)
        if missing:
            X_dummy = np.zeros((len(missing), X.shape[1]), dtype=X.dtype)
            target_dummy = np.asarray(missing, dtype=hard_target.dtype)
            weight_dummy = np.zeros(len(missing), dtype=weights.dtype)
            X = np.concatenate([X, X_dummy], axis=0)
            hard_target = np.concatenate([hard_target, target_dummy])
            weights = np.concatenate([weights, weight_dummy])

        self.clf.fit(X, hard_target, sample_weight=weights)

        # regression: per-(anchor, axis) booster, target clip ±0.005
        rt_clipped = np.clip(residual_targets, -ANCHOR_RADIUS, ANCHOR_RADIUS)
        for k in range(7):
            for axis in range(3):
                self.reg[k * 3 + axis].fit(X, rt_clipped[:, k, axis])
        return self

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """returns (probs (N, 7), reg_offset (N, 7, 3) tanh-bounded ±0.005m)."""
        # 7-class zero-pad guard (§6.2)
        probs_raw = self.clf.predict_proba(X)
        probs = np.zeros((X.shape[0], 7), dtype=np.float32)
        for col_idx, class_label in enumerate(self.clf.classes_):
            probs[:, int(class_label)] = probs_raw[:, col_idx]

        reg_raw = np.stack([
            np.stack([self.reg[k * 3 + axis].predict(X) for axis in range(3)], axis=1)
            for k in range(7)
        ], axis=1).astype(np.float32)
        reg_offset = (np.tanh(reg_raw) * ANCHOR_RADIUS).astype(np.float32)
        return probs, reg_offset


# ── GRU dual head (§7.2) ───────────────────────────────────────────────


class GRUDualHead(nn.Module):
    def __init__(self, seq_dim: int = 9, hidden: int = 64, flat_dim: int = 35, dropout: float = 0.1):
        super().__init__()
        self.gru = nn.GRU(input_size=seq_dim, hidden_size=hidden, num_layers=1,
                          batch_first=True, bidirectional=False)
        self.dropout = nn.Dropout(dropout)
        self.clf_head = nn.Linear(hidden + flat_dim, 7)
        self.reg_head = nn.Linear(hidden + flat_dim, 21)
        # ANCHORS register_buffer (§7.3.0 MINOR — host→device 복사 회피)
        self.register_buffer(
            "ANCHORS",
            torch.tensor([
                (0.000, 0.000, 0.000),
                (+0.005, 0.000, 0.000),
                (-0.005, 0.000, 0.000),
                (0.000, +0.005, 0.000),
                (0.000, -0.005, 0.000),
                (0.000, 0.000, +0.005),
                (0.000, 0.000, -0.005),
            ], dtype=torch.float32),
        )

    def forward(self, seq: torch.Tensor, flat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # seq (B, 11, 9), flat (B, 35)
        out, _ = self.gru(seq)
        seq_hidden = self.dropout(out[:, -1, :])                       # (B, hidden)
        combined = torch.cat([seq_hidden, flat], dim=-1)               # (B, hidden+flat_dim)
        logits = self.clf_head(combined)                                # (B, 7)
        reg_raw = self.reg_head(combined).view(-1, 7, 3)
        reg_offset = torch.tanh(reg_raw) * ANCHOR_RADIUS                # (B, 7, 3) bounded ±0.5cm
        return logits, reg_offset

    def predict_world(
        self,
        seq: torch.Tensor,
        flat: torch.Tensor,
        R_wfn: torch.Tensor,
        origin: torch.Tensor,
    ) -> torch.Tensor:
        """§7.3.0 — Frenet → world final pred (training loop helper)."""
        logits, reg_offset = self.forward(seq, flat)
        probs = torch.softmax(logits, dim=1)                            # (B, 7)
        combined = self.ANCHORS[None, :, :] + reg_offset                # (B, 7, 3) Frenet
        final_frenet = (probs[:, :, None] * combined).sum(dim=1)        # (B, 3)
        # R_wfn columns=[t̂,n̂,b̂] → world = R_wfn @ frenet (no transpose)
        final_world = torch.einsum("nij,nj->ni", R_wfn, final_frenet) + origin
        return final_world
