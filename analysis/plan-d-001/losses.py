"""plan-d-001 c3 — combined loss (노트북 cell 10 상수 그대로).

loss = soft_hit + 126.309·huber(δ=0.001026) + 1e-4·accel_reg
좌표 단위 = m (프로젝트 데이터). 상수 전부 cell 10 verbatim.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

ACCEL_REG_WEIGHT = 1e-4
HUBER_WEIGHT = 126.309
HUBER_DELTA = 0.001026
SOFTHIT_THR = 0.011178
SOFTHIT_SHARP = 332.259


def soft_hit_loss(pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    d = torch.norm(pred - y, dim=1)
    return (1 - torch.sigmoid(-(d - SOFTHIT_THR) * SOFTHIT_SHARP)).mean()


def accel_reg(last_accels: list[torch.Tensor]) -> torch.Tensor:
    # per-sample 축-합(||a||²) → batch-mean → RK4 4단계 평균 (cell 10 그대로)
    if not last_accels:
        return torch.zeros((), device="cpu")
    return sum(a.pow(2).sum(-1).mean() for a in last_accels) / len(last_accels)


def combined_loss(pred: torch.Tensor, y: torch.Tensor, last_accels: list[torch.Tensor]) -> torch.Tensor:
    sh = soft_hit_loss(pred, y)
    huber = F.huber_loss(pred, y, delta=HUBER_DELTA)
    reg = accel_reg(last_accels)
    return sh + HUBER_WEIGHT * huber + ACCEL_REG_WEIGHT * reg
