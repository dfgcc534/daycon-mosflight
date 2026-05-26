"""plan-d-001 c1 — data diffs/p_last + local yaw frame (θ, R, speed).

R = local→world yaw 회전행렬 (rotation by +θ about z).
global→local = Rᵀ = `vec @ R` (rotate_xy(·,θ) 와 항등 — frame_self_check 로 assert).
"""
from __future__ import annotations

import numpy as np
import torch

DT_STEP = 0.04  # 40ms


def diffs_of(X: torch.Tensor) -> torch.Tensor:
    """X (N,T,3) → step 변위 (N,T-1,3)."""
    return X[:, 1:] - X[:, :-1]


def p_last_of(X: torch.Tensor) -> torch.Tensor:
    """X (N,T,3) → 마지막 관측 위치 (N,3)."""
    return X[:, -1]


def theta_of(diffs: torch.Tensor) -> torch.Tensor:
    """마지막 step 속도 v_last = diffs[:,-1]/DT 의 yaw = atan2(v_y, v_x). (N,)."""
    v_last = diffs[:, -1] / DT_STEP
    return torch.atan2(v_last[:, 1], v_last[:, 0])


def speed_of(diffs: torch.Tensor) -> torch.Tensor:
    """기준 속력 = ||v_last|| (N,)."""
    v_last = diffs[:, -1] / DT_STEP
    return torch.norm(v_last, dim=1)


def rot_matrix(theta: torch.Tensor) -> torch.Tensor:
    """θ (N,) → R (N,3,3) = local→world (+θ about z). `vec@R` = Rᵀ·vec = world→local."""
    n = theta.shape[0]
    c, s = torch.cos(theta), torch.sin(theta)
    R = torch.zeros((n, 3, 3), dtype=theta.dtype, device=theta.device)
    R[:, 0, 0] = c
    R[:, 0, 1] = -s
    R[:, 1, 0] = s
    R[:, 1, 1] = c
    R[:, 2, 2] = 1.0
    return R


def frame_self_check() -> None:
    """`diffs@R` (world→local) == rotate_xy(·,θ) 항등성 + R·Rᵀ=I assert (§4.1)."""
    torch.manual_seed(0)
    v = torch.randn(5, 3, dtype=torch.float64)
    theta = torch.rand(5, dtype=torch.float64) * 6.28 - 3.14
    R = rot_matrix(theta)
    # world→local via vec@R
    local = torch.einsum("nj,njk->nk", v, R)  # = Rᵀ·v
    c, s = torch.cos(theta), torch.sin(theta)
    expect = torch.stack([v[:, 0] * c + v[:, 1] * s, -v[:, 0] * s + v[:, 1] * c, v[:, 2]], dim=-1)
    assert torch.allclose(local, expect, atol=1e-10), "diffs@R != rotate_xy(·,θ)"
    # local→world round-trip
    back = torch.einsum("nij,nj->ni", R, local)
    assert torch.allclose(back, v, atol=1e-10), "R·(Rᵀ·v) != v"


if __name__ == "__main__":
    frame_self_check()
    print("frame_self_check OK")
