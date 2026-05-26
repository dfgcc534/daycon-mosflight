"""plan-d-001 c3 — SimpleNeuralODEModel (노트북 cell 8 그대로 이식).

pos+vel 6D 상태계 위 학습 가속도장 + RK4 단일스텝 적분 + 학습 damping + local/global bias.
구조·init 상수는 notes/[LB_0.6+] Neural ODE 기반 예측모델.ipynb cell 8 verbatim.
"""
from __future__ import annotations

import torch
import torch.nn as nn

DT_STEP = 0.04  # 관측 step 간격 (40ms) — init_vel = diffs_local[:,-1]/DT_STEP


class ResBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(dim, dim),
        )
        self.ln = nn.LayerNorm(dim)

    def forward(self, x):
        return self.ln(x + self.net(x))


class SimpleAccelerationField(nn.Module):
    def __init__(self, latent_dim: int = 64):
        super().__init__()
        # 입력: pos(3) + vel(3) + latent(latent_dim) + θ(1) + speed(1) = 8 + latent_dim
        self.net = nn.Sequential(
            nn.Linear(3 + 3 + latent_dim + 2, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            ResBlock(64),
            nn.Linear(64, 3),
        )

    def forward(self, pos, vel, latent, theta, speed):
        if theta.dim() == 1:
            theta = theta.unsqueeze(-1)
        if speed.dim() == 1:
            speed = speed.unsqueeze(-1)
        inputs = torch.cat([pos, vel, latent, theta, speed], dim=-1)
        return self.net(inputs)


class SimpleNeuralODEModel(nn.Module):
    def __init__(self, input_dim: int = 24, latent_dim: int = 64):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
            ResBlock(latent_dim),
        )
        self.accel_field = SimpleAccelerationField(latent_dim=latent_dim)

        # 물리 기반 학습 파라미터
        self.learned_damping = nn.Parameter(torch.tensor([0.1, 0.1, 0.1]))  # vel 3축 elementwise
        self.local_bias = nn.Parameter(torch.zeros(3))
        self.global_bias = nn.Parameter(torch.zeros(3))

        self.dt_physical = 0.08  # 80ms 단일 적분
        self._last_accels: list[torch.Tensor] = []

    def _ode_derivative(self, pos, vel, latent, theta, speed):
        accel = self.accel_field(pos, vel, latent, theta, speed)
        dpos = vel
        dvel = -self.learned_damping * vel + accel
        return dpos, dvel, accel

    def _rk4_single_step(self, init_pos, init_vel, latent, theta, speed):
        dt = self.dt_physical

        dp1, dv1, a1 = self._ode_derivative(init_pos, init_vel, latent, theta, speed)
        pos_k2 = init_pos + 0.5 * dt * dp1
        vel_k2 = init_vel + 0.5 * dt * dv1

        dp2, dv2, a2 = self._ode_derivative(pos_k2, vel_k2, latent, theta, speed)
        pos_k3 = init_pos + 0.5 * dt * dp2
        vel_k3 = init_vel + 0.5 * dt * dv2

        dp3, dv3, a3 = self._ode_derivative(pos_k3, vel_k3, latent, theta, speed)
        pos_k4 = init_pos + dt * dp3
        vel_k4 = init_vel + dt * dv3

        dp4, dv4, a4 = self._ode_derivative(pos_k4, vel_k4, latent, theta, speed)
        final_pos = init_pos + (dt / 6.0) * (dp1 + 2.0 * dp2 + 2.0 * dp3 + dp4)
        final_vel = init_vel + (dt / 6.0) * (dv1 + 2.0 * dv2 + 2.0 * dv3 + dv4)

        self._last_accels = [a1, a2, a3, a4]
        return final_pos, final_vel

    def forward(self, features, diffs, p_last, theta, speed, R):
        latent = self.backbone(features)
        diffs_local = torch.matmul(diffs, R)  # world→local (= Rᵀ·diffs)

        init_pos = torch.zeros_like(p_last)
        init_vel = diffs_local[:, -1] / DT_STEP

        pos, vel = self._rk4_single_step(init_pos, init_vel, latent, theta, speed)

        pred_local = pos + self.local_bias
        pred_global = p_last + torch.einsum("nij,nj->ni", R, pred_local) + self.global_bias  # local→world
        return pred_global
