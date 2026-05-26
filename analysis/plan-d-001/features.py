"""plan-d-001 c2 — extract_features 재구현 (§4.2 24D dim→식 정본 표).

외부 model.utils.extract_features 부재 → 노트북 markdown 의미 매칭 재구성 (byte-exact 아님).
반환 11-tuple: (ft, df, p_last, theta, _, _, _, R, speed, mean_stats, std_stats).
사용 slot 0,1,2,3,7,8,9,10 (4,5,6 = None). 전부 torch.float32, X 의 device.
"""
from __future__ import annotations

import torch

from frame import DT_STEP, diffs_of, p_last_of, rot_matrix, speed_of, theta_of

EPS = 1e-8  # std 0-division 가드


def _build_24d(X: torch.Tensor, diffs: torch.Tensor, p_last: torch.Tensor,
               theta: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    """§4.2 표대로 24D ft (N,24) 산출. 모든 std = ddof=0."""
    diffs_local = torch.matmul(diffs, R)          # (N,T-1,3) world→local
    vL = diffs_local / DT_STEP                     # (N,T-1,3) per-step velocity (local)
    aL = vL[:, 1:] - vL[:, :-1]                    # (N,T-2,3)
    jL = aL[:, 1:] - aL[:, :-1]                    # (N,T-3,3)
    XL = torch.matmul(X - p_last.unsqueeze(1), R)  # (N,T,3) positions in local frame

    cols = []
    cols.append(vL[:, -1])                                  # d0-2  last velocity
    cols.append(torch.norm(vL[:, -1], dim=1, keepdim=True)) # d3    ||v||
    cols.append(aL[:, -1])                                  # d4-6  last accel
    cols.append(torch.norm(aL[:, -1], dim=1, keepdim=True)) # d7    ||a||
    cols.append(jL[:, -1])                                  # d8-10 last jerk
    cols.append(torch.norm(jL[:, -1], dim=1, keepdim=True)) # d11   ||j||
    cols.append(torch.stack([torch.sin(theta), torch.cos(theta)], dim=-1))  # d12-13
    cols.append(torch.norm(vL, dim=2).mean(dim=1, keepdim=True))            # d14 mean speed
    cols.append(diffs_local.std(dim=1, unbiased=False))     # d15-17 step 변위 축별 std (ddof=0)
    cols.append(vL.mean(dim=1))                             # d18-20 mean velocity vector
    # d21 turn-cos ⟨vL[-2],vL[-1]⟩/(||·||·||·||), 분모 0→1.0
    v_a, v_b = vL[:, -2], vL[:, -1]
    denom = torch.norm(v_a, dim=1) * torch.norm(v_b, dim=1)
    turn = (v_a * v_b).sum(dim=1) / denom
    turn = torch.where(denom > EPS, turn, torch.ones_like(turn))
    cols.append(turn.unsqueeze(-1))                         # d21
    cols.append((XL[:, :, 2].amax(dim=1) - XL[:, :, 2].amin(dim=1)).unsqueeze(-1))  # d22 z-range
    cols.append(torch.norm(diffs, dim=2).sum(dim=1, keepdim=True))                  # d23 path-length

    ft = torch.cat(cols, dim=1)
    assert ft.shape[1] == 24, f"ft dim {ft.shape[1]} != 24"
    return ft


def extract_features(X: torch.Tensor, mean=None, std=None):
    """X (N,11,3) float32 → 11-tuple. mean/std None 시 ft 통계 계산 모드."""
    X = X.float()
    diffs = diffs_of(X)
    p_last = p_last_of(X)
    theta = theta_of(diffs)
    speed = speed_of(diffs)
    R = rot_matrix(theta)

    ft_raw = _build_24d(X, diffs, p_last, theta, R)
    if mean is None or std is None:
        mean = ft_raw.mean(dim=0)
        std = ft_raw.std(dim=0, unbiased=False)  # ddof=0
    ft = (ft_raw - mean) / (std + EPS)
    return (ft, diffs, p_last, theta, None, None, None, R, speed, mean, std)


# §4.2 dim→식 표 (results_node001.json 박제용)
DIM_TABLE = {
    "0-2": "vL[-1] (last velocity, local)", "3": "||vL[-1]||",
    "4-6": "aL[-1]", "7": "||aL[-1]||", "8-10": "jL[-1]", "11": "||jL[-1]||",
    "12-13": "(sinθ, cosθ)", "14": "mean_t ||vL[t]||",
    "15-17": "std_t(diffs_local) per-axis (ddof=0)", "18-20": "mean_t vL[t]",
    "21": "turn-cos <vL[-2],vL[-1]>/(||·||·||·||), denom 0->1.0",
    "22": "z-range max-min XL[:,:,2]", "23": "path-length sum_t ||diffs[t]||",
}
