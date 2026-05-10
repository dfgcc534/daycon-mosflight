"""Wingbeat FFT feature (R004 ablation).

Per plan-003 §4.3 + caveat #4 (label "wing-beat" = 11pt 저주파 oscillation
패턴; 실제 wingbeat 는 11pt 의 Nyquist 위 → aliasing 으로 entry 부재).
"""
from __future__ import annotations

import numpy as np


def wingbeat_fft(X_relative: np.ndarray, n_bins: int = 3) -> np.ndarray:
    """X_relative: (n, T, 3) → (n, T, 3 * n_bins).

    각 axis 의 T-길이 시퀀스 rfft → magnitude 의 첫 n_bins 성분
    (DC + harmonics). sequence-level 요약을 timestep 차원에 broadcast.
    """
    if X_relative.ndim != 3:
        raise ValueError(f"X_relative must be (n, T, d); got {X_relative.shape}")
    n, T, d = X_relative.shape
    coefs = np.fft.rfft(X_relative, axis=1)  # (n, T//2 + 1, d)
    if n_bins > coefs.shape[1]:
        raise ValueError(
            f"n_bins {n_bins} > available rfft bins {coefs.shape[1]} (T={T})"
        )
    mags = np.abs(coefs)[:, :n_bins, :]  # (n, n_bins, d)
    feats = mags.transpose(0, 2, 1).reshape(n, d * n_bins)
    return np.broadcast_to(feats[:, None, :], (n, T, d * n_bins)).copy()
