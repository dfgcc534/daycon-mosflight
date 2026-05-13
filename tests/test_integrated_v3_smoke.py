"""plan-013 c2 §4.2 — smoke test for src/pb_0_6822/integrated_v3.py."""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.pb_0_6822.integrated_v3 import (
    STEP4_BEST_BASIS,
    InICCorrectorWrapper,
    InICEmbedder,
    Step4CoeffMLP,
    check_in_ic_frozen,
    load_25_cand_set,
)

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def r001_ckpt_present() -> bool:
    return (REPO / InICEmbedder.DEFAULT_CKPT).exists()


def test_in_ic_frozen_load(r001_ckpt_present):
    embedder = InICEmbedder(strict_load=r001_ckpt_present)
    # all parameters frozen
    for p in embedder.parameters():
        assert not p.requires_grad
    # forward shape sanity
    x = torch.randn(4, 11, 3)
    out = embedder(x)
    assert out.shape == (4, 11, 64)


def test_step4_basis_invariant():
    # plan-007 박제 basis 와 일치 — step4_basis_drift severe check 의 anchor
    s = Step4CoeffMLP(mode="F0_only")
    assert tuple(s.basis_names) == STEP4_BEST_BASIS


def test_step4_f0_only_shape():
    s = Step4CoeffMLP(mode="F0_only")
    feat = torch.randn(4, 13)
    coeff = s(feat)
    assert coeff.shape == (4, 8)


def test_step4_27ext_shape():
    s = Step4CoeffMLP(mode="27ext")
    feat = torch.randn(4, 13)
    coeff = s(feat, candidate_idx=17)
    assert coeff.shape == (4, 8)


def test_step4_27ext_requires_candidate_idx():
    s = Step4CoeffMLP(mode="27ext")
    feat = torch.randn(4, 13)
    with pytest.raises(ValueError, match="candidate_idx"):
        s(feat)


def test_step4_invalid_mode():
    with pytest.raises(ValueError, match="unknown mode"):
        Step4CoeffMLP(mode="bogus")


def test_25_cand_load_shape():
    g1_dir = REPO / "runs/baseline/G001_candidate-redefine"
    cand_files = [g1_dir / "cand_set.json", g1_dir / "cand_set.npy"]
    if not any(p.exists() for p in cand_files):
        pytest.skip(
            f"plan-008 G1 cand_set 미존재: {cand_files} — c3 preflight 에서 cand_set_25_drift severe trigger anchor"
        )
    cands = load_25_cand_set(str(g1_dir))
    assert len(cands) == 25


def test_25_cand_missing_raises():
    with pytest.raises(FileNotFoundError):
        load_25_cand_set("/nonexistent/path_for_smoke_test")


def test_in_ic_corrector_wrapper_forward(r001_ckpt_present):
    wrap = InICCorrectorWrapper(strict_load=r001_ckpt_present, dim_cf=96, hidden=64)
    cf = torch.randn(4, 27, 32)
    traj = torch.randn(4, 11, 3)
    # 2-arg 직접 호출 (test 용)
    delta, env = wrap._impl_forward(cf, traj)
    assert delta.shape == (4, 27, 3)
    assert env.shape == (4, 27, 4)


def test_in_ic_corrector_wrapper_1arg_adapter(r001_ckpt_present):
    wrap = InICCorrectorWrapper(strict_load=r001_ckpt_present, dim_cf=96, hidden=64)
    cf = torch.randn(4, 27, 32)
    traj = torch.randn(4, 11, 3)
    wrap._cached_trajectory = traj
    delta, env = wrap(cf)
    assert delta.shape == (4, 27, 3)
    assert env.shape == (4, 27, 4)


def test_in_ic_corrector_wrapper_1arg_without_cache_raises(r001_ckpt_present):
    wrap = InICCorrectorWrapper(strict_load=r001_ckpt_present, dim_cf=96, hidden=64)
    cf = torch.randn(4, 27, 32)
    with pytest.raises(RuntimeError, match="_cached_trajectory"):
        wrap(cf)


def test_in_ic_frozen_invariant(r001_ckpt_present):
    wrap = InICCorrectorWrapper(strict_load=r001_ckpt_present, dim_cf=96, hidden=64)
    # 초기 상태 — frozen invariant holds
    assert check_in_ic_frozen(wrap)
    # base_corrector 학습 시뮬 (1 step) — embedder 는 변경 0
    opt = torch.optim.AdamW(wrap.parameters(), lr=1e-3)
    cf = torch.randn(2, 27, 32)
    traj = torch.randn(2, 11, 3)
    wrap._cached_trajectory = traj
    delta, _env = wrap(cf)
    loss = delta.sum()
    loss.backward()
    opt.step()
    # embedder 는 여전히 frozen
    assert check_in_ic_frozen(wrap)
