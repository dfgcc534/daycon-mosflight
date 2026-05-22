"""plan-026 c4 — smoke tests (spec §4.5)."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parent.parent
_PLAN026 = _REPO / "analysis" / "plan-026"
_PLAN025 = _REPO / "analysis" / "plan-025"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def modules():
    if str(_REPO) not in sys.path:
        sys.path.insert(0, str(_REPO))
    return {
        "bmb": _load(_PLAN026 / "block_mask_builder.py", "t_bmb"),
        "p025_build": _load(_PLAN025 / "build_feat_1080.py", "t_p025_build"),
        "p022_anchors": _load(_REPO / "analysis/plan-022/anchors.py", "t_p022_a"),
        "p021_build": _load(_REPO / "analysis/plan-021/build_input.py", "t_p021"),
        "qc": _load(_REPO / "analysis/plan-024/quantile_carry.py", "t_qc"),
        "bf": _load(_REPO / "analysis/plan-020/baseline_f0.py", "t_bf"),
    }


def test_block_ranges_sum_1080(modules):
    bmb = modules["bmb"]
    total = 0
    for k, (s, e) in bmb.BLOCK_RANGES.items():
        total += e - s
    assert total == 1080


def test_exclusion_map(modules):
    bmb = modules["bmb"]
    assert bmb.EXCLUSION_MAP == {"A1": "block2", "A2": "block3", "A3": "block4"}


def test_expected_d_masked(modules):
    bmb = modules["bmb"]
    assert bmb.EXPECTED_D_MASKED == {"A1": 952, "A2": 1058, "A3": 320}


def _build_masked_helper(modules, N=4, cell="A1"):
    rng = np.random.default_rng(20260522)
    X = rng.standard_normal((N, 11, 3)).astype(np.float32)
    R_wfn = modules["p021_build"].build_frenet_basis_3d(X, end_idx=10)
    qc = modules["qc"].build(X, R_wfn)
    return modules["bmb"].build_feat_masked(
        X, modules["p022_anchors"].ANCHORS_A6, modules["bf"].f0_baseline, qc, cell,
    ), X


def test_build_feat_masked_dims_A1(modules):
    out, X = _build_masked_helper(modules, N=4, cell="A1")
    assert out.shape == (4 * 14, 952)


def test_build_feat_masked_dims_A2(modules):
    out, X = _build_masked_helper(modules, N=4, cell="A2")
    assert out.shape == (4 * 14, 1058)


def test_build_feat_masked_dims_A3(modules):
    out, X = _build_masked_helper(modules, N=4, cell="A3")
    assert out.shape == (4 * 14, 320)


def test_build_feat_masked_preserve_columns(modules):
    """A1 (drop block2) 의 keep column 이 full 1080D 의 동일 column 과 일치."""
    rng = np.random.default_rng(20260522)
    N = 3
    X = rng.standard_normal((N, 11, 3)).astype(np.float32)
    R_wfn = modules["p021_build"].build_frenet_basis_3d(X, end_idx=10)
    qc = modules["qc"].build(X, R_wfn)
    feat_full = modules["p025_build"].build_feat_1080(
        X, modules["p022_anchors"].ANCHORS_A6, modules["bf"].f0_baseline, qc,
    )
    feat_A1 = modules["bmb"].build_feat_masked(
        X, modules["p022_anchors"].ANCHORS_A6, modules["bf"].f0_baseline, qc, "A1",
    )
    # block ② = col 170..298 제거. block ① (0..170) + block ③④ (298..1080) 가 keep.
    expected_keep = np.concatenate([feat_full[:, 0:170], feat_full[:, 298:1080]], axis=1)
    assert np.allclose(feat_A1, expected_keep, atol=1e-6)


def test_prereq_results_C1_exists():
    prereq_path = _PLAN025 / "results_C1.json"
    assert prereq_path.exists(), f"prereq missing: {prereq_path}"
    import json
    with open(prereq_path) as f:
        d = json.load(f)
    assert "hit_1cm" in d
    assert "hit_1p5cm" in d
    assert "max_class_ratio" in d
