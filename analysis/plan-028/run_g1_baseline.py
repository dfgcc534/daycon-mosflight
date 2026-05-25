"""plan-028 c6 — G1 baseline carry reproduce (F0 + plan-022 winner + plan-025 C1).

§0.5 c6 박제: 3 carry reproduce + baseline_carry.json 박제.

plan-025 run_g1_reproduce() 가 F0 + plan-022 winner 동시 reproduce. plan-025 C1
(= 본 plan B4 cell = 1080D full + weight ON + τ=0.001 + LgbmSelectorRowExpanded) 은
별도로 run_oof_cell_subset("B4") 호출.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

_THIS = Path(__file__).resolve().parent
_REPO = _THIS.parent.parent
_PLAN025 = _THIS.parent / "plan-025"

if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


_p025_run = _load(_PLAN025 / "run_oof.py", "p028_g1_p025run")
_subset_run = _load(_THIS / "run_oof_subset.py", "p028_g1_subset")

from src.io import load_all_samples, load_labels  # noqa: E402


def _dataset_hash() -> str:
    """sample id list + labels SHA256 hash (drift 검출용)."""
    _ids, _ = load_all_samples()
    _ids_gt, gt = load_labels()
    h = hashlib.sha256()
    for sid in _ids:
        h.update(str(sid).encode())
    h.update(gt.tobytes())
    return h.hexdigest()[:16]


def main() -> dict:
    t0 = time.time()

    # G1.a + G1.b: plan-025 run_g1_reproduce — F0 + plan-022 winner
    print("[G1.a+b] running plan-025 run_g1_reproduce (F0 + plan-022 winner)...")
    p025_g1 = _p025_run.run_g1_reproduce(verbose=True)

    # G1.c: plan-025 C1 reproduce (= 본 plan B4 cell)
    print("\n[G1.c] running B4 (= plan-025 C1) 5-fold OOF...")
    b4 = _subset_run.run_oof_cell_subset("B4", verbose=True)

    # Tight band 검증
    f0_hit = p025_g1["F0"]["hit_1cm"]
    f0_hit_15 = p025_g1["F0"]["hit_1p5cm"]
    p022_hit = p025_g1["plan022_winner"]["hit_1cm"]
    p022_hit_15 = p025_g1["plan022_winner"]["hit_1p5cm"]
    assert 0.6315 <= f0_hit <= 0.6325, f"F0 drift: {f0_hit}"
    assert 0.8028 <= f0_hit_15 <= 0.8038, f"F0 drift: {f0_hit_15}"
    assert 0.6526 <= p022_hit <= 0.6536, f"plan-022 drift: {p022_hit}"
    assert 0.8103 <= p022_hit_15 <= 0.8113, f"plan-022 drift: {p022_hit_15}"
    assert 0.6315 <= b4["hit_1cm"] <= 0.6325, f"plan-025 C1/B4 drift: {b4['hit_1cm']}"
    assert 0.8028 <= b4["hit_1p5cm"] <= 0.8038, f"plan-025 C1/B4 drift: {b4['hit_1p5cm']}"

    baseline = {
        "f0": {"hit_1cm": f0_hit, "hit_1p5cm": f0_hit_15},
        "plan022_winner": {
            "hit_1cm": p022_hit,
            "hit_1p5cm": p022_hit_15,
            "cell": "A6_bcc14_tau001",
        },
        "plan025_C1": {
            "hit_1cm": b4["hit_1cm"],
            "hit_1p5cm": b4["hit_1p5cm"],
            "max_class_ratio": b4["max_class_ratio"],
            "top1_acc": b4["top1_acc"],
            "input_dim": 1080,
        },
        "dataset_hash": _dataset_hash(),
        "fold_seed": "stable_fold_id MD5",
        "n_folds": 5,
        "runtime_s_total": time.time() - t0,
    }

    out_path = _THIS / "baseline_carry.json"
    with open(out_path, "w") as f:
        json.dump(baseline, f, indent=2, default=str)
    print(f"\n[G1] baseline_carry.json saved: {out_path}")
    print(f"  F0:    {f0_hit:.4f} / {f0_hit_15:.4f}")
    print(f"  p022:  {p022_hit:.4f} / {p022_hit_15:.4f}")
    print(f"  B4:    {b4['hit_1cm']:.4f} / {b4['hit_1p5cm']:.4f} "
          f"(max_class_ratio={b4['max_class_ratio']:.4f})")
    print(f"  total runtime: {time.time() - t0:.1f}s")

    # B4 결과 별도 저장 (c10 단계의 results_B4.json carry)
    b4_path = _THIS / "results_B4.json"
    with open(b4_path, "w") as f:
        json.dump(b4, f, indent=2, default=str)
    print(f"[c10 carry] results_B4.json saved: {b4_path}")

    return baseline


if __name__ == "__main__":
    main()
