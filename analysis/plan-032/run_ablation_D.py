"""plan-032 c2 — Ablation D (input axis 단독, informative).

variants:
  D1: 잔차 (a) drop (residual_a_gru + residual_a_kv 모두 zero) — GRU input 97→95 + K/V H+5 의 잔차 zero
  D2: 잔차 (b) drop (query_64 의 col [22:57] = 잔차 b 35D zero)

G2 (fold-0 1-fold) only — informative measurement (lift 산출 X).

monkey-patch _build_per_sample_artifacts.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

_THIS = Path(__file__).resolve().parent
_REPO = _THIS.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_train_mod = _load(_THIS.parent / "plan-031" / "train.py", "p032_d_train")
_orig_build = _train_mod._build_per_sample_artifacts


def _patched_D1(X, R_wfn, F0, cand_ext):
    r = _orig_build(X, R_wfn, F0, cand_ext)
    r["residual_a_gru"] = np.zeros_like(r["residual_a_gru"])
    r["residual_a_kv"] = np.zeros_like(r["residual_a_kv"])
    return r


def _patched_D2(X, R_wfn, F0, cand_ext):
    r = _orig_build(X, R_wfn, F0, cand_ext)
    # query_64 col [22:57] = 잔차 (b) 35D zero
    r["query_64"] = r["query_64"].copy()
    r["query_64"][:, :, 22:57] = 0.0
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=["D1", "D2"])
    args = ap.parse_args()

    if args.variant == "D1":
        _train_mod._build_per_sample_artifacts = _patched_D1
    else:
        _train_mod._build_per_sample_artifacts = _patched_D2

    from src.io import load_all_samples, load_labels
    from src.pb_0_6822.selector import stable_fold_id

    t0 = time.perf_counter()
    _ids, X = load_all_samples()
    _, gt = load_labels()
    X = X.astype(np.float32); gt = gt.astype(np.float32)
    folds = np.asarray([stable_fold_id(str(s), _train_mod.N_FOLDS) for s in _ids], dtype=int)
    mask = folds == 0
    X_tr, X_te = X[~mask], X[mask]
    gt_tr, gt_te = gt[~mask], gt[mask]

    log = _train_mod.train_one_fold(0, X_tr, X_te, gt_tr, verbose=False)
    err = np.linalg.norm(log["world_pred_te"] - gt_te, axis=1)
    h = float((err <= 0.01).mean())
    h15 = float((err <= 0.015).mean())
    mcr = float(np.bincount(log["probs_te"].argmax(1), minlength=14).max() / len(gt_te))
    elapsed = time.perf_counter() - t0

    # baseline: plan-031 G1 (fold-0) = 0.6450
    delta = h - 0.6450

    result = {
        "ablation_axis": "D",
        "variant": args.variant,
        "drop": "residual_a" if args.variant == "D1" else "residual_b",
        "gate": "G2",
        "fold": 0,
        "N_te": int(len(gt_te)),
        "hit_1cm": h, "hit_1p5cm": h15,
        "max_class_ratio": mcr,
        "delta_vs_plan_031_G1": delta,
        "elapsed_s": elapsed,
        "final_score_std": log["score_std_trajectory"][-1],
    }
    out_path = _THIS / f"results_{args.variant}.json"
    with out_path.open("w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    print(f"[done] {args.variant} ({result['drop']} drop): hit_1cm={h:.4f}, "
          f"Δ vs plan-031 G1={delta:+.4f}, elapsed={elapsed:.1f}s")


if __name__ == "__main__":
    main()
