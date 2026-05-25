"""plan-031 run_oof — G1 (1-fold) + G3 (5-fold OOF) orchestrator."""
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
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


_train_mod = _load(_THIS / "train.py", "p031_runoof_train")


def run_g1(verbose: bool = True) -> dict:
    """G1: fold-0 (plan-031 PASS threshold = F0 + 0.001 = 0.6330)."""
    from src.io import load_all_samples, load_labels
    from src.pb_0_6822.selector import stable_fold_id

    t0 = time.perf_counter()
    _ids, X = load_all_samples()
    _ids_gt, gt = load_labels()
    assert _ids == _ids_gt
    X = X.astype(np.float32)
    gt = gt.astype(np.float32)

    folds = np.asarray([stable_fold_id(str(sid), _train_mod.N_FOLDS) for sid in _ids], dtype=int)
    fold = 0
    train_idx = np.where(folds != fold)[0]
    test_idx = np.where(folds == fold)[0]
    X_tr, X_te = X[train_idx], X[test_idx]
    gt_tr, gt_te = gt[train_idx], gt[test_idx]

    log = _train_mod.train_one_fold(fold=fold, X_tr=X_tr, X_te=X_te, gt_tr=gt_tr, verbose=verbose)
    err = np.linalg.norm(log["world_pred_te"] - gt_te, axis=1)
    hit_1cm = float((err <= 0.01).mean())
    hit_1p5cm = float((err <= 0.015).mean())
    top1_argmax = log["probs_te"].argmax(axis=1)
    max_class_ratio = float(np.bincount(top1_argmax, minlength=_train_mod.K_ANCHORS).max() / len(test_idx))

    elapsed = time.perf_counter() - t0
    return {
        "gate": "G1",
        "fold": fold,
        "N_te": int(len(test_idx)),
        "hit_1cm": hit_1cm,
        "hit_1p5cm": hit_1p5cm,
        "max_class_ratio": max_class_ratio,
        "elapsed_s": elapsed,
        "final_score_std": log["score_std_trajectory"][-1],
        "score_std_trajectory": log["score_std_trajectory"],
        "loss_trajectory": log["loss_trajectory"],
        "PASS_threshold_hit_1cm": 0.6330,  # plan-031 conservative (F0 + 0.001)
        "PASS": hit_1cm > 0.6330,
    }


def run_g3(verbose: bool = True) -> dict:
    """G3: 5-fold OOF (PASS ≥ 0.6360, STRONG ≥ 0.6387, EXCELLENT ≥ 0.65)."""
    result = _train_mod.run_5fold_oof(verbose=verbose)
    h = result["hit_1cm"]
    if h >= 0.65:
        band = "EXCELLENT"
    elif h >= 0.6387:
        band = "STRONG"
    elif h >= 0.6360:
        band = "PASS"
    elif h >= 0.6320:
        band = "BORDERLINE"
    else:
        band = "FAIL_regression"
    result["gate"] = "G3"
    result["band"] = band
    result["PASS"] = h >= 0.6360
    return result


def _dump(result: dict, path: Path):
    arrays = {}
    for k in ("oof_pred", "oof_probs"):
        if k in result:
            arrays[k] = result.pop(k)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(result, f, indent=2)
    if arrays:
        np.savez_compressed(path.with_suffix(".npz"), **arrays)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", choices=["g1", "g3"], required=True)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    if args.gate == "g1":
        result = run_g1(verbose=not args.quiet)
    else:
        result = run_g3(verbose=not args.quiet)

    out_path = _THIS / f"results_{args.gate}.json"
    _dump(result, out_path)
    summary = {k: v for k, v in result.items()
               if k not in ("oof_pred", "oof_probs", "fold_logs", "gt_anchor_label",
                            "score_std_trajectory", "loss_trajectory")}
    print(json.dumps(summary, indent=2))
    print(f"[done] {args.gate} dumped to {out_path}")


if __name__ == "__main__":
    main()
