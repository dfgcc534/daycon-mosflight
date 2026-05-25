"""plan-032 c1 — Ablation A (label τ_cls 완화).

plan-031 train.py carry + TAU_CLS override 만.
Variants: A1 (τ=0.005), A2 (τ=0.01).
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


_train_mod = _load(_THIS.parent / "plan-031" / "train.py", "p032_a_train")


VARIANTS = {
    "A1": 0.005,
    "A2": 0.01,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=list(VARIANTS.keys()))
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    tau_cls = VARIANTS[args.variant]
    _train_mod.TAU_CLS = tau_cls

    t0 = time.perf_counter()
    result = _train_mod.run_5fold_oof(verbose=not args.quiet)
    elapsed = time.perf_counter() - t0

    h = result["hit_1cm"]
    band = ("EXCELLENT" if h >= 0.6624 else
            "PASS" if h >= 0.6511 else
            "STRONG" if h >= 0.6387 else
            "BORDERLINE" if h >= 0.6320 else
            "FAIL_regression")
    result["ablation_axis"] = "A"
    result["variant"] = args.variant
    result["tau_cls_override"] = tau_cls
    result["delta_vs_plan_031"] = h - 0.6397
    result["band"] = band
    result["total_elapsed_runner_s"] = elapsed

    out_path = _THIS / f"results_{args.variant}.json"
    arrays = {}
    for k in ("oof_pred", "oof_probs"):
        if k in result:
            arrays[k] = result.pop(k)
    with out_path.open("w") as f:
        json.dump(result, f, indent=2)
    if arrays:
        np.savez_compressed(out_path.with_suffix(".npz"), **arrays)

    summary = {k: v for k, v in result.items()
               if k not in ("fold_logs", "gt_anchor_label", "oof_pred", "oof_probs")}
    print(json.dumps(summary, indent=2))
    print(f"[done] {args.variant} (τ_cls={tau_cls}): hit_1cm={h:.4f}, "
          f"Δ vs plan-031={h-0.6397:+.4f}, band={band}, elapsed={elapsed:.1f}s")


if __name__ == "__main__":
    main()
