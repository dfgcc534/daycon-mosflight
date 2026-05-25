"""plan-029 c6 — run_oof.py orchestrator.

CLI:
  --g1   : G1 reproduce (F0 + plan-022 winner) via plan-025 baseline_carry.json tight band assert
           or fallback 재산출 (F0 baseline + plan-022 selector_only_eval_5fold)
  --cell X1 : X1 5-fold OOF (4 lever 동시) — final metric + paradigm_analysis

outputs:
  analysis/plan-029/results_X1.json
  analysis/plan-029/train_X1.log
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
for p in (_REPO, _THIS.parent / "plan-025", _THIS.parent / "plan-020"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


_train = _load(_THIS / "train.py", "p029_oof_train")
BASELINE_CARRY_PATH = _THIS.parent / "plan-025" / "baseline_carry.json"


def run_g1(verbose: bool = True) -> dict:
    """G1 carry verification: plan-025 baseline_carry.json + tight band assert.

    Tight bands (§5.1):
      F0 hit_1cm   ∈ [0.6315, 0.6325]
      F0 hit_1p5cm ∈ [0.8028, 0.8038]
      p022 hit_1cm   ∈ [0.6523, 0.6533]
      p022 hit_1p5cm ∈ [0.8099, 0.8109]
    """
    if not BASELINE_CARRY_PATH.exists():
        raise FileNotFoundError(
            f"baseline_carry.json not found at {BASELINE_CARRY_PATH}. "
            f"§5.2 fallback: re-run plan-025 G1 reproduce or use --cell G1 (TODO)."
        )
    with open(BASELINE_CARRY_PATH) as f:
        carry = json.load(f)

    F0_hit_1cm = float(carry["F0"]["hit_1cm"])
    F0_hit_1p5cm = float(carry["F0"]["hit_1p5cm"])
    p022_hit_1cm = float(carry["plan022_winner"]["hit_1cm"])
    p022_hit_1p5cm = float(carry["plan022_winner"]["hit_1p5cm"])

    if verbose:
        print(f"[G1] carry from {BASELINE_CARRY_PATH.name}")
        print(f"  F0  hit_1cm={F0_hit_1cm:.4f}, hit_1p5cm={F0_hit_1p5cm:.4f}")
        print(f"  p022 hit_1cm={p022_hit_1cm:.4f}, hit_1p5cm={p022_hit_1p5cm:.4f}")

    assert 0.6315 <= F0_hit_1cm <= 0.6325, f"F0 hit_1cm drift: {F0_hit_1cm}"
    assert 0.8028 <= F0_hit_1p5cm <= 0.8038, f"F0 hit_1p5cm drift: {F0_hit_1p5cm}"
    assert 0.6523 <= p022_hit_1cm <= 0.6533, f"p022 hit_1cm drift: {p022_hit_1cm}"
    assert 0.8099 <= p022_hit_1p5cm <= 0.8109, f"p022 hit_1p5cm drift: {p022_hit_1p5cm}"

    if verbose:
        print("[G1] ✓ tight band PASS")
    return {
        "F0_hit_1cm": F0_hit_1cm, "F0_hit_1p5cm": F0_hit_1p5cm,
        "p022_hit_1cm": p022_hit_1cm, "p022_hit_1p5cm": p022_hit_1p5cm,
        "source": str(BASELINE_CARRY_PATH),
        "verdict": "PASS",
    }


def run_x1(verbose: bool = True) -> dict:
    """X1 5-fold OOF — 4 lever 동시 paradigm-level 검증."""
    t0 = time.perf_counter()
    if verbose:
        print(f"[X1] starting 5-fold OOF (epochs={_train.EPOCHS}, batch={_train.BATCH_SIZE}, "
              f"hidden={_train.HIDDEN}, anchor_embed_init={_train.ANCHOR_EMBED_INIT_SCALE})")
    result = _train.run_5fold_oof(verbose=verbose)
    elapsed = time.perf_counter() - t0

    # Result summary
    hit_1cm = result["hit_1cm"]
    hit_1p5cm = result["hit_1p5cm"]
    max_class_ratio = result["max_class_ratio"]
    top1_acc = result["top1_acc"]
    grad_norm_traj = [
        log["grad_norm_trajectory"] for log in result["fold_logs"]
    ]
    grad_norm_ep5 = [traj[4] if len(traj) > 4 else 0.0 for traj in grad_norm_traj]

    # G2.X1 gate checks
    g2_finite = np.isfinite([hit_1cm, hit_1p5cm, max_class_ratio, top1_acc]).all()
    g2_max_class = max_class_ratio < 0.95
    g2_grad_at_ep5 = min(grad_norm_ep5) > 1e-4
    g2_grad_final = min([traj[-1] for traj in grad_norm_traj]) > 0
    g2_pass = bool(g2_finite and g2_max_class and g2_grad_at_ep5 and g2_grad_final)

    # G3 paradigm verdict
    if hit_1cm > 0.6528:
        verdict = "PASS"
    elif hit_1cm > 0.6387:
        verdict = "partial_above_p024"
    elif hit_1cm >= 0.6320:
        verdict = "partial_below_p024"
    else:
        verdict = "regression"

    # mode_collapse warn
    mode_collapse_warn = 0.05 <= max_class_ratio < 0.10
    mode_collapse_attention_warn = min(grad_norm_ep5) <= 1e-4

    summary = {
        "cell": "X1",
        "hit_1cm": hit_1cm,
        "hit_1p5cm": hit_1p5cm,
        "max_class_ratio": max_class_ratio,
        "top1_acc": top1_acc,
        "verdict_G3": verdict,
        "g2_x1": {
            "finite": bool(g2_finite),
            "max_class_ratio_under_0_95": bool(g2_max_class),
            "grad_ep5_over_1e-4": bool(g2_grad_at_ep5),
            "grad_final_over_0": bool(g2_grad_final),
            "pass": g2_pass,
        },
        "warns": {
            "mode_collapse": bool(mode_collapse_warn),
            "mode_collapse_attention": bool(mode_collapse_attention_warn),
        },
        "elapsed_total_s": elapsed,
        "N_total": result["N_total"],
        "K": result["K"],
        "hparams": result["hparams"],
        "fold_logs": result["fold_logs"],
    }

    # paired Δ vs plan-022 winner / F0 (from baseline_carry.json)
    if BASELINE_CARRY_PATH.exists():
        with open(BASELINE_CARRY_PATH) as f:
            carry = json.load(f)
        summary["delta_vs_F0_1cm"] = hit_1cm - float(carry["F0"]["hit_1cm"])
        summary["delta_vs_p022_1cm"] = hit_1cm - float(carry["plan022_winner"]["hit_1cm"])
        summary["oracle_14anchor"] = 0.7928  # MEMORY 박제
        summary["recall_rate_vs_oracle"] = hit_1cm / 0.7928

    if verbose:
        print(f"\n[X1] DONE — hit_1cm={hit_1cm:.4f} hit_1p5cm={hit_1p5cm:.4f} "
              f"max_class_ratio={max_class_ratio:.4f} top1_acc={top1_acc:.4f}")
        print(f"[G3] verdict = {verdict}")
        print(f"[G2.X1] pass = {g2_pass} (finite={g2_finite}, "
              f"max_class<0.95={g2_max_class}, grad_ep5>1e-4={g2_grad_at_ep5}, "
              f"grad_final>0={g2_grad_final})")
        print(f"[elapsed] {elapsed:.1f}s ({elapsed/60:.1f}min)")

    # Dump
    out_json = _THIS / "results_X1.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[dump] {out_json}")

    # Dump oof_pred / oof_probs / gt_anchor_label for c10 paradigm_analysis
    np.savez_compressed(
        _THIS / "oof_X1.npz",
        oof_pred=result["oof_pred"],
        oof_probs=result["oof_probs"],
        gt_anchor_label=np.asarray(result["gt_anchor_label"], dtype=np.int64),
    )
    print(f"[dump] {_THIS / 'oof_X1.npz'}")

    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--g1", action="store_true", help="G1 carry verification")
    ap.add_argument("--cell", type=str, default=None, choices=[None, "X1"], help="cell name")
    ap.add_argument("--verbose", action="store_true", default=True)
    args = ap.parse_args()

    if args.g1:
        run_g1(verbose=args.verbose)
    if args.cell == "X1":
        run_x1(verbose=args.verbose)
    if not args.g1 and args.cell is None:
        ap.print_help()


if __name__ == "__main__":
    main()
