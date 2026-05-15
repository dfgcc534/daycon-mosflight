"""plan-019 c2 (STAGE 0, G0) — EDA sanity + A0 baseline import.

§4.1 spec carry (plan-018 §4 답습). 신규 작성 (§10 정책 — import X).

3 task:
  (a) 데이터셋 shape + per-axis std sanity (plan-001 §3 carry).
  (b) const-velocity baseline MAE check (plan-007 §4.1 carry).
  (c) plan-007 mlp_coeff.json 의 OOF=0.6482 import + G0 합격 (∈ [0.6479, 0.6485]).

본 plan 의 A0 5-fold OOF reproduce 는 c4 (`src/plan019/baseline_a0.py`) 에서.
본 file 은 *sanity precheck only*.

Usage:
    python analysis/plan-019/eda_check.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.io import load_all_samples, load_labels  # noqa: E402


# §3.2 G0 합격 기준
A0_OOF_TARGET_MIN = 0.6479
A0_OOF_TARGET_MAX = 0.6485


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-json", type=Path, default=Path("analysis/plan-019/eda_check.json"))
    args = ap.parse_args()

    t0 = time.time()
    print("[plan-019 G0] (a) shape + per-axis std sanity ...", flush=True)
    ids, X = load_all_samples("train")
    label_ids, Y = load_labels()
    if ids != label_ids:
        print("  ERROR: id ordering mismatch", file=sys.stderr)
        return 1
    X = X.astype(np.float64)
    Y = Y.astype(np.float64)

    shape_ok = X.shape == (10000, 11, 3) and Y.shape == (10000, 3)
    print(f"  X.shape={X.shape}, Y.shape={Y.shape}, shape_ok={shape_ok}", flush=True)

    axis_std = X.reshape(-1, 3).std(axis=0)
    std_ok = bool(np.all((axis_std > 0.4) & (axis_std < 1.5)))
    print(f"  axis_std={axis_std.tolist()}, std_ok={std_ok}", flush=True)

    # (b) const-velocity 2-step horizon baseline MAE (plan-007 convention)
    print("[plan-019 G0] (b) const-velocity baseline MAE ...", flush=True)
    vel = np.diff(X, axis=1).mean(axis=1)
    const_vel_pred = X[:, -1] + 2.0 * vel
    mae_per_axis_max = float(np.abs(const_vel_pred - Y).mean(axis=0).max())
    mae_ok = mae_per_axis_max < 0.020
    print(f"  const_vel_mae_per_axis_max={mae_per_axis_max:.4f}, mae_ok={mae_ok}", flush=True)

    hit_const = float((np.linalg.norm(const_vel_pred - Y, axis=1) < 0.01).mean())
    print(f"  const_vel hit_rate@1cm = {hit_const:.4f}", flush=True)

    # (c) A0 baseline import (plan-007 mlp_coeff.json)
    print("[plan-019 G0] (c) A0 baseline import (plan-007 mlp_coeff.json) ...", flush=True)
    mlp_path = REPO_ROOT / "analysis/plan-007/mlp_coeff.json"
    if not mlp_path.exists():
        print(f"  ERROR: {mlp_path} missing", file=sys.stderr)
        return 1
    mlp_coeff = json.loads(mlp_path.read_text())
    a0_oof = float(mlp_coeff["oof_hit"])
    in_range = A0_OOF_TARGET_MIN <= a0_oof <= A0_OOF_TARGET_MAX
    print(
        f"  A0 OOF (plan-007 step 4) = {a0_oof:.4f}, "
        f"target ∈ [{A0_OOF_TARGET_MIN}, {A0_OOF_TARGET_MAX}], in_range={in_range}",
        flush=True,
    )

    basis_path = REPO_ROOT / "analysis/plan-007/basis_ablation.json"
    basis = json.loads(basis_path.read_text())
    best_basis_vars = basis["best_basis_vars"]
    print(f"  best_basis_vars (n={len(best_basis_vars)}) = {best_basis_vars}", flush=True)

    g0_passed = shape_ok and std_ok and mae_ok and in_range

    elapsed = time.time() - t0
    summary = {
        "exp_id": "F013_eda_check",
        "plan_version": "v1.1",
        "shape_ok": shape_ok,
        "axis_std": axis_std.tolist(),
        "std_ok": std_ok,
        "const_vel_mae_per_axis_max_m": mae_per_axis_max,
        "mae_ok": mae_ok,
        "const_vel_hit_1cm": hit_const,
        "a0_oof_hit_1cm_imported": a0_oof,
        "a0_oof_target_min": A0_OOF_TARGET_MIN,
        "a0_oof_target_max": A0_OOF_TARGET_MAX,
        "a0_in_range_imported": in_range,
        "best_basis_vars": best_basis_vars,
        "n_coeffs": len(best_basis_vars),
        "all_assertions_pass": g0_passed,
        "g0_imported_passed": g0_passed,
        "note": "본 file 은 plan-007 step 4 OOF import-only sanity. "
                "실제 A0 5-fold OOF reproduce 는 c4 baseline_a0.py 에서 별도 수행.",
        "elapsed_seconds": elapsed,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(
        f"\n[plan-019 G0 precheck] passed={g0_passed}, artifact -> {args.out_json} ({elapsed:.1f}s)",
        flush=True,
    )
    return 0 if g0_passed else 1


if __name__ == "__main__":
    sys.exit(main())
