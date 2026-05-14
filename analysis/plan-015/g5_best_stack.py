"""plan-015 c8 (G5) — best stack 선정 + submission.

G1 negative → drop rule § 발동: G2~G4 skip. best = G0 baseline.

best_stack = plan-014 G5 best_stack (= E0c K-Means K=9 + boundary_weight_on,
F0 frozen, OOF=0.6425). submission = plan-014 best_stack submission 재사용
(plan-015 별도 학습 안 함, deterministic same config).

§9 spec v2.4 carry: candidates argmax + drop rule + band 분류.

Usage:
    python analysis/plan-015/g5_best_stack.py
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-json", type=Path, default=Path("analysis/plan-015/g5_phase4.json"))
    ap.add_argument("--run-dir", type=Path, default=Path("runs/baseline/plan015_g5"))
    args = ap.parse_args()

    t_start = time.time()

    # Load G0 + G1 artifacts
    g0 = json.loads(Path("analysis/plan-015/preflight.json").read_text())
    g1 = json.loads(Path("analysis/plan-015/g1_e1.json").read_text())

    print("[plan-015 G5] === best stack 선정 (candidates argmax + drop rule) ===", flush=True)
    candidates = {
        "baseline": g0["reproduce_5fold_oof"],
        "E1 (A)": g1["e1_oof"],
        # G2/G3/G4 skipped per drop rule
    }
    print(f"  candidates = {candidates}", flush=True)

    # drop rule: G1 negative → E2/E3/E4 already excluded from candidates
    if g1["status"] == "negative":
        print(f"  drop rule: G1 negative ({g1['status']}) → E2/E3/E4 candidates 에서 이미 제외", flush=True)

    # argmax with tie-break (insertion order first)
    best_name = max(candidates, key=candidates.get)
    best_oof = candidates[best_name]
    delta_oof = best_oof - g0["reproduce_5fold_oof"]

    G5_THRESHOLD = 0.005
    G5_passed = delta_oof >= G5_THRESHOLD
    G5_warn = None if G5_passed else "g5_no_improvement"

    # band classification
    if best_oof >= 0.66:
        band = "positive"
    elif best_oof >= 0.65:
        band = "partial"
    else:
        band = "negative"

    print(f"\n[plan-015 G5] === result ===", flush=True)
    print(f"  baseline_oof = {g0['reproduce_5fold_oof']:.4f}", flush=True)
    print(f"  best_name    = {best_name}", flush=True)
    print(f"  best_oof     = {best_oof:.4f}", flush=True)
    print(f"  delta_oof    = {delta_oof:+.4f} (threshold +{G5_THRESHOLD})", flush=True)
    print(f"  G5_passed    = {G5_passed}, warn = {G5_warn}", flush=True)
    print(f"  band         = **{band}**", flush=True)

    # Submission: best = baseline → plan-014 best_stack submission 재사용
    args.run_dir.mkdir(parents=True, exist_ok=True)
    plan014_submission_src = Path("runs/baseline/plan014_g5_phase4/submission_best.csv")
    plan015_submission_dst = args.run_dir / "submission_best.csv"

    if best_name == "baseline":
        # plan-014 submission 재사용
        if plan014_submission_src.exists():
            shutil.copy2(plan014_submission_src, plan015_submission_dst)
            submission_source = "plan-014 best_stack carry (G1 negative drop rule)"
            print(f"  submission   = {plan015_submission_dst} (carry from {plan014_submission_src})",
                  flush=True)
        else:
            print(f"  WARNING: plan-014 submission not found at {plan014_submission_src}",
                  flush=True)
            submission_source = "missing"
    else:
        # E1+ winner — 별도 학습 후 submission 산출 path (이번엔 발동 안 됨)
        submission_source = f"E1 winner (not implemented this run, fallback to baseline)"
        if plan014_submission_src.exists():
            shutil.copy2(plan014_submission_src, plan015_submission_dst)

    elapsed = time.time() - t_start
    artifact = {
        "exp_id": "H047_g5_best_stack_5fold",
        "candidates": candidates,
        "best_name": best_name,
        "best_oof": best_oof,
        "baseline_oof": g0["reproduce_5fold_oof"],
        "delta_oof": delta_oof,
        "G5_threshold": G5_THRESHOLD,
        "G5_passed": G5_passed,
        "G5_warn": G5_warn,
        "band": band,
        "submission_best_path": str(plan015_submission_dst),
        "submission_source": submission_source,
        "drop_rule_activated": g1["status"] == "negative",
        "skipped_stages": ["G2", "G3", "G4"] if g1["status"] == "negative" else [],
        "elapsed_seconds": elapsed,
        "plan_version": "v2.4",
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(artifact, indent=2, ensure_ascii=False))
    print(f"[plan-015 G5] artifact -> {args.out_json}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
