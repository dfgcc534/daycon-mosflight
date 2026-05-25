"""plan-028 c16 — G2.A analysis: 5가설 verdict + branch decision.

§4.5 decide_branch((b)+(d) 5 cell) + §4.6 verdict 함수 (a/b/c/d/e) 통합.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

_THIS = Path(__file__).resolve().parent

def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_branch = _load(_THIS / "branch_decision.py", "p028_branch_a")


def load_cell(cell_id: str) -> dict:
    p = _THIS / f"results_{cell_id}.json"
    with open(p) as f:
        return json.load(f)


def verdict_5hypothesis(cells: dict[str, dict]) -> dict[str, str]:
    """§4.6 5가설 verdict 함수 (a/b/c/d/e)."""
    B1 = cells["B1"]["hit_1cm"]
    B2 = cells["B2"]["hit_1cm"]
    B3 = cells["B3"]["hit_1cm"]
    B4 = cells["B4"]["hit_1cm"]
    W1 = cells["W1"]["hit_1cm"]
    T1 = cells["T1"]["hit_1cm"]
    T2 = cells["T2"]["hit_1cm"]
    S1 = cells["S1"]["hit_1cm"]
    R1 = cells["R1"]["hit_1cm"]

    def _verdict(positive, negative):
        if positive:
            return "confirmed"
        if negative:
            return "rejected"
        return "inconclusive"

    return {
        "(a) tau_cls_sharp_gap": _verdict(
            max(T1, T2) > B4 + 0.005,
            max(T1, T2) < B4 - 0.003,
        ),
        "(b) sample_weight": _verdict(
            W1 > B4 + 0.005,
            W1 < B4 - 0.003,
        ),
        "(c) subclass_self_consistency": _verdict(
            S1 > B4 + 0.005,
            S1 < B4 - 0.003,
        ),
        "(d) broadcast_dominance": _verdict(
            B1 > B3 + 0.005 or B2 > B4 + 0.005,
            B2 < B4 - 0.003 and B1 < B3 - 0.003,
        ),
        "(e) seq_compression_lossy": _verdict(
            R1 > B4 + 0.005,
            R1 < B4 - 0.003,
        ),
    }


def main():
    cells = {cid: load_cell(cid) for cid in ["B1", "B2", "B3", "B4", "W1", "T1", "T2", "S1", "R1"]}

    # 5가설 verdict
    verdict = verdict_5hypothesis(cells)

    # branch decision ((b)+(d) only)
    branch = _branch.decide_branch(
        B1=cells["B1"]["hit_1cm"],
        B2=cells["B2"]["hit_1cm"],
        B3=cells["B3"]["hit_1cm"],
        B4=cells["B4"]["hit_1cm"],
        W1=cells["W1"]["hit_1cm"],
    )

    # 표 출력
    print("=" * 70)
    print(f"{'cell':6s} {'hit_1cm':>9s} {'Δ_F0':>9s} {'Δ_p022':>9s} {'Δ_B4':>9s} "
          f"{'max_ratio':>10s} {'top1':>8s} {'runtime':>8s}")
    print("-" * 70)
    F0 = 0.6320
    P022 = 0.6530  # G1 reproduce 값
    B4_hit = cells["B4"]["hit_1cm"]
    rows = []
    for cid in ["B1", "B2", "B3", "B4", "W1", "T1", "T2", "S1", "R1"]:
        h = cells[cid]["hit_1cm"]
        d_f0 = h - F0
        d_p022 = h - P022
        d_b4 = h - B4_hit
        mcr = cells[cid]["max_class_ratio"]
        top1 = cells[cid]["top1_acc"]
        rt = cells[cid]["runtime_s"]
        print(f"{cid:6s} {h:>9.4f} {d_f0:>+9.4f} {d_p022:>+9.4f} {d_b4:>+9.4f} "
              f"{mcr:>10.4f} {top1:>8.4f} {rt:>7.1f}s")
        rows.append({
            "cell": cid,
            "hit_1cm": h,
            "hit_1p5cm": cells[cid]["hit_1p5cm"],
            "delta_vs_f0": d_f0,
            "delta_vs_p022": d_p022,
            "delta_vs_B4": d_b4,
            "max_class_ratio": mcr,
            "top1_acc": top1,
            "runtime_s": rt,
        })
    print("=" * 70)

    print(f"\n5가설 verdict:")
    for k, v in verdict.items():
        print(f"  {k}: {v}")

    print(f"\nbranch decision: {branch}")

    out = {
        "cells": rows,
        "hypothesis_verdict": verdict,
        "g2b_branch": branch,
        "baselines": {"F0": F0, "plan022_winner": P022, "B4": B4_hit},
    }
    out_path = _THIS / "paradigm_analysis_g2a.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved: {out_path}")


if __name__ == "__main__":
    main()
