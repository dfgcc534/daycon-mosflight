"""plan-023 §6.2 + §6.3 — 5-fold OOF runner for 12-cell large-N sweep.

plan-022 `run_oof_cell` 직접 import (importlib pattern) — 본 plan code 변경 X.
오로지 `analysis/plan-023/anchors_largeN.py` 의 LAYOUT_NAMES_B (B1~B4) 위 12 회 호출.

CLI:
  # 단일 cell:
  python analysis/plan-023/run_oof_largeN.py --layout B1_dodeca20 --tau 0.001 \
         --out-json /tmp/cell.json

  # 단일 layout (3 τ):
  python analysis/plan-023/run_oof_largeN.py --layout B2_trunc_octa24 \
         --out-json analysis/plan-023/results_B2.json

  # 12-cell 전수 sweep:
  python analysis/plan-023/run_oof_largeN.py --all \
         --out-dir analysis/plan-023/
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

_THIS = Path(__file__).resolve().parent
_REPO = _THIS.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.io import load_all_samples, load_labels                       # noqa: E402
from src.pb_0_6822.selector import stable_fold_id                       # noqa: E402

# plan-023 anchors (local)
_spec = importlib.util.spec_from_file_location("anchors_largeN", _THIS / "anchors_largeN.py")
anchors_mod = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(anchors_mod)
# anchors_mod.LAYOUT_NAMES_B, ANCHORS_B1..B4

# plan-022 modules (reused for run_oof_cell)
_P022 = _THIS.parent / "plan-022"
_spec = importlib.util.spec_from_file_location("p022_run", _P022 / "run_oof.py")
p022_run = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(p022_run)
# p022_run.run_oof_cell, run_sub_exp signature, N_FOLDS, TAU_SCAN

N_FOLDS = 5
TAU_SCAN = [0.001, 0.003, 0.005]


# ── helpers ────────────────────────────────────────────────────────────


def assign_folds(ids: list[str]) -> np.ndarray:
    return np.asarray([stable_fold_id(str(sid), N_FOLDS) for sid in ids], dtype=int)


# ── §6.3 per-sub-exp (1 layout × 3 τ_cls) — plan-022 run_sub_exp carry ─


def run_sub_exp_largeN(
    X: np.ndarray,
    Y: np.ndarray,
    folds: np.ndarray,
    layout_name: str,
    anchors: np.ndarray,
    verbose: bool = True,
) -> dict:
    """layout 의 3 τ_cls cell 측정 → dict {tau_str: cell_dict, ...}.

    plan-022 `p022_run.run_sub_exp` 정확 carry — anchors / layout name 만 swap.
    """
    out: dict[str, Any] = {}
    for tau in TAU_SCAN:
        t0 = time.time()
        if verbose:
            print(
                f"  [{layout_name}] τ_cls={tau:.3f} K={anchors.shape[0]} ...",
                flush=True,
            )
        cell = p022_run.run_oof_cell(X, Y, folds, anchors, tau, verbose=verbose)
        cell["elapsed_sec"] = float(time.time() - t0)
        if verbose:
            print(
                f"  [{layout_name}] τ_cls={tau:.3f}  "
                f"hit@1cm={cell['hit_1cm']:.4f}  hit@1.5cm={cell['hit_1.5cm']:.4f}  "
                f"Δ_1cm={cell['delta_1cm']:+.4f}  Δ_1.5cm={cell['delta_1.5cm']:+.4f}  "
                f"pass={cell['pass_both']}  "
                f"max_class={cell['max_class_ratio']:.3f}  "
                f"{cell['elapsed_sec']:.0f}s",
                flush=True,
            )
        out[f"tau_{tau:.3f}"] = cell
    return out


# ── §6.3 full sweep (12 cell = 4 layout × 3 τ_cls) ────────────────────


def run_sweep_largeN(
    X: np.ndarray, Y: np.ndarray, folds: np.ndarray, verbose: bool = True
) -> dict[str, dict]:
    """4 layout × 3 τ_cls = 12 cell 전수 측정."""
    sweep: dict[str, dict] = {}
    for layout_name, anchors in anchors_mod.LAYOUT_NAMES_B.items():
        print(f"\n=== {layout_name} ===", flush=True)
        sweep[layout_name] = run_sub_exp_largeN(
            X, Y, folds, layout_name, anchors, verbose=verbose
        )
    return sweep


# ── CLI ────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--layout",
        default=None,
        choices=list(anchors_mod.LAYOUT_NAMES_B.keys()),
        help="단일 layout (3 τ_cls cell 측정). 미지정 시 --all 필수.",
    )
    ap.add_argument("--tau", type=float, default=None,
                    help="단일 cell 측정 (--layout 와 같이). 미지정 시 3 τ scan.")
    ap.add_argument("--all", action="store_true", help="12-cell 전수 sweep")
    ap.add_argument("--out-json", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="--all 모드에서 layout 별 results_B{n}.json 자동 저장.")
    args = ap.parse_args()

    if not args.all and args.layout is None:
        ap.error("--layout <name> 또는 --all 중 하나 필요")

    t0 = time.time()
    print("[plan-023 run_oof_largeN] loading data ...", flush=True)
    ids, X = load_all_samples(split="train")
    label_ids, Y = load_labels()
    assert ids == label_ids, "ids mismatch"
    X = X.astype(np.float64)
    Y = Y.astype(np.float64)
    folds = assign_folds(ids)
    print(
        f"[plan-023] N={X.shape[0]} folds={np.bincount(folds).tolist()}",
        flush=True,
    )

    if args.all:
        sweep = run_sweep_largeN(X, Y, folds)
        if args.out_dir:
            args.out_dir.mkdir(parents=True, exist_ok=True)
            for layout_name, sub_exp in sweep.items():
                sub_exp_key = layout_name.split("_")[0]                # "B1", "B2", "B3", "B4"
                path = args.out_dir / f"results_{sub_exp_key}.json"
                path.write_text(json.dumps(sub_exp, indent=2))
                print(f"  wrote {path}", flush=True)
        if args.out_json:
            args.out_json.write_text(json.dumps(sweep, indent=2))
            print(f"  wrote {args.out_json}", flush=True)
    elif args.tau is not None:
        anchors = anchors_mod.LAYOUT_NAMES_B[args.layout]
        cell = p022_run.run_oof_cell(X, Y, folds, anchors, args.tau)
        print(json.dumps(cell, indent=2, default=str), flush=True)
        if args.out_json:
            args.out_json.write_text(json.dumps(cell, indent=2, default=str))
            print(f"  wrote {args.out_json}", flush=True)
    else:
        anchors = anchors_mod.LAYOUT_NAMES_B[args.layout]
        sub_exp = run_sub_exp_largeN(X, Y, folds, args.layout, anchors)
        if args.out_json:
            args.out_json.write_text(json.dumps(sub_exp, indent=2))
            print(f"  wrote {args.out_json}", flush=True)

    print(f"\n[plan-023 run_oof_largeN] total {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
