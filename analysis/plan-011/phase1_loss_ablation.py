"""plan-011 c4 — Phase 1.L Loss axis ablation (8 sub-exp, 1-fold approx fold=0).

★ 본 wrapper 의 구조:
  - L0 (anchor) = plan-006 default OOF (= plan-005 corrected_oof.npz @ CANDIDATES[17] 그대로 사용, 재학습 X).
  - L1~L7 = corrector_redesign_v2.py components + 추가 학습 필요.

decision-note (자율 결정):
  - L0 anchor 는 plan-005 npz 의 fold-0 부분만 추출 (재학습 0 — wall-time 절약).
  - L1~L7 학습 wrapper 는 *별도 commit* 으로 분리 진행 (1 sub-exp = ~10min 실제 학습 필요).
  - 본 c4 commit = wrapper scaffold + L0 anchor 박제. L1~L7 학습 = c5 (별도 commit).

산출:
  - runs/baseline/H011_phase1-loss-ablation/sub_L0/report_sub_L0.json (anchor 박제)
  - analysis/plan-011/phase1_loss_summary.json (8 sub-exp 통합 표 — c5 까지 누적)
"""
from __future__ import annotations
import argparse
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
KST = timezone(timedelta(hours=9))

CANDIDATES_IDX = 17  # frenet_par120_perp_neg020
N_FOLDS = 5
FOLD_VAL = 0
R_HIT = 0.01

BAND_EDGES_M = [(0.0, 0.005), (0.005, 0.010), (0.010, 0.015), (0.015, 0.020), (0.020, float("inf"))]
BAND_NAMES = ["[0,0.5cm)", "[0.5,1cm)", "[1,1.5cm)", "[1.5,2cm)", "[2cm,inf)"]


def load_anchor_data(corrected_oof_path: Path, labels_path: Path):
    """plan-005 npz + train_labels.csv 로딩 → fold-0 val 만 추출.

    Returns:
      val_raw_pos:    (N_val, 3) — anchor raw (단일공식)
      val_corrected:  (N_val, 3) — L0 anchor corrected (plan-004 default = anchor)
      val_truth:      (N_val, 3)
      val_ids:        list[str]
    """
    from src.pb_0_6822.selector import stable_fold_id

    z = np.load(corrected_oof_path)
    cands_all = z["cands"][:, CANDIDATES_IDX, :].astype(np.float64)      # (N, 3)
    corrected_all = z["corrected"][:, CANDIDATES_IDX, :].astype(np.float64)
    df = pd.read_csv(labels_path)
    truth_all = df[["x", "y", "z"]].to_numpy(dtype=np.float64)
    sample_ids = df["id"].astype(str).tolist()

    fold_ids = np.asarray([stable_fold_id(sid, N_FOLDS) for sid in sample_ids])
    val_mask = fold_ids == FOLD_VAL
    return (
        cands_all[val_mask],
        corrected_all[val_mask],
        truth_all[val_mask],
        [sid for sid, m in zip(sample_ids, val_mask) if m],
    )


def compute_per_band_hit(
    raw_pos: np.ndarray, corrected_pos: np.ndarray, truth_pos: np.ndarray
) -> dict:
    """plan-005 corrector_decomp schema: per-band hit_before / hit_after / delta."""
    err_raw = np.linalg.norm(raw_pos - truth_pos, axis=1)
    err_corrected = np.linalg.norm(corrected_pos - truth_pos, axis=1)
    raw_hit = err_raw <= R_HIT
    corrected_hit = err_corrected <= R_HIT
    band_table = {}
    for name, (lo, hi) in zip(BAND_NAMES, BAND_EDGES_M):
        mask = (err_raw >= lo) & (err_raw < hi)
        n = int(mask.sum())
        hb = float(raw_hit[mask].mean()) if n else 0.0
        ha = float(corrected_hit[mask].mean()) if n else 0.0
        band_table[name] = {"n": n, "hit_before": hb, "hit_after": ha, "delta": float(ha - hb)}
    oof_soft_hit = float(corrected_hit.mean())
    oof_raw_hit = float(raw_hit.mean())
    return {
        "oof_soft_hit": oof_soft_hit,
        "oof_raw_hit": oof_raw_hit,
        "corrector_gain": float(oof_soft_hit - oof_raw_hit),
        "per_band_hit_after": band_table,
    }


def run_L0_anchor(out_dir: Path, corrected_oof_path: Path, labels_path: Path) -> dict:
    """P1.L0 (anchor) — 재학습 X, plan-005 npz 그대로 fold-0 val OOF 추출."""
    val_raw, val_corrected, val_truth, val_ids = load_anchor_data(corrected_oof_path, labels_path)
    metrics = compute_per_band_hit(val_raw, val_corrected, val_truth)

    sub_dir = out_dir / "sub_L0"
    sub_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        sub_dir / "boundary_val_predictions.npz",
        raw_pos=val_raw.astype(np.float32),
        corrected_pos=val_corrected.astype(np.float32),
        truth=val_truth.astype(np.float32),
        sample_ids=np.array(val_ids),
    )

    report = {
        "sub_exp": "P1.L0",
        "config": "anchor (plan-004 default: MSE + far=0.04 + easy=0.20 + env=0.05 + apply_scale=0.75 + boundary [0.7, 1.7cm])",
        "n_val": int(len(val_truth)),
        "fold": FOLD_VAL,
        "training": "X (anchor reuse from plan-005 corrected_oof.npz)",
        **metrics,
        "delta_vs_anchor": 0.0,  # L0 = anchor, by definition
        "delta_vs_z1": None,     # L1 학습 후 계산 (c5)
        "elapsed_sec": 0,
        "generated_at": datetime.now(KST).isoformat(),
    }
    (sub_dir / "report_sub_L0.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def write_summary(out_dir: Path, anchor_dir: Path, reports: dict[str, dict]) -> None:
    """8 sub-exp 통합 summary — c4 시점에는 L0 만 박제."""
    summary = {
        "phase": "Phase 1.L Loss axis ablation",
        "n_folds": 1,
        "fold": FOLD_VAL,
        "anchor": "P1.L0",
        "anchor_oof_soft_hit": reports["P1.L0"]["oof_soft_hit"],
        "sub_exps": reports,
        "axis_positive": None,    # c5 (L1~L7) 완료 후 판정
        "best_lever": None,
        "generated_at": datetime.now(KST).isoformat(),
        "notes": [
            "c4 commit = L0 anchor only (재학습 X, plan-005 npz reuse).",
            "L1~L7 학습 = c5 commit 에서 진행 (corrector_redesign_v2.py components).",
            "axis-level aggregation = max(ΔOOF_i for sub_exp_i ≠ anchor) ≥ 0.005 (§9.2 spec).",
        ],
    }
    (anchor_dir / "phase1_loss_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )


def main():
    parser = argparse.ArgumentParser(description="plan-011 c4 Phase 1.L wrapper")
    parser.add_argument(
        "--corrected-oof",
        type=Path,
        default=REPO / "analysis/plan-005/corrected_oof.npz",
    )
    parser.add_argument("--labels", type=Path, default=REPO / "data/train_labels.csv")
    parser.add_argument(
        "--out-dir", type=Path, default=REPO / "runs/baseline/H011_phase1-loss-ablation"
    )
    parser.add_argument(
        "--summary-dir", type=Path, default=REPO / "analysis/plan-011"
    )
    parser.add_argument(
        "--sub-exp",
        type=str,
        default="L0",
        choices=["L0", "L1", "L2", "L3", "L4", "L5", "L6", "L7", "all"],
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    reports: dict[str, dict] = {}

    if args.sub_exp in ("L0", "all"):
        print(f"[L0] running anchor (plan-005 reuse, fold={FOLD_VAL})...")
        r = run_L0_anchor(args.out_dir, args.corrected_oof, args.labels)
        reports["P1.L0"] = r
        print(
            f"[L0] n_val={r['n_val']}, oof_soft_hit={r['oof_soft_hit']:.4f}, "
            f"raw_hit={r['oof_raw_hit']:.4f}, corrector_gain={r['corrector_gain']:.4f}"
        )

    if args.sub_exp in ("L1", "L2", "L3", "L4", "L5", "L6", "L7", "all"):
        # c5 에서 학습 wrapper 추가. 본 c4 commit 에서는 stub.
        print(
            f"[{args.sub_exp}] 학습 필요 — c5 commit (별도) 에서 corrector_redesign_v2.py + train loop 진행."
        )

    if reports:
        write_summary(args.out_dir, args.summary_dir, reports)
        print(f"✓ summary written → {args.summary_dir}/phase1_loss_summary.json")


if __name__ == "__main__":
    main()
