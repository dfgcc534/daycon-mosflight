"""plan-012 c11~c14 (G3) — Phase 3 Aux Ablation on winner E0a (3 axis × 4 additional sub-exp).

E6 Boundary sample weighting: 1 추가 sub-exp (1cm 근처 sample × 3)
E7 Scorer arch:               1 추가 sub-exp (last-step MLP)
E8 r=0 logit prior:           2 추가 sub-exp (+0.5, +1.0)
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

from src.pb_0_6822 import ring_classifier as rc                       # noqa: E402
from src.pb_0_6822 import ring_classifier_train as rct                # noqa: E402
from src.pb_0_6822 import selector as base                            # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="data")
    parser.add_argument("--winner", type=str, default="analysis/plan-012/phase1_winner.json")
    parser.add_argument("--out", type=str, default="analysis/plan-012/phase3_results.json")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--val-fold", type=int, default=0)
    parser.add_argument("--n-folds", type=int, default=5)
    args = parser.parse_args()

    with open(args.winner) as f:
        winner_data = json.load(f)
    anchor_oof = winner_data["winner_oof"]
    print(f"[phase3] winner anchor OOF = {anchor_oof:.4f}", flush=True)

    # data
    root = Path(args.root)
    train_ids, train_y = base.read_labels(root / "train_labels.csv")
    train_x = base.load_stack(root / "train", train_ids).astype(np.float64)
    train_y = train_y.astype(np.float64)
    N, T, _ = train_x.shape
    end_idx = T - 1
    fold_id = np.array([base.stable_fold_id(sid, args.n_folds) for sid in train_ids], dtype=np.int64)

    F0_pred = rc.f0_predict_frenet_par120_perp_neg020(train_x, end_idx=end_idx)
    R_wfn = rc.build_frenet_basis_3d(train_x, end_idx=end_idx)
    seq_feat = base.make_seq_features(train_x, end_idx=end_idx, direction=1.0).astype(np.float32)
    anchors_abs = rc.compute_anchors_absolute()

    # E6 boundary weight 산출
    err_F0 = np.linalg.norm(F0_pred - train_y, axis=-1)
    boundary_mask = (err_F0 > 0.005) & (err_F0 < 0.015)
    sw_e6 = np.where(boundary_mask, 3.0, 1.0).astype(np.float32)
    print(f"[phase3] E6 boundary 샘플: {int(boundary_mask.sum())}/{N} (weight=3)", flush=True)

    # sub-exp matrix (4 additional)
    sub_exps = [
        # E6: boundary weight on
        {"id": "E6.bweight_on", "axis": "E6_boundary_weight", "sample_weight": sw_e6,
         "temperature": 0.03, "r0_logit_prior": 0.0, "scorer_arch": "attn_gru"},
        # E7: last-step MLP
        {"id": "E7.last_step_mlp", "axis": "E7_scorer_arch", "sample_weight": None,
         "temperature": 0.03, "r0_logit_prior": 0.0, "scorer_arch": "last_step_mlp"},
        # E8: r=0 prior +0.5
        {"id": "E8.r0_plus0_5", "axis": "E8_r0_prior", "sample_weight": None,
         "temperature": 0.03, "r0_logit_prior": 0.5, "scorer_arch": "attn_gru"},
        # E8: r=0 prior +1.0
        {"id": "E8.r0_plus1_0", "axis": "E8_r0_prior", "sample_weight": None,
         "temperature": 0.03, "r0_logit_prior": 1.0, "scorer_arch": "attn_gru"},
    ]

    results: list[dict] = []
    t0 = time.time()
    for spec in sub_exps:
        ts = time.time()
        r = rct.run_sub_exp(
            sub_exp_id=spec["id"],
            codebook_id="absolute",
            anchors_local_per_fold=None,
            anchors_local_static=anchors_abs,
            R_wfn=None,
            F0_pred=F0_pred,
            train_y=train_y,
            seq_feat=seq_feat,
            fold_id=fold_id,
            val_fold=args.val_fold,
            epochs=args.epochs,
            batch_size=args.batch_size,
            patience=args.patience,
            temperature=spec["temperature"],
            r0_logit_prior=spec["r0_logit_prior"],
            sample_weight=spec.get("sample_weight"),
            scorer_arch=spec["scorer_arch"],
        )
        r["axis"] = spec["axis"]
        r["r0_logit_prior"] = spec["r0_logit_prior"]
        r["sample_weight_used"] = spec["sample_weight"] is not None
        r["scorer_arch"] = spec["scorer_arch"]
        r["delta_oof_vs_anchor"] = r["best_val_hit"] - anchor_oof
        r["elapsed_seconds"] = round(time.time() - ts, 1)
        results.append(r)
        print(f"[{spec['id']}] best={r['best_val_hit']:.4f} ΔOOF={r['delta_oof_vs_anchor']:+.4f} DCM={r['best_dcm']:.5f} ({r['elapsed_seconds']}s)", flush=True)

    # axis summary
    axis_summary = {}
    for axis in ["E6_boundary_weight", "E7_scorer_arch", "E8_r0_prior"]:
        subs = [r for r in results if r["axis"] == axis]
        if not subs:
            continue
        deltas = [r["delta_oof_vs_anchor"] for r in subs]
        best_sub = max(subs, key=lambda r: r["delta_oof_vs_anchor"])
        axis_summary[axis] = {
            "n_sub_exp": len(subs),
            "deltas": {r["sub_exp_id"]: r["delta_oof_vs_anchor"] for r in subs},
            "max_delta": max(deltas),
            "best_sub_id": best_sub["sub_exp_id"],
            "best_val_hit": best_sub["best_val_hit"],
            "positive_lever": max(deltas) >= 0.005,
        }

    positive_axes = [ax for ax, summ in axis_summary.items() if summ.get("positive_lever")]
    g3_passed = True  # G3 = informational only

    out = {
        "exp_ids": ["H026_phase3-boundary-weight", "H027_phase3-scorer-arch", "H028_phase3-r0-prior"],
        "winner_id": winner_data["winner_id"],
        "anchor_oof": anchor_oof,
        "n_sub_exp": len(sub_exps),
        "axis_summary": axis_summary,
        "results_per_sub_exp": results,
        "G3_passed": g3_passed,
        "positive_axes": positive_axes,
        "elapsed_total_seconds": round(time.time() - t0, 1),
    }
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"[phase3] wrote {args.out}", flush=True)
    print(f"[phase3] G3 informational complete (positive axes: {positive_axes})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
