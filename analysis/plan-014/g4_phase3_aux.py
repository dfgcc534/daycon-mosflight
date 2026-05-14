"""plan-014 c8 (STAGE 4, G4) — Phase 3 aux ablation 3 (E6/E7/E8).

G2 winner + G3 anchor (E2b/E3c/E4a/E5b) 위 *지정 axis 1 변경*. fold=0.
F0 frozen 공통. informational mode (G3 g3_marginal_only warn carry).

E6 boundary sample weighting (on/off, sw=where(boundary_mask, 3.0, 1.0))
E7 scorer arch (BiGRU h=128 vs LastStep MLP)
E8 r=0 logit prior (0/+0.5/+1.0, inference-only)

Usage:
    python analysis/plan-014/g4_phase3_aux.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.io import load_all_samples, load_labels  # noqa: E402
from src.pb_0_6822 import plan014_paradigm as pp  # noqa: E402


def fold0_split(ids, X, Y):
    fold_of = np.array([pp.stable_hash_fold(s) for s in ids])
    val_mask = fold_of == 0
    train_mask = ~val_mask
    return X[train_mask], Y[train_mask], X[val_mask], Y[val_mask]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-json", type=Path, default=Path("analysis/plan-014/g4_phase3.json"))
    ap.add_argument("--epochs", type=int, default=pp.DEFAULT_EPOCHS)
    ap.add_argument("--patience", type=int, default=pp.DEFAULT_PATIENCE)
    args = ap.parse_args()

    t_start = time.time()
    print("[plan-014 G4 v4] loading data ...", flush=True)
    ids, X = load_all_samples("train")
    label_ids, Y = load_labels()
    assert ids == label_ids
    X = X.astype(np.float32); Y = Y.astype(np.float32)
    X_tr, Y_tr, X_va, Y_va = fold0_split(ids, X, Y)

    g2 = json.loads(Path("analysis/plan-014/g2_phase1.json").read_text())
    g3 = json.loads(Path("analysis/plan-014/g3_phase2.json").read_text())
    winner_codebook = g2["winner_codebook"]
    print(f"[plan-014 G4] G2 winner = {g2['winner_id']} ({winner_codebook}). G3 anchor (E2b/E3c/E4a/E5b)", flush=True)

    f0_function = pp.Plan014F0Function()

    anchor_cfg = pp.TrainConfig(
        name="anchor", K=7, encoder_name="bigru", codebook=winner_codebook,
        use_reg_head=True, use_hinge=False,
        temperature=0.03, r0_logit_prior=0.0, boundary_weight_on=False,
        lr=pp.DEFAULT_LR, batch_size=pp.DEFAULT_BATCH,
        epochs=args.epochs, patience=args.patience, seed=pp.DEFAULT_SEED,
    )

    # anchor reuse (G3 anchor 의 OOF carry — 재학습 안 함, G3 결과 직접 reuse 가능)
    anchor_oof = g3["anchor_oof_fold0"]
    print(f"\n[plan-014 G4] anchor (G3 carry): val_hit={anchor_oof:.4f}", flush=True)

    def run_sub_exp(sub_id: str, cfg: pp.TrainConfig):
        t = time.time()
        res = pp.train_one_fold(
            cfg, fold_id=0,
            X_train=X_tr, Y_train=Y_tr,
            X_val=X_va, Y_val=Y_va,
            f0_function=f0_function,
        )
        elapsed = time.time() - t
        return res, elapsed

    results_per_sub_exp: list[dict] = []

    # E6 boundary weight (E6b = on)
    print(f"\n[plan-014 G4] === E6b (boundary_weight_on) ===", flush=True)
    cfg_e6b = replace(anchor_cfg, name="E6b", boundary_weight_on=True)
    res, elapsed = run_sub_exp("E6b", cfg_e6b)
    sub_oof = res["best_val_hit"]
    delta = sub_oof - anchor_oof
    print(f"  E6b: val_hit={sub_oof:.4f} (Δ={delta:+.4f}) dcm={res['dcm']:.4f} "
          f"epoch={res['best_epoch']} elapsed={elapsed:.1f}s", flush=True)
    results_per_sub_exp.append({
        "sub_exp_id": "E6b", "name": "boundary_weight_on", "axis": "E6",
        "val_hit": sub_oof, "dcm": res["dcm"], "delta_oof_vs_anchor": delta,
        "best_epoch": res["best_epoch"], "elapsed_seconds": elapsed,
    })

    # E7 scorer arch (E7b = LastStep MLP)
    print(f"\n[plan-014 G4] === E7b (laststep_mlp) ===", flush=True)
    cfg_e7b = replace(anchor_cfg, name="E7b", encoder_name="laststep_mlp")
    res, elapsed = run_sub_exp("E7b", cfg_e7b)
    sub_oof = res["best_val_hit"]
    delta = sub_oof - anchor_oof
    print(f"  E7b: val_hit={sub_oof:.4f} (Δ={delta:+.4f}) dcm={res['dcm']:.4f} "
          f"epoch={res['best_epoch']} elapsed={elapsed:.1f}s", flush=True)
    results_per_sub_exp.append({
        "sub_exp_id": "E7b", "name": "laststep_mlp", "axis": "E7",
        "val_hit": sub_oof, "dcm": res["dcm"], "delta_oof_vs_anchor": delta,
        "best_epoch": res["best_epoch"], "elapsed_seconds": elapsed,
    })

    # E8 r=0 logit prior (E8b 0.5, E8c 1.0). inference-only — anchor 재학습 후 prior 변경 eval.
    print(f"\n[plan-014 G4] === E8 r=0 logit prior (inference-only) ===", flush=True)
    for sub_id_var, prior in [("E8b", 0.5), ("E8c", 1.0)]:
        cfg_e8 = replace(anchor_cfg, name=sub_id_var, r0_logit_prior=prior)
        # 학습 = anchor (prior=0), eval = prior 변경 — but our train_one_fold 은 r0_logit_prior 를
        # forward/hybrid_predict 시 동적 사용하므로 cfg 에 박은 채로 학습+eval 동일. 단,
        # learning loss 는 prior 무관 (use_reg_head 만 영향). 학습 trajectory 가 prior 무관함을
        # 보장하려면 anchor checkpoint 직접 reuse 필요 — 일단 cfg 학습 + best_val_hit 보고.
        res, elapsed = run_sub_exp(sub_id_var, cfg_e8)
        sub_oof = res["best_val_hit"]
        delta = sub_oof - anchor_oof
        print(f"  {sub_id_var} prior={prior}: val_hit={sub_oof:.4f} (Δ={delta:+.4f}) "
              f"dcm={res['dcm']:.4f} epoch={res['best_epoch']} elapsed={elapsed:.1f}s", flush=True)
        results_per_sub_exp.append({
            "sub_exp_id": sub_id_var, "name": f"r0_prior_{prior}", "axis": "E8",
            "r0_prior": prior, "val_hit": sub_oof, "dcm": res["dcm"],
            "delta_oof_vs_anchor": delta, "best_epoch": res["best_epoch"],
            "elapsed_seconds": elapsed,
        })

    # axis summary
    print(f"\n[plan-014 G4] === axis summary ===", flush=True)
    axis_summary: dict[str, dict] = {}
    for axis in ["E6", "E7", "E8"]:
        axis_subs = [r for r in results_per_sub_exp if r.get("axis") == axis and "val_hit" in r]
        if not axis_subs:
            axis_summary[axis] = {"n_sub_exp": 0, "skipped": True}
            continue
        deltas = {r["sub_exp_id"]: r["delta_oof_vs_anchor"] for r in axis_subs}
        max_delta = max(deltas.values())
        best_sub_id = max(deltas, key=deltas.get)
        positive = max_delta > 0
        axis_summary[axis] = {
            "n_sub_exp": len(axis_subs),
            "deltas": deltas,
            "max_delta": max_delta,
            "best_sub_id": best_sub_id,
            "best_val_hit": next(r["val_hit"] for r in axis_subs if r["sub_exp_id"] == best_sub_id),
            "positive_lever": positive,
        }
        print(f"  {axis}: n={len(axis_subs)}, max_delta={max_delta:+.4f}, "
              f"best={best_sub_id}, positive={positive}", flush=True)

    positive_axes = [a for a, info in axis_summary.items() if info.get("positive_lever")]

    elapsed_total = time.time() - t_start
    artifact = {
        "exp_id": "H040_g4_phase3_aux3",
        "winner_id": g2["winner_id"],
        "anchor_oof_fold0": anchor_oof,
        "n_sub_exp": len([r for r in results_per_sub_exp if "val_hit" in r]),
        "axis_summary": axis_summary,
        "positive_axes": positive_axes,
        "G4_passed": True,  # informational only
        "G4_mode": "informational" + (" (G3 g3_marginal_only carry)" if not g3.get("G3_passed") else ""),
        "results_per_sub_exp": results_per_sub_exp,
        "elapsed_total_seconds": elapsed_total,
        "plan_version": "v4.5",
        "f0_frozen_baseline": "plan-006_frenet_par120_perp_neg020",
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(artifact, indent=2, ensure_ascii=False))

    print(f"\n[plan-014 G4] === final ===", flush=True)
    print(f"  positive_axes={positive_axes}, mode=informational", flush=True)
    print(f"  elapsed_total={elapsed_total:.1f}s ({elapsed_total/60:.2f} min)", flush=True)
    print(f"[plan-014 G4] artifact -> {args.out_json}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
