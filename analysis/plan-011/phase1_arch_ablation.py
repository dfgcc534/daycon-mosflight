"""plan-011 c8 — Phase 1.M Architecture axis ablation (7 sub-exp + M0 anchor).

7 sub-exp 학습 (fixed L0-style training: huber + weight schedule, In-A anchor, F0 anchor):
  - M0 (anchor): RedesignedCorrectionNet depth=2 hidden=64
  - M1: GateHeadCorrector (gate head only, no asymmetric loss in L0)
  - M2: SplitHeadCorrector (direction + magnitude)
  - M3: BinClassifierCorrector (3-axis factorized bin_dim=60)
  - M4: IterativeRefinementCorrector (3-step, per_step_cap=3mm)
  - M5: GMMCorrector (μ + diagonal σ NLL)
  - M6: WiderShallowCorrector (depth=1, hidden=256)
"""
from __future__ import annotations
import argparse
import json
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
import numpy as np
import torch
from torch import nn

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase1_loss_train as base  # type: ignore

from src.pb_0_6822 import selector as sel
from src.pb_0_6822 import corrector_redesign_v2 as v2

REPO = Path(__file__).resolve().parents[2]
KST = timezone(timedelta(hours=9))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

R_HIT = 0.01
HORIZON = 2
FOLD_VAL = 0
SEED_BASE = 20260513


def make_arch(sub_exp: str, dim_cf: int = 32, hidden: int = 64) -> nn.Module:
    """sub_exp 별 corrector arch (M0~M6)."""
    if sub_exp == "M0":
        return v2.RedesignedCorrectionNet(dim_cf=dim_cf, hidden=hidden, dim_encoder=0)
    if sub_exp == "M1":
        return v2.GateHeadCorrector(dim_cf=dim_cf, hidden=hidden, dim_encoder=0)
    if sub_exp == "M2":
        return v2.SplitHeadCorrector(dim_cf=dim_cf, hidden=hidden, dim_encoder=0)
    if sub_exp == "M3":
        return v2.BinClassifierCorrector(dim_cf=dim_cf, hidden=hidden, dim_encoder=0)
    if sub_exp == "M4":
        return v2.IterativeRefinementCorrector(dim_cf=dim_cf, hidden=hidden, dim_encoder=0, n_steps=3, per_step_cap=0.003)
    if sub_exp == "M5":
        return v2.GMMCorrector(dim_cf=dim_cf, hidden=hidden, dim_encoder=0)
    if sub_exp == "M6":
        return v2.WiderShallowCorrector(dim_cf=dim_cf, hidden=256, dim_encoder=0)
    raise ValueError(sub_exp)


def compute_arch_loss(
    sub_exp: str,
    delta: torch.Tensor,
    aux: dict,
    target: torch.Tensor,
    err_raw: torch.Tensor,
    cand: torch.Tensor,
    truth: torch.Tensor,
) -> torch.Tensor:
    """L0-style training: huber + weight schedule. M5 만 별도 GMM NLL."""
    w = base.weight_schedule(err_raw)
    if sub_exp == "M5":
        # M5 = NLL loss; delta = mu, aux["logsigma"]
        return v2.gmm_nll_loss(delta, aux["logsigma"], target)
    if sub_exp == "M4":
        # M4 stage-wise loss: 매 step 의 cand_t err — simplification: 누적 delta 만 huber
        per = v2.huber_loss(delta, target)
        # add stage-wise: 각 step delta 의 partial-target 추정
        for step_delta in aux.get("per_step_deltas", []):
            per = per + v2.huber_loss(step_delta, target / len(aux["per_step_deltas"]))
        return (per * w).sum() / (w.sum() + 1e-8)
    per = v2.huber_loss(delta, target)
    return (per * w).sum() / (w.sum() + 1e-8)


def train_arch_subexp(sub_exp, data_tr, data_va, args):
    t0 = time.time()
    torch.manual_seed(SEED_BASE + 200 + int(sub_exp[1]))
    np.random.seed(SEED_BASE + 200 + int(sub_exp[1]))

    model = make_arch(sub_exp).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    def to_t(d, k): return torch.from_numpy(d[k]).to(DEVICE)
    cf_tr, target_tr, err_tr = to_t(data_tr, "cf"), to_t(data_tr, "target"), to_t(data_tr, "err")
    cand_tr, truth_tr = to_t(data_tr, "cand"), to_t(data_tr, "truth")
    cf_va, target_va, err_va = to_t(data_va, "cf"), to_t(data_va, "target"), to_t(data_va, "err")
    cand_va, truth_va = to_t(data_va, "cand"), to_t(data_va, "truth")

    n_tr = cf_tr.shape[0]
    best_hit, best_state, wait = -1.0, None, 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(n_tr, device=DEVICE)
        total, n = 0.0, 0
        for start in range(0, n_tr, args.batch):
            sel_ = perm[start:start + args.batch]
            cf_b, tg_b, er_b = cf_tr[sel_], target_tr[sel_], err_tr[sel_]
            cd_b, tr_b = cand_tr[sel_], truth_tr[sel_]

            opt.zero_grad(set_to_none=True)
            delta, aux = model(cf_b)
            loss = compute_arch_loss(sub_exp, delta, aux, tg_b, er_b, cd_b, tr_b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            total += float(loss.detach()) * len(cf_b)
            n += len(cf_b)

        model.eval()
        with torch.no_grad():
            delta_va, _ = model(cf_va)
            corrected_pos_va = cand_va + v2.cap_6mm(delta_va)
            err_after = torch.norm(corrected_pos_va - truth_va, dim=1)
            hit = float((err_after <= R_HIT).float().mean())

        if hit > best_hit:
            best_hit, wait = hit, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1

        if epoch % 5 == 0 or wait >= args.patience:
            print(f"  [{sub_exp}] ep{epoch:3d} loss={total/max(n,1):.4f} val_hit={hit:.4f} best={best_hit:.4f} wait={wait}")
        if wait >= args.patience and epoch >= args.min_epochs:
            break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        delta_va, _ = model(cf_va)
        delta_va_np = delta_va.cpu().numpy()
    corrected_pos_va_np = data_va["cand"] + base.cap_6mm_np(delta_va_np)
    err_after = np.linalg.norm(corrected_pos_va_np - data_va["truth"], axis=1)
    err_before = data_va["err"]
    return {
        "sub_exp": f"P1.{sub_exp}",
        "n_val": int(len(err_after)),
        "fold": FOLD_VAL,
        "oof_soft_hit": float((err_after <= R_HIT).mean()),
        "oof_raw_hit": float((err_before <= R_HIT).mean()),
        "corrector_gain": float((err_after <= R_HIT).mean() - (err_before <= R_HIT).mean()),
        "per_band_hit_after": base.compute_per_band(err_before, err_after),
        "elapsed_sec": time.time() - t0,
        "best_val_hit": best_hit,
        "device": str(DEVICE),
        "n_params": sum(p.numel() for p in model.parameters()),
    }, corrected_pos_va_np, delta_va_np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=REPO / "data")
    parser.add_argument("--out-dir", type=Path, default=REPO / "runs/baseline/H013_phase1-arch-ablation")
    parser.add_argument("--summary-dir", type=Path, default=REPO / "analysis/plan-011")
    parser.add_argument("--sub-exps", type=str, default="M0,M1,M2,M3,M4,M5,M6")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--min-epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    data = base.load_data(args.data_root)
    data_tr, data_va = base.split_fold(data)
    print(f"[fold-0] train n={len(data_tr['cf'])}, val n={len(data_va['cf'])}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    sub_exps = [s.strip() for s in args.sub_exps.split(",") if s.strip()]
    reports = {}
    for sub_exp in sub_exps:
        print(f"\n=== {sub_exp} ===")
        report, corrected, delta = train_arch_subexp(sub_exp, data_tr, data_va, args)
        reports[f"P1.{sub_exp}"] = report
        sub_dir = args.out_dir / f"sub_{sub_exp}"
        sub_dir.mkdir(parents=True, exist_ok=True)
        (sub_dir / f"report_sub_{sub_exp}.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
        np.savez(sub_dir / "boundary_val_predictions.npz",
                 corrected_pos=corrected.astype(np.float32), delta=delta.astype(np.float32))
        print(f"  → oof_soft_hit={report['oof_soft_hit']:.4f}, gain={report['corrector_gain']:+.4f}, "
              f"params={report['n_params']}, elapsed={report['elapsed_sec']:.1f}s")

    # summary
    summary_path = args.summary_dir / "phase1_arch_summary.json"
    anchor_oof = reports.get("P1.M0", {}).get("oof_soft_hit")
    for k, r in reports.items():
        if anchor_oof is not None:
            r["delta_vs_anchor"] = r["oof_soft_hit"] - anchor_oof
    non_anchor = [v for k, v in reports.items() if k != "P1.M0" and "delta_vs_anchor" in v]
    max_delta = max((v["delta_vs_anchor"] for v in non_anchor), default=0.0)
    summary = {
        "phase": "Phase 1.M Architecture axis ablation",
        "n_folds": 1, "fold": FOLD_VAL, "anchor": "P1.M0",
        "anchor_oof_soft_hit": anchor_oof,
        "sub_exps": reports,
        "axis_positive_threshold_0p005": bool(max_delta >= 0.005),
        "max_delta_vs_anchor": float(max_delta),
        "best_lever": max(non_anchor, key=lambda v: v["delta_vs_anchor"], default={}).get("sub_exp"),
        "generated_at": datetime.now(KST).isoformat(),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\n✓ summary → {summary_path.relative_to(REPO)}: best={summary['best_lever']}, max_Δ={max_delta:+.4f}")


if __name__ == "__main__":
    main()
