"""plan-032 c4 — Ablation C (fine-distill / label smoothing).

plan-031 train.py 의 fine phase soft CE 의 label 만 변경:
  q_smooth = (1 - alpha) * q_orig + alpha / K   (label smoothing α 적용)

simplest variant (PB teacher 없이 cheap):
  C1: α = 0.1 (uniform mix 10%)
  C2: α = 0.15 (uniform mix 15%)

self-distillation (PB teacher 변환 비용 회피).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from math import sqrt
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_THIS = Path(__file__).resolve().parent
_REPO = _THIS.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_train_mod = _load(_THIS.parent / "plan-031" / "train.py", "p032_c_train")


VARIANTS = {
    "C1": 0.10,
    "C2": 0.15,
}


def _train_one_fold_smoothed(label_smooth_alpha: float, fold: int,
                              X_tr, X_te, gt_tr, verbose=False):
    """plan-031 train_one_fold carry + fine phase 의 q_b → label smoothed.

    이전 fold inline 호출 (train_one_fold 재정의 X) — fine phase 의 loss 만 patch.
    가장 쉬운 방법: train_one_fold 함수 자체를 monkey-patch 한 wrapper.
    """
    # plan-031 train_one_fold 의 fine phase 안 q_b 만 mixing.
    # cleanest = train_one_fold 의 사본을 만들어서 label smoothing 적용.
    # 더 cheap = run_5fold_oof 호출 전 build_soft_label_with_tau 결과를 후처리.

    # === inline copy of train_one_fold with smoothing ===
    M = _train_mod
    t0 = time.perf_counter()

    R_wfn_tr = M._p021_build.build_frenet_basis_3d(X_tr, end_idx=10).astype(np.float32)
    R_wfn_te = M._p021_build.build_frenet_basis_3d(X_te, end_idx=10).astype(np.float32)
    F0_tr = M._bf.f0_baseline(X_tr, end_idx=10).astype(np.float32)
    F0_te = M._bf.f0_baseline(X_te, end_idx=10).astype(np.float32)
    qc = M._quantile_mod.build(X_tr, R_wfn_tr)
    from src.pb_0_6822.selector import fit_regime_bins, assign_regimes
    regime_bins_tr = fit_regime_bins(X_tr, end_idx=10)
    regimes_tr = assign_regimes(X_tr, end_idx=10, bins=regime_bins_tr)
    regimes_te = assign_regimes(X_te, end_idx=10, bins=regime_bins_tr)
    regime_anchor_table_tr = M._aqe.build_regime_anchor_lookup(
        gt_train=gt_tr, regimes_train=regimes_tr, ANCHORS_A6=M._p022_anchors.ANCHORS_A6,
        R_wfn_train=R_wfn_tr, F0_train=F0_tr,
        regime_count=M.REGIME_COUNT, laplace=1.0,
    )
    cand_ext_tr = M._aqe.build(
        X_tr, R_wfn_tr, F0_tr, M._p022_anchors.ANCHORS_A6, M._bf.f0_baseline,
        regimes=regimes_tr, quantile_carry=qc,
        regime_count=M.REGIME_COUNT, regime_anchor_table=regime_anchor_table_tr,
    )
    cand_ext_te = M._aqe.build(
        X_te, R_wfn_te, F0_te, M._p022_anchors.ANCHORS_A6, M._bf.f0_baseline,
        regimes=regimes_te, quantile_carry=qc,
        regime_count=M.REGIME_COUNT, regime_anchor_table=regime_anchor_table_tr,
    )
    seq_tr = M._seq_mod.build(X_tr, R_wfn_tr, M._p022_anchors.ANCHORS_A6, M._bf.f0_baseline, qc)
    seq_te = M._seq_mod.build(X_te, R_wfn_te, M._p022_anchors.ANCHORS_A6, M._bf.f0_baseline, qc)
    art_tr = M._build_per_sample_artifacts(X_tr, R_wfn_tr, F0_tr, cand_ext_tr)
    art_te = M._build_per_sample_artifacts(X_te, R_wfn_te, F0_te, cand_ext_te)
    seq97_tr = np.concatenate([seq_tr, art_tr["residual_a_gru"]], axis=-1).astype(np.float32)
    seq97_te = np.concatenate([seq_te, art_te["residual_a_gru"]], axis=-1).astype(np.float32)

    q_tr = M._p022_soft.build_soft_label_with_tau(
        gt_tr, R_wfn_tr, F0_tr, M._p022_anchors.ANCHORS_A6, tau_cls=M.TAU_CLS,
    )
    # === label smoothing (axis C) ===
    K = M.K_ANCHORS
    q_tr = (1.0 - label_smooth_alpha) * q_tr + label_smooth_alpha / K
    # row sum 자동 유지 (q_orig row sum=1, uniform/K sum=1)

    R_t_tr = np.transpose(R_wfn_tr, (0, 2, 1))
    gt_res_f = np.einsum("nij,nj->ni", R_t_tr, gt_tr - F0_tr)
    gt_anchor_idx_tr = np.linalg.norm(
        M._p022_anchors.ANCHORS_A6[None, :, :] - gt_res_f[:, None, :], axis=-1,
    ).argmin(axis=1).astype(np.int64)
    class_prior_global = M._prior_mod.build_class_prior_global(gt_anchor_idx_tr, K=K)
    regime_anchor_prior_tr = M._prior_mod.build_regime_anchor_prior(regime_anchor_table_tr, regimes_tr)

    torch.manual_seed(M.SEED + fold)
    anchors_torch = torch.from_numpy(M._p022_anchors.ANCHORS_A6.astype(np.float32))
    model = M._model_mod.GRUNetX3(
        seq_dim=M.SEQ_PLUS_RES_DIM, query_dim=M.QUERY_DIM,
        head_summary_dim=M.HEAD_SUMMARY_DIM, slim7_dim=M.SLIM7_DIM,
        hidden=M.HIDDEN, attn_dim=M.ATTN_DIM, head_hidden=M.HEAD_HIDDEN,
        gru_dropout=M.GRU_DROPOUT, head_dropout=M.HEAD_DROPOUT,
        K=K, residual_a_kv_dim=M.RESIDUAL_A_KV_DIM,
        anchors=anchors_torch,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=M.PRE_LR, weight_decay=M.WEIGHT_DECAY)
    scheduler_pre = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[
            torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-6, end_factor=1.0, total_iters=M.WARMUP_EP),
            torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=M.PRE_EPOCHS - M.WARMUP_EP),
        ],
        milestones=[M.WARMUP_EP],
    )

    N_tr = X_tr.shape[0]
    for epoch in range(M.PRE_EPOCHS):
        model.train()
        rng_ep = np.random.default_rng(M.SEED + epoch * 1000 + fold)
        perm = rng_ep.permutation(N_tr)
        for b_start in range(0, N_tr, M.BATCH_SIZE):
            idx = perm[b_start : b_start + M.BATCH_SIZE]
            seq_b = M._to_torch(seq97_tr[idx])
            res_kv_b = M._to_torch(art_tr["residual_a_kv"][idx])
            q64_b = M._to_torch(art_tr["query_64"][idx])
            hs_b = M._to_torch(art_tr["head_summary_51"][idx])
            slim7_b = M._to_torch(art_tr["slim7"][idx])
            F0_b = M._to_torch(F0_tr[idx])
            R_b = M._to_torch(R_wfn_tr[idx])
            q_b = torch.from_numpy(q_tr[idx]).float()
            optimizer.zero_grad()
            _wp, probs = model(seq_b, res_kv_b, q64_b, hs_b, slim7_b, F0_b, R_b)
            log_probs = torch.log(probs.clamp_min(1e-12))
            loss = -(q_b * log_probs).sum(dim=-1).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), M.GRAD_CLIP)
            optimizer.step()
        scheduler_pre.step()

    for g in optimizer.param_groups:
        g["lr"] = M.FINE_LR
        g["initial_lr"] = M.FINE_LR
    scheduler_fine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=M.FINE_EPOCHS)

    def _forward_score(model, seq_b, res_kv_b, q64_b, hs_b, slim7_b, F0_b, R_b):
        B = seq_b.shape[0]; Kk = model.K
        gru_out, _ = model.gru(seq_b)
        gru_hidden_last = gru_out[:, -1, :]
        kv_raw = torch.cat([gru_out, res_kv_b], dim=-1)
        kv = model.kv_proj(kv_raw)
        q = model.q_proj(q64_b)
        attn_logits = torch.einsum("bka,bta->bkt", q, kv) / sqrt(model.attn_dim)
        attn_w = torch.softmax(attn_logits, dim=-1)
        attn_context = torch.einsum("bkt,bta->bka", attn_w, kv)
        sample_summary_full = torch.cat([gru_hidden_last, hs_b], dim=-1)
        sample_bias = sample_summary_full.unsqueeze(1).expand(-1, Kk, -1)
        head_in = torch.cat([attn_context, sample_bias, slim7_b], dim=-1)
        score = model.head_mlp(head_in).squeeze(-1)
        probs = torch.softmax(score, dim=-1)
        residual_frenet = probs @ model.ANCHORS_A6
        residual_world = torch.einsum("bij,bj->bi", R_b, residual_frenet)
        world_pred = F0_b + residual_world
        return world_pred, probs, score

    class_prior_t = torch.from_numpy(class_prior_global).float()
    for epoch in range(M.FINE_EPOCHS):
        model.train()
        rng_ep = np.random.default_rng(M.SEED + (M.PRE_EPOCHS + epoch) * 1000 + fold)
        perm = rng_ep.permutation(N_tr)
        for b_start in range(0, N_tr, M.BATCH_SIZE):
            idx = perm[b_start : b_start + M.BATCH_SIZE]
            seq_b = M._to_torch(seq97_tr[idx])
            res_kv_b = M._to_torch(art_tr["residual_a_kv"][idx])
            q64_b = M._to_torch(art_tr["query_64"][idx])
            hs_b = M._to_torch(art_tr["head_summary_51"][idx])
            slim7_b = M._to_torch(art_tr["slim7"][idx])
            F0_b = M._to_torch(F0_tr[idx])
            R_b = M._to_torch(R_wfn_tr[idx])
            q_b = torch.from_numpy(q_tr[idx]).float()
            gt_idx_b = torch.from_numpy(gt_anchor_idx_tr[idx]).long()
            reg_prior_b = torch.from_numpy(regime_anchor_prior_tr[idx]).float()

            optimizer.zero_grad()
            _wp, probs, score = _forward_score(model, seq_b, res_kv_b, q64_b, hs_b, slim7_b, F0_b, R_b)
            log_probs = torch.log(probs.clamp_min(1e-12))
            loss_soft = -(q_b * log_probs).sum(dim=-1).mean()
            loss_pair = M._pairwise_mod.pairwise_margin_loss(score, gt_idx_b, margin=M.PAIRWISE_MARGIN)
            loss_prior = M._prior_mod.regime_class_prior_loss(
                score, reg_prior_b, class_prior_t,
                regime_strength=M.REGIME_STRENGTH, class_strength=M.CLASS_STRENGTH,
            )
            loss = M.W_SOFT_CE * loss_soft + M.W_PAIRWISE * loss_pair + M.W_PRIOR * loss_prior
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), M.GRAD_CLIP)
            optimizer.step()
        scheduler_fine.step()

    model.eval()
    pred_te_list = []
    probs_te_list = []
    with torch.no_grad():
        for e_start in range(0, X_te.shape[0], M.EVAL_BATCH):
            sl = slice(e_start, e_start + M.EVAL_BATCH)
            seq_b = M._to_torch(seq97_te[sl])
            res_kv_b = M._to_torch(art_te["residual_a_kv"][sl])
            q64_b = M._to_torch(art_te["query_64"][sl])
            hs_b = M._to_torch(art_te["head_summary_51"][sl])
            slim7_b = M._to_torch(art_te["slim7"][sl])
            F0_b = M._to_torch(F0_te[sl])
            R_b = M._to_torch(R_wfn_te[sl])
            world_pred, probs = model(seq_b, res_kv_b, q64_b, hs_b, slim7_b, F0_b, R_b)
            pred_te_list.append(world_pred.cpu().numpy())
            probs_te_list.append(probs.cpu().numpy())
    pred_te = np.concatenate(pred_te_list, axis=0).astype(np.float32)
    probs_te = np.concatenate(probs_te_list, axis=0).astype(np.float32)
    t_fold = time.perf_counter() - t0
    return {
        "fold": fold, "world_pred_te": pred_te, "probs_te": probs_te,
        "R_wfn_te": R_wfn_te, "F0_te": F0_te, "elapsed_s": t_fold,
    }


def run_5fold_oof_smoothed(label_smooth_alpha: float, verbose: bool = False) -> dict:
    t_total = time.perf_counter()
    from src.io import load_all_samples, load_labels
    from src.pb_0_6822.selector import stable_fold_id

    _ids, X = load_all_samples()
    _ids_gt, gt = load_labels()
    assert _ids == _ids_gt
    X = X.astype(np.float32); gt = gt.astype(np.float32)
    N_total = X.shape[0]
    folds = np.asarray([stable_fold_id(str(sid), _train_mod.N_FOLDS) for sid in _ids], dtype=int)

    K = _train_mod.K_ANCHORS
    oof_pred = np.zeros((N_total, 3), dtype=np.float32)
    oof_probs = np.zeros((N_total, K), dtype=np.float32)
    F0_all = np.zeros((N_total, 3), dtype=np.float32)
    R_all = np.zeros((N_total, 3, 3), dtype=np.float32)

    fold_logs = []
    for fold in range(_train_mod.N_FOLDS):
        train_idx = np.where(folds != fold)[0]
        test_idx = np.where(folds == fold)[0]
        X_tr, X_te = X[train_idx], X[test_idx]
        gt_tr = gt[train_idx]
        log = _train_one_fold_smoothed(label_smooth_alpha, fold, X_tr, X_te, gt_tr, verbose)
        oof_pred[test_idx] = log["world_pred_te"]
        oof_probs[test_idx] = log["probs_te"]
        F0_all[test_idx] = log["F0_te"]
        R_all[test_idx] = log["R_wfn_te"]
        fold_logs.append({"fold": fold, "elapsed_s": log["elapsed_s"], "N_te": int(len(test_idx))})

    err = np.linalg.norm(oof_pred - gt, axis=1)
    hit_1cm = float((err <= 0.01).mean())
    hit_1p5 = float((err <= 0.015).mean())
    top1 = oof_probs.argmax(1)
    mcr = float(np.bincount(top1, minlength=K).max() / N_total)
    R_t = np.transpose(R_all, (0, 2, 1))
    gt_res_f = np.einsum("nij,nj->ni", R_t, gt - F0_all)
    gt_label = np.linalg.norm(
        _train_mod._p022_anchors.ANCHORS_A6[None, :, :] - gt_res_f[:, None, :], axis=-1
    ).argmin(1)
    top1_acc = float((top1 == gt_label).mean())

    return {
        "hit_1cm": hit_1cm, "hit_1p5cm": hit_1p5,
        "max_class_ratio": mcr, "top1_acc": top1_acc,
        "elapsed_total_s": time.perf_counter() - t_total,
        "fold_logs": fold_logs,
        "label_smooth_alpha": label_smooth_alpha,
        "oof_pred": oof_pred, "oof_probs": oof_probs,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=list(VARIANTS.keys()))
    args = ap.parse_args()

    alpha = VARIANTS[args.variant]
    result = run_5fold_oof_smoothed(alpha, verbose=False)
    h = result["hit_1cm"]
    band = ("EXCELLENT" if h >= 0.6624 else
            "PASS" if h >= 0.6511 else
            "STRONG" if h >= 0.6387 else
            "BORDERLINE" if h >= 0.6320 else
            "FAIL_regression")
    result.update({"ablation_axis": "C", "variant": args.variant,
                   "delta_vs_plan_031": h - 0.6397, "band": band})

    out_path = _THIS / f"results_{args.variant}.json"
    arrays = {k: result.pop(k) for k in ("oof_pred", "oof_probs") if k in result}
    with out_path.open("w") as f:
        json.dump(result, f, indent=2)
    if arrays:
        np.savez_compressed(out_path.with_suffix(".npz"), **arrays)
    print(json.dumps({k: v for k, v in result.items() if k not in ("fold_logs",)}, indent=2))
    print(f"[done] {args.variant} (α={alpha}): hit_1cm={h:.4f}, Δ={h-0.6397:+.4f}, band={band}")


if __name__ == "__main__":
    main()
