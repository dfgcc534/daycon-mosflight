"""plan-031 c4 — train.py multi-phase (pre + fine).

plan-030 train.py + plan-031 §3.4 multi-phase split:

Phase A — pre (15 epoch):
  - lr = 7e-4, cosine T_max=15 with warmup 5 ep
  - loss = soft CE only (plan-030 carry)
  - 목적: 기본 GRU + attention representation 학습

Phase B — fine (35 epoch):
  - lr = 2e-4 (= 7e-4 / 3.5), cosine T_max=35 from pre last
  - loss = 0.5 * soft_CE + 0.3 * pairwise_margin(0.12) + 0.2 * regime_class_prior(0.65/0.45)
  - optimizer state carry (continue)
  - 목적: logit sharpening + anchor discrimination + prior 보강

total 50 epoch budget (plan-030 동일).
"""
from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_THIS = Path(__file__).resolve().parent
_REPO = _THIS.parent.parent
_PLAN030 = _THIS.parent / "plan-030"

for p in (_REPO, _THIS.parent / "plan-020", _THIS.parent / "plan-021",
          _THIS.parent / "plan-022", _THIS.parent / "plan-024",
          _THIS.parent / "plan-029", _PLAN030, _THIS):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


# carry modules
_bf = _load(_THIS.parent / "plan-020" / "baseline_f0.py", "p031_train_bf")
_p021_build = _load(_THIS.parent / "plan-021" / "build_input.py", "p031_train_bi")
_p022_anchors = _load(_THIS.parent / "plan-022" / "anchors.py", "p031_train_av")
_p022_soft = _load(_THIS.parent / "plan-022" / "selector_only_model.py", "p031_train_soft")
_seq_mod = _load(_THIS.parent / "plan-024" / "seq_builder.py", "p031_train_seq")
_quantile_mod = _load(_THIS.parent / "plan-024" / "quantile_carry.py", "p031_train_qc")
_aqe = _load(_THIS.parent / "plan-029" / "anchor_query_extend.py", "p031_train_aqe")
_residual_mod = _load(_PLAN030 / "residual_builder.py", "p031_train_residual")
_query_mod = _load(_PLAN030 / "query_builder.py", "p031_train_query")
_head_mod = _load(_PLAN030 / "head_summary.py", "p031_train_head")

_model_mod = _load(_THIS / "model.py", "p031_train_model")
_pairwise_mod = _load(_THIS / "pairwise_loss.py", "p031_train_pairwise")
_prior_mod = _load(_THIS / "prior_loss.py", "p031_train_prior")

from src.io import load_all_samples, load_labels  # noqa: E402
from src.pb_0_6822.selector import stable_fold_id, fit_regime_bins, assign_regimes  # noqa: E402

N_FOLDS = 5
K_ANCHORS = 14
T_SEQ = 7
SEQ_PLUS_RES_DIM = 97
QUERY_DIM = 64
HEAD_SUMMARY_DIM = 51
SLIM7_DIM = 7
RESIDUAL_A_KV_DIM = 5
HIDDEN = 196
ATTN_DIM = 128
HEAD_HIDDEN = 196              # plan-031 slim (plan-030 = 384)
GRU_DROPOUT = 0.10
HEAD_DROPOUT = 0.08
BATCH_SIZE = 64
EVAL_BATCH = 256

# multi-phase epoch budgets
PRE_EPOCHS = 15
FINE_EPOCHS = 35
TOTAL_EPOCHS = PRE_EPOCHS + FINE_EPOCHS                # 50
PRE_LR = 7e-4
FINE_LR = 2e-4                                          # = PRE_LR / 3.5
WARMUP_EP = 5                                           # within pre phase
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
TAU_CLS = 0.001

# fine phase loss weights
W_SOFT_CE = 0.5
W_PAIRWISE = 0.3
W_PRIOR = 0.2
PAIRWISE_MARGIN = 0.12
REGIME_STRENGTH = 0.65
CLASS_STRENGTH = 0.45

REGIME_COUNT = 18
SEED = 20260524


def _to_torch(arr: np.ndarray) -> torch.Tensor:
    return torch.nan_to_num(
        torch.from_numpy(np.ascontiguousarray(arr)).float(),
        nan=0.0, posinf=1e3, neginf=-1e3,
    )


def _build_per_sample_artifacts(X, R_wfn, F0_pred, cand_ext):
    """plan-030 carry — residual/query/head_summary 산출."""
    res = _residual_mod.build_residuals(X, R_wfn, _p022_anchors.ANCHORS_A6, _bf.f0_baseline)
    slim7 = _query_mod.extract_slim7_from_cand_ext_165(cand_ext)
    cand_feat_150 = cand_ext[:, :, :150]
    query_64 = _query_mod.build_query(cand_feat_150, res["residual_b"], slim7)
    macro9 = _p021_build._macro_stat_9d(X, end_idx=10)
    _L2, L4 = _p021_build._build_L2_L4(X, R_wfn, _bf.f0_baseline)
    L4_flat = L4.reshape(L4.shape[0], -1).astype(np.float32)
    head_summary_51 = _head_mod.build_head_summary(cand_feat_150, macro9, L4_flat)
    return {
        "residual_a_gru": res["residual_a_gru"],
        "residual_a_kv": res["residual_a"],
        "query_64": query_64,
        "head_summary_51": head_summary_51,
        "slim7": slim7,
    }


def train_one_fold(
    fold: int,
    X_tr: np.ndarray, X_te: np.ndarray,
    gt_tr: np.ndarray,
    verbose: bool = True,
) -> dict:
    t0 = time.perf_counter()

    # 1) Frenet basis + F0
    R_wfn_tr = _p021_build.build_frenet_basis_3d(X_tr, end_idx=10).astype(np.float32)
    R_wfn_te = _p021_build.build_frenet_basis_3d(X_te, end_idx=10).astype(np.float32)
    F0_tr = _bf.f0_baseline(X_tr, end_idx=10).astype(np.float32)
    F0_te = _bf.f0_baseline(X_te, end_idx=10).astype(np.float32)

    # 2) quantile_carry
    qc = _quantile_mod.build(X_tr, R_wfn_tr)

    # 3) regimes
    regime_bins_tr = fit_regime_bins(X_tr, end_idx=10)
    regimes_tr = assign_regimes(X_tr, end_idx=10, bins=regime_bins_tr)
    regimes_te = assign_regimes(X_te, end_idx=10, bins=regime_bins_tr)

    # 4) regime_anchor_table (train-fold only)
    regime_anchor_table_tr = _aqe.build_regime_anchor_lookup(
        gt_train=gt_tr, regimes_train=regimes_tr, ANCHORS_A6=_p022_anchors.ANCHORS_A6,
        R_wfn_train=R_wfn_tr, F0_train=F0_tr,
        regime_count=REGIME_COUNT, laplace=1.0,
    )

    # 5) cand_ext
    cand_ext_tr = _aqe.build(
        X_tr, R_wfn_tr, F0_tr, _p022_anchors.ANCHORS_A6, _bf.f0_baseline,
        regimes=regimes_tr, quantile_carry=qc,
        regime_count=REGIME_COUNT, regime_anchor_table=regime_anchor_table_tr,
    )
    cand_ext_te = _aqe.build(
        X_te, R_wfn_te, F0_te, _p022_anchors.ANCHORS_A6, _bf.f0_baseline,
        regimes=regimes_te, quantile_carry=qc,
        regime_count=REGIME_COUNT, regime_anchor_table=regime_anchor_table_tr,
    )

    # 6) seq
    seq_tr = _seq_mod.build(X_tr, R_wfn_tr, _p022_anchors.ANCHORS_A6, _bf.f0_baseline, qc)
    seq_te = _seq_mod.build(X_te, R_wfn_te, _p022_anchors.ANCHORS_A6, _bf.f0_baseline, qc)

    # 7-9) per-sample artifacts
    art_tr = _build_per_sample_artifacts(X_tr, R_wfn_tr, F0_tr, cand_ext_tr)
    art_te = _build_per_sample_artifacts(X_te, R_wfn_te, F0_te, cand_ext_te)

    # 10) seq_97 concat
    seq97_tr = np.concatenate([seq_tr, art_tr["residual_a_gru"]], axis=-1).astype(np.float32)
    seq97_te = np.concatenate([seq_te, art_te["residual_a_gru"]], axis=-1).astype(np.float32)

    # 11) soft label q_tr
    q_tr = _p022_soft.build_soft_label_with_tau(
        gt_tr, R_wfn_tr, F0_tr, _p022_anchors.ANCHORS_A6, tau_cls=TAU_CLS,
    )

    # 12) gt_anchor_idx (pairwise loss용)
    R_t_tr = np.transpose(R_wfn_tr, (0, 2, 1))
    gt_res_f = np.einsum("nij,nj->ni", R_t_tr, gt_tr - F0_tr)
    gt_anchor_idx_tr = np.linalg.norm(
        _p022_anchors.ANCHORS_A6[None, :, :] - gt_res_f[:, None, :], axis=-1,
    ).argmin(axis=1).astype(np.int64)

    # 13) priors (train-fold만)
    class_prior_global = _prior_mod.build_class_prior_global(gt_anchor_idx_tr, K=K_ANCHORS)
    regime_anchor_prior_tr = _prior_mod.build_regime_anchor_prior(regime_anchor_table_tr, regimes_tr)

    # 14) Model
    torch.manual_seed(SEED + fold)
    anchors_torch = torch.from_numpy(_p022_anchors.ANCHORS_A6.astype(np.float32))
    model = _model_mod.GRUNetX3(
        seq_dim=SEQ_PLUS_RES_DIM, query_dim=QUERY_DIM,
        head_summary_dim=HEAD_SUMMARY_DIM, slim7_dim=SLIM7_DIM,
        hidden=HIDDEN, attn_dim=ATTN_DIM, head_hidden=HEAD_HIDDEN,
        gru_dropout=GRU_DROPOUT, head_dropout=HEAD_DROPOUT,
        K=K_ANCHORS, residual_a_kv_dim=RESIDUAL_A_KV_DIM,
        anchors=anchors_torch,
    )

    # 15) Phase A — pre (15 ep, soft CE only, lr=7e-4, warmup 5 + cosine 10)
    optimizer = torch.optim.AdamW(model.parameters(), lr=PRE_LR, weight_decay=WEIGHT_DECAY)
    scheduler_pre = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[
            torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=1e-6, end_factor=1.0, total_iters=WARMUP_EP,
            ),
            torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=PRE_EPOCHS - WARMUP_EP),
        ],
        milestones=[WARMUP_EP],
    )

    N_tr = X_tr.shape[0]
    score_std_trajectory = []
    loss_trajectory = []
    for epoch in range(PRE_EPOCHS):
        model.train()
        rng_ep = np.random.default_rng(SEED + epoch * 1000 + fold)
        perm = rng_ep.permutation(N_tr)
        last_loss = 0.0
        last_score_std = 0.0
        for b_start in range(0, N_tr, BATCH_SIZE):
            idx = perm[b_start : b_start + BATCH_SIZE]
            seq_b = _to_torch(seq97_tr[idx])
            res_kv_b = _to_torch(art_tr["residual_a_kv"][idx])
            q64_b = _to_torch(art_tr["query_64"][idx])
            hs_b = _to_torch(art_tr["head_summary_51"][idx])
            slim7_b = _to_torch(art_tr["slim7"][idx])
            F0_b = _to_torch(F0_tr[idx])
            R_b = _to_torch(R_wfn_tr[idx])
            q_b = torch.from_numpy(q_tr[idx]).float()

            optimizer.zero_grad()
            _wp, probs = model(seq_b, res_kv_b, q64_b, hs_b, slim7_b, F0_b, R_b)
            log_probs = torch.log(probs.clamp_min(1e-12))
            loss = -(q_b * log_probs).sum(dim=-1).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            last_loss = loss.item()
            with torch.no_grad():
                last_score_std = log_probs.std().item()
        scheduler_pre.step()
        score_std_trajectory.append(last_score_std)
        loss_trajectory.append(last_loss)
        if verbose and (epoch + 1) % 5 == 0:
            print(f"  [pre]  fold={fold} ep={epoch + 1:2d}/{PRE_EPOCHS} loss={last_loss:.4f} "
                  f"score_std={last_score_std:.3e} lr={scheduler_pre.get_last_lr()[0]:.3e}")

    # 16) Phase B — fine (35 ep, multi loss, lr=2e-4 from this point, cosine T_max=35)
    # optimizer state continue, but lr reset to FINE_LR
    for g in optimizer.param_groups:
        g["lr"] = FINE_LR
        g["initial_lr"] = FINE_LR
    scheduler_fine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=FINE_EPOCHS)

    # forward-once: score 직접 받기 위해 head_mlp output (raw logit) 사용
    # GRUNetX2.forward 는 (world_pred, probs) 만 반환 → score 추출 wrapper
    def _forward_score(model, seq_b, res_kv_b, q64_b, hs_b, slim7_b, F0_b, R_b):
        """raw logit score (B, K) 도 함께 반환하는 wrapper."""
        from math import sqrt
        B = seq_b.shape[0]
        K = model.K
        H = model.hidden
        gru_out, _ = model.gru(seq_b)
        gru_hidden_last = gru_out[:, -1, :]
        kv_raw = torch.cat([gru_out, res_kv_b], dim=-1)
        kv = model.kv_proj(kv_raw)
        q = model.q_proj(q64_b)
        attn_logits = torch.einsum("bka,bta->bkt", q, kv) / sqrt(model.attn_dim)
        attn_w = torch.softmax(attn_logits, dim=-1)
        attn_context = torch.einsum("bkt,bta->bka", attn_w, kv)
        sample_summary_full = torch.cat([gru_hidden_last, hs_b], dim=-1)
        sample_bias = sample_summary_full.unsqueeze(1).expand(-1, K, -1)
        head_in = torch.cat([attn_context, sample_bias, slim7_b], dim=-1)
        score = model.head_mlp(head_in).squeeze(-1)
        probs = torch.softmax(score, dim=-1)
        residual_frenet = probs @ model.ANCHORS_A6
        residual_world = torch.einsum("bij,bj->bi", R_b, residual_frenet)
        world_pred = F0_b + residual_world
        return world_pred, probs, score

    class_prior_t = torch.from_numpy(class_prior_global).float()

    for epoch in range(FINE_EPOCHS):
        model.train()
        rng_ep = np.random.default_rng(SEED + (PRE_EPOCHS + epoch) * 1000 + fold)
        perm = rng_ep.permutation(N_tr)
        last_loss = 0.0
        last_score_std = 0.0
        for b_start in range(0, N_tr, BATCH_SIZE):
            idx = perm[b_start : b_start + BATCH_SIZE]
            seq_b = _to_torch(seq97_tr[idx])
            res_kv_b = _to_torch(art_tr["residual_a_kv"][idx])
            q64_b = _to_torch(art_tr["query_64"][idx])
            hs_b = _to_torch(art_tr["head_summary_51"][idx])
            slim7_b = _to_torch(art_tr["slim7"][idx])
            F0_b = _to_torch(F0_tr[idx])
            R_b = _to_torch(R_wfn_tr[idx])
            q_b = torch.from_numpy(q_tr[idx]).float()
            gt_idx_b = torch.from_numpy(gt_anchor_idx_tr[idx]).long()
            reg_prior_b = torch.from_numpy(regime_anchor_prior_tr[idx]).float()

            optimizer.zero_grad()
            _wp, probs, score = _forward_score(model, seq_b, res_kv_b, q64_b, hs_b, slim7_b, F0_b, R_b)

            log_probs = torch.log(probs.clamp_min(1e-12))
            loss_soft = -(q_b * log_probs).sum(dim=-1).mean()
            loss_pair = _pairwise_mod.pairwise_margin_loss(score, gt_idx_b, margin=PAIRWISE_MARGIN)
            loss_prior = _prior_mod.regime_class_prior_loss(
                score, reg_prior_b, class_prior_t,
                regime_strength=REGIME_STRENGTH, class_strength=CLASS_STRENGTH,
            )
            loss = W_SOFT_CE * loss_soft + W_PAIRWISE * loss_pair + W_PRIOR * loss_prior
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            last_loss = loss.item()
            with torch.no_grad():
                last_score_std = score.std().item()
        scheduler_fine.step()
        score_std_trajectory.append(last_score_std)
        loss_trajectory.append(last_loss)
        if verbose and (epoch + 1) % 5 == 0:
            print(f"  [fine] fold={fold} ep={epoch + 1:2d}/{FINE_EPOCHS} loss={last_loss:.4f} "
                  f"(CE/pair/prior={loss_soft.item():.3f}/{loss_pair.item():.3f}/{loss_prior.item():.3f}) "
                  f"score_std={last_score_std:.3e} lr={scheduler_fine.get_last_lr()[0]:.3e}")

    # 17) eval
    model.eval()
    pred_te_list = []
    probs_te_list = []
    with torch.no_grad():
        for e_start in range(0, X_te.shape[0], EVAL_BATCH):
            sl = slice(e_start, e_start + EVAL_BATCH)
            seq_b = _to_torch(seq97_te[sl])
            res_kv_b = _to_torch(art_te["residual_a_kv"][sl])
            q64_b = _to_torch(art_te["query_64"][sl])
            hs_b = _to_torch(art_te["head_summary_51"][sl])
            slim7_b = _to_torch(art_te["slim7"][sl])
            F0_b = _to_torch(F0_te[sl])
            R_b = _to_torch(R_wfn_te[sl])
            world_pred, probs = model(seq_b, res_kv_b, q64_b, hs_b, slim7_b, F0_b, R_b)
            pred_te_list.append(world_pred.cpu().numpy())
            probs_te_list.append(probs.cpu().numpy())

    pred_te = np.concatenate(pred_te_list, axis=0).astype(np.float32)
    probs_te = np.concatenate(probs_te_list, axis=0).astype(np.float32)

    t_fold = time.perf_counter() - t0
    if verbose:
        print(f"[fold {fold}] done in {t_fold:.1f}s, N_te={X_te.shape[0]}, "
              f"final_score_std={score_std_trajectory[-1]:.3e}")

    return {
        "fold": fold,
        "world_pred_te": pred_te,
        "probs_te": probs_te,
        "R_wfn_te": R_wfn_te,
        "F0_te": F0_te,
        "score_std_trajectory": score_std_trajectory,
        "loss_trajectory": loss_trajectory,
        "elapsed_s": t_fold,
    }


def run_5fold_oof(verbose: bool = True) -> dict:
    t_total = time.perf_counter()
    _ids, X = load_all_samples()
    _ids_gt, gt = load_labels()
    assert _ids == _ids_gt
    X = X.astype(np.float32)
    gt = gt.astype(np.float32)
    N_total = X.shape[0]

    folds = np.asarray([stable_fold_id(str(sid), N_FOLDS) for sid in _ids], dtype=int)

    oof_pred = np.zeros((N_total, 3), dtype=np.float32)
    oof_probs = np.zeros((N_total, K_ANCHORS), dtype=np.float32)
    R_wfn_all = np.zeros((N_total, 3, 3), dtype=np.float32)
    F0_all = np.zeros((N_total, 3), dtype=np.float32)

    fold_logs = []
    for fold in range(N_FOLDS):
        train_idx = np.where(folds != fold)[0]
        test_idx = np.where(folds == fold)[0]
        X_tr, X_te = X[train_idx], X[test_idx]
        gt_tr = gt[train_idx]
        log = train_one_fold(fold, X_tr, X_te, gt_tr, verbose=verbose)
        oof_pred[test_idx] = log["world_pred_te"]
        oof_probs[test_idx] = log["probs_te"]
        R_wfn_all[test_idx] = log["R_wfn_te"]
        F0_all[test_idx] = log["F0_te"]
        fold_logs.append({
            "fold": fold,
            "elapsed_s": log["elapsed_s"],
            "score_std_trajectory": log["score_std_trajectory"],
            "loss_trajectory": log["loss_trajectory"],
            "N_te": int(X_te.shape[0]),
        })

    err = np.linalg.norm(oof_pred - gt, axis=1)
    hit_1cm = float((err <= 0.01).mean())
    hit_1p5cm = float((err <= 0.015).mean())
    top1_argmax = oof_probs.argmax(axis=1)
    max_class_ratio = float(np.bincount(top1_argmax, minlength=K_ANCHORS).max() / N_total)

    R_t_all = np.transpose(R_wfn_all, (0, 2, 1))
    gt_res_f = np.einsum("nij,nj->ni", R_t_all, gt - F0_all)
    gt_anchor_label = np.linalg.norm(
        _p022_anchors.ANCHORS_A6[None, :, :] - gt_res_f[:, None, :], axis=-1,
    ).argmin(axis=1)
    top1_acc = float((top1_argmax == gt_anchor_label).mean())

    elapsed_total = time.perf_counter() - t_total

    return {
        "cell": "plan031-pb-multiphase",
        "hit_1cm": hit_1cm,
        "hit_1p5cm": hit_1p5cm,
        "max_class_ratio": max_class_ratio,
        "top1_acc": top1_acc,
        "N_total": N_total,
        "K": K_ANCHORS,
        "elapsed_total_s": elapsed_total,
        "fold_logs": fold_logs,
        "hparams": {
            "pre_epochs": PRE_EPOCHS, "fine_epochs": FINE_EPOCHS,
            "pre_lr": PRE_LR, "fine_lr": FINE_LR, "warmup_ep": WARMUP_EP,
            "wd": WEIGHT_DECAY, "batch": BATCH_SIZE,
            "hidden": HIDDEN, "attn_dim": ATTN_DIM, "head_hidden": HEAD_HIDDEN,
            "gru_dropout": GRU_DROPOUT, "head_dropout": HEAD_DROPOUT,
            "grad_clip": GRAD_CLIP, "tau_cls": TAU_CLS,
            "w_soft_ce": W_SOFT_CE, "w_pairwise": W_PAIRWISE, "w_prior": W_PRIOR,
            "pairwise_margin": PAIRWISE_MARGIN,
            "regime_strength": REGIME_STRENGTH, "class_strength": CLASS_STRENGTH,
            "regime_count": REGIME_COUNT, "seed": SEED,
        },
        "oof_pred": oof_pred,
        "oof_probs": oof_probs,
        "gt_anchor_label": gt_anchor_label.tolist(),
    }


def _smoke() -> None:
    global PRE_EPOCHS, FINE_EPOCHS, WARMUP_EP
    PRE_EPOCHS, FINE_EPOCHS, WARMUP_EP = 2, 2, 1
    rng = np.random.default_rng(SEED)
    N = 16
    X_tr = rng.standard_normal((N, 11, 3)).astype(np.float32) * 0.5
    X_te = rng.standard_normal((6, 11, 3)).astype(np.float32) * 0.5
    gt_tr = X_tr[:, 10, :] + rng.standard_normal((N, 3)).astype(np.float32) * 0.005
    log = train_one_fold(fold=0, X_tr=X_tr, X_te=X_te, gt_tr=gt_tr, verbose=False)
    assert log["world_pred_te"].shape == (6, 3)
    assert log["probs_te"].shape == (6, K_ANCHORS)
    assert len(log["score_std_trajectory"]) == 4  # 2 pre + 2 fine
    print(f"[smoke] train OK: pred={log['world_pred_te'].shape}, "
          f"probs={log['probs_te'].shape}, elapsed={log['elapsed_s']:.2f}s, "
          f"score_std_traj len={len(log['score_std_trajectory'])}, "
          f"last_score_std={log['score_std_trajectory'][-1]:.3e}")


if __name__ == "__main__":
    _smoke()
