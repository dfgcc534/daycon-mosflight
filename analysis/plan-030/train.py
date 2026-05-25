"""plan-030 c5 — train.py.

PyTorch 5-fold OOF training (plan-030 spec §4):
- epoch=50, lr=7e-4, SequentialLR (warmup 5ep + cosine 45ep), AdamW (wd=1e-4)
- batch=64, GRU dropout=0.10, grad_clip=1.0, soft CE with τ_cls=0.001
- plan-029 X1 carry (single phase). PB multi-phase 는 §7 deferred.

per-fold pipeline:
  1) Frenet basis + F0 (plan-021/020 carry)
  2) train-fold quantile_carry (fold-leakage 차단)
  3) regime_bins + regimes (train-fold inject)
  4) regime_anchor_table (train-fold lookup)
  5) cand_ext_tr/te (plan-029 anchor_query_extend)
  6) seq_tr/te (plan-024 seq_builder)
  7) residual_a/_gru/_b (plan-030 residual_builder)
  8) query_64 (plan-030 query_builder)
  9) head_summary_51 (plan-030 head_summary + plan-021 macro_9 + L4)
 10) soft label q_tr (plan-022 build_soft_label_with_tau, τ_cls=0.001)
 11) seq_97 = concat([seq_95, residual_a_gru], -1)
 12) GRUNetX2 init (ANCHORS_A6 buffer) + AdamW + SequentialLR
 13) epoch loop: forward (7 args) → log_softmax soft CE → backward → clip → step
 14) eval: forward → (world_pred, probs)
 15) OOF accumulate world_pred + probs
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_THIS = Path(__file__).resolve().parent              # analysis/plan-030/
_REPO = _THIS.parent.parent                          # repo root

for p in (_REPO, _THIS.parent / "plan-020", _THIS.parent / "plan-021",
          _THIS.parent / "plan-022", _THIS.parent / "plan-024",
          _THIS.parent / "plan-029"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


# carry modules
_bf = _load(_THIS.parent / "plan-020" / "baseline_f0.py", "p030_train_bf")
_p021_build = _load(_THIS.parent / "plan-021" / "build_input.py", "p030_train_bi")
_p022_anchors = _load(_THIS.parent / "plan-022" / "anchors.py", "p030_train_av")
_p022_soft = _load(_THIS.parent / "plan-022" / "selector_only_model.py", "p030_train_soft")
_seq_mod = _load(_THIS.parent / "plan-024" / "seq_builder.py", "p030_train_seq")
_quantile_mod = _load(_THIS.parent / "plan-024" / "quantile_carry.py", "p030_train_qc")
_aqe = _load(_THIS.parent / "plan-029" / "anchor_query_extend.py", "p030_train_aqe")

# plan-030 modules
_residual_mod = _load(_THIS / "residual_builder.py", "p030_train_residual")
_query_mod = _load(_THIS / "query_builder.py", "p030_train_query")
_head_mod = _load(_THIS / "head_summary.py", "p030_train_head")
_model_mod = _load(_THIS / "model.py", "p030_train_model")

from src.io import load_all_samples, load_labels  # noqa: E402
from src.pb_0_6822.selector import stable_fold_id, fit_regime_bins, assign_regimes  # noqa: E402

N_FOLDS = 5
K_ANCHORS = 14
T_SEQ = 7
SEQ_DIM = 95
SEQ_PLUS_RES_DIM = 97          # seq 95 + residual_a_gru 2
QUERY_DIM = 64
HEAD_SUMMARY_DIM = 51
SLIM7_DIM = 7
RESIDUAL_A_KV_DIM = 5
HIDDEN = 196
ATTN_DIM = 128
HEAD_HIDDEN = 384
GRU_DROPOUT = 0.10
HEAD_DROPOUT = 0.08
BATCH_SIZE = 64
EVAL_BATCH = 256
EPOCHS = 50
WARMUP_EP = 5
LR = 7e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
TAU_CLS = 0.001
REGIME_COUNT = 18
SEED = 20260524


def _to_torch(arr: np.ndarray) -> torch.Tensor:
    """np → torch float32 + nan_to_num safety."""
    return torch.nan_to_num(
        torch.from_numpy(np.ascontiguousarray(arr)).float(),
        nan=0.0, posinf=1e3, neginf=-1e3,
    )


def _build_per_sample_artifacts(
    X: np.ndarray,
    R_wfn: np.ndarray,
    F0_pred: np.ndarray,
    cand_ext: np.ndarray,
) -> dict[str, np.ndarray]:
    """plan-030 input artifact builder (per fold side, train/test 각각 호출).

    Returns:
        seq_97:          (N, T=7, 97)
        residual_a_kv:   (N, T=7, 5)
        query_64:        (N, K=14, 64)
        head_summary_51: (N, 51)
        slim7:           (N, K=14, 7)
    """
    # residual block (plan-030 c1)
    res = _residual_mod.build_residuals(
        X, R_wfn, _p022_anchors.ANCHORS_A6, _bf.f0_baseline,
    )
    # query 64D (plan-030 c2): slim7 from cand_ext 165D
    slim7 = _query_mod.extract_slim7_from_cand_ext_165(cand_ext)  # (N, K, 7)
    cand_feat_150 = cand_ext[:, :, :150]                          # plan-024 base
    query_64 = _query_mod.build_query(cand_feat_150, res["residual_b"], slim7)  # (N, K, 64)
    # head_summary 51D (plan-030 c3): plan-021 macro_9 + L4
    macro9 = _p021_build._macro_stat_9d(X, end_idx=10)            # (N, 9)
    _L2, L4 = _p021_build._build_L2_L4(X, R_wfn, _bf.f0_baseline)
    L4_flat = L4.reshape(L4.shape[0], -1).astype(np.float32)      # (N, 14)
    head_summary_51 = _head_mod.build_head_summary(cand_feat_150, macro9, L4_flat)
    return {
        "residual_a_gru": res["residual_a_gru"],
        "residual_a_kv": res["residual_a"],     # 5coord 그대로
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
    """plan-030 1-fold training. Returns dict with world_pred_te, probs_te, R_wfn_te, F0_te, etc."""
    t0 = time.perf_counter()

    # 1) Frenet basis + F0
    R_wfn_tr = _p021_build.build_frenet_basis_3d(X_tr, end_idx=10).astype(np.float32)
    R_wfn_te = _p021_build.build_frenet_basis_3d(X_te, end_idx=10).astype(np.float32)
    F0_tr = _bf.f0_baseline(X_tr, end_idx=10).astype(np.float32)
    F0_te = _bf.f0_baseline(X_te, end_idx=10).astype(np.float32)

    # 2) quantile_carry (train-fold only)
    qc = _quantile_mod.build(X_tr, R_wfn_tr)

    # 3) regimes (train-fold bins → inject)
    regime_bins_tr = fit_regime_bins(X_tr, end_idx=10)
    regimes_tr = assign_regimes(X_tr, end_idx=10, bins=regime_bins_tr)
    regimes_te = assign_regimes(X_te, end_idx=10, bins=regime_bins_tr)

    # 4) regime_anchor_table (train-fold only)
    regime_anchor_table_tr = _aqe.build_regime_anchor_lookup(
        gt_train=gt_tr, regimes_train=regimes_tr, ANCHORS_A6=_p022_anchors.ANCHORS_A6,
        R_wfn_train=R_wfn_tr, F0_train=F0_tr,
        regime_count=REGIME_COUNT, laplace=1.0,
    )

    # 5) cand_ext 165D (plan-029 anchor_query_extend)
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

    # 6) seq 95D × 7 step (plan-024 seq_builder)
    seq_tr = _seq_mod.build(X_tr, R_wfn_tr, _p022_anchors.ANCHORS_A6, _bf.f0_baseline, qc)
    seq_te = _seq_mod.build(X_te, R_wfn_te, _p022_anchors.ANCHORS_A6, _bf.f0_baseline, qc)

    # 7-9) per-sample artifacts (plan-030 builders)
    art_tr = _build_per_sample_artifacts(X_tr, R_wfn_tr, F0_tr, cand_ext_tr)
    art_te = _build_per_sample_artifacts(X_te, R_wfn_te, F0_te, cand_ext_te)

    # 11) seq_97 = concat(seq_95, residual_a_gru)
    seq97_tr = np.concatenate([seq_tr, art_tr["residual_a_gru"]], axis=-1).astype(np.float32)
    seq97_te = np.concatenate([seq_te, art_te["residual_a_gru"]], axis=-1).astype(np.float32)

    # 10) soft label
    q_tr = _p022_soft.build_soft_label_with_tau(
        gt_tr, R_wfn_tr, F0_tr, _p022_anchors.ANCHORS_A6, tau_cls=TAU_CLS,
    )

    # 12) Model + Optimizer + Scheduler
    torch.manual_seed(SEED + fold)
    anchors_torch = torch.from_numpy(_p022_anchors.ANCHORS_A6.astype(np.float32))
    model = _model_mod.GRUNetX2(
        seq_dim=SEQ_PLUS_RES_DIM, query_dim=QUERY_DIM,
        head_summary_dim=HEAD_SUMMARY_DIM, slim7_dim=SLIM7_DIM,
        hidden=HIDDEN, attn_dim=ATTN_DIM, head_hidden=HEAD_HIDDEN,
        gru_dropout=GRU_DROPOUT, head_dropout=HEAD_DROPOUT,
        K=K_ANCHORS, residual_a_kv_dim=RESIDUAL_A_KV_DIM,
        anchors=anchors_torch,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[
            torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=1e-6, end_factor=1.0, total_iters=WARMUP_EP,
            ),
            torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS - WARMUP_EP),
        ],
        milestones=[WARMUP_EP],
    )

    # 13) training loop
    N_tr = X_tr.shape[0]
    score_std_trajectory = []
    for epoch in range(EPOCHS):
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
            _world_pred, probs = model(seq_b, res_kv_b, q64_b, hs_b, slim7_b, F0_b, R_b)
            # loss: -Σ q · log(probs). probs 가 이미 softmax 결과라 log() 호출.
            #   numerical safety: clamp probs >= 1e-12 (log_softmax 를 score 에 직접 호출 안 함 — model 안에서 softmax 후 반환).
            log_probs = torch.log(probs.clamp_min(1e-12))
            loss = -(q_b * log_probs).sum(dim=-1).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            last_loss = loss.item()
            # last batch score std (raw logit-scale for §5 fallback rule monitoring)
            with torch.no_grad():
                # score = log(probs) without normalization is equivalent up to constant — use log(probs.clamp) std
                last_score_std = log_probs.std().item()
        scheduler.step()
        score_std_trajectory.append(last_score_std)
        if verbose and (epoch + 1) % 10 == 0:
            print(f"  fold={fold} ep={epoch + 1:2d}/{EPOCHS} loss={last_loss:.4f} "
                  f"score_std={last_score_std:.3e} lr={scheduler.get_last_lr()[0]:.3e}")

    # 14) eval
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
        "elapsed_s": t_fold,
    }


def run_5fold_oof(verbose: bool = True) -> dict:
    """5-fold concat OOF for plan-030 baseline. Returns metric dict."""
    t_total = time.perf_counter()

    _ids, X = load_all_samples()
    _ids_gt, gt = load_labels()
    assert _ids == _ids_gt, "id mismatch"
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
        "cell": "plan030-baseline",
        "hit_1cm": hit_1cm,
        "hit_1p5cm": hit_1p5cm,
        "max_class_ratio": max_class_ratio,
        "top1_acc": top1_acc,
        "N_total": N_total,
        "K": K_ANCHORS,
        "elapsed_total_s": elapsed_total,
        "fold_logs": fold_logs,
        "hparams": {
            "epochs": EPOCHS, "warmup_ep": WARMUP_EP, "lr": LR, "wd": WEIGHT_DECAY,
            "batch": BATCH_SIZE, "hidden": HIDDEN, "attn_dim": ATTN_DIM,
            "head_hidden": HEAD_HIDDEN, "gru_dropout": GRU_DROPOUT,
            "head_dropout": HEAD_DROPOUT, "grad_clip": GRAD_CLIP, "tau_cls": TAU_CLS,
            "regime_count": REGIME_COUNT, "seed": SEED,
        },
        "oof_pred": oof_pred,
        "oof_probs": oof_probs,
        "gt_anchor_label": gt_anchor_label.tolist(),
    }


def _smoke() -> None:
    """smoke — 1 fold tiny mock, epoch=2."""
    global EPOCHS, WARMUP_EP
    EPOCHS, WARMUP_EP = 2, 1
    rng = np.random.default_rng(SEED)
    N = 16
    X_tr = rng.standard_normal((N, 11, 3)).astype(np.float32) * 0.5
    X_te = rng.standard_normal((6, 11, 3)).astype(np.float32) * 0.5
    gt_tr = X_tr[:, 10, :] + rng.standard_normal((N, 3)).astype(np.float32) * 0.005
    log = train_one_fold(fold=0, X_tr=X_tr, X_te=X_te, gt_tr=gt_tr, verbose=False)
    assert log["world_pred_te"].shape == (6, 3)
    assert log["probs_te"].shape == (6, K_ANCHORS)
    assert log["R_wfn_te"].shape == (6, 3, 3)
    print(f"[smoke] train OK: pred={log['world_pred_te'].shape}, "
          f"probs={log['probs_te'].shape}, elapsed={log['elapsed_s']:.2f}s, "
          f"final_score_std={log['score_std_trajectory'][-1]:.3e}")


if __name__ == "__main__":
    _smoke()
