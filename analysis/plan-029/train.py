"""plan-029 c5 — train.py.

PyTorch 5-fold OOF training loop (§6.1 spec):
- epoch=50 fixed (no early stop)
- lr=7e-4, SequentialLR([LinearLR warmup 5 ep, CosineAnnealingLR T_max=45 ep])
- AdamW (wd=1e-4)
- GRU dropout=0.10, gradient_clip=1.0
- batch=64 (last batch < 64 그대로 사용)
- soft cross-entropy loss

per-fold:
  1) Frenet basis + F0 (train/test 각각 산출)
  2) train-fold quantile_carry (fold-leakage 차단)
  3) regime_bins + regimes (train-fold bins → test-fold inject)
  4) regime_anchor_table (train-fold gt → train-fold lookup, test-fold inject)
  5) cand_ext_tr / cand_ext_te (anchor_query_extend.build with table inject)
  6) seq_tr / seq_te (seq_builder.build with train-fold qc)
  7) soft label q_tr (build_soft_label_with_tau)
  8) GRUNetX1 init + AdamW + SequentialLR
  9) epoch loop: nan_to_num → forward → soft CE → backward → clip → step
 10) per-epoch anchor_embed grad norm log
 11) eval: nan_to_num → forward → softmax → final_pred (F0 + R_wfn @ residual_frenet)
 12) fold-별 R_wfn / F0 / oof_pred / oof_probs 누적
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

_THIS = Path(__file__).resolve().parent              # analysis/plan-029/
_REPO = _THIS.parent.parent                          # repo root

for p in (_REPO, _THIS.parent / "plan-020", _THIS.parent / "plan-021",
          _THIS.parent / "plan-022", _THIS.parent / "plan-024"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


# plan-020/021/022/024 carry
_bf = _load(_THIS.parent / "plan-020" / "baseline_f0.py", "p029_train_bf")
_p021_build = _load(_THIS.parent / "plan-021" / "build_input.py", "p029_train_bi")
_p022_anchors = _load(_THIS.parent / "plan-022" / "anchors.py", "p029_train_av")
_p022_soft = _load(_THIS.parent / "plan-022" / "selector_only_model.py", "p029_train_soft")
_seq_mod = _load(_THIS.parent / "plan-024" / "seq_builder.py", "p029_train_seq")
_quantile_mod = _load(_THIS.parent / "plan-024" / "quantile_carry.py", "p029_train_qc")

# plan-029 modules — local importlib (peer 모듈)
_aqe = _load(_THIS / "anchor_query_extend.py", "p029_train_aqe")
_model_mod = _load(_THIS / "model.py", "p029_train_model")

from src.io import load_all_samples, load_labels  # noqa: E402
from src.pb_0_6822.selector import stable_fold_id, fit_regime_bins, assign_regimes  # noqa: E402

N_FOLDS = 5
K_ANCHORS = 14
T_SEQ = 7
SEQ_DIM = 95
CAND_EXT_DIM = 165
HIDDEN = 196
ANCHOR_EMBED_DIM = 8
ANCHOR_EMBED_INIT_SCALE = 0.1
GRU_DROPOUT = 0.10
BATCH_SIZE = 64
EVAL_BATCH = 256
EPOCHS = 50
WARMUP_EP = 5
LR = 7e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
TAU_CLS = 0.001
REGIME_COUNT = 18
SEED = 20260522


def _to_torch(arr: np.ndarray) -> torch.Tensor:
    """np → torch float32 + nan_to_num safety net."""
    return torch.nan_to_num(
        torch.from_numpy(np.ascontiguousarray(arr)).float(),
        nan=0.0, posinf=1e3, neginf=-1e3,
    )


def train_one_fold(
    fold: int,
    X_tr: np.ndarray, X_te: np.ndarray,
    gt_tr: np.ndarray,
    train_idx: np.ndarray, test_idx: np.ndarray,
    verbose: bool = True,
) -> dict:
    """Train one fold. Returns dict with oof_pred_te, oof_probs_te, R_wfn_te, F0_te, grad_norm_trajectory."""
    t0 = time.perf_counter()

    # 1) Frenet basis + F0
    R_wfn_tr = _p021_build.build_frenet_basis_3d(X_tr, end_idx=10).astype(np.float32)
    R_wfn_te = _p021_build.build_frenet_basis_3d(X_te, end_idx=10).astype(np.float32)
    F0_tr = _bf.f0_baseline(X_tr, end_idx=10).astype(np.float32)
    F0_te = _bf.f0_baseline(X_te, end_idx=10).astype(np.float32)

    # 2) train-fold quantile_carry
    qc = _quantile_mod.build(X_tr, R_wfn_tr)

    # 3) regime_bins + regimes (train-fold bins → test-fold inject)
    regime_bins_tr = fit_regime_bins(X_tr, end_idx=10)
    regimes_tr = assign_regimes(X_tr, end_idx=10, bins=regime_bins_tr)
    regimes_te = assign_regimes(X_te, end_idx=10, bins=regime_bins_tr)

    # 4) regime_anchor_table (train-fold only)
    regime_anchor_table_tr = _aqe.build_regime_anchor_lookup(
        gt_train=gt_tr, regimes_train=regimes_tr, ANCHORS_A6=_p022_anchors.ANCHORS_A6,
        R_wfn_train=R_wfn_tr, F0_train=F0_tr,
        regime_count=REGIME_COUNT, laplace=1.0,
    )

    # 5) cand_ext (train + test, train-fold table inject)
    cand_ext_tr = _aqe.build(
        X_tr, R_wfn_tr, F0_tr, _p022_anchors.ANCHORS_A6, _bf.f0_baseline,
        regimes=regimes_tr, quantile_carry=qc,
        regime_count=REGIME_COUNT,
        regime_anchor_table=regime_anchor_table_tr,
    )
    cand_ext_te = _aqe.build(
        X_te, R_wfn_te, F0_te, _p022_anchors.ANCHORS_A6, _bf.f0_baseline,
        regimes=regimes_te, quantile_carry=qc,
        regime_count=REGIME_COUNT,
        regime_anchor_table=regime_anchor_table_tr,
    )

    # 6) seq (train + test, train-fold qc)
    seq_tr = _seq_mod.build(X_tr, R_wfn_tr, _p022_anchors.ANCHORS_A6, _bf.f0_baseline, qc)
    seq_te = _seq_mod.build(X_te, R_wfn_te, _p022_anchors.ANCHORS_A6, _bf.f0_baseline, qc)

    # 7) soft label
    q_tr = _p022_soft.build_soft_label_with_tau(
        gt_tr, R_wfn_tr, F0_tr, _p022_anchors.ANCHORS_A6, tau_cls=TAU_CLS,
    )

    # 8) Model + Optimizer + Scheduler
    torch.manual_seed(SEED + fold)
    model = _model_mod.GRUNetX1(
        seq_dim=SEQ_DIM, cand_in_dim=CAND_EXT_DIM, hidden=HIDDEN,
        anchor_embed_dim=ANCHOR_EMBED_DIM,
        anchor_embed_init_scale=ANCHOR_EMBED_INIT_SCALE,
        gru_dropout=GRU_DROPOUT, K=K_ANCHORS,
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

    # 9) training loop
    N_tr = X_tr.shape[0]
    grad_norm_trajectory = []
    for epoch in range(EPOCHS):
        model.train()
        rng_ep = np.random.default_rng(SEED + epoch * 1000 + fold)
        perm = rng_ep.permutation(N_tr)
        for b_start in range(0, N_tr, BATCH_SIZE):
            idx = perm[b_start : b_start + BATCH_SIZE]
            seq_batch = _to_torch(seq_tr[idx])
            cand_batch = _to_torch(cand_ext_tr[idx])
            q_batch = torch.from_numpy(q_tr[idx]).float()
            optimizer.zero_grad()
            score = model(seq_batch, cand_batch)
            log_probs = F.log_softmax(score, dim=-1)
            loss = -(q_batch * log_probs).sum(dim=-1).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
        scheduler.step()
        # per-epoch anchor_embed grad norm (last batch)
        grad_norm = (
            model.anchor_embed.grad.norm().item()
            if model.anchor_embed.grad is not None else 0.0
        )
        grad_norm_trajectory.append(grad_norm)
        if verbose and (epoch + 1) % 10 == 0:
            print(f"  fold={fold} ep={epoch + 1:2d}/{EPOCHS} loss={loss.item():.4f} "
                  f"grad_norm={grad_norm:.4e} lr={scheduler.get_last_lr()[0]:.3e}")

    # 10) eval
    model.eval()
    probs_te_list = []
    with torch.no_grad():
        for e_start in range(0, X_te.shape[0], EVAL_BATCH):
            sb = _to_torch(seq_te[e_start : e_start + EVAL_BATCH])
            cb = _to_torch(cand_ext_te[e_start : e_start + EVAL_BATCH])
            sc = model(sb, cb)
            probs_te_list.append(F.softmax(sc, dim=-1).cpu().numpy())
    probs_te = np.concatenate(probs_te_list, axis=0).astype(np.float32)  # (N_te, K)

    # final pred = F0 + R_wfn @ (probs @ anchors_frenet)
    residual_frenet = (probs_te[:, :, None] * _p022_anchors.ANCHORS_A6[None, :, :]).sum(axis=1)
    residual_world = np.einsum("nij,nj->ni", R_wfn_te, residual_frenet)
    final_pred = (F0_te + residual_world).astype(np.float32)

    t_fold = time.perf_counter() - t0
    if verbose:
        print(f"[fold {fold}] done in {t_fold:.1f}s, N_te={X_te.shape[0]}, "
              f"final_grad_norm={grad_norm_trajectory[-1]:.4e}")

    return {
        "fold": fold,
        "oof_pred_te": final_pred,
        "oof_probs_te": probs_te,
        "R_wfn_te": R_wfn_te,
        "F0_te": F0_te,
        "grad_norm_trajectory": grad_norm_trajectory,
        "elapsed_s": t_fold,
    }


def run_5fold_oof(verbose: bool = True) -> dict:
    """5-fold concat OOF for X1 cell. Returns metric dict + per-fold logs."""
    t_total = time.perf_counter()

    # Dataset-wide load
    _ids, X = load_all_samples()
    _ids_gt, gt = load_labels()
    assert _ids == _ids_gt, "id mismatch"
    X = X.astype(np.float32)
    gt = gt.astype(np.float32)
    N_total = X.shape[0]

    folds = np.asarray([stable_fold_id(str(sid), N_FOLDS) for sid in _ids], dtype=int)

    # Pre-alloc
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
        log = train_one_fold(fold, X_tr, X_te, gt_tr, train_idx, test_idx, verbose=verbose)
        oof_pred[test_idx] = log["oof_pred_te"]
        oof_probs[test_idx] = log["oof_probs_te"]
        R_wfn_all[test_idx] = log["R_wfn_te"]
        F0_all[test_idx] = log["F0_te"]
        # log 에서 array 제거 (json 직렬화용)
        fold_logs.append({
            "fold": fold,
            "elapsed_s": log["elapsed_s"],
            "grad_norm_trajectory": log["grad_norm_trajectory"],
            "N_te": int(X_te.shape[0]),
        })

    # Concat OOF metric
    err = np.linalg.norm(oof_pred - gt, axis=1)
    hit_1cm = float((err <= 0.01).mean())
    hit_1p5cm = float((err <= 0.015).mean())
    top1_argmax = oof_probs.argmax(axis=1)
    max_class_ratio = float(np.bincount(top1_argmax, minlength=K_ANCHORS).max() / N_total)

    # gt_anchor_label
    R_t_all = np.transpose(R_wfn_all, (0, 2, 1))
    gt_res_f = np.einsum("nij,nj->ni", R_t_all, gt - F0_all)
    gt_anchor_label = np.linalg.norm(
        _p022_anchors.ANCHORS_A6[None, :, :] - gt_res_f[:, None, :], axis=-1,
    ).argmin(axis=1)
    top1_acc = float((top1_argmax == gt_anchor_label).mean())

    elapsed_total = time.perf_counter() - t_total

    return {
        "cell": "X1",
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
            "batch": BATCH_SIZE, "hidden": HIDDEN, "cand_in_dim": CAND_EXT_DIM,
            "anchor_embed_dim": ANCHOR_EMBED_DIM,
            "anchor_embed_init_scale": ANCHOR_EMBED_INIT_SCALE,
            "gru_dropout": GRU_DROPOUT, "grad_clip": GRAD_CLIP, "tau_cls": TAU_CLS,
            "regime_count": REGIME_COUNT, "seed": SEED,
        },
        "oof_pred": oof_pred,         # (N_total, 3) — caller 가 dump 결정
        "oof_probs": oof_probs,       # (N_total, K) — caller 가 dump 결정
        "gt_anchor_label": gt_anchor_label.tolist(),
    }


def _smoke() -> None:
    """very small smoke — 1 fold tiny mock data, epoch=2."""
    global EPOCHS, WARMUP_EP
    EPOCHS, WARMUP_EP = 2, 1
    rng = np.random.default_rng(SEED)
    N = 16
    X_tr = rng.standard_normal((N, 11, 3)).astype(np.float32)
    X_te = rng.standard_normal((6, 11, 3)).astype(np.float32)
    gt_tr = X_tr[:, 10, :] + rng.standard_normal((N, 3)).astype(np.float32) * 0.005
    log = train_one_fold(
        fold=0, X_tr=X_tr, X_te=X_te, gt_tr=gt_tr,
        train_idx=np.arange(N), test_idx=np.arange(6),
        verbose=False,
    )
    assert log["oof_pred_te"].shape == (6, 3)
    assert log["oof_probs_te"].shape == (6, K_ANCHORS)
    assert log["R_wfn_te"].shape == (6, 3, 3)
    print(f"smoke OK: pred={log['oof_pred_te'].shape}, probs={log['oof_probs_te'].shape}, "
          f"elapsed={log['elapsed_s']:.2f}s, grad_norm[-1]={log['grad_norm_trajectory'][-1]:.4e}")


if __name__ == "__main__":
    _smoke()
