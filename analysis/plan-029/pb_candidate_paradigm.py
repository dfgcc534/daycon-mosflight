"""plan-029 E2 — PB candidate paradigm switch.

GRU-attention architecture (plan-029 GRUNetX1) 그대로 + candidate 만 PB 의
27 F0 hypothesis (`src/pb_0_6822/selector.make_candidates`) 로 교체.

paradigm 차이:
- plan-029 X1: F0=baseline 고정 + 14 anchor (Frenet residual), final = F0 + R_wfn @ (probs @ ANCHORS)
- E2:          F0 학습 (27 variants), final = (probs[:,:,None] * candidates_world).sum(axis=1)

paradigm switch 단독 효과 isolation. +0.02 회복 (≥ 0.65) 시 H_main 확정.
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

_THIS = Path(__file__).resolve().parent
_REPO = _THIS.parent.parent
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


from src.io import load_all_samples, load_labels
from src.pb_0_6822.selector import (
    make_candidates, make_candidate_features, stable_fold_id,
    CANDIDATES,
)

_seq_mod = _load(_THIS.parent / "plan-024" / "seq_builder.py", "e2_seq")
_quantile_mod = _load(_THIS.parent / "plan-024" / "quantile_carry.py", "e2_qc")
_p022_anchors = _load(_THIS.parent / "plan-022" / "anchors.py", "e2_av")
_bf = _load(_THIS.parent / "plan-020" / "baseline_f0.py", "e2_bf")
_p021_build = _load(_THIS.parent / "plan-021" / "build_input.py", "e2_bi")
_model_mod = _load(_THIS / "model.py", "e2_model")

N_FOLDS = 5
K_PB = len(CANDIDATES)            # 27
CAND_FEAT_DIM_PB = 32
SEQ_DIM = 95
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
SEED = 20260522


def _to_torch(arr: np.ndarray) -> torch.Tensor:
    return torch.nan_to_num(
        torch.from_numpy(np.ascontiguousarray(arr)).float(),
        nan=0.0, posinf=1e3, neginf=-1e3,
    )


def build_soft_label_world(gt: np.ndarray, candidates_world: np.ndarray, tau: float) -> np.ndarray:
    """gt (N,3) vs candidates_world (N,K,3) → soft label q (N,K) via softmax(-dist/tau).

    plan-022 build_soft_label_with_tau 의 paradigm 호환 식 (Frenet residual 대신 world distance).
    """
    dist = np.linalg.norm(candidates_world - gt[:, None, :], axis=2)             # (N, K)
    z = -dist / tau
    z = z - z.max(axis=1, keepdims=True)
    q = np.exp(z)
    q /= q.sum(axis=1, keepdims=True)
    return q.astype(np.float32)


def train_one_fold(fold: int, X_tr, X_te, gt_tr, train_idx, test_idx, verbose=True) -> dict:
    t0 = time.perf_counter()

    # PB candidate (world frame) + cand_feat
    cand_tr = make_candidates(X_tr, end_idx=10).astype(np.float32)                # (N_tr, 27, 3) world
    cand_te = make_candidates(X_te, end_idx=10).astype(np.float32)                # (N_te, 27, 3) world
    feat_tr = make_candidate_features(X_tr, end_idx=10, candidates=cand_tr).astype(np.float32)  # (N_tr, 27, 32)
    feat_te = make_candidate_features(X_te, end_idx=10, candidates=cand_te).astype(np.float32)

    # seq (plan-024 seq_builder 95D, train-fold quantile_carry)
    R_wfn_tr = _p021_build.build_frenet_basis_3d(X_tr, end_idx=10).astype(np.float32)
    R_wfn_te = _p021_build.build_frenet_basis_3d(X_te, end_idx=10).astype(np.float32)
    qc = _quantile_mod.build(X_tr, R_wfn_tr)
    seq_tr = _seq_mod.build(X_tr, R_wfn_tr, _p022_anchors.ANCHORS_A6, _bf.f0_baseline, qc)
    seq_te = _seq_mod.build(X_te, R_wfn_te, _p022_anchors.ANCHORS_A6, _bf.f0_baseline, qc)

    # soft label: gt vs candidates_world distance
    q_tr = build_soft_label_world(gt_tr, cand_tr, tau=TAU_CLS)                    # (N_tr, 27)

    torch.manual_seed(SEED + fold)
    model = _model_mod.GRUNetX1(
        seq_dim=SEQ_DIM, cand_in_dim=CAND_FEAT_DIM_PB, hidden=HIDDEN,
        anchor_embed_dim=ANCHOR_EMBED_DIM,
        anchor_embed_init_scale=ANCHOR_EMBED_INIT_SCALE,
        gru_dropout=GRU_DROPOUT, K=K_PB, head_skip_mode="none",
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[
            torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-6, end_factor=1.0,
                                              total_iters=WARMUP_EP),
            torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS - WARMUP_EP),
        ],
        milestones=[WARMUP_EP],
    )

    N_tr = X_tr.shape[0]
    grad_traj = []
    for epoch in range(EPOCHS):
        model.train()
        rng_ep = np.random.default_rng(SEED + epoch * 1000 + fold)
        perm = rng_ep.permutation(N_tr)
        for b_start in range(0, N_tr, BATCH_SIZE):
            idx = perm[b_start : b_start + BATCH_SIZE]
            seq_batch = _to_torch(seq_tr[idx])
            cand_batch = _to_torch(feat_tr[idx])
            q_batch = torch.from_numpy(q_tr[idx]).float()
            optimizer.zero_grad()
            score = model(seq_batch, cand_batch)                                   # (B, 27)
            log_probs = F.log_softmax(score, dim=-1)
            loss = -(q_batch * log_probs).sum(dim=-1).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
        scheduler.step()
        gn = model.anchor_embed.grad.norm().item() if model.anchor_embed.grad is not None else 0.0
        grad_traj.append(gn)
        if verbose and (epoch + 1) % 10 == 0:
            print(f"  fold={fold} ep={epoch + 1:2d}/{EPOCHS} loss={loss.item():.4f} grad={gn:.4e}")

    # eval — final_pred = (probs[:,:,None] * cand_te).sum(axis=1) — PB 식
    model.eval()
    probs_list = []
    with torch.no_grad():
        for e_start in range(0, X_te.shape[0], EVAL_BATCH):
            sb = _to_torch(seq_te[e_start : e_start + EVAL_BATCH])
            cb = _to_torch(feat_te[e_start : e_start + EVAL_BATCH])
            probs_list.append(F.softmax(model(sb, cb), dim=-1).cpu().numpy())
    probs_te = np.concatenate(probs_list, axis=0).astype(np.float32)               # (N_te, 27)
    final_pred = (probs_te[:, :, None] * cand_te).sum(axis=1).astype(np.float32)   # (N_te, 3)

    t_fold = time.perf_counter() - t0
    if verbose:
        print(f"[fold {fold}] done in {t_fold:.1f}s, final_grad={grad_traj[-1]:.4e}")
    return {
        "fold": fold, "elapsed_s": t_fold,
        "final_pred_te": final_pred, "probs_te": probs_te,
        "grad_traj": grad_traj, "N_te": int(X_te.shape[0]),
    }


def run_5fold():
    t_total = time.perf_counter()
    _ids, X = load_all_samples()
    _ids_gt, gt = load_labels()
    assert _ids == _ids_gt
    X = X.astype(np.float32)
    gt = gt.astype(np.float32)
    N_total = X.shape[0]
    folds = np.asarray([stable_fold_id(str(sid), N_FOLDS) for sid in _ids], dtype=int)

    oof_pred = np.zeros((N_total, 3), dtype=np.float32)
    oof_probs = np.zeros((N_total, K_PB), dtype=np.float32)
    fold_logs = []
    for fold in range(N_FOLDS):
        train_idx = np.where(folds != fold)[0]
        test_idx = np.where(folds == fold)[0]
        X_tr, X_te = X[train_idx], X[test_idx]
        gt_tr = gt[train_idx]
        log = train_one_fold(fold, X_tr, X_te, gt_tr, train_idx, test_idx, verbose=True)
        oof_pred[test_idx] = log["final_pred_te"]
        oof_probs[test_idx] = log["probs_te"]
        fold_logs.append({"fold": fold, "elapsed_s": log["elapsed_s"],
                          "grad_traj": log["grad_traj"], "N_te": log["N_te"]})

    err = np.linalg.norm(oof_pred - gt, axis=1)
    hit_1cm = float((err <= 0.01).mean())
    hit_1p5cm = float((err <= 0.015).mean())
    top1_argmax = oof_probs.argmax(axis=1)
    max_class_ratio = float(np.bincount(top1_argmax, minlength=K_PB).max() / N_total)

    elapsed = time.perf_counter() - t_total
    summary = {
        "cell": "PB_paradigm",
        "hit_1cm": hit_1cm,
        "hit_1p5cm": hit_1p5cm,
        "max_class_ratio": max_class_ratio,
        "elapsed_total_s": elapsed,
        "N_total": N_total,
        "K": K_PB,
        "fold_logs": fold_logs,
        "hparams": {"epochs": EPOCHS, "lr": LR, "batch": BATCH_SIZE,
                    "hidden": HIDDEN, "cand_in_dim": CAND_FEAT_DIM_PB,
                    "K_pb": K_PB, "tau_cls": TAU_CLS, "seed": SEED},
        "paradigm": "PB F0-hypothesis selection (27 candidates × world frame), "
                    "final = (probs × candidates).sum",
    }

    # Comparison
    with open(_THIS / "results_X1.json") as f:
        x1 = json.load(f)
    with open(_THIS / "ablation_head_skip.json") as f:
        abl = json.load(f)

    print(f"\n{'=' * 60}\nPB candidate paradigm switch result\n{'=' * 60}")
    print(f"PB selector OOF (plan-004):   hit_1cm = 0.6511 (baseline target)")
    print(f"plan-029 X1 (anchor residual): hit_1cm = {x1['hit_1cm']:.4f}")
    print(f"plan-029 X3 (head full skip):  hit_1cm = {abl['X3']['hit_1cm']:.4f}")
    print(f"E2 PB paradigm switch:         hit_1cm = {hit_1cm:.4f}")
    print(f"  vs X1: Δ = {hit_1cm - x1['hit_1cm']:+.4f}")
    print(f"  vs PB selector OOF (0.6511): Δ = {hit_1cm - 0.6511:+.4f}")
    print(f"  vs F0 (0.6320):              Δ = {hit_1cm - 0.6320:+.4f}")
    print(f"  elapsed: {elapsed:.1f}s ({elapsed / 60:.1f} min)")

    summary["compare"] = {
        "PB_selector_OOF": 0.6511,
        "X1_anchor_residual": x1["hit_1cm"],
        "X3_head_full_skip": abl["X3"]["hit_1cm"],
        "E2_PB_paradigm": hit_1cm,
        "delta_vs_X1": hit_1cm - x1["hit_1cm"],
        "delta_vs_PB_selector": hit_1cm - 0.6511,
        "delta_vs_F0": hit_1cm - 0.6320,
    }

    out = _THIS / "results_PB_paradigm.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[dump] {out}")
    np.savez_compressed(_THIS / "oof_PB_paradigm.npz",
                         oof_pred=oof_pred, oof_probs=oof_probs)


if __name__ == "__main__":
    run_5fold()
