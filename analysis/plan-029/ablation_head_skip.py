"""plan-029 follow-up — head_skip_mode ablation (사용자 요청, 2026-05-22).

X2 cell: head_in = concat(event_ctx, cand_ext minus anchor_spec 9D) — anchor identity 만 차단
X3 cell: head_in = concat(event_ctx, cand_ext full 165D) — PB framework 유사 raw skip 부활

가설: lever (d) head raw skip 차단이 paradigm-essential 이었다면
  X2 / X3 hit_1cm > X1 hit_1cm = 0.6316 (회복)
  특히 X3 ≥ plan-024 ceiling 0.6387 → "raw skip = main signal carrier" 입증
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


_train = _load(_THIS / "train.py", "p029_abl_train")
_model = _load(_THIS / "model.py", "p029_abl_model")


def run_one_cell(cell_name: str, head_skip_mode: str, verbose: bool = True) -> dict:
    """Same as _train.run_5fold_oof but with head_skip_mode override."""
    t_total = time.perf_counter()
    from src.io import load_all_samples, load_labels
    from src.pb_0_6822.selector import stable_fold_id, fit_regime_bins, assign_regimes

    _ids, X = load_all_samples()
    _ids_gt, gt = load_labels()
    assert _ids == _ids_gt
    X = X.astype(np.float32)
    gt = gt.astype(np.float32)
    N_total = X.shape[0]

    folds = np.asarray([stable_fold_id(str(sid), _train.N_FOLDS) for sid in _ids], dtype=int)

    oof_pred = np.zeros((N_total, 3), dtype=np.float32)
    oof_probs = np.zeros((N_total, _train.K_ANCHORS), dtype=np.float32)
    R_wfn_all = np.zeros((N_total, 3, 3), dtype=np.float32)
    F0_all = np.zeros((N_total, 3), dtype=np.float32)
    fold_logs = []

    for fold in range(_train.N_FOLDS):
        train_idx = np.where(folds != fold)[0]
        test_idx = np.where(folds == fold)[0]
        X_tr, X_te = X[train_idx], X[test_idx]
        gt_tr = gt[train_idx]
        # train_one_fold logic 재사용 - 단 model 생성 부분만 override
        # 가장 깔끔한 방법은 train_one_fold 안쪽 복사. 단 cell 옵션 추가가 spec drift 라
        # 본 ablation 에서는 train_one_fold 의 forward + model 부분만 inline 모사.
        t0 = time.perf_counter()
        R_wfn_tr = _train._p021_build.build_frenet_basis_3d(X_tr, end_idx=10).astype(np.float32)
        R_wfn_te = _train._p021_build.build_frenet_basis_3d(X_te, end_idx=10).astype(np.float32)
        F0_tr = _train._bf.f0_baseline(X_tr, end_idx=10).astype(np.float32)
        F0_te = _train._bf.f0_baseline(X_te, end_idx=10).astype(np.float32)
        qc = _train._quantile_mod.build(X_tr, R_wfn_tr)
        regime_bins_tr = fit_regime_bins(X_tr, end_idx=10)
        regimes_tr = assign_regimes(X_tr, end_idx=10, bins=regime_bins_tr)
        regimes_te = assign_regimes(X_te, end_idx=10, bins=regime_bins_tr)
        regime_anchor_table_tr = _train._aqe.build_regime_anchor_lookup(
            gt_train=gt_tr, regimes_train=regimes_tr,
            ANCHORS_A6=_train._p022_anchors.ANCHORS_A6,
            R_wfn_train=R_wfn_tr, F0_train=F0_tr,
            regime_count=_train.REGIME_COUNT, laplace=1.0,
        )
        cand_ext_tr = _train._aqe.build(
            X_tr, R_wfn_tr, F0_tr, _train._p022_anchors.ANCHORS_A6, _train._bf.f0_baseline,
            regimes=regimes_tr, quantile_carry=qc,
            regime_count=_train.REGIME_COUNT, regime_anchor_table=regime_anchor_table_tr,
        )
        cand_ext_te = _train._aqe.build(
            X_te, R_wfn_te, F0_te, _train._p022_anchors.ANCHORS_A6, _train._bf.f0_baseline,
            regimes=regimes_te, quantile_carry=qc,
            regime_count=_train.REGIME_COUNT, regime_anchor_table=regime_anchor_table_tr,
        )
        seq_tr = _train._seq_mod.build(
            X_tr, R_wfn_tr, _train._p022_anchors.ANCHORS_A6, _train._bf.f0_baseline, qc)
        seq_te = _train._seq_mod.build(
            X_te, R_wfn_te, _train._p022_anchors.ANCHORS_A6, _train._bf.f0_baseline, qc)
        q_tr = _train._p022_soft.build_soft_label_with_tau(
            gt_tr, R_wfn_tr, F0_tr, _train._p022_anchors.ANCHORS_A6, tau_cls=_train.TAU_CLS,
        )

        torch.manual_seed(_train.SEED + fold)
        model = _model.GRUNetX1(
            seq_dim=_train.SEQ_DIM, cand_in_dim=_train.CAND_EXT_DIM, hidden=_train.HIDDEN,
            anchor_embed_dim=_train.ANCHOR_EMBED_DIM,
            anchor_embed_init_scale=_train.ANCHOR_EMBED_INIT_SCALE,
            gru_dropout=_train.GRU_DROPOUT, K=_train.K_ANCHORS,
            head_skip_mode=head_skip_mode,
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=_train.LR, weight_decay=_train.WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[
                torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-6, end_factor=1.0,
                                                  total_iters=_train.WARMUP_EP),
                torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=_train.EPOCHS - _train.WARMUP_EP),
            ],
            milestones=[_train.WARMUP_EP],
        )

        N_tr = X_tr.shape[0]
        grad_norm_trajectory = []
        for epoch in range(_train.EPOCHS):
            model.train()
            rng_ep = np.random.default_rng(_train.SEED + epoch * 1000 + fold)
            perm = rng_ep.permutation(N_tr)
            for b_start in range(0, N_tr, _train.BATCH_SIZE):
                idx = perm[b_start : b_start + _train.BATCH_SIZE]
                seq_batch = _train._to_torch(seq_tr[idx])
                cand_batch = _train._to_torch(cand_ext_tr[idx])
                q_batch = torch.from_numpy(q_tr[idx]).float()
                optimizer.zero_grad()
                score = model(seq_batch, cand_batch)
                log_probs = F.log_softmax(score, dim=-1)
                loss = -(q_batch * log_probs).sum(dim=-1).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), _train.GRAD_CLIP)
                optimizer.step()
            scheduler.step()
            grad_norm = (model.anchor_embed.grad.norm().item()
                         if model.anchor_embed.grad is not None else 0.0)
            grad_norm_trajectory.append(grad_norm)
            if verbose and (epoch + 1) % 10 == 0:
                print(f"  [{cell_name}] fold={fold} ep={epoch + 1:2d}/{_train.EPOCHS} "
                      f"loss={loss.item():.4f} grad_norm={grad_norm:.4e}")

        model.eval()
        probs_te_list = []
        with torch.no_grad():
            for e_start in range(0, X_te.shape[0], _train.EVAL_BATCH):
                sb = _train._to_torch(seq_te[e_start : e_start + _train.EVAL_BATCH])
                cb = _train._to_torch(cand_ext_te[e_start : e_start + _train.EVAL_BATCH])
                sc = model(sb, cb)
                probs_te_list.append(F.softmax(sc, dim=-1).cpu().numpy())
        probs_te = np.concatenate(probs_te_list, axis=0).astype(np.float32)
        residual_frenet = (probs_te[:, :, None] * _train._p022_anchors.ANCHORS_A6[None, :, :]).sum(axis=1)
        residual_world = np.einsum("nij,nj->ni", R_wfn_te, residual_frenet)
        final_pred = (F0_te + residual_world).astype(np.float32)

        t_fold = time.perf_counter() - t0
        if verbose:
            print(f"[{cell_name} fold {fold}] done in {t_fold:.1f}s, "
                  f"final_grad={grad_norm_trajectory[-1]:.4e}")

        oof_pred[test_idx] = final_pred
        oof_probs[test_idx] = probs_te
        R_wfn_all[test_idx] = R_wfn_te
        F0_all[test_idx] = F0_te
        fold_logs.append({
            "fold": fold, "elapsed_s": t_fold,
            "grad_norm_trajectory": grad_norm_trajectory,
            "N_te": int(X_te.shape[0]),
        })

    err = np.linalg.norm(oof_pred - gt, axis=1)
    hit_1cm = float((err <= 0.01).mean())
    hit_1p5cm = float((err <= 0.015).mean())
    top1_argmax = oof_probs.argmax(axis=1)
    max_class_ratio = float(np.bincount(top1_argmax, minlength=14).max() / N_total)

    elapsed_total = time.perf_counter() - t_total
    return {
        "cell": cell_name,
        "head_skip_mode": head_skip_mode,
        "head_in_dim": _model.GRUNetX1(head_skip_mode=head_skip_mode).head_in_dim,
        "hit_1cm": hit_1cm,
        "hit_1p5cm": hit_1p5cm,
        "max_class_ratio": max_class_ratio,
        "elapsed_total_s": elapsed_total,
        "fold_logs": fold_logs,
    }


def main():
    results = {}
    for cell, mode in [("X2", "no_anchor_spec"), ("X3", "full")]:
        print(f"\n{'=' * 60}\n[{cell}] head_skip_mode={mode}\n{'=' * 60}")
        r = run_one_cell(cell, mode, verbose=True)
        results[cell] = r
        print(f"\n[{cell}] DONE — hit_1cm={r['hit_1cm']:.4f} hit_1p5cm={r['hit_1p5cm']:.4f} "
              f"max_class_ratio={r['max_class_ratio']:.4f} elapsed={r['elapsed_total_s']:.1f}s")

    # X1 비교 (carry from results_X1.json)
    with open(_THIS / "results_X1.json") as f:
        x1 = json.load(f)
    print(f"\n{'=' * 60}\nablation_summary\n{'=' * 60}")
    print(f"X1 (head_in=event_ctx 196 only):        hit_1cm={x1['hit_1cm']:.4f}")
    print(f"X2 (head_in=event_ctx + cand 156):      hit_1cm={results['X2']['hit_1cm']:.4f} "
          f"Δ={results['X2']['hit_1cm'] - x1['hit_1cm']:+.4f}")
    print(f"X3 (head_in=event_ctx + cand_full 165): hit_1cm={results['X3']['hit_1cm']:.4f} "
          f"Δ={results['X3']['hit_1cm'] - x1['hit_1cm']:+.4f}")
    print(f"  F0 baseline = 0.6320 / plan-024 ceiling = 0.6387 / plan-022 winner = 0.6531")

    summary = {
        "X1": {"hit_1cm": x1["hit_1cm"], "hit_1p5cm": x1["hit_1p5cm"],
               "max_class_ratio": x1["max_class_ratio"], "head_in_dim": 196,
               "head_skip_mode": "none"},
        "X2": results["X2"],
        "X3": results["X3"],
        "deltas": {
            "X2_vs_X1": results["X2"]["hit_1cm"] - x1["hit_1cm"],
            "X3_vs_X1": results["X3"]["hit_1cm"] - x1["hit_1cm"],
            "X3_vs_F0": results["X3"]["hit_1cm"] - 0.6320,
            "X3_vs_p024_ceiling": results["X3"]["hit_1cm"] - 0.6387,
            "X3_vs_p022_winner": results["X3"]["hit_1cm"] - 0.6531,
        },
    }
    out = _THIS / "ablation_head_skip.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[dump] {out}")


if __name__ == "__main__":
    main()
