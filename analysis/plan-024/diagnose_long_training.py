"""plan-024 사용자 재의심 — epoch 100 + constant lr (no decay) 로 진짜 수렴 확인.

기존 spec: 22 epoch + cosine decay (lr 7e-4 → 0 over 22 epoch).
사용자 의심: GPU 5-7h 가정 대비 167s 너무 빠름 → 학습 부족 의심.

진단: cosine decay 가 epoch 11 이후 lr 빠르게 ↓. epoch 22 가 충분한지 검증.

setup:
  - 1 fold (fold 0) 만
  - epoch 100 (4.5× longer)
  - constant lr = 5e-4 (no decay)
  - no warmup
  - no early stop (epoch 100 다 돌림)
  - 매 5 epoch hit_1cm + train/val loss 로깅
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

_THIS = Path(__file__).resolve().parent
_REPO = _THIS.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.io import load_all_samples, load_labels
from src.pb_0_6822.selector import stable_fold_id, fit_regime_bins, assign_regimes

_spec = importlib.util.spec_from_file_location("p020_bf", _REPO / "analysis" / "plan-020" / "baseline_f0.py")
bf = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(bf)
_spec = importlib.util.spec_from_file_location("p022_anchors", _REPO / "analysis" / "plan-022" / "anchors.py")
anchors_mod = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(anchors_mod)
_spec = importlib.util.spec_from_file_location("p022_som", _REPO / "analysis" / "plan-022" / "selector_only_model.py")
som = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(som)
_spec = importlib.util.spec_from_file_location("p021_build", _REPO / "analysis" / "plan-021" / "build_input.py")
p021_build = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(p021_build)
_spec = importlib.util.spec_from_file_location("p024_qc", _THIS / "quantile_carry.py")
qc_mod = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(qc_mod)
_spec = importlib.util.spec_from_file_location("p024_seq", _THIS / "seq_builder.py")
seq_builder = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(seq_builder)
_spec = importlib.util.spec_from_file_location("p024_cand", _THIS / "cand_builder.py")
cand_builder = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(cand_builder)
_spec = importlib.util.spec_from_file_location("p024_model", _THIS / "model.py")
model_mod = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(model_mod)


def main():
    t0 = time.time()
    EPOCHS = 100
    LR = 5e-4
    BATCH = 256
    LOG_EVERY = 5

    print(f"[long-diag] EPOCHS={EPOCHS}, lr={LR} constant, batch={BATCH}, log every {LOG_EVERY}", flush=True)
    print(f"[long-diag] load ...", flush=True)
    ids, X = load_all_samples(split="train")
    label_ids, Y = load_labels()
    X = X.astype(np.float64); Y = Y.astype(np.float64)
    folds = np.asarray([stable_fold_id(str(s), 5) for s in ids], dtype=int)
    common = p021_build.build_input_common(X, bf.f0_baseline)
    R_wfn_all = common["R_wfn"]; pred_F0_world_all = common["pred_F0_world"]
    L1_frenet_all = common["L1"]
    regime_bins = fit_regime_bins(X, end_idx=10)
    regimes_all = assign_regimes(X, end_idx=10, bins=regime_bins)
    anchors = anchors_mod.ANCHORS_A6
    q_true_all = som.build_soft_label_with_tau(Y, R_wfn_all, pred_F0_world_all, anchors, tau_cls=0.001)

    # fold 0 only
    tr = np.where(folds != 0)[0]
    va = np.where(folds == 0)[0]
    tr_sorted = tr[np.argsort([ids[i] for i in tr])]
    qc = qc_mod.build(X[tr_sorted], R_wfn_all[tr_sorted])
    mw_path = _THIS / "multiwindow_trim.json"

    print(f"[long-diag] build seq/cand ...", flush=True)
    seq_tr = seq_builder.build(X[tr_sorted], R_wfn_all[tr_sorted], anchors, bf.f0_baseline,
                                quantile_carry=qc, tau_past=0.003)
    cand_tr = cand_builder.build(X[tr_sorted], R_wfn_all[tr_sorted], pred_F0_world_all[tr_sorted],
                                  anchors, bf.f0_baseline, regimes=regimes_all[tr_sorted],
                                  quantile_carry=qc, multiwindow_trim_path=mw_path)
    seq_va = seq_builder.build(X[va], R_wfn_all[va], anchors, bf.f0_baseline,
                                quantile_carry=qc, tau_past=0.003)
    cand_va = cand_builder.build(X[va], R_wfn_all[va], pred_F0_world_all[va], anchors, bf.f0_baseline,
                                  regimes=regimes_all[va], quantile_carry=qc,
                                  multiwindow_trim_path=mw_path)
    q_true_tr = q_true_all[tr_sorted]

    torch.manual_seed(20260521); np.random.seed(20260521)
    n_tr = seq_tr.shape[0]
    val_split = int(n_tr * 0.8)
    seq_train = torch.from_numpy(seq_tr[:val_split]).float()
    cand_train = torch.from_numpy(cand_tr[:val_split]).float()
    q_train = torch.from_numpy(q_true_tr[:val_split]).float()
    seq_val_t = torch.from_numpy(seq_tr[val_split:]).float()
    cand_val_t = torch.from_numpy(cand_tr[val_split:]).float()
    q_val_t = torch.from_numpy(q_true_tr[val_split:]).float()
    seq_va_t = torch.from_numpy(seq_va).float()
    cand_va_t = torch.from_numpy(cand_va).float()

    model = model_mod.CrossAttentionAnchorSelector()
    optim = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.02)
    # NO scheduler — constant lr

    log = []
    t_train_start = time.time()
    best_hit_1cm = 0.0
    best_val_loss = float("inf")
    best_epoch = -1

    for epoch in range(EPOCHS):
        t_ep = time.time()
        model.train()
        perm = np.random.permutation(val_split)
        train_losses = []
        for s_idx in range(0, val_split, BATCH):
            idx = perm[s_idx:s_idx + BATCH]
            q_pred, _ = model(seq_train[idx], cand_train[idx])
            loss = -(q_train[idx] * torch.log(q_pred + 1e-12)).sum(-1).mean()
            optim.zero_grad(); loss.backward(); optim.step()
            train_losses.append(loss.item())
        train_loss = sum(train_losses) / len(train_losses)

        # val (only log every LOG_EVERY epochs)
        if epoch % LOG_EVERY == 0 or epoch == EPOCHS - 1:
            model.eval()
            with torch.no_grad():
                q_v, _ = model(seq_val_t, cand_val_t)
                val_loss = -(q_val_t * torch.log(q_v + 1e-12)).sum(-1).mean().item()
                q_va, _ = model(seq_va_t, cand_va_t)
                q_va_np = q_va.cpu().numpy()
                anchors_world_va = (
                    np.einsum("nij,kj->nki", R_wfn_all[va], anchors.astype(np.float32))
                    + pred_F0_world_all[va, None, :]
                )
                final_va = (q_va_np[:, :, None] * anchors_world_va).sum(axis=1)
                hit_1cm = float((np.linalg.norm(final_va - Y[va], axis=1) <= 0.01).mean())

            if hit_1cm > best_hit_1cm:
                best_hit_1cm = hit_1cm
                best_epoch = epoch
            if val_loss < best_val_loss:
                best_val_loss = val_loss

            ep_time = time.time() - t_ep
            msg = (f"[long-diag] epoch {epoch:3d}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                   f"hit_1cm={hit_1cm:.4f} {ep_time:.1f}s")
            print(msg, flush=True)
            log.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                        "hit_1cm": hit_1cm, "ep_time_s": ep_time})

    t_train = time.time() - t_train_start
    print(f"\n[long-diag] === SUMMARY ===", flush=True)
    print(f"[long-diag] total epochs={EPOCHS} time={t_train:.0f}s", flush=True)
    print(f"[long-diag] best hit_1cm={best_hit_1cm:.4f} @ epoch {best_epoch}", flush=True)
    print(f"[long-diag] best val_loss={best_val_loss:.4f}", flush=True)
    print(f"[long-diag] *** plan-022 winner 0.6528 = {best_hit_1cm - 0.6528:+.4f} ***", flush=True)

    out = {
        "config": {"epochs": EPOCHS, "lr": LR, "batch": BATCH, "scheduler": "constant_no_decay"},
        "train_time_s": t_train,
        "best_hit_1cm": best_hit_1cm,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "epoch_log": log,
    }
    out_path = _THIS / "diagnose_long_training.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"[long-diag] → {out_path}", flush=True)


if __name__ == "__main__":
    main()
