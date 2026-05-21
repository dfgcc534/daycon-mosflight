"""plan-024 G2 fail 학습 진단 — 1 fold verbose training.

사용자 의심: CPU 학습 167s 가 비정상적으로 빠름.
- 단일 fold (fold 0) 만 학습
- epoch 별 train_loss + val_loss + hit_1cm + time + no_improve count + best_epoch 박제
- step-level batch time 측정
- early-stop trigger 정확한 시점 확인
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
_spec = importlib.util.spec_from_file_location("p024_mw", _THIS / "multiwindow_trim_build.py")
mw_build = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(mw_build)
_spec = importlib.util.spec_from_file_location("p024_seq", _THIS / "seq_builder.py")
seq_builder = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(seq_builder)
_spec = importlib.util.spec_from_file_location("p024_cand", _THIS / "cand_builder.py")
cand_builder = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(cand_builder)
_spec = importlib.util.spec_from_file_location("p024_model", _THIS / "model.py")
model_mod = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(model_mod)


def main():
    t0 = time.time()
    print(f"[diag] PyTorch={torch.__version__}, threads={torch.get_num_threads()}, cuda={torch.cuda.is_available()}", flush=True)
    print(f"[diag] load data ...", flush=True)
    ids, X = load_all_samples(split="train")
    label_ids, Y = load_labels()
    X = X.astype(np.float64); Y = Y.astype(np.float64)
    folds = np.asarray([stable_fold_id(str(s), 5) for s in ids], dtype=int)

    print(f"[diag] build_input_common ...", flush=True)
    common = p021_build.build_input_common(X, bf.f0_baseline)
    R_wfn_all = common["R_wfn"]
    pred_F0_world_all = common["pred_F0_world"]
    L1_frenet_all = common["L1"]

    mw_path = _THIS / "multiwindow_trim.json"
    if not mw_path.exists():
        mw_build.build_and_save(L1_frenet_all, output_path=mw_path)

    regime_bins = fit_regime_bins(X, end_idx=10)
    regimes_all = assign_regimes(X, end_idx=10, bins=regime_bins)
    anchors = anchors_mod.ANCHORS_A6
    q_true_all = som.build_soft_label_with_tau(Y, R_wfn_all, pred_F0_world_all, anchors, tau_cls=0.001)

    # fold 0 only
    tr = np.where(folds != 0)[0]
    va = np.where(folds == 0)[0]
    tr_sorted = tr[np.argsort([ids[i] for i in tr])]

    print(f"[diag] fold 0: n_tr={len(tr_sorted)} n_va={len(va)}", flush=True)

    qc = qc_mod.build(X[tr_sorted], R_wfn_all[tr_sorted])
    print(f"[diag] quantile_carry: {qc}", flush=True)

    t1 = time.time()
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
    q_true_va = q_true_all[va]
    print(f"[diag] seq/cand build elapsed {time.time()-t1:.1f}s", flush=True)
    print(f"[diag] seq_tr.shape={seq_tr.shape} cand_tr.shape={cand_tr.shape}", flush=True)

    # ── train w/ verbose log ─────────────────────────────────────
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
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[diag] model params: {n_params:,}", flush=True)

    optim = torch.optim.AdamW(model.parameters(), lr=7e-4, weight_decay=0.02)
    PRE = 12; FINE = 10; BATCH = 256; PATIENCE = 3
    total_steps = (PRE + FINE) * (val_split // BATCH + 1)
    warm_steps = max(1, total_steps // 10)
    def lr_at(step):
        if step < warm_steps: return step / max(1, warm_steps)
        progress = (step - warm_steps) / max(1, total_steps - warm_steps)
        return 0.5 * (1 + np.cos(np.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_at)

    best_val_loss = float("inf")
    best_state = None
    no_improve = 0
    best_epoch = -1
    log = []

    t_train_start = time.time()
    for epoch in range(PRE + FINE):
        t_ep = time.time()
        model.train()
        perm = np.random.permutation(val_split)
        train_losses = []
        for s_idx in range(0, val_split, BATCH):
            idx = perm[s_idx:s_idx + BATCH]
            q_pred, _ = model(seq_train[idx], cand_train[idx])
            loss = -(q_train[idx] * torch.log(q_pred + 1e-12)).sum(-1).mean()
            optim.zero_grad(); loss.backward(); optim.step(); scheduler.step()
            train_losses.append(loss.item())
        train_loss = sum(train_losses) / len(train_losses)

        # validation loss
        model.eval()
        with torch.no_grad():
            q_v, _ = model(seq_val_t, cand_val_t)
            val_loss = -(q_val_t * torch.log(q_v + 1e-12)).sum(-1).mean().item()
            # check fold-0 valid hit_1cm with current model
            q_va, _ = model(seq_va_t, cand_va_t)
            q_va_np = q_va.cpu().numpy()
            anchors_world_va = (
                np.einsum("nij,kj->nki", R_wfn_all[va], anchors.astype(np.float32))
                + pred_F0_world_all[va, None, :]
            )
            final_va = (q_va_np[:, :, None] * anchors_world_va).sum(axis=1)
            hit_1cm = float((np.linalg.norm(final_va - Y[va], axis=1) <= 0.01).mean())

        improved = val_loss < best_val_loss - 1e-5
        if improved:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            no_improve = 0
        else:
            no_improve += 1

        ep_time = time.time() - t_ep
        n_batch = len(train_losses)
        ms_per_batch = ep_time * 1000 / max(n_batch, 1)
        msg = (f"[diag] epoch {epoch:2d}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
               f"hit_1cm(va)={hit_1cm:.4f} | best@{best_epoch} no_improve={no_improve} | "
               f"n_batch={n_batch} {ep_time:.1f}s ({ms_per_batch:.0f}ms/batch)")
        print(msg, flush=True)
        log.append({
            "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
            "hit_1cm_fold0": hit_1cm, "best_epoch": best_epoch, "no_improve": no_improve,
            "n_batch": n_batch, "ep_time_s": ep_time, "ms_per_batch": ms_per_batch,
            "improved": improved,
        })

        if no_improve >= PATIENCE:
            print(f"[diag] early stop @ epoch {epoch} (patience={PATIENCE})", flush=True)
            break

    t_train = time.time() - t_train_start
    if best_state is not None:
        model.load_state_dict(best_state)

    # final predict on fold 0
    model.eval()
    with torch.no_grad():
        q_va, _ = model(seq_va_t, cand_va_t)
        q_va_np = q_va.cpu().numpy()
    anchors_world_va = (
        np.einsum("nij,kj->nki", R_wfn_all[va], anchors.astype(np.float32))
        + pred_F0_world_all[va, None, :]
    )
    final_va = (q_va_np[:, :, None] * anchors_world_va).sum(axis=1)
    hit_1cm_final = float((np.linalg.norm(final_va - Y[va], axis=1) <= 0.01).mean())
    hit_15cm_final = float((np.linalg.norm(final_va - Y[va], axis=1) <= 0.015).mean())

    print(f"\n[diag] === SUMMARY ===", flush=True)
    print(f"[diag] train_time={t_train:.1f}s total={time.time()-t0:.1f}s", flush=True)
    print(f"[diag] n_epoch ran={len(log)} (PRE+FINE={PRE+FINE})", flush=True)
    print(f"[diag] best_epoch={best_epoch} best_val_loss={best_val_loss:.4f}", flush=True)
    print(f"[diag] final fold 0 hit_1cm={hit_1cm_final:.4f} hit_1.5cm={hit_15cm_final:.4f}", flush=True)

    diag_out = {
        "pytorch_version": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "cuda": torch.cuda.is_available(),
        "n_params": n_params,
        "n_tr": len(tr_sorted), "n_va": len(va),
        "val_split": val_split,
        "train_time_s": t_train,
        "n_epoch_ran": len(log),
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "final_hit_1cm_fold0": hit_1cm_final,
        "final_hit_1.5cm_fold0": hit_15cm_final,
        "epoch_log": log,
    }
    out_path = _THIS / "diagnose_training.json"
    out_path.write_text(json.dumps(diag_out, indent=2))
    print(f"[diag] → {out_path}", flush=True)


if __name__ == "__main__":
    main()
