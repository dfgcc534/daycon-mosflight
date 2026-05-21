"""plan-024 c7 — 5-fold OOF runner (§6, v1.1-rev2).

Pipeline per fold:
  1. quantile_carry.build(X[tr], R_wfn[tr])  → train-only quantile dict.
  2. seq/cand build (train + valid, same quantile_carry, fold-leakage 차단).
  3. CrossAttentionAnchorSelector fit
     (AdamW lr=7e-4 weight_decay=0.02, cosine + warm-up 10%, 12 pre + 10 fine
     epoch, batch=256, dropout 0.10/0.15, early stop patience=3 on val 20%).
  4. Predict q_pred on valid.
  5. final_position = R_wfn[va] @ (q_pred @ anchors_frenet) + pred_F0_world[va].

OOF concat → metric:
  - hit_1cm / hit_1.5cm / Δ_F0
  - max_class_ratio (probs.mean(0).max())
  - q_true_max, dist_match_KL, top1_acc, soft_CE
  - **gap_ranking** = oracle_1cm - argmax_hit  (§3.6 식)

CLI:
  python analysis/plan-024/run_oof.py --out-json analysis/plan-024/results_xattn.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

_THIS = Path(__file__).resolve().parent
_REPO = _THIS.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.io import load_all_samples, load_labels                          # noqa: E402
from src.pb_0_6822.selector import stable_fold_id, fit_regime_bins, assign_regimes  # noqa: E402

# plan-020 baseline_f0
_spec = importlib.util.spec_from_file_location(
    "p020_bf", _REPO / "analysis" / "plan-020" / "baseline_f0.py"
)
bf = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(bf)

# plan-022 anchors
_spec = importlib.util.spec_from_file_location(
    "p022_anchors", _REPO / "analysis" / "plan-022" / "anchors.py"
)
anchors_mod = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(anchors_mod)

# plan-022 build_soft_label_with_tau (output target)
_spec = importlib.util.spec_from_file_location(
    "p022_som", _REPO / "analysis" / "plan-022" / "selector_only_model.py"
)
som = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(som)

# plan-021 build_input (R_wfn, pred_F0_world)
_spec = importlib.util.spec_from_file_location(
    "p021_build", _REPO / "analysis" / "plan-021" / "build_input.py"
)
p021_build = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(p021_build)

# plan-024 modules
_spec = importlib.util.spec_from_file_location(
    "p024_quantile", _THIS / "quantile_carry.py"
)
quantile_carry_mod = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(quantile_carry_mod)

_spec = importlib.util.spec_from_file_location(
    "p024_mw_build", _THIS / "multiwindow_trim_build.py"
)
mw_build = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(mw_build)

_spec = importlib.util.spec_from_file_location(
    "p024_seq", _THIS / "seq_builder.py"
)
seq_builder = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(seq_builder)

_spec = importlib.util.spec_from_file_location(
    "p024_cand", _THIS / "cand_builder.py"
)
cand_builder = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(cand_builder)

_spec = importlib.util.spec_from_file_location(
    "p024_model", _THIS / "model.py"
)
model_mod = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(model_mod)


N_FOLDS = 5
R_HIT = 0.01
R_HIT_LOOSE = 0.015
TAU_CLS = 0.001
TAU_PAST = 0.003
SEED = 20260521
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ── fold split ─────────────────────────────────────────────────────────


def assign_folds(ids: list[str]) -> np.ndarray:
    return np.asarray([stable_fold_id(str(s), N_FOLDS) for s in ids], dtype=int)


# ── training loop ──────────────────────────────────────────────────────


def train_one_fold(
    seq_tr: np.ndarray, cand_tr: np.ndarray, q_true_tr: np.ndarray,
    seq_va: np.ndarray, cand_va: np.ndarray,
    *,
    pre_epochs: int = 12,
    fine_epochs: int = 10,
    batch_size: int = 256,
    lr: float = 7e-4,
    weight_decay: float = 0.02,
    val_frac: float = 0.2,
    patience: int = 3,
    seed: int = SEED,
    device: str = DEVICE,
) -> np.ndarray:
    """fit one fold + early stop on val split + predict valid. returns q_pred (n_va, 14)."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    n_tr = seq_tr.shape[0]
    val_split = int(n_tr * (1.0 - val_frac))
    # deterministic ordering: caller가 sample_id sort 된 인덱스로 들어옴
    seq_train = torch.from_numpy(seq_tr[:val_split]).float()
    cand_train = torch.from_numpy(cand_tr[:val_split]).float()
    q_train = torch.from_numpy(q_true_tr[:val_split]).float()
    seq_val = torch.from_numpy(seq_tr[val_split:]).float().to(device)
    cand_val = torch.from_numpy(cand_tr[val_split:]).float().to(device)
    q_val = torch.from_numpy(q_true_tr[val_split:]).float().to(device)

    model = model_mod.CrossAttentionAnchorSelector().to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = (pre_epochs + fine_epochs) * (val_split // batch_size + 1)
    warm_steps = max(1, total_steps // 10)
    def lr_at(step: int) -> float:
        if step < warm_steps:
            return step / max(1, warm_steps)
        progress = (step - warm_steps) / max(1, total_steps - warm_steps)
        return 0.5 * (1 + np.cos(np.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_at)

    best_val_loss = float("inf")
    best_state = None
    no_improve = 0
    global_step = 0

    for epoch in range(pre_epochs + fine_epochs):
        model.train()
        perm = np.random.permutation(val_split)
        for s_idx in range(0, val_split, batch_size):
            idx = perm[s_idx:s_idx + batch_size]
            seq_b = seq_train[idx].to(device, non_blocking=True)
            cand_b = cand_train[idx].to(device, non_blocking=True)
            q_b = q_train[idx].to(device, non_blocking=True)
            q_pred, _ = model(seq_b, cand_b)
            loss = -(q_b * torch.log(q_pred + 1e-12)).sum(-1).mean()
            optim.zero_grad()
            loss.backward()
            optim.step()
            scheduler.step()
            global_step += 1

        # validation
        model.eval()
        with torch.no_grad():
            q_v, _ = model(seq_val, cand_val)
            val_loss = -(q_val * torch.log(q_v + 1e-12)).sum(-1).mean().item()

        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    # load best state
    if best_state is not None:
        model.load_state_dict(best_state)

    # predict valid (input arg: seq_va, cand_va — the actual va fold)
    model.eval()
    seq_va_t = torch.from_numpy(seq_va).float().to(device)
    cand_va_t = torch.from_numpy(cand_va).float().to(device)
    with torch.no_grad():
        q_pred_va, _ = model(seq_va_t, cand_va_t)
    return q_pred_va.cpu().numpy().astype(np.float32)


# ── 5-fold OOF + metric ────────────────────────────────────────────────


def run_oof(
    X: np.ndarray, Y: np.ndarray, ids: list[str],
    *,
    multiwindow_trim_path: str | Path = _THIS / "multiwindow_trim.json",
    verbose: bool = True,
) -> dict:
    """5-fold OOF. X (N, 11, 3), Y (N, 3)."""
    N = X.shape[0]
    K = 14
    folds = assign_folds(ids)
    if verbose:
        print(f"[run_oof] N={N} folds={np.bincount(folds).tolist()}", flush=True)

    # plan-021 carry: build_input_common → R_wfn, pred_F0_world (sample 별 동일 식)
    common = p021_build.build_input_common(X, bf.f0_baseline)
    R_wfn_all = common["R_wfn"]               # (N, 3, 3)
    pred_F0_world_all = common["pred_F0_world"]   # (N, 3)
    L1_frenet_all = common["L1"]              # (N, 11, 9) — for trim_build

    # multi-window trim build (STAGE 0, fold-invariant single trim)
    trim_path = Path(multiwindow_trim_path)
    if not trim_path.exists():
        if verbose:
            print(f"[run_oof] multiwindow_trim.json 없음 → build (full train)", flush=True)
        mw_build.build_and_save(L1_frenet_all, output_path=trim_path)

    # regime bins (full train fit, fold-invariant)
    regime_bins = fit_regime_bins(X, end_idx=10)
    regimes_all = assign_regimes(X, end_idx=10, bins=regime_bins)

    # anchor codebook
    anchors = anchors_mod.ANCHORS_A6        # (14, 3) Frenet

    # output soft label (full data, fold-leakage 면제 since pure function of Y/R/F0/anchors)
    q_true_all = som.build_soft_label_with_tau(
        Y, R_wfn_all, pred_F0_world_all, anchors, tau_cls=TAU_CLS
    )                                         # (N, 14)

    # per fold
    probs_all = np.zeros((N, K), dtype=np.float32)

    for k in range(N_FOLDS):
        t0 = time.time()
        tr = np.where(folds != k)[0]
        va = np.where(folds == k)[0]
        # deterministic ordering: tr by sample_id ascending sort
        tr_sorted = tr[np.argsort([ids[i] for i in tr])]

        # train-only quantile carry
        qc = quantile_carry_mod.build(X[tr_sorted], R_wfn_all[tr_sorted])

        # build seq + cand for train (sorted) + valid
        seq_tr = seq_builder.build(
            X[tr_sorted], R_wfn_all[tr_sorted], anchors, bf.f0_baseline,
            quantile_carry=qc, tau_past=TAU_PAST,
        )
        cand_tr = cand_builder.build(
            X[tr_sorted], R_wfn_all[tr_sorted], pred_F0_world_all[tr_sorted],
            anchors, bf.f0_baseline,
            regimes=regimes_all[tr_sorted], quantile_carry=qc,
            multiwindow_trim_path=trim_path,
        )
        seq_va = seq_builder.build(
            X[va], R_wfn_all[va], anchors, bf.f0_baseline,
            quantile_carry=qc, tau_past=TAU_PAST,
        )
        cand_va = cand_builder.build(
            X[va], R_wfn_all[va], pred_F0_world_all[va], anchors, bf.f0_baseline,
            regimes=regimes_all[va], quantile_carry=qc,
            multiwindow_trim_path=trim_path,
        )
        q_true_tr = q_true_all[tr_sorted]

        if verbose:
            print(f"  fold {k}: tr={len(tr_sorted)} va={len(va)} seq.shape={seq_tr.shape} "
                  f"cand.shape={cand_tr.shape}", flush=True)

        # train + predict
        q_pred_va = train_one_fold(seq_tr, cand_tr, q_true_tr, seq_va, cand_va)
        probs_all[va] = q_pred_va

        # per-fold hit rate (sanity)
        anchors_world = (
            np.einsum("nij,kj->nki", R_wfn_all[va], anchors.astype(np.float32))
            + pred_F0_world_all[va, None, :]
        )                                       # (n_va, 14, 3)
        final_world = (q_pred_va[:, :, None] * anchors_world).sum(axis=1)  # (n_va, 3)
        d = np.linalg.norm(final_world - Y[va], axis=1)
        if verbose:
            print(f"    fold {k} hit@1cm={(d <= R_HIT).mean():.4f} time={time.time()-t0:.0f}s",
                  flush=True)

    # ── OOF metric concat ────────────────────────────────────────────
    anchors_world_all = (
        np.einsum("nij,kj->nki", R_wfn_all, anchors.astype(np.float32))
        + pred_F0_world_all[:, None, :]
    )                                            # (N, 14, 3)
    final_world_all = (probs_all[:, :, None] * anchors_world_all).sum(axis=1)
    d_cell = np.linalg.norm(final_world_all - Y, axis=1)
    pred_F0_dist = np.linalg.norm(pred_F0_world_all - Y, axis=1)

    hit_1cm = float((d_cell <= R_HIT).mean())
    hit_15cm = float((d_cell <= R_HIT_LOOSE).mean())
    f0_hit_1cm = float((pred_F0_dist <= R_HIT).mean())
    f0_hit_15cm = float((pred_F0_dist <= R_HIT_LOOSE).mean())

    # gap_ranking (§3.6)
    oracle_dist = np.linalg.norm(anchors_world_all - Y[:, None, :], axis=2)  # (N, 14)
    oracle_1cm = float((oracle_dist.min(axis=1) <= R_HIT).mean())
    argmax_idx = probs_all.argmax(axis=1)                                     # (N,)
    argmax_pos = anchors_world_all[np.arange(N), argmax_idx]
    argmax_hit = float((np.linalg.norm(argmax_pos - Y, axis=1) <= R_HIT).mean())
    gap_ranking = oracle_1cm - argmax_hit

    # distribution diagnostics
    probs_mean = probs_all.mean(axis=0)
    q_true_mean = q_true_all.mean(axis=0)
    max_class_ratio = float(probs_mean.max())
    q_true_max = float(q_true_mean.max())
    eps = 1e-12
    dist_match_KL = float(
        (probs_mean * (np.log(probs_mean + eps) - np.log(q_true_mean + eps))).sum()
    )
    pred_top1 = probs_all.argmax(axis=1)
    true_top1 = q_true_all.argmax(axis=1)
    top1_acc = float((pred_top1 == true_top1).mean())
    soft_CE = float(
        -(q_true_all.astype(np.float64) * np.log(probs_all.astype(np.float64) + eps))
        .sum(axis=1).mean()
    )

    return {
        "N": int(N),
        "hit_1cm": hit_1cm,
        "hit_1.5cm": hit_15cm,
        "delta_1cm": hit_1cm - f0_hit_1cm,
        "delta_1.5cm": hit_15cm - f0_hit_15cm,
        "f0_hit_1cm": f0_hit_1cm,
        "f0_hit_1.5cm": f0_hit_15cm,
        "max_class_ratio": max_class_ratio,
        "q_true_max": q_true_max,
        "dist_match_KL": dist_match_KL,
        "top1_acc": top1_acc,
        "soft_CE": soft_CE,
        "oracle_1cm": oracle_1cm,
        "argmax_hit": argmax_hit,
        "gap_ranking": gap_ranking,
    }


# ── CLI ────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-json", type=Path, default=_THIS / "results_xattn.json")
    args = ap.parse_args()

    t0 = time.time()
    print(f"[plan-024 run_oof] loading data ... device={DEVICE}", flush=True)
    ids, X = load_all_samples(split="train")
    label_ids, Y = load_labels()
    assert ids == label_ids, "ids mismatch"
    X = X.astype(np.float64)
    Y = Y.astype(np.float64)

    out = run_oof(X, Y, ids)
    out["elapsed_sec"] = float(time.time() - t0)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(out, indent=2))
    print(f"\n[run_oof] hit_1cm={out['hit_1cm']:.4f} hit_1.5cm={out['hit_1.5cm']:.4f} "
          f"Δ_1cm={out['delta_1cm']:+.4f} gap_ranking={out['gap_ranking']:.4f}", flush=True)
    print(f"[run_oof] total {out['elapsed_sec']:.1f}s → {args.out_json}", flush=True)


if __name__ == "__main__":
    main()
