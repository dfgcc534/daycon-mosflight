"""GRU diagnostic experiments for plan-021 (자체실험).

Hypothesis tests:
  - V0 (baseline): hidden=64, layers=1, flat=35D, epochs=30  → reproduce 0.6408 / 0.8100
  - V1 (+feat):    hidden=64, layers=1, flat=71D (+9 macro +27 EWMA), epochs=30
  - V2 (+cap):     hidden=128, layers=2, flat=35D, epochs=50
  - V3 (+both):    hidden=128, layers=2, flat=71D, epochs=50

5-fold OOF, 3 seeds, best-on-train selection (per plan-021 spec).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.io import load_all_samples, load_labels  # noqa: E402
from src.pb_0_6822.selector import stable_fold_id  # noqa: E402

_THIS_DIR = Path(__file__).parent
_PLAN020_DIR = REPO_ROOT / "analysis" / "plan-020"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


bf = _load("baseline_f0", _PLAN020_DIR / "baseline_f0.py")
bi = _load("build_input", _THIS_DIR / "build_input.py")
dh = _load("dual_head_model", _THIS_DIR / "dual_head_model.py")


N_FOLDS = 5
END_IDX = 10
R_HIT = 0.01
R_HIT_LOOSE = 0.015
ANCHOR_RADIUS = 0.005


# ── flexible GRU (hidden/layers/flat configurable) ─────────────────────


class FlexGRUDualHead(nn.Module):
    def __init__(self, seq_dim=9, hidden=64, num_layers=1, flat_dim=35, dropout=0.1):
        super().__init__()
        self.gru = nn.GRU(input_size=seq_dim, hidden_size=hidden, num_layers=num_layers,
                          batch_first=True, bidirectional=False,
                          dropout=dropout if num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.clf_head = nn.Linear(hidden + flat_dim, 7)
        self.reg_head = nn.Linear(hidden + flat_dim, 21)
        self.register_buffer("ANCHORS", torch.tensor(bi.ANCHORS_FRENET, dtype=torch.float32))

    def forward(self, seq, flat):
        out, _ = self.gru(seq)
        seq_hidden = self.dropout(out[:, -1, :])
        combined = torch.cat([seq_hidden, flat], dim=-1)
        logits = self.clf_head(combined)
        reg_raw = self.reg_head(combined).view(-1, 7, 3)
        reg_offset = torch.tanh(reg_raw) * ANCHOR_RADIUS
        return logits, reg_offset

    def predict_world(self, seq, flat, R_wfn, pred_F0_world):
        logits, reg_offset = self.forward(seq, flat)
        probs = torch.softmax(logits, dim=1)
        combined = self.ANCHORS[None] + reg_offset
        final_frenet = (probs[:, :, None] * combined).sum(dim=1)
        return torch.einsum("nij,nj->ni", R_wfn, final_frenet) + pred_F0_world


def assign_folds(ids):
    return np.asarray([stable_fold_id(str(s), N_FOLDS) for s in ids], dtype=int)


def hit_rates(pred, gt, folds):
    d = np.linalg.norm(pred - gt, axis=1)
    per_1, per_15 = [], []
    for k in range(N_FOLDS):
        m = folds == k
        per_1.append(float((d[m] <= R_HIT).mean()))
        per_15.append(float((d[m] <= R_HIT_LOOSE).mean()))
    return {
        "hit_1cm_5fold_concat": float((d <= R_HIT).mean()),
        "hit_1.5cm_5fold_concat": float((d <= R_HIT_LOOSE).mean()),
        "hit_1cm_per_fold": per_1,
        "hit_1.5cm_per_fold": per_15,
        "fold_variance_1cm": float(np.std(per_1)),
        "fold_variance_1.5cm": float(np.std(per_15)),
    }


def paired_delta(p_c, p_f, gt, R):
    d_c = np.linalg.norm(p_c - gt, axis=1)
    d_f = np.linalg.norm(p_f - gt, axis=1)
    return float((d_c <= R).astype(float).mean() - (d_f <= R).astype(float).mean())


def run_variant(
    X, Y, folds, pred_f0,
    hidden=64, num_layers=1, use_extra_features=False, epochs=30,
    seeds=(20260518, 20260519, 20260520), device="cuda:1",
    batch_size=256, lr=1e-3, weight_decay=1e-4, early_stop_patience=10,
    label="V0", verbose=True,
):
    dev = torch.device(device) if (torch.cuda.is_available() and device.startswith("cuda")) else torch.device("cpu")

    common = bi.build_input_common(X, bf.f0_baseline)
    N = X.shape[0]
    flat_base = np.concatenate([common["L2"].reshape(N, 21), common["L4"].reshape(N, 14)], axis=1)  # (N,35)
    if use_extra_features:
        extra = bi.build_input_lgbm_extra(X, L1=common["L1"])  # (N,36)
        flat_np = np.concatenate([flat_base, extra], axis=1).astype(np.float32)
    else:
        flat_np = flat_base.astype(np.float32)
    flat_dim = flat_np.shape[1]
    if verbose:
        print(f"  [{label}] flat_dim={flat_dim} hidden={hidden} layers={num_layers} epochs={epochs}", flush=True)

    seq_all = torch.from_numpy(common["L1"]).float()
    flat_all = torch.from_numpy(flat_np).float()
    R_all = torch.from_numpy(common["R_wfn"]).float()
    pf0_all = torch.from_numpy(common["pred_F0_world"]).float()
    Y_t = torch.from_numpy(Y.astype(np.float32))
    q_all = torch.from_numpy(bi.build_soft_label(Y, common["R_wfn"], common["pred_F0_world"])).float()

    pred_world = np.zeros((N, 3), dtype=np.float32)
    train_hit_log: dict[int, Any] = {}

    for k in range(N_FOLDS):
        train_idx = np.where(folds != k)[0]
        val_idx = np.where(folds == k)[0]
        best_seed, best_train_hit, best_state, best_epoch = None, -1.0, None, -1
        seed_log: dict[int, dict] = {}

        for seed in seeds:
            torch.manual_seed(seed); np.random.seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

            model = FlexGRUDualHead(hidden=hidden, num_layers=num_layers, flat_dim=flat_dim).to(dev)
            opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

            seq_tr = seq_all[train_idx].to(dev); flat_tr = flat_all[train_idx].to(dev)
            R_tr = R_all[train_idx].to(dev); pf0_tr = pf0_all[train_idx].to(dev)
            Y_tr = Y_t[train_idx].to(dev); q_tr = q_all[train_idx].to(dev)
            N_tr = len(train_idx)

            plateau = 0
            best_epoch_seed, best_train_hit_seed, best_state_seed = 0, -1.0, None
            stopped_epoch = epochs - 1

            for epoch in range(epochs):
                model.train()
                perm = torch.randperm(N_tr, device=dev)
                for i in range(0, N_tr, batch_size):
                    idx = perm[i:i + batch_size]
                    logits, reg_off = model(seq_tr[idx], flat_tr[idx])
                    probs = torch.softmax(logits, dim=1)
                    combined = model.ANCHORS[None] + reg_off
                    final_frenet = (probs[:, :, None] * combined).sum(dim=1)
                    final_world = torch.einsum("nij,nj->ni", R_tr[idx], final_frenet) + pf0_tr[idx]
                    tau, ub = dh.tau_for_epoch(epoch)
                    loss = dh.soft_ce_loss(logits, q_tr[idx]) + dh.smooth_hit_loss(
                        final_world, Y_tr[idx], tau=tau, use_boundary=ub,
                    )
                    opt.zero_grad(); loss.backward(); opt.step()

                model.eval()
                with torch.no_grad():
                    tp = []
                    for i in range(0, N_tr, batch_size):
                        seg = slice(i, i + batch_size)
                        tp.append(model.predict_world(seq_tr[seg], flat_tr[seg], R_tr[seg], pf0_tr[seg]))
                    train_pred = torch.cat(tp, 0)
                d_tr = (train_pred - Y_tr).norm(dim=1)
                train_hit = float((d_tr <= R_HIT).float().mean().item())

                if train_hit > best_train_hit_seed + 1e-5:
                    best_train_hit_seed = train_hit
                    best_epoch_seed = epoch
                    best_state_seed = {kk: vv.detach().clone() for kk, vv in model.state_dict().items()}
                    plateau = 0
                else:
                    plateau += 1
                    if plateau >= early_stop_patience:
                        stopped_epoch = epoch
                        break

            seed_log[seed] = {
                "best_train_hit": best_train_hit_seed,
                "best_epoch": best_epoch_seed,
                "stopped_epoch": stopped_epoch,
            }
            if best_train_hit_seed > best_train_hit:
                best_train_hit = best_train_hit_seed
                best_seed = seed; best_state = best_state_seed; best_epoch = best_epoch_seed

        train_hit_log[k] = {"seeds": seed_log, "best_seed": best_seed, "best_epoch": best_epoch}
        if verbose:
            print(f"  [{label}] fold {k}: seed={best_seed} ep={best_epoch} train_hit={best_train_hit:.4f}", flush=True)

        model = FlexGRUDualHead(hidden=hidden, num_layers=num_layers, flat_dim=flat_dim).to(dev)
        model.load_state_dict(best_state); model.eval()
        with torch.no_grad():
            vp = []
            for i in range(0, len(val_idx), batch_size):
                vi = val_idx[i:i + batch_size]
                final = model.predict_world(seq_all[vi].to(dev), flat_all[vi].to(dev),
                                             R_all[vi].to(dev), pf0_all[vi].to(dev))
                vp.append(final.cpu().numpy())
            pred_world[val_idx] = np.concatenate(vp, axis=0)

    m = hit_rates(pred_world, Y, folds)
    m["candidate"] = label
    m["n_samples"] = N
    m["delta_1cm"] = paired_delta(pred_world, pred_f0, Y, R_HIT)
    m["delta_1.5cm"] = paired_delta(pred_world, pred_f0, Y, R_HIT_LOOSE)
    m["pass_both"] = bool(m["delta_1cm"] >= 0.005 and m["delta_1.5cm"] >= 0.005)
    m["config"] = {"hidden": hidden, "num_layers": num_layers, "flat_dim": flat_dim, "epochs": epochs}
    m["train_hit_log"] = train_hit_log
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=["V0", "V1", "V2", "V3"])
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--out", type=Path, default=_THIS_DIR / "diag_gru_results.json")
    args = ap.parse_args()

    print("[diag] loading data ...", flush=True)
    ids, X = load_all_samples(split="train")
    labid, Y = load_labels()
    assert ids == labid
    X = X.astype(np.float64); Y = Y.astype(np.float64)
    folds = assign_folds(ids)
    print(f"[diag] N={X.shape[0]} folds={np.bincount(folds).tolist()}", flush=True)

    pred_f0 = bf.f0_baseline(X, end_idx=END_IDX)
    f0_hit_1 = float((np.linalg.norm(pred_f0 - Y, axis=1) <= R_HIT).mean())
    f0_hit_15 = float((np.linalg.norm(pred_f0 - Y, axis=1) <= R_HIT_LOOSE).mean())
    print(f"[diag] F0: hit@1cm={f0_hit_1:.4f} hit@1.5cm={f0_hit_15:.4f}", flush=True)

    specs = {
        "V0_baseline": dict(hidden=64, num_layers=1, use_extra_features=False, epochs=30),
        "V1_feat":     dict(hidden=64, num_layers=1, use_extra_features=True, epochs=30),
        "V2_cap":      dict(hidden=128, num_layers=2, use_extra_features=False, epochs=50),
        "V3_both":     dict(hidden=128, num_layers=2, use_extra_features=True, epochs=50),
    }
    short2full = {"V0": "V0_baseline", "V1": "V1_feat", "V2": "V2_cap", "V3": "V3_both"}

    results: dict[str, Any] = {"f0_baseline": {"hit_1cm": f0_hit_1, "hit_1.5cm": f0_hit_15}}
    for v in args.variants:
        full = short2full.get(v, v)
        if full not in specs:
            print(f"[diag] skip unknown {v}", flush=True); continue
        print(f"\n[diag] === {full} ===", flush=True)
        t0 = time.time()
        results[full] = run_variant(X, Y, folds, pred_f0, label=full, device=args.device, **specs[full])
        results[full]["wall_time_s"] = time.time() - t0
        print(f"  [{full}] elapsed {results[full]['wall_time_s']:.1f}s "
              f"hit@1cm={results[full]['hit_1cm_5fold_concat']:.4f} "
              f"hit@1.5cm={results[full]['hit_1.5cm_5fold_concat']:.4f} "
              f"Δ_1cm={results[full]['delta_1cm']:+.4f} "
              f"Δ_1.5cm={results[full]['delta_1.5cm']:+.4f}", flush=True)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=2, default=str))

    print(f"\n[diag] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
