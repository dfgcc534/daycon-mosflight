"""plan-032 c3 — Ablation B (boundary corrector 14-anchor 재구현).

post-process MLP: plan-031 OOF probs + final_pred + sample state → 3D delta correction.

5-fold cross-validation (corrector 학습 fold leakage 차단):
  - plan-031 results_g3.npz 에서 oof_pred (N=10000, 3) + oof_probs (N, 14) load
  - 5-fold split (plan-031 carry stable_fold_id)
  - corrector input per sample (M dim):
      top3_prob (3) + top3_anchor_world (9 = 3 anchor × 3 axis) + F0_pred (3) + R_wfn flat (9) + plan021_macro9 (9)
      = 33D
  - MLP: Linear(33, 64) → SiLU → Dropout(0.1) → Linear(64, 32) → SiLU → Linear(32, 3)
  - loss = huber(pred + delta, gt)  (delta 는 raw pred 의 correction)
  - 학습: 50 epoch AdamW lr=1e-3, batch=128

inference: 5-fold corrector inject → corrected_pred = pred + delta.
metric: hit_1cm corrected vs plan-031 oof_pred.
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


_train_mod = _load(_THIS.parent / "plan-031" / "train.py", "p032_b_train")
_p021_build = _load(_THIS.parent / "plan-021" / "build_input.py", "p032_b_bi")
_p022_anchors = _load(_THIS.parent / "plan-022" / "anchors.py", "p032_b_av")
_bf = _load(_THIS.parent / "plan-020" / "baseline_f0.py", "p032_b_bf")


class TinyCorrectionNet(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.SiLU(),
            nn.Linear(hidden // 2, 3),
        )
        # init last layer near zero (delta 작게)
        nn.init.zeros_(self.net[-1].bias)
        with torch.no_grad():
            self.net[-1].weight.mul_(0.01)

    def forward(self, x):
        return self.net(x)


def build_corrector_features(
    pred: np.ndarray,       # (N, 3) plan-031 final pred
    probs: np.ndarray,      # (N, 14)
    F0: np.ndarray,         # (N, 3)
    R_wfn: np.ndarray,      # (N, 3, 3)
    X: np.ndarray,          # (N, 11, 3)
) -> np.ndarray:
    """Returns (N, 33) corrector input."""
    N, K = probs.shape
    anchors_f = _p022_anchors.ANCHORS_A6.astype(np.float64)
    # top3 anchor: per sample
    top3_idx = np.argsort(-probs, axis=1)[:, :3]  # (N, 3)
    top3_prob = np.take_along_axis(probs, top3_idx, axis=1).astype(np.float32)  # (N, 3)
    # top3 anchor world pos = F0 + R_wfn @ anchors[top3]
    anchor_frenet_top3 = anchors_f[top3_idx]                    # (N, 3, 3)
    anchor_world_top3 = F0[:, None, :] + np.einsum("nij,nkj->nki", R_wfn, anchor_frenet_top3)  # (N, 3, 3)
    anchor_world_flat = anchor_world_top3.reshape(N, 9).astype(np.float32)
    F0_f = F0.astype(np.float32)
    R_flat = R_wfn.reshape(N, 9).astype(np.float32)
    macro9 = _p021_build._macro_stat_9d(X, end_idx=10).astype(np.float32)
    feat = np.concatenate([top3_prob, anchor_world_flat, F0_f, R_flat, macro9], axis=-1)  # (N, 3+9+3+9+9=33)
    return feat.astype(np.float32)


def train_corrector_5fold(
    feat: np.ndarray,       # (N, 33)
    pred: np.ndarray,       # (N, 3) plan-031 pred (=initial)
    gt: np.ndarray,         # (N, 3)
    folds: np.ndarray,      # (N,) fold id
    epochs: int = 50,
    batch: int = 128,
    lr: float = 1e-3,
    huber_delta: float = 0.005,
) -> np.ndarray:
    """5-fold OOF corrected pred. Returns (N, 3)."""
    N = feat.shape[0]
    corrected = np.zeros_like(pred)
    for fold in range(5):
        tr_mask = folds != fold
        te_mask = folds == fold
        feat_tr = torch.from_numpy(feat[tr_mask]).float()
        pred_tr = torch.from_numpy(pred[tr_mask]).float()
        gt_tr = torch.from_numpy(gt[tr_mask]).float()
        feat_te = torch.from_numpy(feat[te_mask]).float()
        pred_te = torch.from_numpy(pred[te_mask]).float()

        torch.manual_seed(20260524 + fold)
        model = TinyCorrectionNet(in_dim=feat.shape[1], hidden=64, dropout=0.1)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

        N_tr = feat_tr.shape[0]
        for ep in range(epochs):
            model.train()
            perm = torch.randperm(N_tr)
            for b in range(0, N_tr, batch):
                idx = perm[b:b+batch]
                delta = model(feat_tr[idx])
                corrected_b = pred_tr[idx] + delta
                loss = F.huber_loss(corrected_b, gt_tr[idx], delta=huber_delta)
                opt.zero_grad()
                loss.backward()
                opt.step()
            sched.step()

        model.eval()
        with torch.no_grad():
            delta_te = model(feat_te).numpy()
        corrected[te_mask] = pred[te_mask] + delta_te
    return corrected


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=50)
    args = ap.parse_args()

    from src.io import load_all_samples, load_labels
    from src.pb_0_6822.selector import stable_fold_id

    t0 = time.perf_counter()

    # 1) load plan-031 OOF artifacts
    npz = np.load(_THIS.parent / "plan-031" / "results_g3.npz")
    oof_pred = npz["oof_pred"]            # (N, 3)
    oof_probs = npz["oof_probs"]          # (N, 14)
    N = oof_pred.shape[0]

    # 2) reload raw data + F0, R_wfn per fold (carry from plan-031)
    _ids, X = load_all_samples()
    _, gt = load_labels()
    X = X.astype(np.float32); gt = gt.astype(np.float32)
    folds = np.asarray([stable_fold_id(str(s), 5) for s in _ids], dtype=int)

    # F0, R_wfn 은 per fold 의 train-fold 만으로 학습되었음.
    # corrector 학습은 OOF 기반이라 sample 별 1개 R_wfn / F0 필요 — 각 fold 의 test set 에서 산출본 사용.
    # 단 fold 별 다른 build → fold 별 산출 후 inject.
    F0_all = np.zeros((N, 3), dtype=np.float32)
    R_all = np.zeros((N, 3, 3), dtype=np.float32)
    for fold in range(5):
        mask = folds == fold
        if mask.any():
            F0_all[mask] = _bf.f0_baseline(X[mask], end_idx=10).astype(np.float32)
            R_all[mask] = _p021_build.build_frenet_basis_3d(X[mask], end_idx=10).astype(np.float32)

    # 3) build corrector features
    feat = build_corrector_features(oof_pred, oof_probs, F0_all, R_all, X)
    print(f"feat shape={feat.shape}, dtype={feat.dtype}")

    # 4) 5-fold train corrector + inference
    corrected = train_corrector_5fold(feat, oof_pred, gt, folds, epochs=args.epochs)

    # 5) metrics
    err_before = np.linalg.norm(oof_pred - gt, axis=1)
    err_after = np.linalg.norm(corrected - gt, axis=1)
    hit_before = float((err_before <= 0.01).mean())
    hit_after = float((err_after <= 0.01).mean())
    hit15_after = float((err_after <= 0.015).mean())
    delta = hit_after - hit_before

    elapsed = time.perf_counter() - t0
    band = ("SUPERIOR" if hit_after >= 0.6718 else
            "EXCELLENT" if hit_after >= 0.6624 else
            "PASS" if hit_after >= 0.6511 else
            "STRONG" if hit_after >= 0.6387 else
            "BORDERLINE" if hit_after >= 0.6320 else
            "FAIL_regression")

    result = {
        "ablation_axis": "B",
        "variant": "boundary_corrector_14a",
        "in_dim": int(feat.shape[1]),
        "epochs": args.epochs,
        "hit_1cm_before": hit_before,
        "hit_1cm_after": hit_after,
        "hit_1p5cm_after": hit15_after,
        "delta_vs_plan_031": delta,
        "band": band,
        "elapsed_s": elapsed,
    }
    out_path = _THIS / "results_B.json"
    np.savez_compressed(out_path.with_suffix(".npz"),
                        corrected_pred=corrected, original_pred=oof_pred)
    with out_path.open("w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    print(f"[done] B (boundary corrector): {hit_before:.4f} → {hit_after:.4f} (Δ={delta:+.4f}), "
          f"band={band}, elapsed={elapsed:.1f}s")


if __name__ == "__main__":
    main()
