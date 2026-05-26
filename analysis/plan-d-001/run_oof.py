"""plan-d-001 c4 — Neural ODE 5-fold OOF runner (NODE001).

5-fold = stable_fold_id(sid,5). per-fold: train-fold ft 통계 선계산 → 학습 → val 예측 OOF 누적.
OOF hit_1cm (world Euclid < R_HIT) + paired permutation 10k vs F0.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis" / "plan-020"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.io import load_all_samples, load_labels  # noqa: E402
from src.pb_0_6822.selector import stable_fold_id  # noqa: E402
from baseline_f0 import R_HIT, R_HIT_LOOSE, f0_baseline  # noqa: E402  (analysis/plan-020)

from features import extract_features  # noqa: E402
from losses import combined_loss  # noqa: E402
from model import SimpleNeuralODEModel  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent


def load_aligned():
    ids_x, X = load_all_samples("train")
    ids_y, y = load_labels()
    assert ids_x == ids_y, "train coords/labels id 순서 불일치"
    return ids_x, X.astype(np.float32), y.astype(np.float32)


def fold_ids(ids, k=5):
    return np.array([stable_fold_id(s, k) for s in ids])


def feats_for(Xt, mean=None, std=None):
    ft, df, pl, th, _, _, _, R, sp, m, s = extract_features(Xt, mean, std)
    return dict(ft=ft, df=df, pl=pl, th=th, R=R, sp=sp), m, s


def val_hits(model, V, yv, device):
    model.eval()
    with torch.no_grad():
        pred = model(V["ft"], V["df"], V["pl"], V["th"], V["sp"], V["R"])
        d = torch.norm(pred - yv, dim=1)
    return pred.cpu().numpy(), (d <= R_HIT).float().mean().item()


def train_fold(Xtr, ytr, Xvl, yvl, device, epochs, batch, seed, log):
    torch.manual_seed(seed)
    np.random.seed(seed)
    Xtr_t = torch.from_numpy(Xtr).to(device)
    Xvl_t = torch.from_numpy(Xvl).to(device)
    ytr_t = torch.from_numpy(ytr).to(device)
    yvl_t = torch.from_numpy(yvl).to(device)

    _, mean, std = feats_for(Xtr_t)          # stats 계산 모드
    Tr, _, _ = feats_for(Xtr_t, mean, std)
    Vl, _, _ = feats_for(Xvl_t, mean, std)

    model = SimpleNeuralODEModel(input_dim=24, latent_dim=64).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=4e-3, weight_decay=1e-3)

    n = Xtr.shape[0]
    ep_hits = []
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            pred = model(Tr["ft"][idx], Tr["df"][idx], Tr["pl"][idx],
                         Tr["th"][idx], Tr["sp"][idx], Tr["R"][idx])
            loss = combined_loss(pred, ytr_t[idx], model._last_accels)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss ep{ep}")
            loss.backward()
            opt.step()
        _, hr = val_hits(model, Vl, yvl_t, device)
        ep_hits.append(hr)
        if log:
            print(f"    ep{ep:2d}/{epochs} val_hit@1cm={hr*100:.3f}%")
    val_pred, _ = val_hits(model, Vl, yvl_t, device)
    return val_pred, ep_hits


def paired_permutation(hit_a, hit_b, n_resample=10000, seed=0):
    """H0: hit_a, hit_b 동분포. 통계 = mean(hit_a) - mean(hit_b), sign-flip resample."""
    rng = np.random.default_rng(seed)
    d = hit_a.astype(np.float64) - hit_b.astype(np.float64)
    obs = d.mean()
    flips = rng.choice([1.0, -1.0], size=(n_resample, d.shape[0]))
    null = (flips * d).mean(axis=1)
    p = float((np.abs(null) >= abs(obs)).mean())
    return obs, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="1-fold 1-epoch finite check")
    ap.add_argument("--g1", action="store_true", help="1-fold full-epoch (G1)")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ids, X, y = load_aligned()
    fid = fold_ids(ids, args.folds)
    print(f"[plan-d-001] N={len(ids)} device={device} folds={args.folds} "
          f"epochs={args.epochs} seed={args.seed}")

    if args.smoke:
        m = fid == 0
        vp, eh = train_fold(X[~m], y[~m], X[m], y[m], device, epochs=1, batch=args.batch,
                            seed=args.seed, log=True)
        assert np.isfinite(vp).all(), "smoke: non-finite val pred"
        print(f"[SMOKE] OK finite, fold0 1ep val_hit={eh[-1]*100:.3f}%")
        return

    if args.g1:
        m = fid == 0
        vp, eh = train_fold(X[~m], y[~m], X[m], y[m], device, epochs=args.epochs,
                            batch=args.batch, seed=args.seed, log=True)
        g1_pass = bool(np.isfinite(vp).all() and eh[-1] >= eh[0])
        print(f"[G1] fold0 epoch1={eh[0]*100:.3f}% final={eh[-1]*100:.3f}% "
              f"PASS={g1_pass} (final>=epoch1, 비단조 noise 허용)")
        json.dump({"epoch_hits": eh, "g1_pass": g1_pass},
                  open(OUT_DIR / "g1_node001.json", "w"), indent=2)
        return

    # full 5-fold OOF
    oof = np.zeros_like(y)
    fold_final = []
    for f in range(args.folds):
        m = fid == f
        print(f"  FOLD {f+1}/{args.folds} (val n={m.sum()})")
        vp, eh = train_fold(X[~m], y[~m], X[m], y[m], device, epochs=args.epochs,
                            batch=args.batch, seed=args.seed, log=False)
        oof[m] = vp
        fold_final.append(eh[-1])
        print(f"    fold{f} final val_hit={eh[-1]*100:.3f}%")

    err = np.linalg.norm(oof - y, axis=1)
    hit_node = (err <= R_HIT)
    hit_1cm = float(hit_node.mean())
    hit_1p5 = float((err <= R_HIT_LOOSE).mean())

    f0 = f0_baseline(X.astype(np.float64), end_idx=X.shape[1] - 1)
    hit_f0 = (np.linalg.norm(f0 - y, axis=1) <= R_HIT)
    delta, pval = paired_permutation(hit_node, hit_f0)

    band = ("EXCELLENT" if hit_1cm >= 0.6854 else "STRONG" if hit_1cm >= 0.6622
            else "PASS" if hit_1cm >= 0.6320 else "FAIL_transfer")
    res = {
        "exp_id": "NODE001_notebook-repro",
        "oof_hit_1cm": hit_1cm, "oof_hit_1p5cm": hit_1p5,
        "f0_hit_1cm": float(hit_f0.mean()),
        "delta_vs_f0": delta, "perm_p": pval,
        "band": band, "fold_final_hits": fold_final,
        "n": len(ids), "epochs": args.epochs, "seed": args.seed,
        "mean_err": float(err.mean()),
    }
    json.dump(res, open(OUT_DIR / "results_node001.json", "w"), indent=2)
    np.savez(OUT_DIR / "results_node001.npz", oof_pred=oof, y=y, err=err, fold_id=fid)
    print(f"[NODE001] OOF hit_1cm={hit_1cm:.4f} ({band}) | hit_1p5cm={hit_1p5:.4f} | "
          f"vs F0 {hit_f0.mean():.4f} Δ={delta:+.4f} p={pval:.4f}")


if __name__ == "__main__":
    main()
