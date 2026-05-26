"""plan-c-001 §4.3 — F0(perp=0.0) 잔차 GRU OOF orchestrator (FR001).

KR002 paradigm (잔차 GRU + 입력 yaw) 의 baseline predictor 만 Kalman CV → f0_perp0 swap
+ F0 per-step 잔차 자기진단 feature (§1.3, --f0-resid-feats). W-aux λ=0 (Kalman target 회피).
plan-a-001 의 {yaw, features, model, losses} 를 import 재사용 (원본 미수정, KR repro 보존).

Usage:
  python analysis/plan-c-001/run_oof.py --gate smoke
  python analysis/plan-c-001/run_oof.py --gate g1   --baseline f0-perp0 --f0-resid-feats --input-yaw
  python analysis/plan-c-001/run_oof.py --gate full --baseline f0-perp0 --f0-resid-feats --input-yaw \
         --aux-w-weight 0 --out results_fr001.json
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
from sklearn.preprocessing import StandardScaler

_THIS = Path(__file__).resolve().parent
_PA1 = _THIS.parent / "plan-a-001"
_REPO = _THIS.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.io import load_all_samples, load_labels  # noqa: E402
from src.pb_0_6822.selector import stable_fold_id  # noqa: E402


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_yaw = _load("pc_yaw", _PA1 / "yaw.py")
_feat = _load("pc_feat", _PA1 / "features.py")
_model = _load("pc_model", _PA1 / "model.py")
_losses = _load("pc_losses", _PA1 / "losses.py")
_f0 = _load("pc_f0", _THIS / "f0_baseline.py")
_frf = _load("pc_frf", _THIS / "f0_residual_feats.py")

R_HIT = 0.01
R_HIT_LOOSE = 0.015

CONFIGS = {
    "A": dict(hidden_size=64, num_layers=1, fc_hidden=128, lr=5e-4, p=0.3, wd=1e-4),
    "B": dict(hidden_size=64, num_layers=1, fc_hidden=128, lr=1e-3, p=0.1, wd=1e-4),
}


def hit_rate(pred, y, thr=R_HIT):
    return float((np.linalg.norm(pred - y, axis=-1) <= thr).mean())


def hit_mask(pred, y, thr=R_HIT):
    return (np.linalg.norm(pred - y, axis=-1) <= thr)


def rotate_seq_input(seq, theta):
    """seq (N,11,>=9) 의 base rel(0-2)/v(3-5)/a(6-8) triplet 을 rotate_xy(θ) (KR002).
    9 이후 채널(F0-잔차 r_f)은 이미 yaw frame → 회전 미적용."""
    out = seq.copy()
    N, T, _ = seq.shape
    th = np.repeat(theta, T)
    for s in (0, 3, 6):
        flat = seq[:, :, s:s + 3].reshape(-1, 3)
        out[:, :, s:s + 3] = _yaw.rotate_xy(flat, th).reshape(N, T, 3)
    return out.astype(np.float32)


def train_one(cfg, seq_tr, scal_tr, tgt_main, tgt_F, tgt_W,
              seq_va, scal_va, theta_va, base_va, y_va,
              seed, device, epochs, patience, lambda_F, lambda_W,
              n_channels, batch=256, quiet=True):
    """단일 config×seed×fold 학습 → (best_val_out_rot, best_rhit)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    net = _model.GRUModelMultiAux(
        n_channels=n_channels, scal_dim=scal_tr.shape[1], hidden_size=cfg["hidden_size"],
        num_layers=cfg["num_layers"], fc_hidden=cfg["fc_hidden"], p=cfg["p"],
        aux_dims=[3, 3], aux_clips=[None, None], main_out_scale_cm=2.0,
    ).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    sq = torch.as_tensor(seq_tr, device=device)
    sc = torch.as_tensor(scal_tr, device=device)
    tm = torch.as_tensor(tgt_main, dtype=torch.float32, device=device)
    tF = torch.as_tensor(tgt_F, dtype=torch.float32, device=device)
    tW = torch.as_tensor(tgt_W, dtype=torch.float32, device=device)
    sq_va = torch.as_tensor(seq_va, device=device)
    sc_va = torch.as_tensor(scal_va, device=device)
    n = sq.shape[0]

    best_rhit, best_out, no_imp = -1.0, None, 0
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            om, ax = net(sq[idx], sc[idx])
            loss = (_losses.loss_combo(om, tm[idx])
                    + lambda_F * _losses.loss_aux_euclid(ax[0], tF[idx])
                    + lambda_W * _losses.loss_aux_euclid(ax[1], tW[idx]))
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        sched.step()
        net.eval()
        with torch.no_grad():
            om_va, _ = net(sq_va, sc_va)
        out_rot = om_va.cpu().numpy()
        pred = base_va + _yaw.inverse_rotate_xy(out_rot, theta_va)
        rh = hit_rate(pred, y_va)
        if rh > best_rhit:
            best_rhit, best_out, no_imp = rh, out_rot.copy(), 0
        else:
            no_imp += 1
            if no_imp >= patience:
                break
    return best_out, best_rhit


def run_config(cfg_name, X, y, fold_ids, theta, base_pred, seq, scal,
               tgt_main, tgt_F, tgt_W, folds, seeds, epochs, patience,
               device, quiet, lambda_F, lambda_W):
    cfg = CONFIGS[cfg_name]
    N = X.shape[0]
    n_channels = seq.shape[2]
    oof_rot = np.zeros((N, 3), dtype=np.float64)
    for f in folds:
        tr = np.where(fold_ids != f)[0]
        va = np.where(fold_ids == f)[0]
        sc_seq = StandardScaler().fit(seq[tr].reshape(-1, n_channels))
        sc_scal = StandardScaler().fit(scal[tr])
        seq_tr = _feat.normalize_seq(seq[tr], sc_seq)
        seq_va = _feat.normalize_seq(seq[va], sc_seq)
        scal_tr = sc_scal.transform(scal[tr]).astype(np.float32)
        scal_va = sc_scal.transform(scal[va]).astype(np.float32)
        seed_outs = []
        for s in seeds:
            out_rot, rh = train_one(
                cfg, seq_tr, scal_tr, tgt_main[tr], tgt_F[tr], tgt_W[tr],
                seq_va, scal_va, theta[va], base_pred[va], y[va],
                s, device, epochs, patience, lambda_F, lambda_W, n_channels, quiet=quiet)
            seed_outs.append(out_rot)
            if not quiet:
                print(f"    [{cfg_name}] fold{f} seed{s} val_rhit={rh:.4f}", flush=True)
        oof_rot[va] = np.mean(seed_outs, axis=0)
    return _yaw.inverse_rotate_xy(oof_rot, theta)


def build_inputs(X, theta, input_yaw, use_resid, use_cache=True):
    """seq (N,11,C), scal (N,D), n_channels, scal_dim 반환. C=9 or 12, D=40 or 48.
    use_cache=False (truncated gate) 시 noise 캐시 미사용 — 부분 N 이 full 캐시 오염 방지."""
    cache = (_THIS / "noise_cache.npz") if use_cache else None
    noise = _feat.compute_noise(X, cache_path=cache, key="train", with_loo=True)
    seq = _feat.build_seq_t3(X)                      # (N,11,9)
    scal, names = _feat.build_scalar_40d(X, noise["poly2"], noise["savgol"], noise["loo"])
    if input_yaw:
        seq = rotate_seq_input(seq, theta)
    if use_resid:
        feats = _frf.f0_resid_feats(X, theta)
        seq = np.concatenate([seq, feats["seq_resid"]], axis=2).astype(np.float32)  # 9→12
        scal = np.concatenate(
            [scal, feats["f0_conf"], feats["ewma"], feats["sta_lta"]], axis=1).astype(np.float32)  # 40→48
    return seq, scal


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", choices=["smoke", "g1", "full"], default="full")
    ap.add_argument("--baseline", choices=["kalman-cv", "f0-perp0"], default="f0-perp0")
    ap.add_argument("--f0-resid-feats", action="store_true", help="§1.3 F0 잔차 자기진단 concat")
    ap.add_argument("--input-yaw", action="store_true", help="KR002: 입력 seq rel/v/a 회전")
    ap.add_argument("--aux-w-weight", type=float, default=0.0, help="W-aux λ (FR001=0)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    if args.gate == "smoke":
        folds, seeds, epochs, configs, max_n = [0], [0], 1, ["A"], 512
    elif args.gate == "g1":
        folds, seeds, epochs, configs, max_n = [0], [0], args.epochs, ["A"], None
    else:
        folds, seeds, epochs, configs, max_n = list(range(5)), [0, 1, 2], args.epochs, ["A", "B"], None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    print(f"[plan-c-001 FR001] gate={args.gate} baseline={args.baseline} "
          f"f0_resid={args.f0_resid_feats} input_yaw={args.input_yaw} aux_w={args.aux_w_weight} "
          f"device={device} epochs={epochs} configs={configs} seeds={seeds}", flush=True)

    ids, X = load_all_samples("train")
    lab_ids, y = load_labels()
    assert ids == lab_ids, "id 정렬 불일치"
    if max_n is not None:
        X, y, ids = X[:max_n], y[:max_n], ids[:max_n]
    N = X.shape[0]

    fold_ids = np.array([stable_fold_id(i, 5) for i in ids])
    pm = np.isin(fold_ids, folds)
    theta = _yaw.yaw_from_last_step(X)

    # baseline predictor (F0(perp=0.0) numpy 비학습)
    if args.baseline == "f0-perp0":
        base_pred = _f0.f0_perp0(X)
    else:
        _kal = _load("pc_kal", _PA1 / "kalman.py")
        base_pred = _kal.kalman_predict(X, sigma_obs=_kal.SIGMA_OBS_MAIN, sigma_proc=_kal.SIGMA_PROC_MAIN)

    # targets (rotated frame). W target = zeros (λ_W=0, Kalman-free)
    tgt_main = _yaw.rotate_xy(y - base_pred, theta).astype(np.float32)
    tgt_F = _yaw.rotate_xy(y - X[:, -1], theta).astype(np.float32)
    tgt_W = np.zeros_like(tgt_main, dtype=np.float32)

    seq, scal = build_inputs(X, theta, args.input_yaw, args.f0_resid_feats, use_cache=(max_n is None))
    print(f"  inputs: seq {seq.shape} scal {scal.shape}", flush=True)

    # baseline-alone floor (비교 기준점)
    base_alone_hit = hit_rate(base_pred[pm], y[pm])
    base_alone_hit15 = hit_rate(base_pred[pm], y[pm], R_HIT_LOOSE)

    oof_res = {}
    for c in configs:
        oof_res[c] = run_config(
            c, X, y, fold_ids, theta, base_pred, seq, scal,
            tgt_main, tgt_F, tgt_W, folds, seeds, epochs, args.patience,
            device, args.quiet, lambda_F=_losses.LAMBDA_AUX, lambda_W=args.aux_w_weight)
        h = hit_rate((base_pred + oof_res[c])[pm], y[pm])
        print(f"  config {c} OOF hit_1cm={h:.4f}", flush=True)

    ens_res = np.mean([oof_res[c] for c in configs], axis=0)
    pred_uncal = base_pred + ens_res
    hit_1cm = hit_rate(pred_uncal[pm], y[pm])
    hit_1p5cm = hit_rate(pred_uncal[pm], y[pm], R_HIT_LOOSE)
    per_sample_hit = hit_mask(pred_uncal, y)

    # band (fold-평균 기준). floor = base_alone (f0_perp0)
    floor = base_alone_hit
    band = ("STRONG_kalman_par" if hit_1cm >= 0.6663 else "PASS" if hit_1cm >= floor + 0.02
            else "WEAK_above_floor" if hit_1cm >= floor else "FAIL_no_lift")

    result = dict(
        exp="FR001", gate=args.gate, baseline=args.baseline,
        f0_resid_feats=args.f0_resid_feats, input_yaw=args.input_yaw,
        aux_w_weight=args.aux_w_weight, device=device,
        N=int(N), n_predicted=int(pm.sum()),
        folds=[int(f) for f in folds], seeds=[int(s) for s in seeds],
        epochs=epochs, configs=configs,
        seq_dim=int(seq.shape[2]), scal_dim=int(scal.shape[1]),
        hit_1cm=hit_1cm, hit_1p5cm=hit_1p5cm,
        config_hit_1cm={c: hit_rate((base_pred + oof_res[c])[pm], y[pm]) for c in configs},
        baseline_alone_hit_1cm=base_alone_hit, baseline_alone_hit_1p5cm=base_alone_hit15,
        floor_f0perp0=floor, pass_bar=floor + 0.02, strong_bar=0.6663,
        g_f0_band=band,
        runtime_sec=round(time.time() - t0, 1),
    )
    print(f"[done] hit_1cm={hit_1cm:.4f} band={band} "
          f"(floor={floor:.4f}, +0.02bar={floor+0.02:.4f}, KR002=0.6663) "
          f"runtime={result['runtime_sec']}s", flush=True)

    out_name = args.out or f"results_fr001_{args.gate}.json"
    (_THIS / out_name).write_text(json.dumps(result, indent=2, ensure_ascii=False))
    if args.gate == "full":
        npz_name = out_name.replace(".json", ".npz")
        np.savez(_THIS / npz_name, oof_residual=ens_res, oof_pred=pred_uncal,
                 per_sample_hit=per_sample_hit, y=y, base_pred=base_pred, fold_ids=fold_ids)
    print(f"[saved] {out_name}", flush=True)
    return result


if __name__ == "__main__":
    main()
