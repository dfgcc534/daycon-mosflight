"""plan-a-001 OOF orchestrator (§4.4).

KR001 (notebook repro) / KR002 (--input-yaw) 5-fold OOF hit_1cm 박제.
- 5-fold = stable_fold_id (노트북 KFold 대신, 프로젝트 OOF 호환 — decision-note).
- 2 config(A/B) × 3 seed, per-fold StandardScaler(train fit / val transform),
  rotated-frame seed 평균 → inverse_rotate world 복원 → kalman+residual.
- headline = uncalibrated. calibrated = add-on (1,0.95,1 하드코드 + OOF-fit α grid).
- test 예측 out-of-scope (OOF only; DACON 미제출).

Usage:
  python analysis/plan-a-001/run_oof.py --gate smoke
  python analysis/plan-a-001/run_oof.py --gate g1
  python analysis/plan-a-001/run_oof.py --gate full --out results_kr001.json
  python analysis/plan-a-001/run_oof.py --gate full --input-yaw --out results_kr002.json
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
_REPO = _THIS.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.io import load_all_samples, load_labels  # noqa: E402
from src.pb_0_6822.selector import stable_fold_id  # noqa: E402


def _load(name):
    spec = importlib.util.spec_from_file_location(f"pa_{name}", _THIS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_kalman = _load("kalman")
_yaw = _load("yaw")
_feat = _load("features")
_model = _load("model")
_losses = _load("losses")

R_HIT = 0.01
R_HIT_LOOSE = 0.015
BEST_ALPHAS = np.array([1.000, 0.950, 1.000])  # 노트북 하드코드 (cell 31)

CONFIGS = {
    "A": dict(hidden_size=64, num_layers=1, fc_hidden=128, lr=5e-4, p=0.3, wd=1e-4),
    "B": dict(hidden_size=64, num_layers=1, fc_hidden=128, lr=1e-3, p=0.1, wd=1e-4),
}


def hit_rate(pred, y, thr=R_HIT):
    return float((np.linalg.norm(pred - y, axis=-1) <= thr).mean())


def hit_mask(pred, y, thr=R_HIT):
    return (np.linalg.norm(pred - y, axis=-1) <= thr)


def rotate_seq_input(seq, theta):
    """seq (N,11,9) 의 rel(0-2)/v(3-5)/a(6-8) triplet 을 rotate_xy(θ) (KR002)."""
    out = seq.copy()
    N, T, _ = seq.shape
    th = np.repeat(theta, T)  # (N*T,)
    for s in (0, 3, 6):
        flat = seq[:, :, s:s + 3].reshape(-1, 3)
        out[:, :, s:s + 3] = _yaw.rotate_xy(flat, th).reshape(N, T, 3)
    return out.astype(np.float32)


def train_one(cfg, seq_tr, scal_tr, tgt_main, tgt_F, tgt_W,
              seq_va, scal_va, theta_va, kal_va, y_va,
              seed, device, epochs, patience, batch=256, quiet=True):
    """단일 config×seed×fold 학습 → (best_val_out_rot (n_va,3), best_rhit)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    net = _model.GRUModelMultiAux(
        n_channels=9, scal_dim=scal_tr.shape[1], hidden_size=cfg["hidden_size"],
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
                    + _losses.LAMBDA_AUX * _losses.loss_aux_euclid(ax[0], tF[idx])
                    + _losses.LAMBDA_AUX * _losses.loss_aux_euclid(ax[1], tW[idx]))
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        sched.step()
        # val R-Hit
        net.eval()
        with torch.no_grad():
            om_va, _ = net(sq_va, sc_va)
        out_rot = om_va.cpu().numpy()
        out_world = _yaw.inverse_rotate_xy(out_rot, theta_va)
        pred = kal_va + out_world
        rh = hit_rate(pred, y_va)
        if rh > best_rhit:
            best_rhit, best_out, no_imp = rh, out_rot.copy(), 0
        else:
            no_imp += 1
            if no_imp >= patience:
                break
    return best_out, best_rhit


def run_config(cfg_name, X, y, ids, fold_ids, theta, kalman_main, seq, scal,
               tgt_main, tgt_F, tgt_W, folds, seeds, epochs, patience, device, quiet):
    """config 1개의 5-fold OOF residual(rotated→world) → (oof_res_world (N,3))."""
    cfg = CONFIGS[cfg_name]
    N = X.shape[0]
    oof_rot = np.zeros((N, 3), dtype=np.float64)
    for f in folds:
        tr = np.where(fold_ids != f)[0]
        va = np.where(fold_ids == f)[0]
        sc_seq = StandardScaler().fit(seq[tr].reshape(-1, 9))
        sc_scal = StandardScaler().fit(scal[tr])
        seq_tr = _feat.normalize_seq(seq[tr], sc_seq)
        seq_va = _feat.normalize_seq(seq[va], sc_seq)
        scal_tr = sc_scal.transform(scal[tr]).astype(np.float32)
        scal_va = sc_scal.transform(scal[va]).astype(np.float32)
        seed_outs = []
        for s in seeds:
            out_rot, rh = train_one(
                cfg, seq_tr, scal_tr, tgt_main[tr], tgt_F[tr], tgt_W[tr],
                seq_va, scal_va, theta[va], kalman_main[va], y[va],
                s, device, epochs, patience, quiet=quiet)
            seed_outs.append(out_rot)
            if not quiet:
                print(f"    [{cfg_name}] fold{f} seed{s} val_rhit={rh:.4f}", flush=True)
        oof_rot[va] = np.mean(seed_outs, axis=0)
    oof_res_world = _yaw.inverse_rotate_xy(oof_rot, theta)
    return oof_res_world


def fit_alpha_grid(residual, kalman, y):
    """per-axis α grid {0.85..1.05 step .025} OOF-fit (각 축 독립 hit 최대화 — 근사)."""
    grid = np.round(np.arange(0.85, 1.0501, 0.025), 3)
    # 전역 hit 최대화는 축결합이라, 노트북식으로 per-axis 독립 sweep 후 조합 평가
    best_alpha, best_h = np.array([1.0, 1.0, 1.0]), -1.0
    # coordinate ascent 3 round
    alpha = np.array([1.0, 1.0, 1.0])
    for _ in range(3):
        for ax in range(3):
            cand_h = []
            for g in grid:
                a = alpha.copy(); a[ax] = g
                cand_h.append(hit_rate(kalman + residual * a[None, :], y))
            alpha[ax] = grid[int(np.argmax(cand_h))]
    best_alpha = alpha
    best_h = hit_rate(kalman + residual * best_alpha[None, :], y)
    return best_alpha, best_h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", choices=["smoke", "g1", "full"], default="full")
    ap.add_argument("--input-yaw", action="store_true", help="KR002: 입력 seq rel/v/a 회전")
    ap.add_argument("--out", default=None, help="결과 json 파일명 (analysis/plan-a-001/ 하위)")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    # gate 별 budget
    if args.gate == "smoke":
        folds, seeds, epochs, configs, max_n = [0], [0], 1, ["A"], 512
    elif args.gate == "g1":
        folds, seeds, epochs, configs, max_n = [0], [0], args.epochs, ["A"], None
    else:
        folds, seeds, epochs, configs, max_n = list(range(5)), [0, 1, 2], args.epochs, ["A", "B"], None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    print(f"[plan-a-001 OOF] gate={args.gate} input_yaw={args.input_yaw} device={device} "
          f"epochs={epochs} configs={configs} seeds={seeds} folds={folds}", flush=True)

    ids, X = load_all_samples("train")
    lab_ids, y = load_labels()
    assert ids == lab_ids, "id 정렬 불일치 (load_all_samples vs load_labels)"
    if max_n is not None:
        X, y, ids = X[:max_n], y[:max_n], ids[:max_n]
    N = X.shape[0]

    fold_ids = np.array([stable_fold_id(i, 5) for i in ids])
    predicted_mask = np.isin(fold_ids, folds)  # full=전체, g1=fold subset (비교 일관)
    theta = _yaw.yaw_from_last_step(X)
    kalman_main = _kalman.kalman_predict(X, sigma_obs=_kalman.SIGMA_OBS_MAIN, sigma_proc=_kalman.SIGMA_PROC_MAIN)
    kalman_alt = _kalman.kalman_predict(X, sigma_obs=_kalman.SIGMA_OBS_ALT, sigma_proc=_kalman.SIGMA_PROC_ALT)

    # targets (rotated frame)
    tgt_main = _yaw.rotate_xy(y - kalman_main, theta).astype(np.float32)
    tgt_F = _yaw.rotate_xy(y - X[:, -1], theta).astype(np.float32)
    tgt_W = _yaw.rotate_xy(y - kalman_alt, theta).astype(np.float32)

    # features
    cache = _THIS / "noise_cache.npz"
    noise = _feat.compute_noise(X, cache_path=(None if max_n else cache), key="train", with_loo=True)
    seq = _feat.build_seq_t3(X)
    scal, scal_names = _feat.build_scalar_40d(X, noise["poly2"], noise["savgol"], noise["loo"])
    rot_class = {n: "invariant" for n in scal_names}  # 40D 전부 회전불변
    if args.input_yaw:
        seq = rotate_seq_input(seq, theta)
    seq_rot_class = {c: ("rotate" if i < 6 else ("rotate" if i < 9 else "invariant"))
                     for i, c in enumerate(_feat.SEQ_CHANNELS)}  # rel/v/a 전부 rotate 대상

    # Kalman-alone baseline (G1 비교 기준점 — GRU/잔차 미적용). predicted_mask subset 기준.
    pm = predicted_mask
    kalman_alone_hit = hit_rate(kalman_main[pm], y[pm])
    kalman_alone_hit15 = hit_rate(kalman_main[pm], y[pm], R_HIT_LOOSE)

    # per-config OOF residual
    oof_res = {}
    for c in configs:
        oof_res[c] = run_config(
            c, X, y, ids, fold_ids, theta, kalman_main, seq, scal,
            tgt_main, tgt_F, tgt_W, folds, seeds, epochs, args.patience, device, args.quiet)
        h = hit_rate((kalman_main + oof_res[c])[pm], y[pm])
        print(f"  config {c} OOF hit_1cm={h:.4f}", flush=True)

    # ensemble residual (mean of configs). headline hit = predicted_mask subset.
    ens_res = np.mean([oof_res[c] for c in configs], axis=0)
    pred_uncal = kalman_main + ens_res
    hit_1cm = hit_rate(pred_uncal[pm], y[pm])
    hit_1p5cm = hit_rate(pred_uncal[pm], y[pm], R_HIT_LOOSE)
    per_sample_hit = hit_mask(pred_uncal, y)  # full array (npz 용)

    # calibration add-on (predicted_mask subset 기준)
    pred_hard = kalman_main + ens_res * BEST_ALPHAS[None, :]
    hit_hardcal = hit_rate(pred_hard[pm], y[pm])
    alpha_fit, hit_fitcal = fit_alpha_grid(ens_res[pm], kalman_main[pm], y[pm])
    # overfit-risk flag (c0 rev1): |α_fit − hardcode| > 0.05 어느 축 또는 calibrated < uncalibrated
    flag_dev = bool(np.any(np.abs(alpha_fit - BEST_ALPHAS) > 0.05))
    flag_drop = bool(hit_fitcal < hit_1cm)
    overfit_flag = flag_dev or flag_drop

    band = ("EXCELLENT" if hit_1cm >= 0.6600 else "STRONG" if hit_1cm >= 0.6528
            else "PASS" if hit_1cm >= 0.6320 else "FAIL_transfer")
    g1_pass = bool(hit_1cm > kalman_alone_hit)

    result = dict(
        exp=("KR002" if args.input_yaw else "KR001"),
        gate=args.gate, input_yaw=args.input_yaw, device=device,
        N=int(N), n_predicted=int(pm.sum()),
        folds=[int(f) for f in folds], seeds=[int(s) for s in seeds],
        epochs=epochs, configs=configs,
        hit_1cm=hit_1cm, hit_1p5cm=hit_1p5cm,
        config_hit_1cm={c: hit_rate((kalman_main + oof_res[c])[pm], y[pm]) for c in configs},
        kalman_alone_hit_1cm=kalman_alone_hit, kalman_alone_hit_1p5cm=kalman_alone_hit15,
        g_repro_band=band, g1_pass_gt_kalman_alone=g1_pass,
        calibration=dict(
            hardcode_alpha=BEST_ALPHAS.tolist(), hit_hardcal=hit_hardcal,
            fit_alpha=alpha_fit.tolist(), hit_fitcal=hit_fitcal,
            overfit_risk_flag=overfit_flag, flag_dev=flag_dev, flag_drop=flag_drop,
            headline="uncalibrated",
        ),
        rotation_class=dict(scalar=rot_class, seq=seq_rot_class),
        runtime_sec=round(time.time() - t0, 1),
    )
    print(f"[done] hit_1cm={hit_1cm:.4f} band={band} (kalman_alone={kalman_alone_hit:.4f}, "
          f"g1_pass={g1_pass}) runtime={result['runtime_sec']}s", flush=True)

    out_dir = _THIS
    out_name = args.out or (f"results_{'kr002' if args.input_yaw else 'kr001'}_{args.gate}.json")
    (out_dir / out_name).write_text(json.dumps(result, indent=2, ensure_ascii=False))
    if args.gate == "full":
        npz_name = out_name.replace(".json", ".npz")
        np.savez(out_dir / npz_name, oof_residual=ens_res, oof_pred=pred_uncal,
                 per_sample_hit=per_sample_hit, y=y, kalman_main=kalman_main,
                 fold_ids=fold_ids)
    print(f"[saved] {out_name}", flush=True)
    return result


if __name__ == "__main__":
    main()
