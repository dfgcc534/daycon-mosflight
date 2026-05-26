"""plan-a-002 OOF orchestrator (§4.3).

KR003 (--innov --filtered-v --cv-ca) / KR004 (+--filtered-yaw) 5-fold OOF hit_1cm.
- plan-a-001 building block (model/losses/yaw/kalman/features) import 재사용.
- train_one/run_config 는 n_channels 를 seq dim 에서 동적 추론 (plan-a-001 의 9-하드코딩 일반화).
- baseline(KR002) carry: 2cfg(A/B)×5fold stable_fold_id×3seed×200ep, combo+aux F/W.
- headline = uncalibrated OOF hit_1cm. --compare-to npz 와 paired permutation (G_kalman/G_frame).

Usage:
  python analysis/plan-a-002/run_oof.py --gate smoke --innov --filtered-v --cv-ca --input-yaw
  python analysis/plan-a-002/run_oof.py --gate full --innov --filtered-v --cv-ca --input-yaw \
      --out results_kr003.json --compare-to ../plan-a-001/results_kr002.npz --exp KR003
  python analysis/plan-a-002/run_oof.py --gate full --innov --filtered-v --cv-ca --input-yaw \
      --filtered-yaw --out results_kr004.json --compare-to results_kr003.npz --exp KR004
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
_A001 = _THIS.parent / "plan-a-001"
_REPO = _THIS.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.io import load_all_samples, load_labels  # noqa: E402
from src.pb_0_6822.selector import stable_fold_id  # noqa: E402


def _load(base: Path, name: str):
    spec = importlib.util.spec_from_file_location(f"m_{name}", base / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_kalman = _load(_A001, "kalman")
_yaw = _load(_A001, "yaw")
_model = _load(_A001, "model")
_losses = _load(_A001, "losses")
_kf = _load(_THIS, "kalman_features")
_fe = _load(_THIS, "features_ext")

R_HIT = 0.01
R_HIT_LOOSE = 0.015

CONFIGS = {
    "A": dict(hidden_size=64, num_layers=1, fc_hidden=128, lr=5e-4, p=0.3, wd=1e-4),
    "B": dict(hidden_size=64, num_layers=1, fc_hidden=128, lr=1e-3, p=0.1, wd=1e-4),
}


def hit_rate(pred, y, thr=R_HIT):
    return float((np.linalg.norm(pred - y, axis=-1) <= thr).mean())


def hit_mask(pred, y, thr=R_HIT):
    return np.linalg.norm(pred - y, axis=-1) <= thr


def paired_perm(hit_b, hit_a, n_resample=10000, seed=0):
    """paired sign-flip permutation. stat=mean(hit_b−hit_a). (delta, p). (compare.py 동일.)"""
    d = hit_b.astype(np.float64) - hit_a.astype(np.float64)
    obs = d.mean()
    rng = np.random.default_rng(seed)
    signs = rng.choice([1.0, -1.0], size=(n_resample, d.shape[0]))
    null = (signs * d[None, :]).mean(axis=1)
    p = float((np.abs(null) >= abs(obs)).mean())
    return float(obs), p


def train_one(cfg, seq_tr, scal_tr, tgt_main, tgt_F, tgt_W,
              seq_va, scal_va, theta_va, kal_va, y_va,
              seed, device, epochs, patience, batch=256,
              seq_test=None, scal_test=None):
    """단일 config×seed×fold 학습 → (best_val_out_rot, best_rhit, test_out_rot|None). n_channels 동적.

    test_out_rot = best-val-epoch state 로 예측한 test (rotated frame). seq_test 미지정 시 None.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    n_ch = seq_tr.shape[2]
    net = _model.GRUModelMultiAux(
        n_channels=n_ch, scal_dim=scal_tr.shape[1], hidden_size=cfg["hidden_size"],
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

    best_rhit, best_out, best_state, no_imp = -1.0, None, None, 0
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
        net.eval()
        with torch.no_grad():
            om_va, _ = net(sq_va, sc_va)
        out_rot = om_va.cpu().numpy()
        pred = kal_va + _yaw.inverse_rotate_xy(out_rot, theta_va)
        rh = hit_rate(pred, y_va)
        if rh > best_rhit:
            best_rhit, best_out, no_imp = rh, out_rot.copy(), 0
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            no_imp += 1
            if no_imp >= patience:
                break
    test_out = None
    if seq_test is not None:
        net.load_state_dict(best_state)
        net.eval()
        with torch.no_grad():
            om_te, _ = net(torch.as_tensor(seq_test, device=device),
                           torch.as_tensor(scal_test, device=device))
        test_out = om_te.cpu().numpy()
    return best_out, best_rhit, test_out


def run_config(cfg_name, X, y, fold_ids, theta, kalman_main, seq, scal,
               tgt_main, tgt_F, tgt_W, folds, seeds, epochs, patience, device, quiet,
               seq_test=None, scal_test=None, theta_test=None):
    """config 1개의 5-fold → (oof_res_world (N,3), test_res_world (Nt,3)|None). n_channels 동적.

    test 는 fold 별 seed-평균(rotated)→inverse_rotate→fold 평균 (plan-a-001 동일).
    """
    cfg = CONFIGS[cfg_name]
    N, _, C = seq.shape
    oof_rot = np.zeros((N, 3), dtype=np.float64)
    test_world_folds = []
    for f in folds:
        tr = np.where(fold_ids != f)[0]
        va = np.where(fold_ids == f)[0]
        sc_seq = StandardScaler().fit(seq[tr].reshape(-1, C))
        sc_scal = StandardScaler().fit(scal[tr])
        seq_tr = _fe.normalize_seq(seq[tr], sc_seq)
        seq_va = _fe.normalize_seq(seq[va], sc_seq)
        scal_tr = sc_scal.transform(scal[tr]).astype(np.float32)
        scal_va = sc_scal.transform(scal[va]).astype(np.float32)
        seq_te_n = _fe.normalize_seq(seq_test, sc_seq) if seq_test is not None else None
        scal_te_n = sc_scal.transform(scal_test).astype(np.float32) if scal_test is not None else None
        seed_outs, seed_test = [], []
        for s in seeds:
            out_rot, rh, test_rot = train_one(
                cfg, seq_tr, scal_tr, tgt_main[tr], tgt_F[tr], tgt_W[tr],
                seq_va, scal_va, theta[va], kalman_main[va], y[va],
                s, device, epochs, patience,
                seq_test=seq_te_n, scal_test=scal_te_n)
            seed_outs.append(out_rot)
            if test_rot is not None:
                seed_test.append(test_rot)
            if not quiet:
                print(f"    [{cfg_name}] fold{f} seed{s} val_rhit={rh:.4f}", flush=True)
        oof_rot[va] = np.mean(seed_outs, axis=0)
        if seed_test:
            test_world_folds.append(_yaw.inverse_rotate_xy(np.mean(seed_test, axis=0), theta_test))
    oof_world = _yaw.inverse_rotate_xy(oof_rot, theta)
    test_world = np.mean(test_world_folds, axis=0) if test_world_folds else None
    return oof_world, test_world


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", choices=["smoke", "g1", "full"], default="full")
    ap.add_argument("--input-yaw", action="store_true", help="입력 seq triplet 회전 (KR002 carry)")
    ap.add_argument("--innov", action="store_true", help="innovation seq +3채널")
    ap.add_argument("--filtered-v", action="store_true", help="filtered velocity seq +3채널")
    ap.add_argument("--cv-ca", action="store_true", help="CV/CA 불일치 scalar +4")
    ap.add_argument("--filtered-yaw", action="store_true", help="KR004: θ=yaw(filtered v_last)")
    ap.add_argument("--exp", default=None, help="exp 라벨 (KR003/KR004)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--compare-to", default=None, help="baseline npz (paired permutation)")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--seeds-cpu", type=int, default=1, help="CPU 시 seed 수 (decision-note carry)")
    ap.add_argument("--predict-test", action="store_true", help="test 10000 예측 → submission_{exp}.csv (full only)")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    if args.gate == "smoke":
        folds, seeds, epochs, configs, max_n = [0], [0], 1, ["A"], 512
    elif args.gate == "g1":
        folds, seeds, epochs, configs, max_n = [0], [0], args.epochs, ["A"], None
    else:
        folds, seeds, epochs, configs, max_n = list(range(5)), [0, 1, 2], args.epochs, ["A", "B"], None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu" and args.gate == "full":
        seeds = list(range(args.seeds_cpu))  # decision-note carry: CPU 시 seed 감소
    t0 = time.time()
    need_internals = args.innov or args.filtered_v or args.filtered_yaw
    print(f"[plan-a-002 OOF] exp={args.exp} gate={args.gate} device={device} epochs={epochs} "
          f"flags(input_yaw={args.input_yaw} innov={args.innov} filtered_v={args.filtered_v} "
          f"cv_ca={args.cv_ca} filtered_yaw={args.filtered_yaw}) configs={configs} seeds={seeds}",
          flush=True)

    ids, X = load_all_samples("train")
    lab_ids, y = load_labels()
    assert ids == lab_ids, "id 정렬 불일치"
    if max_n is not None:
        X, y, ids = X[:max_n], y[:max_n], ids[:max_n]
    N = X.shape[0]
    fold_ids = np.array([stable_fold_id(i, 5) for i in ids])
    pm = np.isin(fold_ids, folds)

    # Kalman internals (innov/filtered_v) — leakage-safe 관측창 산출
    innov_arr = filtered_v_arr = None
    if need_internals:
        _, innov_arr, filtered_v_arr = _kf.kalman_with_internals(X)

    # theta: filtered-yaw(KR004) → yaw(filtered v_last), else raw v_last yaw (KR002 carry)
    if args.filtered_yaw:
        theta = _yaw.yaw_angle(filtered_v_arr[:, -1, :].astype(np.float64))
    else:
        theta = _yaw.yaw_from_last_step(X)

    kalman_main = _kalman.kalman_predict(X, sigma_obs=_kalman.SIGMA_OBS_MAIN, sigma_proc=_kalman.SIGMA_PROC_MAIN)
    kalman_alt = _kalman.kalman_predict(X, sigma_obs=_kalman.SIGMA_OBS_ALT, sigma_proc=_kalman.SIGMA_PROC_ALT)
    tgt_main = _yaw.rotate_xy(y - kalman_main, theta).astype(np.float32)
    tgt_F = _yaw.rotate_xy(y - X[:, -1], theta).astype(np.float32)
    tgt_W = _yaw.rotate_xy(y - kalman_alt, theta).astype(np.float32)

    cache = _THIS / "noise_cache.npz"
    noise = _fe.compute_noise(X, cache_path=(None if max_n else cache), key="train", with_loo=True)
    seq, seq_names = _fe.build_seq_ext(
        X, innov_arr=(innov_arr if args.innov else None),
        filtered_v_arr=(filtered_v_arr if args.filtered_v else None),
        theta=theta, input_yaw=args.input_yaw)
    cv_ca_arr = _kf.cv_ca_disagreement(X) if args.cv_ca else None
    scal, scal_names, scal_rot = _fe.build_scalar_ext(
        X, noise["poly2"], noise["savgol"], noise["loo"],
        cv_ca_arr=cv_ca_arr, theta=theta, input_yaw=args.input_yaw)
    seq_rot = {n: "rotate" for n in seq_names}  # 전부 벡터 triplet

    # test 예측 features (--predict-test) — DACON submission. test internals 도 관측창 산출(leakage-safe).
    seq_test = scal_test = theta_test = kalman_test = test_ids = None
    if args.predict_test:
        if max_n is not None:
            raise SystemExit("--predict-test 는 full gate 전용 (max_n=None)")
        test_ids, X_test = load_all_samples("test")
        innov_te = fv_te = None
        if need_internals:
            _, innov_te, fv_te = _kf.kalman_with_internals(X_test)
        theta_test = (_yaw.yaw_angle(fv_te[:, -1, :].astype(np.float64)) if args.filtered_yaw
                      else _yaw.yaw_from_last_step(X_test))
        kalman_test = _kalman.kalman_predict(
            X_test, sigma_obs=_kalman.SIGMA_OBS_MAIN, sigma_proc=_kalman.SIGMA_PROC_MAIN)
        noise_te = _fe.compute_noise(X_test, cache_path=cache, key="test", with_loo=False)
        seq_test, _ = _fe.build_seq_ext(
            X_test, innov_arr=(innov_te if args.innov else None),
            filtered_v_arr=(fv_te if args.filtered_v else None),
            theta=theta_test, input_yaw=args.input_yaw)
        cv_ca_te = _kf.cv_ca_disagreement(X_test) if args.cv_ca else None
        scal_test, _, _ = _fe.build_scalar_ext(
            X_test, noise_te["poly2"], noise_te["savgol"], noise_te["loo"],
            cv_ca_arr=cv_ca_te, theta=theta_test, input_yaw=args.input_yaw)

    kalman_alone_hit = hit_rate(kalman_main[pm], y[pm])

    oof_res, test_res = {}, {}
    for c in configs:
        oof_res[c], test_res[c] = run_config(
            c, X, y, fold_ids, theta, kalman_main, seq, scal,
            tgt_main, tgt_F, tgt_W, folds, seeds, epochs, args.patience, device, args.quiet,
            seq_test=seq_test, scal_test=scal_test, theta_test=theta_test)
        h = hit_rate((kalman_main + oof_res[c])[pm], y[pm])
        print(f"  config {c} OOF hit_1cm={h:.4f}", flush=True)

    ens_res = np.mean([oof_res[c] for c in configs], axis=0)
    pred = kalman_main + ens_res
    hit_1cm = hit_rate(pred[pm], y[pm])
    hit_1p5cm = hit_rate(pred[pm], y[pm], R_HIT_LOOSE)
    per_sample_hit = hit_mask(pred, y)

    # baseline 비교 (paired permutation)
    compare = None
    if args.compare_to:
        cpath = Path(args.compare_to)
        if not cpath.is_absolute():
            cpath = _THIS / cpath
        zb = np.load(cpath)
        hit_base = zb["per_sample_hit"]
        assert np.allclose(zb["y"], y), "baseline y 정렬 불일치"
        delta, pval = paired_perm(per_sample_hit[pm], hit_base[pm])
        base_hit = float(hit_base[pm].mean())
        # band: KR003(vs KR002)=no-regression; KR004(vs KR003)=positive/neutral/negative
        if args.exp == "KR003":
            if delta >= 0.002 and pval < 0.05:
                band = "positive"
            elif hit_1cm >= base_hit - 0.001:
                band = "no_regression_PASS"
            else:
                band = "FAIL_regression"
        else:  # KR004 vs KR003 (G_frame)
            if delta >= 0.002 and pval < 0.05:
                band = "positive"
            elif delta <= -0.002 and pval < 0.05:
                band = "negative"
            else:
                band = "neutral"
        compare = dict(compare_to=str(cpath.name), base_hit_1cm=base_hit,
                       delta=delta, p=pval, band=band)
        print(f"  [compare vs {cpath.name}] base={base_hit:.4f} Δ={delta:+.4f} p={pval:.4g} → {band}",
              flush=True)

    # submission (--predict-test): uncalibrated test 예측 (OOF headline 동일 기준)
    submission_name = None
    if args.predict_test and test_ids is not None:
        import pandas as pd
        ens_test_res = np.mean([test_res[c] for c in configs], axis=0)
        test_pred = kalman_test + ens_test_res
        sub = pd.DataFrame({"id": test_ids, "x": test_pred[:, 0],
                            "y": test_pred[:, 1], "z": test_pred[:, 2]})
        submission_name = f"submission_{(args.exp or 'kr00x').lower()}.csv"
        sub.to_csv(_THIS / submission_name, index=False)
        print(f"[submission] {submission_name} ({len(sub)} rows, uncalibrated, "
              f"finite={bool(np.isfinite(test_pred).all())})", flush=True)

    result = dict(
        exp=args.exp, gate=args.gate, device=str(device),
        flags=dict(input_yaw=args.input_yaw, innov=args.innov, filtered_v=args.filtered_v,
                   cv_ca=args.cv_ca, filtered_yaw=args.filtered_yaw),
        N=int(N), n_predicted=int(pm.sum()),
        folds=[int(f) for f in folds], seeds=[int(s) for s in seeds],
        epochs=epochs, configs=configs,
        seq_dim=int(seq.shape[2]), scal_dim=int(scal.shape[1]),
        hit_1cm=hit_1cm, hit_1p5cm=hit_1p5cm,
        config_hit_1cm={c: hit_rate((kalman_main + oof_res[c])[pm], y[pm]) for c in configs},
        kalman_alone_hit_1cm=kalman_alone_hit,
        compare=compare,
        submission=submission_name,
        rotation_class=dict(seq=seq_rot, scalar=scal_rot),
        runtime_sec=round(time.time() - t0, 1),
    )
    print(f"[done] exp={args.exp} hit_1cm={hit_1cm:.4f} (kalman_alone={kalman_alone_hit:.4f}) "
          f"runtime={result['runtime_sec']}s", flush=True)

    out_name = args.out or f"results_{(args.exp or 'kr00x').lower()}_{args.gate}.json"
    (_THIS / out_name).write_text(json.dumps(result, indent=2, ensure_ascii=False))
    if args.gate == "full":
        np.savez(_THIS / out_name.replace(".json", ".npz"),
                 oof_residual=ens_res, oof_pred=pred, per_sample_hit=per_sample_hit,
                 y=y, kalman_main=kalman_main, fold_ids=fold_ids)
    print(f"[saved] {out_name}", flush=True)
    return result


if __name__ == "__main__":
    main()
