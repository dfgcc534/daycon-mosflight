"""plan-020 §5/§6/§7 — 5-fold OOF runner.

CLI:
    python analysis/plan-020/run_oof.py --candidate f0_baseline --out-json ...
    python analysis/plan-020/run_oof.py --candidate all --out-json ...
    python analysis/plan-020/run_oof.py --candidate N01_mlp_coef --device cuda:1

Dispatch:
  - f0_baseline                       → numpy 5-fold OOF reproduce (G1)
  - C01_helix..C14_trajectory_knn    → per-fold CMA-ES fit (if c6 available) + apply
  - N01_mlp_coef / N02_tcn_coef / N05_moe → per-fold multi-seed train + best-on-train

Output: JSON per candidate, schema follows §5.2 / §6.2 / §7.3.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.io import load_all_samples, load_labels  # noqa: E402
from src.pb_0_6822.selector import stable_fold_id, fit_regime_bins, assign_regimes  # noqa: E402

# direct file load for hyphenated plan-020 dir
_THIS_DIR = Path(__file__).parent


def _load_module(name: str):
    spec = importlib.util.spec_from_file_location(name, _THIS_DIR / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


bf = _load_module("baseline_f0")
fd = _load_module("formula_deterministic")
try:
    fnn = _load_module("formula_nn")
except Exception as exc:
    fnn = None
    _nn_import_err = exc
try:
    cma = _load_module("cma_es_fit")
except Exception:
    cma = None  # c6 가 아직 없을 수 있음 — None default param fallback


N_FOLDS = 5
END_IDX = 10  # T-1 (T = 11)
R_HIT = 0.01
R_HIT_LOOSE = 0.015


# ── shared helpers ────────────────────────────────────────────────────


def assign_folds(ids: list[str]) -> np.ndarray:
    """stable_fold_id(sample_id_str, 5) → fold index. Returns (N,) int."""
    return np.asarray([stable_fold_id(str(sid), N_FOLDS) for sid in ids], dtype=int)


def hit_rates_per_fold(pred: np.ndarray, gt: np.ndarray, folds: np.ndarray) -> tuple[list[float], list[float], float, float]:
    """Returns (per_fold_hit_1cm, per_fold_hit_1.5cm, concat_hit_1cm, concat_hit_1.5cm)."""
    d = np.linalg.norm(pred - gt, axis=1)
    per_1cm, per_loose = [], []
    for k in range(N_FOLDS):
        m = folds == k
        per_1cm.append(float((d[m] <= R_HIT).mean()) if m.any() else 0.0)
        per_loose.append(float((d[m] <= R_HIT_LOOSE).mean()) if m.any() else 0.0)
    concat_1cm = float((d <= R_HIT).mean())
    concat_loose = float((d <= R_HIT_LOOSE).mean())
    return per_1cm, per_loose, concat_1cm, concat_loose


def paired_delta_sample_level(pred_cand: np.ndarray, pred_f0: np.ndarray, gt: np.ndarray, R: float) -> float:
    """§3.3 sample-level paired Δ = mean_i(1{||pred_cand-gt||≤R} − 1{||pred_F0-gt||≤R})."""
    d_cand = np.linalg.norm(pred_cand - gt, axis=1)
    d_f0 = np.linalg.norm(pred_f0 - gt, axis=1)
    return float((d_cand <= R).astype(float).mean() - (d_f0 <= R).astype(float).mean())


# ── F0 baseline OOF (G1) ──────────────────────────────────────────────


def run_f0_baseline_oof(X: np.ndarray, Y: np.ndarray, folds: np.ndarray) -> dict[str, Any]:
    """G1 — numpy F0 산식 그대로 5-fold (deterministic, fold 와 무관 — fold split 만 reporting 용)."""
    pred = bf.f0_baseline(X, end_idx=END_IDX)
    per_1, per_15, concat_1, concat_15 = hit_rates_per_fold(pred, Y, folds)
    return {
        "candidate": "f0_baseline",
        "n_samples": int(X.shape[0]),
        "hit_1cm_5fold_concat": concat_1,
        "hit_1.5cm_5fold_concat": concat_15,
        "hit_1cm_per_fold": per_1,
        "hit_1.5cm_per_fold": per_15,
        "fold_variance_1cm": float(np.std(per_1)),
        "fold_variance_1.5cm": float(np.std(per_15)),
        "pred_array": pred,  # 메모리 keep 위해 caller 가 dump 시 drop
    }


# ── deterministic candidate OOF ───────────────────────────────────────


def run_deterministic_oof(
    name: str,
    fn: Callable,
    X: np.ndarray,
    Y: np.ndarray,
    folds: np.ndarray,
    pred_f0: np.ndarray,
    cma_kwargs: dict | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """C01..C14 — per-fold CMA-ES fit (cma module 가 있으면) + apply.
    0-param 후보 (C02/C03/C06/C07/C10[default]/C11/C13) 는 cma 호출 skip → fit_params=None."""
    cma_kwargs = cma_kwargs or {}
    pred = np.empty_like(Y)
    fit_log: dict[int, Any] = {}

    has_param = name not in {"C02_ctra", "C03_ctrv", "C06_quintic_hermite", "C07_jerk_quartic", "C11_se3_twist", "C13_levy_prior"}
    for k in range(N_FOLDS):
        train_idx = np.where(folds != k)[0]
        val_idx = np.where(folds == k)[0]
        fit_params: dict | None = None
        if has_param and cma is not None and hasattr(cma, "fit_candidate"):
            try:
                fit_params = cma.fit_candidate(
                    name, X[train_idx], Y[train_idx], end_idx=END_IDX, **cma_kwargs
                )
            except Exception as exc:
                if verbose:
                    print(f"  [{name}] fold {k}: CMA-ES fit failed ({type(exc).__name__}), fallback to default.", flush=True)
                fit_params = None

        # C05 per-regime requires regimes assignment (fold-internal). bins comes from fit_params.
        if name == "C05_per_regime_f0" and fit_params is not None and "regimes" not in fit_params:
            bins = fit_params.get("bins") or fit_regime_bins(X[train_idx], end_idx=END_IDX)
            fit_params["regimes"] = assign_regimes(X[val_idx], end_idx=END_IDX, bins=bins)

        pred_val = fn(X[val_idx], end_idx=END_IDX, fit_params=fit_params)
        pred[val_idx] = pred_val
        fit_log[k] = {"params_present": fit_params is not None}
        if verbose:
            d = np.linalg.norm(pred_val - Y[val_idx], axis=1)
            print(f"  [{name}] fold {k}: val_hit@1cm={float((d <= R_HIT).mean()):.4f}", flush=True)

    per_1, per_15, concat_1, concat_15 = hit_rates_per_fold(pred, Y, folds)
    return {
        "candidate": name,
        "n_samples": int(X.shape[0]),
        "hit_1cm": concat_1,
        "hit_1.5cm": concat_15,
        "delta_1cm": paired_delta_sample_level(pred, pred_f0, Y, R_HIT),
        "delta_1.5cm": paired_delta_sample_level(pred, pred_f0, Y, R_HIT_LOOSE),
        "hit_1cm_per_fold": per_1,
        "hit_1.5cm_per_fold": per_15,
        "fold_variance_1cm": float(np.std(per_1)),
        "fold_variance_1.5cm": float(np.std(per_15)),
        "fit_log": fit_log,
        "pred_array": pred,
    }


# ── NN candidate OOF (multi-seed best-on-train) ───────────────────────


def build_seq_feats_11(x: np.ndarray) -> np.ndarray:
    """Helper: x (N, 11, 3) → (N, 11, 9) [px,py,pz, vx,vy,vz, ax,ay,az] displacement units."""
    N, T, _ = x.shape
    out = np.zeros((N, T, 9), dtype=np.float64)
    for t in range(T):
        out[:, t, 0:3] = x[:, t]
        if t >= 1:
            out[:, t, 3:6] = x[:, t] - x[:, t - 1]
        if t >= 2:
            out[:, t, 6:9] = out[:, t, 3:6] - (x[:, t - 1] - x[:, t - 2])
    return out


def run_nn_oof(
    name: str,
    X: np.ndarray,
    Y: np.ndarray,
    folds: np.ndarray,
    pred_f0: np.ndarray,
    seeds: list[int],
    device: str = "cuda:1",
    epochs: int = 50,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    early_stop_patience: int = 10,
    expert_preds_global: np.ndarray | None = None,  # (N_total, K=4, 3) for N5
    verbose: bool = True,
) -> dict[str, Any]:
    if fnn is None:
        raise RuntimeError(f"formula_nn import failed: {_nn_import_err}")
    import torch

    seq_feats_11 = build_seq_feats_11(X).astype(np.float32)
    seq_feats_3 = seq_feats_11[:, -3:, :]  # (N, 3, 9)
    Y_f32 = Y.astype(np.float32)

    # factory + dispatcher per candidate
    if name == "N01_mlp_coef":
        factory = lambda: fnn.N01_MLPCoef()
        pred_fn = fnn.make_pred_fn_n1(bf.f0_form_torch)
        train_seq_np = seq_feats_3
    elif name == "N02_tcn_coef":
        factory = lambda: fnn.N02_TCNCoef()
        pred_fn = fnn.make_pred_fn_n2(bf.f0_form_torch)
        train_seq_np = seq_feats_11
    elif name == "N05_moe":
        factory = lambda: fnn.N05_MoE()
        pred_fn = fnn.make_pred_fn_n5()
        train_seq_np = seq_feats_11
        assert expert_preds_global is not None, "N05 requires expert_preds_global (N_total, K=4, 3)"
    else:
        raise ValueError(f"unknown NN candidate {name}")

    pred = np.empty_like(Y_f32)
    train_hit_log: dict[int, dict[int, float]] = {}

    for k in range(N_FOLDS):
        train_idx = np.where(folds != k)[0]
        val_idx = np.where(folds == k)[0]
        train_seq = torch.from_numpy(train_seq_np[train_idx])
        train_y = torch.from_numpy(Y_f32[train_idx])
        val_seq = torch.from_numpy(train_seq_np[val_idx])

        train_expert = None
        val_expert = None
        if expert_preds_global is not None:
            train_expert = torch.from_numpy(expert_preds_global[train_idx].astype(np.float32))
            val_expert = torch.from_numpy(expert_preds_global[val_idx].astype(np.float32))

        best_seed = None
        best_train_hit = -1.0
        best_state = None
        seed_log: dict[int, float] = {}
        for seed in seeds:
            model, train_hit, _hist = fnn.train_nn_fold(
                factory, pred_fn, train_seq, train_y,
                train_expert_preds=train_expert,
                epochs=epochs, batch_size=batch_size, lr=lr, weight_decay=weight_decay,
                seed=seed, device=device, early_stop_patience=early_stop_patience,
            )
            seed_log[seed] = train_hit
            if train_hit > best_train_hit:
                best_train_hit = train_hit
                best_seed = seed
                best_state = {k_: v.detach().clone() for k_, v in model.state_dict().items()}
        train_hit_log[k] = {"seeds": seed_log, "best_seed": best_seed, "best_train_hit": best_train_hit}
        if verbose:
            print(f"  [{name}] fold {k}: best_seed={best_seed} train_hit={best_train_hit:.4f}", flush=True)

        # eval best on val_k
        import torch
        dev = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")
        model = factory().to(dev)
        model.load_state_dict(best_state)
        model.eval()
        val_seq_d = val_seq.to(dev)
        val_expert_d = val_expert.to(dev) if val_expert is not None else None
        with torch.no_grad():
            val_pred = pred_fn(model, val_seq_d, val_expert_d).cpu().numpy()
        pred[val_idx] = val_pred

    per_1, per_15, concat_1, concat_15 = hit_rates_per_fold(pred, Y, folds)
    return {
        "candidate": name,
        "n_samples": int(X.shape[0]),
        "hit_1cm": concat_1,
        "hit_1.5cm": concat_15,
        "delta_1cm": paired_delta_sample_level(pred, pred_f0, Y, R_HIT),
        "delta_1.5cm": paired_delta_sample_level(pred, pred_f0, Y, R_HIT_LOOSE),
        "hit_1cm_per_fold": per_1,
        "hit_1.5cm_per_fold": per_15,
        "fold_variance_1cm": float(np.std(per_1)),
        "fold_variance_1.5cm": float(np.std(per_15)),
        "train_hit_log": train_hit_log,
        "pred_array": pred,
    }


# ── CLI ────────────────────────────────────────────────────────────────


def _drop_arrays(d: dict) -> dict:
    """Remove numpy arrays from dict before json dump."""
    return {k: v for k, v in d.items() if not isinstance(v, np.ndarray)}


def main():
    ap = argparse.ArgumentParser(description="plan-020 5-fold OOF runner")
    ap.add_argument("--candidate", type=str, required=True,
                    help="f0_baseline | C01_helix..C14_trajectory_knn | N01_mlp_coef | N02_tcn_coef | N05_moe | all_deterministic | all_nn | all")
    ap.add_argument("--out-json", type=Path, default=None)
    ap.add_argument("--device", type=str, default="cuda:1")
    ap.add_argument("--seeds", type=int, nargs="+", default=[20260518, 20260519, 20260520, 20260521, 20260522])
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--cma-popsize", type=int, default=20, help="CMA-ES popsize (deterministic 학습 후보).")
    ap.add_argument("--cma-maxiter", type=int, default=200, help="CMA-ES maxiter.")
    ap.add_argument("--cma-seeds", type=int, nargs="+", default=None,
                    help="CMA-ES seed list (default: --seeds 와 동일).")
    args = ap.parse_args()

    t0 = time.time()
    print(f"[plan-020 run_oof] loading train data ...", flush=True)
    ids, X = load_all_samples(split="train")
    label_ids, Y = load_labels()
    assert ids == label_ids, "id mismatch between samples and labels"
    print(f"[plan-020 run_oof] N={X.shape[0]}, shape={X.shape}", flush=True)

    folds = assign_folds(ids)
    print(f"[plan-020 run_oof] fold sizes: {np.bincount(folds).tolist()}", flush=True)

    # F0 baseline first (needed for paired Δ on all other candidates)
    f0_out = run_f0_baseline_oof(X, Y, folds)
    pred_f0 = f0_out["pred_array"]
    print(f"[plan-020 run_oof] F0 baseline: hit@1cm={f0_out['hit_1cm_5fold_concat']:.4f} hit@1.5cm={f0_out['hit_1.5cm_5fold_concat']:.4f}", flush=True)

    results: dict[str, dict] = {"f0_baseline": _drop_arrays(f0_out)}

    deterministic_names = list(fd.C01_TO_C14.keys())
    nn_names = ["N01_mlp_coef", "N02_tcn_coef", "N05_moe"]

    targets: list[str] = []
    if args.candidate == "f0_baseline":
        targets = []
    elif args.candidate == "all_deterministic":
        targets = deterministic_names
    elif args.candidate == "all_nn":
        targets = nn_names
    elif args.candidate == "all":
        targets = deterministic_names + nn_names
    else:
        targets = [args.candidate]

    for name in targets:
        t_sub = time.time()
        print(f"\n[plan-020 run_oof] === {name} ===", flush=True)
        if name in fd.C01_TO_C14:
            cma_kwargs = {
                "popsize": args.cma_popsize,
                "maxiter": args.cma_maxiter,
                "seed_list": args.cma_seeds or args.seeds,
            }
            res = run_deterministic_oof(name, fd.C01_TO_C14[name], X, Y, folds, pred_f0, cma_kwargs=cma_kwargs)
        elif name in nn_names:
            # N5 expert_preds = pre-computed F0 / helix / hermite / ctra over full dataset
            expert_preds_global = None
            if name == "N05_moe":
                preds = np.stack([
                    bf.f0_baseline(X, end_idx=END_IDX),
                    fd.C01_TO_C14["C01_helix"](X, end_idx=END_IDX),
                    fd.C01_TO_C14["C06_quintic_hermite"](X, end_idx=END_IDX),
                    fd.C01_TO_C14["C02_ctra"](X, end_idx=END_IDX),
                ], axis=1)  # (N, 4, 3)
                expert_preds_global = preds
            res = run_nn_oof(
                name, X, Y, folds, pred_f0,
                seeds=args.seeds, device=args.device, epochs=args.epochs,
                batch_size=args.batch_size, lr=args.lr,
                expert_preds_global=expert_preds_global,
            )
        else:
            raise SystemExit(f"unknown candidate: {name}")
        results[name] = _drop_arrays(res)
        elapsed = time.time() - t_sub
        print(f"  [{name}] elapsed {elapsed:.1f}s", flush=True)

    elapsed_total = time.time() - t0
    print(f"\n[plan-020 run_oof] total {elapsed_total:.1f}s", flush=True)

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(results, indent=2, default=str))
        print(f"[plan-020 run_oof] wrote {args.out_json}", flush=True)
    else:
        print(json.dumps(results, indent=2, default=str)[:2000])


if __name__ == "__main__":
    main()
