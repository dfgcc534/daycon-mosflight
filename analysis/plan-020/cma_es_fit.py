"""plan-020 c6 — CMA-ES fitter for 학습 후보 + annealed hit-direct objective.

Public API:
    fit_candidate(name, X_train, Y_train, end_idx, popsize=20, maxiter=200,
                  seed_list=[...]) → fit_params dict | None

각 후보의 CMA-ES spec (init / sigma / bounds / param names) 은 CANDIDATE_CMA_SPEC.
C05 per-regime / C14 KNN 은 별도 fit_per_regime_f0 / fit_knn 로 dispatch.
C02/C03/C06/C07/C11 (0-param) / C13 (degenerate=F0) 는 None 반환.

annealed objective τ schedule (manual CMA-ES iter loop):
  iter ∈ [0, M/3)        → τ=0.003
  iter ∈ [M/3, 2M/3)     → τ=0.001
  iter ∈ [2M/3, M)       → τ=0.0003
final eval (best-on-train): hard hit (τ→0 = 1-indicator).
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import cma
import numpy as np

_THIS_DIR = Path(__file__).parent


def _load_local(name: str):
    spec = importlib.util.spec_from_file_location(name, _THIS_DIR / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


fd = _load_local("formula_deterministic")

R_HIT_MAIN = 0.01
R_HIT_LOOSE = 0.015
DEFAULT_SEEDS = [20260518, 20260519, 20260520, 20260521, 20260522]

# ── CMA-ES spec per candidate ──────────────────────────────────────────


CANDIDATE_CMA_SPEC: dict[str, dict] = {
    "C01_helix": {
        "init": [1.0, 1.0, 1.0],
        "sigma": 0.3,
        "param_names": ["alpha", "beta", "gamma"],
        "bounds": None,  # 자유
    },
    "C04_imm": {
        "init": [1.0, 1.0, 1.0],
        "sigma": 0.5,
        "param_names": ["w_diag"],  # vec3 — special handling: w_diag 자체가 vec
        "bounds": [[0.1, 0.1, 0.1], [10.0, 10.0, 10.0]],
        "_w_diag_vec3": True,
    },
    "C08_singer": {
        "init": [0.100],
        "sigma": 0.05,
        "param_names": ["tau_a"],
        "bounds": [[0.020], [1.000]],
    },
    "C09_kalman_smoother": {
        "init": [-6.0, -4.0],
        "sigma": 1.0,
        "param_names": ["log_q", "log_r"],
        "bounds": [[-12.0, -12.0], [0.0, 0.0]],
    },
    "C10_bishop_frame": {
        "init": [1.0],
        "sigma": 0.5,
        "param_names": ["lam"],
        "bounds": [[-2.0], [2.0]],
    },
    "C12_wingbeat_corrected": {
        "init": [8.0],
        "sigma": 1.0,
        "param_names": ["f_c"],
        "bounds": [[2.27], [12.5]],
    },
}


def _smooth_hit_objective(pred: np.ndarray, gt: np.ndarray, tau: float) -> float:
    """L = − mean_i [ sigmoid((R_main − d)/τ) + 0.5 · sigmoid((R_loose − d)/τ) ]."""
    d = np.linalg.norm(pred - gt, axis=1)
    sh_main = 1.0 / (1.0 + np.exp(-(R_HIT_MAIN - d) / tau))
    sh_loose = 1.0 / (1.0 + np.exp(-(R_HIT_LOOSE - d) / tau))
    return -float((sh_main + 0.5 * sh_loose).mean())


def _hard_hit_objective(pred: np.ndarray, gt: np.ndarray) -> float:
    d = np.linalg.norm(pred - gt, axis=1)
    return -float((d <= R_HIT_MAIN).mean())


def _params_to_fit_params(name: str, p_flat: np.ndarray) -> dict:
    spec = CANDIDATE_CMA_SPEC[name]
    if spec.get("_w_diag_vec3"):
        return {"w_diag": np.asarray(p_flat, dtype=float)}
    return {pn: float(p_flat[i]) for i, pn in enumerate(spec["param_names"])}


def _tau_for_iter(it: int, max_it: int) -> float:
    if it < max_it / 3:
        return 0.003
    if it < 2 * max_it / 3:
        return 0.001
    return 0.0003


def _run_cma(
    name: str,
    cand_fn: Any,
    X: np.ndarray,
    Y: np.ndarray,
    end_idx: int,
    popsize: int,
    maxiter: int,
    seed: int,
) -> tuple[np.ndarray, float]:
    spec = CANDIDATE_CMA_SPEC[name]
    opts = {
        "popsize": popsize,
        "maxiter": maxiter,
        "tolfun": 1e-5,
        "seed": seed,
        "verbose": -9,
    }
    if spec.get("bounds") is not None:
        opts["bounds"] = spec["bounds"]
    es = cma.CMAEvolutionStrategy(spec["init"], spec["sigma"], opts)
    for it in range(maxiter):
        if es.stop():
            break
        sols = es.ask()
        tau = _tau_for_iter(it, maxiter)
        fits = []
        for s in sols:
            fp = _params_to_fit_params(name, np.asarray(s))
            try:
                pred = cand_fn(X, end_idx, fp)
                if not np.isfinite(pred).all():
                    fits.append(1e10)
                else:
                    fits.append(_smooth_hit_objective(pred, Y, tau))
            except Exception:
                fits.append(1e10)
        es.tell(sols, fits)
    best = es.result.xbest if es.result.xbest is not None else np.asarray(spec["init"])
    # hard-hit eval
    fp_best = _params_to_fit_params(name, best)
    try:
        pred = cand_fn(X, end_idx, fp_best)
        hard = _hard_hit_objective(pred, Y) if np.isfinite(pred).all() else 0.0
    except Exception:
        hard = 0.0
    return np.asarray(best), hard


def fit_candidate(
    name: str,
    X_train: np.ndarray,
    Y_train: np.ndarray,
    end_idx: int,
    popsize: int = 20,
    maxiter: int = 200,
    seed_list: list[int] | None = None,
    verbose: bool = False,
) -> dict | None:
    """Dispatch CMA-ES / per-regime / KNN by candidate name. Returns fit_params dict or None."""
    seed_list = seed_list or DEFAULT_SEEDS
    if name == "C05_per_regime_f0":
        return fit_per_regime_f0(X_train, Y_train, end_idx, popsize=popsize, maxiter=maxiter, seed=seed_list[0])
    if name == "C14_trajectory_knn":
        return fit_knn(X_train, Y_train, end_idx)
    if name in {"C02_ctra", "C03_ctrv", "C06_quintic_hermite", "C07_jerk_quartic", "C11_se3_twist", "C13_levy_prior"}:
        return None  # 0-param / degenerate

    if name not in CANDIDATE_CMA_SPEC:
        raise ValueError(f"unknown candidate name: {name}")

    cand_fn = fd.C01_TO_C14[name]
    best_score = float("inf")
    best_params: np.ndarray | None = None
    seed_scores: dict[int, float] = {}
    for seed in seed_list:
        params, hard_score = _run_cma(name, cand_fn, X_train, Y_train, end_idx, popsize, maxiter, seed)
        seed_scores[seed] = -hard_score  # hard_score is −hit_rate; store hit_rate
        if hard_score < best_score:
            best_score = hard_score
            best_params = params
        if verbose:
            print(f"  [{name}] seed={seed} train_hit={-hard_score:.4f}", flush=True)
    assert best_params is not None
    fp = _params_to_fit_params(name, best_params)
    fp["_seed_scores"] = seed_scores  # diagnostic
    fp["_best_train_hit"] = -best_score
    return fp


# ── C05 per-regime F0 ──────────────────────────────────────────────────


def fit_per_regime_f0(
    X: np.ndarray,
    Y: np.ndarray,
    end_idx: int,
    popsize: int = 20,
    maxiter: int = 100,
    seed: int = 20260518,
    min_samples: int = 100,
) -> dict:
    """18 regime 별 (d1, par, perp) CMA-ES fit. Bins + regimes 함께 반환."""
    import sys
    sys.path.insert(0, str(_THIS_DIR.parent.parent))
    from src.pb_0_6822.selector import fit_regime_bins, assign_regimes

    bins = fit_regime_bins(X, end_idx)
    regimes_train = assign_regimes(X, end_idx, bins)
    regime_params: dict[int, tuple[float, float, float]] = {}
    for r in range(18):
        mask = regimes_train == r
        if int(mask.sum()) < min_samples:
            regime_params[r] = (1.98, 1.20, -0.20)
            continue
        X_r, Y_r = X[mask], Y[mask]

        def _obj(p):
            pred = fd._f0_apply(X_r, end_idx, p[0], p[1], p[2])
            if not np.isfinite(pred).all():
                return 1e10
            d = np.linalg.norm(pred - Y_r, axis=1)
            # smooth hit (single τ for speed)
            sh = 1.0 / (1.0 + np.exp(-(R_HIT_MAIN - d) / 0.001))
            return -float(sh.mean())

        es = cma.CMAEvolutionStrategy([1.98, 1.20, -0.20], 0.3, {
            "popsize": popsize, "maxiter": maxiter, "tolfun": 1e-5, "seed": seed, "verbose": -9,
        })
        es.optimize(_obj)
        best = es.result.xbest if es.result.xbest is not None else np.array([1.98, 1.20, -0.20])
        regime_params[r] = (float(best[0]), float(best[1]), float(best[2]))

    return {"regime_params": regime_params, "bins": bins, "regimes_train": regimes_train}


# ── C14 KNN ────────────────────────────────────────────────────────────


def fit_knn(X: np.ndarray, Y: np.ndarray, end_idx: int, k_grid: tuple[int, ...] = (1, 3, 5, 10, 20)) -> dict:
    """C14: build v_last frame normalized 33D query + sklearn KNN, pick best k by train hit@1cm.
    Note: transductive (full-fit then predict-on-train) for k selection — slight overfit risk
          mitigated by §6.1 C14 spec 의 fold-internal usage (run_oof 는 train_(not k) 만 전달)."""
    from sklearn.neighbors import KNeighborsRegressor

    R_s, origin_s = fd._v_last_frame(X, end_idx)
    T_obs = end_idx + 1
    rel = X[:, :T_obs] - origin_s[:, None, :]
    traj_frame = np.einsum("nij,ntj->nti", R_s, rel)
    query = traj_frame.reshape(X.shape[0], -1)  # (N, 33)
    disp_frame = np.einsum("nij,nj->ni", R_s, Y - origin_s)  # (N, 3)

    best_k, best_hit = int(k_grid[0]), -1.0
    k_log: dict[int, float] = {}
    for k in k_grid:
        knn = KNeighborsRegressor(n_neighbors=k, n_jobs=1)
        knn.fit(query, disp_frame)
        disp_pred = knn.predict(query)
        pred = origin_s + np.einsum("nij,nj->ni", R_s.transpose(0, 2, 1), disp_pred)
        hit = float((np.linalg.norm(pred - Y, axis=1) <= R_HIT_MAIN).mean())
        k_log[k] = hit
        if hit > best_hit:
            best_hit, best_k = hit, int(k)

    final_knn = KNeighborsRegressor(n_neighbors=best_k, n_jobs=1)
    final_knn.fit(query, disp_frame)
    return {"k": best_k, "knn": final_knn, "k_grid_train_hit": k_log}
