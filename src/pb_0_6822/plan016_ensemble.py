"""plan-016 multi-seed × multi-fold ensemble runner.

§5 G1 (Path A) / §6 G2 (Path B) / §7-9 G3-G5 (Path C-B/C/D) 공통 사용.

Spec (plan-016 §5.2 OOF aggregation rule):
  1. 각 fold f 마다 5 seed val prediction `pred_seed_k_fold_f (N_val_f, 3)` 산출.
  2. 좌표 mean over seeds: `oof_pred_fold_f = mean over k of pred_seed_k_fold_f`.
  3. 5 fold concat → `oof_pred_all (10000, 3)` (각 sample 1번씩).
  4. **5-fold concat OOF hit@1cm = mean(‖oof_pred_all − y_true‖₂ ≤ 0.01m)**.

Test ensemble:
  - 25 model 의 test prediction 좌표 mean → single submission.

Usage (programmatic):
    from src.pb_0_6822 import plan016_ensemble as pe
    result = pe.run_multiseed_kfold(
        ids_train, X_train, Y_train, ids_test, X_test,
        config_base, seeds=[20260514, ...], f0_function=...,
    )
    # result["overall_oof_hit_1cm"], result["test_pred"], result["fold_oof"], result["fold_results"]
"""

from __future__ import annotations

import time
from dataclasses import asdict, replace
from typing import Any

import numpy as np

from src.pb_0_6822 import plan014_paradigm as pp
from src.pb_0_6822 import plan015_train as p15t


def run_multiseed_kfold(
    ids_train: list[str],
    X_train: np.ndarray,
    Y_train: np.ndarray,
    ids_test: list[str],
    X_test: np.ndarray,
    config_base: pp.TrainConfig,
    seeds: list[int],
    f0_function: pp.Plan014F0Function | None = None,
    progress_cb=None,
) -> dict[str, Any]:
    """Multi-seed × multi-fold 학습 + OOF aggregation + test ensemble.

    Args:
        config_base: seed 외 모든 lever 고정. 각 seed 마다 `replace(config_base, seed=s)` 변형.
        seeds: 5 seed list (§5.2 spec).

    Returns:
        dict with:
            - "overall_oof_hit_1cm": 단일 scalar (좌표 mean over seeds → concat → hit@1cm)
            - "oof_pred_all": (N_train, 3) — 좌표 mean OOF
            - "test_pred": (N_test, 3) — 좌표 mean over all (seed × fold) test predictions
            - "fold_results": list of dicts (seed, fold, val_hit, dcm, best_epoch, elapsed)
            - "per_seed_oof_hit_1cm": list of len(seeds) — 각 seed 의 5-fold concat OOF hit@1cm
            - "fold_oof_hit_per_fold": dict {fold: hit} — fold 별 (seed-mean) OOF hit (sub-metric)
    """
    if f0_function is None:
        f0_function = pp.Plan014F0Function()

    fold_of = np.array([pp.stable_hash_fold(s) for s in ids_train])
    N_train = X_train.shape[0]
    N_test = X_test.shape[0]
    n_seeds = len(seeds)
    n_folds = pp.N_FOLDS

    # per-seed per-fold val prediction container: shape (n_seeds, N_train, 3), nan for non-val
    oof_per_seed = np.full((n_seeds, N_train, 3), np.nan, dtype=np.float32)
    # per-seed per-fold test prediction container: (n_seeds, n_folds, N_test, 3)
    test_per_seed_fold = np.full((n_seeds, n_folds, N_test, 3), np.nan, dtype=np.float32)
    fold_results = []

    for si, seed in enumerate(seeds):
        cfg_seed = replace(config_base, seed=seed)
        for f in range(n_folds):
            train_mask = fold_of != f
            val_mask = fold_of == f
            t0 = time.time()
            res = pp.train_one_fold(
                cfg_seed, fold_id=f,
                X_train=X_train[train_mask], Y_train=Y_train[train_mask],
                X_val=X_train[val_mask], Y_val=Y_train[val_mask],
                f0_function=f0_function,
                X_test=X_test,
            )
            elapsed = time.time() - t0
            oof_per_seed[si, val_mask, :] = res["oof_pred"]
            test_per_seed_fold[si, f, :, :] = res["test_pred"]
            entry = {
                "seed": seed, "fold": f,
                "best_val_hit": res["best_val_hit"],
                "best_val_loss": res["best_val_loss"],
                "best_epoch": res["best_epoch"],
                "dcm": res["dcm"],
                "elapsed_seconds": elapsed,
            }
            fold_results.append(entry)
            if progress_cb is not None:
                progress_cb(si, f, seed, res, elapsed)

    # Per-seed concat OOF hit (sub-metric)
    per_seed_oof_hit = []
    for si in range(n_seeds):
        oof_si = oof_per_seed[si]
        completed = ~np.isnan(oof_si).any(axis=1)
        err = np.linalg.norm(oof_si[completed] - Y_train[completed], axis=-1)
        per_seed_oof_hit.append(float((err <= 0.01).mean()))

    # Coordinate mean over seeds → final OOF (§5.2 spec step 2)
    oof_pred_all = np.nanmean(oof_per_seed, axis=0).astype(np.float32)
    completed_mask = ~np.isnan(oof_pred_all).any(axis=1)
    overall_oof_hit = float((np.linalg.norm(
        oof_pred_all[completed_mask] - Y_train[completed_mask], axis=-1) <= 0.01).mean())

    # Per-fold sub-metric (seed-mean OOF restricted to that fold)
    fold_oof_hit_per_fold = {}
    for f in range(n_folds):
        m = (fold_of == f) & completed_mask
        if m.sum() > 0:
            err = np.linalg.norm(oof_pred_all[m] - Y_train[m], axis=-1)
            fold_oof_hit_per_fold[int(f)] = float((err <= 0.01).mean())

    # Test ensemble: coord mean over (seed × fold)
    test_pred_all = test_per_seed_fold.reshape(n_seeds * n_folds, N_test, 3).mean(axis=0).astype(np.float32)

    return {
        "overall_oof_hit_1cm": overall_oof_hit,
        "oof_pred_all": oof_pred_all,
        "test_pred": test_pred_all,
        "fold_results": fold_results,
        "per_seed_oof_hit_1cm": per_seed_oof_hit,
        "fold_oof_hit_per_fold": fold_oof_hit_per_fold,
        "config_base": asdict(config_base),
        "seeds": list(seeds),
        "n_seeds": n_seeds,
        "n_folds": n_folds,
    }


def run_multiseed_kfold_v2(
    ids_train: list[str],
    X_train: np.ndarray,
    Y_train: np.ndarray,
    ids_test: list[str],
    X_test: np.ndarray,
    config_base: pp.TrainConfig,
    seeds: list[int],
    feature_flags: dict[str, bool],
    f0_function: pp.Plan014F0Function | None = None,
    progress_cb=None,
) -> dict[str, Any]:
    """plan-015 train_one_fold_v2 variant with feature_flags. Used for plan-016 Path C
    (Feature B/C/D 단독).

    encoder input_dim 은 feature_flags 에 의해 결정 (10D/18D/15D). weight 재초기화 (Kaiming + seed carry).
    OOF aggregation / test ensemble logic 은 run_multiseed_kfold 와 동일.
    """
    if f0_function is None:
        f0_function = pp.Plan014F0Function()

    fold_of = np.array([pp.stable_hash_fold(s) for s in ids_train])
    N_train = X_train.shape[0]
    N_test = X_test.shape[0]
    n_seeds = len(seeds)
    n_folds = pp.N_FOLDS

    oof_per_seed = np.full((n_seeds, N_train, 3), np.nan, dtype=np.float32)
    test_per_seed_fold = np.full((n_seeds, n_folds, N_test, 3), np.nan, dtype=np.float32)
    fold_results = []
    feature_dim_observed = None

    for si, seed in enumerate(seeds):
        cfg_seed = replace(config_base, seed=seed)
        for f in range(n_folds):
            train_mask = fold_of != f
            val_mask = fold_of == f
            t0 = time.time()
            res = p15t.train_one_fold_v2(
                cfg_seed, fold_id=f,
                X_train=X_train[train_mask], Y_train=Y_train[train_mask],
                X_val=X_train[val_mask], Y_val=Y_train[val_mask],
                f0_function=f0_function,
                feature_flags=feature_flags,
                X_test=X_test,
            )
            elapsed = time.time() - t0
            oof_per_seed[si, val_mask, :] = res["oof_pred"]
            test_per_seed_fold[si, f, :, :] = res["test_pred"]
            feature_dim_observed = res["feature_dim"]
            entry = {
                "seed": seed, "fold": f,
                "best_val_hit": res["best_val_hit"],
                "best_val_loss": res["best_val_loss"],
                "best_epoch": res["best_epoch"],
                "dcm": res["dcm"],
                "feature_dim": res["feature_dim"],
                "elapsed_seconds": elapsed,
            }
            fold_results.append(entry)
            if progress_cb is not None:
                progress_cb(si, f, seed, res, elapsed)

    per_seed_oof_hit = []
    for si in range(n_seeds):
        oof_si = oof_per_seed[si]
        completed = ~np.isnan(oof_si).any(axis=1)
        err = np.linalg.norm(oof_si[completed] - Y_train[completed], axis=-1)
        per_seed_oof_hit.append(float((err <= 0.01).mean()))

    oof_pred_all = np.nanmean(oof_per_seed, axis=0).astype(np.float32)
    completed_mask = ~np.isnan(oof_pred_all).any(axis=1)
    overall_oof_hit = float((np.linalg.norm(
        oof_pred_all[completed_mask] - Y_train[completed_mask], axis=-1) <= 0.01).mean())

    fold_oof_hit_per_fold = {}
    for f in range(n_folds):
        m = (fold_of == f) & completed_mask
        if m.sum() > 0:
            err = np.linalg.norm(oof_pred_all[m] - Y_train[m], axis=-1)
            fold_oof_hit_per_fold[int(f)] = float((err <= 0.01).mean())

    test_pred_all = test_per_seed_fold.reshape(n_seeds * n_folds, N_test, 3).mean(axis=0).astype(np.float32)

    return {
        "overall_oof_hit_1cm": overall_oof_hit,
        "oof_pred_all": oof_pred_all,
        "test_pred": test_pred_all,
        "fold_results": fold_results,
        "per_seed_oof_hit_1cm": per_seed_oof_hit,
        "fold_oof_hit_per_fold": fold_oof_hit_per_fold,
        "config_base": asdict(config_base),
        "seeds": list(seeds),
        "n_seeds": n_seeds,
        "n_folds": n_folds,
        "feature_flags": feature_flags,
        "feature_dim": feature_dim_observed,
    }
