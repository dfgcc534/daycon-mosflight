"""plan-028 c4 — 9 cell (G2.A) OOF runner — plan-025 carry wrapper.

§0.5 c4 박제: plan-025 carry runner 위에 (a) input slice fn 주입 + (b) sample_weight
flag (on/off) + (c) τ_cls value + (d) model wrapping (subclass vs base) + (e) block ④
산식 (8-stat vs raw) 5 축 모두 wrapper 내부에서 cell 별 inject.

plan-025 file 자체는 read-only 유지 (c2 cherry-pick 박제 정합) — 본 wrapper 가 fit
kwargs / soft label / slice / model class 만 patch.

CLI:
    python analysis/plan-028/run_oof_subset.py --cell {B1, B2, B3, B4, W1, T1, T2, S1, R1}
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
import warnings
from pathlib import Path
from typing import Callable

import numpy as np

# ── importlib carry (plan-025 와 동일 패턴) ──────────────────────────────
_THIS = Path(__file__).resolve().parent              # analysis/plan-028/
_REPO = _THIS.parent.parent                           # repo root
_PLAN020 = _THIS.parent / "plan-020"
_PLAN021 = _THIS.parent / "plan-021"
_PLAN022 = _THIS.parent / "plan-022"
_PLAN024 = _THIS.parent / "plan-024"
_PLAN025 = _THIS.parent / "plan-025"

for p in (_REPO, _PLAN021, _PLAN024):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


_bf = _load(_PLAN020 / "baseline_f0.py", "p028_bf")
_p021_build = _load(_PLAN021 / "build_input.py", "p028_p021_build")
_p022_anchors = _load(_PLAN022 / "anchors.py", "p028_p022_anchors")
_som = _load(_PLAN022 / "selector_only_model.py", "p028_som")
_quantile_mod = _load(_PLAN024 / "quantile_carry.py", "p028_qc")
_p025_run = _load(_PLAN025 / "run_oof.py", "p028_p025_run")
_subset = _load(_THIS / "build_feat_subset.py", "p028_subset")

from src.io import load_all_samples, load_labels  # noqa: E402
from src.pb_0_6822.selector import stable_fold_id  # noqa: E402


# ── constants (plan-025 carry) ───────────────────────────────────────────
LGBM_RANDOM_STATE = 20260522
TAU_BASELINE = 0.001
K_ANCHORS = 14
N_FOLDS = 5


# ── cell configs (§4.3 9 cell) ───────────────────────────────────────────
CELL_CONFIGS: dict[str, dict] = {
    "B1": {"slice": "B1", "weight": "ON", "tau": 0.001, "model": "subclass", "hypothesis": "d"},
    "B2": {"slice": "B2", "weight": "ON", "tau": 0.001, "model": "subclass", "hypothesis": "d"},
    "B3": {"slice": "B3", "weight": "ON", "tau": 0.001, "model": "subclass", "hypothesis": "d"},
    "B4": {"slice": "B4", "weight": "ON", "tau": 0.001, "model": "subclass", "hypothesis": "baseline"},
    "W1": {"slice": "B4", "weight": "OFF", "tau": 0.001, "model": "subclass", "hypothesis": "b"},
    "T1": {"slice": "B4", "weight": "ON", "tau": 0.01,  "model": "subclass", "hypothesis": "a"},
    "T2": {"slice": "B4", "weight": "ON", "tau": 0.1,   "model": "subclass", "hypothesis": "a"},
    "S1": {"slice": "B4", "weight": "ON", "tau": 0.001, "model": "base",     "hypothesis": "c"},
    "R1": {"slice": "R1", "weight": "ON", "tau": 0.001, "model": "subclass", "hypothesis": "e"},
}

SLICE_FNS: dict[str, Callable] = {
    "B1": _subset.slice_B1_anchor22,
    "B2": _subset.slice_B2_combo192,
    "B3": _subset.slice_B3_no_anchor1058,
    "B4": _subset.slice_B4_full1080,
}


# ── LgbmSelector with weight flag (§4.3 OFF 산식) ────────────────────────
class LgbmSelectorRowExpandedWeightFlag(_p025_run.LgbmSelectorRowExpanded):
    """fit(X_expanded, q, weighted=True): weighted=False → sample_weight=1.0 균등.

    W1 cell 의 (b) 가설 single-variable isolation — weight 값만 ON/OFF, 다른 산식
    (row-expand reshape / label / objective / num_class) 모두 plan-025 carry 동일.
    """

    def fit(self, X_expanded, q, weighted: bool = True, eval_set=None, early_stopping_rounds=None):
        N, K = q.shape
        assert K == self.K, f"K={K} != self.K={self.K}"
        assert X_expanded.shape[0] == N * K

        y_expanded = np.tile(np.arange(K), N)
        if weighted:
            sample_weight = q.flatten().astype(np.float32)
        else:
            # W1: sample_weight=1.0 균등, row-expand reshape 자체는 ON 동일
            sample_weight = np.ones(N * K, dtype=np.float32)

        mask = sample_weight > 1e-6

        present_classes = set(y_expanded[mask].tolist())
        missing = [k for k in range(K) if k not in present_classes]
        if missing:
            X_dummy = np.zeros((len(missing), X_expanded.shape[1]), dtype=X_expanded.dtype)
            y_dummy = np.array(missing, dtype=np.int64)
            w_dummy = np.full(len(missing), 1e-6, dtype=sample_weight.dtype)
            X_fit = np.concatenate([X_expanded[mask], X_dummy], axis=0)
            y_fit = np.concatenate([y_expanded[mask], y_dummy], axis=0)
            w_fit = np.concatenate([sample_weight[mask], w_dummy], axis=0)
        else:
            X_fit = X_expanded[mask]
            y_fit = y_expanded[mask]
            w_fit = sample_weight[mask]

        self.clf.fit(X_fit, y_fit, sample_weight=w_fit)
        return self


# ── S1: base LGBMClassifier multiclass 직접 사용 (가설 c) ────────────────
class BaseLgbmMulticlass:
    """plan-022 LgbmSelectorOnly subclass 우회 — base lightgbm.LGBMClassifier 직접 사용.

    row-expand reshape + sample_weight = soft_label-weighted + label = anchor_idx 산식은
    본 class 내부에서 명시적 처리 (subclass wrapper 의 의도적 우회).
    """

    def __init__(self, K: int = 14, random_state: int = LGBM_RANDOM_STATE):
        import lightgbm as lgb
        self.K = K
        self.clf = lgb.LGBMClassifier(
            objective="multiclass",
            num_class=K,
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=63,
            random_state=random_state,
            verbose=-1,
        )

    def fit(self, X_expanded, q, weighted: bool = True):
        N, K = q.shape
        assert K == self.K
        assert X_expanded.shape[0] == N * K

        y_expanded = np.tile(np.arange(K), N)
        if weighted:
            sample_weight = q.flatten().astype(np.float32)
        else:
            sample_weight = np.ones(N * K, dtype=np.float32)

        mask = sample_weight > 1e-6

        present_classes = set(y_expanded[mask].tolist())
        missing = [k for k in range(K) if k not in present_classes]
        if missing:
            X_dummy = np.zeros((len(missing), X_expanded.shape[1]), dtype=X_expanded.dtype)
            y_dummy = np.array(missing, dtype=np.int64)
            w_dummy = np.full(len(missing), 1e-6, dtype=sample_weight.dtype)
            X_fit = np.concatenate([X_expanded[mask], X_dummy], axis=0)
            y_fit = np.concatenate([y_expanded[mask], y_dummy], axis=0)
            w_fit = np.concatenate([sample_weight[mask], w_dummy], axis=0)
        else:
            X_fit = X_expanded[mask]
            y_fit = y_expanded[mask]
            w_fit = sample_weight[mask]

        self.clf.fit(X_fit, y_fit, sample_weight=w_fit)
        return self

    def predict_proba(self, X_expanded):
        return self.clf.predict_proba(X_expanded)


# ── runner ───────────────────────────────────────────────────────────────
def run_oof_cell_subset(
    cell_id: str,
    n_folds: int = N_FOLDS,
    seed: int = LGBM_RANDOM_STATE,
    verbose: bool = True,
) -> dict:
    """5-fold OOF for plan-028 G2.A 9 cell."""
    assert cell_id in CELL_CONFIGS, f"unsupported cell: {cell_id}"
    cfg = CELL_CONFIGS[cell_id]

    t0 = time.time()
    _ids, X = load_all_samples()
    _ids_gt, gt = load_labels()
    assert _ids == _ids_gt, "id mismatch between samples and labels"
    gt = gt.astype(np.float32)
    X = X.astype(np.float32)
    N = X.shape[0]
    anchors = _p022_anchors.ANCHORS_A6
    folds = np.asarray([stable_fold_id(str(sid), N_FOLDS) for sid in _ids], dtype=int)

    oof_pred = np.zeros((N, 3), dtype=np.float32)
    oof_probs_sel = np.zeros((N, K_ANCHORS), dtype=np.float32)
    per_fold: list[dict] = []

    if verbose:
        print(f"[{cell_id}] hypothesis={cfg['hypothesis']} slice={cfg['slice']} "
              f"weight={cfg['weight']} tau={cfg['tau']} model={cfg['model']}")

    for fold in range(n_folds):
        t_fold = time.time()
        train_idx = np.where(folds != fold)[0]
        test_idx = np.where(folds == fold)[0]
        N_tr, N_te = len(train_idx), len(test_idx)

        X_train, X_test = X[train_idx], X[test_idx]
        gt_train, gt_test = gt[train_idx], gt[test_idx]

        # Frenet basis + F0
        R_wfn_train = _p021_build.build_frenet_basis_3d(X_train, end_idx=10)
        R_wfn_test = _p021_build.build_frenet_basis_3d(X_test, end_idx=10)
        F0_train = _bf.f0_baseline(X_train, end_idx=10).astype(np.float32)
        F0_test = _bf.f0_baseline(X_test, end_idx=10).astype(np.float32)

        # Quantile carry (train fold only)
        qc = _quantile_mod.build(X_train, R_wfn_train)

        # Build feature 1080D + (R1 위해) seq_raw 도 동시 export
        if cfg["slice"] == "R1":
            feat_train_1080, seq_train = _subset.build_feat_1080_with_raw_seq(
                X_train, anchors, _bf.f0_baseline, qc,
            )
            feat_test_1080, seq_test = _subset.build_feat_1080_with_raw_seq(
                X_test, anchors, _bf.f0_baseline, qc,
            )
            feat_train = _subset.build_R1_seq_raw(feat_train_1080, seq_train)
            feat_test = _subset.build_R1_seq_raw(feat_test_1080, seq_test)
        else:
            feat_train_1080 = _p025_run.build_feat_1080(
                X_train, anchors, _bf.f0_baseline, qc,
            )
            feat_test_1080 = _p025_run.build_feat_1080(
                X_test, anchors, _bf.f0_baseline, qc,
            )
            slice_fn = SLICE_FNS[cfg["slice"]]
            feat_train = slice_fn(feat_train_1080)
            feat_test = slice_fn(feat_test_1080)

        # Soft label (cell.tau)
        q_train = _som.build_soft_label_with_tau(
            gt_train, R_wfn_train, F0_train, anchors, cfg["tau"],
        )

        # Model dispatch
        if cfg["model"] == "subclass":
            model = LgbmSelectorRowExpandedWeightFlag(K=K_ANCHORS)
            try:
                model.clf.set_params(random_state=seed)
            except Exception:
                pass
            weighted = (cfg["weight"] == "ON")
            model.fit(feat_train, q_train, weighted=weighted)
        elif cfg["model"] == "base":
            model = BaseLgbmMulticlass(K=K_ANCHORS, random_state=seed)
            weighted = (cfg["weight"] == "ON")
            model.fit(feat_train, q_train, weighted=weighted)
        else:
            raise ValueError(f"unknown model: {cfg['model']}")

        # Predict — row-expand selector
        probs_test_expanded = model.clf.predict_proba(feat_test)
        anchor_idx = np.tile(np.arange(K_ANCHORS), N_te)
        probs_sel = probs_test_expanded[np.arange(N_te * K_ANCHORS), anchor_idx].reshape(N_te, K_ANCHORS)
        probs_sel = probs_sel / probs_sel.sum(axis=1, keepdims=True)

        # Frenet → world
        residual_frenet = (probs_sel[:, :, None] * anchors[None, :, :]).sum(axis=1)
        residual_world = np.einsum("nij,nj->ni", R_wfn_test, residual_frenet)
        final_pred = F0_test + residual_world

        oof_pred[test_idx] = final_pred
        oof_probs_sel[test_idx] = probs_sel

        # Per-fold metric
        err = np.linalg.norm(final_pred - gt_test, axis=1)
        hit_1cm_fold = float((err <= 0.01).mean())
        hit_1p5cm_fold = float((err <= 0.015).mean())
        runtime_fold = time.time() - t_fold
        per_fold.append({
            "fold": fold,
            "n_test": int(N_te),
            "hit_1cm": hit_1cm_fold,
            "hit_1p5cm": hit_1p5cm_fold,
            "max_class_ratio_fold": float(probs_sel.mean(axis=0).max()),
            "runtime_s_fold": runtime_fold,
        })
        if verbose:
            print(f"  fold {fold} hit_1cm={hit_1cm_fold:.4f} runtime={runtime_fold:.1f}s")

    # 5-fold concat
    err = np.linalg.norm(oof_pred - gt, axis=1)
    hit_1cm = float((err <= 0.01).mean())
    hit_1p5cm = float((err <= 0.015).mean())
    max_class_ratio = float(oof_probs_sel.mean(axis=0).max())

    # top1_acc (= argmax(probs) vs gt_anchor_label)
    R_wfn_all = _p021_build.build_frenet_basis_3d(X, end_idx=10)
    F0_all = _bf.f0_baseline(X, end_idx=10).astype(np.float32)
    residual_world_true = gt - F0_all
    residual_frenet_true = np.einsum(
        "nij,nj->ni",
        np.transpose(R_wfn_all, (0, 2, 1)).astype(np.float32),
        residual_world_true,
    )
    diff = anchors[None, :, :] - residual_frenet_true[:, None, :]
    gt_anchor_label = np.linalg.norm(diff, axis=2).argmin(axis=1)
    top1_acc = float((oof_probs_sel.argmax(axis=1) == gt_anchor_label).mean())

    runtime_s = time.time() - t0
    if verbose:
        print(f"[{cell_id}] FINAL hit_1cm={hit_1cm:.4f} hit_1p5cm={hit_1p5cm:.4f} "
              f"max_class_ratio={max_class_ratio:.4f} top1_acc={top1_acc:.4f} "
              f"runtime={runtime_s:.1f}s")

    return {
        "cell_id": cell_id,
        "hypothesis": cfg["hypothesis"],
        "config": cfg,
        "hit_1cm": hit_1cm,
        "hit_1p5cm": hit_1p5cm,
        "top1_acc": top1_acc,
        "max_class_ratio": max_class_ratio,
        "per_fold": per_fold,
        "runtime_s": runtime_s,
        "seed": seed,
        "n_folds": n_folds,
        "K": K_ANCHORS,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cell", type=str, required=True, choices=list(CELL_CONFIGS.keys()))
    p.add_argument("--out", type=str, default=None,
                   help="JSON output path (default: analysis/plan-028/results_{cell}.json)")
    p.add_argument("--seed", type=int, default=LGBM_RANDOM_STATE)
    args = p.parse_args()

    result = run_oof_cell_subset(args.cell, seed=args.seed)

    out_path = Path(args.out) if args.out else _THIS / f"results_{args.cell}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"[{args.cell}] result saved to {out_path}")


if __name__ == "__main__":
    main()
