"""plan-025 c4 — 5-fold OOF runner for C1 / C2 cells.

K=14 BCC + τ_cls=0.001 fix (plan-022 winner cell carry).
input = 1080D row-expanded (build_feat_1080).

CLI:
    python analysis/plan-025/run_oof.py --cell {C1,C2,G1} [--seed 20260522]

decision-note carriers (spec §0.5):
- 5-fold split = stable_fold_id (plan-020/021/022/023/024 carry, MD5 deterministic)
- C1 hparam = plan-022 LgbmSelectorOnly default (n_est=500, lr=0.05, num_leaves=63)
- C2 hparam = C1 + (n_est=2000, lr=0.03, feature_fraction=0.7, min_data_in_leaf=50,
             early_stopping_rounds=100, inner-val 20% sample-stratified seed=20260522)
- LGBM random_state = 20260522 (본 plan, plan-022 reproduce 와 분리 layer)
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

# ── importlib carry ────────────────────────────────────────────────
_THIS = Path(__file__).resolve().parent              # analysis/plan-025/
_REPO = _THIS.parent.parent
_PLAN020 = _THIS.parent / "plan-020"
_PLAN021 = _THIS.parent / "plan-021"
_PLAN022 = _THIS.parent / "plan-022"
_PLAN024 = _THIS.parent / "plan-024"

for p in (_REPO, _PLAN021, _PLAN022, _PLAN024):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


_bf = _load(_PLAN020 / "baseline_f0.py", "p025_oof_bf")
_p021_build = _load(_PLAN021 / "build_input.py", "p025_oof_p021_build")
_p022_anchors = _load(_PLAN022 / "anchors.py", "p025_oof_p022_anchors")
_som = _load(_PLAN022 / "selector_only_model.py", "p025_oof_som")
_quantile_mod = _load(_PLAN024 / "quantile_carry.py", "p025_oof_qc")

# build_feat_1080 — same dir, importlib
_p025_build = _load(_THIS / "build_feat_1080.py", "p025_build_feat_1080")
build_feat_1080 = _p025_build.build_feat_1080

from src.io import load_all_samples, load_labels  # noqa: E402
from src.pb_0_6822.selector import stable_fold_id  # noqa: E402


# ── LgbmSelectorRowExpanded — spec §4.4 선택 B subclass ───────────
# plan-022 LgbmSelectorOnly.fit 는 sample-level X (N, D) 전제 + 내부 row-expand.
# 본 plan 의 build_feat_1080 는 per-anchor feature 포함 → 이미 (N*K, D) row-expanded.
# 따라서 fit override 필요. plan-022 의 sample-weight expansion + missing class
# dummy inject 로직만 carry, 외부 np.repeat 단계 skip.
class LgbmSelectorRowExpanded(_som.LgbmSelectorOnly):
    """fit(X_expanded (N*K, D), q (N, K)) — X 자체가 이미 row-expanded."""

    def fit(self, X_expanded, q, eval_set=None, early_stopping_rounds=None):
        N = q.shape[0]
        K = self.K
        assert q.shape == (N, K), f"q shape {q.shape} != ({N}, {K})"
        assert X_expanded.shape[0] == N * K, f"X_expanded shape {X_expanded.shape} != ({N*K}, *)"

        y_expanded = np.tile(np.arange(K), N)
        sample_weight = q.flatten()
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

        fit_kwargs = {"sample_weight": w_fit}
        if eval_set is not None and early_stopping_rounds is not None:
            # LightGBM API: eval_set 의 y 도 1D class index 형태 필요 → soft label argmax 사용
            es_kwargs = []
            for X_es, q_es in eval_set:
                N_es = q_es.shape[0]
                # eval set 도 동일 row-expand + sample-weight 방식 적용
                y_es = np.tile(np.arange(K), N_es)
                w_es = q_es.flatten()
                m_es = w_es > 1e-6
                es_kwargs.append((X_es[m_es], y_es[m_es]))
            try:
                from lightgbm import early_stopping
                fit_kwargs["eval_set"] = es_kwargs
                fit_kwargs["callbacks"] = [early_stopping(stopping_rounds=early_stopping_rounds, verbose=False)]
            except ImportError:
                pass

        self.clf.fit(X_fit, y_fit, **fit_kwargs)
        return self


# ── cell configs (spec §3.5) ──────────────────────────────────────
CELL_CONFIGS: dict[str, dict] = {
    "C1": {
        # plan-022 default carry — LgbmSelectorOnly 생성자 그대로
        "hparam_override": {},
        "use_early_stopping": False,
    },
    "C2": {
        # 5 hparam 동시 adjust (spec §3.5)
        "hparam_override": {
            "n_estimators": 2000,
            "learning_rate": 0.03,
            "colsample_bytree": 0.7,
            "min_child_samples": 50,
        },
        "use_early_stopping": True,
        "early_stopping_rounds": 100,
        "inner_val_size": 0.20,
        "inner_val_seed": 20260522,
    },
}

LGBM_RANDOM_STATE = 20260522            # 본 plan seed (G1 reproduce 의 plan-022 seed=20260519 와 분리)
TAU_CLS = 0.001                          # plan-022 winner
K_ANCHORS = 14
N_FOLDS = 5


# ── helpers ───────────────────────────────────────────────────────
def _normalize_p022_result(d: dict) -> dict:
    """plan-022 run_oof_cell result 의 key 를 본 plan canonical 표기로 변환 (spec §4.2)."""
    mapping = {
        "hit_1.5cm": "hit_1p5cm",
        "hit_15mm": "hit_1p5cm",
        "hit_at_1cm": "hit_1cm",
        "hit_at_1.5cm": "hit_1p5cm",
    }
    out = {}
    for k, v in d.items():
        out[mapping.get(k, k)] = v
    return out


def _row_expand_idx(sample_idx: np.ndarray, K: int) -> np.ndarray:
    """sample index → row index 확장 (sample-major)."""
    return (sample_idx[:, None] * K + np.arange(K)[None, :]).ravel()


def _stratified_inner_split(
    q_train: np.ndarray,
    test_size: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """sample 단위 stratified split (q.argmax 위 stratify)."""
    from sklearn.model_selection import train_test_split
    N = q_train.shape[0]
    idx_all = np.arange(N)
    labels = q_train.argmax(axis=1)
    idx_tr, idx_val = train_test_split(
        idx_all, test_size=test_size, stratify=labels, random_state=seed,
    )
    return idx_tr, idx_val


# ── G1 reproduce (F0 + plan-022 winner) ────────────────────────────
def run_g1_reproduce(verbose: bool = True) -> dict:
    """G1: F0 baseline + plan-022 A6_bcc14_tau001 reproduce."""
    _ids, X = load_all_samples()
    _ids_gt, gt = load_labels()
    assert _ids == _ids_gt, "id mismatch between samples and labels"

    # G1 a: F0 baseline 5-fold concat OOF
    F0_pred = _bf.f0_baseline(X, end_idx=10)
    err_F0 = np.linalg.norm(F0_pred - gt, axis=1)
    hit_1cm_F0 = float((err_F0 <= 0.01).mean())
    hit_1p5cm_F0 = float((err_F0 <= 0.015).mean())
    if verbose:
        print(f"[G1 a] F0 hit_1cm={hit_1cm_F0:.4f} hit_1p5cm={hit_1p5cm_F0:.4f}")
    assert 0.6315 <= hit_1cm_F0 <= 0.6325, f"F0 hit_1cm drift: {hit_1cm_F0}"
    assert 0.8028 <= hit_1p5cm_F0 <= 0.8038, f"F0 hit_1p5cm drift: {hit_1p5cm_F0}"

    # G1 b: plan-022 winner A6_bcc14 + τ=0.001 reproduce
    # plan-022 run_oof_cell signature 확인 후 호출
    _run_oof_p022 = _load(_PLAN022 / "run_oof.py", "p025_oof_p022_run")
    folds = np.asarray([stable_fold_id(str(sid), N_FOLDS) for sid in _ids], dtype=int)
    if verbose:
        print(f"[G1 b] running plan-022 run_oof_cell (5-fold LGBM)...")
    result_p022 = _run_oof_p022.run_oof_cell(
        X=X.astype(np.float32),
        Y=gt.astype(np.float32),
        anchors=_p022_anchors.ANCHORS_A6,
        folds=folds,
        tau_cls=0.001,
        verbose=verbose,
    )
    result_p022 = _normalize_p022_result(result_p022)
    hit_1cm_p022 = float(result_p022["hit_1cm"])
    hit_1p5cm_p022 = float(result_p022["hit_1p5cm"])
    if verbose:
        print(f"[G1 b] plan-022 reproduce hit_1cm={hit_1cm_p022:.4f} hit_1p5cm={hit_1p5cm_p022:.4f}")
    assert 0.6523 <= hit_1cm_p022 <= 0.6533, f"plan-022 hit_1cm drift: {hit_1cm_p022}"
    assert 0.8099 <= hit_1p5cm_p022 <= 0.8109, f"plan-022 hit_1p5cm drift: {hit_1p5cm_p022}"

    return {
        "F0": {"hit_1cm": hit_1cm_F0, "hit_1p5cm": hit_1p5cm_F0},
        "plan022_winner": {
            "hit_1cm": hit_1cm_p022,
            "hit_1p5cm": hit_1p5cm_p022,
            "raw": result_p022,
        },
    }


# ── G2 main runner ───────────────────────────────────────────────
def run_oof_plan025(
    cell_id: str,
    n_folds: int = N_FOLDS,
    seed: int = LGBM_RANDOM_STATE,
    verbose: bool = True,
) -> dict:
    """5-fold OOF for cell C1 or C2 (spec §6.2)."""
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
    oof_best_iter: list[int | None] = [None] * n_folds
    per_fold: list[dict] = []

    for fold in range(n_folds):
        t_fold = time.time()
        train_idx = np.where(folds != fold)[0]
        test_idx = np.where(folds == fold)[0]
        N_tr, N_te = len(train_idx), len(test_idx)

        X_train, X_test = X[train_idx], X[test_idx]
        gt_train, gt_test = gt[train_idx], gt[test_idx]

        # Frenet basis + F0 (per fold)
        R_wfn_train = _p021_build.build_frenet_basis_3d(X_train, end_idx=10)
        R_wfn_test = _p021_build.build_frenet_basis_3d(X_test, end_idx=10)
        F0_train = _bf.f0_baseline(X_train, end_idx=10).astype(np.float32)
        F0_test = _bf.f0_baseline(X_test, end_idx=10).astype(np.float32)

        # Quantile carry (train fold only — fold-leakage 차단)
        qc = _quantile_mod.build(X_train, R_wfn_train)

        # Build feature 1080D (per fold)
        feat_train = build_feat_1080(X_train, anchors, _bf.f0_baseline, qc)
        feat_test = build_feat_1080(X_test, anchors, _bf.f0_baseline, qc)

        # Soft label (plan-022 carry)
        q_train = _som.build_soft_label_with_tau(
            gt_train, R_wfn_train, F0_train, anchors, TAU_CLS,
        )                                              # (N_tr, K)

        # Model (LgbmSelectorRowExpanded — 본 plan 의 (N*K, D) row-expanded X 처리)
        model = LgbmSelectorRowExpanded(K=K_ANCHORS)
        try:
            model.clf.set_params(random_state=seed)
        except Exception:
            pass

        if cell_id == "C2":
            model.clf.set_params(**cfg["hparam_override"])

        if cfg["use_early_stopping"]:
            idx_tr2, idx_val = _stratified_inner_split(
                q_train, test_size=cfg["inner_val_size"], seed=cfg["inner_val_seed"],
            )
            row_idx_tr2 = _row_expand_idx(idx_tr2, K_ANCHORS)
            row_idx_val = _row_expand_idx(idx_val, K_ANCHORS)
            X_tr2 = feat_train[row_idx_tr2]
            X_val = feat_train[row_idx_val]
            q_tr2 = q_train[idx_tr2]
            q_val = q_train[idx_val]
            try:
                model.fit(X_tr2, q_tr2, eval_set=[(X_val, q_val)],
                          early_stopping_rounds=cfg["early_stopping_rounds"])
                best_it = getattr(model.clf, "best_iteration_", None)
                oof_best_iter[fold] = int(best_it) if best_it else None
            except (TypeError, ValueError, AttributeError) as e:
                warnings.warn(
                    f"C2 early_stopping fallback to default fit ({type(e).__name__}: {e}) "
                    f"— decision-note: early_stop_fallback"
                )
                model.fit(feat_train, q_train)
                oof_best_iter[fold] = None
        else:
            model.fit(feat_train, q_train)

        # Predict — row-expand selector
        probs_test_expanded = model.clf.predict_proba(feat_test)    # (N_te*K, K)
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
            "top1_acc": float((probs_sel.argmax(axis=1) == probs_sel.argmax(axis=1)).mean()),  # placeholder
            "max_class_ratio_fold": float(probs_sel.mean(axis=0).max()),
            "runtime_s_fold": runtime_fold,
            "best_iteration": oof_best_iter[fold],
        })
        if verbose:
            print(f"[{cell_id}] fold {fold} hit_1cm={hit_1cm_fold:.4f} hit_1p5cm={hit_1p5cm_fold:.4f} runtime={runtime_fold:.1f}s")

    # 5-fold concat OOF metric
    err = np.linalg.norm(oof_pred - gt, axis=1)
    hit_1cm = float((err <= 0.01).mean())
    hit_1p5cm = float((err <= 0.015).mean())
    max_class_ratio = float(oof_probs_sel.mean(axis=0).max())

    # gt_anchor_label 계산 (= argmin_k ‖a_k - residual_true_frenet‖)
    # for top1_acc 산식
    R_wfn_all = _p021_build.build_frenet_basis_3d(X, end_idx=10)
    F0_all = _bf.f0_baseline(X, end_idx=10).astype(np.float32)
    residual_world_true = gt - F0_all
    residual_frenet_true = np.einsum(
        "nij,nj->ni",
        np.transpose(R_wfn_all, (0, 2, 1)).astype(np.float32),
        residual_world_true,
    )
    diff = anchors[None, :, :] - residual_frenet_true[:, None, :]   # (N, K, 3)
    gt_anchor_label = np.linalg.norm(diff, axis=2).argmin(axis=1)   # (N,)
    top1_acc = float((oof_probs_sel.argmax(axis=1) == gt_anchor_label).mean())

    runtime_s = time.time() - t0
    if verbose:
        print(f"[{cell_id}] FINAL hit_1cm={hit_1cm:.4f} hit_1p5cm={hit_1p5cm:.4f} "
              f"max_class_ratio={max_class_ratio:.3f} top1_acc={top1_acc:.4f} "
              f"runtime={runtime_s:.1f}s")

    return {
        "cell_id": cell_id,
        "hit_1cm": hit_1cm,
        "hit_1p5cm": hit_1p5cm,
        "top1_acc": top1_acc,
        "max_class_ratio": max_class_ratio,
        "per_fold": per_fold,
        "runtime_s": runtime_s,
        "best_iteration_per_fold": oof_best_iter if cell_id == "C2" else None,
        "seed": seed,
        "n_folds": n_folds,
        "tau_cls": TAU_CLS,
        "K": K_ANCHORS,
    }


# ── CLI ────────────────────────────────────────────────────────────
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cell", required=True, choices=["G1", "C1", "C2"])
    p.add_argument("--seed", type=int, default=LGBM_RANDOM_STATE)
    p.add_argument("--out", type=str, default=None, help="JSON output path")
    args = p.parse_args()

    if args.cell == "G1":
        result = run_g1_reproduce(verbose=True)
        out_path = args.out or str(_THIS / "baseline_carry.json")
    else:
        result = run_oof_plan025(args.cell, seed=args.seed, verbose=True)
        out_path = args.out or str(_THIS / f"results_{args.cell}.json")

    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=lambda x: int(x) if isinstance(x, np.integer) else float(x))
    print(f"saved → {out_path}")


if __name__ == "__main__":
    main()
