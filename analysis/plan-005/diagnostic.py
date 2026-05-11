"""plan-005 diagnostic — PB_0.6822 framework ceiling/gap audit.

STAGE 0 (c2/c3) infrastructure + plan-004 artifact loader. STAGE 1~5 (c4~c8) follow.

CLI:
    python -m analysis.plan_005.diagnostic verify          # G0 entry — load 4 artifacts
    python -m analysis.plan_005.diagnostic rerun-corrector # c3 — corrector full-fit + corrected_*.npz
    python -m analysis.plan_005.diagnostic stage1          # c4 — oracle 4-tier
    python -m analysis.plan_005.diagnostic stage2          # c5 — selector decomp
    python -m analysis.plan_005.diagnostic stage3          # c6 — corrector decomp
    python -m analysis.plan_005.diagnostic stage4a         # c7a — Variant A retrain
    python -m analysis.plan_005.diagnostic stage4b         # c7b — Variant B + 3-way
    python -m analysis.plan_005.diagnostic stage5          # c8 — failure + B001
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from src.pb_0_6822 import boundary, selector

REPO = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO / "data"
PLAN004_DIR = REPO / "runs/baseline/P001_pb-0-6822-fullrun"
PLAN004_REGIME_JSON = REPO / "analysis/plan-004/regime_distribution.json"
ANALYSIS_DIR = REPO / "analysis/plan-005"
DEVICE = "cuda:1"  # plan-004 일관성 (server agent GPU 1)

R_HIT = selector.R_HIT
EPS = selector.EPS
N_REGIMES = 18
N_CANDIDATES = 27
SEED_BOUNDARY = 20260606  # plan-004 yaml seed_boundary

REQUIRED_PLAN004_ARTIFACTS = (
    PLAN004_DIR / "oof_selector_scores.npz",
    PLAN004_DIR / "test_selector_scores.npz",
    PLAN004_DIR / "boundary_tiny_correction_report.json",
    PLAN004_REGIME_JSON,
)


# ─────────────────────────────────────────────────────────────────────────────
# G0 entry — plan-004 artifact verification
# ─────────────────────────────────────────────────────────────────────────────


def verify_plan004_artifacts() -> dict:
    """G0 entry — plan-004 의 4 산출이 모두 로드 가능한지 검증.

    Raises AssertionError if any required artifact is missing → severe `plan004_artifacts_missing`.
    Returns a dict with loaded summaries (keys: oof_scores shape, test_scores shape,
    boundary_report keys, regime_distribution keys).
    """
    for p in REQUIRED_PLAN004_ARTIFACTS:
        assert p.exists(), f"plan-004 산출 부재: {p}"

    oof = np.load(REQUIRED_PLAN004_ARTIFACTS[0], allow_pickle=True)
    test = np.load(REQUIRED_PLAN004_ARTIFACTS[1], allow_pickle=True)
    rep = json.loads(REQUIRED_PLAN004_ARTIFACTS[2].read_text())
    regime_dist = json.loads(REQUIRED_PLAN004_ARTIFACTS[3].read_text())

    # smoke shape check (codified expectations from plan-004 lock-in)
    assert oof["ens_scores"].shape[1] == N_CANDIDATES, oof["ens_scores"].shape
    assert oof["cands"].shape[1] == N_CANDIDATES, oof["cands"].shape
    assert oof["cands"].shape[2] == 3, f"expected 3-D coords, got {oof['cands'].shape}"
    assert test["ens_scores"].shape[1] == N_CANDIDATES, test["ens_scores"].shape
    assert "regime_histogram" in regime_dist
    assert len(regime_dist["regime_histogram"]) == N_REGIMES
    assert sum(regime_dist["regime_histogram"]) == int(regime_dist["n_total"])

    return {
        "oof_shape": list(oof["ens_scores"].shape),
        "test_shape": list(test["ens_scores"].shape),
        "boundary_report_keys": sorted(rep.keys()),
        "regime_n_total": int(regime_dist["n_total"]),
        "regime_histogram": regime_dist["regime_histogram"],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Train / test data loaders (plan-004 와 동일 경로)
# ─────────────────────────────────────────────────────────────────────────────


def load_train_data() -> tuple[list[str], np.ndarray, np.ndarray]:
    """plan-004 와 동일 경로로 train ids / x / y 로드.

    Returns:
        ids: list[str] (length N_train)
        train_x: np.ndarray (N_train, T, 3) float32
        train_y: np.ndarray (N_train, 3) float32 — same coord space as cands / corrected
    """
    ids, train_y = selector.read_labels(DATA_ROOT / "train_labels.csv")
    train_x = selector.load_stack(DATA_ROOT / "train", ids)
    return ids, train_x, train_y


def load_test_data() -> tuple[list[str], np.ndarray]:
    test_ids = selector.read_submission_ids(DATA_ROOT / "sample_submission.csv")
    test_x = selector.load_stack(DATA_ROOT / "test", test_ids)
    return test_ids, test_x


# ─────────────────────────────────────────────────────────────────────────────
# §3.1 Shared input manifest loader
# ─────────────────────────────────────────────────────────────────────────────


def _load_shared_inputs(*, recompute_y_from_csv: bool = False) -> dict:
    """플랜 §3.1 공유 입력 변수 manifest — 한 곳에서 로드.

    decision-note (spec-default): plan §3.1 의 *형식적* 차이를 다음과 같이 해소:

      1. **3-D coords** — plan §3.1 은 train_y / cands 를 (N, 2) 로 명시하나 실제 코드 경로는 (N, 3)
         (xyz; submission CSV 도 id,x,y,z). 본 모듈은 (N, 3) / (N, 27, 3) 그대로 사용. R_HIT 비교는
         L2 norm over last axis 라 차원 무관.

      2. **`oof_scores` semantic** — plan §3.1 은 `ens_scores` 를 *pre-bias* (model logit only) 로
         규정하나 실제 selector.py L1752 는 `scores = predict_scores(...) + val_bias` 로 저장.
         즉 `ens_scores` 는 *post-bias*, `ens_prior` 가 *bias-only* 항. 본 모듈은:
            - `final_scores` (§6) = `ens_scores` 직접 사용 (이중 가산 방지)
            - Variant B 의 score_B (§8.3 "no gru") = `ens_prior` (bias-only)
            - Variant A retrain 의 score_A 도 동일하게 *post-bias* (학습-시 bias 가산됨)
            - `pre_bias_scores` 가 필요할 때 = `ens_scores - ens_prior` (헬퍼 제공)

      3. **regimes per-sample 라벨** — plan §3.1 은 plan-004 산출에서 우선 로드를 명시하나
         `analysis/plan-004/regime_distribution.json` 은 18-bin histogram 만 박제 (per-sample 없음).
         본 모듈은 `selector.fit_regime_bins(train_x) + assign_regimes(train_x)` 로 재계산하고
         18-bin 히스토그램을 plan-004 박제와 sanity-check (`regime_histogram_drift_max=0` 기대).

      4. **physics_bias / regime_bias_table** — plan §3.1 은 selector 모듈에서 import 를 명시하나
         두 항은 *fold-dependent* (per-fold 학습 데이터로 fit). full-train 으로 한 번 fit 한 값을
         `_load_shared_inputs` 에서 캐시. plan-005 §6 의 `final_scores = ens_scores` 가 직접
         post-bias 라 두 bias 분해는 §8.3 Variant B 만 필요 — 거기선 `ens_prior` (per-fold 평균 bias)
         가 ground-truth bias-only 점수라 더 정확. fitted physics_bias / regime_bias_table 은
         per-family / per-regime 진단용 보조 (미사용 가능).

    Args:
        recompute_y_from_csv: True 면 train_labels.csv 에서 train_y 재로드 (sanity-check).
                              False 면 oof_selector_scores.npz["y"] 사용 (caching 으로 빠름).
    """
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    oof = np.load(PLAN004_DIR / "oof_selector_scores.npz", allow_pickle=True)
    test = np.load(PLAN004_DIR / "test_selector_scores.npz", allow_pickle=True)
    regime_dist = json.loads(PLAN004_REGIME_JSON.read_text())

    train_y = oof["y"].astype(np.float32)
    train_cands = oof["cands"].astype(np.float32)
    ens_scores = oof["ens_scores"].astype(np.float32)
    ens_prior = oof["ens_prior"].astype(np.float32)
    candidate_names = list(map(str, oof["candidate_names"]))
    covered = oof["covered"].astype(bool)

    test_cands = test["cands"].astype(np.float32)
    test_ens_scores = test["ens_scores"].astype(np.float32)

    # candidate_family [27] (0~5) — selector.CANDIDATE_FAMILY exposes this
    cand_family = np.asarray(selector.CANDIDATE_FAMILY, dtype=np.int64)
    assert cand_family.shape == (N_CANDIDATES,), cand_family.shape
    assert cand_family.max() <= 5 and cand_family.min() >= 0

    # train_x for regime / corrector — load from CSV
    ids, train_x_raw, train_y_csv = load_train_data()
    assert len(ids) == train_y.shape[0], f"id count mismatch: csv={len(ids)} vs npz={train_y.shape[0]}"
    if recompute_y_from_csv:
        # tolerance 1e-5 (float32 round-trip)
        max_diff = float(np.abs(train_y - train_y_csv).max())
        assert max_diff < 1e-4, f"train_y npz vs csv mismatch: {max_diff}"

    # regime per-sample labels — full-train fit (consistent with full_train_predict path)
    end_idx = train_x_raw.shape[1] - 1
    regime_bins = selector.fit_regime_bins(train_x_raw, end_idx)
    regimes = selector.assign_regimes(train_x_raw, end_idx, regime_bins).astype(np.int64)
    assert regimes.shape == (train_y.shape[0],)
    assert int(regimes.min()) >= 0 and int(regimes.max()) < N_REGIMES

    # sanity vs plan-004 histogram (drift assert)
    hist_local = np.bincount(regimes, minlength=N_REGIMES).tolist()
    hist_p4 = list(regime_dist["regime_histogram"])
    drift = max(abs(a - b) for a, b in zip(hist_local, hist_p4))
    if drift != 0:
        # warn (not severe) — record + continue with locally fit regimes
        print(
            f"[WARN] regime_distribution_path_drift: max bin diff={drift} "
            f"(local={hist_local} vs plan004={hist_p4})",
            file=sys.stderr,
        )

    # full-train physics_bias / regime_bias_table (cached for §8 if needed)
    physics_bias_full = (selector.candidate_physics_bias(train_cands, train_y) * 0.65).astype(np.float32)
    regime_bias_table_full = selector.candidate_regime_bias(
        train_cands, train_y, regimes, regime_count=N_REGIMES
    ).astype(np.float32)
    assert regime_bias_table_full.shape == (N_REGIMES, N_CANDIDATES)

    return {
        # primary OOF arrays (use these in stage analysis)
        "ids": ids,
        "covered": covered,                       # [N_train] bool
        "train_y": train_y,                       # [N_train, 3]
        "train_cands": train_cands,               # [N_train, 27, 3]
        "ens_scores": ens_scores,                 # [N_train, 27]   POST-BIAS (≡ final_scores §6)
        "ens_prior": ens_prior,                   # [N_train, 27]   bias-only (≡ Variant B score_B §8.3)
        "candidate_names": candidate_names,
        # test arrays (sanity-check; main analysis uses OOF only)
        "test_cands": test_cands,                 # [N_test, 27, 3]
        "test_ens_scores": test_ens_scores,       # [N_test, 27]
        # auxiliaries
        "train_x": train_x_raw,                   # [N_train, T, 3]
        "regimes": regimes,                       # [N_train] int (0~17)
        "regime_bins": regime_bins,               # dict
        "regime_histogram_local": hist_local,
        "regime_histogram_plan004": hist_p4,
        "regime_histogram_drift": int(drift),
        "cand_family": cand_family,               # [27] int (0~5)
        "physics_bias_full": physics_bias_full,           # [27]   pre-scaled with prior_strength=0.65
        "regime_bias_table_full": regime_bias_table_full, # [18, 27] (regime_prior_strength multiplier NOT applied)
        # constants
        "R_HIT": R_HIT,
    }


def pre_bias_scores(ens_scores: np.ndarray, ens_prior: np.ndarray) -> np.ndarray:
    """Helper — per-fold bias 를 OOF score 에서 제거하여 model-only logit 복원."""
    return (ens_scores - ens_prior).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# CLI dispatcher
# ─────────────────────────────────────────────────────────────────────────────


def _cmd_verify(args: argparse.Namespace) -> int:
    summary = verify_plan004_artifacts()
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def _cmd_smoke_load(args: argparse.Namespace) -> int:
    """c2 smoke — _load_shared_inputs() 가 정상 로드 + dimension/shape 일치 확인."""
    bundle = _load_shared_inputs(recompute_y_from_csv=True)
    summary = {
        "n_train": int(bundle["train_y"].shape[0]),
        "train_y_shape": list(bundle["train_y"].shape),
        "train_cands_shape": list(bundle["train_cands"].shape),
        "ens_scores_shape": list(bundle["ens_scores"].shape),
        "ens_prior_shape": list(bundle["ens_prior"].shape),
        "test_cands_shape": list(bundle["test_cands"].shape),
        "regime_histogram_drift": bundle["regime_histogram_drift"],
        "regime_histogram_local": bundle["regime_histogram_local"],
        "cand_family": bundle["cand_family"].tolist(),
        "candidate_names_first5": bundle["candidate_names"][:5],
        "physics_bias_l1": float(np.abs(bundle["physics_bias_full"]).sum()),
        "regime_bias_table_l1": float(np.abs(bundle["regime_bias_table_full"]).sum()),
        "covered_count": int(bundle["covered"].sum()),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("verify", help="G0 entry — verify plan-004 artifacts").set_defaults(_fn=_cmd_verify)
    sub.add_parser("smoke-load", help="c2 smoke — _load_shared_inputs sanity").set_defaults(_fn=_cmd_smoke_load)
    args = parser.parse_args(argv)
    return args._fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
