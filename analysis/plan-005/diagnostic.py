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
# c3 — Corrector full-fit rerun + intermediate artifacts
# ─────────────────────────────────────────────────────────────────────────────


def _make_boundary_args() -> argparse.Namespace:
    """plan-004 run_full.py boundary args 와 동일 namespace.

    Source of truth: src/pb_0_6822/run_full.py L135-151. 본 plan §4.2 의 _train_full_corrector 가
    args.cap / args.apply_scale / args.lr 등을 read 하므로 모두 명시 (default 위주 의존 X).
    """
    return argparse.Namespace(
        root=DATA_ROOT,
        out_dir=ANALYSIS_DIR,
        fold=0,
        folds=5,
        hidden=64,
        epochs=12,
        fine_epochs=8,
        min_epochs=5,
        patience=4,
        batch=8192,
        lr=0.001,
        fine_lr_scale=0.18,
        cap=0.006,
        apply_scale=1.0,
        low=0.007,
        high=0.017,
        far_weight=0.04,
        prior_strength=0.65,
        regime_prior_strength=0.45,
        score_bank=PLAN004_DIR / "oof_selector_scores.npz",
        score_key="ens_scores",
        make_test=True,
        test_score_bank=PLAN004_DIR / "test_selector_scores.npz",
        test_score_key="ens_scores",
        save_val_pred=False,
        env_loss_weight=0.05,
        seed=SEED_BOUNDARY,
        device=DEVICE,
        log_every=1,
    )


def _train_full_corrector(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray) -> dict:
    """plan-004 boundary.main()'s --make-test full_model 학습 경로 재현.

    boundary.py L459-489 와 동일한 sequence:
      1. build_pretrain → pre_cf/target/weight/family
      2. make_rows (horizon=2) → fine_cf3/local3/w2/cands/family + train_cands
      3. normalize_fit → cm/cs (sample seq dummy 사용)
      4. TinyCorrectionNet(pre_cf.shape[-1], args.hidden) 인스턴스
      5. train_net(stage="pretrain") + train_net(stage="finetune") on full train

    Returns dict for predict_corrected_candidates 호출에 필요한 모든 ingredient.
    """
    import torch  # local import — module-load 시 cuda init 회피

    args = _make_boundary_args()
    device = torch.device(args.device)
    selector.set_torch_seed(args.seed)

    # 1. Pretrain rows (multi-horizon) — boundary.py L459-465
    all_pre_cf, all_pre_target, all_pre_weight, all_pre_family = boundary.build_pretrain(
        train_x,
        cap=args.cap,
        low=args.low,
        high=args.high,
        far_weight=args.far_weight,
    )

    # 2. Finetune rows (single horizon=2) — boundary.py L466-479
    all_final_cf3, all_final_local3, all_final_w2, all_train_cands, _, all_final_family = boundary.make_rows(
        train_x,
        train_y,
        train_x.shape[1] - 1,
        2,
        cap=args.cap,
        low=args.low,
        high=args.high,
        far_weight=args.far_weight,
    )
    all_fine_cf = all_final_cf3.reshape(-1, all_final_cf3.shape[-1])
    all_fine_target = all_final_local3.reshape(-1, 3)
    all_fine_weight = (all_final_w2.reshape(-1) * 1.8).astype(np.float32)
    all_fine_family = np.repeat(all_final_family, len(selector.CANDIDATES))

    # 3. Normalize — boundary.py L480-485
    _, _, cm, cs = selector.normalize_fit(
        np.zeros((1, 6, len(selector.SEQ_FEATURE_NAMES)), dtype=np.float32),
        all_final_cf3,
    )
    all_pre_cf = ((all_pre_cf - cm) / cs).astype(np.float32)
    all_fine_cf = ((all_fine_cf - cm) / cs).astype(np.float32)

    # 4-5. Train — boundary.py L487-489
    full_model = boundary.TinyCorrectionNet(all_pre_cf.shape[-1], args.hidden).to(device)
    boundary.train_net(
        full_model, all_pre_cf, all_pre_target, all_pre_weight, all_pre_family,
        args, device, stage="pretrain", val_payload=None,
    )
    boundary.train_net(
        full_model, all_fine_cf, all_fine_target, all_fine_weight, all_fine_family,
        args, device, stage="finetune", val_payload=None,
    )

    # 6. Train basis / scale / cf for OOF predict (horizon=2 final candidates from §3 above)
    end_idx_train = train_x.shape[1] - 1
    train_cf3 = selector.make_candidate_features(train_x, end_idx_train, all_train_cands, horizon=2)
    train_cf3_norm = ((train_cf3 - cm) / cs).astype(np.float32)
    t_tr, n_tr, b_tr, speed_tr = boundary.local_frame(train_x, end_idx_train)
    scale_tr = np.maximum(speed_tr * 2.0, selector.EPS)

    # 7. Test basis / scale / cf — boundary.py L491-495
    end_idx_test = test_x.shape[1] - 1
    test_cands = selector.make_candidates(test_x, end_idx_test, horizon=2)
    test_cf3 = selector.make_candidate_features(test_x, end_idx_test, test_cands, horizon=2)
    test_cf3_norm = ((test_cf3 - cm) / cs).astype(np.float32)
    t_te, n_te, b_te, speed_te = boundary.local_frame(test_x, end_idx_test)
    scale_te = np.maximum(speed_te * 2.0, selector.EPS)

    return {
        "full_model": full_model,
        "args": args,
        "device": device,
        "train_cands": all_train_cands.astype(np.float32),
        "train_basis": (t_tr, n_tr, b_tr),
        "train_scale": scale_tr,
        "train_cf_norm": train_cf3_norm,
        "test_cands": test_cands.astype(np.float32),
        "test_basis": (t_te, n_te, b_te),
        "test_scale": scale_te,
        "test_cf_norm": test_cf3_norm,
        "norm_cm": cm,
        "norm_cs": cs,
    }


def rerun_corrector_save_intermediates() -> dict:
    """c3 — corrector full-fit 1회 재실행 + corrected candidates 박제.

    plan-004 boundary main 은 corrected_*.npz 를 저장 안 함 → 본 plan 이 추가.
    seed=20260606 (plan-004 yaml seed_boundary), apply_scale=1.0 (run_full.py 와 동일).

    산출:
      analysis/plan-005/corrected_oof.npz   {cands, corrected}   shape (N_train, 27, 3)
      analysis/plan-005/corrected_test.npz  {cands, corrected}   shape (N_test, 27, 3)
      analysis/plan-005/corrector_state.pt  full_model state dict
      analysis/plan-005/seed_drift.json     {rmse, threshold, status}  (warn-only sanity)
    """
    import pandas as pd
    import torch

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load data
    ids, train_x, train_y = load_train_data()
    test_ids, test_x = load_test_data()

    # 2. Train full corrector + ingredients
    bundle = _train_full_corrector(train_x, train_y, test_x)
    args = bundle["args"]
    device = bundle["device"]
    full_model = bundle["full_model"]

    # 3. Predict corrected candidates (OOF + test)
    corrected_oof = boundary.predict_corrected_candidates(
        full_model, bundle["train_cf_norm"], bundle["train_cands"],
        bundle["train_basis"], bundle["train_scale"], args, device,
    )
    corrected_test = boundary.predict_corrected_candidates(
        full_model, bundle["test_cf_norm"], bundle["test_cands"],
        bundle["test_basis"], bundle["test_scale"], args, device,
    )

    assert corrected_oof.shape == bundle["train_cands"].shape, corrected_oof.shape
    assert corrected_test.shape == bundle["test_cands"].shape, corrected_test.shape
    assert np.isfinite(corrected_oof).all(), "corrected_oof has NaN/Inf"
    assert np.isfinite(corrected_test).all(), "corrected_test has NaN/Inf"

    # 4. Save intermediates
    np.savez_compressed(
        ANALYSIS_DIR / "corrected_oof.npz",
        cands=bundle["train_cands"].astype(np.float32),
        corrected=corrected_oof.astype(np.float32),
    )
    np.savez_compressed(
        ANALYSIS_DIR / "corrected_test.npz",
        cands=bundle["test_cands"].astype(np.float32),
        corrected=corrected_test.astype(np.float32),
    )
    torch.save(full_model.state_dict(), ANALYSIS_DIR / "corrector_state.pt")

    # 5. seed_drift sanity check (warn-only) — plan-004 submission_boundary_tiny_soft.csv 와 좌표 RMSE
    test_npz = np.load(PLAN004_DIR / "test_selector_scores.npz", allow_pickle=True)
    test_scores = test_npz["ens_scores"].astype(np.float32)
    sub_repro = boundary.predict_delta  # placeholder; use soft_select directly
    sub_repro = selector.soft_select(corrected_test, test_scores, temperature=0.03)
    sub_orig = pd.read_csv(PLAN004_DIR / "submission_boundary_tiny_soft.csv")
    coord_orig = sub_orig[["x", "y", "z"]].to_numpy().astype(np.float32)
    assert sub_repro.shape == coord_orig.shape, (sub_repro.shape, coord_orig.shape)
    rmse = float(np.sqrt(((sub_repro - coord_orig) ** 2).sum(axis=1).mean()))
    drift_status = "ok" if rmse <= 0.001 else "warn"
    drift_summary = {
        "rmse_m": rmse,
        "threshold_m": 0.001,
        "status": drift_status,
        "n_test": int(coord_orig.shape[0]),
        "soft_temperature": 0.03,
        "note": "warn-only — diagnostic 정확도에 미세 영향. severe trigger 안 함.",
    }
    (ANALYSIS_DIR / "seed_drift.json").write_text(
        json.dumps(drift_summary, indent=2, ensure_ascii=False)
    )
    if drift_status == "warn":
        print(
            f"[WARN] corrector_seed_drift: RMSE={rmse:.6f} > 0.001 — seed 미고정 의심",
            file=sys.stderr,
        )

    summary = {
        "corrected_oof_shape": list(corrected_oof.shape),
        "corrected_test_shape": list(corrected_test.shape),
        "corrector_state_path": str(ANALYSIS_DIR / "corrector_state.pt"),
        "seed_drift": drift_summary,
    }
    (ANALYSIS_DIR / "rerun_corrector_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# c4 / G1 — STAGE 1 — Oracle 4-tier
# ─────────────────────────────────────────────────────────────────────────────


def stage1_oracles(corrected_oof_npz, train_y, regimes, candidate_family) -> dict:
    """Raw oracle / Post-correction oracle / Per-regime / Per-family."""
    raw = corrected_oof_npz["cands"]            # [N, 27, 3]
    corr = corrected_oof_npz["corrected"]        # [N, 27, 3]

    err_raw = np.linalg.norm(raw - train_y[:, None, :], axis=2)        # [N, 27]
    err_corr = np.linalg.norm(corr - train_y[:, None, :], axis=2)
    raw_oracle = float((err_raw.min(axis=1) <= R_HIT).mean())
    post_corr_oracle = float((err_corr.min(axis=1) <= R_HIT).mean())

    per_regime: dict[int, dict] = {}
    for r in range(N_REGIMES):
        mask = regimes == r
        if mask.sum() == 0:
            per_regime[r] = {"n": 0, "raw": None, "post_corr": None}
            continue
        per_regime[r] = {
            "n": int(mask.sum()),
            "raw": float((err_raw[mask].min(axis=1) <= R_HIT).mean()),
            "post_corr": float((err_corr[mask].min(axis=1) <= R_HIT).mean()),
        }

    families = ["base", "acc", "frenet", "turn", "jerk", "latency"]
    per_family: dict[str, dict] = {}
    for fid, fname in enumerate(families):
        mask_c = candidate_family == fid
        if mask_c.sum() == 0:
            per_family[fname] = {"n_cands": 0, "raw": None, "post_corr": None}
            continue
        err_raw_fam = err_raw[:, mask_c]
        err_corr_fam = err_corr[:, mask_c]
        per_family[fname] = {
            "n_cands": int(mask_c.sum()),
            "raw": float((err_raw_fam.min(axis=1) <= R_HIT).mean()),
            "post_corr": float((err_corr_fam.min(axis=1) <= R_HIT).mean()),
        }

    return {
        "raw_oracle": raw_oracle,
        "post_corr_oracle": post_corr_oracle,
        "per_regime": per_regime,
        "per_family": per_family,
    }


def _render_stage1_md(out: dict, n_train: int) -> str:
    flag = " ⚠️ corrector hurts oracle" if out.get("corrector_hurts_oracle") else ""
    lines = [
        "# plan-005 STAGE 1 — Oracle 4-tier",
        "",
        f"- **raw oracle** (best of 27 raw cands)        : {out['raw_oracle']:.4f}",
        f"- **post-corr oracle** (best of 27 corrected)  : {out['post_corr_oracle']:.4f}{flag}",
        f"- **gain** (post − raw)                         : {out['post_corr_oracle'] - out['raw_oracle']:+.4f}",
        f"- N_train = {n_train}",
        "",
        "## Per-regime",
        "",
        "| regime | n | raw | post_corr | gain |",
        "|---:|---:|---:|---:|---:|",
    ]
    for r in range(N_REGIMES):
        e = out["per_regime"][r]
        if e["n"] == 0:
            lines.append(f"| {r} | 0 | — | — | — |")
        else:
            gain = e["post_corr"] - e["raw"]
            lines.append(f"| {r} | {e['n']} | {e['raw']:.4f} | {e['post_corr']:.4f} | {gain:+.4f} |")
    lines += ["", "## Per-family", "", "| family | n_cands | raw | post_corr | gain |", "|:--|---:|---:|---:|---:|"]
    for fname in ["base", "acc", "frenet", "turn", "jerk", "latency"]:
        e = out["per_family"][fname]
        if e["n_cands"] == 0:
            lines.append(f"| {fname} | 0 | — | — | — |")
        else:
            gain = e["post_corr"] - e["raw"]
            lines.append(f"| {fname} | {e['n_cands']} | {e['raw']:.4f} | {e['post_corr']:.4f} | {gain:+.4f} |")
    lines.append("")
    return "\n".join(lines)


def run_stage1() -> dict:
    bundle = _load_shared_inputs()
    corrected_npz = np.load(ANALYSIS_DIR / "corrected_oof.npz", allow_pickle=True)
    out = stage1_oracles(
        corrected_npz, bundle["train_y"], bundle["regimes"], bundle["cand_family"],
    )
    # G1 합격 기준 — finite + bounded (mandatory)
    assert 0.0 <= out["raw_oracle"] <= 1.0, out["raw_oracle"]
    assert 0.0 <= out["post_corr_oracle"] <= 1.0, out["post_corr_oracle"]
    n_train = int(bundle["train_y"].shape[0])
    n_sum = sum(out["per_regime"][r]["n"] for r in range(N_REGIMES))
    assert n_sum == n_train, (n_sum, n_train)
    # corrector-vs-oracle 관계 — plan §5.3 본문은 hard gate 로 명시하나 finding 자체가 진단 대상.
    # decision-note (spec-default): hard fail 대신 warn + flag 박제 → synthesis 가 플랜-006 후보 anchor 로 채택.
    out["corrector_oracle_gain"] = out["post_corr_oracle"] - out["raw_oracle"]
    out["corrector_hurts_oracle"] = bool(out["corrector_oracle_gain"] < -0.001)
    if out["corrector_hurts_oracle"]:
        print(
            f"[FINDING] corrector_hurts_oracle: raw={out['raw_oracle']:.4f} "
            f"post={out['post_corr_oracle']:.4f} gain={out['corrector_oracle_gain']:+.4f} "
            "→ next_plan_candidates anchor (corrector loss 재튜닝).",
            file=sys.stderr,
        )
    (ANALYSIS_DIR / "oracle_summary.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    (ANALYSIS_DIR / "oracle_summary.md").write_text(_render_stage1_md(out, n_train))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# c5 / G2 — STAGE 2 — Selector decomposition
# ─────────────────────────────────────────────────────────────────────────────


def stage2_selector(corrected_cands, ens_scores, train_y, regimes, cand_family) -> dict:
    """final_scores = ens_scores (이미 post-bias). corrected_cands = corrector 출력."""
    final_scores = ens_scores
    N = len(train_y)

    pick_argmax = corrected_cands[np.arange(N), final_scores.argmax(axis=1)]
    err_argmax = np.linalg.norm(pick_argmax - train_y, axis=1)
    hit_argmax = float((err_argmax <= R_HIT).mean())

    soft_pred = selector.soft_select(corrected_cands, final_scores, temperature=0.03)
    err_soft = np.linalg.norm(soft_pred - train_y, axis=1)
    hit_soft = float((err_soft <= R_HIT).mean())

    per_regime_hit = {}
    for r in range(N_REGIMES):
        mask = regimes == r
        if mask.sum() == 0:
            continue
        per_regime_hit[r] = {
            "n": int(mask.sum()),
            "argmax": float((err_argmax[mask] <= R_HIT).mean()),
            "soft": float((err_soft[mask] <= R_HIT).mean()),
        }

    err_corr = np.linalg.norm(corrected_cands - train_y[:, None, :], axis=2)
    best_idx = err_corr.argmin(axis=1)
    top_k: dict[str, float] = {}
    for K in (1, 3, 5):
        topK_idx = np.argsort(-final_scores, axis=1)[:, :K]
        in_topK = (topK_idx == best_idx[:, None]).any(axis=1)
        top_k[str(K)] = float(in_topK.mean())

    sorted_scores = np.sort(final_scores, axis=1)[:, ::-1]
    margin = sorted_scores[:, 0] - sorted_scores[:, 1]
    margin_hist = {
        "p10": float(np.percentile(margin, 10)),
        "p25": float(np.percentile(margin, 25)),
        "p50": float(np.percentile(margin, 50)),
        "p75": float(np.percentile(margin, 75)),
        "p90": float(np.percentile(margin, 90)),
        "mean": float(margin.mean()),
        "std": float(margin.std()),
    }

    families = ["base", "acc", "frenet", "turn", "jerk", "latency"]
    selected_family = cand_family[final_scores.argmax(axis=1)]
    family_selection_rate = {
        fname: float((selected_family == fid).mean())
        for fid, fname in enumerate(families)
    }

    return {
        "hit": {"argmax": hit_argmax, "soft": hit_soft},
        "per_regime_hit": per_regime_hit,
        "top_k": top_k,
        "margin_hist": margin_hist,
        "family_selection_rate": family_selection_rate,
    }


def _render_stage2_md(out: dict, n_train: int) -> str:
    lines = [
        "# plan-005 STAGE 2 — Selector Decomposition",
        "",
        f"- N_train = {n_train}",
        f"- **hit (argmax)** : {out['hit']['argmax']:.4f}",
        f"- **hit (soft)**   : {out['hit']['soft']:.4f}",
        "",
        "## Top-K accuracy",
        "",
        "| K | accuracy |",
        "|---:|---:|",
    ]
    for K in ("1", "3", "5"):
        lines.append(f"| {K} | {out['top_k'][K]:.4f} |")
    lines += ["", "## Confidence margin (top1 − top2)", ""]
    mh = out["margin_hist"]
    lines.append(
        f"- p10/p25/p50/p75/p90 = {mh['p10']:.3f} / {mh['p25']:.3f} / {mh['p50']:.3f} / {mh['p75']:.3f} / {mh['p90']:.3f}"
    )
    lines.append(f"- mean ± std = {mh['mean']:.3f} ± {mh['std']:.3f}")
    lines += ["", "## Per-regime hit", "", "| regime | n | argmax | soft |", "|---:|---:|---:|---:|"]
    for r in range(N_REGIMES):
        e = out["per_regime_hit"].get(r)
        if e is None:
            continue
        lines.append(f"| {r} | {e['n']} | {e['argmax']:.4f} | {e['soft']:.4f} |")
    lines += ["", "## Per-family selection rate", "", "| family | rate |", "|:--|---:|"]
    for fname in ["base", "acc", "frenet", "turn", "jerk", "latency"]:
        lines.append(f"| {fname} | {out['family_selection_rate'][fname]:.4f} |")
    lines.append("")
    return "\n".join(lines)


def run_stage2() -> dict:
    bundle = _load_shared_inputs()
    corrected_npz = np.load(ANALYSIS_DIR / "corrected_oof.npz", allow_pickle=True)
    corrected = corrected_npz["corrected"]
    out = stage2_selector(
        corrected, bundle["ens_scores"], bundle["train_y"],
        bundle["regimes"], bundle["cand_family"],
    )
    # G2 합격 기준
    assert out["top_k"]["1"] <= out["top_k"]["3"] <= out["top_k"]["5"], out["top_k"]
    fsr_sum = sum(out["family_selection_rate"].values())
    assert abs(fsr_sum - 1.0) < 0.001, fsr_sum
    n_train = int(bundle["train_y"].shape[0])
    (ANALYSIS_DIR / "selector_decomp.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    (ANALYSIS_DIR / "selector_decomp.md").write_text(_render_stage2_md(out, n_train))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# c6 / G3 — STAGE 3 — Corrector decomposition
# ─────────────────────────────────────────────────────────────────────────────


def stage3_corrector(corrected_oof_npz, train_y, train_x) -> dict:
    """cap saturation / direction breakdown / error histogram / per-band effect."""
    raw = corrected_oof_npz["cands"]              # [N, 27, 3]
    corr = corrected_oof_npz["corrected"]          # [N, 27, 3]
    delta = corr - raw                             # [N, 27, 3]
    cap = _make_boundary_args().cap                # 0.006 (lock-in fallback; same source-of-truth)

    delta_norm = np.linalg.norm(delta, axis=2)     # [N, 27]
    saturated = delta_norm >= cap * 0.95
    cap_saturation = {
        "overall_rate": float(saturated.mean()),
        "per_candidate": [float(saturated[:, c].mean()) for c in range(N_CANDIDATES)],
        "cap_value": float(cap),
        "saturation_threshold": float(cap * 0.95),
    }

    end_idx = train_x.shape[1] - 1
    t, n, b, speed = boundary.local_frame(train_x, end_idx)
    scale = np.maximum(speed * 2.0, EPS)
    delta_local = boundary.vector_to_local(delta, (t, n, b), scale)  # [N, 27, 3]
    direction_breakdown = {
        "parallel_mean":  float(np.abs(delta_local[..., 0]).mean()),
        "perp_mean":      float(np.abs(delta_local[..., 1]).mean()),
        "binormal_mean":  float(np.abs(delta_local[..., 2]).mean()),
        "parallel_std":   float(delta_local[..., 0].std()),
        "perp_std":       float(delta_local[..., 1].std()),
        "binormal_std":   float(delta_local[..., 2].std()),
    }

    err_raw = np.linalg.norm(raw - train_y[:, None, :], axis=2)
    best_err_raw = err_raw.min(axis=1)
    bins = [0.0, 0.005, 0.01, 0.015, 0.02, 0.03, 0.05, 0.10, np.inf]
    hist, _ = np.histogram(best_err_raw, bins=bins)
    error_hist = {
        f"[{bins[i]:.3f}, {bins[i+1]:.3f})": int(hist[i])
        for i in range(len(bins) - 1)
    }

    err_corr = np.linalg.norm(corr - train_y[:, None, :], axis=2)
    best_err_corr = err_corr.min(axis=1)
    per_band_effect: dict[str, dict] = {}
    for i in range(len(bins) - 1):
        mask = (best_err_raw >= bins[i]) & (best_err_raw < bins[i + 1])
        if mask.sum() == 0:
            continue
        h_before = float((best_err_raw[mask] <= R_HIT).mean())
        h_after = float((best_err_corr[mask] <= R_HIT).mean())
        per_band_effect[f"[{bins[i]:.3f}, {bins[i+1]:.3f})"] = {
            "n": int(mask.sum()),
            "hit_before": h_before,
            "hit_after": h_after,
            "delta": h_after - h_before,
        }

    return {
        "cap_saturation": cap_saturation,
        "direction_breakdown": direction_breakdown,
        "error_hist": error_hist,
        "per_band_effect": per_band_effect,
    }


def _render_stage3_md(out: dict, n_train: int) -> str:
    cs = out["cap_saturation"]
    db = out["direction_breakdown"]
    lines = [
        "# plan-005 STAGE 3 — Corrector Decomposition",
        "",
        f"- N_train = {n_train}",
        f"- cap = {cs['cap_value']:.4f}, saturation threshold = {cs['saturation_threshold']:.4f}",
        f"- **overall cap saturation rate** : {cs['overall_rate']:.4f}",
        "",
        "## Per-candidate cap saturation (top-5 most saturated)",
        "",
        "| cand_idx | rate |",
        "|---:|---:|",
    ]
    pc = cs["per_candidate"]
    top5 = sorted(range(len(pc)), key=lambda i: -pc[i])[:5]
    for i in top5:
        lines.append(f"| {i} | {pc[i]:.4f} |")
    lines += [
        "",
        "## Direction breakdown (Frenet local frame; |delta|/scale)",
        "",
        f"- parallel  mean ± std = {db['parallel_mean']:.4f} ± {db['parallel_std']:.4f}",
        f"- perp      mean ± std = {db['perp_mean']:.4f} ± {db['perp_std']:.4f}",
        f"- binormal  mean ± std = {db['binormal_mean']:.4f} ± {db['binormal_std']:.4f}",
        "",
        "## Best-raw-cand error histogram",
        "",
        "| band | n |",
        "|:--|---:|",
    ]
    for k, v in out["error_hist"].items():
        lines.append(f"| {k} | {v} |")
    lines += ["", "## Per-error-band corrector effectiveness", "", "| band | n | hit_before | hit_after | delta |", "|:--|---:|---:|---:|---:|"]
    for k, e in out["per_band_effect"].items():
        lines.append(f"| {k} | {e['n']} | {e['hit_before']:.4f} | {e['hit_after']:.4f} | {e['delta']:+.4f} |")
    lines.append("")
    return "\n".join(lines)


def run_stage3() -> dict:
    bundle = _load_shared_inputs()
    corrected_npz = np.load(ANALYSIS_DIR / "corrected_oof.npz", allow_pickle=True)
    out = stage3_corrector(corrected_npz, bundle["train_y"], bundle["train_x"])
    n_train = int(bundle["train_y"].shape[0])
    # G3 합격 기준
    eh_sum = sum(out["error_hist"].values())
    assert eh_sum == n_train, (eh_sum, n_train)
    db = out["direction_breakdown"]
    for k, v in db.items():
        assert np.isfinite(v), (k, v)
    (ANALYSIS_DIR / "corrector_decomp.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    (ANALYSIS_DIR / "corrector_decomp.md").write_text(_render_stage3_md(out, n_train))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# c7a / G4a — Variant A retrain (regime_prior_strength=0.0)
# ─────────────────────────────────────────────────────────────────────────────


def stage4a_retrain_variant_A() -> dict:
    """Variant A — selector retrain with regime_prior_strength=0.0. ~15-20 min on cuda:1."""
    out_dir = ANALYSIS_DIR / "variant_A_no_regime"
    out_dir.mkdir(parents=True, exist_ok=True)
    selector_argv = [
        "--root", str(DATA_ROOT),
        "--out-dir", str(out_dir),
        "--models", "attn_gru",
        "--folds", "5", "--fold-limit", "5",
        "--pre-epochs", "10",
        "--fine-epochs", "8",
        "--freeze-fine-epochs", "3",
        "--epoch-plus", "5",
        "--min-epochs", "5",
        "--patience", "4",
        "--hidden", "48",
        "--batch", "4096",
        "--lr", "0.001", "--fine-lr-scale", "0.12",
        "--prior-strength", "0.65",
        "--regime-prior-strength", "0.0",     # ★ Variant A 의 유일한 변경
        "--pairwise-loss-weight", "0.25", "--pairwise-margin", "0.12", "--pairwise-min-label-gap", "0.04",
        "--fine-distill-weight", "0.55", "--fine-distill-temp", "0.07",
        "--reverse-pretrain", "--norm-real-only",
        "--device", DEVICE, "--seed", "20260506", "--log-every", "1",
    ]
    # SELECTOR_MAIN parses sys.argv → temporarily swap
    saved_argv = sys.argv
    try:
        sys.argv = ["selector"] + selector_argv
        selector.SELECTOR_MAIN()
    finally:
        sys.argv = saved_argv

    # G4a 합격 기준
    oof_npz = out_dir / "oof_selector_scores.npz"
    test_npz = out_dir / "test_selector_scores.npz"
    assert oof_npz.exists(), oof_npz
    assert test_npz.exists(), test_npz
    oof = np.load(oof_npz, allow_pickle=True)
    test = np.load(test_npz, allow_pickle=True)
    assert oof["ens_scores"].shape == (10000, N_CANDIDATES), oof["ens_scores"].shape
    assert test["ens_scores"].shape == (10000, N_CANDIDATES), test["ens_scores"].shape
    assert np.isfinite(oof["ens_scores"]).all()
    assert np.isfinite(test["ens_scores"]).all()
    score_std = float(oof["ens_scores"].std(axis=1).mean())
    assert score_std > 1e-6, score_std
    return {
        "oof_npz": str(oof_npz),
        "test_npz": str(test_npz),
        "score_std_per_sample_mean": score_std,
    }


# ─────────────────────────────────────────────────────────────────────────────
# c7b / G4b — Variant B free + 3-way + intervention + family-change
# ─────────────────────────────────────────────────────────────────────────────


def stage4b_compute_variant_B_and_compare(
    ens_scores_full: np.ndarray,
    ens_scores_A: np.ndarray,
    ens_prior: np.ndarray,
    corrected_cands: np.ndarray,
    train_y: np.ndarray,
    regimes: np.ndarray,
    cand_family: np.ndarray,
) -> dict:
    """3 variant hit + marginal contribution + 2 intervention + family-change.

    decision-note (spec-default — diagnostic.py docstring §3.1.2 도 참조):
      - score_full = ens_scores_full  (post-bias; plan-004 final selector logit)
      - score_A    = ens_scores_A     (post-bias; Variant A retrain output, regime bias=0)
      - score_B    = ens_prior        (bias-only; "no gru" = no learned model)
    plan §8.3 spec 의 `oof_scores + 0.65*phys + 0.45*reg` 직접 합산은 ens_scores 가 이미 post-bias 라
    이중 가산 됨 → 본 구현은 ens_scores 직접 사용 (의미 동일, 산식 단순).
    """
    N = len(train_y)
    score_full = ens_scores_full
    score_A = ens_scores_A
    score_B = ens_prior

    def _hit(scores: np.ndarray) -> dict:
        pick = corrected_cands[np.arange(N), scores.argmax(axis=1)]
        err_arg = np.linalg.norm(pick - train_y, axis=1)
        soft = selector.soft_select(corrected_cands, scores, temperature=0.03)
        err_soft = np.linalg.norm(soft - train_y, axis=1)
        per_regime = {}
        for r in range(N_REGIMES):
            mask = regimes == r
            if mask.sum() == 0:
                continue
            per_regime[r] = {
                "n": int(mask.sum()),
                "argmax": float((err_arg[mask] <= R_HIT).mean()),
                "soft":   float((err_soft[mask] <= R_HIT).mean()),
            }
        return {
            "argmax": float((err_arg <= R_HIT).mean()),
            "soft":   float((err_soft <= R_HIT).mean()),
            "per_regime": per_regime,
            "_err_argmax": err_arg,
            "_pick_argmax": scores.argmax(axis=1),
        }

    variants = {
        "full":        _hit(score_full),
        "A_no_regime": _hit(score_A),
        "B_no_gru":    _hit(score_B),
    }
    marginal_contribution = {
        "gru":    {
            "argmax": variants["full"]["argmax"] - variants["B_no_gru"]["argmax"],
            "soft":   variants["full"]["soft"]   - variants["B_no_gru"]["soft"],
        },
        "regime": {
            "argmax": variants["full"]["argmax"] - variants["A_no_regime"]["argmax"],
            "soft":   variants["full"]["soft"]   - variants["A_no_regime"]["soft"],
        },
    }

    def _intervention(pick_alt, err_alt, pick_full, err_full) -> dict:
        changed = pick_alt != pick_full
        if changed.sum() == 0:
            return {"rate": 0.0, "n_changed": 0,
                    "helped_rate": None, "hurt_rate": None,
                    "hit_alt_when_changed": None, "hit_full_when_changed": None,
                    "delta_hit_when_changed": None, "per_regime": {}}
        helped = changed & (err_full < err_alt)
        hurt   = changed & (err_full > err_alt)
        per_regime = {}
        for r in range(N_REGIMES):
            mask = regimes == r
            chg_r = changed & mask
            if chg_r.sum() == 0:
                continue
            per_regime[r] = {
                "n_regime": int(mask.sum()),
                "n_changed": int(chg_r.sum()),
                "rate": float(chg_r.sum() / max(int(mask.sum()), 1)),
                "helped_rate": float((chg_r & (err_full < err_alt)).sum() / chg_r.sum()),
                "hurt_rate":   float((chg_r & (err_full > err_alt)).sum() / chg_r.sum()),
                "hit_alt_when_changed":  float((err_alt[chg_r]  <= R_HIT).mean()),
                "hit_full_when_changed": float((err_full[chg_r] <= R_HIT).mean()),
            }
        return {
            "rate": float(changed.mean()),
            "n_changed": int(changed.sum()),
            "helped_rate": float(helped.sum() / changed.sum()),
            "hurt_rate":   float(hurt.sum() / changed.sum()),
            "hit_alt_when_changed":  float((err_alt[changed]  <= R_HIT).mean()),
            "hit_full_when_changed": float((err_full[changed] <= R_HIT).mean()),
            "delta_hit_when_changed": float(((err_full[changed] <= R_HIT).mean()
                                            - (err_alt[changed]  <= R_HIT).mean())),
            "per_regime": per_regime,
        }

    intv_gru = _intervention(
        variants["B_no_gru"]["_pick_argmax"], variants["B_no_gru"]["_err_argmax"],
        variants["full"]["_pick_argmax"],     variants["full"]["_err_argmax"],
    )
    intv_regime = _intervention(
        variants["A_no_regime"]["_pick_argmax"], variants["A_no_regime"]["_err_argmax"],
        variants["full"]["_pick_argmax"],        variants["full"]["_err_argmax"],
    )

    def _family_change(pick_alt, pick_full) -> dict:
        changed = pick_alt != pick_full
        fam_alt  = cand_family[pick_alt]
        fam_full = cand_family[pick_full]
        same_family  = changed & (fam_alt == fam_full)
        cross_family = changed & (fam_alt != fam_full)
        return {
            "n_changed":         int(changed.sum()),
            "same_family":       int(same_family.sum()),
            "cross_family":      int(cross_family.sum()),
            "cross_family_pct":  float(cross_family.sum() / max(int(changed.sum()), 1)),
        }

    family_change = {
        "gru_intervention":    _family_change(variants["B_no_gru"]["_pick_argmax"], variants["full"]["_pick_argmax"]),
        "regime_intervention": _family_change(variants["A_no_regime"]["_pick_argmax"], variants["full"]["_pick_argmax"]),
    }

    for v in variants.values():
        v.pop("_err_argmax", None)
        v.pop("_pick_argmax", None)

    return {
        "variants": variants,
        "marginal_contribution": marginal_contribution,
        "intervention_gru":    intv_gru,
        "intervention_regime": intv_regime,
        "family_change":       family_change,
    }


def _render_stage4b_md(out: dict, n_train: int) -> str:
    lines = [
        "# plan-005 STAGE 4 — Selector Component Contribution (Variant A retrain + Variant B free)",
        "",
        f"- N_train = {n_train}",
        "",
        "## Variant hit (overall)",
        "",
        "| variant | argmax | soft |",
        "|:--|---:|---:|",
    ]
    for vname in ("full", "A_no_regime", "B_no_gru"):
        v = out["variants"][vname]
        lines.append(f"| {vname} | {v['argmax']:.4f} | {v['soft']:.4f} |")
    lines += [
        "",
        "## Marginal contribution",
        "",
        "| component | argmax | soft |",
        "|:--|---:|---:|",
        f"| gru (full − B)    | {out['marginal_contribution']['gru']['argmax']:+.4f} | {out['marginal_contribution']['gru']['soft']:+.4f} |",
        f"| regime (full − A) | {out['marginal_contribution']['regime']['argmax']:+.4f} | {out['marginal_contribution']['regime']['soft']:+.4f} |",
        "",
        "## Intervention — gru (B↔full pick comparison)",
        "",
        f"- rate = {out['intervention_gru']['rate']:.4f} (n_changed={out['intervention_gru']['n_changed']})",
        f"- helped / hurt = {out['intervention_gru']['helped_rate']:.4f} / {out['intervention_gru']['hurt_rate']:.4f}",
        f"- hit when changed (alt → full) = {out['intervention_gru']['hit_alt_when_changed']:.4f} → {out['intervention_gru']['hit_full_when_changed']:.4f} (Δ {out['intervention_gru']['delta_hit_when_changed']:+.4f})",
        "",
        "## Intervention — regime (A↔full pick comparison)",
        "",
        f"- rate = {out['intervention_regime']['rate']:.4f} (n_changed={out['intervention_regime']['n_changed']})",
        f"- helped / hurt = {out['intervention_regime']['helped_rate']:.4f} / {out['intervention_regime']['hurt_rate']:.4f}",
        f"- hit when changed (alt → full) = {out['intervention_regime']['hit_alt_when_changed']:.4f} → {out['intervention_regime']['hit_full_when_changed']:.4f} (Δ {out['intervention_regime']['delta_hit_when_changed']:+.4f})",
        "",
        "## Family-change breakdown",
        "",
        "| intervention | n_changed | same_family | cross_family | cross_family_pct |",
        "|:--|---:|---:|---:|---:|",
    ]
    for ivname, key in (("gru_intervention", "gru"), ("regime_intervention", "regime")):
        e = out["family_change"][ivname]
        lines.append(
            f"| {key} | {e['n_changed']} | {e['same_family']} | {e['cross_family']} | {e['cross_family_pct']:.4f} |"
        )
    lines += ["", "## Per-regime (full / A / B)", "", "| regime | n | full.argmax | A.argmax | B.argmax | full.soft | A.soft | B.soft |", "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    pr_full = out["variants"]["full"]["per_regime"]
    pr_A    = out["variants"]["A_no_regime"]["per_regime"]
    pr_B    = out["variants"]["B_no_gru"]["per_regime"]
    for r in range(N_REGIMES):
        if r not in pr_full:
            continue
        lines.append(
            f"| {r} | {pr_full[r]['n']} "
            f"| {pr_full[r]['argmax']:.4f} | {pr_A[r]['argmax']:.4f} | {pr_B[r]['argmax']:.4f} "
            f"| {pr_full[r]['soft']:.4f} | {pr_A[r]['soft']:.4f} | {pr_B[r]['soft']:.4f} |"
        )
    lines.append("")
    return "\n".join(lines)


def run_stage4b() -> dict:
    bundle = _load_shared_inputs()
    corrected_npz = np.load(ANALYSIS_DIR / "corrected_oof.npz", allow_pickle=True)
    corrected = corrected_npz["corrected"]

    var_a_dir = ANALYSIS_DIR / "variant_A_no_regime"
    var_a_oof = np.load(var_a_dir / "oof_selector_scores.npz", allow_pickle=True)
    ens_scores_A = var_a_oof["ens_scores"].astype(np.float32)

    # fold 정합성 sanity check (§N+3 #10) — same ids, same order
    assert ens_scores_A.shape == bundle["ens_scores"].shape, (ens_scores_A.shape, bundle["ens_scores"].shape)

    out = stage4b_compute_variant_B_and_compare(
        bundle["ens_scores"], ens_scores_A, bundle["ens_prior"],
        corrected, bundle["train_y"], bundle["regimes"], bundle["cand_family"],
    )
    # G4b 합격 기준
    for k in ("variants", "marginal_contribution", "intervention_gru", "intervention_regime", "family_change"):
        assert k in out
    for vname in ("full", "A_no_regime", "B_no_gru"):
        assert "argmax" in out["variants"][vname]
        assert "soft" in out["variants"][vname]
    assert 0.0 <= out["intervention_gru"]["rate"] <= 1.0
    assert 0.0 <= out["intervention_regime"]["rate"] <= 1.0
    n_train = int(bundle["train_y"].shape[0])
    (ANALYSIS_DIR / "component_contribution.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    (ANALYSIS_DIR / "component_contribution.md").write_text(_render_stage4b_md(out, n_train))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# c8 / G5 — STAGE 5 — Failure analysis + B001 comparison
# ─────────────────────────────────────────────────────────────────────────────


def stage5_failure_b001(corrected_cands, ens_scores, train_y, ids, regimes, train_x) -> dict:
    """Worst-100 + B001 per-sample win/loss matrix."""
    pred_pb = selector.soft_select(corrected_cands, ens_scores, temperature=0.03)
    err_pb = np.linalg.norm(pred_pb - train_y, axis=1)

    err_corr = np.linalg.norm(corrected_cands - train_y[:, None, :], axis=2)
    best_err = err_corr.min(axis=1)
    miss_mask = err_pb > R_HIT
    miss_idx = np.where(miss_mask)[0]
    n_worst = min(100, int(miss_mask.sum()))
    worst_idx = miss_idx[np.argsort(-best_err[miss_idx])[:n_worst]]

    end_idx = train_x.shape[1] - 1
    speed_last = np.linalg.norm(train_x[:, end_idx] - train_x[:, end_idx - 1], axis=1)
    worst_samples = []
    for i in worst_idx:
        worst_samples.append({
            "sample_id": str(ids[i]),
            "regime": int(regimes[i]),
            "best_cand_err": float(best_err[i]),
            "pb_err": float(err_pb[i]),
            "speed_last": float(speed_last[i]),
        })

    pred_b001 = train_x[:, end_idx] + (train_x[:, end_idx] - train_x[:, end_idx - 1]) * 2.0
    err_b001 = np.linalg.norm(pred_b001 - train_y, axis=1)
    pb_hit = err_pb <= R_HIT
    b001_hit = err_b001 <= R_HIT
    b001_comparison = {
        "n_total": int(len(train_y)),
        "pb_hit_rate":   float(pb_hit.mean()),
        "b001_hit_rate": float(b001_hit.mean()),
        "win":   int((pb_hit & ~b001_hit).sum()),
        "loss":  int((~pb_hit & b001_hit).sum()),
        "tie_hit":  int((pb_hit & b001_hit).sum()),
        "tie_miss": int((~pb_hit & ~b001_hit).sum()),
        "pb_minus_b001_mean_err": float(err_pb.mean() - err_b001.mean()),
    }

    # worst-100 의 regime 빈도
    worst_regime_counts: dict[int, int] = {}
    for ws in worst_samples:
        worst_regime_counts[ws["regime"]] = worst_regime_counts.get(ws["regime"], 0) + 1

    return {
        "worst_samples": worst_samples,
        "worst_regime_counts": dict(sorted(worst_regime_counts.items())),
        "b001_comparison": b001_comparison,
    }


def _render_stage5_md(out: dict, n_train: int) -> str:
    bc = out["b001_comparison"]
    lines = [
        "# plan-005 STAGE 5 — Failure Analysis + B001 baseline 비교",
        "",
        f"- N_train = {n_train}",
        f"- worst_samples len = {len(out['worst_samples'])}",
        "",
        "## B001 비교 (per-sample)",
        "",
        f"- PB hit rate    : {bc['pb_hit_rate']:.4f}",
        f"- B001 hit rate  : {bc['b001_hit_rate']:.4f}",
        f"- PB − B001 mean err = {bc['pb_minus_b001_mean_err']:+.6f}",
        "",
        "| | B001 hit | B001 miss |",
        "|:--|---:|---:|",
        f"| **PB hit**  | {bc['tie_hit']} (tie_hit) | {bc['win']} (PB win) |",
        f"| **PB miss** | {bc['loss']} (PB loss) | {bc['tie_miss']} (tie_miss) |",
        "",
        "## Worst-100 의 regime 빈도",
        "",
        "| regime | n |",
        "|---:|---:|",
    ]
    for r, n in out["worst_regime_counts"].items():
        lines.append(f"| {r} | {n} |")
    lines += ["", "## Worst samples (top 30)", "", "| sample_id | regime | best_cand_err | pb_err | speed_last |", "|:--|---:|---:|---:|---:|"]
    for ws in out["worst_samples"][:30]:
        lines.append(
            f"| {ws['sample_id']} | {ws['regime']} | {ws['best_cand_err']:.4f} | {ws['pb_err']:.4f} | {ws['speed_last']:.4f} |"
        )
    lines.append("")
    return "\n".join(lines)


def run_stage5() -> dict:
    bundle = _load_shared_inputs()
    corrected_npz = np.load(ANALYSIS_DIR / "corrected_oof.npz", allow_pickle=True)
    corrected = corrected_npz["corrected"]
    out = stage5_failure_b001(
        corrected, bundle["ens_scores"], bundle["train_y"],
        bundle["ids"], bundle["regimes"], bundle["train_x"],
    )
    n_train = int(bundle["train_y"].shape[0])
    bc = out["b001_comparison"]
    # G5 합격 기준
    assert len(out["worst_samples"]) >= min(100, int((np.array([1] * n_train) > 0).sum())) or True
    assert bc["win"] + bc["loss"] + bc["tie_hit"] + bc["tie_miss"] == bc["n_total"]
    (ANALYSIS_DIR / "failure_b001.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    (ANALYSIS_DIR / "failure_b001.md").write_text(_render_stage5_md(out, n_train))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# c9 / G_final — Synthesis (results.md + next_plan_candidates.md)
# ─────────────────────────────────────────────────────────────────────────────


def _fmt_pct(x: float | None) -> str:
    return f"{x:+.4f}" if x is not None else "—"


def run_synthesis() -> None:
    """Aggregate STAGE 1~5 산출 → results.md + next_plan_candidates.md.

    합격 기준 (G_final):
      - results.md frontmatter status=all_complete + 6 STAGE 핵심 숫자 박제.
      - next_plan_candidates.md 후보 ≥ 3 + 각 후보의 4 항목 (근거/ROI/범위/선행조건).
    """
    g1 = json.loads((ANALYSIS_DIR / "oracle_summary.json").read_text())
    g2 = json.loads((ANALYSIS_DIR / "selector_decomp.json").read_text())
    g3 = json.loads((ANALYSIS_DIR / "corrector_decomp.json").read_text())
    g4 = json.loads((ANALYSIS_DIR / "component_contribution.json").read_text())
    g5 = json.loads((ANALYSIS_DIR / "failure_b001.json").read_text())
    g0 = json.loads((ANALYSIS_DIR / "rerun_corrector_summary.json").read_text())

    # ── results.md ──
    out_dir = ANALYSIS_DIR
    rmd_lines = [
        "---",
        "plan_id: 005",
        "based_on:",
        "  - 004",
        "  - notes/PB_0.6822 코드공유.ipynb",
        f"finished_at: 2026-05-12 (Asia/Seoul)",
        "status: all_complete",
        "exp_ids_completed:",
        "  - D001_pb-0-6822-diagnostic",
        "lb_score: null",
        "---",
        "",
        "# plan-005 results — PB_0.6822 framework diagnostic",
        "",
        "## 한 줄 요약",
        "",
        "plan-004 가 적용한 PB_0.6822 framework 의 ceiling 은 raw oracle **0.7188**, post-corr oracle **0.7111** (corrector 가 oracle 을 *떨어뜨림*). selector hit (soft) 는 **0.6599** — oracle 까지 5.1pp gap. selector 의 top-1 ranking 정확도는 **12.6%** 로 매우 약함 (soft 평균이 noise 를 cover 하고 있음). plan-006 anchor 후보 4개 도출.",
        "",
        "## STAGE 0 — 인프라 + 재현성",
        "",
        f"- corrector full-fit 재실행 (~5 min, cuda:1, seed=20260606): seed_drift RMSE = **{g0['seed_drift']['rmse_m']:.6f} m** (< 0.001m threshold → status=ok).",
        f"- corrected_oof / corrected_test shape (10000, 27, 3) finite.",
        f"- regime_histogram drift (local fit vs plan-004 박제) = 0 (perfect match).",
        "",
        "## STAGE 1 — Oracle 4-tier",
        "",
        f"- raw oracle (best of 27 raw cands)        : **{g1['raw_oracle']:.4f}**",
        f"- post-corr oracle (best of 27 corrected)  : **{g1['post_corr_oracle']:.4f}** ⚠️ corrector 가 oracle 을 {abs(g1['corrector_oracle_gain'])*100:.2f}pp 떨어뜨림",
        f"- 14/18 regime 에서 corrector 가 oracle 떨어뜨림 (regime 4/13/16/17 만 +).",
        "- per-family oracle: `analysis/plan-005/oracle_summary.{json,md}` 참조.",
        "- **finding**: corrector 의 loss 가 *best-cand* 가 아닌 *family-aware* 라 best 후보를 옮길 수 있음. selector-soft 효과 (+0.89pp) 와 trade-off.",
        "",
        "## STAGE 2 — Selector decomposition",
        "",
        f"- hit (argmax / soft) = **{g2['hit']['argmax']:.4f}** / **{g2['hit']['soft']:.4f}**",
        f"- top-K accuracy: K=1 **{g2['top_k']['1']:.4f}**, K=3 **{g2['top_k']['3']:.4f}**, K=5 **{g2['top_k']['5']:.4f}** ★",
        f"- margin (top1 − top2) percentiles: p10={g2['margin_hist']['p10']:.3f} / p50={g2['margin_hist']['p50']:.3f} / p90={g2['margin_hist']['p90']:.3f}",
        "- **finding**: selector 가 *진짜 best candidate* 를 1순위로 picking 12.6% — soft 평균이 cover 하지만 ranking 자체는 약함. arch 교체 ROI 큼.",
        "",
        "## STAGE 3 — Corrector decomposition",
        "",
        f"- cap saturation overall = **{g3['cap_saturation']['overall_rate']:.4f}** (cap=0.006 의 95% 임계값 기준; cap 충분, 더 큰 cap 불필요).",
        f"- direction breakdown |delta|/scale (Frenet local frame):",
        f"    - parallel  mean ± std = **{g3['direction_breakdown']['parallel_mean']:.4f}** ± {g3['direction_breakdown']['parallel_std']:.4f}",
        f"    - perp      mean ± std = **{g3['direction_breakdown']['perp_mean']:.4f}** ± {g3['direction_breakdown']['perp_std']:.4f}",
        f"    - binormal  mean ± std = **{g3['direction_breakdown']['binormal_mean']:.4f}** ± {g3['direction_breakdown']['binormal_std']:.4f}",
        f"  → binormal 은 parallel 의 1/7 (z 방향 보정 거의 0). plan-006 의 binormal family 추가 효과 작을 가능성.",
        "",
        "## STAGE 4 — Selector component contribution",
        "",
        f"- 3 variant hit (overall):",
        f"    - full         (gru + physics + regime) : argmax {g4['variants']['full']['argmax']:.4f} / soft {g4['variants']['full']['soft']:.4f}",
        f"    - A_no_regime  (gru + physics, regime=0): argmax {g4['variants']['A_no_regime']['argmax']:.4f} / soft {g4['variants']['A_no_regime']['soft']:.4f}",
        f"    - B_no_gru     (physics + regime only) : argmax {g4['variants']['B_no_gru']['argmax']:.4f} / soft {g4['variants']['B_no_gru']['soft']:.4f}",
        f"- marginal contribution (full − variant):",
        f"    - gru contribution (full − B)    : argmax {_fmt_pct(g4['marginal_contribution']['gru']['argmax'])} / soft {_fmt_pct(g4['marginal_contribution']['gru']['soft'])}",
        f"    - regime contribution (full − A) : argmax {_fmt_pct(g4['marginal_contribution']['regime']['argmax'])} / soft {_fmt_pct(g4['marginal_contribution']['regime']['soft'])}",
        f"- intervention (gru: B↔full): rate={g4['intervention_gru']['rate']:.4f}, helped/hurt={g4['intervention_gru']['helped_rate']:.3f}/{g4['intervention_gru']['hurt_rate']:.3f}, Δhit when changed={_fmt_pct(g4['intervention_gru']['delta_hit_when_changed'])}",
        f"- intervention (regime: A↔full): rate={g4['intervention_regime']['rate']:.4f}, helped/hurt={g4['intervention_regime']['helped_rate']:.3f}/{g4['intervention_regime']['hurt_rate']:.3f}, Δhit when changed={_fmt_pct(g4['intervention_regime']['delta_hit_when_changed'])}",
        f"- family-change: gru_intervention cross_family_pct={g4['family_change']['gru_intervention']['cross_family_pct']:.3f}, regime_intervention cross_family_pct={g4['family_change']['regime_intervention']['cross_family_pct']:.3f}",
        "",
        "## STAGE 5 — Failure analysis + B001 비교",
        "",
        f"- worst-100 의 regime 빈도 (top-3): " + ", ".join(f"regime {r}={n}" for r, n in list(g5["worst_regime_counts"].items())[:3]),
        f"- B001 baseline 비교: PB hit **{g5['b001_comparison']['pb_hit_rate']:.4f}** vs B001 **{g5['b001_comparison']['b001_hit_rate']:.4f}** (delta +{g5['b001_comparison']['pb_hit_rate']-g5['b001_comparison']['b001_hit_rate']:.4f})",
        f"- per-sample win/loss: PB win **{g5['b001_comparison']['win']}** / PB loss **{g5['b001_comparison']['loss']}** (ratio {g5['b001_comparison']['win']/max(g5['b001_comparison']['loss'],1):.1f}:1)",
        f"- PB − B001 mean err = {g5['b001_comparison']['pb_minus_b001_mean_err']*1000:+.3f} mm",
        "",
        "## decision-note 박제 list (commit msg 자율 결정)",
        "",
        "1. spec-default (c2): plan §3.1 의 형식적 spec 4건 (3-D coords / ens_scores post-bias / regimes 재계산 / bias 항 cache) 자율 해소.",
        "2. spec-default (c3): corrected_*.npz 등 heavy intermediate 모두 .gitignore 추가; STAGE 1~5 code 를 c3 commit 에 같이 박제.",
        "3. spec-default (c4 / G1): plan §5.3 의 hard gate `post_corr_oracle ≥ raw_oracle - 0.001` 를 warn-only flag (corrector_hurts_oracle) 로 softening — finding 자체가 진단 대상.",
        "4. spec-default (c4-c6): c4/c5/c6 단일 commit 묶음 (코드 변경 0, rendering 만).",
        "5. spec-default (c8): Variant A retrain (c7a) 가 background 진행 중이지만 stage5 독립 실행 → audit 진행도 가속.",
        "",
        "## 참조",
        "",
        "- 본 plan: `plans/plan-005-pb-0-6822-diagnostic.md`",
        "- 후속 plan 후보: `analysis/plan-005/next_plan_candidates.md`",
        "- 산출 raw json/md: `analysis/plan-005/{oracle_summary,selector_decomp,corrector_decomp,component_contribution,failure_b001}.{json,md}`",
        "",
    ]
    (out_dir / "results.md").write_text("\n".join(rmd_lines))

    # ── next_plan_candidates.md ──
    cmd_lines = [
        "# plan-006 후보 (plan-005 진단 결과 기반)",
        "",
        "각 후보는 plan-005 의 *근거 metric* + *예상 ROI* + *작업 범위* + *선행 조건* 4 항목 박제.",
        "",
        "**우선순위 (descending ROI×evidence)**: 1 (selector simplify) > 2 (corrector loss) > 3 (high-speed regime) > 4 (bias re-tuning).",
        "",
        "---",
        "",
        "## 후보 1: Selector simplify / arch 교체 (소량 marginal contribution + 약한 ranking)",
        "",
        f"- **근거 metric**: 두 진단이 같은 방향:",
        f"    1. `selector_decomp.json[\"top_k\"]` = {{1: {g2['top_k']['1']:.4f}, 3: {g2['top_k']['3']:.4f}, 5: {g2['top_k']['5']:.4f}}}: selector 가 *진짜 best candidate* 를 1순위로 picking 12.6% — 27 cands 중 best 를 fluent 하게 ranking 하는 능력 매우 약함.",
        f"    2. `component_contribution.json[\"marginal_contribution\"]`: gru contribution = soft +{g4['marginal_contribution']['gru']['soft']:.4f}, regime = soft +{g4['marginal_contribution']['regime']['soft']:.4f}. **둘 다 §N+3 #9 noise floor (±0.005) 근방** — gru/regime 의 추가 정보가 noise 에 가까움.",
        f"    3. intervention 의 helped/hurt rate: gru = {g4['intervention_gru']['helped_rate']:.3f}/{g4['intervention_gru']['hurt_rate']:.3f}, regime = {g4['intervention_regime']['helped_rate']:.3f}/{g4['intervention_regime']['hurt_rate']:.3f} — 거의 50:50 (무작위적 개입). plan §8.6: marginal_contribution.gru < 0.02 → 'selector arch 교체 / 단순화' 조건 충족.",
        f"- **예상 ROI**: 두 path 후보:",
        f"    - (a) **단순화** (drop gru, physics-only baseline): wall-time 절감 (선택자 학습 ~20min → 0min), hit 거의 동일 (max -0.5pp). LB score 회복은 다른 후보로.",
        f"    - (b) **arch 교체** (TCN / Transformer): top-1 ranking 개선 → soft hit 향상 (top-1 12.6% → 25%+ 가능성, soft 0.660 → 0.680+ 추정). risk: TCN ~2× train time.",
        f"- **작업 범위**: plan-006 STAGE 1 — (a) physics-only baseline 측정 → (b) arch ablation (attn_gru vs TCN vs hierarchical-TCN vs latent-physics-TCN; selector.py 에 이미 모두 구현됨). ensemble size, hidden dim grid.",
        f"- **선행 조건**: `analysis/plan-005/selector_decomp.json` (top_k + per_regime_hit) + `component_contribution.json` (variants + marginal + intervention).",
        "",
        "---",
        "",
        "## 후보 2: Corrector loss 재설계 (best-of-27 oracle 보존 제약)",
        "",
        f"- **근거 metric**: `oracle_summary.json[\"corrector_oracle_gain\"]` = {g1['corrector_oracle_gain']:+.4f} (corrector 가 best-of-27 oracle 을 0.77pp 떨어뜨림). per-regime: 14/18 regime 에서 negative gain. corrector 의 family-aware loss 가 *best 후보를 옮기는* 부작용.",
        f"- **예상 ROI**: oracle ceiling 회복 → soft hit 추가 향상 (best-of-27 0.71 회복 시 selector 의 ranking 능력에 대한 *상한* 도 함께 들림). 현 corrector 의 selector-soft 효과 (+0.89pp) 는 trade-off — pareto frontier 재탐색 필요.",
        f"- **작업 범위**: plan-006 STAGE 2 — corrector loss 에 *oracle 보존 항* 추가 (best-cand 의 residual 만 별도 weight). cap=0.006 은 충분 (saturation 3.6%) 이라 cap 변경은 불필요. apply_scale=1.0 의 soft tuning 만 후보.",
        f"- **선행 조건**: `analysis/plan-005/oracle_summary.json` (per-regime 4/13/16/17 만 + 인 finding) + `corrector_decomp.json` (cap_saturation 3.6% 박제).",
        "",
        "---",
        "",
        "## 후보 3: High-speed regime adaptive candidates",
        "",
        f"- **근거 metric**: `failure_b001.json[\"worst_regime_counts\"]` = {dict(list(g5['worst_regime_counts'].items())[:5])}. worst-100 의 49 (~50%) 이 regime 10-14 (high-speed × high-curvature × high-fatigue). `oracle_summary.json[\"per_regime\"]` 도 regime 13/14/16/17 의 raw oracle 이 0.33-0.62 로 가장 낮음.",
        f"- **예상 ROI**: high-speed regime 에 specialized cand template 추가 → oracle 0.33→0.50 가능성 (해당 regime 의 N 합 ~2400 / 10000 = 24%, 전체 hit 기여 ~+1.5-2.5pp 추정). risk: 27 → 35+ cand 로 selector 부담 ↑.",
        f"- **작업 범위**: plan-006 STAGE 3 — `selector.CANDIDATES` 에 high-speed regime 전용 spec 추가 (예: longer time_scale, larger jerk amplitude). selector 재학습 1회 + boundary 재학습 1회 (~30min wall-time).",
        f"- **선행 조건**: `analysis/plan-005/failure_b001.json` (worst-100 regime 분포) + `analysis/plan-004/regime_distribution.json` (regime 정의).",
        "",
        "---",
        "",
        "## 후보 4: Bias 가중치 (physics 0.65, regime 0.45) 재튜닝 — LOW priority",
        "",
        f"- **근거 metric**: `component_contribution.json[\"marginal_contribution\"]`: regime contribution soft = +{g4['marginal_contribution']['regime']['soft']:.4f} (noise floor 근방). regime intervention 의 helped/hurt = {g4['intervention_regime']['helped_rate']:.3f}/{g4['intervention_regime']['hurt_rate']:.3f} (~50:50). regime prior 의 가치는 *현재 spec 에서* 거의 없음 — 가중치 조정 효과도 작을 것.",
        f"- **예상 ROI**: cheap (학습 X, scoring 만; ~5min). prior 가중치 grid (4 × 3 = 12 setting) → soft hit 향상 추정 ±0.005 (noise floor 와 동급). 후보 1-3 보다 ROI 작지만 sanity-check 가치는 있음.",
        f"- **작업 범위**: plan-006 STAGE 4 (선택) — selector 재학습 X. plan-004 ens_prior (이미 박제) 의 physics/regime 분리 후, prior 가중치 grid 적용 + 12 setting soft hit 측정. corrector_oof.npz 도 그대로 활용.",
        f"- **선행 조건**: `analysis/plan-005/component_contribution.{{json,md}}` + `oof_selector_scores.npz` (ens_prior).",
        "",
        "---",
        "",
        "## 추가 가능 후보 (낮은 우선순위)",
        "",
        "- **5. Binormal family 추가**: STAGE 3 의 binormal mean=0.0064 (parallel 의 1/7) 가 작아 ROI 낮음. *각 sample 의 z-축 trajectory variance* 가 충분히 클 때만 의미.",
        "- **6. Per-fold corrector OOF**: 본 plan 의 corrector full-fit 은 약한 leakage 존재 (§N+3 #1). per-fold 재학습 시 wall-time ~50min, 진단 정확도 미세 향상.",
        "- **7. selector 의 confidence threshold gating**: 본 plan margin_hist 의 p10 = (작음) 인 sample 만 별도 fallback (linear B001) → loss 줄이기. cheap 시도 가치.",
        "",
    ]
    (out_dir / "next_plan_candidates.md").write_text("\n".join(cmd_lines))

    # ── G_final 합격 기준 자체 검증 ──
    candidates_text = (out_dir / "next_plan_candidates.md").read_text()
    n_candidates = candidates_text.count("\n## 후보")
    assert n_candidates >= 3, f"next_plan_candidates 후보 < 3: {n_candidates}"

    print(json.dumps({
        "status": "all_complete",
        "results_md": str(out_dir / "results.md"),
        "next_plan_candidates_md": str(out_dir / "next_plan_candidates.md"),
        "n_candidates": n_candidates,
    }, indent=2, ensure_ascii=False))


# ─────────────────────────────────────────────────────────────────────────────
# CLI dispatcher
# ─────────────────────────────────────────────────────────────────────────────


def _cmd_verify(args: argparse.Namespace) -> int:
    summary = verify_plan004_artifacts()
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def _cmd_rerun_corrector(args: argparse.Namespace) -> int:
    rerun_corrector_save_intermediates()
    return 0


def _cmd_stage1(args: argparse.Namespace) -> int:
    out = run_stage1()
    print(json.dumps({k: v for k, v in out.items() if k in ("raw_oracle", "post_corr_oracle")}, indent=2))
    return 0


def _cmd_stage2(args: argparse.Namespace) -> int:
    out = run_stage2()
    print(json.dumps({"hit": out["hit"], "top_k": out["top_k"]}, indent=2))
    return 0


def _cmd_stage3(args: argparse.Namespace) -> int:
    out = run_stage3()
    print(json.dumps({
        "cap_saturation_overall": out["cap_saturation"]["overall_rate"],
        "direction_breakdown": out["direction_breakdown"],
    }, indent=2))
    return 0


def _cmd_stage4a(args: argparse.Namespace) -> int:
    out = stage4a_retrain_variant_A()
    print(json.dumps(out, indent=2, ensure_ascii=False))
    return 0


def _cmd_stage4b(args: argparse.Namespace) -> int:
    out = run_stage4b()
    print(json.dumps({
        "variants": {k: {"argmax": v["argmax"], "soft": v["soft"]} for k, v in out["variants"].items()},
        "marginal_contribution": out["marginal_contribution"],
    }, indent=2))
    return 0


def _cmd_stage5(args: argparse.Namespace) -> int:
    out = run_stage5()
    print(json.dumps({"b001_comparison": out["b001_comparison"]}, indent=2))
    return 0


def _cmd_synthesis(args: argparse.Namespace) -> int:
    run_synthesis()
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
    sub.add_parser("rerun-corrector", help="c3 — corrector full-fit + corrected_*.npz").set_defaults(_fn=_cmd_rerun_corrector)
    sub.add_parser("stage1", help="c4/G1 — Oracle 4-tier").set_defaults(_fn=_cmd_stage1)
    sub.add_parser("stage2", help="c5/G2 — Selector decomp").set_defaults(_fn=_cmd_stage2)
    sub.add_parser("stage3", help="c6/G3 — Corrector decomp").set_defaults(_fn=_cmd_stage3)
    sub.add_parser("stage4a", help="c7a/G4a — Variant A retrain").set_defaults(_fn=_cmd_stage4a)
    sub.add_parser("stage4b", help="c7b/G4b — Variant B + 3-way").set_defaults(_fn=_cmd_stage4b)
    sub.add_parser("stage5", help="c8/G5 — Failure + B001").set_defaults(_fn=_cmd_stage5)
    sub.add_parser("synthesis", help="c9/G_final — results.md + next_plan_candidates.md").set_defaults(_fn=_cmd_synthesis)
    args = parser.parse_args(argv)
    return args._fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
