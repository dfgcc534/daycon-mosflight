"""plan-008 c6+c7: Selector 재학습 (Variant A path) — extended pool with schema v2.2.

spec @ plans/plan-008-candidate-redefine-corrector-redesign.md §6.

Inputs:
  - analysis/plan-008/prune_summary.json   (Step 2a: kept_indices)
  - analysis/plan-008/greedy_set_cover.json (Step 2b: pool_specs_final → kept_families)
Monkey-patch:
  - selector.CANDIDATES        = EXTENDED_CANDIDATES (= base_kept + new family specs)
  - selector.make_candidates   = wrapper around candidates_extended.make_candidates_extended

Outputs:
  - runs/baseline/G001_candidate-redefine/oof_selector_scores.npz
  - runs/baseline/G001_candidate-redefine/test_selector_scores.npz
  - runs/baseline/G001_candidate-redefine/submission_step3.csv (생성만, LB 미제출)
  - analysis/plan-008/selector_retrain.json (G2 metrics)
"""
from __future__ import annotations

import copy
import inspect
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO / "data"
RUN_DIR = REPO / "runs/baseline/G001_candidate-redefine"
ANALYSIS_DIR = REPO / "analysis/plan-008"
SANITY_JSON = ANALYSIS_DIR / "sanity_baseline_27.json"

R_HIT = 0.01

_FAMILY_ID_NAME = {1: "trig", 2: "arc", 3: "frenet_serret_3d", 5: "higher_order", 6: "cross_term"}


def _call_main(main_func, argv: list) -> None:
    old = sys.argv[:]
    try:
        sys.argv = [main_func.__name__, *[str(a) for a in argv]]
        start = time.time()
        main_func()
        print(f"[DONE] {main_func.__name__} elapsed={time.time() - start:.1f}s", flush=True)
    finally:
        sys.argv = old


def _verify_variant_a(run_dir: Path) -> dict:
    """§6.2 v2.6 강화 — 학습 후 regime_bias_table 분산 < 1e-10 검증."""
    z = np.load(run_dir / "oof_selector_scores.npz")
    info: dict = {"keys": list(z.files)}
    if "regime_bias_table" in z.files:
        rbt = z["regime_bias_table"]
        rbt_var = float(np.var(rbt))
        info["regime_bias_table_shape"] = list(rbt.shape)
        info["regime_bias_table_var"] = rbt_var
        # Variant A path 강제 → regime bias 영향 zero
        assert rbt_var < 1e-10, (
            f"regime_residue: regime_bias_table var {rbt_var} >= 1e-10 (Variant A 위배)"
        )
        info["variant_a_safe"] = True
    else:
        info["regime_bias_table"] = "(absent — Variant A path enforced)"
        info["variant_a_safe"] = True
    return info


def main() -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load Step 2 산출 ──
    prune = json.loads((ANALYSIS_DIR / "prune_summary.json").read_text())
    greedy = json.loads((ANALYSIS_DIR / "greedy_set_cover.json").read_text())
    KEPT_INDICES = prune["kept_indices"]
    KEPT_FAMILIES = sorted({
        _FAMILY_ID_NAME[s["family_id"]]
        for s in greedy["pool_specs_final"]
        if s["family_id"] in _FAMILY_ID_NAME
    })
    print(
        f"[c6] kept_indices ({len(KEPT_INDICES)}): {KEPT_INDICES}\n"
        f"[c6] kept_families ({len(KEPT_FAMILIES)}): {KEPT_FAMILIES}"
    )

    from src.pb_0_6822 import selector
    from src.pb_0_6822 import candidates_extended as cx

    # ── Backup ORIGINAL_27 (§6.0 도 동일) ──
    ORIGINAL_27_CANDIDATES = copy.deepcopy(selector.CANDIDATES)
    ORIGINAL_make_candidates = selector.make_candidates

    # ── EXTENDED_CANDIDATES (CandidateSpec list) ──
    EXTENDED_CANDIDATES = cx.get_extended_candidates_list(KEPT_INDICES, KEPT_FAMILIES)
    print(
        f"[c6] EXTENDED_CANDIDATES len = {len(EXTENDED_CANDIDATES)} "
        f"(= base_kept {len(KEPT_INDICES)} + new {len(EXTENDED_CANDIDATES) - len(KEPT_INDICES)})"
    )

    # §6.1.5 schema_v22 assert (pre-training)
    sample_spec = selector.CandidateSpec("dummy", 1.0, 0.0, 0.0)
    assert hasattr(sample_spec, "omega_scale"), "schema v2.2 미적용"
    assert hasattr(sample_spec, "arc_curvature"), "schema v2.2 미적용"
    assert hasattr(sample_spec, "z_scale"), "schema v2.2 미적용"
    assert hasattr(sample_spec, "family_id"), "schema v2.2 미적용"

    # ── Monkey-patch (Variant A path) ──
    selector.CANDIDATES = EXTENDED_CANDIDATES

    def _patched_make_candidates(x, end_idx, horizon=2):
        # v2.7 fix: `selector.make_candidates` 가 module-level `CANDIDATES` 를 iterate.
        # 우리가 CANDIDATES = EXTENDED 로 patch 했으므로, base 27 을 다시 얻으려면
        # 일시적으로 CANDIDATES 를 ORIGINAL 로 swap 후 호출해야 함.
        saved = selector.CANDIDATES
        selector.CANDIDATES = ORIGINAL_27_CANDIDATES
        try:
            cands_base_27 = ORIGINAL_make_candidates(x, end_idx, horizon)
        finally:
            selector.CANDIDATES = saved
        cands_base_kept = cands_base_27[:, KEPT_INDICES, :]
        new_cands_list = [cands_base_kept]
        for fam in ("trig", "arc", "frenet_serret_3d", "higher_order", "cross_term"):
            if fam in KEPT_FAMILIES:
                new_cands_list.append(cx.FAMILY_TO_MAKE[fam](x, end_idx, horizon))
        return np.concatenate(new_cands_list, axis=1).astype(np.float32)

    selector.make_candidates = _patched_make_candidates

    # §6.1.5 (2) base family count check
    base_specs = [c for c in selector.CANDIDATES if c.family_id == 0]
    assert len(base_specs) >= 1, "base family (family_id=0) 후보 1개 이상 필요"
    for c in base_specs:
        assert c.omega_scale == 0.0
        assert c.arc_curvature == 0.0

    # §6.1.5 (3) cand_feat dim verification on tiny sample
    train_ids, train_y = selector.read_labels(DATA_ROOT / "train_labels.csv")
    train_x_head = selector.load_stack(DATA_ROOT / "train", train_ids[:5])
    end_idx = train_x_head.shape[1] - 1
    test_cands = selector.make_candidates(train_x_head, end_idx, horizon=2)
    test_feat = selector.make_candidate_features(
        train_x_head, end_idx, test_cands, horizon=2,
        candidates_list=EXTENDED_CANDIDATES,
    )
    expected_dim = 3 + 16 + 9 + 4   # par/perp/dist + spec_v22 + ctx + interactions_v22 = 32
    assert test_feat.shape[2] == expected_dim, (
        f"cand_dim mismatch: {test_feat.shape[2]} != {expected_dim}"
    )
    print(f"[c6] schema_v22 assert OK: cand_feat shape = {test_feat.shape}")

    # ── Train (§6.1 Variant A hyperparam) ──
    cli_args = [
        "--root", DATA_ROOT,
        "--out-dir", RUN_DIR,
        "--models", "attn_gru",
        "--folds", 5, "--fold-limit", 5,
        "--regime-prior-strength", 0,   # ⭐ Variant A
        "--pre-epochs", 10, "--fine-epochs", 8, "--freeze-fine-epochs", 3,
        "--epoch-plus", 5, "--patience", 4,
        "--hidden", 48, "--batch", 4096,
        "--lr", 0.001, "--fine-lr-scale", 0.12,
        "--prior-strength", 0.65,
        "--pairwise-loss-weight", 0.25, "--pairwise-margin", 0.12, "--pairwise-min-label-gap", 0.04,
        "--fine-distill-weight", 0.55, "--fine-distill-temp", 0.07,
        "--reverse-pretrain", "--norm-real-only",
        "--device", "cuda:1", "--seed", 20260506, "--log-every", 1,
        # full-fit 도 수행 (test inference 위해)
    ]
    # §6.2 v2.6 학습 *전* assert: regime-prior-strength=0 in argv
    assert "--regime-prior-strength" in [str(a) for a in cli_args]
    rps_idx = [str(a) for a in cli_args].index("--regime-prior-strength")
    assert str(cli_args[rps_idx + 1]) == "0", "regime_residue: regime_prior_strength != 0"

    _call_main(selector.SELECTOR_MAIN, cli_args)

    # ── Variant A path 검증 (§6.2 v2.6 학습 *후*) ──
    variant_a_check = _verify_variant_a(RUN_DIR)
    print(f"[c6] variant_a_check: {variant_a_check}")

    # ── 기존 corrector full-fit (plan-004 default) + submission generation ──
    # §6.1: "corrector 는 기존 (plan-004) 그대로 적용 (Step 3 단계, Step 4 에서 재설계)"
    score_bank = RUN_DIR / "oof_selector_scores.npz"
    test_score_bank = RUN_DIR / "test_selector_scores.npz"
    assert score_bank.exists() and test_score_bank.exists(), (
        f"selector full-fit 결과 누락 — {score_bank}, {test_score_bank}"
    )
    from src.pb_0_6822 import boundary
    _call_main(boundary.BOUNDARY_MAIN, [
        "--root", DATA_ROOT,
        "--out-dir", RUN_DIR,
        "--fold", 0, "--folds", 5,
        "--score-bank", score_bank,
        "--test-score-bank", test_score_bank,
        "--epochs", 12, "--fine-epochs", 8, "--min-epochs", 5, "--patience", 4,
        "--hidden", 64, "--batch", 8192,
        "--lr", 0.001, "--fine-lr-scale", 0.18,
        "--cap", 0.006, "--apply-scale", 1.0,
        "--device", "cuda:1", "--seed", 20260606, "--save-val-pred",
        "--make-test",
    ])

    # Copy soft submission to step3 canonical name
    soft_csv = RUN_DIR / "submission_boundary_tiny_soft.csv"
    step3_csv = RUN_DIR / "submission_step3.csv"
    if soft_csv.exists():
        step3_csv.write_bytes(soft_csv.read_bytes())
        print(f"[c6] submission_step3.csv ← {soft_csv.name}")

    # ── OOF metrics + submission (Step 4 분리: corrector 미실행, 기존 corrector
    #     full-fit 은 §7.3 monkey-patch 와 별도 — 본 단계는 selector OOF만) ──
    score_path = RUN_DIR / "oof_selector_scores.npz"
    z = np.load(score_path)
    oof_scores = z["ens_scores"]
    n_cand = int(oof_scores.shape[1])

    train_x = selector.load_stack(DATA_ROOT / "train", train_ids)
    end_idx_full = train_x.shape[1] - 1
    cands_full = selector.make_candidates(train_x, end_idx_full, horizon=2)
    assert cands_full.shape[1] == n_cand, (
        f"oof score n_cand {n_cand} mismatch with extended pool {cands_full.shape[1]}"
    )

    argmax_idx = oof_scores.argmax(axis=1)
    argmax_pred = cands_full[np.arange(len(train_y)), argmax_idx]
    argmax_hit = float((np.linalg.norm(argmax_pred - train_y, axis=1) <= R_HIT).mean())
    soft_pred = selector.soft_select(cands_full, oof_scores, temperature=0.03)
    soft_hit = float((np.linalg.norm(soft_pred - train_y, axis=1) <= R_HIT).mean())

    err = np.linalg.norm(cands_full - train_y[:, None, :], axis=2)
    oracle = float((err.min(axis=1) <= R_HIT).mean())
    best_idx = err.argmin(axis=1)
    top1_acc = float((argmax_idx == best_idx).mean())
    gap_ranking = oracle - argmax_hit

    # Family effect computation (against sanity baseline)
    family_effect = None
    sanity_band_ok = None
    if SANITY_JSON.exists():
        sanity = json.loads(SANITY_JSON.read_text())
        sanity_soft = sanity["sanity_baseline_27_oof_soft"]
        family_effect = soft_hit - sanity_soft
        sanity_band_ok = sanity.get("in_baseline_band", None)
        print(
            f"[c6] family_effect = {soft_hit:.4f} − {sanity_soft:.4f} = {family_effect:+.4f}"
        )
    else:
        print(f"[c6] WARN: sanity_baseline_27.json missing — family_effect skipped")

    # G2 합격 판정
    g2_oof_pass = soft_hit >= 0.70
    g2_family_effect_pass = (family_effect is not None) and (family_effect >= 0.03)
    g2_family_effect_marginal = (family_effect is not None) and (family_effect < 0.02)
    g2_ranking_gap_pass = gap_ranking <= 0.07

    summary = {
        "exp_id": "G001-candidate-redefine",
        "n_train": int(len(train_y)),
        "n_candidates": n_cand,
        "kept_indices": KEPT_INDICES,
        "kept_families": KEPT_FAMILIES,
        "oof_argmax_hit": argmax_hit,
        "oof_soft_hit": soft_hit,
        "oracle_extended_pool": oracle,
        "top1_ranking_accuracy": top1_acc,
        "gap_ranking": gap_ranking,
        "family_effect": family_effect,
        "sanity_baseline_in_band": sanity_band_ok,
        "variant_a_check": variant_a_check,
        "g2_pass": {
            "oof_geq_0.70": g2_oof_pass,
            "family_effect_geq_0.03": g2_family_effect_pass,
            "family_effect_marginal_lt_0.02": g2_family_effect_marginal,
            "ranking_gap_leq_0.07": g2_ranking_gap_pass,
        },
    }
    (ANALYSIS_DIR / "selector_retrain.json").write_text(json.dumps(summary, indent=2))
    print(
        f"[c7] OOF: argmax={argmax_hit:.4f} soft={soft_hit:.4f} "
        f"oracle={oracle:.4f} gap_ranking={gap_ranking:.4f} "
        f"family_effect={family_effect if family_effect is None else f'{family_effect:+.4f}'}"
    )
    # Restore monkey-patch (idempotent for subsequent imports)
    selector.CANDIDATES = ORIGINAL_27_CANDIDATES
    selector.make_candidates = ORIGINAL_make_candidates


if __name__ == "__main__":
    main()
