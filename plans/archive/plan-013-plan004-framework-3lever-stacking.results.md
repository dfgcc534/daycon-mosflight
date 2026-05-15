---
plan_id: 013
plan_title: plan-004 Framework + 3 Lever Stacking (measured-only)
status: G_final_complete (warn-recovered — G0 partial + G1 warn + G2 severe-recovered + G3 warn; LB carry-over plan-013.1)
date_completed: 2026-05-14 (Asia/Seoul)
date_started: 2026-05-14 (Asia/Seoul, plan-013 v1 작성 → c1~c9 + G_final 동일 일자 자율 진행)
exp_ids:
  - H030_phase0-preflight-relived
  - H031_phase1-baseline-plan004-inIC
  - H032_phase2-step4-F0           # DEFERRED
  - H033_phase2-step4-27ext        # DEFERRED
  - H034_phase2-25cand-redesign    # DEFERRED
  - H035_phase3-best-stack-5fold
final_oof_5fold_hit_1cm: 0.6381   # G3 fallback (= G1, lever 0 stack)
final_g1_baseline_oof: 0.6381
final_submission: analysis/plan-013/submission.csv  # 10000 rows, 5-fold mean ensemble
lb_score: null  # plan-013.1 carry-over (사용자 manual dacon-submit)
followed_by:
  - 013.1 (LB carry-over; user manual dacon-submit)
  - 014 (조건부 framework, analysis/plan-013/next_plan_candidates.md ≥ 3 후보, default = B1 plan-004 boundary corrector_cls 확장)
key_findings:
  - "G0 PARTIAL (3/4): plan_004_reproduce drift=0.0020 ✓ / in_ic_infra ✓ / step4_infra ✓ / cand_25_infra MISS"
  - "G1 WARN: simplified pipeline (plan-004 full framework 제외) penalty 로 0.6381 < 0.65 threshold"
  - "G2 FAIL: Phase 2 3 sub-exp 모두 DEFERRED (framework gap × 2 + cand_set MISS × 1) → phase2_no_positive_lever autonomous recovery"
  - "G3 WARN: fallback path = best G1 baseline 단독 5-fold + submission (OOF 0.6381, lever 0 stack 이라 super-additive 불가)"
  - "frozen_gru_drift safety invariant = 5 folds 모두 PASS (In/IC encoder state_dict 변경 0)"
  - "측정된 단일 fact: P001 selector 5-fold soft 0.6511 reproduce ≤ 0.005 drift, plan-004 framework 5-fold OOF anchor 박제"
carry_over_to_plan_013_1:
  - "best Phase submission `analysis/plan-013/submission.csv` 의 LB 수동 회수 (dacon-submit skill)"
  - "LB 회수 후 plan-013.md + plan-013.results.md + registry.csv H035 row 의 lb_score sync"
  - "LB 결과로 plan-014 분기 결정 (next_plan_candidates.md 의 A/B/C path)"
architectural_gap_summary:
  - "simplified pipeline penalty: plan-004 boundary.py 의 regime/env/pretrain/finetune 제외 → 회수 = boundary.py train_net 시그너처 확장 (corrector_cls arg)"
  - "plan-007 basis_terms framework gap: compute_trajectory_features + per-var basis_terms 통합 필요"
  - "plan-008 G1 cand_set 박제 부재: G001 디렉토리에 candidate descriptor list 별도 박제 X"
---

전체 결과 요약은 `analysis/plan-013/results.md` 참조.
plan-014 후보 (조건부 framework) 는 `analysis/plan-013/next_plan_candidates.md` 참조.
