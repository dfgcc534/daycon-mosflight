---
plan_id: 012
plan_title: Codebook Bake-off Classification + Regression Hybrid (paradigm reframe, 3D)
status: G_final_complete_INVALID_REFERENCE
disclaimer: 본 plan 의 모든 측정값은 plan-004 `CandidateAttentionGRUSelector` + plan-006 numpy `frenet_par120_perp_neg020` 의 *재사용 위* 에서 측정됨. plan-014 진단 (재사용 강박 = single root cause) 따르면 신뢰성 없음 — **타 plan 의 design choice / codebook geometry / target band / ablation lever 결정 의 reference 사용 금지**. historical record 로만 보존.
date_completed: 2026-05-14 (Asia/Seoul; GPU spec-faithful re-run)
date_started: 2026-05-13 (Asia/Seoul, plan-012 v2 spec replacement → c2~c16 + G_final 동일 일자 자율 진행)
exp_ids:
  - H019_phase0-preflight-codebook
  - H020_phase1-codebook-bakeoff
  - H022_phase2-codebook-K
  - H023_phase2-temperature
  - H024_phase2-loss
  - H025_phase2-reg-head
  - H026_phase3-boundary-weight
  - H027_phase3-scorer-arch
  - H028_phase3-r0-prior
  - H029_phase4-final-5fold
final_oof_5fold_hit_1cm: 0.6350  # GPU best stack — INVALID for reference (코드 재사용 환경 위 측정)
final_oof_5fold_anchor_baseline: 0.6344  # GPU anchor — INVALID
final_submission: runs/baseline/H029_phase4-final-5fold/submission_anchor_fallback.csv
lb_score: null
followed_by:
  - 012.1 (LB carry-over)
  - 013 (paradigm 폐기 + plan-004 framework 회귀)
  - 014 (paradigm 부활: 재사용 끊고 from-scratch 재실험)
---

# plan-012 — Results (⚠️ INVALID FOR REFERENCE)

> ## ⚠️ Disclaimer — 본 결과는 코드 재사용 환경 위 측정, 다른 plan 의 reference 로 사용 금지
>
> 본 plan 의 모든 측정값 — **codebook bake-off winner (E0a Absolute), 8 ablation 의 ΔOOF, 5-fold OOF 0.6350 plateau, "paradigm 자체 limit" 결론** — 은 plan-004 `CandidateAttentionGRUSelector` + plan-006 numpy `frenet_par120_perp_neg020` 의 *재사용 위* 에서 측정됨.
>
> **plan-014 진단** (`plan-014-plan012-failure-inversion.md` §1.2 / §1.3): 재사용 강박 = single root cause → 본 plan 의 ablation 이 다 marginal 인 것은 *원인이 아닌 증상 치료*. "paradigm 자체 limit (0.6350)" 결론도 *재사용 환경 위 결론* 이므로 paradigm 의 *진짜 잠재력* 에 대한 판단 근거 아님.
>
> → 본 결과의 *어떤 측정값도* 후속 plan 의 **design choice / codebook geometry / target band / ablation lever** 결정 의 reference 로 사용 금지. paradigm 의 진짜 잠재력 측정 = **plan-014 G3** (5-fold OOF band 분류).
>
> historical record 로만 보존. 아래 outcome 박제 = *재사용 환경 위 마지노선* 의미만.

## Historical outcome (= 재사용 환경 위 측정, INVALID for reference)

- 5-fold OOF hit@1cm = **0.6350** (GPU best stack, CPU 0.6340)
- Phase 1 codebook bake-off winner = **E0a Absolute** (tie-break, gap < 0.005 — 3-way 모두 marginal)
- 8 ablation 합쳐도 +0.005 미만 마진 (max single lever = +0.0015 GPU)
- "paradigm reframe (codebook + classifier + regression hybrid) 은 F0 raw hit 위 +0.002~0.003 만 추가" 결론 박제 — 단 *재사용 환경 위 결론*
- LB submission = `submission_anchor_fallback.csv` (manual submit pending)

## 후속 plan (분기)

- **plan-013**: paradigm 폐기 → plan-004 framework 회귀 (Candidate C corrector + hybrid). G1 fallback 0.6381.
- **plan-014**: paradigm 부활 → 재사용 끊고 from-scratch 재실험 (premise: 재사용 강박이 root cause). 진행 중.

## carry-over

plan-012.1 — 사용자 manual `dacon-submit` (skill) 후 lb_score 박제 (plan-013/014 분기 trigger).
