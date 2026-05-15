---
plan_id: 008
based_on: [004, 005, 006, 007]
status: partial (carry-over)
exp_ids:
  - G001-sanity-27
  - G001-candidate-redefine
  - G002-corrector-band  # DEFERRED
lb_score: null  # TBD — quota_exhausted_2026-05-12, deferred to plan-008.1
lb_submitted_at: null
finished_at: 2026-05-12T21:05+09:00
severe_flags:
  - redefinition_severely_insufficient (G1)
  - selector_no_improvement (G2)
warn_flags:
  - diagnostic_inconclusive (G0)
  - sanity_baseline_drift (G2)
  - family_effect_marginal (G2)
deferred:
  - G002-corrector-band (G3) → plan-009 (boundary.py LOSS_ATTR hook 신설 필요)
  - LB 회수 → plan-008.1 (할당량 소진)
---

# plan-008 — Results pair (1-줄 요약)

> **Result**: Candidate pool 27 → 25 (12 base_kept + 4 templates greedy add), oracle 0.7188 → 0.7543 (target 0.85 의 30% 도달, SEVERE), selector OOF 0.6503 (Variant A 0.6570 보다 −0.007, SEVERE), family_effect +0.0037 (marginal). G3 corrector deferred (boundary.py hook 부재 → plan-009). LB carry-over.

## 핵심 finding

- **main_bottleneck = "ranking"** (diag c2 확정, gap_ranking 0.0516 ≫ gap_drift −0.0004).
- 후보 풀 확장은 oracle 천장 +0.04 회수만, selector 가 it 을 hit 으로 follow 불가.
- **plan-008 본질적 결론**: ranking 능력 한계 (caveat #13) → plan-009 main task = selector arch 교체.

## 산출

- `analysis/plan-008/diagnostic.{py,json,md}` (c2, G0)
- `analysis/plan-008/prune_and_redefine.py` + `prune_summary.json` + `greedy_set_cover.json` + `redefine.md` (c4+c5, G1)
- `analysis/plan-008/sanity_baseline_27.{py,json}` (c5.5)
- `analysis/plan-008/selector_retrain.{py,json}` (c6+c7, G2)
- `analysis/plan-008/corrector_band.{py,json}` (c9, G3 DEFERRED)
- `analysis/plan-008/results.md` + `next_plan_candidates.md` (c14, G_final)
- `runs/baseline/G001_sanity-27/*` + `runs/baseline/G001_candidate-redefine/*` (5 submission csv variants)
- `src/pb_0_6822/candidates_extended.py` (5 family, 15 templates)
- `src/pb_0_6822/selector.py` (CandidateSpec schema v2.2 — 4 신규 fields, partial)

## Commit chain (all DONE)

- ebd4979 c2+G0, d2a105f sync, 89f3b3f c2.5, b5845a5 sync
- 9e8b61f c3, b22f86c c4+c5+G1 (SEVERE), 8ca7abd sync
- 637f7e2 c5.5+c6, 48adfed c6 boundary, 62b6344 c6 recursion fix
- 1a8c05c c7+G2 (SEVERE), 5215c64 sync
- 4277a21 c9+G3 (DEFERRED)
- (이 commit) c14 + G_final

## plan-009 carry-over tasks

1. **selector arch 교체** (TCN / Transformer / MLP coeff) — main task
2. **boundary.py compute_corrector_loss hook 신설** + band-specific monkey-patch (plan-008 §7 carry-over)
3. (선택) test_internal + plan-007 MLP coeff 재시도

## plan-008.1 carry-over

- LB submit `runs/baseline/G001_candidate-redefine/submission_step3.csv` (1 회)
- 결과 박제: lb_score → 3 파일 (frontmatter top + results.md + analysis/plan-008/results.md)
