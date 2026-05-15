---
plan_id: 006
exp_id: E001_minimal-variant-e
lb_exp_id: E001_minimal-variant-e
lb_score: 0.6692
lb_submitted_at: 2026-05-12T11:13:37+09:00
lb_recovered_at: 2026-05-12T11:26:49+09:00
status: all_complete
date: 2026-05-12 (Asia/Seoul)
---

# plan-006 results — Minimal Variant E LB Validation (physics_bias + soft averaging only)

본 plan = plan-005 의 통찰 "PB framework 의 95% 가 장식이고 진짜 엔진은 *27 후보 + physics_bias + soft averaging* 3 ingredients 뿐" 을 *1 LB 제출* 로 직접 검증하는 cheap experiment.

**LB carry-over closed (c5.1)**: dacon-submit 응답 `{isSubmitted: True, detail: 'Success'}` (2026-05-12T11:13:37+09:00) → 사용자 회수 LB = **0.6692** (2026-05-12T11:26:49+09:00). plan §0 명제 1 (`lb_score ≥ 0.6606`, inclusive) ✓ → **시나리오 A — plan-005 통찰 입증**.

상세 분석: `analysis/plan-006/results.md` 참조.

## §1. Exp summary

| field | value |
|---|---|
| exp_id | `E001_minimal-variant-e` |
| plan_id | 006 |
| based_on | plan-004 (full framework) + plan-005 (component decomposition) |
| compute | local CPU + plan-005 `corrected_*.npz` 재사용 (재학습 0) |
| wall_time | < 1 min (analysis-only) |
| Variant 정의 | `score[i, c] = 0.65 × physics_bias[c]`, soft averaging temp=0.03, GRU 제거, regime 제거 |

## §2. 핵심 수치

| Metric | Value | 비교 |
|---|---|---|
| **E_corrected.soft (OOF)** | **0.6524** | plan-005 추정 0.6517 → 추정 정확도 7bp |
| E_corrected.argmax (OOF) | 0.6491 | score sample-invariant — informational |
| E_raw.soft (OOF) | 0.6250 | corrector 효과 +0.0274pp |
| F_corrected.soft (uniform, sanity) | 0.6520 | < E_corrected (strict) — physics_bias 가 uniform 보다 +0.0004pp |
| **LB lb_score** | **0.6692** | full LB 0.6806 대비 -0.0114pp, `≥ 0.6606` cutoff ✓ |
| **OOF→LB gap** | **+0.0168** | plan-004 full gap +0.0207 와 거의 일관 (-0.0039) |

## §3. plan-005 통찰 LB 입증 — 시나리오 A 채택

- 명제 1: `lb_score ≥ 0.6606` (inclusive) → **0.6692 ≥ 0.6606 ✓ 입증**.
- plan-005 의 "PB framework 의 95% 가 장식, 진짜 엔진은 27 후보 + physics_bias + soft averaging" 통찰 — *LB 단위로 검증*.
- GRU + regime 의 marginal LB 기여 ≈ +0.0114 (full 0.6806 − Variant E 0.6692). plan-005 의 OOF marginal +0.0075 (full 0.6599 − E 0.6524) 와 거의 동일 (LB ↔ OOF 일관).
- → 후속 plan-007 = **시나리오 A**: 후보 다양화 (A1) + corrector 재설계 (A2) 우선.

## §4. 변경 이력

- 2026-05-12 (KST 11:13): c5 — dacon 제출 성공 `{isSubmitted: True, detail: Success}`. `lb_score: TBD` (carry-over open).
- 2026-05-12 (KST 11:26): c5.1 — 사용자 LB 회수 = **0.6692**. 3 파일 frontmatter 동시 갱신, status `all_complete`. 시나리오 A 결정.
