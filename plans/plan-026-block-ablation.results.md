---
plan_id: 026
finished_at: 2026-05-22 (Asia/Seoul)
status: all_complete
best_cell: A2_no_block3
best_hit_1cm: 0.6509
best_hit_1p5cm: 0.8118
best_delta_1cm: 0.0189
best_delta_1p5cm: 0.0085
band: paradigm_reversal
exp_ids_completed:
  - Z026_A1_no_block2
  - Z026_A2_no_block3
  - Z026_A3_no_block4
exp_ids_skipped: []
---

# plan-026.results — Block Ablation (FINAL)

## 핵심 결과 요약

- **best**: A2 (no block ③) — hit_1cm=**0.6509**, hit_1p5cm=**0.8118**
- **plan-025 C1 baseline 0.6320 대비 +0.0189 lift** (block ③ 제거가 hit 향상)
- **plan-022 winner 0.6531 대비 -0.0022** (99.66% 회수)
- **band: paradigm_reversal** — H1 가설 정반대 결과 (block ③ = noise lever 가 아닌, **selector mode collapse 원인**)
- 14-anchor oracle 회수율 = 82.10% (plan-022 82.38% 와 거의 동등)

## 3 cell 결과

| Cell | D_masked | hit_1cm | hit_1p5cm | Δ vs C1 | max_class_ratio | runtime |
|:--|--:|--:|--:|--:|--:|--:|
| A1 (no block ②) | 952 | 0.6320 | 0.8033 | 0.0000 | 0.071 | 302s |
| **A2 (no block ③) 🏆** | 1058 | **0.6509** | **0.8118** | **+0.0189** | 0.106 | 779s |
| A3 (no block ④) | 320 | 0.6320 | 0.8033 | 0.0000 | 0.071 | 132s |

## Paradigm finding

**block ③ 22D per-anchor feature = LGBM row-expand selector 의 mode collapse 원인**.

- block ③ = anchor identity 직접 encode (sign/group/idx scalar)
- row-expand 에서 row k 가 항상 class k 예측하도록 trivial 학습 → self-prediction
- diag extraction + row-normalize → uniform 1/14 분포 → soft-mean(ANCHORS) ≈ origin → F0 prediction
- block ③ 제거 시: meaningful sample-conditional 학습 → lift +0.0189

A2 (1058D) 가 plan-022 winner (170D) 의 99.66% 회수 → 추가 feature lever (block ② + ④) 는 saturation.

## Severe / warn

- Severe 0건
- `attribution_negative` warn (A2): drop = -0.0189 (block ③ 제거가 hit 향상)
- spec H1 가설 정반대 결과 → paradigm finding 박제

## Runtime

- 총 plan-026: ~25min CPU (G0 + 3 ablation × 평균 7min + analysis)
- A2 가 가장 길음 (779s, 1058D 대응 fit 수렴 더 많이)

## Follow-up plan 후보

- **plan-027 ensemble**: baseline 후보 = plan-026 A2 (0.6509). band 격상 → 3-way ensemble (p022 + p023 + p026_A2) valid.
- **plan-028 F0 ML**: anchor selection ceiling = 0.6531 ≈ 14-oracle 82.4% → F0 자체 개선이 가장 큰 잠재력.
- **plan-029 (가칭)** row-expand selector redesign: block ③ self-prediction 우회 (anchor identity 별도 head OR sample-level model).

## Cross-refs

- spec: `plans/plan-026-block-ablation.md` (v1.1)
- analysis: `analysis/plan-026/` (block_mask_builder, run_oof, 3 results, attribution.json, results.md)
- carry: plan-025 build_feat_1080 + LgbmSelectorRowExpanded, plan-022 anchors + LgbmSelectorOnly + run_oof_cell, plan-024 cand/seq/torsion/quantile/multiwindow_trim, plan-021 build_input, plan-020 baseline_f0
