# plan-026 — Block Ablation (FINAL)

> **🎯 PARADIGM REVERSAL**: A2 (no block ③) hit_1cm=**0.6509** (+0.0189 vs plan-025 C1 0.6320 baseline). H1 가설 정반대. block ③ 22D per-anchor 가 **mode collapse 원인** (LGBM row-expand self-prediction trivial 학습). 1058D LGBM 가 plan-022 winner 0.6531 의 99.66% 회수.

## 핵심 결과

| Cell | D_masked | hit_1cm | hit_1p5cm | Δ vs baseline (C1) | max_class_ratio | runtime |
|:--|--:|--:|--:|--:|--:|--:|
| baseline (plan-025 C1) | 1080 | 0.6320 | 0.8033 | — | 0.0714 | 334s (carry) |
| A1 (no block ②) | 952 | 0.6320 | 0.8033 | 0.0000 | 0.0714 | 302s |
| **A2 (no block ③)** 🏆 | 1058 | **0.6509** | **0.8118** | **+0.0189** | 0.106 | 779s |
| A3 (no block ④) | 320 | 0.6320 | 0.8033 | 0.0000 | 0.0714 | 132s |

Δ vs plan-022 winner (0.6531): A2 = -0.0022 (99.66% 회수).

## G-gate 표

| Gate | Status | Commit | 결과 |
|:--|:--|:--|:--|
| G0 | ✅ | 04ba0bf | 8/8 pytest, prereq_check ✓ |
| G1 | ✅ | (prereq_check) | plan-025 C1 baseline 0.6320 carry ✓ |
| G2.A1 | ✅ | d9daaf8 | 0.6320 (no effect) |
| G2.A2 | ✅ | d9daaf8 | **0.6509 (+0.0189, mode collapse 해소)** |
| G2.A3 | ✅ | d9daaf8 | 0.6320 (no effect) |
| G3 | ✅ | (본 commit) | attribution_negative warn (block ③ = noise lever, REVERSE 가설) |
| G_final | ✅ | (본 commit) | 3-file sync + follow-up 3건 |

## 가설 검증

| 가설 | 측정 | 결과 |
|:--|:--|:--|
| H1 (강): block ③ 제거 시 큰 drop | A2: 0.6320 → 0.6509 = **+0.0189 LIFT** | **REVERSE — H1 정반대 결과** |
| H2 (약): block ② 제거 시 작은 drop | A1: 0.6320 → 0.6320 = 0.0000 | NEUTRAL (block ② = irrelevant) |
| H3 (약): block ④ 제거 시 중간 drop | A3: 0.6320 → 0.6320 = 0.0000 | NEUTRAL (block ④ = irrelevant) |

## Paradigm finding (가장 중요)

**block ③ 22D per-anchor feature 가 LGBM row-expand selector 의 *mode collapse 원인***:

- block ③ = anchor 별 다른 값 (par/perp/dist + anchor spec + interactions)
- row-expand 에서 row k 의 X feature 중 block ③ 가 row k 의 anchor identity 를 *직접 encode* (anchor 0..13 의 sign/group/idx scalar 등)
- LGBM 이 trivial 학습: "row k 위에서는 class k 예측" → self-prediction
- predict_proba (N*K, K) 의 row i*K + k 에서 class k 위 확률 ≈ 1, others ≈ 0
- diag extraction `probs_sel[i, k] = probs_expanded[i*K+k, k]` ≈ 1 for all k
- row-normalize 후 분포 ≈ uniform 1/14 = 0.0714
- soft-mean(ANCHORS_A6) = mean(ANCHORS) ≈ Frenet origin
- world prediction = F0 + origin = F0 (selector 무의미)

block ③ 제거 시: row k 의 X 가 어느 anchor 인지 모름 → meaningful sample-conditional 학습 → 의미 있는 selector probs → lift +0.0189.

## A2 의 비교 위치

- A2 hit_1cm = 0.6509 vs plan-022 winner 0.6531 = **-0.0022 (99.66% 회수)**
- A2 hit_1p5cm = 0.8118 vs plan-022 winner 0.8108 = **+0.0010 (1.5cm 에서는 미세 lift)**
- A2 oracle 회수율 = 0.6509 / 0.7928 = **82.10%** (plan-022 winner 82.38% 와 거의 동등)

→ 1058D LGBM 가 plan-022 170D 와 본질적으로 동등 성능. **feature 추가가 추가 lift 안 만든다** (saturation).

## Follow-up

- **plan-027 ensemble**: baseline 후보 갱신 → plan-026 A2 (0.6509) 가 plan-025 C1 (0.6320) 보다 우월. band 조건 분기 = (plan-026 A2 가 plan-022 winner 0.6531 의 -0.0022 → "near-positive" band) → 3-way ensemble (p022 + p023 + p026_A2) 시도 valid.
- **plan-028 F0 ML**: anchor selection 의 ceiling (plan-022/023 winner = 0.6531/0.6532) 자체가 14-anchor oracle 0.7928 의 82.4% → **F0 baseline 개선이 가장 큰 남은 lever**.
- **plan-029 (가칭)** row-expand selector redesign: block ③ self-prediction 우회. (a) anchor identity 를 별도 head, OR (b) sample-level model paradigm shift.

## Cross-refs

- spec: `plans/plan-026-block-ablation.md`
- carry: plan-025 build_feat_1080 + run_oof (LgbmSelectorRowExpanded), plan-022 anchors/model, plan-024 cand/seq builders
- analysis dir: `analysis/plan-026/` (block_mask_builder, run_oof, 3 results, attribution.json, results.md)
- memory: `project_next_plan_direction.md` (2026-05-22 user 한 줄 재정의)
