# plan-021 STAGE 2 G2.A — Sub-exp A LGBM 5-fold OOF (v1.3)

## 결과 (2026-05-18, n_estimators=500, lr=0.05, num_leaves=63)

| metric | F0 baseline | A LGBM v1.3 | Δ | pass (≥+0.005) |
|---|---|---|---|---|
| hit@1cm | 0.6320 | **0.6488** | **+0.0168** | ✓ |
| hit@1.5cm | 0.8033 | 0.8070 | +0.0037 | ✗ |
| fold variance 1cm | 0.0052 | 0.0066 | — | informational |
| fold variance 1.5cm | 0.0087 | 0.0077 | — | informational |

**pass_both = False** (hit@1.5cm Δ +0.0037 < +0.005 threshold).

Wall time: 334 s (CPU, 5-fold × 22 booster). G2.A 합격 (metric finite, no NaN/Inf).

## Per-fold

| fold | hit@1cm |
|---|---|
| 0 | 0.6584 |
| 1 | 0.6463 |
| 2 | 0.6528 |
| 3 | 0.6480 |
| 4 | 0.6386 |
| concat | **0.6488** |

per-fold 안정 (1cm σ=0.0066). plan-020 C05 의 fold variance (σ=0.0056) 보다 약간 큼.

## plan-020 winner 와 비교

| candidate | Δ_1cm | Δ_1.5cm | pass_both |
|---|---|---|---|
| plan-020 C05 per-regime F0 | +0.0183 | +0.0053 | ✓ |
| **plan-021 A LGBM v1.3** | **+0.0168** | **+0.0037** | **partial (1cm only)** |
| plan-020 N1 MLP coef | +0.0069 | -0.0010 | ✗ |

→ 본 plan 의 input augment 4 lever + dual head 가 **1cm metric 에서 C05 의 92% 효과** 도달, 1.5cm 에서 70% 만 달성.

paradigm-level 진단:
- **1cm tight metric**: 4 lever input augment (Frenet trajectory + F0 잔차 sequence + F0 soft hit + soft label) + 7-anchor classifier 가 task target distribution 의 핵심 신호 흡수 — C05 의 18-regime discrete partition 과 거의 동등.
- **1.5cm loose metric**: 0.5cm anchor radius × ±0.005m reg_offset bounded = max ±1cm Frenet 영역 → 1.5cm hit zone (정답이 prediction 1.5cm 안) 의 일부 sample 이 anchor 영역 밖으로 새어나감. anchor radius 확장 또는 reg_offset bound 완화가 follow-up lever.

## v1.3 conceptual fix 의 효과

| version | hit@1cm | Δ_1cm |
|---|---|---|
| v1.2 (anchor reference = x[end_idx], conceptual error) | 0.0600 | -0.5720 |
| **v1.3 (anchor reference = pred_F0_world)** | **0.6488** | **+0.0168** |

→ +0.589 hit@1cm 회복 (paradigm 의 *실제 효과* 발현). c7 actual run 이 spec 의 conceptual error catch — silent design bug 발견 + 즉시 fix.

## G2.A PASS (metric finite + dispatch sanity)

- A LGBM 5-fold OOF metric finite ✓
- NaN/Inf 0건 ✓
- 위반 trigger (`lgbm_numerical`) 미발동.
- 단, pass criterion (paired Δ ≥ +0.005 *둘 다*) 의 1.5cm 측 미달 → G3 단독 PASS 후보 X (B GRU 결과 후 paradigm-level 결정).
