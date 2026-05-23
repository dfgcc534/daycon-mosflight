---
plan_id: plan-032
status: complete
based_on: plan-031
title: PB target 0.6511 도달 시도 (results)
g_final_hit_1cm: 0.6438
g_final_band: STRONG
best_axis: B (boundary corrector 14a)
best_axis_lift: +0.0041
pb_target_reached: false
gap_to_pb_selector_ensemble: -0.0073
followed_by: null (반전 실패, plan-031 + B patch 또는 DACON submit 권장)
---

# plan-032 results

## §0. 한 줄 결론

**반전 부분 성공 / PB target 미달**: 4 axis ablation 중 **axis B (boundary corrector 14a) 만 +0.0041 lift** (plan-031 0.6397 → 0.6438 STRONG). axis A (τ_cls 완화) / C (label smoothing) / D (input axis) 모두 lift 없음 (각각 -0.0035 / +0.0002 / 0). multi-axis 결합 = B 단독 (A/C nega). **PB target 0.6511 까지 -0.0073 잔여, plan-032 의 cheap axis 들로는 도달 불가능**.

## §0.5 Result Quick Reference

| 항목 | 값 |
|---|---|
| baseline (plan-031 G3) | 0.6397 STRONG |
| **best result (plan-031 + axis B)** | **0.6438 STRONG** |
| best axis | B (boundary corrector 14a) |
| best axis lift | +0.0041 |
| PB target (selector ensemble) | 0.6511 — 미달 (-0.0073) |
| PB target (selector soft) | 0.6624 — 미달 (-0.0186) |
| PB superior (selector + boundary) | 0.6718 — 미달 (-0.0280) |
| **반전 성공?** | **부분 성공** (lift +0.0041, but PB target 미달) |

## §1. Ablation 결과 표

| axis | variant | cost | hit_1cm | Δ vs plan-031 | band | 결론 |
|---|---|---|---|---|---|---|
| A | τ_cls = 0.005 | 9.7분 | 0.6362 | **-0.0035** | BORDERLINE | **FAIL** — label sharpness mismatch 가설 X |
| A | τ_cls = 0.01 | — | (skipped) | — | — | A1 fail 후 skip |
| **B** | corrector 33D ep50 | 17초 | 0.6438 | **+0.0041** | STRONG | **MARGINAL POSITIVE** (유일 lift axis) |
| B | corrector 57D ep80 (B2) | 42초 | 0.6439 | +0.0042 | STRONG | corrector ceiling 도달 (feature richness 무관) |
| C | label smoothing α=0.1 | 10.4분 | 0.6399 | +0.0002 | STRONG | **FAIL** (zero effect) |
| D | 잔차 (a) drop (G2) | 2.1분 | 0.6455 (fold-0) | +0.0005 vs G1 | informative | **net contribution 0** |
| D | 잔차 (b) drop (G2) | 2.1분 | 0.6450 (fold-0) | +0.0000 vs G1 | informative | **net contribution 0** |

**핵심 발견**:

### 1. axis D 의 충격적 결과 — plan-030 input axis 가 dead weight
- 잔차 (a) 35D + 잔차 (b) 35D × 14 anchor 모두 drop → fold-0 hit_1cm 변화 0.
- plan-030/031 의 lift +0.0103 의 **100% 가 training procedure (multi-phase + pairwise + prior + head slim)** 에서 옴.
- 잔차 input 은 dead weight (negative 도 아니지만 lift 도 안 줌).
- 함의: plan-030 의 잔차 paradigm fix 가 **의미 없는 패치**였음. plan-031 의 진짜 가치 = training procedure 만.

### 2. axis A FAIL — label sharpness 가설 틀림
- τ_cls 0.001 → 0.005 (sharpness 1000 → 200) 했더니 -0.0035 worse.
- "model 의 logit magnitude 가 label sharpness 와 mismatch" 가설은 wrong direction.
- 오히려 sharp target 이 model 학습에 *필요한* 신호. 완화 시 model 이 게으르게 학습.

### 3. axis B 의 +0.0041 단독 lift — corrector ceiling
- 33D feature ep50 / 57D feature ep80 모두 ~+0.004 plateau.
- 14-anchor paradigm 의 fundamental expressivity limit 도달.
- PB carry +0.0207 의 1/5 수준 — 27 candidate hypothesis diversity 부재 → corrector 가 학습할 신호 빈약.

### 4. axis C FAIL — label smoothing 효과 zero
- α=0.1 uniform mixing 시 lift +0.0002 (noise level).
- regime/class prior loss (plan-031 carry) 가 이미 distribution smoothing 효과 가짐 → 추가 smoothing redundant.

### 5. multi-axis 결합 = B 단독
- A nega + C nega → 결합 의미 없음
- B 만 적용 = plan-031 의 train 결과에 post-process corrector inject
- → **final result = 0.6438 STRONG band**

## §2. PB target 0.6511 미달 원인 분석

| layer | 진단 |
|---|---|
| paradigm | **14 anchor (anchor residual) vs 27 candidate (hypothesis)** 의 fundamental 차이. corrector 학습 신호 빈약 |
| ensemble 부재 | PB 는 TCN + GRU 결합 ensemble. plan-031 = single GRU model |
| epoch budget | 50 sweet spot (epoch 100 -0.0089 over-fit) — capacity 한계 |
| boundary corrector | PB 27-cand 기반 + larger feature engineering. plan-032 의 14-anchor 재구현은 +0.0041 ceiling |
| GPU infrastructure | PB = GPU batch 4096 + hidden 48. plan-031 = CPU batch 64 + hidden 196. dynamics 다름 |

**PB target 0.6511 도달 위한 잔여 lever** (높은 cost):
1. TCN + GRU ensemble (별도 model 학습 + score blend) — +0.005~0.010 추정
2. anchor 14 → 27 hypothesis 확장 (paradigm 변경) — high cost
3. GPU 인프라 도입 (batch=4096 + hidden=48) — infrastructure 의존
4. PB selector 의 boundary corrector 의 selector 분리 inject (PB selector 의 output 을 plan-031 으로 blend)

## §3. Commit chain 결과

| commit | axis | status |
|---|---|---|
| c0 spec | — | [DONE] |
| c1 ablation A1 | A | [DONE] FAIL (-0.0035) |
| c2 ablation D1/D2 | D | [DONE] informative (net=0) |
| c3 ablation B (B1+B2) | B | [DONE] +0.0041~+0.0042 (only lift) |
| c4 ablation C1 | C | [DONE] FAIL (+0.0002) |
| c5 multi-axis 결합 | — | [DEFERRED — A/C nega 라 결합 = B 단독 0.6438] |
| c6 results | — | [DONE] 본 문서 |

## §4. Decision

본 plan **반전 부분 성공** (B +0.0041, plan-031 STRONG band 안전 margin 확보).
- **PB target 0.6511 미달** (-0.0073).
- 추가 lever 모두 high cost (ensemble / paradigm 변경 / GPU 인프라).

### 권장 다음 단계 (사용자 결정):

| option | 비용 | 예상 lift | 비고 |
|---|---|---|---|
| **DACON submit (plan-031 단독)** | 0 | LB 검증 | quota 의무 confirm |
| **DACON submit (plan-031 + axis B patch)** | 0 (코드 ready) | LB +0.004 추정 | 본 plan 의 best result |
| plan-033 TCN+GRU ensemble | high (별도 model 학습) | +0.005~0.010 | PB level ensemble carry |
| plan-033 PB selector blend | medium | +0.010~0.020 | PB selector inference 결과를 plan-031 와 weighted blend |
| 종료 (STRONG band 만족) | 0 | — | plan-031 0.6397 ≥ plan-024 0.6387 ceiling |

## §5. Artifact

- `analysis/plan-032/results_A1.json` + `.npz` — axis A FAIL
- `analysis/plan-032/results_D1.json`, `results_D2.json` — axis D informative
- `analysis/plan-032/results_B.json` + `.npz` (corrected_pred) — axis B best (+0.0041)
- `analysis/plan-032/results_C1.json` + `.npz` — axis C FAIL

## §6. Decision-note

decision-note: plan-032 multi-axis ablation 완료. 4 axis (A/B/C/D) 단독 측정 → B 만 +0.0041 lift (corrector ceiling). PB target 0.6511 -0.0073 미달. **plan-030 의 잔차 input axis 가 dead weight (axis D informative 결과)** — plan-031 lift 100% = training procedure. multi-axis 결합 의미 없음 (A/C nega). 사용자 결정 = DACON submit / plan-033 / 종료 중 선택. plan-031 STRONG (≥ plan-024 honest ceiling 0.6387) 만족 충분.
