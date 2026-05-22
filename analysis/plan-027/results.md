# plan-027 — Ensemble 3-way (FINAL)

> **G3 = negative_ensemble band**. best E3 (weighted [0.3, 0.3, 0.4]) = **0.6529** vs base_max (p022=p023=0.6530) = -0.0001. ensemble effective lift X. hit_1p5cm 만 +0.0010 미세 lift.

## 핵심 결과

| Cell | hit_1cm | hit_1p5cm | weights (p022, p023, p026_A2) | Δ vs base_max |
|:--|--:|--:|:--|--:|
| base p022 | 0.6530 | 0.8108 | — | — |
| base p023 | 0.6530 | 0.8113 | — | — |
| base p026_A2 | 0.6509 | 0.8118 | — | -0.0021 |
| **base_max** | **0.6530** | — | — | 0.0000 |
| pass_threshold | 0.6550 | — | — | +0.0020 |
| E1 (equal 3-way) | 0.6523 | 0.8118 | (1/3, 1/3, 1/3) | -0.0007 |
| E2 (equal 2-way) | 0.6524 | 0.8110 | (0.5, 0.5, 0.0) | -0.0006 |
| **E3 (weighted best)** | 0.6529 | 0.8118 | (0.3, 0.3, 0.4) | -0.0001 |

**G3 verdict**: negative_ensemble (best < base_max). hit_1p5cm 만 +0.0010 미세 lift (E1 / E3).

## E3 weight grid (9-point)

| weights | hit_1cm | hit_1p5cm |
|:--|--:|--:|
| (0.5, 0.5, 0.0) | 0.6524 | 0.8110 |
| (0.4, 0.4, 0.2) | 0.6520 | 0.8118 |
| (0.4, 0.3, 0.3) | 0.6520 | 0.8119 |
| (0.3, 0.3, 0.4) 🏆 | 0.6529 | 0.8118 |
| (0.34, 0.33, 0.33) | 0.6522 | 0.8118 |
| (0.6, 0.4, 0.0) | 0.6527 | 0.8113 |
| (0.4, 0.6, 0.0) | 0.6522 | 0.8112 |
| (0.3, 0.5, 0.2) | 0.6518 | 0.8116 |
| (0.5, 0.3, 0.2) | 0.6525 | 0.8118 |

**관찰**: p026_A2 (가장 낮은 base hit 0.6509) 가 best weight 에서 가장 높은 가중치 (0.4) — diversity 기여는 있으나 base hit 손실로 net 0. hit_1p5cm 은 p026_A2 포함된 모든 weight 에서 0.8118 도달 (base p022/p023 0.8108/0.8113 보다 ↑).

## G-gate

| Gate | Status | Commit | 결과 |
|:--|:--|:--|:--|
| G0 | ✅ | e7ed15a | run_oof.py 작성 + import OK |
| G1 | ✅ | 16e1397 (실행 bqxddnj9i) | base predictor 3-way reproduce: 0.6530/0.6530/0.6509 (모두 tight band ✓) |
| G2.E1 | ✅ | (본 commit) | equal 3-way = 0.6523 |
| G2.E2 | ✅ | (본 commit) | equal 2-way = 0.6524 |
| G2.E3 | ✅ | (본 commit) | weighted best = 0.6529 |
| G3 | ✅ | (본 commit) | negative_ensemble (best < base_max) |
| G_final | ✅ | (본 commit) | 3-file sync + follow-up 2건 |

## Paradigm finding

**3 base predictor 의 prediction error pattern correlated** — anchor codebook 차이 (K=14 BCC vs K=50 fib vs K=14 1058D) 만으로는 ensemble diversity 부족.

근거:
- p022 (K=14 BCC, 170D) hit_1cm = 0.6530
- p023 (K=50 fib, 170D) hit_1cm = 0.6530 (anchor 만 다르고 input 동일)
- p026_A2 (K=14, 1058D, no block③) hit_1cm = 0.6509

모두 same model (LgbmSelectorOnly variant) + same fold + similar input pipeline → prediction 매우 유사. averaging 으로 individual best 손실, noise reduction 효과 X.

진짜 ensemble diversity 위해서는 *paradigm 차이* 필요:
- F0 baseline (numpy, deterministic) + selector (LGBM)
- vs GRU sequence model
- vs cross-attention (plan-024 paradigm — 단 cross-attention 자체는 fail)
- vs sample-level model (per-anchor 의 self-prediction 없는 다른 architecture)

→ **plan-029 (selector redesign)** 가 더 큰 lever.

## Runtime

- p022 reproduce: 5 × 46s = 228s (~4min)
- p023 reproduce: 5 × 313s = 1568s (~26min, K=50 LGBM 무거움)
- p026_A2 reproduce: 5 × 150s = 752s (~13min)
- 합계: ~43min CPU. ensemble grid 자체는 즉시.

## Follow-up

- **plan-028 (F0 ML)**: anchor selection ceiling = max(p022, p023, p026_A2) = 0.6530 (14-oracle 0.7928 의 82.4%). 잔여 17.6% lever 의 대부분이 *F0 baseline 의 systematic forward bias*. F0 자체를 ML 화 하면 ceiling 격상 가능.
- **plan-029 (selector redesign)**: row-expand LGBM 의 self-prediction 약점 (plan-026 paradigm reversal) + ensemble diversity 부족 (본 plan finding) → architecture 차이 paradigm. sample-level model (per-anchor masking 우회) OR 진짜 cross-modal architecture (residual + attention) 시도.

## Cross-refs

- spec: `plans/plan-027-ensemble.md` (v1.1)
- analysis: `analysis/plan-027/run_oof.py`, `results_ensemble.json`, `run.log`
- base predictors carry:
  - plan-022: `analysis/plan-022/{anchors.py:ANCHORS_A6, selector_only_model.py, run_oof.py}`
  - plan-023: `analysis/plan-023/anchors_largeN.py:ANCHORS_B4`
  - plan-026 A2: `analysis/plan-026/{block_mask_builder.py, run_oof.py}`
- carry: plan-025 LgbmSelectorRowExpanded, plan-024 cand/seq builders, plan-021 build_input, plan-020 baseline_f0
