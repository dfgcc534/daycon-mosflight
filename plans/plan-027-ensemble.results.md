---
plan_id: 027
finished_at: 2026-05-22 (Asia/Seoul)
status: all_complete
best_cell: E3_weighted
best_hit_1cm: 0.6529
best_hit_1p5cm: 0.8118
best_delta_1cm: -0.0001
best_delta_1p5cm: 0.0005
band: negative_ensemble
exp_ids_completed:
  - Z027_E1_eq_2way
  - Z027_E2_eq_3way
  - Z027_E3_weighted_optim
exp_ids_skipped: []
---

# plan-027.results — Ensemble 3-way (FINAL)

## 핵심 결과

- **best**: E3 weighted [0.3, 0.3, 0.4] — hit_1cm=**0.6529**, hit_1p5cm=**0.8118**
- **base_max** (p022 = p023 = 0.6530) 대비 **-0.0001** (effectively tie)
- **band: negative_ensemble** — anchor codebook 만으로 ensemble diversity 부족
- hit_1p5cm 만 **+0.0010 미세 lift** (E1/E3, p026_A2 포함 시)

## 3 base predictor + 3 ensemble cell

| | hit_1cm | hit_1p5cm |
|:--|--:|--:|
| p022 (170D, K=14 BCC) | 0.6530 | 0.8108 |
| p023 (170D, K=50 fib) | 0.6530 | 0.8113 |
| p026_A2 (1058D, K=14 no-block③) | 0.6509 | 0.8118 |
| E1 equal 3-way | 0.6523 | 0.8118 |
| E2 equal 2-way (p022+p023) | 0.6524 | 0.8110 |
| **E3 weighted (0.3,0.3,0.4)** 🏆 | **0.6529** | **0.8118** |

## Paradigm finding

3 base predictor 의 prediction error pattern *highly correlated* — same model + same fold + similar input pipeline + anchor 만 다름 → ensemble averaging 으로 individual best 손실, noise reduction 효과 X.

진짜 ensemble lift 위해서는 **paradigm diversity** 필요 (F0 vs selector vs GRU vs cross-attention 등 architecture 차이).

## Severe / warn

- Severe 0건
- `negative_ensemble` warn (G3 best < base_max) — paradigm finding 박제

## Runtime

총 ~43min CPU (p023 K=50 LGBM 26min dominant + p026_A2 13min + p022 4min). ensemble grid 자체 즉시.

## Follow-up

- **plan-028 (F0 ML)**: ceiling = 0.6530 = 14-oracle 82.4%. 잔여 17.6% = F0 systematic forward bias. F0 자체 ML화 가장 큰 lever.
- **plan-029 (selector redesign)**: row-expand LGBM self-prediction 약점 + ensemble diversity 부족 → architecture paradigm shift.
