# plan-029 results — GRU-attention Input Max (X1 cell)

**date**: 2026-05-22
**status**: complete
**G3 verdict**: **regression** (hit_1cm 0.6316 < F0 0.6320 by 0.0004)

## TL;DR

> plan-024 의 sample-invariant query 약점을 4 lever (a/b/c/d 동시) 로 fix 한 결과, **attention 학습은 됐으나 (loss 2.64 → 1.5~1.98, top1_acc 0.1266 > random 0.071, grad_norm 정상) hit metric 은 F0 baseline 아래로 떨어짐 (0.6316)**. paradigm framework (cross-attention GRU selector) 자체가 F0 hardlimit 미달 — 사용자 진단 (query sample invariant = plan-024 fail root cause) 의 part 1 (query 학습 가능) ✓, part 2 (hit_1cm 회복) ✗. follow-up = paradigm-distinct lever (F0 ML / corrector / KNN) 전환, plan-030 single-lever ablation 우선순위 낮음.

## Hyperparams

| Hparam | 값 |
|---|---|
| cell | X1 (4 lever 동시) |
| hidden | 196 |
| seq_dim | 95 |
| cand_in_dim | 165 (= plan-024 150 + N_new 15) |
| anchor_embed_dim | 8 (learnable, init randn × 0.1) |
| epochs | 50 fixed (early stop disabled) |
| batch | 64 |
| lr | 7e-4 |
| schedule | SequentialLR([LinearLR warmup 5 ep, CosineAnnealingLR T_max=45 ep]) |
| AdamW wd | 1e-4 |
| GRU dropout | 0.10 (2-layer) |
| gradient clip | 1.0 |
| τ_cls | 0.001 |
| K | 14 (BCC, ANCHORS_A6) |
| loss | soft cross-entropy = -(q × log_softmax(score)).sum(-1).mean() |
| seed | 20260522 |
| total param | 478,745 |

## Metric

| metric | plan-029 X1 | F0 baseline | plan-024 ceiling | plan-022 winner | 14-oracle |
|---|---:|---:|---:|---:|---:|
| **hit_1cm** | **0.6316** | 0.6320 | 0.6387 | 0.6531 | 0.7928 |
| hit_1p5cm | 0.8039 | 0.8033 | — | 0.8108 | — |
| Δ vs F0 | **-0.0004** | — | — | — | — |
| Δ vs plan-024 ceiling | -0.0071 | — | — | — | — |
| Δ vs plan-022 winner | -0.0215 | — | — | — | — |
| recall vs oracle | 79.7% | 79.7% | 80.6% | 82.4% | 100% |
| max_class_ratio | 0.1328 | — | — | 0.1054 | — |
| top1_acc | 0.1266 | — | — | — | — |

## G3 판정

- **PASS** (>0.6528): ❌
- **partial_above_p024** (0.6387~0.6528): ❌
- **partial_below_p024** (0.6320~0.6387): ❌
- **regression** (<0.6320): ✓ — hit_1cm = 0.6316

## Paradigm finding (4 lever 동시 검증 결과)

### 무엇이 작동했는가
1. **attention path 학습 정상**:
   - loss 2.64 (uniform) → 1.5~1.98 (25-43% 감소, fold 별)
   - per-fold grad_norm 정상: ep5 grad ≥ 0.058 (임계 1e-4 의 580×), final grad ≥ 0.024
   - anchor_embed / anchor_key_proj gradient 흐름 정상
2. **anchor identity discrimination 일부 성공**:
   - top1_acc = 0.1266 (random 0.0714 의 1.78×)
   - per-anchor argmax: anchor 1 = 13.3%, anchor 9 = 3.9% (3.4× spread, collapse 없음)

### 무엇이 작동하지 않았는가
1. **task 성능 변환 실패**: attention 학습 성공 ≠ hit metric 향상
2. **F0 baseline 미달**: 0.6316 < 0.6320 (-0.0004)
3. **plan-024 ceiling 미달**: 0.6316 < 0.6387 (-0.0071)
4. **soft prediction 의 residual quality**: probs @ ANCHORS_A6 산출이 F0 hardlimit (0.6320) 아래로 떨어짐 — model 이 anchor 를 *고를 줄은* 알아도 *고른 anchor 로의 residual 가 F0 보다 정확하지는 않은* 결과

### 사용자 진단 검증

| 사용자 진단 part | 검증 결과 |
|---|---|
| (1) plan-024 의 query 가 sample invariant — attention 학습 자체 불가 | **part 검증**: 4 lever 적용 시 attention 학습 가능 (top1_acc 1.78× random). 즉 진단 자체의 mechanism (query 의 sample×anchor interaction 부족) 은 *학습 가능성* 측면에서 유효 |
| (2) 진단 직접 fix → hit_1cm > 0.6528 (LGBM floor 회복) | **반증**: attention 학습 성공해도 hit metric 회복 안 됨. paradigm framework 자체가 F0 hardlimit 아래 |

## H1~H4 검증 결과

| 가설 | 임계 | 결과 |
|---|---|---|
| H1: hit_1cm > 0.6528 (LGBM floor 회복) | strict > | ❌ (0.6316) |
| H1a: hit_1cm > 0.6387 (plan-024 ceiling 위) | strict > | ❌ (0.6316) |
| H3: max_class_ratio < 0.95 (mode collapse 없음) | < 0.95 | ✓ (0.1328) |
| H4: anchor_embed cosine off-diag mean < 0.5 | < 0.5 | ⏸ unmeasured (5-fold model.anchor_embed dump 미수행, G3 = regression 으로 follow-up 우선순위 낮음) |

## Runtime + Param

- elapsed: **565.3s = 9.4 min** (5-fold OOF, spec 예상 7-15 min ✓)
- per-fold: 113~117 s (= 2 min/fold)
- total model param: **478,745** (GRU ~404K + query_mlp ~73K + anchor_key_proj ~1.76K + anchor_embed 112 + head 197)

## Follow-up direction

본 plan 의 G3 = regression 결과 + plan-024 plateau (0.6370~0.6387) 평가 종합:

1. **paradigm-distinct lever 전환 권장** (paradigm-level 검증 1회 plan 의 본 plan 의도 따라):
   - F0 ML (F0 baseline 자체 학습)
   - corrector (selector 가 아닌 직접 residual correction)
   - KNN (instance-based prediction)
2. **plan-030 single-lever ablation 우선순위 낮음**: 4 lever 모두 적용해도 regression — a/b/c/d 단독 ablation 도 비슷한 결과 예상. 시간 비용 대비 paradigm-level information gain 낮음.
3. **만약 plan-030 ablation 진행한다면**: lever (a) query enrichment 단독 + (d) head raw skip 차단 단독 가 가장 informative — 전자는 plan-024 의 *interaction channel 시간 axis 확장* 효과 단독 측정, 후자는 paradigm-confound source 차단 단독 효과 측정.

## Files

- `analysis/plan-029/anchor_query_extend.py` (c3, lever a + D channel lookup)
- `analysis/plan-029/model.py` (c4, GRUNetX1 — 4 lever 통합)
- `analysis/plan-029/train.py` (c5, 5-fold OOF training)
- `analysis/plan-029/run_oof.py` (c6, orchestrator + G1 + X1 + G2/G3 verdict)
- `analysis/plan-029/paradigm_analysis.py` (c10, 후처리 분석)
- `tests/test_plan029_smoke.py` (c7, 19 pytest)
- `analysis/plan-029/results_X1.json` (X1 결과 summary)
- `analysis/plan-029/oof_X1.npz` (oof_pred + oof_probs + gt_anchor_label)
- `analysis/plan-029/paradigm_analysis.{json,md}` (c10 분석본)
- `analysis/plan-029/g1_run.log` + `train_X1.log` (실행 로그)

## Commit chain

c1 (spec) → c2 (cherry-pick) → c3 (anchor_query_extend) → c4 (model) → c5 (train) → c6 (run_oof) → c7 (tests) + G0 → c8 (G1) → c9 (G2.X1) → c10 (paradigm_analysis) → G3 = regression → c11 (results + sync) → G_final.
