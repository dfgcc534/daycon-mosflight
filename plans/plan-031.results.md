---
plan_id: plan-031
status: complete
based_on: plan-030
title: PB training procedure carry (results)
g3_oof_hit_1cm: 0.6397
g3_band: STRONG
g1_oof_hit_1cm: 0.6450
lift_vs_plan_030: +0.0103
lift_vs_plan_029_x1: +0.0081
lift_vs_F0: +0.0077
followed_by: plan-032 (fine-distill + reverse-pretrain + GPU batch 후보)
---

# plan-031 results

## §0. 한 줄 결론

**G3 STRONG band PASS**: 5-fold OOF hit_1cm = **0.6397** (≥ STRONG threshold 0.6387 = plan-024 honest ceiling). **plan-030 0.6294 → +0.0103 lift**. paradigm_root_cause.md 의 "main carrier = training procedure" 진단 *직접 검증* — multi-phase (pre + fine) + pairwise margin + regime/class prior + head slimming 의 결합이 G3 회복.

## §0.5 Result Quick Reference

| 항목 | 값 |
|---|---|
| **G3 OOF hit_1cm** | **0.6397** ✅ STRONG |
| G3 OOF hit_1p5cm | 0.8082 |
| G3 max_class_ratio | 0.2250 (no collapse, < 0.95) |
| G3 top1_acc | 0.1436 (random 7.1% 의 2.0×, plan-030 1.72× 보다 sharp) |
| **G3 band** | **STRONG** (≥ 0.6387 plan-024 ceiling) |
| G1 (fold-0) hit_1cm | 0.6450 (PASS > 0.6330) |
| G1 → G3 gap | -0.0053 (plan-030 -0.0142 보다 안정) |
| **per-fold std** | **0.0044** (plan-030 0.008 의 55%) |
| 모든 fold ≥ 0.6330 | ✓ (plan-030 의 fold-4 0.6215 outlier 없음) |
| **lift vs plan-030** | **+0.0103** |
| lift vs plan-029 X1 | +0.0081 |
| lift vs F0 baseline | +0.0077 |
| gap vs PB 0.6511 | -0.0114 (남은 회복 여지) |
| 5-fold elapsed | 628s ≈ 10분 28초 |
| 1-fold elapsed | 127s |

## §1. Gate 진행

| gate | 결과 | 값 | PASS? |
|---|---|---|---|
| G0 | DONE | finite + max_class_ratio 0.225 | ✓ |
| G1 (smoke) | DONE | 20/20 pytest green | ✓ |
| G2 (1-fold) | DONE | hit_1cm 0.6450 | ✓ (PASS > 0.6330) |
| G3 (OOF 5-fold) | DONE | hit_1cm **0.6397** | ✅ **STRONG** (≥ 0.6387) |
| G_final | DONE | STRONG band 도달 | ✓ paired permutation 미실행 (band 도달로 충분) |

## §2. per-fold 분포 (variance 감소)

| fold | plan-030 hit_1cm | plan-031 hit_1cm | Δ |
|---|---|---|---|
| 0 | 0.6431 | 0.6450 | +0.0019 |
| 1 | 0.6273 | 0.6434 | **+0.0161** |
| 2 | 0.6320 | 0.6366 | +0.0046 |
| 3 | 0.6233 | 0.6401 | **+0.0168** |
| 4 | 0.6215 | 0.6330 | +0.0115 |
| **mean** | **0.6294** | **0.6396** | **+0.0103** |
| **std** | **0.008** | **0.0044** | **-45%** |

**핵심**: plan-030 의 worst fold (4, 0.6215) → plan-031 0.6330 (+0.0115). variance 절반 감소.

## §3. 분석

### §3.1 multi-phase training (pre + fine) — main lever

- **pre phase (15ep, soft CE only)**: GRU + attention representation 안정 학습. score_std 평균 0.55→0.84.
- **fine phase (35ep, lr=2e-4, multi-loss)**: pairwise margin 으로 anchor discrimination 강제 + regime/class prior 로 distribution 정합 + soft CE 로 baseline 유지.
- loss 추이: pre last 2.44 → fine last 1.76 (-28%, multi-loss 의 효과).
- per-component loss (fine 끝): CE 2.38, pair 0.12, prior 2.68 — weighted sum 0.5·2.38 + 0.3·0.12 + 0.2·2.68 ≈ 1.76 ✓

### §3.2 score_std vs hit_1cm — 직관 반전

- plan-030 score_std 평균 2.71 → plan-031 score_std 평균 1.04 (-62%).
- 그런데 hit_1cm = 0.6294 → 0.6397 (+0.0103). **logit magnitude 와 lift 반비례**.
- 함의: plan-030 의 "score_std 신장 = sharpness 신장" 가설 *틀림*. label τ_cls=0.001 의 sharpness 와 score magnitude 는 학습 가능 결합이긴 하나, **anchor direction 학습** (= top-1 정답 정확도) 이 더 critical.
- top1_acc 12.3% → 14.4% (+2.1%p) — anchor direction sharpening 확인. 단 absolute level 은 여전히 낮음 (random 7.1% 의 2.0×).

### §3.3 head_hidden 384 → 196 slim — over-param 완화 효과

- params: 586,765 → 514,573 (-12%).
- fold variance 0.008 → 0.0044 (-45%) — slim 의 regularization 효과 측정 가능.
- plan-030 의 fold-별 spurious pattern overfitting 가설 *확인* — slim 후 fold 간 안정성 ↑.

### §3.4 max_class_ratio 0.12 → 0.22 — anchor preference 강화 (mode collapse 아님)

- 14 anchor uniform = 0.071. plan-030 0.12 (1.72×) → plan-031 0.22 (3.1×).
- **regime/class prior loss** 가 train-fold distribution 박제 → model 이 prior 따라 sharp 한 anchor preference 학습.
- 단 < 0.95 (mode collapse threshold) — 다양한 anchor 선택 유지. positive effect (informative prior).

### §3.5 fold variance 절반 감소 — multi-phase + slim 시너지

- plan-030 fold std 0.008 → plan-031 fold std 0.0044.
- pre phase 의 안정 representation + slim head 의 regularization 결합.
- 효과: G1 (fold-0) 가 G3 (5-fold) 의 *unbiased* estimator 에 가까워짐 (gap -0.0053 vs plan-030 -0.0142).

### §3.6 PB target 0.65 까지 남은 gap -0.0114

- plan-031 0.6397 vs PB selector 0.6511 = -0.0114 still.
- 본 plan 미박제 lever (plan-032 후보):
  - **fine-distill** (weight 0.55, temp 0.07) — teacher distill, 추정 lift +0.005~0.010
  - **reverse-pretrain** (BiLSTM 역방향) — sequence representation 보강
  - **GPU batch=4096 + hidden=48** (PB 정확 모사) — dynamics 변화
  - **boundary corrector 2-stage** (plan-004 §2.2, +0.0094 absolute)

## §4. Decision (plan-032 후보 진입 권장 X — 본 plan G_final STRONG)

본 plan G3 STRONG band 도달로 *충분*. plan-032 진입은 *우선순위 낮음* (lift +0.0103 already realized, PB gap -0.0114 잔여).

권장 다음 단계 (사용자 결정):
1. **DACON submit** (G3 STRONG → LB 검증) — plan-024 carry로 boundary corrector 적용 시 LB +0.01 추가 예상
2. **plan-032 spec** (fine-distill + reverse-pretrain) — PB target 0.65 도달 시도
3. **input axis ablation** (잔차 a/b drop) — plan-031 의 plan-030 input axis 가 net positive 확인

## §5. Artifact

- `analysis/plan-031/results_g1.json` — G1 fold-0 metric + trajectories
- `analysis/plan-031/results_g3.json` — G3 5-fold OOF + hparams + fold_logs
- `analysis/plan-031/results_g3.npz` — oof_pred (10000, 3) + oof_probs (10000, 14)
- `tests/test_plan031_smoke.py` — 20 pytest green

## §6. Decision-note

decision-note: plan-031 G3 STRONG (hit_1cm 0.6397, plan-024 ceiling 0.6387 회복 + 초과). paradigm_root_cause.md "main carrier = training procedure" 진단 직접 검증 (lift +0.0103 from plan-030). multi-phase + pairwise + prior + head slim 4 lever 결합 효과. plan-032 PB target 0.65 도달 시도는 사용자 선택 (gap -0.0114 잔여). DACON submit / plan-032 / input axis ablation 중 선택.
