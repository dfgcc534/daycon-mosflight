---
plan_id: plan-030
status: complete
based_on: plan-029
title: GRU-attention residual injection (results)
g3_oof_hit_1cm: 0.6294
g3_band: FAIL_regression
g1_oof_hit_1cm: 0.6436
followed_by: plan-031 (PB training procedure carry — main carrier per paradigm_root_cause.md)
---

# plan-030 results

## §0. 한 줄 결론

**G3 FAIL_regression**: 5-fold OOF hit_1cm = **0.6294** < F0 baseline 0.6320 (Δ = **−0.0026**), plan-029 X1 0.6316 대비도 **−0.0022 worse**. 잔차 (a)/(b) input feature axis fix 만으로는 부족 — `analysis/plan-029/paradigm_root_cause.md` 의 "main carrier = training procedure" 진단 확정. **plan-031 PB training procedure carry 진입 권장**.

## §0.5 Result Quick Reference

| 항목 | 값 |
|---|---|
| **G3 OOF hit_1cm** | **0.6294** |
| G3 OOF hit_1p5cm | 0.8033 |
| G3 max_class_ratio | 0.1227 (no mode collapse) |
| G3 top1_acc | 0.123 |
| G3 band | **FAIL_regression** (< F0 0.6320) |
| G1 (fold-0) hit_1cm | 0.6436 (PASS, F0 +0.0116) |
| G1 → G3 gap | **−0.0142** (cross-fold variance 큼) |
| F0 baseline | 0.6320 |
| plan-029 X1 (carry) | 0.6316 |
| plan-024 honest ceiling | 0.6387 |
| 5-fold elapsed | 681s ≈ 11분 24초 (CPU) |
| 1-fold elapsed | 140s |
| N_total | 10000 |
| K (anchors) | 14 |

## §1. Gate 진행

| gate | 결과 | 값 | PASS? | 사유 |
|---|---|---|---|---|
| G0 (data) | DONE | finite, max_class_ratio 0.1227 | ✓ | upstream cache 정합 |
| G1 (smoke) | DONE | 20/20 pytest green | ✓ | builder + model + train + finite |
| G2 (1-fold) | DONE | hit_1cm 0.6436 | ✓ (PASS > 0.6290) | fold-0 noise 가능성 — G3 와 큰 gap |
| G3 (OOF 5-fold) | DONE | hit_1cm 0.6294 | ✗ **FAIL_regression** | F0 baseline -0.0026, plan-029 X1 -0.0022 |
| G_final | N/A | — | ✗ | G3 fail → §5 fallback rule = plan-031 escalate |

## §2. 분석

### §2.1 G1 (0.6436) vs G3 (0.6294) gap = -0.0142 — 비정상적 큰 cross-fold variance

- fold-0 의 N_te = 2020, 5-fold concat OOF N_total = 10000.
- single fold 의 noise ≈ 1.96 × sqrt(0.65 × 0.35 / 2020) ≈ 0.021. G3 의 cross-fold noise (1/5 sqrt) ≈ 0.009.
- 관측 gap −0.0142 는 single fold noise 범위 안 (≤ 0.021) 이나, *systematic* drift 가능성.
- 추정 원인: 잔차 (a)/(b) input 의 zero-pad step (i=5,6, t_wall=-1,0) 이 fold 별로 다른 distribution shift 야기. fold 0 는 마침 zero-pad 가 less harmful 한 sample 분포.
- 또는 head MLP input dim 382 가 over-parameterized 로 fold 별 overfitting → cross-fold variance ↑.

### §2.2 plan-029 X1 0.6316 → plan-030 0.6294 = -0.0022 worse

- 잔차 (a) 35D + (b) 35D per anchor 추가가 **net negative** lift.
- 가설 검증 결과: paradigm_root_cause.md 의 진단 = "main carrier = training procedure (multi-phase + pairwise + prior + distill + reverse-pretrain)" — 본 plan 의 input feature axis fix 만으로는 회복 불가능 확인.
- 잔차 raw 추가 효과 ≈ 0 또는 slight negative. K/V 의 raw bottleneck-bypass 가 attention 압축 손실 회피 목표였으나, single-head dot-product attention 의 K/V tied 구조 (parameter 절약) 가 잔차 raw 신호를 효과적으로 활용 못함 가능성.

### §2.3 score_std trajectory = 0.78 → 2.74 (logit-scale)

- 50 epoch 끝 logit-score std ≈ 2.7 — fallback rule threshold (100) 보다 훨씬 작음. label τ_cls=0.001 의 sharpness factor 1/τ_cls = 1000 까지 신장 *실패*.
- 단 PASS 조건 (G2 threshold) 은 통과 → fallback rule trigger 안 됨.
- 함의: model 이 logit magnitude 를 더 키워야 label distribution 정합 향상 가능. learnable temperature scalar τ_model (fallback rule §5) 또는 score scale auxiliary loss 가 plan-031 후보.

### §2.4 max_class_ratio 0.1227 — no mode collapse

- 14 anchor 의 uniform = 1/14 ≈ 0.0714. 0.1227 = 1.72× uniform — 약간의 anchor preference 있지만 mode collapse (≥ 0.95) 와는 거리 멀음.
- top1_acc 0.123 도 비슷 — top-1 prediction 의 정확도가 random (1/14=0.0714) 의 1.72× 수준. anchor selection 자체는 *informative* 하지만 충분히 sharp 하지 않음.

## §3. Decision (plan-031 진입)

plan-030 §5 fallback rule 따라:
> **G3 fail → plan-031 spec 진입 (= PB training procedure carry)**

plan-031 의 main lever 후보 (paradigm_root_cause.md L67-78 carry):
1. **PB multi-phase training** (pre + fine-tune + freeze_fine + epoch_plus) — 추정 lift +0.015 ~ +0.020
2. **pairwise margin loss** (margin 0.12) — anchor discrimination 강제
3. **regime/class prior** (strength 0.65 / 0.45) — inductive bias
4. **fine-distill** (weight 0.55, temp 0.07) — teacher distill
5. **reverse-pretrain** (BiLSTM 역방향)
6. **GPU batch=4096 + hidden=48** (PB 정확 모사)

본 plan 의 input feature axis (residual block + Q/K 정합) 은 plan-031 carry — 잔차 raw 가 PB training procedure 와 결합 시 lift 발생 가능성 검증 필요.

## §4. Ablation (§7 deferred, not run)

§5 fallback rule 의 c10~c13 ablation (잔차 a/b/slim7/head sample summary drop) 은 본 plan 에서 *미실행* — main lever (training procedure) 가 부재한 상태에서의 ablation 은 정보 가치 낮음. plan-031 의 PB carry 후 input axis ablation 재진행 권장.

## §5. Artifact

- `analysis/plan-030/results_g1.json` — G1 fold-0 metric + score_std_trajectory
- `analysis/plan-030/results_g3.json` — G3 5-fold OOF metric + hparams + fold_logs
- `analysis/plan-030/results_g3.npz` — oof_pred (10000, 3) + oof_probs (10000, 14)
- `tests/test_plan030_smoke.py` — 20 pytest green

## §6. Decision-note 박제

decision-note: plan-030 G3 FAIL_regression (hit_1cm 0.6294 < F0 0.6320). cross-fold variance −0.0142 비정상. paradigm_root_cause.md "main carrier = training procedure" 진단 확정. §5 fallback rule 따라 plan-031 PB training procedure carry 진입 권장. input feature axis (residual block + Q/K 정합) 는 plan-031 carry, ablation c10~c13 은 plan-031 의 input axis ablation 단계로 미룸.
