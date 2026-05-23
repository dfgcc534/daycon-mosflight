---
plan_id: plan-032
status: complete
based_on: plan-031
followed_by: null (반전 부분 성공, axis B 단독 +0.0041, PB target -0.0073 미달)
title: PB target 0.6511 도달 시도 — multi-axis ablation (τ_cls 완화 + boundary corrector + fine-distill)
data_window: train_labels.csv 전체 (plan-024/29/30/31 carry)
fold_pin: stable_fold_id(sid, 5) (plan-024 carry, MD5)
horizon: 80ms (2 step × DT=40ms)
g_final_hit_1cm: 0.6438
g_final_band: STRONG
best_axis: B
best_axis_lift: +0.0041
pb_target_reached: false
---

# plan-032 — PB target 0.6511 도달 시도 (multi-axis ablation)

## §0. 한 줄 목적

plan-031 G_final STRONG = 0.6397, PB selector ensemble argmax = 0.6511 (gap **-0.0114**), PB selector + boundary corrector = 0.6718 (gap **-0.0321**). plan-032 = **multi-axis ablation** 으로 gap 회복 시도 + 각 axis 의 단독 기여 측정. epoch 100 ablation (under-fit 가설 검증) = **over-fitting confirm** (Δ=-0.0089), 50 epoch sweet spot 박제.

## §0.5 Quick Reference

| 항목 | 값 |
|---|---|
| baseline (plan-031 G3) | hit_1cm 0.6397, STRONG band |
| PB selector ensemble argmax | 0.6511 (gap -0.0114) |
| PB selector soft | 0.6624 (gap -0.0227) |
| PB selector + boundary corrector OOF | 0.6718 (gap -0.0321) |
| epoch 100 ablation | hit_1cm 0.6361 (Δ=-0.0089) — **over-fit confirm** |
| **plan-032 PASS band** | **hit_1cm ≥ 0.6511** (PB ensemble argmax 도달) |
| plan-032 EXCELLENT band | hit_1cm ≥ 0.6624 (PB selector soft) |

### Axis 우선순위 (cheap first)

| # | axis | cost | 예상 lift | 가능성 | blocker |
|---|---|---|---|---|---|
| A | **label τ_cls 완화** (0.001 → 0.005 / 0.01) | 매우 낮음 (1 hparam, 10분 G3 × 2) | +0.003~+0.010 | ★★★★★ | 없음 |
| B | **boundary corrector 재구현** (14-anchor 정합) | medium (재구현 + 학습) | +0.005~+0.015 | ★★★ | PB 27-cand ≠ plan-031 14-anchor paradigm 변환 |
| C | **fine-distill** (self-distillation 또는 label smoothing) | medium | +0.005~+0.010 | ★★★ | (PB teacher 변환 시) 27→14 cast |
| D | input axis ablation (잔차 a/b drop) | 낮음 (1-fold) | 0 (informative) | ★★★★ | — |

### Commit chain

| commit | axis | TODO |
|---|---|---|
| c0 spec | — | DONE (본 commit) |
| c1 ablation A — τ_cls 완화 | A | [TODO] τ_cls ∈ {0.005, 0.01} G3 × 2 |
| c2 ablation D — input axis 단독 | D | [TODO] 잔차 (a)/(b) drop G2 × 2 (informative) |
| c3 ablation B — boundary corrector 재구현 | B | [TODO] 14-anchor 정합 + 학습 + G3 |
| c4 ablation C — fine-distill | C | [TODO] label smoothing G3 |
| c5 axis 결합 (best 2-3) | — | [TODO-cond] 단독 Δ > 0.005 axis 만 결합 G3 |
| c6 G_final + results | — | [TODO] `plans/plan-032.results.md` |

---

## §1. Failure attribution carry (plan-031.results.md §3 carry)

plan-031 0.6397 ceiling 원인:
1. **score_std 1.04** vs label sharpness 1/τ=1000 → 3 ord magnitude gap (학습 불가능 target)
2. **top1_acc 14.4%** = random 2.0× (PB 5-7× 추정)
3. **50 epoch sweet spot** — over-fitting confirmed (epoch 100 → -0.0089)
4. **pair loss 0.04** = ranking 완료, 더 학습 여지 없음

→ main lever = label distribution 정합성 (A) + post-process corrector (B) + label smoothing (C)

---

## §2. Architecture carry (plan-031 그대로)

- GRUNetX3 (head_hidden 196 slim) carry
- Multi-phase training (pre 15 + fine 35 = 50 epoch) carry
- pairwise margin 0.12 + regime/class prior 0.65/0.45 carry
- input axis (잔차 a/b + Q=anchor + slim 7 + head sample 51) carry

각 ablation 의 단일 변수 변경만.

---

## §3. Ablation spec

### §3.1 Ablation A — label τ_cls 완화

**Hypothesis**: τ_cls=0.001 (sharpness 1000) 가 model score_std 1.04 와 mismatch. 완화 시 realizable target.

**Variants**:
- A1: τ_cls = 0.005 (sharpness 200)
- A2: τ_cls = 0.01 (sharpness 100, PB temperature=0.07 비슷)

**Procedure**: plan-031 train.py 의 `TAU_CLS` constant override. 다른 carry. G3 5-fold × 2 = ~20분.

**PASS criterion**: max(ΔA1, ΔA2) > 0.005 → axis A carry 진행.

### §3.2 Ablation B — boundary corrector 재구현 (14-anchor)

**Hypothesis**: PB boundary corrector +0.0207 lift (0.6511 → 0.6718). 14-anchor paradigm 재구현 시 +0.005~0.015 (paradigm 다름, conservative).

**Mechanism** (plan-031 OOF probs/pred 기반 post-process):
1. plan-031 OOF probs (N=10000, K=14) + final_pred (N, 3) 로드
2. per sample: top-3 anchor prob + top-3 anchor world pos (= F0 + R_wfn @ ANCHORS[top3]) + sample state summary (= GRU hidden last + Bz/Tz + macro stats)
3. TinyCorrectionNet (MLP): input ~ [top-3 prob 3 + top-3 anchor world pos 9 + F0 3 + sample summary 51 + R_wfn 9] = 75D → hidden 128 → output 3D delta
4. corrected_pred = final_pred + delta
5. 학습 (5-fold OOF, train fold 의 corrector 가 test fold 에 inject): loss = huber(corrected_pred, gt)

**Procedure**: `analysis/plan-032/boundary_corrector_14a.py` 신규. 5-fold corrector 학습 (~15분).

**PASS criterion**: Δ > 0.010 → carry.

### §3.3 Ablation C — fine-distill (label smoothing first)

**Hypothesis simple version**: fine phase 의 soft label τ_cls=0.001 에 uniform mixing (label smoothing 0.1) 적용. PB teacher 없이 cheap 검증.

**Variant**:
- C1: q_smooth = 0.9 * q_orig + 0.1 / K (label smoothing 0.1)
- C2: q_smooth = 0.85 * q_orig + 0.15 / K (label smoothing 0.15)

**Procedure**: plan-031 train.py 의 fine phase loss 만 변경. G3 5-fold × 1 = ~10분 (best variant).

**PASS criterion**: Δ > 0.005.

### §3.4 Ablation D — input axis 단독 (informative, not lift)

**Variants**:
- D1: 잔차 (a) GRU input + K/V drop (seq 95D 만, K/V H 만)
- D2: 잔차 (b) Q from query drop (Q = 29D)

**Procedure**: plan-031 train.py 의 input flag 추가. 1-fold (G2) × 2 = ~8분.

**Purpose**: informative measurement of input axis 의 net contribution.

### §3.5 Multi-axis 결합 (c5, conditional)

단독 Δ > 0.005 인 axis 만 결합. G3 1회 (~10분).

---

## §4. 합격 기준 + Gate

| gate | 검사 | PASS band |
|---|---|---|
| G0 | finite + max_class_ratio < 0.95 | per ablation |
| G2 (ablation A/B/C) | 5-fold OOF hit_1cm Δ | per axis (§3.1~§3.3) |
| **G3 (multi-axis 결합)** | 5-fold OOF hit_1cm | **≥ 0.6511 (PB selector ensemble)** = PASS, ≥ 0.6624 EXCELLENT, ≥ 0.6718 SUPERIOR |
| G_final | G3 PASS + paired permutation p<0.005 vs plan-031 | report 박제 |

### Fallback

- G3 모든 ablation Δ < 0.005 → **반전 실패**, plan-031 STRONG 유지 + DACON submit 권장.
- G3 일부 PASS → 그 axis 만 plan-031 patch 적용.

---

## §5. References

- `analysis/plan-031/results_g3.json` — baseline hit_1cm 0.6397
- `analysis/plan-031/results_g3.npz` — OOF probs + final pred
- `plans/plan-004-pb-0-6822-fullrun.results.md` — PB metric source
- `src/pb_0_6822/boundary.py` — boundary mechanism (참고)
- `analysis/plan-031/{train,model,pairwise_loss,prior_loss}.py` — carry source

---

## §6. Reference (이전 plan-030 시점 draft, 일부 lever 호환성 분석)

이전 draft 에서 plan-030 적용 가능성 평가 (plan-031 진행 후 invariant lever 만 carry):

| lever | 적용 평가 |
|---|---|
| pairwise margin loss | ★★★★★ — plan-031 c1 박제 (적용 완료) |
| multi-phase (pre/fine) | ★★★★ — plan-031 c4 박제 (적용 완료, freeze 변형 제외) |
| fine-distill (teacher snapshot) | ★★★★★ — 본 plan §3.3 axis C |
| reverse-pretrain | ★★ — BiLSTM 변경 필요, §7 deferred 유지 |
| batch=4096 | ★★ — CPU 환경 제약, §7 deferred |
| hidden=48 | ★★★ — capacity 부족 위험, plan-031 hidden 196 carry |
| norm-real-only | ★★★★★ — 자동 충족 (augmentation 없음) |

decision-note: plan-032 = multi-axis ablation. cheap first (τ_cls → input ablation → boundary → fine-distill). 단독 Δ > 0.005 axis 만 결합. PASS = PB target 0.6511. fallback = plan-031 STRONG 유지 + DACON submit.
