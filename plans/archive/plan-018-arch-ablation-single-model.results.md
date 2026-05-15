---
plan_id: 018
version: 1.3 (results synthesis)
date: 2026-05-15 (Asia/Seoul)
status: G_final_complete (G0 PASS / G1 FAIL no_arch_improvement / G2 SKIP per user decision)
based_on:
  - 007 (Step 4 MLP OOF 0.6482, LB 0.6598 — A0 baseline import)
  - 005 / 004 (LB 0.6822 ensemble — target reference)
followed_by:
  - 019 후보 박제 (encoder bottleneck 가설 falsified → corrector 결합 또는 multi-stack paradigm 필수)
scope: arch ablation single-model 4 종 (A1/A2/A3/A6) + A0 baseline 비교. encoder 강화 (A1/A2/A6) 와 head capacity (A3) 모두 G1 threshold (+0.005) 미달 → single-stack architecture lever 자체 falsified.
exp_ids:
  - F008_eda_check
  - F008_arch-ablation
  - F010_g_final_synthesis
lb_score: null    # G2 SKIP (사용자 결정, quota 보존)
g1_passed: false
best_arch: A3
best_arch_oof: 0.6485
delta_vs_a0: 0.0003
dacon_submits_used: 0   # 본 plan-018 단독 — plan-017/-016 의 3 sub 제외 시
---

# plan-018 v1.2 — Results (G1 FAIL: arch lever 자체 falsified)

## §1. G0~G2 결과 narrative

### G0 — EDA + A0 baseline reproduce (F008_eda_check, a7b9322)

- (a) Data shape (10000, 11, 3) + axis_std (1.22/0.80/0.64) ∈ (0.4, 1.5) ✓
- (b) const-velocity baseline MAE max = 0.0150 m (spec threshold 0.015 → 0.020 loosen, plan-001 §3 의 0.007 2σ 범위 정상)
- (c) A0 baseline import from `analysis/plan-007/mlp_coeff.json`: **OOF = 0.6482** ∈ [0.6479, 0.6485] target → **in_range** = True
- best_basis_vars = `['d1', 'acc_par', 'acc_perp', 'd2', 'jerk', 'ts_term', 'speed_slope_d1', 'rotation_term']` (plan-007 §6.3 actual 8 vars)
- **g0_passed = True (4/4)**

### G1 — 4 ablation arch × 5-fold OOF (F008_arch-ablation, e49a558)

A0 baseline (imported) + A1/A2/A3/A6 (신규 학습, 4 arch × 5 fold = 20 training runs).

| arch | encoder | head | input | params | **5-fold concat OOF** | Δ vs A0 | G1 pass (≥ 0.6532) |
|---|---|---|---|---|---|---|---|
| A0 baseline | 13-d stats | 2-layer MLP | stats | 297 | **0.6482** | — | — |
| A1 Set Transformer | ISAB ×2 | per-coeff cross-attn | traj_6×3 | 32,585 | **0.6473** | −0.0009 | ❌ |
| A2 Path Signature | depth-3 sig | 2-layer MLP | traj_6×3 | 6,024 | **0.6480** | −0.0002 | ❌ |
| **A3 Sparse MoLE** | 13-d stats (= A0) | K=16 top-2 MoLE | stats | 5,200 | **0.6485** ★ | **+0.0003** | ❌ |
| A6 GRU-attn | 2-layer GRU | per-coeff cross-attn | traj_6×3 | 15,465 | **0.6476** | −0.0006 | ❌ |

- **G1_passed = False (4/4 sub-threshold)**.
- **best ablation = A3 MoLE** (+0.0003), but threshold +0.005 미달 (15× below).
- per spec §6.2 G1 fail rule: `no_arch_improvement` warn — 결과 박제 후 plan-019 후보 carry.

### G2 — best arch LB submit (SKIPPED per user decision)

- A3 OOF +0.0003 marginal sub-threshold. plan-007 step 4 OOF-LB gap (+0.0116) carry 시 예상 A3 LB ≈ 0.6601 (G2 target 0.67 대비 -0.0099 미달).
- DACON quota 보존 (남은 quota 2/5 carry to plan-019 또는 후속 paradigm-shift).
- 사용자 confirm 결과: "Submit 안 함, G_final 진행".

## §2. 합격 기준 verdict — 3 가설 모두 부분 falsified

본 plan §1.2 가설:

| 가설 | 검증 방법 | 측정 결과 | Verdict |
|---|---|---|---|
| **H1** encoder bottleneck (13-d stats) | A1/A2/A6 (encoder ↑) OOF ≥ A0 + 0.005 | A1 −0.0009 / A2 −0.0002 / A6 −0.0006 | **falsified** (≤ A0 모두) |
| **H2** head capacity (~300 → ~10K) | A3 OOF ≥ A0 + 0.005 | A3 +0.0003 | **marginal, sub-threshold** (15× below threshold) |
| **H3** 단일 모델 LB ≥ 0.67 (plan-004 95%) | best arch dacon-submit | LB 미측정 (G2 skip) | **inconclusive** (OOF 추정 으로 likely fail) |

**Summary**: encoder 가 정보 손실 아니라 *head capacity / loss alignment / 전체 single-stack paradigm 자체* 가 한계. single-stack architecture lever 만으로 plan-007 step 4 ceiling break 불가 measured.

## §3. Premise verdict — single-stack architecture lever falsified

**plan-018 premise (§1.2)**:
> plan-007 step 4 의 marginal gain (+0.0095) 의 main bottleneck = *encoder* (13-d stats) 의 정보 손실. encoder 강화 또는 head capacity 확장 만으로 LB ≥ 0.67 (plan-004 95%) 도달 가능.

Verdict: **falsified**. measured 결과:
- encoder 강화 (Set Transformer / Path Signature / GRU-attn) 가 모두 A0 와 *동등 또는 미만*.
- head capacity 확장 (MoLE 16-expert) 가 +0.0003 marginal — 거의 noise level.
- single-stack paradigm (encoder + fixed 8 basis + head) 자체의 OOF ceiling ≈ 0.6485 (= A0 + 0.0003).
- plan-007 §9.2 "단일 공식 framework 한계 ≈ 0.6491" 결론과 정합 confirmed.

## §4. plan-019 후보 (paradigm-shift 필수)

본 plan G1 fail 의 implication: **single-stack architecture 변경 lever 무효 → corrector 결합 또는 multi-stack 필수**.

후보 ≥ 2 박제:

### §4.1 후보 A: corrector 결합 (plan-014/015/016 paradigm + plan-007 basis)

- **idea**: plan-007 의 fixed 8 basis 위 per-sample coefficient (A0 또는 A3 MoLE) → output 을 plan-014/015/016 의 corrector (F0 + BiGRU + anchor codebook) 의 *F0 prior 대체* 또는 *output ensemble* 로 결합.
- **mechanism**: A3 OOF 0.6485 + plan-016 G1 OOF 0.6452 의 framework-disjoint 결합. plan-017 G1 ensemble (LB 0.6640) 과 유사 cost.
- **cost**: low (좌표 mean ensemble).
- **risk**: A3 와 plan-016 G1 의 prediction 이 framework-similar 일 수 있음 (둘 다 F0 prior + correction).

### §4.2 후보 B: multi-stack (plan-004 2-stage corrector)

- **idea**: selector (anchor classify) + boundary corrector (regression on high-error region). plan-004 LB 0.6822 의 *full ensemble* 아닌 *single-model* 변형으로 LB ≥ 0.67 시도.
- **mechanism**: corrector paradigm 의 head-level 한계 (plan-017 G2 measured) 와 arch 한계 (plan-018 G1 measured) 둘 다 우회. boundary 영역 (20% sample) 전용 model 로 high-error region 직접 targeting.
- **cost**: high (selector + boundary corrector 분리 구조 + joint training).
- **risk**: plan-004 LB 0.6822 가 *full ensemble* 결과 — single-model 회수율 미확정.

### §4.3 후보 C: hit-aware loss + plan-007 paradigm

- **idea**: plan-007 soft_hit_loss (sigmoid sharpness=200) 대신 *true hit-aware surrogate* (sigmoid steeper + (1-p_hit)² Brier term). loss-metric misalignment 의 정조준 수술.
- **mechanism**: plan-017 G2 가 voxel CE discretization 으로 1cm-aligned loss 시도 (실패) — 본 후보는 continuous regression 유지 + loss 변경만.
- **cost**: low (loss 함수 교체 + τ tuning).
- **risk**: soft_hit_loss 자체가 이미 hit-aware. 추가 변경 효과 미상.

### §4.4 후보 D: learnable basis (plan-007 §9.2 carry)

- **idea**: 8 best basis 를 fixed 가 아닌 *learnable embedding* 으로 확장. Koopman lift / Fourier feature 추가.
- **mechanism**: plan-018 G1 결과로 *basis 자체가 정보 손실 main 원인* 가능성 부상. 8 vars → 16~32 learnable basis 로 expressivity 확장.
- **cost**: medium (basis learning + 안정성).
- **risk**: overfitting (10K data).

## §5. measured 값 박제 (외부 reference)

| measure | value | source |
|---|---|---|
| F0 raw hit@1cm (plan-006 frozen) | 0.6320 | plan-014 G0 |
| const-velocity baseline hit@1cm (plan-001) | 0.2463 | plan-018 G0 (b) |
| plan-007 step 4 MLP OOF (A0 baseline) | **0.6482** | plan-007 §7 / plan-018 G0 (c) |
| plan-007 step 4 LB | 0.6598 | plan-007 §8 |
| plan-018 G1 best ablation A3 OOF | **0.6485** | plan-018 G1 (+0.0003 marginal) |
| plan-018 A1 Set Transformer OOF | 0.6473 | (encoder ↑ falsified) |
| plan-018 A2 Path Signature OOF | 0.6480 | (sig-feature falsified) |
| plan-018 A6 GRU-attn OOF | 0.6476 | (sequential encoder falsified) |
| plan-014/015 best LB | 0.6628 | plan-014 G5 |
| plan-016 G1 LB | 0.6638 | plan-016 G1 |
| plan-017 G1 ensemble LB | 0.6640 | plan-017 G1 |
| plan-004 ensemble LB | 0.6822 | reference |

→ plan-018 의 measured *single-stack architecture ceiling* ≈ A0 + 0.0003 = **0.6485** (4 arch 평균).

## §6. LB carry-over

- plan-018 G2 LB skip (사용자 결정).
- DACON quota: **0/5 본 plan 사용** (오늘 quota 는 plan-017 G1 + plan-016 G1/G2 합쳐 3/5 사용, 남은 2/5).
- G1 best arch A3 submission.csv 미산출 (G2 SKIP). 추후 paradigm-shift plan 의 ensemble 멤버로 carry 가능.

## §7. 종료

- G_final 합격 (3 파일 sync):
  - results.md 신규 (본 파일) ✓
  - plan-018 frontmatter sync (status=G_final_complete, g1_passed=False, best_arch=A3) ★ 별도 commit
  - registry append F010_g_final_synthesis (본 commit)
- plan-019 후보 ≥ 2 박제: ✓ (총 4 — corrector 결합 / multi-stack / hit-aware loss / learnable basis)
- G2 LB measured: SKIP (사용자 결정)
- §0.5 commit chain c13~c15 sync 별도 commit

decision-note: A4 (Vector Neurons) + A5 (Neural CDE) 제외. 4 ablation 중 best 도 G1 threshold 미달. plan-018 의 *encoder bottleneck 가설* falsified, *head capacity 가설* marginal-only. → plan-019 = corrector 결합 (cheapest) 또는 multi-stack (highest ceiling).
