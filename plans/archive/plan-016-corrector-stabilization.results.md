---
plan_id: 016
version: 1.6 (results synthesis)
date: 2026-05-15 (Asia/Seoul)
status: G_final_complete (G1 marginal_under_threshold / G2 negative_drop / G3-G5 sub-threshold, G6 SKIP per user submit 결정)
based_on:
  - 014 (LB 0.6628 baseline carry)
  - 015 (Feature A negative drop rule)
followed_by:
  - 017 (paradigm-shift branch — stabilization 3 lever 모두 sub-threshold, single-feature/multi-seed/val_loss 한계 confirmed)
scope: plan-014/015 corrector paradigm 의 3 limitation (L1 Feature redundancy / L2 5-fold OOF variance noisy / L3 monitor=val_hit discrete jump) 을 직접 닫는 3-path sequential ablation 결과 박제. Path A (multi-seed) 부분 회수 (LB +0.0010, OOF +0.0027 둘 다 +0.005 threshold 미달). Path B (val_loss) negative_drop. Path C (Feature B/C/D 단독) 3/3 sub-threshold. paradigm-level 한계 measured.
exp_ids:
  - H049_g0_preflight
  - H050_g1_path_a_multiseed
  - H051_g2_path_b_val_loss
  - H052_g3_path_c_b
  - H053_g4_path_c_c
  - H054_g5_path_c_d
  - H056_g_final_synthesis
lb_score: 0.6638
lb_band: positive
band: positive
g6_status: skip
best_5fold_oof: 0.6452
delta_oof: 0.0027
oof_lb_gap: 0.0186
dacon_submits_used: 2
---

# plan-016 v1.5 — Results (band=positive, G6 SKIP, paradigm 한계 measured)

## §1. G0~G5 결과 narrative

### G0 — preflight (H049_g0_preflight, fc6c1b1)

- **plan-014/015 baseline 5-fold concat OOF reproduce = 0.6425** (target 0.6425 ± 0.0005, in_range=True, 4자리 정확 일치).
- fold-wise: [0.6658, 0.6412, 0.6507, 0.6318, 0.6230] — plan-015 G0 와 deterministic same.
- seed list [20260514..20260518] len=5 unique, monitor signature (val_hit/val_loss) OK, Feature B/C/D 단독 dim [10/18/15] all sanity OK.
- **G0_passed = True (4/4 checks)**.

### G1 — Path A multi-seed ensemble (H050_g1_path_a_multiseed, 19d57f6)

5 seed × 5 fold = 25 models 학습 + OOF aggregation (좌표 mean over seeds → 5-fold concat).

| metric | value |
|---|---|
| per-seed concat OOF | [0.6425, 0.6459, 0.6428, 0.6432, 0.6420] |
| per-fold (seed-mean) OOF | {0: 0.6678, 1: 0.6431, 2: 0.6502, 3: 0.6308, 4: 0.6343} |
| **multi-seed concat OOF** | **0.6452** |
| Δ vs baseline 0.6425 | +0.0027 |
| OOF Δ pass (+0.005 threshold) | **False** |
| **LB** | **0.6638** |
| Δ vs baseline LB 0.6628 | +0.0010 |
| LB Δ pass | **False** |
| status | **marginal_under_threshold** |
| fold spread (max-min) | 0.0370 |

**L2 closure 분석**: fold spread 0.0370 — plan-014/015 의 0.04 와 거의 동등. multi-seed ensemble 이 *systematic fold variance* 를 못 잡음 (예상된 limitation, plan-016 §1.3 L143 박제 그대로). Variance source 가 seed (학습 randomness) 보다 fold split (data distributional bias) 임.

**Δ 분류 결정 narrative**: §5.3 literal "둘 다 fail → negative" vs §3.2 footnote v1.5 "negative = OOF Δ<0 AND LB Δ<0" 의 spec 충돌. §3.2 footnote 의 *명시적 정의* 가 governing → G1 = **marginal_under_threshold** (positive Δ 양방향, sub-threshold). G2~G5 진행.

### G2 — Path B monitor=val_loss (H051_g2_path_b_val_loss, a1b8ae9)

G1 carry + monitor=val_loss only (single 변경 lever).

| metric | value |
|---|---|
| per-seed concat OOF | [0.6364, 0.6373, 0.6382, 0.6400, 0.6390] |
| **multi-seed concat OOF** | **0.6414** |
| Δ vs G1 OOF 0.6452 | **-0.0038** |
| OOF Δ pass | **False** (negative direction) |
| **best_epoch std** | **2.21** ★ |
| L3 closure (target ≤ 3) | **달성** |
| **LB** | **0.6634** |
| Δ vs G1 LB 0.6638 | **-0.0004** |
| status | **negative_drop** |

**L3 closure 달성** but OOF/LB 둘 다 negative direction → §3.2 footnote v1.5 negative_drop → **drop Path B, G3-G5 base = G1 (val_hit, NOT val_loss)**.

**해석**: monitor=val_loss 는 train objective (`hybrid_combined_loss` = ring_classification + L1 huber + hinge) 를 optimize. eval metric (hit@1cm) 와 *misaligned* — loss 가 줄어도 hit 가 안 늘어남. best_epoch std 감소 (epoch noise 줄임) 효과 있지만 metric improvement 효과 없음 measured.

### G3 — Path C-B Feature B 단독 (H052_g3_path_c_b, efe761b)

| metric | value |
|---|---|
| feature | Feature B (binormal split, step-local Frenet basis) |
| input_dim | 10D (9D base − 1D `perp_norm/speed` + 2D `(acc_normal/speed, acc_binormal/speed)`) |
| per-seed concat OOF | [0.6432, 0.6436, 0.6424, 0.6431, 0.6410] |
| **multi-seed concat OOF** | **0.6443** |
| Δ vs G1 OOF 0.6452 | **-0.0009** |
| OOF Δ pass (+0.003 threshold) | **False** (negative direction) |
| LB | unsubmitted (사용자 결정) |
| status | negative direction (OOF only) |

**해석**: sign-aware binormal split 이 plan-014/015 의 `perp_norm/speed` 1D abs 보다 정보량 더 많을 것이라는 가설 — measured **falsified**. 9D base 의 `acc_par/speed`, `acc_perp/speed` 가 이미 충분.

### G4 — Path C-C Feature C 단독 (H053_g4_path_c_c, efe761b)

| metric | value |
|---|---|
| feature | Feature C (multi-scale stride τ=1,2 stream concat) |
| input_dim | 18D (9D base × 2 stream) |
| per-seed concat OOF | [0.6420, 0.6437, 0.6430, 0.6453, 0.6423] |
| **multi-seed concat OOF** | **0.6458** |
| Δ vs G1 OOF 0.6452 | **+0.0006** |
| OOF Δ pass (+0.003 threshold) | **False** (sub-threshold, +0.0006 < +0.003) |
| LB | unsubmitted |
| status | sub-threshold positive |

**해석**: stride τ=2 의 long-range kinematic 정보가 1D feature 추가 효과 (+0.0006) 정도. BiGRU 가 단일 stride 위 11-step temporal context 만으로 long-range 정보 흡수 한계.

### G5 — Path C-D Feature D 단독 (H054_g5_path_c_d, efe761b)

| metric | value |
|---|---|
| feature | Feature D (pairwise cross-step, 3 pair × 2 stat) |
| input_dim | 15D (9D base + 6D pairwise) |
| per-seed concat OOF | [0.6422, 0.6443, 0.6433, 0.6446, 0.6410] |
| **multi-seed concat OOF** | **0.6461** ★ |
| Δ vs G1 OOF 0.6452 | **+0.0009** ★ |
| OOF Δ pass (+0.003 threshold) | **False** (sub-threshold, +0.0009 < +0.003) |
| LB | unsubmitted |
| status | sub-threshold positive ★ (**best of 3 Path C**) |

**해석**: pairwise (cosine + Δspeed) cross-step interaction 이 가장 큰 positive Δ. 하지만 +0.0009 = noise 수준 (per-seed std ≈ 0.0013). single-feature lever 가 paradigm-level break 못 함.

### G6 — best stack [SKIP]

**SKIP 사유** (사용자 결정 박제):
- Path C 3/3 모두 +0.003 threshold 미달 (G3=-0.0009, G4=+0.0006, G5=+0.0009).
- Δ 가 noise 수준 (per-seed std ≈ 0.0013 ~ 0.0017) → LB 측정해도 signal/noise 비 낮음.
- DACON quota (3 남음) 보존 → plan-017 또는 추후 valid lever 검정 자원으로 carry.
- **G6 effective alias = G1** (G2 dropped + Path C 3/3 sub-threshold + unsubmitted, best LB 확실값 = G1 0.6638).

§10.1 best_c selection 의 v1.5 fix 외 추가 분기 발생: spec 는 "best_c = None → G2 alias" 만 정의, 실제 measured 케이스는 "G2 drop + Path C 3/3 unsubmitted → G1 alias". plan-017 spec 후보 (G6 fallback chain 더 일반화).

## §2. 합격 기준 verdict — plan-016 의 **measured limit**

plan-016 §0.5 ambition:
- **G6 pass = LB ≥ 0.6678** (baseline + 0.005, "stabilization minimal patch" 의 목표)
- **stretch goal = LB ≥ 0.68** (plan-004 정조준)

**Measured 최종 LB = 0.6638** (G1 carry, 단 LB submission 만 사용).
- vs G6 pass 0.6678: **miss by −0.0040**.
- vs stretch 0.68: miss by −0.0162.
- vs baseline 0.6628: **+0.0010 LB 회수**.
- vs plan-013 LB 0.6381: +0.0257.

**Verdict**: G6 pass **미달**. 그러나 LB band = positive (0.6638 ∈ [0.66, 0.68]) 유지.

### LB chain (§1.4 update)

| measure | value | source |
|---|---|---|
| plan-013 LB (fallback) | 0.6381 | plan-013 |
| plan-014/015 best_stack LB (baseline) | 0.6628 | plan-014 G5 |
| **plan-016 G1 LB (multi-seed)** | **0.6638** ★ | plan-016 G1 |
| plan-016 G2 LB (val_loss, dropped) | 0.6634 | plan-016 G2 |
| plan-016 G6 target | 0.6678 | spec |
| plan-016 stretch | 0.68 | spec |
| plan-004 (original) | 0.6806 | reference |

## §3. Premise verdict — 3 limitation 직접 닫기 가설 **부분 falsified**

**plan-016 premise (§0 한 줄 목적)**:
> plan-014/015 의 *3 가지 measured limitation* (L1 Feature redundancy / L2 5-fold OOF variance / L3 monitor=val_hit discrete jump) 을 *직접* 닫음으로써 corrector paradigm 위 LB 0.6628 → 0.68 회수.

**Verdict per closure mechanism** (Mechanism mapping §1.3):

| limit | Path | closure mechanism | measured 결과 |
|---|---|---|---|
| **L2** (fold variance) | Path A multi-seed | fold spread ≤ 0.02 | **不_closed** (0.037, plan-015 0.04 와 동등). multi-seed 가 *seed variance* 만 잡고 *fold distributional bias* 못 잡음. |
| **L3** (early stop jump) | Path B val_loss | best_epoch std ≤ 3 | **closed** (2.21). val_loss continuous 가 epoch noise 감소시킴. 그러나 OOF/LB metric improvement 없음 (train objective ↔ eval metric misalignment). |
| **L1** (Feature redundancy) | Path C-B/C/D 단독 | at least 1 single-feature Δ ≥ +0.003 | **不_closed** (3/3 sub-threshold, max +0.0009). single-feature lever 만으로는 paradigm break 불가. |

**Summary**: 3 limitation 중 1 (L3) closed but metric value 안 늘어남. 다른 2 (L1, L2) 직접 닫기 attempt 모두 failed. corrector paradigm 의 *real ceiling* 이 stabilization 1 ~ +0.002 LB.

## §4. plan-017 후보 — paradigm-shift branch 활성

plan-016 의 stabilization 3 lever 모두 sub-threshold 도달 → **paradigm-level change 필요**.

### 공통 (모든 band, carry)

- **(공통-1) plan-013/014/015/016 ensemble** — 4 plan 의 best submission 좌표 mean. plan-013 0.6381 + plan-014 0.6628 + plan-016 G1 0.6638 의 framework-disjoint 결합. low-cost, dacon-submit 1회 (남은 quota 3 중 1 사용).
- **(공통-2) anchor codebook K 확장** — 현 K=9. K=15/21 검토 (plan-014 G3 E2 axis 의 K-density grid 위 marginal 확장).

### Paradigm-shift branch 활성 (필수, ≥ 3 후보)

- **(shift-1) 2-stage corrector** (plan-004 paradigm carry) ★ — selector (anchor) + boundary corrector (regression). plan-004 LB 0.6806 의 핵심 강점. corrector paradigm 의 input feature 한계 (plan-016 Path C 박제) 를 우회 — *별도 model 이 boundary 영역만 다룸*. 4 sample (~20%) 의 high-error region 직접 targeting.
- **(shift-2) 18×27 regime bias** — plan-004 v3.3 의 carry. trajectory regime (acceleration pattern) × anchor 매핑. plan-016 의 single-feature 한계가 regime 별 *different signal* 일 가능성.
- **(shift-3) hit-aware loss surrogate** — train objective ↔ eval metric misalignment (plan-016 G2 Path B 의 measured 결과) 의 root cause 해소. soft sigmoid hit indicator (e.g. `sigmoid((0.01 - err)/τ)`) 를 loss 에 추가.
- **(shift-4) Diffusion / score-based residual** — F0 residual 의 multi-modal distribution 학습 (특히 1.5cm 안 20% sample 의 *방향* uncertainty multi-modal 가설).
- **(shift-5) F0 prior 자체 교체** — plan-006 frenet_par120_perp_neg020 의 hyperparam (d1=1.98 / par=1.20 / perp=-0.20) 재탐색. 5-fold OOF 정합으로 d1/par/perp grid.

### 가설 검증 우선순위 (cost-ascending)

1. **공통-1** (ensemble) — 1 dacon-submit, no training. low cost.
2. **shift-3** (hit-aware loss) — corrector training pipeline 변경 only. medium cost.
3. **shift-1** (2-stage corrector) — plan-004 paradigm 다시 짜기. high cost but high ceiling (LB 0.68 정조준).
4. **shift-2** (regime bias) — plan-004 v3.3 carry. medium-high cost.
5. **shift-4** (diffusion residual) — completely new training paradigm. very high cost.
6. **shift-5** (F0 prior 교체) — F0 hyperparam grid search. low-medium cost.

## §5. measured 값 박제 (외부 reference)

| measure | value | source |
|---|---|---|
| F0 raw hit@1cm (plan-006 frozen reproduce) | 0.6320 | plan-014 G0 |
| plan-014/015 best_stack 5-fold OOF (baseline) | 0.6425 | plan-014 G5 |
| plan-014/015 best_stack LB (baseline) | 0.6628 | LB carry |
| **plan-016 G1 5-fold OOF** (5-seed multi-seed) | **0.6452** | plan-016 G1 ★ |
| **plan-016 G1 LB** (5-seed × 5-fold = 25 models) | **0.6638** ★ | plan-016 G1 |
| plan-016 G2 OOF (val_loss) | 0.6414 | plan-016 G2 (dropped) |
| plan-016 G2 LB | 0.6634 | plan-016 G2 (dropped) |
| plan-016 G3/G4/G5 OOF (Path C-B/C/D 단독) | 0.6443 / 0.6458 / 0.6461 | plan-016 G3-G5 |
| oracle ceiling (E0b Frenet-ortho, hindsight) | 0.8248 | plan-014 G0 |
| corrector 회수율 (plan-014/015 vs plan-016 G1) | (0.6452 − 0.6320) / (0.8248 − 0.6320) = 6.8% | plan-016 G1 |
| OOF-LB gap (plan-016 G1) | 0.6638 − 0.6452 = +0.0186 | plan-016 G1 (plan-014/015 +0.0203 carry) |

→ corrector paradigm + plan-016 stabilization 의 measured ceiling = **6.8% 회수율** (oracle gap 의 ~7%). plan-014/015 (5.4%) 위 marginal 회수.

## §6. LB carry-over

- **plan-016 G1 submission = LB 0.6638**. plan-014/015 LB 0.6628 + 0.0010.
- DACON quota 사용: **2 / 5** (G1 + G2). 남은 3 회는 plan-017 후보 검정 자원으로 carry.
- G3/G4/G5 submission.csv 산출 완료 (`runs/baseline/plan016_g3_path_c_b/`, `_c_c/`, `_c_d/`) — plan-017 ensemble 시 carry 가능.

## §7. 종료

- G_final 합격 (3 파일 sync):
  - results.md 신규 (본 파일) ✓
  - plan-016 frontmatter sync (status / band / best / followed_by [017]) ★ 별도 commit
  - registry append 6 row (H049~H054 already, H056 본 commit)
- plan-017 후보 ≥ 3 박제: ✓ (공통 2 + paradigm-shift 5 = 총 7)
- band 분류: **positive** (LB 0.6638 ∈ [0.66, 0.68])
- §0.5 c9 [TODO]→[DONE] sync 별도 commit
- DACON quota 2/5 사용 (G3/G4/G5 submit skip per 사용자 결정)
- plan-016 = corrector paradigm stabilization branch 종료, paradigm-shift 필요성 measured
