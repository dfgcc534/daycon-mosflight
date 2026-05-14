---
plan_id: 016
version: 1.1 (spec patch — plan-review-master iter 1 fix 7건. (1) §3.2 Feature B/C/D 산식 inline (plan-015 v2.3 carry, base 위 단독). (2) §3.1 9D base feature 전체 list inline (plan-014 v3.10 carry). (3) §3.2 / §7~§9 base column "G2 if pass else G1" conditional fallback rule 단일화. (4) §4.1 G0 baseline seed=20260514 명시 + 5 seed list 박제. (5) §5.2 OOF aggregation = 좌표 mean 후 hit@1cm 단일화. (6) §10.4 G6 = best Path C sub-exp alias 명시 (별도 학습 없음). (7) §0.5 stretch goal LB ≥ 0.68 표현 + self-label "stabilization + capacity stretch" 사후 승격 룰. v1 → v1.1.)
date: 2026-05-14 (Asia/Seoul)
status: spec
based_on:
  - 014 (LB 0.6628 positive band, OOF 0.6425, oof_lb_gap +0.0203)
  - 015 (Feature A negative drop rule, B/C/D untested)
followed_by: []
scope: plan-014/015 corrector paradigm 의 *3 가지 측정된 limitation* (L1 feature redundancy / L2 5-fold OOF variance noisy / L3 early stop val_hit discrete jump) 을 직접 닫는 sequential ablation. 7 anchor paradigm 유지 (oracle ceiling 0.8248 충분, candidate space 건드림 X). Path A (multi-seed ensemble) → Path B (monitor=val_loss) → Path C (B/C/D 단독 ablation) 누적.
exp_ids: []
lb_score: null
baseline_lb: 0.6628
baseline_oof: 0.6425
---

# plan-016 v1 — Corrector Stabilization (3 limitations 직접 닫기)

## §0. 한 줄 목적

> plan-014/015 LB 0.6628 (positive band) 측정 후 *3 limitation* 진단: **L1 Feature A redundancy** (9D 안에 F0 정보 implicit) / **L2 5-fold OOF variance noisy** (fold spread 0.04 ≫ Δ threshold 0.005, single seed) / **L3 early stop val_hit discrete jump** (fold-별 epoch 4~16 분산). 7 anchor paradigm + candidate space + F0 frozen 모두 carry, 위 3 limitation 만 직접 fix → LB 0.68+ 회수 시도 (plan-004 0.6806 정조준).

---

## §0.5 Quick Reference

### 본 plan 의 task essence — "paradigm 작동 confirm 후 variance + 학습 안정성 회수"

- plan-014/015 LB = **0.6628 positive band**. corrector paradigm 자체 작동.
- OOF–LB gap = +0.0203 → 5-fold OOF systematic underestimate. **OOF 평가 변경 + ensemble variance 감소** 로 측정 정확도 ↑.
- 사용자 명시: candidate space 안 건드림 (oracle ceiling 0.8248 충분).

### 3 limitation 진단 (plan-015 G1 fold-wise 분석)

- **L1 Feature A redundancy**: F0 산식의 input (v_last, acc_par, acc_perp) 이 이미 9D base 의 `acc_par/speed`, `perp_norm/speed` 등에 implicit. A (3D displacement_F0) 추가가 GRU 학습 signal 에 redundant → Δ = −0.001 (negative).
- **L2 OOF variance**: 5-fold concat OOF 의 fold spread 0.62~0.66 (=0.04) ≫ Δ threshold 0.005. single seed 라 fold variance 만으로 +0.005 검증 불가능. LB (0.6628) 가 OOF (0.6425) 보다 +0.02 위 = OOF systematic underestimate.
- **L3 early stop noise**: monitor=val_hit (discrete, sample/N 차이) 가 fold-별 noise 에 민감 → fold 4 epoch 4 vs fold 1 epoch 16 (= 학습 trajectory 길이 4배). 같은 spec 이 fold-별 capacity 다름.

### 본 plan 의 3-path sequential ablation

```
baseline (plan-014/015 best_stack, LB 0.6628)
  →  Path A (multi-seed × multi-fold ensemble)        — L2 닫기
  →  Path B (monitor=val_loss)                         — L3 닫기
  →  Path C-B / C-C / C-D (Feature B/C/D 단독)         — L1 educated 후 A 제외
  →  best stack
```

### 합격 기준 — LB-aware (Q1 v1 결정)

- 모든 stage 의 5-fold OOF + LB head-to-head 측정 (각 stage dacon-submit 1회, plan-016 총 5회 ≤ DACON daily 5 limit).
- per-stage Δ threshold:
  - **OOF Δ ≥ +0.005** → OOF-pass (variance 감소된 후의 검출)
  - **LB Δ ≥ +0.005** → LB-pass (실측 회수)
  - 둘 다 pass → positive; 한 쪽 만 pass → marginal; 둘 다 fail → negative drop
- band classification (LB 기준, v0.5 LB-aware):
  - LB ≥ 0.68 → **plan-004 LB 위 진입** (positive-top)
  - 0.66 ≤ LB < 0.68 → positive (band carry from plan-014/015)
  - 0.65 ≤ LB < 0.66 → partial
  - LB < 0.65 → negative
- LB underestimate 가설 검증: OOF-LB gap 박제 (모든 stage). gap drift ↑ 시 fold split 변경 권장 (plan-017).

### G-gates

- **G0** preflight: plan-014/015 baseline reproduce + 3 path config sanity [TODO]
- **G1 Path A** multi-seed ensemble (5-seed × 5-fold = 25 models): variance 감소 측정 [TODO]
- **G2 Path B** monitor=val_loss (Path A 위 cumulative): 학습 안정성 측정 [TODO]
- **G3 Path C-B** Feature B 단독 (binormal split, A 제외): redundancy 학습 [TODO]
- **G4 Path C-C** Feature C 단독 (multi-scale stride) [TODO]
- **G5 Path C-D** Feature D 단독 (pairwise cross-step) [TODO]
- **G6** best stack 5-fold + submission (Path A+B + best C lever) [TODO]
- **G_final** synthesis + plan-017 후보 + LB carry-over [TODO]

### Target

- baseline LB = **0.6628** (plan-014/015 carry, plan-016 도 동일 시작점).
- **G6 pass (목표) = LB ≥ 0.6678** (= baseline + 0.005). self-label "stabilization (minimal patch)" 의 ambition 안.
- **stretch goal = LB ≥ 0.68** (= plan-004 0.6806 정조준). 본 plan 의 3 변경 변수 (seed 수 / monitor / feature) 만으로는 도달 불확실, 만약 도달 시 self-label 을 *"stabilization + capacity stretch"* 로 사후 승격 (paradigm shift 아님). 미달 시 plan-017 path-pivot (2-stage corrector / regime bias) 후속.

### Commit chain

| # | type | spec section | status |
|---|---|---|---|
| c1 | docs | v1 draft — 3 limitation 진단 + 3-path sequential ablation spec | [DONE] a3d5269 |
| c1.1 | docs | **v1.1 spec patch — plan-review-master iter 1 fix 7건.** (1) §3.2 Feature B/C/D 산식 inline. (2) §3.1 9D base feature 전체 list inline. (3) Path C base fallback conditional rule 단일화. (4) G0 seed=20260514 + 5 seed list 명시. (5) OOF aggregation = 좌표 mean 단일화. (6) G6 alias 명시. (7) stretch goal LB 0.68 + self-label 사후 승격 룰. v1 → v1.1 | [TODO] |
| c2 | code+exp | STAGE 0 (G0) — preflight: baseline reproduce + 3 path config sanity | [TODO] |
| c3 | code+exp | STAGE 1 (G1, Path A) — multi-seed × multi-fold ensemble | [TODO] |
| c4 | code+exp | STAGE 2 (G2, Path B) — monitor=val_loss cumulative | [TODO] |
| c5 | exp | STAGE 3 (G3, Path C-B) — Feature B 단독 (binormal split, 10D base) | [TODO] |
| c6 | exp | STAGE 4 (G4, Path C-C) — Feature C 단독 (multi-scale stride, 18D base) | [TODO] |
| c7 | exp | STAGE 5 (G5, Path C-D) — Feature D 단독 (pairwise, 15D base) | [TODO] |
| c8 | code+exp | STAGE 6 (G6) — best stack 5-fold + submission + dacon-submit | [TODO] |
| c9 | docs+sync | STAGE 7 (G_final) — results.md + frontmatter sync + plan-017 후보 | [TODO] |

---

## §1. 배경 / 동기 (narrative)

### §1.1 plan-014/015 측정 후 evidence

- plan-014/015 best_stack: **OOF 0.6425 vs LB 0.6628** = OOF–LB gap **+0.0203**.
- plan-015 G1 (A only) fold-wise:
  - fold 0: 0.6573 → 0.6638 (+0.0065)
  - fold 1: 0.6283 → 0.6412 (+0.0129)
  - fold 2: 0.6452 → 0.6482 (+0.0030)
  - fold 3: 0.6239 → 0.6313 (+0.0074)
  - fold 4: 0.6251 → 0.6230 (**−0.0021**)
- 4 fold 가 positive, 1 fold 만 marginal negative — concat 으로 −0.001.
- early stopping epoch: fold 4=4 (early), fold 1=16 (late) — 학습 trajectory 분산.

### §1.2 3 limitation 정밀 진단

#### L1. Feature A redundancy

- F0 산식 `F0 = p0 + 1.98·v_last + 1.20·acc_par_vec + (−0.20)·acc_perp_vec` 의 input variables = `v_last, acc_par_vec, acc_perp_vec`.
- 9D base feature 의 `(1) speed = ‖v_last‖`, `(2) prev_speed/speed`, `(3) acc_norm/speed`, `(4) acc_par_scalar/speed`, `(5) perp_norm/speed` 등이 위 모든 components 의 normalized version 을 포함.
- GRU 가 9D 위에서 implicit 으로 F0 prediction 학습 가능. A (3D displacement_F0) 가 추가 정보 없음.

#### L2. 5-fold OOF variance vs Δ threshold mismatch

- fold spread = 0.04 (= max−min over 5 folds), 즉 single fold 의 noise scale.
- Δ threshold = +0.005 → SNR ≈ 0.005 / 0.04 = **1/8**.
- single seed 라 fold variance 만으로 statistical confidence 없음.
- LB 0.6628 (test set scale) 가 OOF 0.6425 보다 +0.02 위 → **OOF 의 systematic underestimate**.

#### L3. monitor=val_hit 의 discrete jump

- val_hit = (samples / N_val) 이라 sample 1개 차이 = 1/2000 ≈ 5e-4 jump.
- patience=5 patience 안에 noise jump 들어가면 fold-별 early stop trigger 분산.
- fold 4 epoch 4 vs fold 1 epoch 16 = 학습 자체가 fold-별 다른 model 산출.

### §1.3 plan-016 의 design philosophy

- **paradigm carry** (변경 안 함): F0 frozen plan-006 / corrector arch (BiGRU h=128) / 7 anchor codebook / baseline lever (boundary_weight_on) / 5-fold scheme.
- **direct fix** (변경 함): seed 수 (1 → 5) / monitor (val_hit → val_loss) / Feature A → B/C/D 교체.
- 위 3 fix 는 *각각 1 lever* — sequential single-variable ablation.
- candidate space 안 건드림 (사용자 명시).

### §1.4 plan-004 / plan-013 / plan-014/015 LB chain

| plan | LB | paradigm | scale |
|---|---|---|---|
| plan-013 fallback | 0.6381 | plan-004 framework + In/IC (deferred lever) | minimal patch |
| **plan-014/015** | **0.6628** | corrector from-scratch + F0 frozen | "재사용 끊기" |
| **plan-016 target** | **≥ 0.6678** (≥ baseline + 0.005) | 위 paradigm + variance/stability fix | direct limitation 해소 |
| **plan-016 ambition** | **≥ 0.68** (= plan-004 정조준) | 7 anchor paradigm 만으로 plan-004 LB 정복 | high-value path |
| plan-004 (original) | 0.6806 | 27 candidate + 2-stage selector+corrector | full notebook |

→ plan-016 = corrector paradigm 의 *true potential* 검증. 사용자 narrative ("candidate 안 건드림") 따라 7 anchor 만으로 plan-004 LB 0.6806 까지 회수 가능한지 measured.

---

## §2. Scope (명시적)

### §2.1 In-scope

| 항목 | 값 |
|---|---|
| Baseline | plan-014/015 best_stack carry (E0c K-Means K=9 + boundary_weight_on, F0 frozen plan-006) |
| 변경 변수 (Path A) | seed 수 1 → 5 (5-seed × 5-fold = 25 models ensemble) |
| 변경 변수 (Path B) | monitor=val_hit → **val_loss** (continuous, fold-별 noise ↓) |
| 변경 변수 (Path C-B/C/D) | Feature A 제외, B/C/D 단독 (10D / 18D / 15D 단독 base) |
| 평가 | 5-fold OOF + LB head-to-head (각 stage dacon-submit 1회) |
| 합격 기준 | OOF Δ ≥ +0.005 AND LB Δ ≥ +0.005 (둘 다 ≥ → positive) |

### §2.2 Out-of-scope

| 항목 | 이유 |
|---|---|
| candidate space (27 후보) 변경 | 사용자 명시 - 7 anchor 유지 (oracle ceiling 0.8248 충분) |
| 2-stage (selector + boundary corrector) | plan-004 paradigm 의 핵심 강점이지만 본 plan scope 아님 (plan-017 후보) |
| 18×27 regime bias | 동일 (plan-017 후보) |
| Feature A | plan-015 G1 falsified → 제외 |
| F0 산식 / corrector arch / lever / 5-fold scheme | plan-014/015 carry |
| ensemble with plan-013 / plan-004 | 정공법 회피 — plan-016 마지막 단계 또는 plan-017 후보 |

---

## §3. 사전 등록 (Pre-registration)

### §3.1 Baseline reference (plan-014/015 carry)

| metric | value | source |
|---|---|---|
| F0 raw hit@1cm | 0.6320 | plan-014 G0 |
| plan-014/015 5-fold OOF | 0.6425 | plan-015 G0 reproduce |
| **plan-014/015 LB** ★ | **0.6628** | plan-015 dacon-submit (사용자 confirm) |
| oracle ceiling (E0b Frenet-ortho) | 0.8248 | plan-014 G0 |
| corrector LB 회수율 | 17% | (0.6628 − 0.6320) / (0.8248 − 0.6320) |

**baseline 9D base feature** (plan-014 v3.10 carry, per-step `s` 산출):
1. `speed_s = ‖v_s‖` where `v_s = X[:, s] − X[:, s−1]`
2. `prev_speed_s / speed_s` where `prev_speed_s = ‖X[:, s−1] − X[:, s−2]‖`
3. `‖acc_s‖ / speed_s` where `acc_s = X[:, s] − 2·X[:, s−1] + X[:, s−2]`
4. `(acc_s · t̂_s) / speed_s` (= acc_par_scalar / speed)
5. `‖acc_perp_s‖ / speed_s` (= perp_norm / speed; **Feature B 의 split 대상**)
6. `‖jerk_s‖ / speed_s` where `jerk_s = acc_s − prev_acc_s`
7. `(v_s · v_{s−1}) / (speed_s · prev_speed_s)` (turn_cos)
8. `‖acc_perp_s‖ / (speed_s + ε)` (curvature placeholder, 의도된 (5) 와 ε 한 항 차이 중복)
9. `direction = 1.0` const broadcast

6-step indices = `range(max(3, end_idx−5), end_idx+1)` = [5,6,7,8,9,10] for `end_idx=10`. `(N, 6, 9)` shape.

### §3.2 Sub-exp matrix (sequential cumulative)

| stage | sub-exp | 변경 사항 | base | dim |
|---|---|---|---|---|
| G0 | preflight | baseline reproduce | — | 9D |
| **G1** | Path A | 5-seed × 5-fold = 25 models, coord mean ensemble | baseline | 9D |
| **G2** | Path B | + monitor=val_loss (patience=5 같음) | G1 cumulative | 9D |
| **G3** | Path C-B | **+ Feature B (binormal split)** — 9D base 의 (5) `perp_norm/speed` 1D 를 step-local Frenet basis 위 `(acc_normal/speed, acc_binormal/speed)` 2D split (sign-aware). plan-015 v2.3 §1.B carry. *base 위 단독 적용* (cumulative 아님) | G2 (if pass) else G1 | **10D = 9D − 1D + 2D** |
| **G4** | Path C-C | **+ Feature C (multi-scale stride τ=1,2 stream concat)** — 9D base 를 stride 1 (기존 indices [5,6,7,8,9,10]) + stride 2 (indices [3,5,7,9] + pad [3,3,3,5,7,9]) 2 stream 산출 후 axis=-1 concat. plan-015 v2.3 §1.C carry. *base 위 단독* | G2 (if pass) else G1 | **18D = 9D × 2 stream** |
| **G5** | Path C-D | **+ Feature D (pairwise cross-step)** — 3 pair × 2 stat = 6D 추가. pair = `(s, s-2)`, `(s, s-4)`, `(s-2, s-4)`. stat = `cosine_similarity(v[s_a], v[s_b])` + `Δspeed = ‖v[s_a]‖ − ‖v[s_b]‖`. plan-015 v2.3 §1.D carry. *base 위 단독* | G2 (if pass) else G1 | **15D = 9D + 6D** |
| G6 | best stack | G2 + Path C best (max LB among C-B/C-C/C-D) | — | varies |

**중요**: Path C 는 **B/C/D 의 *단독* 비교** (cumulative 아님). 이전 plan-015 cumulative A→A+B→A+B+C→A+B+C+D 가 A redundancy 로 막힌 교훈. plan-016 C 는 각 single feature 의 LB head-to-head 만, best 1개만 G6 best_stack 채택.

**Path B fail 시 fallback rule (v1 단일화)**: G2 가 negative (둘 다 fail) 시 *drop Path B* — G3/G4/G5 의 base = **G1 cumulative** (Path A only, monitor=val_hit). G6 best_c 도 G2 가 아닌 G1 위 best feature. 즉 §7.2 / §8.2 / §9.2 의 "G2 carry" 는 실제로 **"G2 if G2 pass else G1" 의 conditional carry**. §3.2 표의 *or* 분기는 모두 본 conditional 의 표기.

### §3.3 G-gate quantitative criteria

#### G0 — preflight (baseline reproduce)

- artifact: `analysis/plan-016/preflight.json`
- (a) plan-014/015 baseline 5-fold OOF reproduce → 0.6425 ± 0.005 일치
- (b) 3 path config sanity: seed list (5 seed 정의) / monitor=val_loss option 동작 / Feature B/C/D 단독 dim (10/18/15)

#### G1 — Path A (multi-seed ensemble)

- artifact: `analysis/plan-016/g1_path_a.json` + `runs/baseline/plan016_g1_path_a/`
- spec: seeds = [20260514, 20260515, 20260516, 20260517, 20260518] (5 seed) × 5 fold = **25 models**.
- 각 model 의 val OOF (좌표 prediction `(N_val, 3)`) 와 test prediction `(N_test, 3)` 산출.
- **OOF aggregation rule (v1 단일화)**: fold `f` 의 OOF = `coord_mean over 5 seeds of (val_pred_seed_k_fold_f)` (즉 *좌표 mean 후 hit@1cm 계산*, 5 hit@1cm 의 mean 아님).
- 5-fold concat OOF hit@1cm = `mean(‖coord_mean_oof_pred − y_true‖₂ ≤ 0.01m)` over all 10000 train samples (plan-015 §3.1 carry).
- test ensemble = 25 model 의 test prediction 좌표 mean (`coord_mean over 25 of test_pred_seed_k_fold_f`).
- submission `runs/baseline/plan016_g1_path_a/submission.csv` → **dacon-submit 1회**.
- criterion: **OOF Δ ≥ +0.005 AND LB Δ ≥ +0.005** (vs baseline 0.6425 OOF / 0.6628 LB).

#### G2 — Path B (monitor=val_loss cumulative)

- artifact: `analysis/plan-016/g2_path_b.json` + run dir
- spec: G1 의 5-seed × 5-fold 같은 ensemble + **monitor=val_loss** (patience=5, plan-014 v3.10 c3.10 spec carry).
- dacon-submit 1회.
- criterion: **OOF Δ ≥ +0.005 AND LB Δ ≥ +0.005** (vs G1).

#### G3/G4/G5 — Path C-B/C-C/C-D (각 Feature 단독)

- artifact: `analysis/plan-016/g[345]_path_c_[bcd].json` + run dir
- spec: G2 의 base (multi-seed + monitor=val_loss) 위 *Feature 단독 적용*. Feature flag `{B: True}` only / `{C: True}` only / `{D: True}` only.
- 각 dacon-submit 1회 (G3, G4, G5).
- criterion: **OOF Δ ≥ +0.003 AND LB Δ ≥ +0.003** (single feature marginal lever, threshold 낮춤).
- best Path C 선정: max LB over {C-B, C-C, C-D} → G6 best_stack.

#### G6 — best stack 5-fold + submission

- artifact: `analysis/plan-016/g6_best_stack.json` + `runs/baseline/plan016_g6/submission.csv`
- spec: G2 cumulative + best Path C lever.
- LB band 분류:
  - LB ≥ 0.68 → **plan-004 정조준 달성** (positive-top)
  - 0.66 ≤ LB < 0.68 → positive
  - 0.65 ≤ LB < 0.66 → partial
  - LB < 0.65 → negative
- **G6 자체는 dacon-submit 안 함** (G6 spec = 직전 best Path 의 LB carry — 추가 dacon 안 필요).

#### G_final — synthesis

- artifact: `plans/plan-016-corrector-stabilization.results.md` 신규 + `registry.csv` append + frontmatter sync
- content:
  - G0~G6 결과 + LB chain (baseline 0.6628 → G1 / G2 / G3 / G4 / G5 / G6 best)
  - OOF–LB gap drift 박제 (모든 stage)
  - L1/L2/L3 closure 검증
  - plan-017 후보 ≥ 3 (band 별)
- fail trigger: 3 파일 sync 누락 → `final_sync_missing` severe

### §3.4 exp_id naming

| exp_id | stage |
|---|---|
| H049_g0_preflight | G0 |
| H050_g1_path_a_multiseed | G1 |
| H051_g2_path_b_val_loss | G2 |
| H052_g3_path_c_b | G3 |
| H053_g4_path_c_c | G4 |
| H054_g5_path_c_d | G5 |
| H055_g6_best_stack | G6 |
| H056_g_final_synthesis | G_final |

### §3.5 DACON submission budget

- Path A / B / C-B / C-C / C-D 각 1회 = **5 dacon-submit** (= daily limit 도달).
- G6 best stack = direct LB carry (재submit 안 함).
- daily limit 도달 → 다음날 follow-up 가능 (G6 별도 검정 원할 시).

---

## §4. STAGE 0 (c2, G0) — preflight [TODO]

### §4.1 산출물

- `analysis/plan-016/preflight.py` — 2 task:
  - (a) plan-014/015 baseline (**seed=20260514**, single-seed, monitor=val_hit) 5-fold OOF reproduce → 0.6425 ± 0.005. plan-015 G0 (H042) 산출과 deterministic 동일.
  - (b) 3 path config sanity: 5 seed list = `[20260514, 20260515, 20260516, 20260517, 20260518]` (plan-014 base seed + 4 sequential) verify / monitor=val_loss option (plan-014 v3.10 carry, patience=5 동일) verify / Feature B/C/D 단독 dim (10D / 18D / 15D) sanity
- `analysis/plan-016/preflight.json`
- registry row: `H049_g0_preflight`

### §4.2 실행

```bash
python analysis/plan-016/preflight.py
```

### §4.3 G0 합격

- (a) reproduce 0.6420 ≤ OOF ≤ 0.6430
- (b) 3 path config 모두 sanity OK (seed list len=5, monitor options 정상, dim 10/18/15)

---

## §5. STAGE 1 (c3, G1, Path A) — multi-seed × multi-fold ensemble [TODO]

### §5.1 산출물

- `src/pb_0_6822/plan016_ensemble.py` — multi-seed ensemble runner
- `analysis/plan-016/g1_path_a.py` — 5 seed × 5 fold = 25 models 학습 + OOF concat + test ensemble
- `runs/baseline/plan016_g1_path_a/submission.csv`
- registry row: `H050_g1_path_a_multiseed`

### §5.2 spec

- seeds = [20260514, 20260515, 20260516, 20260517, 20260518]
- 각 seed × 5 fold = 25 models
- val OOF: 같은 fold 의 5 seed prediction → coord mean → 그 fold 의 OOF
- test: 25 model 의 test prediction → coord mean
- baseline config 외 일체 변경 X (monitor=val_hit 유지, Feature flag 모두 False)
- dacon-submit 1회

### §5.3 G1 합격

- OOF Δ ≥ +0.005 vs 0.6425 (= OOF ≥ 0.6475)
- LB Δ ≥ +0.005 vs 0.6628 (= LB ≥ 0.6678)
- 둘 다 pass → positive, G2 진행
- 한 쪽 만 pass → marginal, G2 진행 + warn
- 둘 다 fail → negative, G2~G5 skip → G_final 직행 (best = baseline)

---

## §6. STAGE 2 (c4, G2, Path B) — monitor=val_loss cumulative [TODO]

### §6.1 산출물

- `analysis/plan-016/g2_path_b.py` — G1 base + monitor=val_loss
- `runs/baseline/plan016_g2_path_b/submission.csv`
- registry row: `H051_g2_path_b_val_loss`

### §6.2 spec

- seeds 동일 (G1 carry)
- **monitor=val_loss** (plan-014 v3.10 c3.10 이전 spec carry, patience=5)
- 외 G1 carry
- dacon-submit 1회

### §6.3 G2 합격

- OOF Δ ≥ +0.005 vs G1 OOF
- LB Δ ≥ +0.005 vs G1 LB
- 둘 다 pass → positive, G3 진행
- 한 쪽 만 pass → marginal, G3 진행 + warn
- 둘 다 fail → drop Path B (= G2 anchor = G1), G3~G5 진행 + best 재산정

---

## §7. STAGE 3 (c5, G3, Path C-B) — Feature B 단독 [TODO]

### §7.1 산출물

- `analysis/plan-016/g3_path_c_b.py` — G2 cumulative + Feature B (10D)
- `runs/baseline/plan016_g3_path_c_b/submission.csv`
- registry row: `H052_g3_path_c_b`

### §7.2 spec

- G2 carry (5-seed ensemble + monitor=val_loss)
- feature_flags = {"A": False, "B": True, "C": False, "D": False}
- input_dim = 10D (단독 B = 9D − 1D + 2D, perp split)
- dacon-submit 1회

### §7.3 G3 합격

- OOF Δ ≥ +0.003 vs G2 OOF (single feature lever, threshold 낮춤)
- LB Δ ≥ +0.003 vs G2 LB
- 둘 다 pass → positive lever, G6 best stack 후보
- 한 쪽 만 pass → marginal, G6 후보 (낮은 priority)
- 둘 다 fail → drop B

---

## §8. STAGE 4 (c6, G4, Path C-C) — Feature C 단독 [TODO]

### §8.1 산출물

- `analysis/plan-016/g4_path_c_c.py` — G2 cumulative + Feature C (18D)
- `runs/baseline/plan016_g4_path_c_c/submission.csv`
- registry row: `H053_g4_path_c_c`

### §8.2 spec

- G2 carry
- feature_flags = {"A": False, "B": False, "C": True, "D": False}
- input_dim = 18D (단독 C = 9D × 2 stream τ=1,2)
- BiGRU input_dim 변경 + weight 재초기화 (plan-015 §7.2 carry)
- dacon-submit 1회

### §8.3 G4 합격

- OOF Δ ≥ +0.003 / LB Δ ≥ +0.003 vs G2

---

## §9. STAGE 5 (c7, G5, Path C-D) — Feature D 단독 [TODO]

### §9.1 산출물

- `analysis/plan-016/g5_path_c_d.py` — G2 cumulative + Feature D (15D)
- `runs/baseline/plan016_g5_path_c_d/submission.csv`
- registry row: `H054_g5_path_c_d`

### §9.2 spec

- G2 carry
- feature_flags = {"A": False, "B": False, "C": False, "D": True}
- input_dim = 15D (단독 D = 9D + 6D pairwise)
- dacon-submit 1회

### §9.3 G5 합격

- OOF Δ ≥ +0.003 / LB Δ ≥ +0.003 vs G2

---

## §10. STAGE 6 (c8, G6) — best stack [TODO]

### §10.1 best 선정

```python
c_candidates = {
    "C-B": (g3_oof, g3_lb),
    "C-C": (g4_oof, g4_lb),
    "C-D": (g5_oof, g5_lb),
}
# argmax LB (LB 가 ground truth, OOF underestimate 신뢰성 낮음)
best_c = max(c_candidates, key=lambda k: c_candidates[k][1])
```

### §10.2 best_stack config

- G2 cumulative (5-seed ensemble + monitor=val_loss)
- + best_c feature 적용 (Feature B/C/D 중 LB 최대)
- = G3/G4/G5 중 LB 최대 sub-exp 자체

### §10.3 G6 합격

- best_c LB ≥ baseline + 0.005 (= 0.6678) → **G6 PASS**
- 0.65 ≤ best_c LB < 0.66 → partial
- best_c LB < 0.65 → negative (paradigm 한계)
- best_c LB ≥ 0.68 → **plan-004 정조준 달성** ★

### §10.4 submission

- 별도 dacon-submit 안 함 (best_c 의 submission 자체가 G6 submission).
- G6 = **alias of best Path C sub-exp** (= argmax LB over {G3, G4, G5}). artifact = best sub-exp 의 결과 carry. 별도 학습/산출 없음.
- registry row `H055_g6_best_stack`: `run_dir` = best sub-exp 의 run_dir alias. `notes` = "alias of <best Path C sub_exp_id>, LB=<best_c LB>".
- §3.4 exp_id `H055` 의 의미 = alias row, *new training 없음* 명시.

---

## §11. STAGE 7 (c9, G_final) — synthesis + plan-017 후보 [TODO]

### §11.1 산출물

- `plans/plan-016-corrector-stabilization.results.md` 신규
- plan-016 frontmatter sync (status / lb_score / band)
- registry append 8 row (H049~H056)
- plan-017 후보 ≥ 3

### §11.2 plan-017 후보 (band 별)

#### 공통 (모든 band)

- **(공통-1) OOF–LB gap 의 fold split 변경 시도** — 현 SHA256 stable_hash 의 spread 0.04 → KFold(shuffle=True) / StratifiedKFold by trajectory_length 등 fold variance 감소

#### Band plan-004 정조준 달성 (LB ≥ 0.68)

- **(top-1) plan-013 + plan-016 best ensemble** — corrector paradigm + plan-004 framework 결합으로 LB further 회수
- **(top-2) 더 큰 encoder** (h=128 → 256, num_layers 3) — 추가 capacity

#### Band positive (0.66 ≤ LB < 0.68)

- **(positive-1) 2-stage corrector 도입** (plan-004 boundary corrector carry) — paradigm hybrid path, 7 anchor 위 boundary refinement 추가
- **(positive-2) regime bias 도입** — 18-regime conditioning (plan-004 carry)

#### Band partial / negative

- **(fallback-1) ensemble with plan-013** — corrector paradigm 한계 시 plan-004 framework hybrid

### §11.3 G_final 합격

- 3 파일 sync + plan-017 후보 ≥ 3
- 누락 시 `final_sync_missing` severe

---

## §N+4. 변경 이력

- v1 (2026-05-14): 초안. plan-014/015 LB 0.6628 measured 후 3 limitation 진단 + sequential ablation spec.

---

## §N+5. 참조

- `plans/plan-014-plan012-failure-inversion.results.md` — LB 0.6628 measured, oof_lb_gap +0.0203
- `plans/plan-015-feature-expansion.results.md` — Feature A negative drop rule, fold-wise 분석
- `plans/plan-013-plan004-framework-3lever-stacking.results.md` — LB 0.6381 baseline reference
- `registry.csv` H041 (plan-014 G5) / H047 (plan-015 G5) — best_stack carry
- `src/pb_0_6822/plan014_paradigm.py` / `plan015_features.py` / `plan015_train.py` — module reuse base
- `runs/baseline/plan015_g5/submission_best.csv` — baseline LB submission (carry)
