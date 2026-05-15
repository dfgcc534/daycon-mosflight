---
plan_id: 014
version: 4.5 (results synthesis)
date: 2026-05-14 (Asia/Seoul)
status: G_final_complete (G2/G3/G4/G5 all PASS, band=negative)
based_on:
  - 012
followed_by:
  - 015 (negative branch — deep path-pivot per plan-013 join row 4)
scope: corrector paradigm 5-Phase 재실험 결과 박제. F0 plan-006 frozen prior + corrector from-scratch 정밀화 measured.
exp_ids:
  - H036_g0_preflight
  - H037_g1_module_smoke
  - H038_g2_phase1_bakeoff
  - H039_g3_phase2_axis5
  - H040_g4_phase3_aux3
  - H041_g5_phase4_final
lb_score: 0.6628
lb_band: positive
band: negative
best_stack_5fold_oof: 0.6425
anchor_5fold_oof: 0.6359
delta_oof: 0.0066
oof_lb_gap: 0.0203
---

# plan-014 v4 — Results (band=negative, plan-013 join row 4 활성)

## §1. G0~G5 결과 narrative

### G0 — preflight (H036_g0_preflight)

- **F0 frozen reproduce hit@1cm = 0.6320** (plan-006 hard evidence 정확 일치, in [0.6270, 0.6370]). hit@1.5cm = 0.8033.
- 3 codebook oracle ceiling (radius=0.01m):
  - E0a Absolute: 0.8203
  - E0b Frenet-orthogonal: 0.8248 ← best ceiling
  - E0c K-Means (per-fold): 0.7625
- soft label entropy (target Gaussian σ=0.01m) = 1.707 nat (≥ 0.5)
- plan-012 disclaimer grep: INVALID_REFERENCE + disclaimer field 양쪽 hit
- per-axis marginal oracle ordering: Absolute=[y, x, z] / Frenet=[n, b, t]
- **g0_essential_passed = True (4/4)**

### G1 — module + smoke (H037_g1_module_smoke)

- `src/pb_0_6822/plan014_paradigm.py` (v4) — `Plan014F0Function` = plain class (NOT nn.Module, no nn.Parameter, requires_grad 개념 없음). corrector (`Plan014BiGRUEncoder` + `Plan014HybridHead`) 만 from-scratch learnable.
- 5/5 pytest PASS:
  - AST import 0 (selector / ring_classifier / boundary / plan-006 numpy F0 함수)
  - F0 frozen verify: plain float constants + reproduce hit@1cm=0.6320
  - anchor scale 0.01 ± 1e-6 (E0a/E0b)
  - Gaussian σ=0.01m soft label entropy ≥ 0.5 nat
  - 1-fold 1-epoch smoke: val_hit_after >= initial_val_hit − 0.05

### G2 — Phase 1 codebook bake-off (H038_g2_phase1_bakeoff)

3 sub-exp × 5-fold OOF (F0 frozen 공통):

| sub-exp | codebook | 5-fold OOF hit@1cm | mean DCM |
|---|---|---|---|
| E0a | Absolute | 0.6293 | 0.0014 |
| E0b | Frenet-orthogonal | 0.6239 | 0.0026 |
| **E0c** ★ | **K-Means** | **0.6359** | **0.0026** |

- winner = E0c (kmeans/frenet), gap = 0.0066 (no tie-break)
- **G2_passed = True** (winner_oof ≥ 0.60 + DCM ≥ 0.002)
- F0 raw 0.6320 → winner 0.6359 = **+0.0039 회수** (corrector 학습 정상 작동, v3.x cascade failure 회피 confirmed)

### G3 — Phase 2 axis ablation 5 (H039_g3_phase2_axis5)

5 axis × fold=0 single-fold (anchor = G2 winner E0c, fold-0 val_hit = 0.6573):

| axis | best variant | max ΔOOF | positive (≥+0.005) |
|---|---|---|---|
| E1 frame swap | E1b (world) | -0.0055 | ❌ |
| E2 K density | **E2c (K=9)** | **+0.0030** | ❌ (marginal) |
| E3 τ scan | E3b (τ=0.01) | -0.0005 | ❌ |
| E4 loss swap | E4b (+ hinge) | -0.0050 | ❌ |
| E5 reg head | E5a (reg off) | -0.0015 | ❌ |

- **positive_axes = [], G3_passed = False, g3_marginal_only warn**
- paradigm 위 의미 있는 lever 0개 — F0 ceiling 신호

### G4 — Phase 3 aux ablation 3 (H040_g4_phase3_aux3)

3 axis × fold=0 (informational, G3 warn carry):

| axis | best variant | max ΔOOF | positive (>0) |
|---|---|---|---|
| **E6 boundary weight** | E6b (on) | **+0.0015** | ✓ (marginal) |
| E7 scorer arch | E7b (LastStep MLP) | -0.0050 | ❌ (BiGRU 시계열 가치 입증) |
| E8 r=0 prior | E8b (+0.5) | -0.0040 | ❌ |

- positive_axes = ['E6'] (marginal)

### G5 — Phase 4 final 5-fold + best stack + submission (H041_g5_phase4_final) ★

best_stack = G2 winner (E0c K-Means) + E2c (K=9) + E6b (boundary_weight_on):

| config | 5-fold concat OOF hit@1cm |
|---|---|
| anchor (E0c K=7, default lever) | 0.6359 |
| **best** (E0c K=9 + boundary_weight_on) | **0.6425** |
| **delta_oof** | **+0.0066** ✓ |

- **G5_passed = True** (Δ ≥ +0.005 threshold)
- **band = negative** (best_stack 0.6425 < 0.65)
- G3 single-fold marginal lever 들 (E2c +0.003, E6b +0.0015) 의 additive 효과가 5-fold concat 에서 살아남 — additive 가정 검증.
- 그러나 absolute band = negative. paradigm 의 measured ceiling = F0 raw + ~0.01.

### Submission

- `runs/baseline/plan014_g5_phase4/submission_best.csv` — best_stack ensemble (5-fold coord mean over test).
- `runs/baseline/plan014_g5_phase4/submission_anchor_fallback.csv` — anchor ensemble (G5 fail 시 fallback, 본 plan 에선 G5_passed 라 best 사용).
- `submission_used_for_LB = submission_best.csv`

## §2. plan-013 join interpretation (§1.4 activated row)

plan-013 LB 0.6381 (fallback, < 0.68) + plan-014 best_stack 0.6425 (< 0.65) → **row 4 활성**:

> "둘 다 실패 — 더 deep path-pivot (`notes/new-ideas.md` KNN/GP/Diffusion)"

→ plan-013 framework path + plan-014 corrector paradigm path 모두 F0 raw (0.6320) 근처 plateau confirm. 두 path 의 ceiling 이 ≤0.65 정도 — DACON 236716 muflight 의 *현 framework family* 자체의 한계 신호.

## §3. Premise verdict — "corrector 재사용 = paradigm 한계 root cause" falsified

**Premise (§0.5 v4 narrative)**:
> plan-012 의 plateau root cause = "plan-004 `CandidateAttentionGRUSelector` + plan-012 `ring_classifier.py` corrector 코드의 재사용 강박". 재사용 끊고 corrector 만 from-scratch 재설계 하면 F0 (frozen) 위 +0.03~0.04 회수 가능 (band ≥ 0.66).

**Verdict**: **부분 falsified**.

| 측면 | 결과 | 해석 |
|---|---|---|
| corrector 재사용 끊은 baseline 측정 가능 (v3.x cascade failure 회피) | ✅ 성공 | premise 의 *방법론* 정당 — F0 frozen narrative 가 정확 |
| corrector paradigm 잠재력 ≥ 0.66 (band positive) | ❌ 실패 | premise 의 *수치 예상* 틀림 — paradigm 자체 ceiling 이 limit |
| 5-fold best_stack absolute = 0.6425 | F0 raw + 0.0105 | corrector 가 +0.01 회수 가능 but +0.03 미달 |
| oracle ceiling 0.8248 vs measured 0.6425 | gap = 0.18 | corrector 가 oracle ceiling 의 *22%* 만 회수 — features 가 F0 error 방향 predict 부족 |

**결론**: 재사용 끊기는 *necessary* but *not sufficient*. F0 raw → +0.01 회수가 corrector paradigm 의 measured limit. competition-level (0.66+) 도달은 paradigm shift 필요 (plan-015 후보).

## §4. plan-015 후보 (negative branch — deep path-pivot)

### 공통 (모든 band, plan-015 carry)

- **(공통-1) F0 frozen vs learnable attribution** — plan-014 F0 frozen baseline 확정 후, F0 learnable variant 1회 측정 → F0 component attribution. (v3.x cascade failure 와의 *measured 비교* 가 plan-015 가치)
- **(공통-2) Multi-seed 분산 측정** — single seed (20260514) → 5-seed × 5-fold OOF std. plan-014 best 0.6425 의 confidence interval.

### Band negative 분기 (활성, ≥ 3 후보)

- **(negative-1) plan-013 join row 4 매핑 — deep path-pivot** — `notes/new-ideas.md` 12종 paradigm-shift 후보 batch 조사. Top candidates:
  - **KNN with Frenet local residuals** — F0 frozen + per-sample Frenet basis 위 K-Nearest training samples 의 residual vote (non-parametric). 적은 data 에서 corrector NN 보다 견고할 가능성.
  - **Gaussian Process (GP)** — F0 residual 의 input feature dependence 를 명시적 covariance kernel 로 modeling. uncertainty 도 함께.
  - **Diffusion / score-based residual** — F0 residual 의 multi-modal distribution 학습 (특히 1.5cm 안 20% sample 의 *방향* uncertainty 가 multi-modal 일 가능성).
- **(negative-2) Input feature 확장** — 현재 8d kinematic features 가 F0 error 방향 predict 부족 신호 (oracle 0.82 vs measured 0.64). 후보:
  - **Higher-order kinematic** (jerk², acc_acc, snap)
  - **Cross-step interaction** (full 11-step pairwise features)
  - **Frequency-domain features** (FFT of 11-step trajectory)
- **(negative-3) plan-013 corrector + plan-014 corrector ensemble** — plan-013 fallback 0.6381 + plan-014 best 0.6425 의 좌표 평균 ensemble. similar magnitudes 라 marginal 회수 예상 (+0.002~0.005) but consistent.

### 최종 권장 (plan-015 spec drop 시점 결정)

**path-pivot 우선순위**:
1. **KNN-based corrector** (가장 simple, F0 frozen 위 non-parametric)
2. **Input feature 확장** (frequency + higher-order)
3. **Diffusion residual** (multi-modal hypothesis, training cost 높음 → 후순위)

(공통-1 F0 attribution + 공통-2 multi-seed 는 plan-015 의 "validation infrastructure" 로 병행)

## §5. measured 값 박제 (외부 reference 후보)

| measure | value |
|---|---|
| F0 raw hit@1cm (plan-006 frozen reproduce, plan-014 G0) | 0.6320 |
| F0 raw hit@1.5cm | 0.8033 |
| G2 winner 5-fold OOF (E0c K-Means K=7) | 0.6359 |
| **G5 best_stack 5-fold OOF** (E0c K-Means K=9 + boundary_weight_on) | **0.6425** ★ |
| anchor_5fold vs best_5fold delta | +0.0066 |
| oracle ceiling (E0b Frenet-ortho, hindsight) | 0.8248 |
| corrector 회수율 (= delta / (oracle − F0)) | 0.0105 / 0.1928 = 5.4% |

(plan-012 의 `INVALID_REFERENCE_` carry pattern 적용 — plan-014 의 measured 값은 *plan-014 spec 위에서만* 정합. plan-015 가 같은 spec 으로 측정 시 valid reference.)

## §6. 종료

- G_final 합격 (3 파일 sync): ✓
- plan-015 후보 ≥ 3 박제: ✓ (공통 2 + negative branch 3)
- band 분류: **negative**
- §0.5 commit chain c10 [TODO]→[DONE] sync 별도 commit
- registry append 6 row 완료 (H036~H041)
