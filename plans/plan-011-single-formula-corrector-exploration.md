---
plan_id: 011
version: 1
date: 2026-05-13 (Asia/Seoul)
status: draft
based_on:
  - 004
  - 005
  - 006
  - 007
  - 008
  - 009
  - 010
  - notes/PB_0.6822 코드공유.ipynb
  - notes/코드공유-upgrade.md
  - notes/prior-ideas.md
  - notes/mosquito-trajectory-ideas.md
followed_by:
  - 011.1 (LB carry-over; user manual dacon-submit)
  - 012 (TBD; candidates @ analysis/plan-011/next_plan_candidates.md)
scope: 단일공식 + corrector path 의 *구체화 + 폭넓은 탐색*. plan-006 의 frenet_par120_perp_neg020 + plan-004 corrector LB 0.6692 baseline 위에 corrector 4 axis (Input × Loss × Arch × Formula) 폭넓은 ablation 으로 *진정한 ceiling* 측정 + lever attribution. Phase 0 diagnostics (재학습 0~minimal) → Phase 1 single-axis 25 sub-exp (1-fold approx, ★ 정보 핵심) → Phase 2 best-axis selection → Phase 3 pairwise 4 sub-exp (5-fold super-additive) → Phase 4 triple stack 1~2 sub-exp → 조건부 Phase 5 iterative / Phase 6 inference augment / Phase 7 synthesis. LB 제출 0 회 (plan-009.1 carry-over 패턴 답습) — plan-011.1 carry-over.
exp_ids:
  - H010_phase0-diagnostics           # G0 — D001 oracle simulation + plan-006 reproduce + decomp 재측정
  - H011_phase1-loss-ablation         # G1.L — P1.L0~L7 (loss axis 8 sub-exp)
  - H012_phase1-input-ablation        # G1.In — P1.IA~IF (input axis 5 sub-exp)
  - H013_phase1-arch-ablation         # G1.M — P1.M0~M6 (arch axis 7 sub-exp)
  - H014_phase1-formula-ablation      # G1.F — P1.F0~F4 (formula axis 4 sub-exp + F0 reuse)
  - H015_phase3-pairwise              # G2 — P3.1~P3.4 (4 pair 5-fold)
  - H016_phase4-triple                # G3 — P4.1 + (조건부) P4.2 (triple stack)
  - H017_phase5-iterative             # G4 (조건부) — iterative refinement on best stack
  - H018_phase6-augment               # G5 (조건부) — TTA + multi-parse inference
lb_score: null
---

# plan-011 v1 — Single-Formula + Corrector Path Exploration (4-axis breadth ablation)

## §0. 한 줄 목적

> **plan-006 Variant E (`frenet_par120_perp_neg020` 단일공식 + plan-004 corrector LB 0.6692) path 의 *구체화 + 진정한 ceiling 측정*. 이 path 의 corrector 는 단일공식과 결합하기에는 *제약 사항이 많고 잘못 설계됨* (plan-004 의 27-후보 selector + boundary 미세조정 역할 기반). 사용한 단일공식 또한 *최고로 좋은 공식이 아니었음에도* 준수한 LB 0.6692 — 이 구조에 맞춰 corrector 의 제약 7 개 + input snapshot 한계를 풀고 *여러 corrector 버전을 4 axis (Input × Loss × Arch × Formula) 위에서 폭넓게 탐색*.**
>
> **narrative 분리 (plan-010 과)**:
> - plan-010 = **depth** (plan-004 corrector 의 7 결함 *defect-by-defect fix*). 4 후보 (Z1+G2 / Z1+G1 / Z3+G2 / Z6) 의 sequential 진입.
> - **plan-011 = breadth** (단일공식 + corrector *path 자체* 의 구체화). 4 axis × ~25 single-axis ablation 으로 *각 axis 의 best lever attribution* 박제 + Phase 3+ 결합 측정.
>
> **두 plan 의 관계**: plan-010 산출 (`src/pb_0_6822/corrector_redesign.py` Z1 module) 을 plan-011 의 *anchor module reuse*. plan-011 은 그 위에 새 component (gate head, anisotropic loss, bell-shape weighting, GMM, bin head, learnable formula) 만 추가하여 *4 axis 탐색 framework* 완성.
>
> **Baseline 확정**: plan-006 LB 0.6692 (단일공식 frenet_par120_perp_neg020 + plan-004 corrector). OOF anchor = plan-007 per_candidate_hit 의 raw 0.6320 (단일공식 corrector 없이) + plan-006 의 0.6491 (corrected).
>
> **Target**: 단일공식 + corrector path 의 *4 axis ceiling* 박제. LB 추정 0.70~0.73 (Phase 4 triple stack 기준). plan-006 LB 0.6692 위 +0.03~0.06.
>
> **LB 제출 정책**: 본 plan 내 LB 제출 **0 회** (할당량 소진 상태 인계, plan-009.1 + plan-010.1 carry-over 패턴 답습). 모든 sub-exp submission.csv 는 *생성·박제만*, LB 회수는 plan-011.1 carry-over.

---

## §0.5 Quick Reference (autonomous loop 매 turn 읽는 section)

### 합격 기준 (G-gate sequence)

- **G0** (Phase 0 diagnostics): D001 oracle simulation (perfect gate ceiling) + plan-006 reproduce (raw single formula OOF ∈ [0.627, 0.637]) + plan-005 corrector_decomp 재측정 + `analysis/plan-011/preflight.json` 생성. 위반 시 `preflight_artifact_missing` severe.
- **G1** (Phase 1 single-axis ablation, ★ 정보 핵심): 4 axis 모두 완료 — L axis 8 sub-exp + In axis 5 sub-exp + M axis 7 sub-exp + F axis 4 sub-exp (F0 reuse). 1-fold approx (fold=0). 각 axis 의 *best lever* 식별. (a) 모든 24 sub-exp informational 완료 (fail 없음 — attribution 목적). (b) 4 axis 중 *최소 2 axis* 에서 +0.005 marginal OOF gain (single-formula + corrector path 의 *부정 방지*). 위반 시 `phase1_no_lever_positive` severe.
- **G2** (Phase 3 pairwise 5-fold): L̂ + In̂, L̂ + M̂, L̂ + F̂, In̂ + M̂ (4 pair). (a) `oof_soft_hit ≥ G1 best + 0.003` (super-additive 입증). (b) 4 pair 중 *최소 1 pair* additive 또는 super-additive (= 결합 OOF ≥ 두 단독 OOF 의 합 − base). 위반 시 `super_additive_fail` warn.
- **G3** (Phase 4 triple stack): L̂ + In̂ + M̂ (P4.1). (a) `oof_soft_hit ≥ G2 best + 0.003`. (b) (조건부) F̂ ΔOOF ≥ +0.005 시 P4.2 (L̂ + In̂ + M̂ + F̂) 추가. 위반 시 `triple_stack_marginal` warn.
- **G4** (Phase 5 iterative, **조건부**): G3 best OOF > 0.69 진입 조건. L̂ + In̂ + M̂ + Z3 iterative (3-step, per-step cap=3mm, parameter 공유). (a) `oof_soft_hit ≥ G3 + 0.005`. (b) `[1, 1.5cm) hit_after ≥ 0.20`. (c) iter_gap (train OOF − val OOF) ≤ 0.05. 위반 시 `iterative_divergence` severe.
- **G5** (Phase 6 inference augment, **조건부**): G3 또는 G4 best 위 TTA rotation 4 + multi-parse inference. (a) `oof_soft_hit ≥ G3 + 0.002` marginal. 위반 시 `augment_no_signal` warn-only (학습 X, 비용 ~free).
- **G_final**: synthesis + plan-012 후보 ≥ 3 + 3 파일 frontmatter 동시 박제 (`lb_score: TBD` carry-over) + best Phase submission 박제 + plan-011.1 carry-over instruction 박제.

### G-gates

- G0: Phase 0 diagnostics + preflight.json 생성 [TODO]
- G1: Phase 1 4-axis ablation (24 sub-exp) — 4 best lever 식별 + 최소 2 axis 에서 +0.005 [TODO]
- G2: Phase 3 pairwise (4 pair 5-fold) — super-additive 입증 [TODO]
- G3: Phase 4 triple stack (P4.1 + 조건부 P4.2) — OOF ≥ G2 + 0.003 [TODO]
- G4: Phase 5 iterative (조건부 G3 > 0.69) — [1,1.5cm) hit ≥ 0.20 + iter_gap ≤ 0.05 [TODO]
- G5: Phase 6 inference augment (조건부) — marginal +0.002 [TODO]
- G_final: synthesis + plan-012 후보 ≥ 3 + 3 파일 frontmatter sync + best Phase submission 박제 + plan-011.1 instruction [TODO]

### Commit chain (next-up)

| # | type | spec section | status |
|---|---|---|---|
| c1 | docs | `plans/plan-011-single-formula-corrector-exploration.md` v1 작성 | [TODO] |
| c2 | code | `analysis/plan-011/preflight.py` — D001 oracle simulation + plan-006 reproduce + corrector_decomp 재측정. spec @ §4 | [TODO] |
| G0 | gate | `preflight.json` 생성 + D001 박제 + reproduce ✓ + decomp drift ✓ | [TODO] |
| c3 | code | `src/pb_0_6822/corrector_redesign_v2.py` — 새 components (GateHead, anisotropic loss, BellShapeWeight, GMMHead, BinHead, IterativeRefiner, LearnableFormula). plan-010 의 `corrector_redesign.py` reuse + extend. spec @ §5.1 | [TODO] |
| c4 | code | `analysis/plan-011/phase1_loss_ablation.py` — P1.L0~L7 wrapper (8 sub-exp). spec @ §5.2 | [TODO] |
| c5 | exp | Phase 1.L 8 sub-exp 실행 (1-fold approx, ~10min/sub-exp ≈ ~80min) | partial G1 |
| c6 | code | `analysis/plan-011/phase1_input_ablation.py` — P1.IA~IF wrapper. spec @ §6 | [TODO] |
| c7 | exp | Phase 1.In 5 sub-exp 실행 (~60min) | partial G1 |
| c8 | code | `analysis/plan-011/phase1_arch_ablation.py` — P1.M0~M6 wrapper. spec @ §7 | [TODO] |
| c9 | exp | Phase 1.M 7 sub-exp 실행 (~100min) | partial G1 |
| c10 | code | `analysis/plan-011/phase1_formula_ablation.py` — P1.F0~F4 wrapper. spec @ §8 | [TODO] |
| c11 | exp | Phase 1.F 4 sub-exp 실행 (~50min, F0 reuse) | **G1** |
| c12 | analysis | `analysis/plan-011/phase1_attribution.md` — 4 axis ΔOOF 표 + best lever 식별. spec @ §9 | [TODO] |
| c13 | code+exp | Phase 3 pairwise: P3.1 (L̂+In̂) + P3.2 (L̂+M̂) + P3.3 (L̂+F̂) + P3.4 (In̂+M̂), 5-fold ~50min × 4 = ~200min. spec @ §10 | **G2** |
| c14 | code+exp | Phase 4 triple stack: P4.1 (L̂+In̂+M̂) 5-fold + (조건부) P4.2. spec @ §11 | **G3** |
| c15 | code+exp | (조건부 G3 > 0.69) Phase 5 iterative refinement. spec @ §12 | G4 |
| c16 | code+exp | (조건부) Phase 6 inference augment (TTA + multi-parse). spec @ §13 | G5 |
| c17 | analysis | `analysis/plan-011/results.md` + `next_plan_candidates.md` (≥ 3 후보) + 3 파일 frontmatter sync + best Phase submission 박제 + plan-011.1 carry-over instruction. spec @ §14 | **G_final** |
| c17.1 | sync | §0.5 [TODO]→[DONE] | — |

### Plan-specific severe (WORKFLOW.md §12.3 default 위 추가분)

- `preflight_artifact_missing` — G0 의 `preflight.json` 미생성 또는 plan-006 reproduce 실패 (|measured − 0.6320| > 0.005)
- `phase1_no_lever_positive` — G1 의 4 axis 중 *어느 axis 도* +0.005 marginal 없음 (= single-formula + corrector path 자체 부정)
- `iterative_divergence` — G4 의 iter_gap > 0.05 또는 [1,1.5cm) < 0.10
- `single_formula_residue` — selector 가 단일공식 외 다른 candidate 사용한 evidence (cand pool size > 1 또는 score variance > 1e-10 in non-F4 sub-exp)
- `frozen_gru_drift` — In-C frozen plan-004 GRU encoder parameter 변경 detected (state_dict diff > 0)
- `gate_collapse` — P1.L2 (C008 gate) 의 gate output 이 모든 sample 에서 < 0.05 또는 > 0.95 (gate 학습 실패)
- (v1.1 제거 유지) `lb_quota_exhausted` — LB 제출 0 회 정책으로 trigger 부재

### Plan-specific paths (WORKFLOW.md §12.5/§12.6 default 위 추가/제외)

- whitelist 추가:
  - `src/pb_0_6822/corrector_redesign_v2.py` (신규 모듈, 본 plan main code)
  - `analysis/plan-011/**` (preflight, phase1_*, phase3_*, phase4_*, phase5_*, phase6_*, results, next_plan_candidates)
- whitelist 제외 (blacklist 추가):
  - `src/pb_0_6822/boundary.py` (touch X — 모든 변경은 `corrector_redesign_v2.py` 신규 모듈에서. `boundary.py` 의 `compute_corrector_loss` hook 은 read-only reference)
  - `src/pb_0_6822/selector.py` (touch X — frozen GRU 는 `selector.AttnGRUCandidateSelector` 의 forward only)
  - `src/pb_0_6822/candidates_extended.py` (plan-008 산출, 본 plan scope X — 단일공식 만 사용)
  - `src/pb_0_6822/corrector_redesign.py` (plan-010 산출, *import only* — 본 plan 의 v2 module 가 reuse)
- 참조 (read-only):
  - `runs/baseline/P001_pb-0-6822-fullrun/**` (plan-004 산출, GRU checkpoint + corrector baseline)
  - `runs/baseline/F001_variant-e/**` (plan-006 산출, 단일공식 baseline)
  - `analysis/plan-005/corrector_decomp.{md,json}` (★ band table baseline)
  - `analysis/plan-007/per_candidate_hit.{md,json}` (★ raw single formula ranking)
  - `analysis/plan-007/mlp_coeff.{py,json}` (★ Step 4 per-sample MLP coeff carry-over)
  - `analysis/plan-010/results.md` (★ plan-010 Z1 결과, anchor)
  - `notes/PB_0.6822 코드공유.ipynb` (cell 6 boundary corrector 원본)

### Decision-note 사용 예 (자율 결정 시 commit msg 박제)

- `decision-note: spec-default — Phase 1 의 모든 axis 는 fixed L0 + In-A + M0 + F0 anchor 위에서 1-axis 만 변경 (attribution clean)`
- `decision-note: spec-default — Phase 1 fold=0 1-fold approx (~10min/sub-exp), Phase 3+ 5-fold concat 강제`
- `decision-note: conditional-skip — D001 < 0.66 시 P1.L2 (C008) skip, M4 iterative 우선`
- `decision-note: conditional-skip — F̂ ΔOOF < +0.005 시 P4.2 skip, F0 anchor 유지`
- `decision-note: conditional-skip — G3 best OOF < 0.69 시 G4 (iterative) skip, plan-011.1 carry-over`
- `decision-note: G1 attribution — L̂ = LX (ΔOOF=+0.0YY), In̂ = InY, M̂ = MZ, F̂ = FW`
- `decision-note: spec-default — C008 gate head bias init = +2.0 (sigmoid(2)=0.88, 시작 시 보정 ON, asymmetric learning)`

---

## §1. 배경 / 이전 plan 인계

### §1.1 plan-006 의 단일공식 + plan-004 corrector 결과 재해석

| 측정 | 값 | 출처 |
|---|---|---|
| plan-006 단일공식 picked | `frenet_par120_perp_neg020` (CANDIDATES[17]) | plan-006 §5.5 |
| raw single formula OOF | 0.6320 (corrector 없이) | plan-007 per_candidate_hit |
| corrected OOF (argmax + plan-004 corrector) | 0.6491 | plan-006 §5.5 |
| **LB (corrected)** | **0.6692** | plan-006 dacon-submit |
| OOF→LB gap | +0.0201 | plan-006 results |

**plan-007 의 4 단계 단일공식 개선 시도** (모두 *결함 corrector* 와 결합한 측정):

| step | 방법 | OOF | LB |
|---|---|---|---|
| plan-006 baseline | frenet_par120_perp_neg020 | 0.6491 | 0.6692 |
| Step 2 | CMA-ES tuned 6 vars | 0.6403 | 0.6570 |
| Step 3 | basis ablation best (8 vars) | 0.6403 | 0.6598 |
| **Step 4** | **per-sample MLP coeff** | **0.6482** | **carry-over 미회수** |

**→ 결론**: 4 측정 모두 *결함 corrector* 와 결합한 결과. 단일공식 framework 의 *진정한 ceiling* 미측정. ★ **plan-007 Step 4 의 LB 미회수** 가 *살아있는 카드*.

### §1.2 plan-005 corrector_decomp 의 destructive evidence (★ C008 motivation)

| band | n | hit_before (raw) | hit_after (corrected) | Δ |
|---|---|---|---|---|
| [0, 0.5cm) | ~5000 | high | high | 0 (already hit) |
| **[0.005, 0.010m)** | **2594** | **high (100%?)** | **lower (-203 hits)** | **★ -7.83pp (destructive band)** |
| [1, 1.5cm) | ~1100 | 0% | 9.77% | +9.77pp (회복 path) |
| [1.5+, 2cm) | ~350 | 0% | 0% | 0 (oracle ceiling) |

**plan-009 H002 sub-exp b 의 측정**: [1,1.5cm) hit_after 9.77% → **4.09%** (오히려 *감소*) — band weight tuning 이 *root cause 못 잡음* (= destructive band 의 회복은 *gate* 가 필요, *weight* 만으로는 X).

★ **C008 do-no-harm gate** = destructive band 의 *직접 fix* — gate 로 "이 sample 은 보정 안 함" 학습 + asymmetric loss 로 `raw_hit && corrected_miss` 페널티.

### §1.3 plan-005 corrector direction breakdown (★ C010 motivation)

| direction | 학습된 delta 평균 magnitude (m) |
|---|---|
| parallel (t-axis) | 0.0451 |
| perpendicular (n-axis) | 0.0214 |
| **binormal (b-axis)** | **0.0064** |

**→ binormal 학습이 parallel 의 1/7 — capacity 낭비**. C010 anisotropic loss (`w_bi=0.1`) = binormal head 학습 신호 축소 → 다른 head 의 capacity 회수.

### §1.4 plan-004 corrector 의 7 결함 (plan-010 anchor)

| # | 결함 | 위치 |
|---|---|---|
| ① | target = cap-truncated residual | boundary.py L108~110 |
| ② | MSE loss vs hit@1cm metric | boundary.py L259 |
| ③ | far_weight 0.04 | boundary.py L114 |
| ④ | easy_weight 0.20 | boundary.py L114 |
| ⑤ | env head (family CE) | boundary.py L185~190 |
| ⑥ | apply_scale 0.75 hack | boundary.py L327 |
| ⑦ | hard-coded band [0.7, 1.7cm] | boundary.py L368~369 |

**plan-010 의 Z1 minimum** = 6 fix (B1 + A2 + C1 + C2 + D1 + E1, ⑦ 만 별도). plan-011 의 *L1* = plan-010 의 Z1 그대로 reuse.

### §1.5 plan-004 corrector 의 input snapshot 한계 (★ Input axis motivation)

`make_candidate_features` 의 `cf` 32-dim 구성:
- candidate-relative 3 (par/perp/dist over scale)
- candidate spec 9 (d1, par, perp, d2, jerk, time_scale, omega_scale, arc_curvature, z_scale)
- ctx 9 (마지막 시점 motion: speed, prev_speed_ratio, acc_norm/speed, ...)
- interactions 4
- family one-hot 7 (extended pool 에서만)

**→ 시계열 정보 zero**. GRU 가 봤던 시계열 흐름 (SEQ_FEATURE_NAMES T step) 을 corrector 가 *전혀 못 봄*. plan-004 의 설계 의도 = "selector (GRU) 가 시계열 처리 + corrector 가 boundary 미세조정". **단일공식 path 에서 corrector 가 main lever 가 되면 *시계열 input 회수* 필수**.

### §1.6 plan-010 의 4 후보 (depth) — plan-011 의 anchor

| plan-010 후보 | plan-011 의 위치 |
|---|---|
| Z1 minimum viable (G1) | plan-011 의 P1.L1 anchor |
| Z1 + frozen GRU (G2) | plan-011 의 P1.IC + L1 결합 (P3 진입 시) |
| Z1 + CNN encoder (G2 변형) | plan-011 의 P1.ID |
| Z3 iterative + frozen GRU (G3) | plan-011 의 P1.M4 + IC 결합 (Phase 5) |
| Z6 e2e (G4 조건부) | plan-011 의 P1.IE |

**→ plan-011 = plan-010 의 *axis 확장* (loss/arch/formula axis 신설) + *combination 명시* (Phase 3+)**.

---

## §2. Scope (명시적)

### §2.1 In-scope

| 항목 | 값 |
|---|---|
| selector | **단일공식 only** — frenet_par120_perp_neg020 (anchor) + Phase 1.F 에서 F1~F4 swap |
| selector arch | 사용 X (단일공식 = K=1 candidate, ranking 없음) — F3/F4 만 별도 (per-sample MLP / learnable) |
| corrector arch | 4 axis: Input (In-A~In-F, 6 variant) × Loss (L0~L7, 8 variant) × Arch (M0~M6, 7 variant) × Formula (F0~F4, 5 variant) |
| LB 제출 | **0 회** (할당량 소진 인계, plan-011.1 carry-over) |
| 학습 데이터 | train 10K (plan-004 동일) |
| Validation | Phase 1: 1-fold OOF (fold=0, N_val≈2020) approx — binomial std ≤0.005. Phase 3+: 5-fold concat 강제 |
| GPU | server cuda:1 (plan-004/005/008/009/010 동일) |

### §2.2 Out-of-scope (절대 안 함)

| 항목 | 이유 |
|---|---|
| 27 후보 selector + corrector | plan-008/009 의 path. 본 plan = 단일공식 path 의 *진정한 ceiling* 측정 분리 |
| Set Transformer over candidates | candidate K=1 (단일공식) 이므로 무의미 |
| KNN / GP / Diffusion (paradigm 교체) | plan-012 후보. 본 plan 의 4-axis triple stack OOF < 0.70 시 plan-012 진입 조건 |
| boundary.py 본문 수정 | whitelist X. 모든 변경은 `corrector_redesign_v2.py` 신규 모듈 |
| selector.py 본문 수정 | whitelist X. GRU 는 frozen forward only |
| candidates_extended.py 사용 | plan-008 산출, 본 plan = 단일공식 만 |
| LB 제출 | 할당량 소진 (plan-009.1 + plan-010.1 까지 사용). 본 plan = carry-over |
| plan-010 의 corrector_redesign.py 본문 수정 | import only, plan-011 의 v2 module 가 reuse + extend |

---

## §3. 사전 등록 (Pre-registration)

### §3.1 Fold split

- 5-fold OOF: `selector.stable_fold_id(sample_id, folds=5)` (plan-004 동일)
- Phase 1 fold=0 only (N_val ≈ 2020, binomial std ≤0.005)
- Phase 3+ 5-fold concat 강제 (overall_oof_hit_soft)

### §3.2 합격 기준

§0.5 G-gate sequence 참조.

### §3.3 평가 점수 / median 집계

- main metric: **5-fold concat OOF soft hit @ 1cm** (Phase 3+) 또는 **1-fold OOF soft hit** (Phase 1)
- soft hit = `base.search_temperature(corrected, scores, true)["metrics"]["hit"]`
- per-band hit_after: `[0, 0.5)`, `[0.5, 1)`, `[1, 1.5)`, `[1.5, 2)`, `[2, ∞)` (plan-005 corrector_decomp schema)
- corrector_oracle_gain = `corrected_oracle_hit − raw_oracle_hit`
- ΔOOF (lever attribution) = `OOF_with_lever − OOF_anchor` per sub-exp

### §3.4 Anchor 정의 (Phase 1 의 모든 ablation 의 기준점)

- L0 (Loss anchor): plan-004 default (MSE on cap-truncated + far_weight 0.04 + easy_weight 0.20 + env_head + apply_scale 0.75 + hard-coded band)
- In-A (Input anchor): `cf` 32-dim snapshot only (plan-004 default)
- M0 (Arch anchor): TinyCorrectionNet (depth=2, hidden=64, plan-004 default)
- F0 (Formula anchor): frenet_par120_perp_neg020 (CANDIDATES[17])

**Anchor combo (= P1.L0 = P1.IA = P1.M0 = P1.F0)** = plan-006 baseline reproduce (corrected OOF 0.6491 ± 0.005).

---

## §4. STAGE 0 (G0) — Phase 0 Diagnostics + preflight

### §4.1 산출물

- `analysis/plan-011/preflight.py` — 3 task 일괄 실행
- `analysis/plan-011/preflight.json` — schema:
```json
{
  "exp_id": "H010_phase0-diagnostics",
  "d001_oracle_simulation": {
    "description": "perfect gate ceiling — destructive samples 모두 skip 시 OOF 상한",
    "plan_005_corrected_oof_npz": "<path>",
    "plan_005_raw_scores_path": "<path>",
    "n_train": 10000,
    "n_destructive_samples": <int>,
    "perfect_gate_oof_5fold": <float>,
    "anchor_oof_5fold": 0.6524,
    "delta": <float>,
    "go_no_go_threshold": 0.66,
    "c008_path_enabled": <bool>
  },
  "plan_006_reproduce": {
    "single_formula": "frenet_par120_perp_neg020",
    "candidate_idx": 17,
    "oof_argmax_hit_raw_measured": <float>,
    "oof_argmax_hit_raw_expected": 0.6320,
    "oof_argmax_hit_corrected_measured": <float>,
    "oof_argmax_hit_corrected_expected": 0.6491,
    "drift": <float>,
    "drift_threshold": 0.005,
    "reproduce_ok": <bool>
  },
  "corrector_decomp_remeasure": {
    "n_train": 10000,
    "band_table": {
      "[0,0.5cm)":   {"n_in_band": <int>, "hit_before": <float>, "hit_after": <float>, "delta": <float>},
      "[0.5,1cm)":   {"n_in_band": <int>, "hit_before": <float>, "hit_after": <float>, "delta": <float>},
      "[1,1.5cm)":   {"n_in_band": <int>, "hit_before": <float>, "hit_after": <float>, "delta": <float>},
      "[1.5,2cm)":   {"n_in_band": <int>, "hit_before": <float>, "hit_after": <float>, "delta": <float>},
      "[2cm,inf)":   {"n_in_band": <int>, "hit_before": <float>, "hit_after": <float>, "delta": <float>}
    },
    "destructive_band_evidence": {
      "band": "[0.5, 1cm)",
      "n_samples": <int>,
      "hits_lost": <int>,
      "plan_005_baseline_lost": -203,
      "drift_ok": <bool>
    },
    "direction_breakdown": {
      "parallel_delta_norm_mean": <float>,
      "parallel_baseline": 0.0451,
      "perp_delta_norm_mean": <float>,
      "perp_baseline": 0.0214,
      "binormal_delta_norm_mean": <float>,
      "binormal_baseline": 0.0064,
      "drift_ok": <bool>
    }
  }
}
```

### §4.2 실행

```bash
python -m analysis.plan-011.preflight \
  --root data \
  --plan-005-corrected-oof <path> \
  --plan-005-raw-scores <path> \
  --plan-006-checkpoint runs/baseline/F001_variant-e/... \
  --out analysis/plan-011/preflight.json
```

### §4.3 G0 합격

- D001 oracle simulation 박제 (재학습 0)
- plan-006 reproduce drift ≤ 0.005 (raw 와 corrected 둘 다)
- corrector_decomp drift ≤ 0.01 per band
- destructive band evidence 박제 (n_samples ≈ 2594, hits_lost ≈ -203)
- direction breakdown 박제 (binormal_baseline 0.0064 ± 10%)

### §4.4 G0 후 판단 (autonomous branching)

- D001 perfect_gate_oof_5fold < 0.66 → `c008_path_enabled=false` → P1.L2 skip + P1.L3 (C009 lite) 만 유지
- D001 perfect_gate_oof_5fold ≥ 0.66 → P1.L2 정상 진입 (★ expected main lever)
- D001 perfect_gate_oof_5fold ≥ 0.70 → P1.L2 격상 (Phase 3 P3.1 진입 시 C008 가 L̂ 후보 1 순위)

---

## §5. STAGE 1.L (Phase 1.L) — Loss Axis Ablation (8 sub-exp)

### §5.1 corrector_redesign_v2.py 신규 모듈 (Loss components)

```python
# src/pb_0_6822/corrector_redesign_v2.py

import torch
from torch import nn
import torch.nn.functional as F
from src.pb_0_6822 import corrector_redesign as v1  # plan-010 reuse


# ── Loss components ──

def huber_loss(pred, target, beta=0.005):
    """L1 (and L4 wrapper): Huber loss, beta=5mm threshold. (B,) per-sample."""
    return F.smooth_l1_loss(pred, target, beta=beta, reduction='none').sum(dim=1)


def asymmetric_loss(pred, target, raw_hit_mask, corrected_pos, lambda_destructive=8.0):
    """L2 (C008) + L3 (C009): asymmetric loss penalizing destructive moves.

    raw_hit_mask: (B,) bool — True if raw candidate (before delta) already hit
    corrected_pos: (B, 3) — cand + delta_applied (cap=0.006)
    """
    base_loss = huber_loss(pred, target)  # (B,)
    err_after = torch.norm(corrected_pos - target, dim=1)
    corrected_miss = (err_after > 0.01)  # R_HIT
    destructive = raw_hit_mask & corrected_miss  # (B,) bool
    penalty = base_loss * lambda_destructive
    return torch.where(destructive, base_loss + penalty, base_loss)


def frenet_anisotropic_loss(pred_local, target_local, w_par=1.0, w_perp=1.0, w_bi=0.1):
    """L4 (C010): Frenet local-frame anisotropic. pred_local/target_local: (B, 3) in (t, n, b).

    decision-note: spec-default — w_bi=0.1 (plan-005 binormal 0.0064 / parallel 0.0451 ≈ 1/7).
    """
    diff = pred_local - target_local
    return w_par * diff[:, 0]**2 + w_perp * diff[:, 1]**2 + w_bi * diff[:, 2]**2


def physics_conservation_loss(delta, recent_acc, recent_jerk, typical_jerk_step=0.004):
    """L5: CPhy-ML — kinematically implausible delta 에 페널티.

    delta: (B, 3), recent_acc/recent_jerk: (B, 3).
    """
    delta_jerk_norm = torch.norm(delta - recent_acc, dim=1)
    penalty = torch.clamp(delta_jerk_norm - typical_jerk_step, min=0.0) ** 2
    return penalty


def bell_shape_weight(err, R_HIT=0.01, sigma=0.005):
    """L6: Gaussian-shaped weight centered at R_HIT. (B,) → (B,)."""
    return torch.exp(-((err - R_HIT) / sigma) ** 2)


def hit_aware_hinge(corrected_pos, target, R_HIT=0.01, smooth=0.005):
    """L7: smooth hinge — max(0, err - R_HIT)² with smooth approx.

    corrected_pos: (B, 3), target: (B, 3). 미분 가능.
    """
    err = torch.norm(corrected_pos - target, dim=1)
    excess = err - R_HIT
    return F.softplus(excess / smooth) * smooth  # smooth approx of max(0, x)
```

### §5.2 Phase 1.L wrapper (`analysis/plan-011/phase1_loss_ablation.py`)

8 sub-exp 일괄 실행 (fixed In-A + M0 + F0, fold=0, ~10min/sub-exp):

| sub-exp | loss config |
|---|---|
| P1.L0 (anchor) | plan-004 default (MSE + far=0.04 + easy=0.20 + env_loss_weight=0.05 + apply_scale=0.75 + boundary [0.7, 1.7cm]) |
| P1.L1 | Z1 minimum: uncapped target + huber(β=0.005) + far=0.5 + easy=0 + env_loss_weight=0 + apply_scale=1 + boundary [0.7, 1.7cm] |
| **P1.L2** (★ 조건부 D001 ≥ 0.66) | Z1 + C008 gate (sigmoid head, bias init +2.0) + asymmetric loss (λ=8) |
| P1.L3 | Z1 + C009 (asymmetric loss only, gate 없이) |
| **P1.L4** | Z1 + C010 Frenet anisotropic (w_par=1, w_perp=1, w_bi=0.1) |
| P1.L5 | Z1 + L5 physics conservation (jerk penalty λ=0.5) |
| P1.L6 | Z1 + L6 bell-shape weight (σ=0.005) |
| P1.L7 | Z1 + L7 hit-aware smooth hinge (combined: huber × 0.5 + hinge × 0.5) |

### §5.3 산출 (per sub-exp)

- `runs/baseline/H011_phase1-loss-ablation/sub_L{N}/`
  - `boundary_val_predictions.npz` (fold 0 val, K=1)
  - `report_sub_L{N}.json` (oof_soft_hit, per-band hit_after, corrector_oracle_gain, gate_stats if L2, elapsed)
- `analysis/plan-011/phase1_loss_summary.json` (8 sub-exp 통합)

### §5.4 G1.L 합격

- 8 sub-exp 모두 informational 완료 (fail 없음 — attribution 목적)
- 최소 1 sub-exp 가 P1.L0 anchor 대비 +0.005 marginal OOF
- best L̂ 식별 (max ΔOOF)

### §5.5 G1.L fail handling

- 모든 sub-exp 가 anchor 대비 ≤ +0.005 → `loss_axis_no_lever_positive` warn. 다른 axis (In/M/F) 가 main lever 가능성 — Phase 1.In/M/F 계속 진행.
- P1.L2 gate output collapse → `gate_collapse` severe, autonomous:
  - 옵션 a: bias init +2.0 → +3.0 retry (sigmoid 더 ON-biased)
  - 옵션 b: λ_destructive 축소 (8 → 4) retry
  - 옵션 c: L2 skip, L3 (C009) 만 신뢰

---

## §6. STAGE 1.In (Phase 1.In) — Input Axis Ablation (5 sub-exp + IA anchor)

### §6.1 corrector_redesign_v2.py — Input encoders

```python
# Input adapters
class TrajectoryStatsFeature(nn.Module):
    """In-B: hand-crafted trajectory statistics (no learning, 20-dim)."""
    def compute(self, trajectory_x):
        # trajectory_x: [N, T, 3] world coords
        # 4 + 3 + 2 + 3 + 3 + 3 + 2 = 20 dim
        # speed: mean/std/last/max (4)
        # acc_norm/speed: mean/std/max (3)
        # acc_par/speed: mean/std (2)
        # acc_perp/speed: mean/std/max (3)
        # jerk: mean/std/max (3)
        # turn_cos: mean/std/last (3)
        # curvature: mean/max (2)
        ...

class FrozenGRUEncoder(nn.Module):
    """In-C: plan-004 GRU encoder, frozen forward only (32-dim hidden)."""
    def __init__(self, plan_004_ckpt_path):
        super().__init__()
        # Load plan-004 selector.AttnGRUCandidateSelector checkpoint
        # extract gru layer, freeze
        ...
    @torch.no_grad()
    def forward(self, x_seq):
        # x_seq [N, T, 9] SEQ_FEATURE_NAMES
        # → GRU hidden [N, 32]
        ...

class TrajectoryCNNEncoder(nn.Module):
    """In-D: 1-D CNN encoder over SEQ feature, learnable (64-dim).

    spec @ plan-010 §6.1 reuse."""
    ...

class MultiParseInput(nn.Module):
    """In-F: raw + Savitzky-Golay smoothing + EMA smoothing (3 parse 평균).

    spec @ notes/prior-ideas.md §3 MTP."""
    def parse(self, trajectory_x, end_idx):
        # 3 parse 각각 cf 계산 → mean
        ...
```

### §6.2 Phase 1.In wrapper

5 sub-exp (fixed L0 + M0 + F0):

| sub-exp | input |
|---|---|
| P1.IA (anchor) | `cf` 32-dim snapshot only |
| **P1.IB** | + trajectory stats 20-dim (cheap, no encoder) |
| **P1.IC** | + frozen plan-004 GRU hidden 32-dim |
| P1.ID | + CNN encoder 64-dim (learnable) |
| P1.IF | + multi-parse (raw + SG + EMA) inference + train |

### §6.3 산출

- `runs/baseline/H012_phase1-input-ablation/sub_{IA,IB,IC,ID,IF}/`
- `analysis/plan-011/phase1_input_summary.json`

### §6.4 G1.In 합격

- 5 sub-exp 모두 완료
- best In̂ 식별

### §6.5 G1.In fail handling

- 모든 sub-exp 가 anchor 대비 ≤ +0.003 → `input_axis_no_lever_positive` warn-only. snapshot 한계가 *진짜 한계 아님* 신호.
- In-C frozen GRU state_dict diff > 0 → `frozen_gru_drift` severe. checkpoint reload retry.

---

## §7. STAGE 1.M (Phase 1.M) — Architecture Axis Ablation (7 sub-exp + M0 anchor)

### §7.1 corrector_redesign_v2.py — Architecture variants

```python
class GateHeadCorrector(v1.RedesignedCorrectionNet):
    """M1: TinyCorrectionNet + gate head (C008 structural).

    Gate output: sigmoid(MLP(features)) ∈ [0,1]
    Final delta = gate × raw_delta
    """
    def __init__(self, dim_cf, hidden=64, dim_encoder=0, gate_bias_init=2.0):
        super().__init__(dim_cf=dim_cf, hidden=hidden, dim_encoder=dim_encoder)
        self.gate_head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 1),
        )
        nn.init.constant_(self.gate_head[-1].bias, gate_bias_init)

    def forward(self, cf, encoder_emb=None):
        # ... stem + blocks (parent)
        raw_delta = self.delta(h)
        gate = torch.sigmoid(self.gate_head(h))  # (B, 1)
        return gate * raw_delta, gate


class SplitHeadCorrector(v1.RedesignedCorrectionNet):
    """M2: direction (unit vector) + magnitude (scalar) split heads."""
    def __init__(self, dim_cf, hidden=64, dim_encoder=0):
        super().__init__(dim_cf=dim_cf, hidden=hidden, dim_encoder=dim_encoder)
        self.delta = None  # remove default
        self.direction_head = nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, hidden // 2), nn.GELU(),
            nn.Linear(hidden // 2, 3),
        )
        self.magnitude_head = nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, hidden // 2), nn.GELU(),
            nn.Linear(hidden // 2, 1), nn.Softplus(),
        )

    def forward(self, cf, encoder_emb=None):
        # ... stem + blocks
        direction = F.normalize(self.direction_head(h), dim=-1)
        magnitude = self.magnitude_head(h)
        return direction * magnitude  # (B, 3)


class BinClassifierCorrector(v1.RedesignedCorrectionNet):
    """M3: bin classification head (60 bins × 1mm = ±3cm coverage)."""
    def __init__(self, dim_cf, hidden=64, dim_encoder=0, bin_dim=60, bin_size=0.001):
        super().__init__(dim_cf=dim_cf, hidden=hidden, dim_encoder=dim_encoder)
        self.delta = None
        self.bin_head = nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, hidden // 2), nn.GELU(),
            nn.Linear(hidden // 2, bin_dim ** 3),  # ★ 주의: bin^3 폭주 — 1D × 3 axis 로 factorize
        )
        # Better: 3 × bin_dim heads (factorized)
        self.bin_heads = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(hidden), nn.Linear(hidden, hidden // 2), nn.GELU(),
                nn.Linear(hidden // 2, bin_dim),
            ) for _ in range(3)
        ])
        self.bin_size = bin_size
        self.bin_dim = bin_dim
        self.bin_centers = torch.linspace(-bin_dim/2 * bin_size, bin_dim/2 * bin_size, bin_dim)

    def forward(self, cf, encoder_emb=None):
        # ... stem + blocks
        # Per-axis softmax: expected delta = Σ prob_i × bin_center_i
        delta_per_axis = []
        for head in self.bin_heads:
            logits = head(h)  # (B, bin_dim)
            prob = F.softmax(logits, dim=-1)
            expected = (prob * self.bin_centers.to(prob.device)).sum(dim=-1)
            delta_per_axis.append(expected)
        return torch.stack(delta_per_axis, dim=-1)  # (B, 3)


class IterativeRefinementCorrector(nn.Module):
    """M4: 3-step iterative refinement (parameter shared). spec @ plan-010 §7.1."""
    # ... plan-010 reuse


class GMMCorrector(v1.RedesignedCorrectionNet):
    """M5: probabilistic — μ + diagonal Σ output. Loss = NLL.

    Inference: expected delta = μ (or μ + samples for uncertainty)."""
    def __init__(self, dim_cf, hidden=64, dim_encoder=0):
        super().__init__(dim_cf=dim_cf, hidden=hidden, dim_encoder=dim_encoder)
        self.delta = None
        self.mu_head = nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, hidden // 2), nn.GELU(),
            nn.Linear(hidden // 2, 3),
        )
        self.logsigma_head = nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, hidden // 2), nn.GELU(),
            nn.Linear(hidden // 2, 3),
        )

    def forward(self, cf, encoder_emb=None):
        # ... stem + blocks
        mu = self.mu_head(h)
        logsigma = self.logsigma_head(h).clamp(min=-6.0, max=0.0)  # numerical stability
        return mu, logsigma


class WiderShallowCorrector(v1.RedesignedCorrectionNet):
    """M6: depth=1, hidden=256. small data 적합 추정."""
    def __init__(self, dim_cf, hidden=256, dim_encoder=0):
        super().__init__(dim_cf=dim_cf, hidden=hidden, dim_encoder=dim_encoder)
        # remove second residual block
        self.blocks = v1.ResidualMLPBlock(hidden)  # depth=1
```

### §7.2 Phase 1.M wrapper

7 sub-exp (fixed L0 + In-A + F0):

| sub-exp | arch |
|---|---|
| P1.M0 (anchor) | TinyCorrectionNet depth=2 hidden=64 |
| **P1.M1** | + gate head (C008 structural, no asymmetric loss in L0) |
| P1.M2 | direction + magnitude split heads |
| P1.M3 | bin classification (3 × 60-bin factorized) |
| **P1.M4** | iterative refinement (3-step, per-step cap=3mm) |
| P1.M5 | GMM (μ, σ) output, NLL loss |
| P1.M6 | wider shallow (depth=1, hidden=256) |

### §7.3 산출

- `runs/baseline/H013_phase1-arch-ablation/sub_M{N}/`
- `analysis/plan-011/phase1_arch_summary.json`

### §7.4 G1.M 합격

- 7 sub-exp 모두 완료
- best M̂ 식별

---

## §8. STAGE 1.F (Phase 1.F) — Single Formula Axis Ablation (4 sub-exp + F0 reuse)

### §8.1 corrector_redesign_v2.py — Formula variants

```python
class PerSampleMLPFormula(nn.Module):
    """F3: per-sample coefficient regression (plan-007 Step 4).

    MLP outputs (par_i, perp_i) for each sample → frenet candidate with per-sample coefs."""
    def __init__(self, in_dim, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden), nn.GELU(),
            nn.Linear(hidden, 2),  # (par, perp)
        )
        # init to plan-006 anchor: par=1.20, perp=-0.20
        with torch.no_grad():
            self.net[-1].bias[0] = 1.20
            self.net[-1].bias[1] = -0.20

    def forward(self, ctx_features):
        # ctx_features: (B, in_dim) — same as plan-007 mlp_coeff input
        return self.net(ctx_features)  # (B, 2) → par, perp per sample


class LearnableSingleCandidate(nn.Module):
    """F4: data-driven learnable candidate (Idea 2 from 코드공유-upgrade.md).

    Learn (d1, d2, par, perp, jerk, time_scale) as single 6-dim parameter via soft-min loss.
    Initialized to F0 (frenet_par120_perp_neg020).
    """
    def __init__(self, init_coef=(1.94, 0.0, 1.10, -0.20, 0.0, 1.0)):
        super().__init__()
        self.coef = nn.Parameter(torch.tensor(init_coef, dtype=torch.float32))

    def forward(self, p0, d1, d2, acc_par, acc_perp, jerk, horizon=2):
        # Generate candidate via current_coef × motion_terms
        ...
```

### §8.2 Phase 1.F wrapper

4 sub-exp (fixed L0 + In-A + M0):

| sub-exp | formula |
|---|---|
| P1.F0 (anchor) | frenet_par120_perp_neg020 — reuse from P1.L0 |
| **P1.F1** | CMA-ES tuned 6 vars (plan-007 Step 2 best_params reuse) |
| P1.F2 | basis ablation best (plan-007 Step 3 8 vars reuse) |
| **P1.F3** | per-sample MLP coefficient regression (plan-007 Step 4 reuse) |
| **P1.F4** | learnable single candidate (Idea 2, soft-min loss + diversity reg) |

### §8.3 산출

- `runs/baseline/H014_phase1-formula-ablation/sub_F{N}/`
- `analysis/plan-011/phase1_formula_summary.json`

### §8.4 G1.F 합격

- 4 sub-exp 완료 (F0 reuse)
- best F̂ 식별 (max ΔOOF vs F0 anchor)

### §8.5 G1.F fail handling

- 모든 F sub-exp ≤ F0 anchor → `formula_swap_marginal` warn. F̂ = F0 fix → P3.3 skip, P4.2 skip
- F3 (per-sample MLP) 결과 ≥ F0 + 0.005 → ★ plan-007 Step 4 LB 미회수 의 *fact 측정* — plan-011.1 carry-over 최우선

---

## §9. STAGE 2 (Phase 2) — Attribution + Best-axis Selection

### §9.1 산출

- `analysis/plan-011/phase1_attribution.md` (10 section):
  1. §1 요약 (4 best lever 식별)
  2. §2 L axis 표 (8 sub-exp × ΔOOF + per-band)
  3. §3 In axis 표 (5 sub-exp)
  4. §4 M axis 표 (7 sub-exp)
  5. §5 F axis 표 (4 sub-exp)
  6. §6 cross-axis informational (예: L1 + IC implicit signal)
  7. §7 decision-note (L̂/In̂/M̂/F̂ 채택 사유)
  8. §8 phase3 진입 후보 list (4 pair)
  9. §9 caveat 검증
  10. §10 변경 이력

### §9.2 G1 (전체) 합격

- 4 axis ablation 모두 완료 (24 sub-exp)
- 최소 2 axis 에서 +0.005 marginal OOF (single-formula + corrector path 의 *부정 방지*)

### §9.3 G1 fail handling

- 4 axis 모두 ≤ +0.005 → `phase1_no_lever_positive` severe. autonomous:
  - 옵션 a: Phase 3 skip, G_final 직접 진입 (best Phase = Phase 1 max + plan-012 paradigm 교체 carry-over)
  - 옵션 b: Phase 5 (iterative) 단독 진입 (Z3 가 plan-005 의 9.77% 회복 가능성)

---

## §10. STAGE 3 (Phase 3) — Pairwise Combinations (G2)

### §10.1 4 pair (5-fold)

| sub-exp | combo | 진입 조건 |
|---|---|---|
| **P3.1** | L̂ + In̂ | 항상 |
| **P3.2** | L̂ + M̂ | 항상 |
| **P3.3** | L̂ + F̂ | F̂ ≠ F0 (P1.F 에서 ΔOOF > 0) |
| **P3.4** | In̂ + M̂ | 항상 |

### §10.2 5-fold OOF 강제

각 pair 5-fold concat OOF 측정 (binomial std ≤0.005 of fold-0 보다 정확).

### §10.3 산출

- `runs/baseline/H015_phase3-pairwise/sub_P3_{1..4}/`
- `analysis/plan-011/phase3_summary.json` (4 pair OOF + per-band + super-additive class)

### §10.4 G2 합격

- (a) `oof_soft_hit ≥ G1 best + 0.003` per pair
- (b) 최소 1 pair 가 additive 또는 super-additive (= 결합 ΔOOF ≥ Σ 단독 ΔOOF − base)

### §10.5 super-additive 분류

```python
delta_pair = oof_pair − oof_anchor
delta_solo_sum = (oof_lever_a − oof_anchor) + (oof_lever_b − oof_anchor)
if delta_pair > delta_solo_sum + 0.003:
    cls = "super-additive"
elif abs(delta_pair - delta_solo_sum) <= 0.003:
    cls = "additive"
else:
    cls = "sub-additive"
```

### §10.6 G2 fail handling

- 모든 pair sub-additive → `super_additive_fail` warn. lever 들이 경쟁 관계 — Phase 4 triple stack 진입 보수적.

---

## §11. STAGE 4 (Phase 4) — Triple Stack (G3)

### §11.1 P4.1 — L̂ + In̂ + M̂ (5-fold)

P3 의 best pair 위 *제3 lever* 추가:
- 진입 조건: P3 best OOF ≥ G1 best + 0.003 (G2 합격)
- spec: L̂ loss + In̂ encoder + M̂ arch, F0 anchor 고정

### §11.2 (조건부) P4.2 — L̂ + In̂ + M̂ + F̂

- 진입 조건: P1.F 의 F̂ ΔOOF ≥ +0.005 (= formula swap 의미 있음)
- spec: P4.1 위에 F̂ formula swap

### §11.3 산출

- `runs/baseline/H016_phase4-triple/sub_P4_{1,2}/`
- `analysis/plan-011/phase4_summary.json`

### §11.4 G3 합격

- P4.1 OOF ≥ P3 best + 0.003

### §11.5 G3 fail handling

- P4.1 OOF < P3 best → `triple_stack_marginal` warn. P4.2 skip + best Phase = P3 best + Phase 5 진입 보수적 결정.

---

## §12. STAGE 5 (Phase 5, 조건부) — Iterative Refinement (G4)

### §12.1 진입 조건

- G3 best OOF > 0.69 (LB 추정 ≥ 0.712 with gap +0.022)
- 시간 여유 ≥ 70min

### §12.2 spec

P4 best 위 Z3 iterative:
- IterativeRefinementCorrector (n_steps=3, per_step_cap=3mm, parameter 공유, step_idx embedding)
- L̂ loss + In̂ encoder + M̂ arch + Z3 wrapper
- 5-fold OOF

### §12.3 G4 합격

- (a) `oof_soft_hit ≥ G3 + 0.005`
- (b) `[1, 1.5cm) hit_after ≥ 0.20`
- (c) iter_gap (train OOF − val OOF) ≤ 0.05

### §12.4 G4 fail handling

- (b) fail → `iterative_divergence` severe. per_step_cap 축소 (3mm → 2mm) + n_steps ↑ (3 → 5) retry. 그래도 fail → G4 skip, G3 best 채택.
- (c) fail (over-fit) → n_steps ↓ (3 → 2) retry.

---

## §13. STAGE 6 (Phase 6, 조건부) — Inference Augmentation (G5)

### §13.1 진입 조건

- G3 또는 G4 best 완료
- 시간 여유 ≥ 30min

### §13.2 spec

**P6.1 — TTA rotation 4**:
```python
# 추론 시 입력 XY 평면 회전 (0°, 90°, 180°, 270°) × 모델 forward × 역회전 평균
# Z축 (중력) 건드리지 않음 — 물리적 대칭성
for theta in [0, 90, 180, 270]:
    x_rot = rotate_xy(test_x, theta)
    delta_rot = model(x_rot)
    delta = rotate_xy_inverse(delta_rot, theta)
    deltas.append(delta)
final_delta = mean(deltas)
```

**P6.2 — Multi-parse inference**:
```python
# 추론 시 입력 raw + SG smoothing + EMA smoothing 3 parse × 모델 forward × 평균
x_raw = test_x
x_sg = savgol_filter(test_x, window=5, order=2, axis=time)
x_ema = ema_smooth(test_x, alpha=0.6)
delta = (model(x_raw) + model(x_sg) + model(x_ema)) / 3
```

### §13.3 산출

- `runs/baseline/H018_phase6-augment/sub_P6_{1,2}/`
- `analysis/plan-011/phase6_summary.json`

### §13.4 G5 합격

- (a) `oof_soft_hit ≥ G3 (또는 G4) best + 0.002` marginal

### §13.5 G5 fail handling

- marginal — warn-only (학습 X, 비용 ~free). best Phase 유지.

---

## §14. STAGE 7 (G_final) — Synthesis + plan-011.1 carry-over

### §14.1 산출

- `analysis/plan-011/results.md` (10 section)
- `analysis/plan-011/next_plan_candidates.md` (≥ 3 후보)
- 3 파일 frontmatter sync:
  - `plans/plan-011-single-formula-corrector-exploration.md` (status: partial/complete + best_submission)
  - `plans/plan-011-single-formula-corrector-exploration.results.md` (frontmatter only stub)
  - `analysis/plan-011/results.md` (자세한 finding)
- best Phase submission 경로 박제: `runs/baseline/<best_H_exp_id>/sub_<name>/submission.csv`
- plan-011.1 carry-over instruction

### §14.2 results.md 필수 항목

1. §1 요약 (best Phase, 4 axis attribution, OOF, LB 추정 / TBD)
2. §2 OOF 표 (전체 Phase 1~6 sub-exp 통합)
3. §3 per-Phase contribution (ΔOOF)
4. §4 4 axis attribution (L̂/In̂/M̂/F̂ + 단독 ΔOOF + 결합 super-additive class)
5. §5 per-band Δ table (plan-005 corrector_decomp 패턴)
6. §6 destructive band recovery 측정 (★ C008 효과 검증)
7. §7 decision-note list
8. §8 plan-012 후보 (≥ 3)
9. §9 변경 이력
10. §10 plan-011.1 carry-over instruction

### §14.3 plan-012 후보 (≥ 3)

- 후보 1: **best Phase + 27 후보 selector 결합** (plan-008/009 baseline 위 단일공식 corrector lever 의 일반화 효과 측정)
- 후보 2: **best Phase + per-sample MLP coeff (F3) 5-fold 강제** (1-fold approx 의 over-fit risk 검증)
- 후보 3 (조건부 best Phase OOF < 0.70): **paradigm 교체** (KNN over single formula candidates / GP posterior mean / Diffusion-style iterative)
- 후보 4 (조건부): **Idea 1 연속 heatmap regime bias + best corrector** (코드공유-upgrade.md Idea 1, single formula 위)

---

## §15. 병렬 실행 정책 (server: CPU 48 core + GPU 1 device:0)

### §15.1 의존성 그래프 (Phase 간 = 직렬 강제)

```
Phase 0 (preflight)
   ↓ D001 결과 → P1.L2 진입 여부 결정 (조건부)
Phase 1 (24 sub-exp, 4 axis ablation) ← ★ sub-exp 병렬 가능
   ↓ 4 axis best 선정 (L̂, In̂, M̂, F̂)
Phase 2 (attribution, 비용 0)
   ↓ 4 best lever 식별
Phase 3 (4 pair, pairwise) ← ★ pair 병렬 가능 + fold 병렬
   ↓ best pair 선정
Phase 4 (triple stack) ← fold 병렬
   ↓
Phase 5 (iterative, 조건부) ← fold 병렬
Phase 6 (augment, 조건부) ← 추론만, 병렬 free
   ↓
Phase 7 (synthesis)
```

**원칙**: Phase 간 직렬 강제 (이전 Phase 결과가 다음 진입 결정). Phase 내 sub-exp 는 anchor 고정 (L0+IA+M0+F0 위 1 lever 만 변경) → *독립 → 병렬 가능*.

### §15.2 3-Layer 병렬 정책

| Layer | 대상 | 병렬 도구 | 단축 효과 |
|---|---|---|---|
| **A. Sub-exp** | Phase 1 의 24 sub-exp + Phase 3 의 4 pair | GPU multi-stream (CUDA streams + multi-process) 4-way | ~60% (4 × 0.4) |
| **B. CV fold** | Phase 3/4/5 의 5-fold concat | multiprocessing.Pool(n=5) — 각 fold = 1 process, GPU memory 분할 | ~75% (5 × 0.25) |
| **C. CPU 데이터 prep** | feature compute / Frenet basis / trajectory stats / OOF assembly | multiprocessing.Pool(n=24) | GPU idle 활용 |

### §15.3 GPU 동시 학습 capacity

| 항목 | 값 |
|---|---|
| TinyCorrectionNet parameter | ~50K (depth=2, hidden=64) |
| forward + backward + optimizer state 모델당 메모리 | ~30 MB |
| batch_size 4096 × 32-dim feature | ~0.5 MB |
| **GPU memory 24~40GB 기준 동시 모델 최대** | ~수백 (이론) |
| **실제 sweet spot (kernel launch overhead 고려)** | **4-way multi-stream** |
| 8-way 이상 | marginal gain (~65% 단축에서 saturate) |

### §15.4 실제 구현 ([analysis/plan-011/_runtime.py](analysis/plan-011/_runtime.py) 신규)

```python
# CPU 데이터 prep 병렬
from multiprocessing import Pool

def compute_features_for_fold(args):
    fold_id, train_x, train_y, candidates = args
    cf = make_candidate_features(...)
    return fold_id, cf

with Pool(processes=24) as pool:
    results = pool.map(compute_features_for_fold, fold_args_list)


# GPU multi-stream (Phase 1 의 sub-exp 4-way 병렬)
import torch
streams = [torch.cuda.Stream() for _ in range(4)]
models = [build_model(cfg) for cfg in batch_of_4_subexp_configs]

for batch in shared_dataloader:
    for i, (model, stream) in enumerate(zip(models, streams)):
        with torch.cuda.stream(stream):
            loss = model(batch)
            loss.backward()
    torch.cuda.synchronize()


# 5-fold 병렬 (Phase 3+ 의 각 sub-exp 안)
from concurrent.futures import ProcessPoolExecutor

def train_one_fold(fold_idx, cfg, gpu_mem_fraction=0.18):
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(gpu_mem_fraction)
    ...

with ProcessPoolExecutor(max_workers=5) as executor:
    futures = [executor.submit(train_one_fold, k, cfg) for k in range(5)]
    results = [f.result() for f in futures]
```

### §15.5 reproducibility 보장

병렬 실행 시 *non-determinism* 위험. 강제 정책:

```python
# 매 sub-exp 진입 시
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.use_deterministic_algorithms(True, warn_only=True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

각 sub-exp 의 seed 는 spec 박제 (예: P1.L0 seed=20260513, P1.L1 seed=20260514, ...). 병렬 실행 후 단일 실행 reproduce 가능해야 함.

### §15.6 fail handling

- GPU OOM (multi-stream 4-way 진입 시) → 2-way 로 fallback retry, decision-note 박제
- multiprocessing deadlock (5-fold ProcessPoolExecutor 의 zombie) → 순차 (n_workers=1) fallback retry
- non-determinism drift (병렬 실행 OOF vs 순차 reproduce OOF |Δ| > 0.005) → `parallel_reproducibility_drift` warn, deterministic algorithm 강제 retry

---

## §N+1. 작업량 총 회계 (3 시나리오)

### §N+1.1 시나리오 A — 순차 (anchor)

| commit | task | 예상 wall-time |
|---|---|---|
| c1 (docs) | plan-011 v1 spec 작성 | 0 (본 commit) |
| c2 (G0) | preflight.py + 3 task (D001 + reproduce + decomp 재측정) | ~15 min |
| c3 (v2 module) | corrector_redesign_v2.py — 8 새 components + Loss 7 functions | ~30 min |
| c4 (P1.L wrapper) | phase1_loss_ablation.py (8 sub-exp orchestrator) | ~15 min |
| c5 (P1.L exec) | 8 sub-exp × ~10min (1-fold approx) | ~80 min |
| c6 (P1.In wrapper) | phase1_input_ablation.py (5 sub-exp) | ~15 min |
| c7 (P1.In exec) | 5 sub-exp × ~12min (CNN 학습 포함) | ~60 min |
| c8 (P1.M wrapper) | phase1_arch_ablation.py (7 sub-exp) | ~15 min |
| c9 (P1.M exec) | 7 sub-exp × ~14min | ~100 min |
| c10 (P1.F wrapper) | phase1_formula_ablation.py (4 sub-exp + F0 reuse) | ~15 min |
| c11 (P1.F exec) | 4 sub-exp × ~12min (per-sample MLP 학습 포함) | ~50 min |
| c12 (attribution) | phase1_attribution.md (4 axis 표 + best 선정) | ~20 min |
| c13 (P3 pairwise) | phase3 4 pair × ~50min (5-fold) | ~200 min |
| c14 (P4 triple) | phase4 1~2 sub-exp × ~60min | ~120 min |
| c15 (P5 iterative, 조건부) | phase5 iterative × ~70min | ~70 min |
| c16 (P6 augment, 조건부) | phase6 TTA + multi-parse × ~30min | ~30 min |
| c17 (synthesis) | results.md + next_plan_candidates.md + 3 파일 frontmatter sync + plan-011.1 instruction | ~30 min |
| **합계** | (조건부 G4/G5 포함) | **~12.7 hr** (조건부 skip 시 ~11 hr) |

### §N+1.2 시나리오 B — 4-way GPU multi-stream (★ 권장, Phase 1 + Phase 3 병렬)

| Phase | 순차 | 4-way 병렬 | 단축 |
|---|---|---|---|
| Phase 0 (preflight) | 15 min | 15 min (CPU only) | — |
| Phase 1.L (8 sub-exp) | 80 min | **~25 min** | 4-way × overhead 1.25 |
| Phase 1.In (5 sub-exp) | 60 min | **~20 min** | |
| Phase 1.M (7 sub-exp) | 100 min | **~32 min** | |
| Phase 1.F (4 sub-exp) | 50 min | **~17 min** | |
| Phase 2 (attribution) | 20 min | 20 min (분석) | — |
| Phase 3 (4 pair × 5-fold) | 200 min | **~50 min** | 4 pair 동시 + per-pair 순차 5-fold |
| Phase 4 (1~2 stack × 5-fold) | 120 min | **~60 min** | 5-fold 병렬 |
| Phase 5 (조건부) | 70 min | **~35 min** | 5-fold 병렬 |
| Phase 6 (조건부) | 30 min | 30 min | 추론만 |
| Phase 7 (synthesis) | 30 min | 30 min | — |
| 코드 작성 (c1/c3/c4/c6/c8/c10) | 105 min | 105 min | — |
| **합계** | ~12.7 hr | **~7.2 hr** (조건부 포함) | **~5.5 hr 단축** |

### §N+1.3 시나리오 C — sub-exp 병렬 + 5-fold 병렬 (max parallelization)

| Phase | 시나리오 B | 시나리오 C (+ 5-fold 병렬) |
|---|---|---|
| Phase 1 (24 sub-exp) | ~94 min | ~94 min (1-fold approx, 변경 없음) |
| Phase 3 (4 pair × 5-fold) | 50 min | **~20 min** (4 pair 동시 + 각 pair 5-fold 동시) |
| Phase 4 | 60 min | **~15 min** (5-fold 동시) |
| Phase 5 | 35 min | **~12 min** (5-fold 동시) |
| Phase 6/7 + 코드 | 165 min | 165 min |
| **합계** | ~7.2 hr | **~5.0 hr** (조건부 포함) |

### §N+1.4 권장

- **시나리오 B 채택** — 안정성 (multi-stream) + 단축 (~5.5 hr) 균형
- 시나리오 C 진입 조건: 시나리오 B 의 Phase 3 OOM 없이 안정 작동 확인 후
- decision-note 박제: `decision-note: parallel-execution — scenario B (4-way GPU multi-stream + CPU 24-worker data prep) 채택, wall-time ~12.7h → ~7.2h`

---

## §N+2. results.md 필수 항목

§14.2 참조 (10 section).

---

## §N+3. 통계 함정 & caveats

1. **1-fold approx 의 informational 한계**: Phase 1 의 fold=0 1-fold approx 는 N_val ≈ 2020 (binomial std ≤0.005). ΔOOF +0.003 이상은 신뢰 가능, ±0.003 은 noise floor. 4 axis best lever 식별은 *informational* (5-fold confirm 은 Phase 3+ 에서).

2. **plan-009 H002 의 1-fold over-fit 교훈**: plan-009 sub-exp b 가 fold=0 +0.0010 OOF gain 하지만 LB 에서 -0.0064 regression — *fold-specific artifact*. plan-011 의 *Phase 1 best lever 식별* 도 동일 risk → Phase 3+ 5-fold confirm 강제.

3. **C008 gate 의 학습 안정성**: sigmoid output 이 모든 sample 에서 collapse (< 0.05 또는 > 0.95) 가능. bias init +2.0 (sigmoid(2)=0.88) 로 *시작 시 ON* 유지 → asymmetric learning 안전. `gate_collapse` severe 시 옵션 a/b 자동 retry.

4. **C010 binormal weight 0.1 의 정당화**: plan-005 측정 binormal 0.0064 / parallel 0.0451 = 0.142. 보수적 round → 0.1. plan-011.1 에서 grid search (0.05 / 0.1 / 0.2) 가능.

5. **frozen GRU encoder 의 task mismatch risk**: plan-004 GRU 는 27-후보 ranking 학습. 단일공식 + corrector 의 task feature 와 다를 가능성 — P1.IC (frozen) vs P1.ID (CNN learnable) 비교가 *feature relevance 검증*.

6. **multi-parse (In-F) 의 학습 비용**: SG/EMA smoothing 은 *deterministic* — 학습 시 random parse 선택 augmentation 또는 3 parse 평균 inference. spec 은 후자 (학습 cost ~동일, 추론 cost ×3).

7. **iterative refinement 의 발산 risk (Phase 5)**: per_step_cap 3mm × 3 step = 9mm 누적. 매 step 방향 재학습 → noise 누적. step_idx embedding + parameter 공유 + huber loss 세 안정장치. `iterative_divergence` 시 옵션 a (step ↑ cap ↓) 자동 retry.

8. **per-sample MLP coeff (F3) 의 5-fold strict**: plan-007 Step 4 의 OOF 0.6482 는 5-fold concat 측정. plan-011 Phase 1.F3 의 1-fold approx 결과는 plan-007 5-fold 와 비교 → drift 검증.

9. **learnable single candidate (F4) 의 mode collapse**: Idea 2 의 soft-min loss 가 *단 1 개 후보* 학습 시 mode collapse 위험 X (1 개 의미 무관). 단 plan-006 anchor (F0) 보다 안 나오면 *learnable 이 직관 + grid search 보다 안 좋음* 신호 — informational.

10. **physics conservation (L5) 의 typical_jerk_step**: train data 의 99-quantile jerk delta 계산. plan-011 preflight 에서 자동 측정 (없으면 0.004 default).

11. **bell-shape weight (L6) 의 σ tuning**: σ=0.005 default (R_HIT 0.01 의 절반). σ 너무 작으면 boundary 외 sample 학습 신호 zero, σ 너무 크면 binary band 와 유사. plan-011.1 grid search (0.003 / 0.005 / 0.008) 가능.

12. **TTA rotation (P6.1) 의 Z축 보존**: XY 평면 회전만 — Z축 (중력 방향) 건드리지 X. 4 rotation (0°, 90°, 180°, 270°) 충분, 더 dense rotation 은 비용 ↑ 신호 ↓.

13. **GMM head (M5) 의 inference**: μ 만 사용 (expected) vs μ + N samples 평균. spec 은 전자 (단순). σ 는 uncertainty 가시화용 (학습 stability).

14. **bin classification (M3) 의 factorize**: per-axis 60 bin 독립 (3 × 60 head) vs joint 60³ = 216K bin. 후자 폭주 → 전자 채택. trade-off: per-axis correlation 손실 ↔ parameter cost 1/3600.

15. **LB 제출 0 회**: 할당량 소진 인계. plan-011.1 carry-over (plan-008.1 + plan-009.1 + plan-010.1 묶음과 동일 정책). 모든 sub-exp submission.csv 는 *생성·박제만*.

16. **plan-006 reproduce 의 raw vs corrected**: plan-006 의 0.6491 = corrected, plan-007 per_candidate_hit 의 0.6320 = raw. G0 reproduce 는 *둘 다* 측정.

17. **(★ caveat for Phase 1) single-axis fix 의 cross-axis bleed**: Phase 1.L 의 모든 sub-exp 가 fixed In-A + M0 + F0 위 측정. 만약 In̂ ≠ IA 또는 M̂ ≠ M0 라면, *true L̂* 가 anchor 와 다를 가능성 (예: 다른 input 위에서 다른 loss 가 best). Phase 3 의 pairwise 가 *partial* 검증 — Phase 4 triple 이 full 검증.

18. **(★ caveat for Phase 3+) F0 anchor maintain 정책**: P3.3 (L̂ + F̂) 만 F̂ swap, 나머지 P3.1/3.2/3.4 는 F0 anchor 유지. P4.1 도 F0, P4.2 만 F̂. *formula axis 의 stack 비용* 보수적.

19. **(★ caveat for Phase 6) augment 의 fold dependence**: TTA/multi-parse 는 *추론 시* augment — 학습 fold split 와 무관. 5-fold OOF 측정 가능 (Phase 4 best 위 추론만 augment).

20. **(★ caveat for §15 parallel) multi-stream reproducibility**: CUDA stream 동시 실행 시 *non-determinism* (kernel order dependence). 강제 정책: `torch.use_deterministic_algorithms(True, warn_only=True)` + `torch.backends.cudnn.deterministic = True` + 매 sub-exp seed 박제. 병렬 OOF 와 순차 reproduce OOF 의 |Δ| > 0.005 시 `parallel_reproducibility_drift` warn → deterministic 강제 retry.

21. **(★ caveat for §15 parallel) 5-fold 병렬 의 GPU memory budget**: ProcessPoolExecutor(max_workers=5) + `torch.cuda.set_per_process_memory_fraction(0.18)` 로 fold 별 GPU memory 18% 할당. TinyCorrectionNet 크기 (~50K params, ~30MB) 라면 5-fold 동시 ~150MB GPU memory 사용 — 24GB GPU 기준 < 1% utilization. CNN encoder (In-D) 포함 시 ~200MB 도 안전. OOM 시 max_workers=2 fallback (decision-note 박제).

22. **(★ caveat for §15 parallel) sub-exp 병렬 의 anchor invariance**: Phase 1 의 4-way multi-stream 학습 시 *4 sub-exp 가 동일 anchor 데이터* (preprocessed cf, train_y) 사용 → shared memory (mmap 또는 PyTorch DataLoader 의 num_workers=0 + persistent 데이터). 데이터 race 없음 (read-only). 단 각 sub-exp 의 *학습된 model state* 는 GPU memory 에 분리.

---

## §N+4. 변경 이력

- v1 (2026-05-13): 초안 — plan-010 의 depth (defect fix) 와 *상호 보완 breadth* (4 axis × ~25 single-axis ablation). notes/코드공유-upgrade.md 의 C008/C009/C010/D001 후보 + notes/prior-ideas.md 의 Physics conservation + Multi-parse + notes/mosquito-trajectory-ideas.md 의 TTA + GMM 통합. Phase 0~7, G0~G_final 7 gate, commit chain c1~c17 + G4/G5 조건부. LB 제출 0 회 (plan-010.1 carry-over 패턴). §15 병렬 실행 정책 신설 (server CPU 48 core + GPU 1 device:0 기준, Phase 1 sub-exp 4-way GPU multi-stream + Phase 3+ 5-fold ProcessPoolExecutor 분할, wall-time ~12.7h → ~7.2h 시나리오 B 권장). caveat #20~#22 (parallel reproducibility / GPU memory / anchor invariance) 추가.

---

## §N+5. 참조

- `plans/plan-004-pb-0-6822-fullrun.md` (corrector arch baseline, GRU encoder source)
- `plans/plan-005-pb-0-6822-diagnostic.md` (corrector_decomp band table, direction breakdown)
- `plans/plan-006-minimal-variant-e-lb.md` (단일공식 baseline + LB 0.6692)
- `plans/plan-007-formula-tuning.md` (per_candidate_hit + Step 2/3/4 carry-over)
- `plans/plan-008-candidate-redefine-corrector-redesign.md` (27 후보 selector + corrector lock-in)
- `plans/plan-009-selector-ranking-loss.md` (corrector 강화 5 sub-exp, LB regression 교훈)
- `plans/plan-010-corrector-redesign-exploration.md` (depth fix, corrector_redesign.py module anchor)
- `notes/PB_0.6822 코드공유.ipynb` (cell 6 boundary corrector 원본 + 부록 §A "한 원칙의 세 면")
- `notes/코드공유-upgrade.md` (★ C008 do-no-harm gate / C009 lite / C010 Frenet anisotropic / D001 oracle simulation + Idea 1 연속 heatmap + Idea 2 학습 후보)
- `notes/prior-ideas.md` (Physics conservation reg / Multi-parse input / Huber loss)
- `notes/mosquito-trajectory-ideas.md` (TTA rotation / GMM output / Residual prediction philosophy)
- `WORKFLOW.md` (§12 Autonomous Execution Protocol)
- `CLAUDE.md` (Autonomous Execution Policy + Push 의무)
