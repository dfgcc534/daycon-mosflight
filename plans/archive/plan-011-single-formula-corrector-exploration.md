---
plan_id: 011
version: 1
date: 2026-05-13 (Asia/Seoul)
status: partial (G1 complete + v1.1/v1.2 post-fix amendments, Phase 3+ autonomous skip)
best_phase: Phase 1 In axis IC (R001 frozen GRU, v1.2)
best_oof_fold0: 0.6446
best_submission: runs/baseline/H012_phase1-input-ablation/sub_IC/submission.csv
v1_1_amendment: F3/F4 formula parity fix (F4 post-fix = 0.6431, +0.0030 vs F0 — F axis 진정한 측정 회복)
v1_2_amendment: IC frozen GRU 활성화 (R001 fold0 reuse, 64-dim). IC OOF 0.6446, In̂ = IC (이전 ID 대체)
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
scope: 단일공식 + corrector path 의 *구체화 + 폭넓은 탐색*. plan-006 의 frenet_par120_perp_neg020 + plan-004 corrector LB 0.6692 baseline 위에 corrector 4 axis (Input × Loss × Arch × Formula) 폭넓은 ablation 으로 *진정한 ceiling* 측정 + lever attribution. Phase 0 diagnostics (재학습 0~minimal) → Phase 1 single-axis 24 sub-exp + F0 reuse (= 표기 25 의 unique 실행은 24; 1-fold approx, ★ 정보 핵심) → Phase 2 best-axis selection → Phase 3 pairwise 4 sub-exp + 3~4 solo prep (5-fold super-additive) → Phase 4 triple stack 1~2 sub-exp → 조건부 Phase 5 iterative / Phase 6 inference augment / Phase 7 synthesis. LB 제출 0 회 (plan-009.1 carry-over 패턴 답습) — plan-011.1 carry-over.
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
- **G1** (Phase 1 single-axis ablation, ★ 정보 핵심): 4 axis 모두 완료 — L axis 8 sub-exp + In axis 5 sub-exp + M axis 7 sub-exp + F axis 4 sub-exp (F0 reuse). 1-fold approx (fold=0). 각 axis 의 *best lever* 식별. (a) 모든 24 sub-exp informational 완료 (fail 없음 — attribution 목적). (b) 4 axis 중 *최소 2 axis* 에서 +0.005 marginal OOF gain — axis-level aggregation 함수 = `max(ΔOOF_i for sub_exp_i in axis where sub_exp_i ≠ anchor) ≥ 0.005` (= "axis 안 *어느 한 sub-exp 라도* anchor 대비 +0.005 이면 그 axis 는 positive"). single-formula + corrector path 의 *부정 방지*. 위반 시 `phase1_no_lever_positive` severe.
- **G2** (Phase 3 pairwise 5-fold): L̂ + In̂, L̂ + M̂, L̂ + F̂, In̂ + M̂ (4 pair). (a) `oof_soft_hit ≥ G1 best + 0.003` (super-additive 입증). (b) 4 pair 중 *최소 1 pair* additive 또는 super-additive (= 결합 OOF ≥ 두 단독 OOF 의 합 − base). 위반 시 `super_additive_fail` warn.
- **G3** (Phase 4 triple stack): L̂ + In̂ + M̂ (P4.1). (a) `oof_soft_hit ≥ G2 best + 0.003`. (b) (조건부) F̂ ΔOOF ≥ +0.005 시 P4.2 (L̂ + In̂ + M̂ + F̂) 추가. 위반 시 `triple_stack_marginal` warn.
- **G4** (Phase 5 iterative, **조건부**): G3 best OOF > 0.69 진입 조건. L̂ + In̂ + M̂ + Z3 iterative (3-step, per-step cap=3mm, parameter 공유). (a) `oof_soft_hit ≥ G3 + 0.005`. (b) `[1, 1.5cm) hit_after ≥ 0.20`. (c) iter_gap (train OOF − val OOF) ≤ 0.05. 위반 시 `iterative_divergence` severe.
- **G5** (Phase 6 inference augment, **조건부**): G3 또는 G4 best 위 TTA rotation 4 + multi-parse inference. (a) `oof_soft_hit ≥ G3 + 0.002` marginal. 위반 시 `augment_no_signal` warn-only (학습 X, 비용 ~free).
- **G_final**: synthesis + plan-012 후보 ≥ 3 + 3 파일 frontmatter 동시 박제 (`lb_score: TBD` carry-over) + best Phase submission 박제 + plan-011.1 carry-over instruction 박제.

### G-gates

- G0: Phase 0 diagnostics + preflight.json 생성 [DONE] (b6e582a — D001=0.6570 < 0.66, c008 path disabled, P1.L2 자동 skip)
- G1: Phase 1 4-axis ablation (24 sub-exp) — L NEG / In POSITIVE (ID +0.0050) / M NEG / F NEG → G1 (b) FAIL (1/4 positive) [DONE] (73a1446 — `phase1_no_lever_positive` warn, autonomous: P3.1 만 진행)
- G2: Phase 3 pairwise (4 pair 5-fold) — super-additive 입증 [SKIP] (autonomous per §9.3 option a — G1 (b) FAIL)
- G3: Phase 4 triple stack (P4.1 + 조건부 P4.2) — OOF ≥ G2 + 0.003 [SKIP] (autonomous — G2 skip)
- G4: Phase 5 iterative (조건부 G3 > 0.69) — [1,1.5cm) hit ≥ 0.20 + iter_gap ≤ 0.05 [SKIP] (autonomous — G3 skip + M4 iterative 자체 -0.0693)
- G5: Phase 6 inference augment (조건부) — marginal +0.002 [SKIP] (autonomous — base 없음)
- G_final: synthesis + plan-012 후보 ≥ 3 + 3 파일 frontmatter sync + best Phase submission 박제 + plan-011.1 instruction [DONE]

### Commit chain (next-up)

| # | type | spec section | status |
|---|---|---|---|
| c1 | docs | `plans/plan-011-single-formula-corrector-exploration.md` v1 작성 | [DONE] (7bfbf81 + c1 fix-up 8619091) |
| c2 | code | `analysis/plan-011/preflight.py` — D001 oracle simulation + plan-006 reproduce + corrector_decomp 재측정. spec @ §4 | [DONE] (b6e582a) |
| G0 | gate | `preflight.json` 생성 + D001 박제 + reproduce ✓ + decomp drift ✓ | [DONE] (b6e582a — D001=0.6570, reproduce drift 0.0, destr -79 vs -203 informational) |
| c3 | code | `src/pb_0_6822/corrector_redesign_v2.py` — 16 components self-contained (Z1 base + 6 arch + 7 loss fn + 3 input + 2 formula). plan-010 c3 미실행 → RedesignedCorrectionNet inline. spec @ §5.1 | [DONE] (ad4d344, smoke test ✓ 16/16) |
| c4 | code | `analysis/plan-011/phase1_loss_ablation.py` — P1.L0~L7 wrapper scaffold + L0 anchor 박제 (fold-0 OOF 0.6545, 재학습 X). spec @ §5.2 | [DONE] (5a7aa97 — L0 only; L1~L7 학습 wrapper = c5) |
| c5 | exp | Phase 1.L L1~L7 학습 (phase1_loss_train.py, L2 skip per G0). 6 sub-exp ~16s total (★ axis NEGATIVE — max ΔOOF=-0.0114, L̂=L3) | partial G1 [DONE] (5bd8353) |
| c6 | code | `analysis/plan-011/phase1_input_ablation.py` — P1.IA~IF wrapper + 학습 일괄 (IC skip per plan-004 GRU 부재). spec @ §6 | [DONE] (234b824) |
| c7 | exp | Phase 1.In 4 sub-exp 실행 (★ AXIS POSITIVE, In̂=ID at +0.0050). ~10s | partial G1 [DONE] (234b824 — c6/c7 통합) |
| c8 | code | `analysis/plan-011/phase1_arch_ablation.py` — P1.M0~M6 wrapper. spec @ §7 | [DONE] (36e5a2c) |
| c9 | exp | Phase 1.M 7 sub-exp ~17s (★ axis NEGATIVE max ΔOOF=0, M̂=M1 tied) | partial G1 [DONE] (36e5a2c) |
| c10 | code | `analysis/plan-011/phase1_formula_ablation.py` — P1.F0~F4 wrapper + 학습. spec @ §8 | [DONE] (73a1446) |
| c11 | exp | Phase 1.F 5 sub-exp (★ formula_swap_marginal, F̂=F0 fix). F3/F4 cand formula 식 selector parity 어긋남 — plan-011.1 carry-over 후보 1순위 | **G1** [DONE] (73a1446 — G1 (b) FAIL: 1 positive axis only) |
| c12 | analysis | `analysis/plan-011/phase1_attribution.md` — 4 axis ΔOOF 표 + best lever 식별. spec @ §9 | [DONE] (b842ce8) |
| c13 | code+exp | Phase 3 pairwise — autonomous SKIP per §9.3 option a (G1 (b) FAIL 0/4) + best Phase submission 박제 (In axis ID full-train). spec @ §10 | **G2** [SKIP] (df055d9 — autonomous decision-note) |
| c14 | code+exp | Phase 4 triple — autonomous SKIP (G2 skip). spec @ §11 | **G3** [SKIP] |
| c15 | code+exp | Phase 5 iterative — autonomous SKIP (G3 skip + M4 iterative -0.0693). spec @ §12 | G4 [SKIP] |
| c16 | code+exp | Phase 6 augment — autonomous SKIP (no base). spec @ §13 | G5 [SKIP] |
| c17 | analysis | `analysis/plan-011/results.md` + `next_plan_candidates.md` (≥ 3 후보) + 3 파일 frontmatter sync + best Phase submission 박제 + plan-011.1 carry-over instruction. spec @ §14 | **G_final** [DONE] |
| c17.1 | sync | §0.5 [TODO]→[DONE] | [DONE] |

### Plan-specific severe (WORKFLOW.md §12.3 default 위 추가분)

- `preflight_artifact_missing` — G0 의 `preflight.json` 미생성 또는 plan-006 reproduce 실패 (|measured − 0.6320| > 0.005)
- `phase1_no_lever_positive` — G1 의 4 axis 중 *어느 axis 도* +0.005 marginal 없음 (= single-formula + corrector path 자체 부정). severity=**severe** 이지만 §9.3 의 autonomous recovery 옵션 (a) Phase 3 skip, G_final 직접 진입 또는 (b) Phase 5 iterative 단독 진입 으로 *halt 아닌 path-pivot* — 사용자 escalate 불필요.
- `iterative_divergence` — G4 의 iter_gap > 0.05 또는 [1,1.5cm) < 0.10
- `single_formula_residue` — selector 가 단일공식 외 다른 candidate 사용한 evidence (cand pool size > 1 또는 score variance > 1e-10 in non-F4 sub-exp)
- `frozen_gru_drift` — In-C frozen plan-004 GRU encoder parameter 변경 detected (state_dict diff > 0)
- `gate_collapse` — P1.L2 (C008 gate-asymmetric loss) **또는 P1.M1 (GateHeadCorrector arch)** 의 gate output 이 모든 sample 에서 < 0.05 또는 > 0.95 (gate 학습 실패). retry 옵션 (bias init bump +2.0 → +3.0; λ_destructive 8 → 4) 은 두 sub-exp 모두 적용 가능.
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
- corrector_oracle_gain = `corrected_hit − raw_hit` (K=1 단일공식 환경에서 "oracle" 은 *유일 candidate* 와 동일 — multi-candidate context 의 best-of-K 개념 아님. 표기상의 잔재이며 actual 계산은 "보정 전 hit − 보정 후 hit" 의 부호 반대 = `corrected − raw`)
- ΔOOF (lever attribution) = `OOF_with_lever − OOF_anchor` per sub-exp

### §3.4 Anchor 정의 (Phase 1 의 모든 ablation 의 기준점)

- L0 (Loss anchor): plan-004 default (MSE on cap-truncated + far_weight 0.04 + easy_weight 0.20 + env_head + apply_scale 0.75 + hard-coded band)
- In-A (Input anchor): `cf` 32-dim snapshot only (plan-004 default)
- M0 (Arch anchor): TinyCorrectionNet (depth=2, hidden=64, plan-004 default)
- F0 (Formula anchor): frenet_par120_perp_neg020 (CANDIDATES[17])

**Anchor combo (= P1.L0 = P1.IA = P1.M0 = P1.F0)** = plan-006 baseline reproduce (corrected OOF 0.6491 ± 0.005).

> *Anchor 두 값의 구분 박제* (§4.1 의 `anchor_oof_5fold: 0.6524` 와 `oof_argmax_hit_corrected_expected: 0.6491` 의 관계):
> - **0.6491** = plan-006 §5.5 의 `oof_argmax_hit_corrected` (= 본 plan G0 reproduce *pass* 기준). drift threshold ±0.005 의 base.
> - **0.6524** = plan-005 5-fold soft hit oof (corrected, *argmax 없이* search_temperature top-1) — `corrector_oracle_gain` 의 `delta` 계산 시 anchor (= D001 oracle simulation 의 base). pass/fail 기준 아님.
> - 두 값은 *다른 metric* 이므로 일치 강제 안 함. drift_threshold (±0.005) 는 `0.6491 base` 의 `oof_argmax_hit_corrected_measured` 에만 적용.

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
  --plan-005-corrected-oof analysis/plan-005/corrector_decomp_oof_corrected.npz \
  --plan-005-raw-scores    analysis/plan-005/corrector_decomp_oof_raw_scores.npz \
  --plan-006-checkpoint    runs/baseline/F001_variant-e/checkpoint_best.pt \
  --plan-006-config        configs/plan-006-variant-e.yaml \
  --out                    analysis/plan-011/preflight.json
```

> *plan-005 npz 경로 확정* (자율 결정 — 산출물명 plan-005 §6.3 의 default name 따름; 부재 시 동일 dir 내 매칭으로 fallback).
> *plan-006 checkpoint* = plan-006 §5.5 의 Variant E (frenet_par120_perp_neg020) best checkpoint.

#### §4.2.1 D001 perfect_gate_oof_5fold 알고리즘 (self-contained spec)

```
INPUT (npz file → numpy array 추출):
  corrected_oof_pred: (N, 3) ← `--plan-005-corrected-oof` npz 의 `pred_pos` key
                                (= plan-005 corrector_decomp 의 5-fold OOF corrected predicted pos).
  raw_oof_pred:       (N, 3) ← `--plan-005-raw-scores` npz 의 `pred_pos` key
                                (= "raw_scores" 라는 파일명 잔재; 실제 저장 내용은 raw predicted pos.
                                 npz key 가 다르면 `argmax_pos` 또는 `top1_pos` 로 fallback 검색).
  truth_pos:          (N, 3) ← data/train.csv 의 ground truth final position (sample_id matching).
  R_HIT = 0.01 (m)
PROCEDURE:
  err_raw       = ‖raw_oof_pred       − truth_pos‖₂                # (N,)
  err_corrected = ‖corrected_oof_pred − truth_pos‖₂                # (N,)
  raw_hit       = err_raw       ≤ R_HIT                           # (N,) bool
  corrected_hit = err_corrected ≤ R_HIT                           # (N,) bool
  destructive   = raw_hit ∧ ¬corrected_hit                         # (N,) bool — "보정 안 했으면 hit 였는데 보정해서 miss"
  # perfect-gate simulation: 위 destructive sample 만 raw 로 되돌림 (= gate 가 destructive 를 완벽히 식별 + skip)
  pred_perfect  = where(destructive, raw_oof_pred, corrected_oof_pred)   # (N, 3)
  err_perfect   = ‖pred_perfect − truth_pos‖₂                       # (N,)
  perfect_gate_oof_5fold = mean(err_perfect ≤ R_HIT)                # scalar in [0, 1]
  # tie 케이스 (err == R_HIT 정확히): ≤ 으로 hit (관행).
  n_destructive_samples = sum(destructive)
```

`perfect_gate_oof_5fold` = preflight.json 의 D001 entry 의 `perfect_gate_oof_5fold` field.
`delta = perfect_gate_oof_5fold − anchor_oof_5fold` (= "destructive samples 완벽 식별 시 회복 가능 ceiling").
`go_no_go_threshold = 0.66` — `perfect_gate_oof_5fold ≥ 0.66` 시 `c008_path_enabled = true`.

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

    pred: (B, 3) — corrector raw delta output (BEFORE gate, BEFORE cap). loss 미분 path.
    target: (B, 3) — uncapped residual (= true_pos − cand_pos).
    raw_hit_mask: (B,) bool — True if raw candidate (before delta) already hit (err ≤ R_HIT=0.01).
    corrected_pos: (B, 3) — caller pre-computes as `cand + cap6mm(gate * pred)`
        - gate: (B, 1) ∈ [0,1] (L2 only — sigmoid of GateHead; for L3 gate ≡ 1)
        - cap6mm(x) = x * min(1, 0.006 / ‖x‖₂)  (plan-010 Z1 cap @ 6mm; outside this fn — caller responsibility)
        corrected_pos 는 미분 path 아님 (destructive mask 판정용 detached input).
    lambda_destructive: float — destructive sample 의 **총** 페널티 배수 (= replacement, NOT additive).
        destructive 케이스 loss = lambda_destructive × base_loss (default 8.0 → ×8).
        normal 케이스 loss     = base_loss.
        gate_collapse 시 retry: lambda_destructive 4.0 (= ×4).
    """
    base_loss = huber_loss(pred, target)  # (B,)
    err_after = torch.norm(corrected_pos - target, dim=1)
    corrected_miss = (err_after > 0.01)  # R_HIT
    destructive = raw_hit_mask & corrected_miss  # (B,) bool
    # ★ replacement (multiplicative substitution), NOT additive — destructive 시 base_loss × lambda 로 *대체*.
    return torch.where(destructive, base_loss * lambda_destructive, base_loss)


def frenet_anisotropic_loss(pred_local, target_local, w_par=1.0, w_perp=1.0, w_bi=0.1):
    """L4 (C010): Frenet local-frame anisotropic. pred_local/target_local: (B, 3) in (t, n, b).

    caller pre-converts world→Frenet via build_frenet_basis() (Frenet basis @ end_idx, self-contained spec below).
    decision-note: spec-default — w_bi=0.1 (plan-005 binormal 0.0064 / parallel 0.0451 ≈ 1/7).

    L4 적용 정책 (★ Z1 huber 와의 결합):
      - 본 fn 은 *replacement* 가 아닌 *additive auxiliary loss* 로 사용.
      - L4 sub-exp 총 loss = `huber_loss(pred, target) + lambda_aniso * frenet_anisotropic_loss(pred_local, target_local)`
      - lambda_aniso default = 1.0 (huber 와 동등 weight; spec-default).
      - 이유: Z1 의 huber 는 *world frame* L1-like robust loss → 기본 학습 신호. C010 의 anisotropic 은
        *Frenet frame* 의 binormal capacity 축소 (w_bi=0.1) 를 통한 *추가 regularization*. 둘은 보완적.
    """
    diff = pred_local - target_local
    return w_par * diff[:, 0]**2 + w_perp * diff[:, 1]**2 + w_bi * diff[:, 2]**2


def build_frenet_basis(trajectory_x, end_idx):
    """Self-contained Frenet basis spec (caller helper for L4).

    trajectory_x: (B, T, 3) world coords. end_idx: (B,) int — last observation index.
    Returns (R_world_to_local: (B, 3, 3)) — rows = (t̂, n̂, b̂).
      - velocity v = trajectory_x[:, end_idx] − trajectory_x[:, end_idx − 1]
      - accel    a = v − (trajectory_x[:, end_idx − 1] − trajectory_x[:, end_idx − 2])
      - t̂ = v / ‖v‖
      - n̂ = (a − (a·t̂) t̂) / ‖·‖   (perpendicular component of accel)
      - b̂ = t̂ × n̂
    Degenerate ‖v‖ < 1e-6 or ‖n‖ < 1e-6 → fallback identity basis (caller skips L4 contribution).
    pred_local = R_world_to_local @ pred_world (matrix-vector; pred_world = corrector delta in world frame).
    """
    ...


def physics_conservation_loss(delta, recent_acc, typical_jerk_step=0.004):
    """L5: CPhy-ML — kinematically implausible delta 에 페널티.

    Units & shapes (모든 양 = step-domain, meter per step^n; horizon h=1 implicit):
      delta:       (B, 3) — corrector output per-step displacement (m/step). caller 가 raw delta (m at horizon=2)
                            를 step-domain 으로 변환: `delta_step = delta / horizon` 후 본 fn 호출.
      recent_acc:  (B, 3) — last-step inter-frame acceleration (m per step²):
                            recent_acc = (x_T − x_{T-1}) − (x_{T-1} − x_{T-2})  computed by caller.
      typical_jerk_step: float (m per step³) — plan-005 median |jerk_step| ≈ 0.004.
    Penalty (★ self-consistent 단일 식):
      jerk_per_step = delta_step − recent_acc                       # (B, 3); 단위 m/step³ (= step-domain jerk proxy)
      norm          = ‖jerk_per_step‖₂                              # (B,);    m/step³
      excess        = max(0, norm − typical_jerk_step)               # (B,);    m/step³
      penalty       = excess²                                         # (B,);    m²/step⁶
    Returns: (B,) — per-sample penalty, caller weights with λ=0.5 in L5.

    Note: recent_jerk 인자 폐기 (이전 spec 의 dead argument). jerk 자체는 본 식의 `delta - recent_acc` 로
    자연스럽게 *implied* 됨 (= "delta 의 변화 = next acc 추정, jerk = next acc − prev acc").
    """
    delta_jerk_norm = torch.norm(delta - recent_acc, dim=1)
    penalty = torch.clamp(delta_jerk_norm - typical_jerk_step, min=0.0) ** 2
    return penalty


def bell_shape_weight(err, R_HIT=0.01, sigma=0.005):
    """L6: Gaussian-shaped weight centered at R_HIT. (B,) → (B,).

    `err` 정의 = `‖cand + cap6mm(delta) − target‖₂` (= corrected_pos 의 err, plan-010 z1 cap 적용 후).
    Caller computes err and passes (B,) tensor; this fn returns weight (B,) ∈ (0, 1].

    P1.L6 적용 식 (★ wrapper-level total loss spec):
        corrected_pos = cand + cap6mm(delta)              # caller
        err           = ‖corrected_pos − target‖₂          # caller (B,)
        w             = bell_shape_weight(err, σ=0.005)    # (B,)
        per_sample    = huber_loss(pred, target)           # (B,) m² units (per-sample sum-of-squares)
        total_loss    = (w * per_sample).mean()            # scalar — err 가 R_HIT 근방 sample 우선 학습.
    σ=0.005m 의 의미: err = R_HIT 일 때 w=1.0; err = R_HIT ± 5mm 일 때 w ≈ e⁻¹ ≈ 0.37.
    far miss (err >> R_HIT) → w ≈ 0 → 영향 적음 (= "near-hit 집중 학습" 의도).
    """
    return torch.exp(-((err - R_HIT) / sigma) ** 2)


def hit_aware_hinge(corrected_pos, target, R_HIT=0.01, smooth=0.005):
    """L7: smooth hinge — squared smoothed-hinge (m² units, dimensional-compatible with huber).

    spec-default 결정: *squared* form (docstring 의 "max(0, err-R_HIT)²" 박제).
    smooth approx: `(softplus(excess/smooth) * smooth)²` — softplus 로 양의 hinge linearization
    후 squared 적용 → smooth 한 quadratic hinge. units = m² (huber 의 sum-of-squares 와 동일 차원).
    excess < 0 (= 이미 hit) → softplus 가 ≈ 0 → loss 거의 0.
    excess > 0 (= miss) → ≈ excess² (linear hinge 의 squared 와 점근적 일치).

    corrected_pos: (B, 3), target: (B, 3). 미분 가능.
    """
    err = torch.norm(corrected_pos - target, dim=1)
    excess = err - R_HIT
    linear_hinge = F.softplus(excess / smooth) * smooth  # smooth approx of max(0, x), units = m
    return linear_hinge ** 2                              # squared → units = m² (huber 와 동차)
```

### §5.2 Phase 1.L wrapper (`analysis/plan-011/phase1_loss_ablation.py`)

8 sub-exp 일괄 실행 (fixed In-A + M0 + F0, fold=0, ~10min/sub-exp):

| sub-exp | loss config | wrapper-level total loss 식 |
|---|---|---|
| P1.L0 (anchor) | plan-004 default | `mse_cap_truncated + far*0.04 + easy*0.20 + env*0.05` (plan-004 그대로) |
| P1.L1 | Z1 minimum: uncapped target + huber(β=0.005) + far=0.5 + easy=0 + env_loss_weight=0 + apply_scale=1 + boundary [0.7, 1.7cm] | `huber_loss(pred, target).mean()` |
| **P1.L2** (★ 조건부 D001 ≥ 0.66) | Z1 + C008 gate (sigmoid head, bias init +2.0) + asymmetric loss (λ=8). corrector arch = **GateHeadCorrector (M1) + `aux["raw_delta"]` 노출 변형** (아래 ★ 박제) | `asymmetric_loss(aux["raw_delta"], target, raw_hit_mask, corrected_pos=cand+cap6mm(delta), λ=8.0).mean()` |
| P1.L3 | Z1 + C009 (asymmetric loss only, gate 없이) | `asymmetric_loss(pred, target, raw_hit_mask, corrected_pos, λ=8.0).mean()` (gate ≡ 1) |
| **P1.L4** | Z1 + C010 Frenet anisotropic (w_par=1, w_perp=1, w_bi=0.1) | `huber_loss(pred, target).mean() + 1.0 * frenet_anisotropic_loss(pred_local, target_local).mean()` (λ_aniso=1.0, additive) |
| P1.L5 | Z1 + L5 physics conservation (jerk penalty λ=0.5) | `huber_loss(pred, target).mean() + 0.5 * physics_conservation_loss(pred / horizon, recent_acc).mean()` (caller scales delta→step domain) |
| P1.L6 | Z1 + L6 bell-shape weight (σ=0.005) | `(bell_shape_weight(err) * huber_loss(pred, target)).mean()` where `err = ‖cand + cap6mm(pred) − target‖₂` |
| P1.L7 | Z1 + L7 hit-aware smooth hinge | `0.5 * huber_loss(pred, target).mean() + 0.5 * hit_aware_hinge(corrected_pos, target).mean()` (둘 다 m² units, equal weight) |

### §5.3 산출 (per sub-exp)

- `runs/baseline/H011_phase1-loss-ablation/sub_L{N}/`
  - `boundary_val_predictions.npz` (fold 0 val, K=1)
  - `report_sub_L{N}.json` (oof_soft_hit, per-band hit_after, corrector_oracle_gain, gate_stats if L2, elapsed)
- `analysis/plan-011/phase1_loss_summary.json` (8 sub-exp 통합)

### §5.4 G1.L 합격

- 8 sub-exp 모두 informational 완료 (fail 없음 — attribution 목적)
- 최소 1 sub-exp 가 P1.L0 anchor 대비 +0.005 marginal OOF
- best L̂ 식별 (max ΔOOF)

> *P1.L2 단일-axis 박제* (Loss + Arch joint 의 의도적 묶음):
> P1.L2 는 *형식상* L axis sub-exp 이지만 GateHeadCorrector (M1 arch) 를 사용하므로 Loss + Arch *joint* change.
> 이는 §1.2 narrative ("C008 = gate head + asymmetric loss 의 *단일 motivation cluster*") 의 의도적 묶음.
> *attribution-clean* 의 관점에서는 *L axis 안의 component 가 gate-loss 묶음* 으로 한 단위; M axis 의 M1 (gate only,
> no asymmetric loss) 와 비교해서 *gate-only contribution* (M1) vs *gate+asymmetric contribution* (P1.L2) 분리 가능.
> Phase 3 P3.2 (L̂=L2 + M̂=M1) 시 mechanism overlap 가 명시적 risk → §10.5 `mechanism_overlap_flag` 박제 + decoupling
> 변형 sub-exp 옵션 (예: L̂=L3 (asymmetric only, no gate) + M̂=M1 으로 mechanism 분리) 사용 가능.
>
> *Z1 base bleed caveat* (P1.L0 anchor vs P1.L1 base vs P1.L2~L7 builds):
> L1 = "Z1 minimum" 자체가 anchor L0 대비 6 결함 (B1+A2+C1+C2+D1+E1) fix → P1.L0 vs P1.L1 ΔOOF 가 *Z1 contribution* 의 직접 측정.
> L2~L7 은 L1 위에 single component (gate / asymmetric / Frenet / physics / bell / hinge) 추가 → ΔOOF(L_i vs L0) 는
> *Z1 + component* 의 합산 효과. *component-only* contribution = `ΔOOF(L_i vs L1)` 으로 별도 보고.
> `phase1_loss_summary.json` 에 `delta_vs_anchor` (L0 base) + `delta_vs_z1` (L1 base) 두 컬럼 박제. attribution §2 표
> (analysis/plan-011/phase1_attribution.md) 도 둘 다 표시.

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
    """In-B: hand-crafted trajectory statistics (no learning, 20-dim).

    Self-contained spec (20-dim breakdown, all 단위 = step-domain, ε=1e-6 for div-by-zero):
      Let v_t = x_{t+1} − x_t (T−1 vectors), a_t = v_{t+1} − v_t (T−2 vectors),
          j_t = a_{t+1} − a_t (T−3 vectors), s_t = ‖v_t‖ (T−1 scalars).
      Frenet basis per t: t̂_t = v_t / max(‖v_t‖, ε); n̂_t = (a_t − (a_t·t̂_t) t̂_t) / max(‖·‖, ε).
      a_par_t  = a_t · t̂_t   (parallel component)
      a_perp_t = a_t · n̂_t   (perpendicular component)
      cos_t    = (v_t · v_{t+1}) / max(‖v_t‖ ‖v_{t+1}‖, ε)
      κ_t      = ‖a_perp_t‖ / max(‖v_t‖², ε)
    20 dim (index [0~19]):
      [0~3]   speed:        mean(s), std(s), s_last (= s_{T-2}), max(s)                         (4)
      [4~6]   acc_norm/v:   mean(‖a‖/s), std(‖a‖/s), max(‖a‖/s)                                  (3)
      [7~8]   a_par/v:      mean(a_par/s), std(a_par/s)                                         (2)
      [9~11]  a_perp/v:     mean(a_perp/s), std(a_perp/s), max(a_perp/s)                        (3)
      [12~14] jerk:         mean(‖j‖), std(‖j‖), max(‖j‖)                                       (3)
      [15~17] turn_cos:     mean(cos), std(cos), cos_last (= cos_{T-3})                          (3)
      [18~19] curvature:    mean(κ), max(κ)                                                      (2)
    Returns: (N, 20) float32 tensor.

    Caller wire-in: 본 모듈은 stateless (no learnable param). caller 는 `module(traj)` 형태로 호출.
    표준 nn.Module 패턴 — `def forward` 가 `compute` 위임:
    """
    def forward(self, trajectory_x):
        return self.compute(trajectory_x)

    def compute(self, trajectory_x):
        # trajectory_x: [N, T, 3] world coords. 위 spec 그대로 계산.
        # (구현 디테일 = inline pseudo-code per docstring; numpy/torch 양쪽 가능.)
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
    """In-F: raw + Savitzky-Golay smoothing + EMA smoothing.

    Single source-of-truth 학습 정책 (★ §13.2 P6.2 + §N+3 caveat #6 와 *완전 동기*):
      - **학습 시**: epoch 매 batch 마다 3 parse 중 *random 1 개* 선택 (augmentation).
        cf = make_candidate_features(parse_k(trajectory_x, end_idx))  where k ~ Uniform{raw, SG, EMA}
        매 step batch 단위 random — sample 단위 아님 (batch 일관성).
      - **추론 시**: 3 parse 모두 forward → cf 단위로 평균 (deterministic ensemble).
        cf = mean([make_cf(raw), make_cf(SG), make_cf(EMA)])
      - parameter: window=5, order=2 (SG), alpha=0.6 (EMA) — §13.2 P6.2 와 동일.
    """
    def parse(self, trajectory_x, end_idx, mode="train"):
        # mode ∈ {"train", "inference"} — 위 docstring 의 정책에 따라 분기.
        # train: random 1 parse augment
        # inference: 3 parse cf 평균
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

> *IE 결번 박제*: §1.6 의 `plan-010 Z6 e2e (G4 조건부)` 가 plan-011 의 `P1.IE` 위치에 매핑되지만 *본 plan 의 scope X*.
> 이유: Z6 e2e 는 GRU encoder + corrector 전체 *non-frozen joint 재학습* 으로 plan-010 (depth) path 의 산출이며, plan-011 (breadth)
> 의 4-axis ablation 의 *single-axis isolation 의도* 와 충돌 (Input axis 만 변경이 아닌 selector 까지 함께 변경). IE 는 plan-012 후보로 carry-over.

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

> *Unified forward return contract* (모든 M0~M6 + L 축 사용 corrector class 통일):
> ```
> forward(cf, encoder_emb=None) -> tuple[delta: Tensor (B, 3), aux: dict]
> ```
> - `delta`: 모든 sub-exp 의 *최종 corrector output* (world frame, m). caller 는 `cand + cap6mm(delta)` 로 corrected_pos 구성.
> - `aux: dict` (sub-exp 별 *부산물* 박제 — 비어 있어도 OK):
>   - `gate: (B, 1) ∈ [0, 1]`     (M1, L2 에서만; 다른 곳 None)
>   - `logsigma: (B, 3)`           (M5 GMM 에서만; NLL 계산용)
>   - `bin_probs: list[(B, K)]`    (M3 BinClassifier 에서만; 진단용)
>   - `direction, magnitude`       (M2 SplitHead 에서만; 진단용)
> - trainer (analysis/plan-011/phase1_*.py) 는 `aux` 의 key 존재 여부로 loss path 분기 (M5: `gmm_nll_loss(aux['logsigma'], delta, target)`; L2/M1: `aux['gate']` 통계 기록; otherwise: huber/asymmetric 등).
> - M0 (anchor) 도 `return delta, {}` 로 contract 통일 — 분기 단순화.

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
        delta = gate * raw_delta
        # ★ raw_delta 노출 — P1.L2 의 asymmetric_loss(pred=raw_delta) 호출용 (gate double-application 회피).
        return delta, {"gate": gate, "raw_delta": raw_delta}


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
        delta = direction * magnitude  # (B, 3)
        return delta, {"direction": direction, "magnitude": magnitude}


class BinClassifierCorrector(v1.RedesignedCorrectionNet):
    """M3: bin classification head — 3 × bin_dim factorized (1D per axis).

    ★ spec-default (확정): factorized 3-axis (`bin_heads`) 만 사용. joint bin^3 head (60³=216K logits) 는
    explosion 으로 사용 X — 본 클래스에 정의 없음.
    """
    def __init__(self, dim_cf, hidden=64, dim_encoder=0, bin_dim=60, bin_size=0.001):
        super().__init__(dim_cf=dim_cf, hidden=hidden, dim_encoder=dim_encoder)
        self.delta = None
        # 3 × bin_dim heads (factorized — joint bin^3 폭주 회피)
        self.bin_heads = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(hidden), nn.Linear(hidden, hidden // 2), nn.GELU(),
                nn.Linear(hidden // 2, bin_dim),
            ) for _ in range(3)
        ])
        self.bin_size = bin_size
        self.bin_dim = bin_dim
        # register_buffer 로 state_dict 포함 + device 자동 이동 (매 forward 재생성 X)
        bin_centers = torch.linspace(-bin_dim / 2 * bin_size, bin_dim / 2 * bin_size, bin_dim, dtype=torch.float32)
        self.register_buffer("bin_centers", bin_centers)

    def forward(self, cf, encoder_emb=None):
        # ... stem + blocks
        # Per-axis softmax: expected delta = Σ prob_i × bin_center_i
        delta_per_axis = []
        bin_probs = []
        for head in self.bin_heads:
            logits = head(h)  # (B, bin_dim)
            prob = F.softmax(logits, dim=-1)
            expected = (prob * self.bin_centers).sum(dim=-1)  # buffer 이미 device 동기
            delta_per_axis.append(expected)
            bin_probs.append(prob)
        delta = torch.stack(delta_per_axis, dim=-1)  # (B, 3)
        return delta, {"bin_probs": bin_probs}


class IterativeRefinementCorrector(nn.Module):
    """M4 / Phase 5 Z3: 3-step iterative refinement (parameter shared).

    Self-contained inline spec (plan-010 §7.1 의 Z3 module reuse; 본 클래스 = wrapper):
    - base_corrector: `v1.RedesignedCorrectionNet(dim_cf=base_dim_cf + 8, hidden=64, dim_encoder=...)` —
      stem.in_features = `dim_cf + 8` 로 인스턴스화 (step_idx_emb concat 수용). wrapper `__init__` 에서 직접 생성.
      parameter 공유 — n_steps 회 동일 weight 반복 호출.
    - n_steps: int (default 3)
    - per_step_cap: float (default 0.003 m = 3mm; cap6mm 의 1/2)
    - step_idx_emb: nn.Embedding(n_steps, dim_step_emb=8) — 매 step 의 idx 를 임베드해 cf 와 concat.
      step_idx_emb 의 init = N(0, 0.02), bias 없음. step_idx_emb output 은 cf 와 같은 dim 으로 broadcast
      되도록 dim_step_emb=8 을 dim_cf 에 단순 concat → 입력 dim 이 dim_cf+8 로 늘어남
      (base_corrector 의 stem.in_features 도 dim_cf+8 로 정의 — wrapper 가 책임).
    - forward 절차:
        cand_t = cand_0 (P1.F0 anchor pos)
        for t in range(n_steps):
            cf_t = concat([cf_base(cand_t), step_idx_emb(t).expand(B, -1)], dim=-1)  # (B, dim_cf + 8)
            delta_t = base_corrector(cf_t, encoder_emb)          # (B, 3) world frame
            delta_t = delta_t * min(1, per_step_cap / ‖delta_t‖) # per-step cap
            cand_t = cand_t + delta_t
        return cand_t − cand_0   # 누적 delta (caller 는 cand_0 + return → corrected_pos)
    - loss: 매 step 의 cand_t 와 target 의 err 를 stage-wise penalize (huber, weight = [1, 1, 1] uniform).

    Unified contract bridge (§7.1 의 `forward(cf, encoder_emb=None) -> (delta, aux)` 호환):
      M4 wrapper 가 corrector class 로 사용될 때:
        def forward(self, cf, encoder_emb=None):
            # cf 안에 cand_0 정보 포함 (P1.F0 anchor pos 는 caller 가 cf 의 candidate-relative 항목으로 인코딩).
            # 본 wrapper 는 iterative refinement 의 *누적 delta* 만 반환 (unified contract).
            accumulated_delta = self._refine(cf, encoder_emb, n_steps=self.n_steps)  # (B, 3)
            aux = {"per_step_deltas": list_of_step_deltas}  # (B, 3) × n_steps, stage-wise loss 진단용
            return accumulated_delta, aux
      caller 측 = `delta, aux = model(cf)` → corrected_pos = cand_0 + cap6mm(delta).
      stage-wise loss = trainer 가 `aux["per_step_deltas"]` 를 받아 매 step 의 corrected_pos_t 와 target 의 huber 합산.
    """
    # implementation: plan-010 §7.1 의 RedesignedIterativeRefiner reuse (import as v1_iter)


class GMMCorrector(v1.RedesignedCorrectionNet):
    """M5: probabilistic — μ + diagonal Σ output. Loss = NLL.

    Inference: expected delta = μ (or μ + samples for uncertainty).

    NLL closed form (per-axis diagonal Gaussian, batch-summed):
        nll_per_sample = 0.5 * Σ_axis (((target_axis − mu_axis) / sigma_axis) ** 2 + 2 * logsigma_axis)
                       = 0.5 * Σ_axis ((target_axis − mu_axis)² * exp(−2 * logsigma_axis) + 2 * logsigma_axis)
        # const term (log(2π) per axis) omitted — constant per sample, doesn't affect gradient.
        return nll_per_sample.mean()
    logsigma clamp [-6.0, 0.0] → sigma ∈ [e⁻⁶, 1] ≈ [2.5e-3 m, 1 m]:
      - lower bound 2.5mm: 학습 초기 σ → 0 폭주 방지 (residual 0.005m order 의 1/2 scale).
      - upper bound 1m  : σ → ∞ trivial 회피 (delta 가 의미 있는 m-scale 내).
    """
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
        # inference 시 expected delta = mu; trainer 는 aux['logsigma'] 로 NLL 계산
        return mu, {"logsigma": logsigma}


def gmm_nll_loss(mu, logsigma, target):
    """GMM NLL closed form (diagonal Gaussian, per-sample sum-over-axes, batch-mean).
    mu: (B, 3), logsigma: (B, 3) clamped to [-6, 0], target: (B, 3).
    Returns scalar.
    """
    inv_var = torch.exp(-2.0 * logsigma)           # 1/σ²
    sq_err  = (target - mu) ** 2                    # (B, 3)
    nll     = 0.5 * (sq_err * inv_var + 2.0 * logsigma).sum(dim=1)  # (B,)
    return nll.mean()


class WiderShallowCorrector(v1.RedesignedCorrectionNet):
    """M6: depth=1, hidden=256. small data 적합 추정."""
    def __init__(self, dim_cf, hidden=256, dim_encoder=0):
        super().__init__(dim_cf=dim_cf, hidden=hidden, dim_encoder=dim_encoder)
        # depth=1 — 부모 forward 가 self.blocks 를 iterable 가정 시 ModuleList 으로 wrap 필수.
        self.blocks = nn.ModuleList([v1.ResidualMLPBlock(hidden)])
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
    """F3: per-sample coefficient regression (plan-007 Step 4 reuse).

    MLP outputs (par_i, perp_i) for each sample → frenet candidate with per-sample coefs.

    in_dim = 12 (self-contained spec, plan-007 Step 4 carry-over):
      - last-step motion (6): speed, prev_speed_ratio, acc_norm/speed, acc_par/speed, acc_perp/speed, turn_cos
      - jerk stats (3): jerk_mean, jerk_std, jerk_max
      - curvature (2): curvature_mean, curvature_max
      - z_scale (1): vertical motion scale
    feature 추출 = `make_ctx_features(trajectory_x, end_idx)` (analysis/plan-011/ctx_features.py, plan-007 reuse).

    Candidate position 식 (★ F3 self-contained spec):
      F3 는 F0 anchor (frenet_par120_perp_neg020) 의 (par=1.20, perp=−0.20) 만 *per-sample 가변*.
      나머지 4 coef = F0 default 고정: k_d1=1.94, k_d2=0.0, k_jerk=0.0, k_time=1.0.
      Candidate position 계산 = LearnableSingleCandidate.forward 의 식 그대로 (§8.1 의 cand_pos 식),
      coef vector = `(k_d1=1.94, k_d2=0.0, par_i, perp_i, k_jerk=0.0, k_time=1.0)` per sample i.
      즉 F3 는 F4 의 *2-coef 만 학습 + 4-coef 고정* 변형. 학습 loss 도 F4 와 동일 (hit-aware + F0 prox term).
    """
    def __init__(self, in_dim=12, hidden=32):
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
        # ctx_features: (B, 12) — see in_dim docstring above
        return self.net(ctx_features)  # (B, 2) → par, perp per sample


class LearnableSingleCandidate(nn.Module):
    """F4: data-driven learnable candidate (Idea 2 from 코드공유-upgrade.md).

    Learn 6 scalar coefficients as a single nn.Parameter via *hit-aware loss*
    (NOT soft-min — K=1 환경에서 soft-min over candidates 는 vacuous).
    Initialized to F0 anchor (frenet_par120_perp_neg020 → par=1.20, perp=-0.20).

    F4 학습 loss spec (K=1 fix, 코드공유-upgrade.md "Idea 2" 의 soft-min 표현 폐기):
        cand_pos = self(p0, ...)
        # 1) hit-aware base — huber + smooth squared hinge (L7 와 동일 형태)
        loss_hit = huber_loss(cand_pos − p0_truth) + hit_aware_hinge(cand_pos, p0_truth)
        # 2) F0 anchor prox term — 6-dim coef 의 init drift 억제 (over-fit 방지)
        loss_prox = lambda_prox * ‖self.coef − init_coef‖²       (lambda_prox = 0.01)
        total = loss_hit.mean() + loss_prox
    diversity reg 없음 (1 candidate). soft-min 항 없음 (1 candidate → softmin = identity).
    """
    def __init__(self, init_coef=(1.94, 0.0, 1.20, -0.20, 0.0, 1.0)):
        # 6-dim parameter: (k_d1, k_d2, k_par, k_perp, k_jerk, k_time)
        #   k_d1   = last-velocity multiplier (init 1.94 ≈ plan-006 F0 anchor 측정 fit; horizon=2 의 trivial 'h=2' 가 아님 —
        #            F0 anchor 의 *실제 학습된 effective multiplier* 가 plan-006 source code 에서 1.94 로 박제됨).
        #   k_d2   = last-acc multiplier (init 0.0 — disabled by default, F0 anchor)
        #   k_par  = parallel scale on Frenet t̂ (init 1.20 = F0 anchor 박제)
        #   k_perp = perpendicular scale on Frenet n̂ (init −0.20 = F0 anchor 박제)
        #   k_jerk = jerk-term scale on jerk_world (init 0.0)
        #   k_time = horizon time-scale exponent (init 1.0; t^k_time scaling, F0 anchor)
        #
        # ★ F4 init = F0 reproduce 검증 (P1.F0 reuse 와 numerical 일치 보장):
        #   preflight (G0) 시 F4 init 으로 candidate 생성 → P1.F0 의 frenet_par120_perp_neg020 candidate 와
        #   per-sample distance ≤ 1e-4 m 확인 (≥ 1e-4 시 init_coef 의 k_d1 또는 k_time 재캘리브 필요).
        #   이 검증은 §4.2.1 D001 알고리즘 직후 1줄 추가 (`f4_init_vs_f0_max_dist: <float>`, threshold 1e-4 m).
        super().__init__()
        self.coef = nn.Parameter(torch.tensor(init_coef, dtype=torch.float32))

    def forward(self, p0, v_last, a_last, jerk_last, t_hat, n_hat, b_hat, horizon=2, coef=None):
        """Generate single learnable candidate position.

        Args:
          p0:        (B, 3) last observed position (world).
          v_last:    (B, 3) last inter-frame velocity (world).
          coef:      (B, 6) optional per-sample coef override (F3 use case).
                     None → self.coef (shape (6,)) broadcast (F4 default).
                     (B, 6) → per-sample coef (F3 PerSampleMLPFormula output 주입).
                     ★ F3 호출 chain: par_perp_i = PerSampleMLPFormula(ctx_features)  # (B, 2)
                                    coef_i = torch.stack([1.94, 0.0, par_i, perp_i, 0.0, 1.0], dim=-1)  # (B, 6)
                                    cand_pos = LearnableSingleCandidate.forward(..., coef=coef_i)
                     (즉 F3 는 LearnableSingleCandidate 인스턴스를 *coef 비활성화* 모드로 호출.
                      spec-default 결정: F3 trainer 는 `self.coef.requires_grad_(False)` 로 frozen — optimizer 제외보다 가독성·재현성 우선.)
          a_last:    (B, 3) last inter-frame acceleration (world).
          jerk_last: (B, 3) last jerk (world).
          t_hat, n_hat, b_hat: (B, 3) Frenet basis at p0 (from build_frenet_basis).
          horizon:   int (default 2 — h step prediction).
        Returns:
          cand_pos:  (B, 3) candidate world position.

        Formula (Idea 2, frenet_par120_perp_neg020 extension):
          term_d1   = self.coef[0] * v_last * (horizon ** self.coef[5])         # extrapolated velocity drift
          term_d2   = self.coef[1] * a_last * (horizon ** self.coef[5]) ** 2 / 2  # accel quadratic term
          term_par  = self.coef[2] * (t_hat * (a_last * t_hat).sum(-1, keepdim=True))   # parallel-acc projected
          term_perp = self.coef[3] * (n_hat * (a_last * n_hat).sum(-1, keepdim=True))   # perpendicular-acc projected
          term_jerk = self.coef[4] * jerk_last * (horizon ** self.coef[5]) ** 3 / 6
          cand_pos  = p0 + term_d1 + term_d2 + term_par + term_perp + term_jerk
        """
        h = horizon
        # F4 default: c = self.coef (broadcast to (B, 6) via implicit). F3: c = coef (B, 6).
        c = coef if coef is not None else self.coef.unsqueeze(0).expand(v_last.shape[0], -1)
        # c shape: (B, 6) — per-sample (F3) 또는 broadcast (F4).
        time_pow1 = h ** c[..., 5:6]   # (B, 1)
        time_pow2 = time_pow1 ** 2
        time_pow3 = time_pow1 ** 3
        term_d1 = c[..., 0:1] * v_last * time_pow1
        term_d2 = c[..., 1:2] * a_last * time_pow2 / 2.0
        proj_par = (a_last * t_hat).sum(-1, keepdim=True)
        proj_perp = (a_last * n_hat).sum(-1, keepdim=True)
        term_par = c[..., 2:3] * t_hat * proj_par
        term_perp = c[..., 3:4] * n_hat * proj_perp
        term_jerk = c[..., 4:5] * jerk_last * time_pow3 / 6.0
        cand_pos = p0 + term_d1 + term_d2 + term_par + term_perp + term_jerk
        return cand_pos
```

### §8.2 Phase 1.F wrapper

4 sub-exp (fixed L0 + In-A + M0):

| sub-exp | formula |
|---|---|
| P1.F0 (anchor) | frenet_par120_perp_neg020 — reuse from P1.L0 |
| **P1.F1** | CMA-ES tuned 6 vars (plan-007 Step 2 best_params reuse) |
| P1.F2 | basis ablation best (plan-007 Step 3 8 vars reuse) |
| **P1.F3** | per-sample MLP coefficient regression (plan-007 Step 4 reuse) |
| **P1.F4** | learnable single candidate (Idea 2 변형 — K=1 fix: hit-aware loss + F0 anchor prox term, soft-min/diversity reg 폐기) |

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
  - axis-level aggregation 함수 = `max(ΔOOF_i for sub_exp_i in axis where sub_exp_i ≠ anchor) ≥ 0.005` (§0.5 G1 (b) 와 일치)
  - **L axis 특수 규칙** (Z1 base bleed 명문화):
    - `axis-positive` 판정 = `max(delta_vs_anchor for L2~L7) ≥ 0.005` (= L0 base).
    - 별도 보고: `delta_vs_z1` (= L1 base) — *component-only contribution*.
    - 판정 분기:
      (a) `max(delta_vs_anchor) ≥ 0.005` AND `max(delta_vs_z1) ≥ 0.003` → **component lever positive** (component 자체로 +0.003 이상 — Z1 외 추가 신호).
      (b) `max(delta_vs_anchor) ≥ 0.005` AND `max(delta_vs_z1) < 0.003` → **Z1-dominant lever** (Z1 minimum 자체가 main contribution — L1 을 L̂ 로 채택).
      (c) `max(delta_vs_anchor) < 0.005` → **L axis no-positive** (`loss_axis_no_lever_positive` warn).
    - Phase 3 entry 시 L̂ 식별: case (a) → `argmax(delta_vs_z1)` 의 sub-exp; case (b) → L1.

### §9.3 G1 fail handling

- 4 axis 모두 ≤ +0.005 → `phase1_no_lever_positive` severe. autonomous:
  - 옵션 a: Phase 3 skip, G_final 직접 진입 (best Phase = Phase 1 max + plan-012 paradigm 교체 carry-over)
  - 옵션 b: Phase 5 (iterative) 단독 진입 (Z3 가 plan-005 의 9.77% 회복 가능성)

---

## §10. STAGE 3 (Phase 3) — Pairwise Combinations (G2)

### §10.1 4 pair (5-fold) + 3~4 solo-5fold prep

> super-additive 식 (§10.5) 의 `oof_lever_a`, `oof_lever_b` 는 5-fold 측정값. Phase 1 1-fold 와 직접 합산 불가
> → Phase 3 entry 시 *solo lever 5-fold 재측정* 필수.

| sub-exp | spec | 진입 조건 | 비용 |
|---|---|---|---|
| **P3.0a** | anchor combo (L0+IA+M0+F0) 5-fold | 항상 (preflight 의 0.6491 reproduce 와 분리 5-fold 측정) | ~50min |
| **P3.0b** | L̂ solo 5-fold | 항상 | ~50min |
| **P3.0c** | In̂ solo 5-fold | 항상 | ~50min |
| **P3.0d** | M̂ solo 5-fold | 항상 | ~50min |
| **P3.0e** | F̂ solo 5-fold | F̂ ≠ F0 (P1.F 에서 ΔOOF > 0) | ~50min |
| **P3.1** | L̂ + In̂ | 항상 | ~50min |
| **P3.2** | L̂ + M̂ | 항상 | ~50min |
| **P3.3** | L̂ + F̂ | F̂ ≠ F0 | ~50min |
| **P3.4** | In̂ + M̂ | 항상 | ~50min |

총 5-fold 재실행 = 4 prep (P3.0a~d) + 4 pair = **8 × ~50min = ~400min** (F̂ 진입 시 +1 prep + 1 pair = 10 × ~50min = ~500min).
§N+1 wall-time 회계 update 필요.

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
# oof_anchor = anchor combo (= P1.L0=IA=M0=F0) 의 *5-fold concat OOF* (Phase 3 시 reproduce, 단일 값)
# oof_lever_a, oof_lever_b = Phase 3 entry 시 *5-fold concat OOF* (Phase 1 의 1-fold 값 reuse 금지 —
#   Phase 1 1-fold 와 Phase 3 5-fold 의 fold-set 다르므로 직접 합산 불가).
# oof_pair = Phase 3 의 L̂+In̂ etc. 5-fold concat OOF.
delta_pair = oof_pair − oof_anchor
delta_solo_sum = (oof_lever_a − oof_anchor) + (oof_lever_b − oof_anchor)
if delta_pair > delta_solo_sum + 0.003:
    cls = "super-additive"
elif abs(delta_pair - delta_solo_sum) <= 0.003:
    cls = "additive"
else:
    cls = "sub-additive"
```

> *caveat — lever 독립성*: P3.2 (L̂ + M̂) 의 경우 L̂ = L2 (C008 gate-asymmetric loss) + M̂ = M1 (GateHead arch) 가
> *같은 mechanism (gate)* 의 다른 면을 누르는 sub-case 가능. 그 경우 super-additive 식의 *독립 가정* 위반 — `delta_pair` 가
> `delta_solo_sum` 보다 작게 측정되어도 mechanism collision 으로 해석 (= sub-additive 가 *진짜 sub-additive 가 아님*).
> Phase 3 산출 `phase3_summary.json` 의 entry 마다 `mechanism_overlap_flag: bool` (L axis pick 과 M axis pick 가
> 둘 다 gate-관련 lever 면 true) 추가, attribution 보고서 §9.7 caveat 검증에 포함.
>
> *decoupling 변형 sub-exp (mechanism_overlap_flag=true 시 진입)*:
>   P3.2' = L̂=**L3** (C009 asymmetric loss only, gate 없이) + M̂=M1 (GateHead arch) — mechanism 분리 변형.
>   진입 조건: `mechanism_overlap_flag(P3.2) == true` AND L3 ΔOOF 가 §9.2 의 axis-positive 임계 통과.
>   비용: +1 5-fold sub-exp (~50min). super-additive 분류 시 P3.2 와 P3.2' 둘 다 보고 → 진짜 mechanism overlap
>   분리 결과로 attribution 명확화. plan-011.1 carry-over 후보 1순위.

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

**P6.1 — TTA rotation 4** (★ entry condition: corrector output = *world frame* only.
L4 (Frenet anisotropic loss) 가 학습 단계에서만 local frame 활용했어도 forward output 은 world frame 이므로 P6.1 적용 OK.
그러나 만약 corrector arch 자체가 Frenet local 출력 (= forward 가 (t,n,b) 좌표계 delta 반환) 으로 바뀐 변형 시
P6.1 skip 또는 회전 시점을 world-domain reconstruct 후 적용 필수.):
```python
# 추론 시 입력 XY 평면 회전 (0°, 90°, 180°, 270°) × 모델 forward × 역회전 평균
# Z축 (중력) 건드리지 않음 — 물리적 대칭성
for theta in [0, 90, 180, 270]:
    x_rot = rotate_xy(test_x, theta)
    delta_rot = model(x_rot)               # world frame output (entry condition)
    delta = rotate_xy_inverse(delta_rot, theta)
    deltas.append(delta)
final_delta = mean(deltas)
```

**P6.2 — Multi-parse inference**:
```python
# 추론 시 입력 raw + SG smoothing + EMA smoothing 3 parse × 모델 forward × 평균
# ★ 학습 시 P6.2 와 동일 parameter (window=5, order=2, alpha=0.6) 사용 강제 — §6 In-F (MultiParseInput) 학습 단계의
#   3 parse augment 정책과 *완전 동기*. 학습-추론 mismatch 시 distribution shift.
#   학습 시 정책 (§N+3 caveat #6): "epoch 매 batch 마다 3 parse 중 *random 1개* sample (augment)" — 추론 시는 "3 parse 평균" (deterministic ensemble).
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
# ★ 각 sub-exp 는 *서로 다른 모델 + 다른 optimizer* — backward graph 격리되어 race-free.
#   단 optimizer step 의 stream 동기화는 명시적으로 처리해야 함 (CUDA spec).
import torch
streams = [torch.cuda.Stream() for _ in range(4)]
models     = [build_model(cfg)     for cfg in batch_of_4_subexp_configs]
optimizers = [build_optimizer(m, cfg) for m, cfg in zip(models, batch_of_4_subexp_configs)]

for batch in shared_dataloader:
    # 1) forward + backward on independent streams (parallel)
    for i, (model, optimizer, stream) in enumerate(zip(models, optimizers, streams)):
        with torch.cuda.stream(stream):
            optimizer.zero_grad(set_to_none=True)
            loss = model(batch)              # 모델 i 의 loss (자체 graph)
            loss.backward()                  # 모델 i 의 param.grad 만 누적
    # 2) stream barrier — 4 backward 완료 보장
    for stream in streams:
        stream.synchronize()
    # 3) optimizer step on default stream (각 모델 param 은 disjoint → race-free)
    for optimizer in optimizers:
        optimizer.step()
    # 4) (선택) 전체 device 동기 — logging / next batch 의 dataloader prefetch 시점 일관
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
| c13a (P3 solo-prep) | phase3 4~5 solo lever 5-fold (P3.0a~e) × ~50min | ~200~250 min |
| c13b (P3 pairwise) | phase3 4 pair × ~50min (5-fold) | ~200 min |
| c14 (P4 triple) | phase4 1~2 sub-exp × ~60min | ~120 min |
| c15 (P5 iterative, 조건부) | phase5 iterative × ~70min | ~70 min |
| c16 (P6 augment, 조건부) | phase6 TTA + multi-parse × ~30min | ~30 min |
| c17 (synthesis) | results.md + next_plan_candidates.md + 3 파일 frontmatter sync + plan-011.1 instruction | ~30 min |
| **합계** | (조건부 G4/G5 포함, P3.0 solo-prep 추가분 +200~250min) | **~16 hr** (조건부 skip 시 ~14.3 hr; F̂ 진입 시 +50 min) |

### §N+1.2 시나리오 B — 4-way GPU multi-stream (★ 권장, Phase 1 + Phase 3 병렬)

| Phase | 순차 | 4-way 병렬 | 단축 |
|---|---|---|---|
| Phase 0 (preflight) | 15 min | 15 min (CPU only) | — |
| Phase 1.L (8 sub-exp) | 80 min | **~25 min** | 4-way × overhead 1.25 |
| Phase 1.In (5 sub-exp) | 60 min | **~20 min** | |
| Phase 1.M (7 sub-exp) | 100 min | **~32 min** | |
| Phase 1.F (4 sub-exp) | 50 min | **~17 min** | |
| Phase 2 (attribution) | 20 min | 20 min (분석) | — |
| Phase 3 (4~5 solo-prep + 4 pair × 5-fold) | 400~450 min | **~100~115 min** | 4-way 동시 + per-task 순차 5-fold |
| Phase 4 (1~2 stack × 5-fold) | 120 min | **~60 min** | 5-fold 병렬 |
| Phase 5 (조건부) | 70 min | **~35 min** | 5-fold 병렬 |
| Phase 6 (조건부) | 30 min | 30 min | 추론만 |
| Phase 7 (synthesis) | 30 min | 30 min | — |
| 코드 작성 (c1/c3/c4/c6/c8/c10) | 105 min | 105 min | — |
| **합계** | ~16 hr | **~8.4 hr** (조건부 포함, P3.0 solo-prep 반영) | **~7.6 hr 단축** |

### §N+1.3 시나리오 C — sub-exp 병렬 + 5-fold 병렬 (max parallelization)

| Phase | 시나리오 B | 시나리오 C (+ 5-fold 병렬) |
|---|---|---|
| Phase 1 (24 sub-exp) | ~94 min | ~94 min (1-fold approx, 변경 없음) |
| Phase 3 (4 solo-prep + 4 pair × 5-fold) | 100~115 min | **~40 min** (8 task 동시 + 각 task 5-fold 동시) |
| Phase 4 | 60 min | **~15 min** (5-fold 동시) |
| Phase 5 | 35 min | **~12 min** (5-fold 동시) |
| Phase 6/7 + 코드 | 165 min | 165 min |
| **합계** | ~8.4 hr | **~5.5 hr** (조건부 포함, P3.0 solo-prep 반영) |

### §N+1.4 권장

- **시나리오 B 채택** — 안정성 (multi-stream) + 단축 (~5.5 hr) 균형
- 시나리오 C 진입 조건: 시나리오 B 의 Phase 3 OOM 없이 안정 작동 확인 후
- decision-note 박제: `decision-note: parallel-execution — scenario B (4-way GPU multi-stream + CPU 24-worker data prep) 채택, wall-time ~16h (P3.0 solo-prep 포함) → ~8.4h`

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

6. **multi-parse (In-F) 의 학습 정책 (★ single source-of-truth)**: SG/EMA smoothing 은 deterministic. spec 결정 = **학습 시 random 1 parse augment** (batch 일관성, sample 단위 아님), **추론 시 3 parse cf 평균** (deterministic ensemble). 학습 cost ~동일, 추론 cost ×3. §6.1 MultiParseInput / §13.2 P6.2 / 본 caveat 3 곳 동기 — 다른 정책 (예: 학습 시 3 parse 평균) 채택 시 3 곳 동시 수정.

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

- v1 (2026-05-13): 초안 — plan-010 의 depth (defect fix) 와 *상호 보완 breadth* (4 axis × ~25 single-axis ablation). notes/코드공유-upgrade.md 의 C008/C009/C010/D001 후보 + notes/prior-ideas.md 의 Physics conservation + Multi-parse + notes/mosquito-trajectory-ideas.md 의 TTA + GMM 통합. Phase 0~7, G0~G_final 7 gate, commit chain c1~c17 + G4/G5 조건부. LB 제출 0 회 (plan-010.1 carry-over 패턴). §15 병렬 실행 정책 신설 (server CPU 48 core + GPU 1 device:0 기준, Phase 1 sub-exp 4-way GPU multi-stream + Phase 3+ 5-fold ProcessPoolExecutor 분할). caveat #20~#22 (parallel reproducibility / GPU memory / anchor invariance) 추가.
- v1 plan-review 7 iter 결과 반영 (2026-05-13): Phase 3 super-additive 식의 5-fold 정합을 위해 P3.0a~e solo-prep 항목 추가 (5-fold lever 단독 재실행 ~200~250min 가산) → wall-time 회계 갱신 ~12.7h → ~16h (시나리오 A), ~7.2h → ~8.4h (시나리오 B). 4-axis 의 loss/arch component spec 자족성 보강 (asymmetric loss replacement multiplier, physics conservation units 일치, GMM NLL closed form 박제, M4 IterativeRefinementCorrector unified forward contract, F3 PerSampleMLPFormula 와 LearnableSingleCandidate coef-override chain). P1.L2 의 GateHead+asymmetric loss joint 명시 + Phase 3 P3.2' decoupling 변형 sub-exp 신설.

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
