---
plan_id: 022
version: 1.0
date: 2026-05-19 (Asia/Seoul)
status: TODO
best_sub_exp: null
best_hit_1cm: null
best_hit_1.5cm: null
best_delta_1cm: null
best_delta_1.5cm: null
based_on:
  - 021 (selector-only ablation finding: LGBM reg_offset = bound 의 0.5% (~0.01mm) — dead. selector-only ≈ full. anchor argmax mode collapse: GRU anchor 1/5/6 dead (합 1.75 %), LGBM anchor 0 over-pick +6 pp. proxy target=t=10 train↔test 분포 동일 (TV 0.017) — covariate shift 없음, mode collapse = inductive bias)
  - 020 (F0 baseline 0.6320 / 0.8033 + 5-fold stable_fold_id MD5)
inspired_by:
  - 021 (anchor layout 효과 미측정 — out-of-scope §56 박제됐던 ablation. selector-only paradigm 의 anchor-counter 효과 정량화)
code_reuse:
  - module: analysis/plan-021/build_input.py
    symbols: [build_frenet_basis_3d, to_frenet, build_input_common, build_input_lgbm_extra]
    reason: 170D LGBM input pipeline 동일 (plan-021 §6.1 그대로). anchor layout 만 swap.
  - module: analysis/plan-021/dual_head_model.py
    symbols: [LgbmDualHead]
    reason: clf head 부분만 차용. reg head 제거 wrapper class 신규 작성.
  - module: analysis/plan-020/baseline_f0.py
    symbols: [f0_baseline, D1, PAR, PERP, R_HIT, R_HIT_LOOSE]
    reason: F0 baseline injection + paired Δ anchor.
  - module: src/io.py
    symbols: [load_all_samples, load_labels]
    reason: data loader.
  - module: src/pb_0_6822/selector.py
    symbols: [stable_fold_id]
    reason: 5-fold stable split (plan-020/021 carry).
followed_by:
  - plan-023 (가칭): best anchor layout (plan-022 결과) 위 corrector reg head 재투입 → 1.5cm metric ceiling 돌파 ablation
  - plan-024 (가칭): best layout 위 GRU sub-exp 비교 + ensemble 잠재력 측정
scope: corrector-free selector-only LGBM 위 anchor layout × τ_cls 2D sweep. 7 layout × 3 τ_cls = 21 cell. pass criterion paired Δ ≥ +0.005 둘 다. GRU / corrector 재투입 / DACON LB / ensemble = out-of-scope.
exp_ids:
  - Z022_A1_octa7
  - Z022_A2_ico13
  - Z022_A3_cubocta13
  - Z022_A4_2shell13
  - Z022_A5_cube8
  - Z022_A6_bcc14
  - Z022_A7_fib13
lb_score: null
band: null
---

# plan-022 v1 — Corrector-free Anchor Layout Sweep (selector-only LGBM, 7 layout × 3 τ_cls)

## §0. 한 줄 목적

> **plan-021 의 corrector reg head 가 LGBM 에서 dead (output ≈ 0) + selector-only ≈ full 동등 + 1.5cm 미달 (Δ +0.0037)** 의 3 finding 위에서, **selector-only 단순화 + anchor layout 변경 + τ_cls 완화** 로 *paired Δ ≥ +0.005 둘 다* 통과를 시도. 7 anchor layout × 3 τ_cls = 21 cell 2D sweep. **모든 input + LGBM hparam = plan-021 §6.1 그대로** (단일 변수 = anchor 좌표 + τ_cls).
>
> **layout 후보 7**:
> 1. **A1** octahedron + center (= 현행 7, baseline)
> 2. **A2** icosahedron + center (13) — Platonic 가장 isotropic
> 3. **A3** cuboctahedron + center (13) — FCC neighbor, uniform edge 0.5cm
> 4. **A4** 2-shell octahedron (13) — center + inner 0.25cm + outer 0.5cm, 잔차 크기 multi-scale coverage
> 5. **A5** cube corners (8) — center 제거, 강제 octant assignment
> 6. **A6** BCC 14 (axis 6 + corner 8) — octahedron + cube 결합
> 7. **A7** Fibonacci spiral 12 + center (13) — quasi-uniform, count 자유 control
>
> **τ_cls scan**: {0.001, 0.003, 0.005}m. plan-021 default = 0.001m (anchor 0.5cm 의 1/5) → anchor 가 13~14 로 늘면 sharp 화 우려 → softening 동시 측정.
>
> **pass criterion**: 21 cell 중 ≥ 1 개가 paired Δ_1cm ≥ +0.005 AND Δ_1.5cm ≥ +0.005 → G3 PASS.
>
> **out-of-scope**: corrector reg head 재투입 / GRU sub-exp / LB 측정 / DACON submit / ensemble / anchor radius ≠ 0.5cm (단, A4 의 inner shell 0.25cm 은 layout 내부 정의로 OK).

---

## §0.5 Quick Reference (autonomous loop 매 turn 읽는 section)

### 합격 기준 (G-gate sequence)

- **G0**: 3 module (anchors / selector_only_model / run_oof) import + smoke + tests green. plan-021 build_input.py import 정상. 위반 시 `infra_drift` severe.
- **G1**: F0 baseline 5-fold concat OOF — hit@1cm ∈ [0.6315, 0.6325] AND hit@1.5cm ∈ [0.8028, 0.8038] (plan-020 / 021 carry). 위반 시 `f0_reproduce_drift` severe.
- **G2.A{n}** (n ∈ 1..7): sub-exp A{n} 의 3 τ_cls cell 모두 metric finite + `max_class_ratio < 0.95` (= soft-mean 의 최대 anchor share, `probs_all.mean(axis=0).max()` 식, §3.3 / §6.2 일관). 위반 시 `lgbm_numerical` severe / `soft_label_collapse` warn.
- **G3 (paradigm-level)**: 21 cell (= 7 layout × 3 τ_cls) 중 ≥ 1 cell 이 paired Δ ≥ +0.005 *둘 다* 통과 → PASS. 0 통과 = `all_negative` warn 박제 후 G_final 진입.
- **G_final**: results.md + best cell 박제 (layout + τ_cls + Δ) + follow-up plan 후보 ≥ 2 건 박제 + 3-file frontmatter sync.

### G-gates

- G0: STAGE 0 인프라 [TODO]
- G1: STAGE 1 F0 baseline reproduce [TODO]
- G2.A1: A1 octa7 3 τ_cls 측정 [TODO]
- G2.A2: A2 ico13 3 τ_cls 측정 [TODO]
- G2.A3: A3 cubocta13 3 τ_cls 측정 [TODO]
- G2.A4: A4 2shell13 3 τ_cls 측정 [TODO]
- G2.A5: A5 cube8 3 τ_cls 측정 [TODO]
- G2.A6: A6 bcc14 3 τ_cls 측정 [TODO]
- G2.A7: A7 fib13 3 τ_cls 측정 [TODO]
- G3: STAGE 3 best cell selection + paradigm finding [TODO]
- G_final: STAGE 4 results + 3-file sync [TODO]

### Commit chain (next-up)

| # | type | spec section | status |
|---|---|---|---|
| c1 | docs | `plans/plan-022-corrector-free-anchor-layout-sweep.md` v1 작성 (plan-review-master 4-iter 자동 fix BLOCKER 0 도달 — iter 1 BLOCKER 2→fix / iter 2 BLOCKER 1→fix / iter 3-4 BLOCKER 0 sustained + AMBIGUITY ~10 fix) | [DONE — 2613835] |
| c2 | code | `analysis/plan-022/anchors.py` (7 layout numpy 상수 + smoke test: 각 layout ‖a‖ ≤ 0.005m + dtype float32 + shape 정합) | [DONE — 908da5a] |
| c3 | code | `analysis/plan-022/selector_only_model.py` (LgbmSelectorOnly class — plan-021 LgbmDualHead.clf_head carry, reg head 제거. predict → probs only. + build_soft_label_with_tau helper) | [DONE — 40935ab] |
| c4 | code | `analysis/plan-022/run_oof.py` (5-fold OOF runner, 7 layout × 3 τ_cls = 21 cell sweep, paired Δ 21 셀 산출 + best cell selection) | [DONE — f3ea888] |
| c5 | test | `tests/test_plan022_smoke.py` (8 pytest: import + 7 layout shape/norm 검증 + LgbmSelectorOnly fit/predict + soft label sum=1 + G1 reproduce sanity) | [TODO] |
| G0 | gate | smoke + tests green — 8/8 pytest pass | [TODO] |
| c6 | exp G1 | F0 baseline 5-fold OOF reproduce → 0.6320 / 0.8033 (plan-020/021 carry exact). `analysis/plan-022/baseline_carry.json` 박제 | [TODO] |
| G1 | gate | F0 hit@1cm ∈ [0.6315, 0.6325] AND hit@1.5cm ∈ [0.8028, 0.8038] | [TODO] |
| c7 | exp G2.A1 | A1 octa7 (baseline reproduce) — 3 τ_cls cell 측정 → `results_A1.{json,md}` | [TODO] |
| G2.A1 | gate | 3 cell metric finite + max-class 비율 < 0.95 | [TODO] |
| c8 | exp G2.A2 | A2 ico13 — 3 τ_cls cell 측정 → `results_A2.{json,md}` | [TODO] |
| G2.A2 | gate | 3 cell metric finite + max-class 비율 < 0.95 | [TODO] |
| c9 | exp G2.A3 | A3 cubocta13 — 3 τ_cls cell 측정 → `results_A3.{json,md}` | [TODO] |
| G2.A3 | gate | 3 cell metric finite + max-class 비율 < 0.95 | [TODO] |
| c10 | exp G2.A4 | A4 2shell13 — 3 τ_cls cell 측정 → `results_A4.{json,md}` | [TODO] |
| G2.A4 | gate | 3 cell metric finite + max-class 비율 < 0.95 | [TODO] |
| c11 | exp G2.A5 | A5 cube8 — 3 τ_cls cell 측정 → `results_A5.{json,md}` | [TODO] |
| G2.A5 | gate | 3 cell metric finite + max-class 비율 < 0.95 | [TODO] |
| c12 | exp G2.A6 | A6 bcc14 — 3 τ_cls cell 측정 → `results_A6.{json,md}` | [TODO] |
| G2.A6 | gate | 3 cell metric finite + max-class 비율 < 0.95 | [TODO] |
| c13 | exp G2.A7 | A7 fib13 — 3 τ_cls cell 측정 → `results_A7.{json,md}` | [TODO] |
| G2.A7 | gate | 3 cell metric finite + max-class 비율 < 0.95 | [TODO] |
| c14 | analysis | 21 cell paired Δ 표 + best cell 식별 + layout-axis marginal + τ_cls-axis marginal + paradigm finding (anchor layout 효과 / τ scale 효과 / mode collapse 완화 여부) → `paradigm_analysis.{json,md}` | [TODO] |
| G3 | gate | ≥ 1 cell paired Δ ≥ +0.005 둘 다 (0 = `all_negative` warn 후 G_final 직진) | [TODO] |
| c15 | docs | 3-file frontmatter sync (status all_complete, band positive/null, best_sub_exp = `<best_layout>_tau<val>`) + `results.md` (필수 항목 11건) + follow-up plan-023/024 박제 | [TODO] |
| G_final | gate | 3-file sync + §0.5 c1~c15 모두 [DONE] + follow-up 2건 박제 | [TODO] |

### Plan-specific severe (WORKFLOW.md §12.3 default 위 추가분)

- `f0_reproduce_drift`: G1 reproduce 가 plan-020/021 hard evidence 0.6320 / 0.8033 ±0.0005 밖. → halt.
- `lgbm_numerical`: 21 cell 중 어느 LGBM classifier 출력 NaN/Inf. soft label CE / softmax 산출 버그 의심.
- `soft_label_collapse`: 21 cell 중 어느 layout 의 selector probs 가 단일 anchor 에 95% 이상 mass (= `max_class_ratio = probs_all.mean(axis=0).max() > 0.95`, soft-mean 정의 — §3.3 / §6.2 와 일관). τ_cls 너무 sharp 의심. warn (severe 아님). **G3 분모 영향**: drop 된 cell N_drop 개에 대해 G3 = "(21 − N_drop) cell 중 ≥ 1 PASS_BOTH". cumulative drop OK (각 cell warn 박제 + paradigm_analysis.json `dropped_cells` 리스트 누적). **halt threshold**: N_drop ≥ 11 (= 21 의 절반 초과) 시 severe `soft_label_collapse_majority` escalate (sweep paradigm 의미 상실).
- `frenet_basis_degenerate`: plan-021 carry — ‖v_last‖ < 1e-9 또는 ‖a_⊥‖ < 1e-9 sample 비율 > 5%. plan-021 fallback (R_wfn ← I_3) 그대로 적용.
- `all_negative`: 21 cell 모두 paired Δ < +0.005 *둘 다*. → warn 박제 후 G_final (paradigm-level evidence — anchor layout lever 의 한계).

### Plan-specific paths (WORKFLOW.md §12.5/§12.6)

- whitelist 추가:
  - `analysis/plan-022/**`
  - `tests/test_plan022_smoke.py`
- blacklist (plan-001~021 산출 자동 변경 금지):
  - `runs/baseline/{B,S,R,P,D,E,F,H,Z020,Z021}*/**`
  - `analysis/plan-{001..021}/**` (단, **read-only import** 는 §4.3 의 plan-021 build_input.py reuse 만 예외)
- 참조 (read-only):
  - `analysis/plan-021/build_input.py` — 170D input pipeline carry
  - `analysis/plan-021/dual_head_model.py:LgbmDualHead` — clf head 구조 carry
  - `analysis/plan-021/results.md` — finding evidence
  - `analysis/plan-020/baseline_oof.json` — F0 0.6320 / 0.8033 hard evidence
  - `analysis/plan-020/baseline_f0.py` — F0 산식
  - `src/pb_0_6822/selector.py:stable_fold_id` — 5-fold split

### Decision-note 사용 예 (자율 결정 시 commit msg 박제)

- `decision-note: spec-default — A2 icosahedron 좌표 = scale × (0, ±1, ±φ) + permutations 정규화 (‖a‖ = 0.005m, scale = 0.005 / √(1+φ²)). φ = (1+√5)/2.`
- `decision-note: spec-default — A4 2-shell octahedron: center (0,0,0) + 6 axis × 0.0025m (inner) + 6 axis × 0.005m (outer) = 13 anchor. inner shell radius 의 spec rationale: outer 의 1/2 (radial gradient).`
- `decision-note: spec-default — A7 Fibonacci spiral N=12 generator: φ_golden = (1+√5)/2, θ_i = 2π·i/φ_golden, z_i = 1 − 2(i+0.5)/N for i ∈ [0,N-1]. r=0.005m × sphere unit vec.`
- `decision-note: spec-default — τ_cls scan = {0.001, 0.003, 0.005}m × 7 layout = 21 cell. 동시 측정 (selection 아님, 전수 박제).`
- `decision-note: spec-default — LGBM hparam = plan-021 LgbmDualHead.clf_head 그대로 (n_estimators=500, lr=0.05, num_leaves=63). multi-output K-class softmax classifier (K=anchor count, 7/8/13/14 가변).`
- `decision-note: spec-default — final_frenet = Σ_a probs[a] · ANCHORS[a] (reg_offset 항 제거, 즉 final 의 Frenet norm ≤ max anchor radius).`
- `decision-note: spec-default — 5-fold = plan-020/021 carry (stable_fold_id MD5 32-bit prefix mod 5). seed X.`

---

## §1. 배경

### §1.1 plan-021 의 5 finding 과 본 plan 의 응답

| plan-021 finding | 본 plan-022 응답 |
|---|---|
| **LGBM reg_offset = bound 의 0.5% (~0.01mm)** — reg head dead | reg head 완전 제거 → corrector-free selector-only 단순화 |
| **selector-only ≈ full** (LGBM Δ 동일, GRU Δ_1cm marginal +0.0025 손실) | selector 만 keep, 효율↑ (학습 절반, 모델 절반) |
| **GRU argmax anchor 1/5/6 dead** (합 1.75 %) + LGBM anchor 0 over-pick (+6 pp) | anchor layout 자체 변경으로 mode collapse 직접 공략 (isotropic / center 제거 / 2-shell / 14 anchor isotropic) |
| **proxy target=t=10 train↔test 분포 동일** (TV 0.017, KL 7e-4) | covariate shift 부재 — train OOF 최적이 test 에서도 valid 가정 안전 |
| **A LGBM Δ_1.5cm +0.0037 < +0.005** (1cm PASS / 1.5cm 미달) | τ_cls 완화 + 13~14 anchor 의 finer grid 로 1.5cm reach 확대 시도 |

### §1.2 사용자 narrative — anchor layout × τ_cls 2D sweep

plan-021 의 7-anchor (octahedron + center) 는 *코드북 한 종류만* 측정 (사용자 명시 — bake-off 안 함). 본 plan-022 = **그 bake-off 를 본격 실행**. layout 차원 7 + τ_cls 차원 3 = 2D 평면 위 21 cell 전수.

**왜 동시 2D scan?** anchor 개수 K 가 7 → 13/14 로 늘면 softmax 가 sharp 화 (각 anchor 의 평균 prob 1/K 가 작아짐). plan-021 의 τ_cls=0.001m 은 K=7 위 spec — K 가 바뀌면 같은 τ_cls 가 다른 effective temperature. 따라서 layout 단독 변경 시 *층화된 효과* (anchor + temperature 결합) 가 분리 안 됨.

### §1.3 가설

**H1 (layout 효과)**: 13~14 anchor isotropic layout (A2/A3/A6) 이 anchor 7 (A1) 대비 paired Δ_1.5cm 더 큼 (1.5cm hit zone coverage ↑).
**H2 (τ 효과)**: τ_cls 완화 (0.001 → 0.003/0.005) 가 mode collapse 완화 + soft-mixture 부드러움 → Δ_1cm 유지 또는 ↑.
**H3 (center 제거 효과)**: A5 (cube 8, no center) 가 A1 대비 anchor 0 over-pick 제거 → 1.5cm 향상 가능. 단, F0-perfect sample 의 1cm metric 손실 risk.
**H4 (2-shell 효과)**: A4 (2-shell 13) 가 multi-scale coverage 로 1.5cm 향상 + 1cm 유지.

### §1.4 baseline = A1_tau0001

A1 (octahedron 7) + τ_cls=0.001 = **plan-021 A LGBM selector-only ablation 의 정확 재현**:
- 예상 hit@1cm = 0.6486 (plan-021 selector-only 측정 0.6486 = full 0.6488 −0.0002)
- 예상 hit@1.5cm = 0.8070 (selector-only ≈ full)
- 예상 Δ_1cm = +0.0166, Δ_1.5cm = +0.0037

**A1_tau0001 cell 의 sanity 위치**: **informational only** — G2.A1 합격 기준은 §3.2 의 일반 G2.A{n} criterion (`metric finite + max_class_ratio < 0.95`) 만 적용. 위 예상 hit 값과의 ±drift 가 cell-level halt trigger 아님 (LGBM `random_state=20260519` 가 plan-021 의 single-seed 결정성과 같으면 정확 일치, drift 시 plan-021 carry 미세 RNG 차이로 추정 — drift 박제 후 진행).

만약 21 cell 모두 baseline 보다 못하면 → **anchor layout lever 자체가 한계** 라는 paradigm-level evidence (negative finding 도 valuable).

---

## §2. 가설 검증 paradigm (한 변수 원칙)

| 변수 | 차원 | 값 |
|---|---|---|
| **anchor layout** | 7 | A1 octa7, A2 ico13, A3 cubocta13, A4 2shell13, A5 cube8, A6 bcc14, A7 fib13 |
| **τ_cls** | 3 | 0.001, 0.003, 0.005 (단위 m, anchor radius 0.5cm 대비 1/5, 3/5, 1/1) |
| 그 외 모두 (input 170D, LGBM hparam, fold split, soft label 산식 frame, final composition 식) | 1 | plan-021 §6.1 / §6.2 default 그대로 |

→ **셀 (A{n}, τ)** 마다 baseline (A1_tau0001) 대비 **2 변수만** 변경 (anchor + τ). plan-021 baseline 대비는 추가로 reg head 제거 (3 변수). 하지만 plan-021 selector-only ablation 결과를 1 변수 carry → 본 plan 내부에서 anchor + τ 가 *uniquely* 변경되는 변수.

---

## §3. 사전 등록 (Pre-registration)

### §3.1 Fold split (plan-020/021 carry)

| 분할 | 값 |
|---|---|
| folds | 5 |
| fold 할당 | `stable_fold_id(str(sample_id), 5)` (plan-020 carry, MD5 32-bit prefix mod 5) |
| seed | fold split deterministic (no seed) |
| F0 baseline | plan-020 `baseline_oof.json` 의 집계 metric (0.6320 / 0.8033) sanity carry. per-sample F0 pred 는 run-time 재계산 (`bf.f0_baseline(X, end_idx=10)`, < 1s). |

### §3.2 합격 기준 (정량)

- **G0**: 3 모듈 (anchors / selector_only_model / run_oof) import + smoke + tests green
- **G1**: F0 reproduce hit@1cm ∈ [0.6315, 0.6325] AND hit@1.5cm ∈ [0.8028, 0.8038]
- **G2.A{n}** (n ∈ 1..7): sub-exp A{n} 의 3 τ_cls cell 모두 metric finite + max-class 비율 < 0.95 (soft_label_collapse 회피)
- **G3**: 21 cell 중 ≥ 1 개가 paired Δ_1cm ≥ +0.005 AND paired Δ_1.5cm ≥ +0.005
  - 0 통과 시 → `all_negative` warn 박제 후 G_final 직진

### §3.3 평가 점수

| metric | 식 | 비교 |
|---|---|---|
| hit@1cm | `mean(‖final_world − gt‖₂ ≤ 0.01)` | F0 baseline 0.6320 |
| hit@1.5cm | `mean(‖final_world − gt‖₂ ≤ 0.015)` | F0 baseline 0.8033 |
| paired Δ | sample-level: `mean_i(1{‖pred_cell_i − gt_i‖ ≤ R} − 1{‖pred_F0_i − gt_i‖ ≤ R})`. 5-fold concat OOF. **`pred_cell_i` = 단일 (layout, τ_cls) cell** — 21 cell 각각 산출. | +0.005 임계 적용 |
| max-class 비율 | `max_a (mean_i probs_i[a])` (= soft-mean 의 최대값) | < 0.95 (collapse 회피 임계) |
| fold variance | per-fold metric std | informational only (G3 binding 없음) |

### §3.4 Anchor layouts (7 codebook, 전수 측정)

각 layout 의 좌표 정의 (Frenet, 단위 m). **Anchor index 일관성**: center 가 있는 layout (A1/A2/A3/A4/A7) 의 **index 0 = center**, 이후 layout-specific 패턴. center 없는 layout (A5/A6) 은 index 0 부터 첫 anchor (예: A5 = `(+s,+s,+s)`, A6 = `(+t̂)`). 이 index 매핑이 argmax 분포 cross-layout 비교 시 reference (예: "anchor 0 over-pick" finding 은 center 가 있는 layout 에서만 직접 비교, 없는 layout 은 "index 0 = first anchor" 로 별도 해석).

#### A1: octahedron + center (7) — baseline
```python
ANCHORS_A1 = np.array([
    (  0,      0,      0    ),   # 0: center
    ( +0.005,  0,      0    ),   # 1: +t̂
    ( -0.005,  0,      0    ),   # 2: -t̂
    (  0,     +0.005,  0    ),   # 3: +n̂
    (  0,     -0.005,  0    ),   # 4: -n̂
    (  0,      0,     +0.005),   # 5: +b̂
    (  0,      0,     -0.005),   # 6: -b̂
], dtype=np.float32)
```

#### A2: icosahedron + center (13)
12 icosahedron vertices: 좌표 `(0, ±1, ±φ), (±1, ±φ, 0), (±φ, 0, ±1)` × scale, **scale = 0.005 / √(1 + φ²)**, φ = (1+√5)/2. 모든 12 점 `‖a‖ = 0.005m` exact. 인접 vertex 의 dot product = constant (= Platonic invariant).

```python
PHI = (1 + np.sqrt(5)) / 2  # 1.6180339...
S = 0.005 / np.sqrt(1 + PHI**2)  # scale, ≈ 0.002629
ICO12 = np.array([
    ( 0, +1, +PHI), ( 0, +1, -PHI), ( 0, -1, +PHI), ( 0, -1, -PHI),
    (+1, +PHI, 0), (+1, -PHI, 0), (-1, +PHI, 0), (-1, -PHI, 0),
    (+PHI, 0, +1), (+PHI, 0, -1), (-PHI, 0, +1), (-PHI, 0, -1),
], dtype=np.float32) * S
ANCHORS_A2 = np.vstack([[(0,0,0)], ICO12]).astype(np.float32)  # (13, 3)
```

#### A3: cuboctahedron + center (13)
12 cuboctahedron vertices: `(±a, ±a, 0), (±a, 0, ±a), (0, ±a, ±a)` with `a = 0.005 / √2 ≈ 0.003536`. 모든 12 점 `‖a‖ = 0.005m`. 인접 edge 거리 = `0.005m` exact (FCC neighbor pattern).

```python
A = 0.005 / np.sqrt(2)
CUB12 = np.array([
    (+A, +A,  0), (+A, -A,  0), (-A, +A,  0), (-A, -A,  0),
    (+A,  0, +A), (+A,  0, -A), (-A,  0, +A), (-A,  0, -A),
    ( 0, +A, +A), ( 0, +A, -A), ( 0, -A, +A), ( 0, -A, -A),
], dtype=np.float32)
ANCHORS_A3 = np.vstack([[(0,0,0)], CUB12]).astype(np.float32)  # (13, 3)
```

#### A4: 2-shell octahedron (13)
center + 6 inner axis (`±0.0025m`) + 6 outer axis (`±0.005m`).

```python
ANCHORS_A4 = np.array([
    (0, 0, 0),
    (+0.0025, 0, 0), (-0.0025, 0, 0),
    (0, +0.0025, 0), (0, -0.0025, 0),
    (0, 0, +0.0025), (0, 0, -0.0025),
    (+0.005, 0, 0), (-0.005, 0, 0),
    (0, +0.005, 0), (0, -0.005, 0),
    (0, 0, +0.005), (0, 0, -0.005),
], dtype=np.float32)  # (13, 3)
```

#### A5: cube corners (8) — no center
```python
s = 0.005 / np.sqrt(3)        # ≈ 0.002887, 모든 8 점 ‖a‖ = √(3·s²) = 0.005m
ANCHORS_A5 = np.array([
    (+s, +s, +s), (+s, +s, -s), (+s, -s, +s), (+s, -s, -s),
    (-s, +s, +s), (-s, +s, -s), (-s, -s, +s), (-s, -s, -s),
], dtype=np.float32)
```

#### A6: BCC 14 (axis 6 + corner 8) — no center
```python
# axis 6 = octahedron vertices (A1 의 1..6, center 0 제외)
# corner 8 = A5 의 8 점
ANCHORS_A6 = np.vstack([
    np.array([
        (+0.005, 0, 0), (-0.005, 0, 0),
        (0, +0.005, 0), (0, -0.005, 0),
        (0, 0, +0.005), (0, 0, -0.005),
    ], dtype=np.float32),
    ANCHORS_A5,
])  # (14, 3) — 모든 점 ‖a‖ = 0.005m
```

#### A7: Fibonacci spiral 12 + center (13)
```python
def fib_sphere(N, r):
    phi_g = (1 + np.sqrt(5)) / 2
    theta = 2 * np.pi * np.arange(N) / phi_g            # azimuth
    z = 1 - 2 * (np.arange(N) + 0.5) / N                # latitude (uniform in z)
    rho = np.sqrt(1 - z**2)
    return np.stack([rho * np.cos(theta), rho * np.sin(theta), z], axis=1) * r

ANCHORS_A7 = np.vstack([[(0,0,0)], fib_sphere(12, 0.005)]).astype(np.float32)  # (13, 3)
```

각 layout invariant (`anchors.py` smoke test):
- `ANCHORS_A{n}.dtype == np.float32`
- `np.linalg.norm(ANCHORS_A{n}, axis=1).max() <= 0.005 + 1e-7`
- 좌표값 finite

### §3.5 τ_cls scan

| τ_cls (m) | anchor 0.5cm 대비 | 의미 |
|---|---|---|
| 0.001 | 1/5 | plan-021 default — sharp (q 거의 one-hot) |
| 0.003 | 3/5 | mid — q 가 인접 anchor 에 약간 spread |
| 0.005 | 1/1 | loose — q 가 다중 anchor 에 smooth |

각 (layout, τ) cell 에서 **soft label 계산만 τ_cls 영향** (model hparam / input / loss 동일). soft label 산식:

```python
# build_soft_label_with_tau (분리된 helper, plan-021 build_input.py 의 hard-coded 0.001 우회):
# Args:
#   gt              : (N, 3) float64  — ground-truth Y, world frame, meters
#   R_wfn           : (N, 3, 3) float32/64 — Frenet basis, columns = [t̂, n̂, b̂], world frame
#                     (build_frenet_basis_3d(x, end_idx=10) 결과)
#   pred_F0_world   : (N, 3) float64 — F0 의 80ms 미래 예측, world frame, meters
#                     (= bf.f0_baseline(X, end_idx=10) 출력, plan-021 build_input_common
#                      반환 dict 의 "pred_F0_world" 와 정합)
#   anchors         : (K, 3) float32 — Frenet 좌표, meters (e.g., ANCHORS_A1 ... A7)
#   tau_cls         : float — softmax temperature (m). plan-021 default = 0.001m
# Returns:
#   q : (N, K) float32 — soft label, row-sum = 1
def build_soft_label_with_tau(gt, R_wfn, pred_F0_world, anchors, tau_cls):
    residual = np.einsum('nij,nj->ni', R_wfn.transpose(0,2,1), gt - pred_F0_world)
    dist = np.linalg.norm(anchors[None, :, :] - residual[:, None, :], axis=2)   # (N, K)
    z = -dist / tau_cls
    z = z - z.max(axis=1, keepdims=True)  # numerical stability
    q = np.exp(z); q /= q.sum(axis=1, keepdims=True)
    return q.astype(np.float32)  # (N, K)
```

→ plan-021 `build_soft_label` 와 산식 identical (τ_cls hardcoded → parameterized).

---

## §4. STAGE 0 — 인프라 (G0)

### §4.1 모듈 layout

```
analysis/plan-022/
├── anchors.py                # 7 layout numpy 상수 + smoke (ANCHORS_A1 ... A7)
├── selector_only_model.py    # LgbmSelectorOnly class + build_soft_label_with_tau
├── run_oof.py                # 21-cell sweep runner + paired Δ + best cell
├── paradigm_analysis.py      # c14 G3 entry — collect_cells / select_best / marginals
├── results_A{n}.{json,md}    # 7 sub-exp results (각 3 τ cell 포함)
├── paradigm_analysis.{json,md}  # c14 G3 — 21 cell 표 + marginals + finding (paradigm_analysis.py 산출)
├── baseline_carry.json       # F0 0.6320 / 0.8033 + dataset hash
└── results.md                # G_final synthesis
```

### §4.2 module top-level export (smoke test lock-in)

| symbol | module | type |
|---|---|---|
| `ANCHORS_A1` .. `ANCHORS_A7` | anchors | `np.ndarray` (K, 3), K ∈ {7, 13, 13, 13, 8, 14, 13}, float32 |
| `LAYOUT_NAMES` | anchors | `dict[str, np.ndarray]` — {"A1_octa7": ANCHORS_A1, ...} |
| `LgbmSelectorOnly` | selector_only_model | sklearn-style class — `fit(X, q)` / `predict(X) → probs` (N, K) |
| `build_soft_label_with_tau` | selector_only_model | `Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float], np.ndarray]` |
| `run_oof_cell` | run_oof | `Callable[[X, Y, folds, anchors, tau_cls], dict]` — 단일 cell 5-fold OOF |
| `run_sweep` | run_oof | `Callable[[X, Y, folds], dict[str, dict]]` — 21 cell 전수 |
| `collect_cells` | paradigm_analysis | `Callable[[Path], dict[str, dict]]` — 7 `results_A{n}.json` load → flatten 21 cell dict |
| `select_best` | paradigm_analysis | `Callable[[dict[str, dict]], tuple[str, dict]]` — `(pass_both, Δ_1cm + Δ_1.5cm)` 순으로 best cell tuple `(cell_key, cell_metrics)` 반환 |
| `marginals` | paradigm_analysis | `Callable[[dict[str, dict]], dict]` — layout-axis (7 row) + τ-axis (3 row) best 비교 dict |

→ AttributeError 시 G0 `infra_drift` severe.

### §4.3 plan-021 module reuse

**전제** (G0 smoke 검증): plan-021 `build_input.py` / `dual_head_model.py` 는 **package-relative import 없음** — 모든 import 가 stdlib / 3rd-party (`numpy`, `torch`, `lightgbm`) + plan-020 `baseline_f0.py` (importlib spec_from_file_location 으로 직접 load) 한정. 위 전제 위반 시 G0 `infra_drift` severe (절대-경로 import 만 plan-021 module 안에 허용).

```python
# analysis/plan-022/selector_only_model.py 상단 (= 본 module 위치 = analysis/plan-022/)
import importlib.util, sys
from pathlib import Path

_THIS = Path(__file__).resolve().parent              # analysis/plan-022/
_REPO = _THIS.parent.parent                          # REPO root (analysis/.. = REPO)
_PLAN020 = _THIS.parent / "plan-020"                  # analysis/plan-020/
_PLAN021 = _THIS.parent / "plan-021"                  # analysis/plan-021/
sys.path.insert(0, str(_REPO))                        # plan-021 module 의 transitive `from src.io ...` 보조
sys.path.insert(0, str(_PLAN021))                     # plan-021 module 의 sibling import 보조

# read-only import (path blacklist 예외 — §0.5 references)
# plan-020 baseline_f0 (F0 산식 carry)
spec = importlib.util.spec_from_file_location("bf", _PLAN020 / "baseline_f0.py")
bf = importlib.util.module_from_spec(spec); spec.loader.exec_module(bf)
# bf.f0_baseline, bf.D1, bf.PAR, bf.PERP, bf.R_HIT, bf.R_HIT_LOOSE

# plan-021 build_input + dual_head_model
spec = importlib.util.spec_from_file_location("p021_build", _PLAN021 / "build_input.py")
p021_build = importlib.util.module_from_spec(spec); spec.loader.exec_module(p021_build)
# p021_build.build_input_common, build_input_lgbm_extra, to_frenet, build_frenet_basis_3d

spec = importlib.util.spec_from_file_location("p021_dh", _PLAN021 / "dual_head_model.py")
p021_dh = importlib.util.module_from_spec(spec); spec.loader.exec_module(p021_dh)
# p021_dh.LgbmDualHead — clf_head 부분만 추출
```

→ 위 `bf` 와 `p021_build` 는 §6.2 의 `run_oof_cell` 본체에서 직접 참조 (`bf.f0_baseline`, `p021_build.build_input_common`). `selector_only_model.py` module 의 top-level 에 적재되므로 같은 module 내 import 한 코드가 자동 접근.

### §4.4 `LgbmSelectorOnly` class spec

```python
class LgbmSelectorOnly:
    """plan-021 LgbmDualHead.clf_head 만 추출. reg head 제거.

    Args:
        K (int): anchor count (= classifier output dim)
    """
    def __init__(self, K: int):
        self.K = K
        # plan-021 LgbmDualHead 의 clf head config 정확히 carry
        # multi-class softmax classifier, n_estimators=500, lr=0.05, num_leaves=63
        from lightgbm import LGBMClassifier
        self.clf = LGBMClassifier(
            objective="multiclass", num_class=K,
            n_estimators=500, learning_rate=0.05, num_leaves=63,
            verbose=-1, random_state=20260519,
        )

    def fit(self, X: np.ndarray, q: np.ndarray):
        """X (N, 170), q (N, K) soft label.

        plan-021 LgbmDualHead.fit 의 clf head 학습과 동일 식 — q 를 sample weight
        로 분해해 hard-label expansion (각 anchor 별 weight 부여 후 stack).
        """
        N = X.shape[0]
        # soft → hard expansion — row order = C-order of q (j-major then i):
        # at row j*K + i: X = X[j], y = i, weight = q[j, i] = q.flatten()[j*K + i]
        X_expanded = np.repeat(X, self.K, axis=0)           # (N*K, 170) — X[0]×K, X[1]×K, ...
        y_expanded = np.tile(np.arange(self.K), N)          # (N*K,) — [0..K-1, 0..K-1, ...]
        sample_weight = q.flatten()                          # (N*K,) — C-order: q[j, i] → j*K+i
        mask = sample_weight > 1e-6                          # drop near-zero weights

        # multi-class safety — LightGBM `objective=multiclass, num_class=K` 가 K class 모두
        # 학습 데이터에 존재함을 요구. soft label q 의 anchor 가 매우 sparse 하면 (예: τ=0.001
        # + isotropic 13-anchor) 어떤 class 의 모든 sample 이 mask 아래로 떨어질 risk.
        # 따라서 누락 class 별로 dummy sample 1 개씩 inject (weight = 1e-6):
        present_classes = set(y_expanded[mask].tolist())
        missing = [k for k in range(self.K) if k not in present_classes]
        if missing:
            X_dummy = np.zeros((len(missing), X.shape[1]), dtype=X.dtype)
            y_dummy = np.array(missing, dtype=np.int64)
            w_dummy = np.full(len(missing), 1e-6, dtype=sample_weight.dtype)
            X_fit = np.concatenate([X_expanded[mask], X_dummy], axis=0)
            y_fit = np.concatenate([y_expanded[mask], y_dummy], axis=0)
            w_fit = np.concatenate([sample_weight[mask], w_dummy], axis=0)
        else:
            X_fit, y_fit, w_fit = X_expanded[mask], y_expanded[mask], sample_weight[mask]
        self.clf.fit(X_fit, y_fit, sample_weight=w_fit)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """returns probs (N, K)."""
        return self.clf.predict_proba(X).astype(np.float32)
```

**rationale**: plan-021 `LgbmDualHead.fit` 의 clf head 정확 carry — sample-weight expansion 으로 soft label CE 근사. K 가 7/8/13/14 가변. 동일 input X 위 K 개 class.

### §4.5 tests (c5)

`tests/test_plan022_smoke.py` 8 항목:
1. `import anchors, selector_only_model, run_oof` (AttributeError 0건)
2. 7 layout 각각: `dtype == float32`, `‖a‖ ≤ 0.005m + 1e-7`, shape `(K, 3)` 정합
3. `LAYOUT_NAMES` dict 7 항목, key 정확 (e.g., "A2_ico13")
4. `LgbmSelectorOnly(K=7).fit(rand(100, 170), softmax(rand(100, 7))).predict(rand(50, 170))` — output shape `(50, 7)` + finite + sum=1
5. `build_soft_label_with_tau` 출력 shape `(N, K)` + sum=1 + finite (3 τ 값 모두)
6. `run_oof_cell(X, Y, folds, ANCHORS_A1, 0.001)` 출력 dict 가 다음 키 superset: `{"K", "tau_cls", "hit_1cm", "hit_1.5cm", "delta_1cm", "delta_1.5cm", "max_class_ratio", "fold_var_1cm", "fold_var_1.5cm", "pass_both"}` (모두 finite + 정합)
7. F0 reproduce sanity (plan-020/021 baseline_oof.json carry, 0.6320 / 0.8033)
8. plan-021 build_input.py reuse — `p021_build.build_input_common(X, bf.f0_baseline)` 출력 shape carry 검증

→ 8/8 pass 시 G0 PASS.

---

## §5. STAGE 1 — F0 baseline reproduce (c6, G1)

### §5.1 실행

**호스트**: `analysis/plan-022/baseline_carry.py` (REPO depth = parent.parent.parent ⇒ `analysis/plan-022/X.py` → `analysis/plan-022/` → `analysis/` → `REPO/`).

```python
# analysis/plan-022/baseline_carry.py
import json, hashlib, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent      # depth = 3
sys.path.insert(0, str(REPO))
from src.io import load_all_samples

# (a) plan-020 metric carry
baseline = json.loads((REPO / "analysis/plan-020/baseline_oof.json").read_text())
f0 = baseline["f0_baseline"]
assert 0.6315 <= f0["hit_1cm_5fold_concat"] <= 0.6325, f"f0 1cm drift: {f0['hit_1cm_5fold_concat']}"
assert 0.8028 <= f0["hit_1.5cm_5fold_concat"] <= 0.8038, f"f0 1.5cm drift: {f0['hit_1.5cm_5fold_concat']}"

# (b) dataset hash sanity (plan-020 / 021 와 동일 train.csv 인지)
ids, X = load_all_samples(split="train")
data_hash = hashlib.sha256(("|".join(sorted(map(str, ids)))).encode()).hexdigest()[:16]
# expected_hash 박제 정책: plan-020 baseline_oof.json 안에 `dataset_hash` 필드가 있으면
# 그 값과 비교. 없으면 (legacy) — 본 plan-022 baseline_carry.json 에 신규 박제하고
# plan-023+ 가 carry. 어느 경우든 mismatch → halt (`f0_reproduce_drift` severe).
expected_hash = baseline.get("dataset_hash")
if expected_hash is not None:
    assert data_hash == expected_hash, f"dataset shift: {data_hash} != {expected_hash}"

(REPO / "analysis/plan-022/baseline_carry.json").write_text(json.dumps({
    "f0_hit_1cm": f0["hit_1cm_5fold_concat"],
    "f0_hit_1.5cm": f0["hit_1.5cm_5fold_concat"],
    "dataset_hash": data_hash,
    "n_samples": len(ids),
    "source_baseline": "analysis/plan-020/baseline_oof.json",
}, indent=2))
```

산출: `analysis/plan-022/baseline_carry.json` (plan-020 metric + dataset hash + n_samples).

### §5.2 G1 합격 (자동)

- plan-020 baseline_oof.json metric ±0.0005 안 통과
- 위반 시 `f0_reproduce_drift` severe (= plan-020/021 environment drift)

---

## §6. STAGE 2 — Sub-exp A1~A7 (c7~c13, G2.A1~G2.A7)

### §6.1 Input spec (= plan-021 §6.1, 170D 그대로)

| 채널 | dim | source |
|---|---|---|
| L1 Frenet trajectory | 99 | plan-021 §6.1.1 |
| L2 F0 residual sequence | 21 | plan-021 §6.1.2 |
| L4 F0 soft hit sequence | 14 | plan-021 §6.1.3 |
| L5 macro statistic | 9 | plan-021 §6.1.4 |
| L6 EWMA | 27 | plan-021 §6.1 ewma_last |
| **total** | **170** | |

→ `p021_build.build_input_common(X, bf.f0_baseline)` + `p021_build.build_input_lgbm_extra(X, L1)` 합성. 본 plan code 변경 X.

**170D 구성 정밀**: `build_input_common` → dict 의 `L1 (N,11,9)` / `L2 (N,7,3)` / `L4 (N,7,2)` → reshape 후 99+21+14=134D. `build_input_lgbm_extra` → `(N, 36)` = L5 macro 9D + L6 EWMA 27D 합성 (plan-021 §6.1.4 / ewma_last 산식 carry). 합계 170D. (plan-021 spec 정합 검증은 §4.5 smoke T8 에서 수행.)

### §6.2 Per-cell 5-fold OOF 식

```python
def run_oof_cell(
    X: np.ndarray,             # (N, 11, 3) float64 — raw 11-step world-frame trajectory
    Y: np.ndarray,             # (N, 3)     float64 — ground-truth at t=12 (80ms 미래), world
    folds: np.ndarray,         # (N,)       int     — fold id ∈ {0..4} from stable_fold_id
    anchors: np.ndarray,       # (K, 3)     float32 — Frenet 좌표, K ∈ {7, 8, 13, 14}
    tau_cls: float,            #            float   — softmax temperature, m
) -> dict:
    """단일 (layout, τ) cell 의 5-fold OOF.

    Returns: dict with keys K, tau_cls, hit_1cm, hit_1.5cm, delta_1cm, delta_1.5cm,
             max_class_ratio, fold_var_1cm, fold_var_1.5cm, pass_both
    """
    common = p021_build.build_input_common(X, bf.f0_baseline)
    extra  = p021_build.build_input_lgbm_extra(X, L1=common["L1"])
    N = X.shape[0]
    X_lgbm = np.concatenate([
        common["L1"].reshape(N, 99),
        common["L2"].reshape(N, 21),
        common["L4"].reshape(N, 14),
        extra,
    ], axis=1).astype(np.float32)
    R_wfn = common["R_wfn"]
    pred_F0 = common["pred_F0_world"]
    K = anchors.shape[0]

    pred_world = np.zeros((N, 3), dtype=np.float32)
    probs_all  = np.zeros((N, K), dtype=np.float32)

    for k in range(5):
        tr = np.where(folds != k)[0]
        va = np.where(folds == k)[0]
        q_train = build_soft_label_with_tau(Y[tr], R_wfn[tr], pred_F0[tr], anchors, tau_cls)

        model = LgbmSelectorOnly(K=K).fit(X_lgbm[tr], q_train)
        probs = model.predict(X_lgbm[va])
        probs_all[va] = probs

        # final_frenet = Σ_a probs[a] · anchors[a]   (no reg_offset)
        final_frenet = probs @ anchors                         # (n_va, 3)
        # final_world = R_wfn @ frenet + pred_F0
        pred_world[va] = np.einsum("nij,nj->ni", R_wfn[va], final_frenet) + pred_F0[va]

    # pred_F0 = build_input_common 의 pred_F0_world (= bf.f0_baseline(X, end_idx=10) frenet→world 동일)
    # 본문 §5.1 carry. d_cell / d_f0 모두 world-frame norm.
    d_cell = np.linalg.norm(pred_world - Y, axis=1)        # (N,)
    d_f0   = np.linalg.norm(pred_F0    - Y, axis=1)        # (N,) — common["pred_F0_world"] 재사용
    hit_cell_1   = float((d_cell <= 0.01 ).mean())
    hit_cell_15  = float((d_cell <= 0.015).mean())
    hit_f0_1     = float((d_f0   <= 0.01 ).mean())
    hit_f0_15    = float((d_f0   <= 0.015).mean())
    # per-fold metric — fold k 의 sample mask `folds == k` 위 hit 계산 후 std
    per_fold_1, per_fold_15 = [], []
    for k in range(5):
        m = (folds == k)
        per_fold_1.append(float((d_cell[m] <= 0.01).mean())  if m.any() else 0.0)
        per_fold_15.append(float((d_cell[m] <= 0.015).mean()) if m.any() else 0.0)
    delta_1  = hit_cell_1  - hit_f0_1
    delta_15 = hit_cell_15 - hit_f0_15
    return {
        "K": K, "tau_cls": tau_cls,
        "hit_1cm":  hit_cell_1,
        "hit_1.5cm": hit_cell_15,
        "delta_1cm":  delta_1,
        "delta_1.5cm": delta_15,
        "max_class_ratio": float(probs_all.mean(axis=0).max()),    # collapse 검출 (= soft-mean 의 최대 anchor share)
        "fold_var_1cm":   float(np.std(per_fold_1)),
        "fold_var_1.5cm": float(np.std(per_fold_15)),
        "pass_both": bool(delta_1 >= 0.005 and delta_15 >= 0.005),  # cell-level paired Δ 둘 다
    }
```

### §6.3 Per-sub-exp 실행 (c7~c13)

각 c{i} (i=7..13) 는 하나의 anchor layout 의 3 τ_cls cell 측정:

```python
# c7 예시 (A1_octa7):
results_A1 = {}
for tau in [0.001, 0.003, 0.005]:
    cell = run_oof_cell(X, Y, folds, ANCHORS_A1, tau)
    results_A1[f"tau_{tau:.3f}"] = cell

# JSON 저장: analysis/plan-022/results_A1.json
# MD 저장: analysis/plan-022/results_A1.md (3 cell 표)
```

소요 시간 예상: 단일 cell ≈ 5 분 (plan-021 LGBM 측정 카리). 7 sub-exp × 3 τ = 21 cell × 5 분 = ~105 분 (parallel 안 함, sequential 박제).

### §6.4 G2.A{n} 합격 (per sub-exp)

- 3 cell 모두 metric finite (NaN/Inf 없음)
- 3 cell 모두 `max_class_ratio < 0.95` (soft_label_collapse 회피)
- 위반 시:
  - finite fail → `lgbm_numerical` severe
  - max_class_ratio ≥ 0.95 → `soft_label_collapse` warn (해당 cell drop, 나머지 진행)

---

## §7. STAGE 3 — Paradigm analysis (c14, G3)

### §7.1 21 cell 표 산출

```python
# analysis/plan-022/paradigm_analysis.py
import json
from pathlib import Path

def collect_cells(plan_dir: Path) -> dict[str, dict]:
    """7 results_A{n}.json 을 flatten → 21 cell dict.

    cell_key 포맷: f"{layout_id}_tau{int(tau*1000):03d}"
                   예: "A1_octa7_tau001", "A2_ico13_tau003", "A3_cubocta13_tau005"
    """
    LAYOUT_IDS = ["A1_octa7", "A2_ico13", "A3_cubocta13", "A4_2shell13",
                  "A5_cube8", "A6_bcc14", "A7_fib13"]
    cells: dict[str, dict] = {}
    for layout_id in LAYOUT_IDS:
        sub_exp_key = layout_id.split("_")[0]                  # "A1", "A2", ...
        path = plan_dir / f"results_{sub_exp_key}.json"
        results_sub_exp = json.loads(path.read_text())          # {"tau_0.001": {...}, ...}
        for tau_key, cell in results_sub_exp.items():
            tau_str = tau_key.replace("tau_", "").replace(".", "")  # "0001", "0003", "0005"
            cell_key = f"{layout_id}_tau{tau_str[1:]}"               # drop leading "0"
            cells[cell_key] = {**cell, "layout": layout_id,
                               "tau_cls": float(tau_key.replace("tau_", ""))}
    return cells   # 21 cell, 각 cell dict 의 key = run_oof_cell 반환 schema
```

### §7.2 Best cell selection

```python
def select_best(cells: dict[str, dict]) -> tuple[str, dict]:
    """(pass_both, Δ_1cm + Δ_1.5cm, -cell_key alphabetic) 우선순위로 best cell 선택.

    tie-break 3rd: cell_key 알파벳 순 (= reverse, `-cell_key` 로 sort 시 alphabetic 우선
    cell 선호 — deterministic). 21 cell 모두 동률 시에도 결정성 보장.
    """
    best_key, best_cell = max(cells.items(), key=lambda kv: (
        kv[1]["pass_both"],                                # 우선순위 1: pass_both = True
        kv[1]["delta_1cm"] + kv[1]["delta_1.5cm"],         # 우선순위 2: Δ 합 최대
        # 우선순위 3: cell_key 알파벳 역순 (= "A1_octa7_tau001" < "A7_fib13_tau005")
        # max(...) 에서 큰 키 선호하므로 alphabetic 앞쪽 cell 선호하려면 음수 처리:
        tuple(-ord(c) for c in kv[0]),
    ))
    return best_key, best_cell
```

→ best cell = `(layout, τ_cls)` tuple 박제 → `best_sub_exp` frontmatter.

### §7.3 Marginal 분석

- **layout axis marginal**: 각 layout 의 3 τ_cls cell 중 best τ 선택 → 7 layout 비교 표
- **τ_cls axis marginal**: 각 τ_cls 의 7 layout cell 중 best layout 선택 → 3 τ 비교 표
- **mode collapse 완화 측정**: 각 cell 의 `max_class_ratio` 와 Δ_1cm/1.5cm 의 상관관계

### §7.4 G3 합격

- 21 cell 중 ≥ 1 개가 paired Δ_1cm ≥ +0.005 AND Δ_1.5cm ≥ +0.005 → PASS, band positive
- 0 통과 → `all_negative` warn 박제, band negative, G_final 직진 (paradigm-level evidence 박제)

---

## §8. STAGE 4 — G_final (c15)

### §8.1 산출

- `analysis/plan-022/results.md` (필수 11 항목):
  1. plan-021 → plan-022 narrative bridge
  2. 21 cell 표 (layout × τ_cls grid)
  3. best cell 박제 (layout + τ + metrics)
  4. layout axis marginal
  5. τ_cls axis marginal
  6. mode collapse 완화 finding
  7. plan-021 A LGBM baseline 대비 향상 (또는 negative finding)
  8. paradigm-level finding (anchor lever 의 효과)
  9. follow-up plan 후보 (plan-023, plan-024)
  10. severe/warn 박제 (`all_negative` 발동 여부 등)
  11. dataset hash + reproducibility 박제

- frontmatter sync (plan + results + 본 plan 의 §0.5):
  - `status: all_complete | partial | failed`
  - `best_sub_exp: <layout>_tau<val>` 또는 `null`
  - `band: positive | negative | null`
  - `best_hit_1cm, best_hit_1.5cm, best_delta_1cm, best_delta_1.5cm`
  - `lb_score: null` (out-of-scope)

### §8.2 G_final 합격

- 3-file frontmatter sync (= WORKFLOW.md §11 의 plan ↔ results ↔ registry 3-축 일치) 정확:
  - **file 1**: `plans/plan-022-corrector-free-anchor-layout-sweep.md` 의 frontmatter (status / best_sub_exp / best_*_hit / best_*_delta / band)
  - **file 2**: `plans/plan-022-corrector-free-anchor-layout-sweep.results.md` 의 frontmatter (status / exp_ids_completed / exp_ids_skipped)
  - **file 3**: `analysis/plan-022/results.md` 의 frontmatter (별도 file — narrative + plan/results 와 token 일관)
  - 위 3 file 의 `best_sub_exp`, `band`, `status` 토큰이 정확히 같은 값 (WORKFLOW.md §4 4-way 토큰 일치 carry).
- §0.5 c1~c15 모두 [DONE] (hash 박제)
- `followed_by` 의 plan-023/024 항목 박제 유지

---

## §9. Out of scope (명시적으로 안 함)

- **corrector reg head 재투입** — plan-021 결과로 dead 확인됨. 본 plan-022 = corrector-free 전수 측정. follow-up plan-023 에서 best layout 위 재투입 ablation.
- **GRU sub-exp** — plan-021 의 GRU 결과 (Δ_1cm +0.0073, Δ_1.5cm +0.0067 PASS_BOTH) 와 비교 위해 측정하려면 별도 plan. 본 plan-022 = LGBM only.
- **anchor radius ≠ 0.5cm 의 1D scan** — 단, A4 의 inner shell 0.0025m 은 layout 내부 정의로 OK. radius 자체 axis sweep 은 follow-up plan-023 또는 plan-024.
- **DACON LB 측정 / submit** — train OOF 만 측정. plan-024 follow-up.
- **Ensemble** (plan-020 C05 ⊕ plan-021 GRU ⊕ 본 plan best) — 사용자 명시 ensemble 아님. 단순 비교.
- **27-pool 통합** — plan-024 follow-up.
- **τ_cls < 0.001 또는 > 0.005** — 3 값 박제 (anchor radius 0.5cm 의 1/5, 3/5, 1/1). 외삽 시 follow-up.

---

## §10. 참조 (read-only — path blacklist 예외)

- `analysis/plan-021/build_input.py` — 170D input pipeline (build_input_common, build_input_lgbm_extra, to_frenet, build_frenet_basis_3d)
- `analysis/plan-021/dual_head_model.py` — LgbmDualHead.clf_head 구조 (n_estimators=500, lr=0.05, num_leaves=63)
- `analysis/plan-021/results.md` — selector-only finding, mode collapse evidence
- `analysis/plan-020/baseline_oof.json` — F0 hard evidence 0.6320 / 0.8033
- `analysis/plan-020/baseline_f0.py` — F0 산식 (계수 carry)
- `src/io.py` — load_all_samples, load_labels
- `src/pb_0_6822/selector.py:stable_fold_id` — 5-fold split
- `WORKFLOW.md §5 / §12` — plan 의무 / autonomous protocol
- `CLAUDE.md` — autonomous execution policy (commit/push 의무)
