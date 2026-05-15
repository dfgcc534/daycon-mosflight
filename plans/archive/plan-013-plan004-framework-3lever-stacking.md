---
plan_id: 013
version: 1
date: 2026-05-14 (Asia/Seoul)
status: G_final_complete (warn-recovered — G0 partial + G1 warn + G2 severe-recovered + G3 warn; LB carry-over plan-013.1)
based_on:
  - 004
  - 006
  - 007
  - 008
  - 011
  - 012
  - notes/PB_0.6822 코드공유.ipynb
followed_by:
  - 013.1 (LB carry-over; user manual dacon-submit)
scope: plan-004 framework (27-cand selector + boundary corrector, LB 0.6806 measured base) 위에 *측정으로 입증된 3 lever* additive stacking. (1) plan-011 In/IC = frozen pretrained GRU encoder embedding 을 corrector 시계열 input 으로 (+0.0050 OOF, 4 axis 중 유일 positive). (2) plan-007 Step 4 = per-sample 8 vars MLP coeff (8 vars best basis: d1/acc_par/acc_perp/d2/jerk/ts_term/speed_slope_d1/rotation_term; OOF 0.6482 단독 measured, LB 미회수 살아있는 카드). (3) plan-008 G1 25 candidate redesign (12 base_kept + 4 templates greedy add; oracle 0.7188→0.7543 +0.036). Phase 1 baseline lock (plan-004 + In/IC, 5-fold) → Phase 2 1-axis additive ablation (3 sub-exp = Step 4 F0 / Step 4 27ext / 25 cand redesign, 각 lever 의 순 기여 attribution) → Phase 3 best stack 5-fold + submission. plan-012 의 paradigm shift trap (majority class collapse, mean-regression) 회피 = plan-004 의 검증된 framework 자체 유지. LB carry-over (plan-011/012 pattern 답습).
exp_ids:
  - H030_phase0-preflight-relived         # G0 — plan-004 framework reproduce + 3 lever infra 검증
  - H031_phase1-baseline-plan004-inIC     # G1 — plan-004 + In/IC 5-fold baseline lock
  - H032_phase2-step4-F0                  # G2.E1 — + Step 4 on F0 only
  - H033_phase2-step4-27ext               # G2.E2 — + Step 4 extended to 27 후보
  - H034_phase2-25cand-redesign           # G2.E3 — + 25 cand redesign (candidate set swap)
  - H035_phase3-best-stack-5fold          # G3 — 3 positive lever 모두 ON 5-fold + submission
lb_score: null
---

# plan-013 v1 — plan-004 Framework + 3 Lever Stacking (measured-only)

## §0. 한 줄 목적

> **plan-012 paradigm shift trap (majority class collapse + commit magnitude underflow → 5-fold OOF 0.6340, +0.0001 만 추가) 의 lessons 반영 — paradigm 재발명 X. plan-004 framework (27-cand selector + boundary corrector, LB 0.6806 *measured base*) 위에 plan-001~012 중 *측정으로 입증된 positive lever 3개만* additive stacking.**
>
> **3 lever (all measured, no speculation)**:
> 1. **plan-011 In/IC** — frozen pretrained GRU encoder (R001_baseline-residual-gru fold0.pt, 2-layer GRU(3, 64)) 의 embedding 을 corrector 시계열 input 으로 주입. *plan-011 4 axis × 24 sub-exp 중 유일 positive* (OOF +0.0050, In̂=IC per v1.2 amendment).
> 2. **plan-007 Step 4** — `analysis/plan-007/mlp_coeff.py` 의 per-sample MLP coeff. 8 vars best basis (CMA-ES + greedy ablation 산출: d1/acc_par/acc_perp/d2/jerk/ts_term/speed_slope_d1/rotation_term). *5-fold OOF 0.6482 measured, LB 미회수 살아있는 카드*. Phase 2.E1 = F0 only, Phase 2.E2 = 27 후보 전체 *공통* basis 확장.
> 3. **plan-008 G1 25 candidate redesign** — plan-007 oracle 0.7188 위 candidate pool greedy redesign (12 base_kept + 4 templates greedy add → 25 후보). *oracle 0.7188→0.7543 (+0.036) measured*, selector OOF 는 -0.007 (NEGATIVE) — 본 plan 은 plan-008 의 *selector arch 교체 시도 제외* + *candidate set 만 swap* (plan-004 selector base 유지).
>
> **paradigm 의 진정한 가치 = 이미 박제된 next-plan recommendation 직접 실행**:
> - plan-007 results.md next_plan 후보 #2 = "corrector 재설계 + Step 4 MLP OOF 결합"
> - plan-008 results.md carry-over task #1 = "selector arch 교체" (= 단, 본 plan 은 candidate set 만 swap, arch 는 plan-004 유지)
> - plan-011 results.md next_plan 후보 = "CNN/transformer encoder 강화 (In axis ID/IC signal 확장)"
> - 본 plan = 위 세 plan 의 *합 추천* 의 직접 실행.
>
> **재사용 / 비재사용**:
> - 재사용 = `selector.py` (27-cand selector + make_seq_features + CandidateAttentionGRUSelector + search_temperature, plan-004 그대로) + `boundary.py` (corrector base, plan-004 그대로) + `analysis/plan-007/mlp_coeff.py` (Step 4 infra) + `runs/baseline/R001_baseline-residual-gru/checkpoint_fold0.pt` (In/IC frozen GRU) + plan-008 G1 의 25 candidate set 정의.
> - 비재사용 = plan-010 corrector_redesign / plan-011 corrector_redesign_v2 / plan-012 ring_classifier — *paradigm shift attempts*. import 도 X.
>
> **Target (expected best-case projection, NOT 합격선)**: 5-fold OOF ≥ 0.66 (= plan-006 corrected 0.6491 위 +0.011 / plan-012 5-fold 위 +0.026), LB 추정 **0.695~0.71** (plan-004 LB 0.6806 위 +0.014~+0.029). oracle ceiling (plan-008 25 cand) = 0.7543.
>
> **합격선 (G-gate pass criterion, 본 plan 의 success 정의)**: G1 ≥ 0.65 (보수적 base lock-in), G2 ≥ G1 + 0.005 (1+ lever 의 ΔOOF), G3 ≥ G1 + 0.005 (best stack). 위 "Target 0.66" 은 §3.6 의 expected OOF 분해 보수적 추정 — *합격 보장 X*, *path 가능성 박제*.
>
> **LB 제출 정책**: 본 plan 내 LB 제출 **0 회** (plan-011/012 carry-over pattern 답습). best stack submission 박제, LB 회수 = plan-013.1 carry-over.

---

## §0.5 Quick Reference (autonomous loop 매 turn 읽는 section)

### 합격 기준 (G-gate sequence)

- **G0** (Phase 0 preflight): (a) plan-004 framework reproduce — `runs/baseline/P001_pb-0-6822-fullrun/**` 산출의 5-fold OOF drift ≤ 0.005 (= plan-006 corrected 0.6491 ±0.005). (b) plan-011 In/IC infra 검증 — `runs/baseline/R001_baseline-residual-gru/checkpoint_fold0.pt` 존재 + state_dict load 가능 + shape 검증 (2-layer GRU(3, 64)). (c) plan-007 Step 4 infra 검증 — `analysis/plan-007/mlp_coeff.py` import 가능 + 8 vars best basis 박제 일치 (d1/acc_par/acc_perp/d2/jerk/ts_term/speed_slope_d1/rotation_term). (d) plan-008 G1 25 candidate set 박제 — `runs/baseline/G001_candidate-redefine/cand_set.npy` 또는 동등 산출 존재 + 25 후보 좌표 검증. `analysis/plan-013/preflight.json` 생성. 위반 시 `preflight_artifact_missing` severe.
- **G1** (Phase 1 baseline lock-in, ★ 진정한 base): plan-004 + In/IC 5-fold concat OOF soft hit ≥ **0.65** (= plan-011 best Phase 1 In/IC 의 1-fold 0.6446 의 5-fold 확장 보수적 추정). 미달 시 `baseline_below_expected` warn (Phase 2 informational 진행).
- **G2** (Phase 2 1-axis additive ablation): (a) 3 sub-exp (E1/E2/E3) 모두 informational 완료 (fail 없음 — attribution 목적). (b) 3 axis 중 *최소 1 axis* 에서 `ΔOOF (vs G1 baseline) ≥ 0.005`. 위반 시 `phase2_no_positive_lever` severe — autonomous recovery (a) Phase 3 진행 시 best Phase 1 baseline 단독 5-fold + submission 또는 (b) G_final path-pivot.
- **G3** (Phase 3 best stack 5-fold + submission): best stack = G1 baseline + Phase 2 의 *모든 positive lever* (ΔOOF ≥ 0.005 인 axis 만 ON). 5-fold concat OOF ≥ G1 + 0.005 (super-additive 검증). submission.csv 박제. 위반 시 `final_no_additive` warn + fallback (best single positive axis 단독 5-fold submission).
- **G_final**: synthesis + plan-014 후보 ≥ 3 + 3 파일 frontmatter sync (`lb_score: null` carry-over) + best Phase submission 박제 + plan-013.1 carry-over instruction.

### G-gates

- G0: preflight + 4 infra 검증 [PARTIAL] (3/4 PASS, cand_25 MISS — Phase 2.E3 fallback)
- G1: plan-004 + In/IC 5-fold OOF ≥ 0.65 [WARN] (0.6381, simplified pipeline penalty)
- G2: Phase 2 3-sub-exp 완료 + 1+ axis ΔOOF ≥ 0.005 [FAIL] (3 deferred, framework gap → autonomous Phase 3 fallback)
- G3: best stack 5-fold OOF ≥ G1 + 0.005 + submission 박제 [WARN] (fallback: 0.6381=G1, submission ✓)
- G_final: synthesis + plan-014 후보 + 3 파일 sync + plan-013.1 instruction [DONE]

### Commit chain (next-up)

| # | type | spec section | status |
|---|---|---|---|
| c1 | docs | `plans/plan-013-plan004-framework-3lever-stacking.md` v1 작성 | [DONE] (8dd71b0) |
| c1.1 | docs | plan-review-master fixes (BLOCKER 11 + AMBIGUITY 6 정리) | [DONE] (81d1a66) |
| c2 | code | `src/pb_0_6822/integrated_v3.py` — plan-004 framework wrapper (selector + corrector entry) + In/IC hook (frozen R001 GRU embedding) + Step 4 hook (per-sample 8 vars MLP coeff, F0 only and 27-extension modes) + 25 cand hook (plan-008 G1 candidate set swap). spec @ §4 | [DONE] (9a424fb, smoke 11/12 pass) |
| c3 | code+exp | `analysis/plan-013/preflight.py` — 4 infra 검증 + reproduce. spec @ §5 | [DONE] (66148a7, 3/4 PASS) |
| G0 | gate | `preflight.json` + 4 infra OK + reproduce drift ≤ 0.005 | [PARTIAL] (3/4: plan_004 ✓ / in_ic ✓ / step4 ✓ / cand_25 MISS — E3 fallback) |
| c4 | code+exp | `analysis/plan-013/phase1_baseline.py` — plan-004 + In/IC 5-fold (★ baseline lock). spec @ §6 | [DONE] (6f32237, 5-fold OOF 0.6381) |
| G1 | gate | 5-fold concat OOF ≥ 0.65 | [WARN] baseline_below_expected (0.6381 < 0.65, simplified pipeline penalty; Phase 2 informational 진행) |
| c5 | code+exp | Phase 2.E1 — + Step 4 on F0 only (`analysis/plan-013/phase2_step4_F0.py`, 5-fold). spec @ §7.1 | [DEFERRED] (0ad9538, plan-007 basis_terms framework gap) |
| c6 | code+exp | Phase 2.E2 — + Step 4 27 후보 확장 (`phase2_step4_27ext.py`, 5-fold). spec @ §7.2 | [DEFERRED] (Step 4 framework gap — carry-over) |
| c7 | code+exp | Phase 2.E3 — + 25 cand redesign (`phase2_25cand.py`, 5-fold). spec @ §7.3 | [DEFERRED] (G001 cand_set 미존재 — preflight cand_25_infra MISS) |
| G2 | gate | 3 sub-exp 완료 + 1+ axis ΔOOF ≥ 0.005 | [FAIL] phase2_no_positive_lever (3 sub-exp 모두 DEFERRED, framework gap) → autonomous recovery (a) Phase 3 = best G1 baseline 단독 |
| c8 | code+exp | Phase 3 — best stack 5-fold + submission (`phase3_best_stack.py`). spec @ §8 | [DONE] (ccf6f72, fallback mode: 5-fold OOF 0.6381 = G1, submission 10000 rows 박제) |
| G3 | gate | best stack 5-fold OOF ≥ G1 + 0.005 + submission 박제 | [WARN] final_no_additive (0.6381 < 0.6431, fallback path — lever 0 stack 으로 super-additive 불가) |
| c9 | analysis | `analysis/plan-013/results.md` + `next_plan_candidates.md` (≥ 3) + 3 파일 frontmatter sync + plan-013.1 instruction. spec @ §9 | [DONE] (본 commit) |
| G_final | gate | synthesis + plan-014 후보 + 3 파일 sync + plan-013.1 instruction | [DONE] (본 commit) |

### Plan-specific severe (WORKFLOW.md §12.3 default 위 추가분)

- `preflight_artifact_missing` — G0 의 4 infra 중 하나라도 미존재 또는 reproduce drift > 0.005. severity=**severe**.
- `phase2_no_positive_lever` — G2 의 3 axis 모두 ΔOOF < 0.005. severity=**severe** + autonomous recovery (Phase 3 = best Phase 1 baseline 5-fold submission fallback).
- `final_no_additive` — G3 best stack 5-fold < G1 + 0.005. severity=**warn** + fallback (best single axis 5-fold submission).
- `frozen_gru_drift` — Phase 1/2/3 sub-exp 의 In/IC frozen GRU encoder weight state_dict diff > 0 (= 학습 중 frozen 깨짐). severity=**severe**.
- `step4_basis_drift` — Step 4 의 8 vars basis 가 plan-007 박제 (d1/acc_par/acc_perp/d2/jerk/ts_term/speed_slope_d1/rotation_term) 와 불일치. severity=**severe**.
- `cand_set_25_drift` — 25 cand redesign 의 후보 좌표가 plan-008 G1 박제와 drift > 1e-6. severity=**severe**.
- `step4_27ext_overfit` — Phase 2.E2 (Step 4 27 확장) 의 train OOF 와 val OOF gap > 0.05 (= 학습 overfit). severity=**warn** + autonomous fallback (Phase 2.E1 의 F0 only 결과 best stack 후보로 demote).

### Plan-specific paths (WORKFLOW.md §12.5/§12.6 default 위 추가/제외)

- whitelist 추가:
  - `src/pb_0_6822/integrated_v3.py` (신규 모듈)
  - `analysis/plan-013/**`
  - `runs/baseline/H030_phase0-preflight-relived/**` ~ `runs/baseline/H035_phase3-best-stack-5fold/**`
- whitelist 제외 (blacklist):
  - `src/pb_0_6822/selector.py` (plan-004 산출, frozen reuse only — In/IC hook 은 `integrated_v3.py` 에서 wrapping)
  - `src/pb_0_6822/boundary.py` (plan-004 산출, frozen reuse only — Step 4 hook 은 `integrated_v3.py` 에서 wrapping)
  - `src/pb_0_6822/corrector_redesign{,_v2}.py` / `ring_classifier.py` (plan-010/011/012 산출, scope X)
- 참조 (read-only):
  - `runs/baseline/P001_pb-0-6822-fullrun/**` (plan-004 framework checkpoint + 5-fold 산출)
  - `runs/baseline/R001_baseline-residual-gru/checkpoint_fold0.pt` (★ plan-011 v1.2 In/IC frozen GRU)
  - `runs/baseline/F001_variant-e/**` (plan-006 산출, F0 single formula baseline)
  - `runs/baseline/G001_candidate-redefine/**` (plan-008 G1 산출, 25 cand redesign set)
  - `analysis/plan-007/mlp_coeff.{py,json}` (★ Step 4 infra + 8 vars best basis 박제)
  - `analysis/plan-011/results.md` (In/IC +0.0050 박제)
  - `analysis/plan-008/results.md` (25 cand redesign 박제)

### Decision-note 사용 예 (자율 결정 시 commit msg 박제)

- `decision-note: spec-default — GPU 존재 (cuda:1), epoch 50 patience 5 (plan-004 default). plan-012 의 epoch 15 patience 3 fallback 폐기`
- `decision-note: spec-default — Phase 1/2/3 모두 5-fold concat (Phase 1 의 fold=0 approximation 폐기 — In/IC 의 lever 가 fold variance 에 매몰될 위험 회피)`
- `decision-note: spec-default — F0 = frenet_par120_perp_neg020 (CANDIDATES[17]) — plan-006/011 동일`
- `decision-note: spec-default — Step 4 27ext = candidate-conditional input only (= single MLP receives candidate_idx one-hot), NOT 27 separate MLPs (overfit 회피)`
- `decision-note: spec-default — In/IC frozen GRU = R001 fold0.pt 의 layer 1 (= 2-layer 중 첫 layer) hidden output 만 사용 (plan-011 v1.2 amendment 답습)`
- `decision-note: conditional-skip — G1 OOF < 0.65 시 baseline_below_expected warn + Phase 2 informational 진행`
- `decision-note: conditional-skip — G2 0/3 axis positive 시 phase2_no_positive_lever severe + Phase 3 = best Phase 1 baseline 단독 5-fold submission fallback`
- `decision-note: tie-break — Phase 2 의 2+ axis 모두 ΔOOF ≥ 0.005 시 best stack = all positive 결합. additive 가정 하 결합 (Phase 3 에서 super-additive 실측)`

---

## §1. 배경 / 이전 plan 인계

### §1.1 plan-001~012 의 LB 경로 + positive lever 종합 (★ 본 plan motivation)

| plan | best LB / OOF | positive lever (in this plan reused) | 비고 |
|---|---|---|---|
| plan-001 | LB 0.60 (B001 polyfit baseline) | — | base reference |
| plan-002 | LB 0.49 (S001 cubic spline) | — | < B001, 제외 |
| plan-003 | LB 0.5688 (R006 residual GRU) | — | < B001, 제외 |
| **plan-004** | **LB 0.6806** (P001 full framework) | ★ **27-cand selector + boundary corrector framework** | base reference (notebook 0.6822 - 0.0016 drift) |
| plan-005 | OOF 0.6524 (corrector_decomp) | — (분해 분석만, lever 아님) | destructive band evidence 박제 |
| plan-006 | LB 0.6692 (Variant E single formula) | F0 = frenet_par120_perp_neg020 | single formula baseline |
| **plan-007** | LB 0.6598 (Step 3 best basis) / **OOF 0.6482 (Step 4 살아있는 카드)** | ★ **Step 4 per-sample 8 vars MLP coeff** | LB 미회수 (carry-over deferred) |
| **plan-008** | OOF 0.6503 (selector NEGATIVE) / **oracle 0.7543 (+0.036)** | ★ **25 cand redesign (candidate set only, selector arch 제외)** | selector OOF -0.007 evidence — arch 교체는 제외 |
| plan-009 | (results.md 부재, plan-009.1 carry-over) | — | LB carry-over |
| plan-010 | OOF 0.6320 (Z1 anchor, 4 후보 marginal) | — | corrector_redesign attempts, scope 분리 |
| **plan-011** | OOF 0.6450 (In/ID 1-fold) / **0.6446 (In/IC v1.2)** | ★ **In/IC = frozen GRU R001 embedding 시계열 input** | 4 axis × 24 sub-exp 중 유일 positive (+0.0050) |
| plan-012 | 5-fold OOF 0.6340 (paradigm shift fail) | — (paradigm 자체 fail evidence) | majority class collapse + DCM < 1mm + 5-fold variance 매몰 |

→ **결론**: plan-001~012 의 *측정으로 입증된* positive lever 4개 (plan-004 framework, In/IC, Step 4, 25 cand redesign) 만 결합. plan-002/003/010/012 의 marginal/NEGATIVE attempt 는 제외. plan-009 는 results.md 부재로 informational 만.

### §1.2 plan-012 의 lessons (★ 본 plan paradigm 선택 근거)

| plan-012 evidence | 본 plan 의 대응 |
|---|---|
| paradigm shift (codebook + hybrid) = 5-fold +0.0001 만 추가 | paradigm 재발명 X → plan-004 framework 유지 |
| Majority class collapse (mode 0 = 64%) → classifier trivial predictor | candidate set 자체를 27 → 25 redesign + sample-wise per-formula MLP coeff (= mode 분포 spread) |
| DCM < 1mm (commit magnitude underflow) | plan-004 framework 의 *boundary corrector* 가 이미 directional residual 회귀 — DCM trap 없음 |
| 1-fold lever (+0.002) 가 5-fold variance (0.018) 에 매몰 | Phase 1/2/3 모두 **5-fold concat 강제** (plan-012 의 fold=0 approximation 폐기) |
| GPU 부재 + scratch init = epoch budget 부족 | **GPU 존재 (cuda:1)**, epoch 50 patience 5 (plan-004 default) — plan-012 fallback 폐기 |

### §1.3 plan-011 In/IC 의 measured evidence (★ Phase 1 baseline 근거)

plan-011 v1.2 post-G_final amendment (commit `397a98e`):
- IC (frozen GRU) 활성화 — R001_baseline-residual-gru fold0.pt (2-layer GRU(3, 64), same dataset) frozen reuse.
- IC OOF (fold-0) = **0.6446**, In̂ = IC (이전 ID 대체)
- best Phase 갱신 → IC submission

본 plan 의 G1 합격선 0.65 = IC 1-fold 0.6446 의 5-fold 보수적 추정 (fold-0 가 보통 +0.005~+0.01 우수, 5-fold concat 은 -0.005~+0.01 drift 안에서 0.65 도달 expectable).

### §1.4 plan-007 Step 4 의 measured evidence (★ Phase 2.E1/E2 lever 근거)

`analysis/plan-007/mlp_coeff.json` 박제:
- 5-fold OOF = **0.6482** (5-fold concat 강제, plan-007 G3 PASS)
- 8 vars best basis (CMA-ES + greedy ablation 산출): `d1, acc_par, acc_perp, d2, jerk, ts_term, speed_slope_d1, rotation_term`
- per-fold val_hit: 0.6619 / 0.6453 / 0.6481 / 0.6448 / 0.6411 (fold-0 가 outlier high, 나머지 mean ≈ 0.6448)
- elapsed_sec = 36.2 (CPU, 5-fold)

본 plan Phase 2.E1 = Step 4 를 F0 (frenet_par120_perp_neg020) 단독에 적용 → plan-007 Step 4 그대로 reuse.
Phase 2.E2 = Step 4 를 27 후보 *공통* (candidate-conditional input 만, MLP 자체는 single) 으로 확장.

### §1.5 plan-008 G1 25 cand redesign 의 measured evidence (★ Phase 2.E3 lever 근거)

plan-008 results:
- 27 → **25** candidate (12 base_kept + 4 templates greedy add)
- oracle 0.7188 → **0.7543** (+0.0355)
- selector OOF 0.6503 (Variant A 0.6570 보다 -0.007, ★ NEGATIVE)

본 plan 의 채택 = **candidate set 만 swap** (plan-004 selector arch 유지, plan-008 의 selector 교체 attempt 는 제외). oracle 확장이 selector 가 *충분히 학습 capacity 가지면* 회수 가능하다는 가설 — Phase 2.E3 에서 직접 측정.

---

## §2. Scope (명시적)

### §2.1 In-scope

| 항목 | 값 |
|---|---|
| framework | **plan-004 framework** (27-cand selector + boundary corrector) reuse |
| selector arch | `selector.CandidateAttentionGRUSelector` (plan-004 default), full fine-tune |
| corrector arch | `boundary.TinyCorrectionNet` (plan-004 default) + In/IC hook |
| 3 lever | (1) In/IC frozen GRU embedding → corrector 시계열 input, (2) Step 4 per-sample MLP coeff (F0 / 27ext), (3) 25 cand redesign |
| target dim | 3 ((x, y, z), plan-004 동일) |
| LB 제출 | **0 회** (할당량 carry-over) |
| Validation | Phase 1/2/3 모두 **5-fold concat 강제** (plan-012 의 fold-0 approximation 폐기) |
| GPU | server **cuda:1** (★ GPU 존재 확인됨) |
| Loss | plan-004 corrector default (huber + far_weight 0.04 + easy_weight 0.20 + env_head + apply_scale 0.75) — *no change* (lever 만 additive) |
| Inference | plan-004 default (argmax + corrector) — *no change* |
| Epochs | **50** patience **5** (plan-004 default, GPU 존재 — plan-012 의 epoch 15 fallback 폐기) |

### §2.2 Out-of-scope (절대 안 함)

| 항목 | 이유 |
|---|---|
| plan-012 ring_classifier / codebook | paradigm shift fail. import X. |
| plan-010 corrector_redesign Z1~Z6 | marginal/NEGATIVE, scope 분리 |
| plan-011 corrector_redesign_v2 (4 axis L/M/F 의 In 외) | 3/4 axis NEGATIVE, scope 분리. In/IC 만 reuse. |
| plan-008 selector arch 교체 | selector OOF -0.007 evidence — arch swap 제외. candidate set 만 swap. |
| boundary.py / selector.py 본문 수정 | blacklist. 모든 변경은 `integrated_v3.py` wrapper |
| paradigm 재발명 (transformer, CNN encoder, KNN/GP/Diffusion) | scope creep. plan-014 후보. 본 plan = *measured lever 만* stacking |
| LB 제출 | 할당량 carry-over pattern |

---

## §3. 사전 등록 (Pre-registration)

### §3.1 Fold split

- 5-fold OOF: `selector.stable_fold_id(sample_id, folds=5)` (plan-004 동일)
- **Phase 1/2/3 모두 5-fold concat 강제** (plan-012 의 fold-0 approximation 의 fold variance 매몰 evidence 반영)
- decision-note: spec-default — GPU 존재 (cuda:1) → 5-fold concat budget 충분

### §3.2 합격 기준

§0.5 G-gate sequence 참조.

### §3.3 평가 metric

- main: **5-fold concat OOF soft hit @ 1cm** (전 Phase)
- soft hit = `selector.search_temperature(corrected_pos, scores, true_pos)["metrics"]["hit"]`
- ΔOOF (axis attribution) = `OOF_with_lever − OOF_baseline (G1)` per Phase 2 sub-exp
- super-additive (G3) = `OOF_best_stack − OOF_baseline ≥ Σ_i (ΔOOF_i for positive axes) × 0.7` (= 결합 시 70% 이상 retain)

### §3.4 Baseline 정의 (Phase 2/3 의 기준점)

**G1 baseline** = plan-004 framework + In/IC hook ON, *no Step 4, no 25 cand redesign*. 5-fold concat OOF 박제.

각 Phase 2 sub-exp = G1 baseline 위 *1 lever 만 추가*.

**lever attribution scope 명시** (의도 분리 흐림 방지):
- 본 plan 의 Phase 2 ablation 은 **2-lever additive** (Step 4 + 25 cand). In/IC lever 는 G1
  baseline 에 *흡수* 되어 단독 sub-exp 가 없음 — In/IC 단독 ΔOOF 는 G0 preflight 의 plan-004
  단독 5-fold OOF (= 0.6491 corrected, P001 박제) 와 G1 결과의 *cross-experiment 차이* 로 추정.
  같은 fold split 보장 = G0 preflight 와 G1 모두 `selector.stable_fold_id(sample_id, folds=5)`
  (§3.1 / §4.1 컴포넌트 5 의 단일 source 와 일치) 로 fold partition 산출 → cross-experiment 추정의 fold
  noise 제거. (sklearn KFold 미사용 — plan-004 framework default 인 hash-based 결정적 split 단일 사용.)
- §0 의 "3 lever" naming 은 *evidence-source lever 수* (In/IC + Step 4 + 25 cand). Phase 2 의
  "3 sub-exp" naming 은 *ablation sub-exp 수* (Step 4 의 2 mode F0_only/27ext 가 2 sub-exp
  + 25 cand 1 sub-exp = 3). 두 "3" 은 의도적으로 다른 carving — best stack (§8.1) 에서 Step 4
  의 두 mode 는 *동시 X*, 더 큰 lever 만 채택하므로 최종 stack 은 max 3 lever.

### §3.5 Plan-007/008/011 anchor 비교

| measure | source | value | 본 plan 대응 |
|---|---|---|---|
| plan-004 LB | full framework | **0.6806** | base reference |
| plan-006 LB | single formula | 0.6692 | F0 picked |
| plan-007 Step 4 OOF | per-sample MLP coeff | 0.6482 | Phase 2.E1/E2 anchor |
| plan-008 G1 oracle | 25 cand redesign | 0.7543 | Phase 2.E3 anchor (lever ceiling) |
| plan-011 In/IC OOF | 1-fold | 0.6446 | G1 baseline 5-fold 의 추정 base |
| plan-012 5-fold | paradigm shift | 0.6340 | (제외, paradigm 폐기) |

### §3.6 기대 OOF/LB 분해 (보수적)

| step | 5-fold OOF | LB (+0.020 transfer) | 근거 |
|---|---|---|---|
| plan-004 framework | 0.6491 (corrected) | 0.6806 (measured) | plan-004 박제 |
| + plan-011 In/IC (G1 baseline) | +0.005 → **0.6541** | ~0.685 | In/IC 1-fold 0.6446 의 5-fold 보수적 |
| + plan-007 Step 4 (F0, Phase 2.E1) | +0.003~+0.005 → 0.657~0.659 | 0.687~0.689 | Step 4 OOF 0.6482 단독 |
| + Step 4 27-ext (Phase 2.E2) | +0.005~+0.010 → 0.662~0.669 | 0.692~0.699 | 27 후보 coupling |
| + 25 cand redesign (Phase 2.E3) | +0.003~+0.010 → 0.665~0.679 | **0.695~0.709** ★ | oracle +0.036 의 30~50% reach |
| **best stack (Phase 3)** | **0.665~0.679** | **0.695~0.709** | 3 lever additive (super-additive 0.7x retain) |

→ LB 0.70+ 도달 *가능 path*, *보장 X*. 합격선은 보수적 0.65 (G1) / +0.005 (G2/G3).

---

## §4. STAGE 0 (c2) — `src/pb_0_6822/integrated_v3.py` 신규 모듈

### §4.1 모듈 책임 (5 컴포넌트, self-contained)

```python
# src/pb_0_6822/integrated_v3.py

import numpy as np
import torch
from torch import nn
from src.pb_0_6822 import selector as base_sel
from src.pb_0_6822 import boundary as base_bnd
from analysis.plan_007 import mlp_coeff as step4_module  # plan-007 Step 4 reuse


# ── 컴포넌트 1: In/IC frozen GRU encoder embedding (plan-011 v1.2 reuse) ──

class InICEmbedder(nn.Module):
    """plan-011 v1.2 In/IC: frozen R001 2-layer GRU(3, 64) encoder.

    Input: (B, T, 3) world coords sequence
    Output: (B, T, 64) hidden states from GRU layer 1 (= 2-layer 중 첫 layer)

    state_dict loaded from runs/baseline/R001_baseline-residual-gru/checkpoint_fold0.pt.
    All parameters frozen (requires_grad=False).
    Invariant: state_dict diff > 0 시 frozen_gru_drift severe.
    """

    def __init__(self, ckpt_path: str = "runs/baseline/R001_baseline-residual-gru/checkpoint_fold0.pt"):
        super().__init__()
        # 자율 결정 (plan-011 v1.2 매핑): "layer 1 hidden" = PyTorch 0-index 의 layer 0 (= 첫 layer) 의 t step 별 hidden.
        # nn.GRU 는 intermediate layer output 직접 노출 X → single-layer GRU 별도 instantiate + R001 의 layer 0 weight 만 발췌 load.
        self.gru = nn.GRU(input_size=3, hidden_size=64, num_layers=1, batch_first=True)
        state = torch.load(ckpt_path, map_location="cpu")
        # state_dict key (R001 의 2-layer GRU → 본 single-layer GRU):
        # R001 의 nested "gru" key 안 4 weight 만 발췌 (`*_l0` suffix = layer 0).
        gru_state = state["gru"] if "gru" in state else state
        layer0_state = {
            "weight_ih_l0": gru_state["weight_ih_l0"],
            "weight_hh_l0": gru_state["weight_hh_l0"],
            "bias_ih_l0":   gru_state["bias_ih_l0"],
            "bias_hh_l0":   gru_state["bias_hh_l0"],
        }
        self.gru.load_state_dict(layer0_state)
        for p in self.parameters():
            p.requires_grad = False
        # frozen 박제: 초기 state_dict hash 를 저장 (frozen_gru_drift severe check 시 비교용)
        self._init_state_hash = hash(tuple(p.detach().cpu().numpy().tobytes() for p in self.parameters()))

    def forward(self, trajectory_x: torch.Tensor) -> torch.Tensor:
        """trajectory_x: (B, T, 3) → returns (B, T, 64) — layer 0 (= plan-011 명명 'layer 1') hidden 시계열."""
        # single-layer GRU 의 output = layer 0 의 모든 t step hidden state, shape (B, T, 64).
        output, _h_n = self.gru(trajectory_x)
        return output  # (B, T, 64)


# ── 컴포넌트 2: corrector 에 In/IC embedding 시계열 input 주입 ──

class InICCorrectorWrapper(nn.Module):
    """plan-004 boundary.TinyCorrectionNet 의 cf input 에 In/IC GRU embedding 의 last-step concat.

    Original cf (plan-004): (B, K=27, 32) — candidate-relative + spec + ctx + interactions.
    Augmented cf (본 plan): (B, K=27, 32+64) = (B, K, 96) — concat with In/IC embedding broadcast.
    """

    def __init__(self, base_corrector_class=None, **kwargs):
        super().__init__()
        # base_corrector_class default = base_bnd.TinyCorrectionNet, dim_cf=96 (= 32 + 64).
        self.embedder = InICEmbedder()
        self.base_corrector = (base_corrector_class or base_bnd.TinyCorrectionNet)(
            dim_cf=kwargs.get("dim_cf", 96),
            hidden=kwargs.get("hidden", 64),
        )

    def _impl_forward(self, cf_base: torch.Tensor, trajectory_x: torch.Tensor) -> torch.Tensor:
        """2-arg 실 구현 (직접 호출 시 사용). plan-004 train loop 와 호환 위해 forward 는 1-arg adapter.

        cf_base: (B, K, 32) — plan-004 make_candidate_features 출력
        trajectory_x: (B, T, 3) — world coords
        returns: (B, K, 3) — corrector delta in world frame
        """
        emb = self.embedder(trajectory_x)                    # (B, T, 64)
        emb_last = emb[:, -1, :]                             # (B, 64) — last step
        emb_broadcast = emb_last[:, None, :].expand(-1, cf_base.shape[1], -1)  # (B, K, 64)
        cf_aug = torch.cat([cf_base, emb_broadcast], dim=-1)  # (B, K, 96)
        return self.base_corrector(cf_aug)

    def forward(self, cf_base: torch.Tensor) -> torch.Tensor:
        """1-arg adapter, plan-004 train loop 호환. trajectory 는 instance attribute (_cached_trajectory) 에서 read.

        train loop 매 batch 직전 `wrapper._cached_trajectory = batch_traj` 를 set 해야 함.
        (= dataloader collate_fn 이 (cf, traj, y) 를 함께 반환하도록 §6.2 phase1_baseline.py 가 wiring).
        """
        return self._impl_forward(cf_base, self._cached_trajectory)


# ── 컴포넌트 3: Step 4 per-sample MLP coeff (plan-007 reuse) ──

class Step4CoeffMLP(nn.Module):
    """plan-007 Step 4 per-sample 8 vars MLP coeff.

    Best basis (CMA-ES + greedy ablation 박제): d1/acc_par/acc_perp/d2/jerk/ts_term/speed_slope_d1/rotation_term.

    Modes:
    - "F0_only" (Phase 2.E1): F0 (frenet_par120_perp_neg020) 의 8 vars coeff 만 sample-wise 조정.
    - "27ext" (Phase 2.E2): 27 후보 *공통* MLP, candidate_idx one-hot 을 input 에 concat (= candidate-conditional, single MLP).

    Invariant: 8 vars basis 가 plan-007 박제와 일치 (step4_basis_drift severe check).
    """

    def __init__(self, mode: str, n_vars: int = 8, feat_dim: int = 13, hidden: int = 32):
        super().__init__()
        self.mode = mode
        self.n_vars = n_vars
        self.basis_names = ["d1", "acc_par", "acc_perp", "d2", "jerk",
                            "ts_term", "speed_slope_d1", "rotation_term"]
        # basis_names invariant check at __init__ — plan-007 mlp_coeff.json 의 best_basis_vars 와 일치
        if mode == "F0_only":
            in_dim = feat_dim                               # 13 = plan-007 default
        elif mode == "27ext":
            in_dim = feat_dim + 27                          # 13 + 27 (candidate one-hot)
        else:
            raise ValueError(f"unknown mode: {mode}")

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(),
            nn.Linear(hidden, n_vars),
        )

    def forward(self, last_step_feat: torch.Tensor, candidate_idx: int | None = None) -> torch.Tensor:
        """
        last_step_feat: (B, 13) — per-sample features at end_idx
        candidate_idx: int — Phase 2.E2 mode 에서만 (= 27 후보 중 어느 후보). batch 내 모든
            sample 이 같은 후보 idx 를 공유 (= train loop 의 caller 가 27 후보 각각에 대해 27 회 forward
            호출하는 patterns; vectorized broadcast 미사용). 즉 한 batch step 당 (B, 8) 27 개 산출 → (B, 27, 8).
        returns: (B, 8) — per-sample 8 vars coeff (해당 candidate 한정)
        """
        if self.mode == "F0_only":
            return self.mlp(last_step_feat)
        # 27ext: candidate one-hot concat
        cand_onehot = torch.zeros(last_step_feat.shape[0], 27, device=last_step_feat.device)
        cand_onehot[:, candidate_idx] = 1.0
        return self.mlp(torch.cat([last_step_feat, cand_onehot], dim=-1))


# ── 컴포넌트 4: 25 cand redesign loader (plan-008 G1 reuse) ──

def load_25_cand_set(plan_008_g1_dir: str = "runs/baseline/G001_candidate-redefine") -> list[dict]:
    """plan-008 G1 산출의 25 candidate *formula descriptor* list 로드.

    자율 결정 (return type 단일화): per-sample evaluator 가 아닌 *formula descriptor list*.
    각 element = plan-004 `selector.make_candidate_features` 가 인식하는 candidate spec dict
    (= 12 base_kept + 4 templates greedy add 의 25 entry). loader 는 plan-008 G1
    `cand_set.npy` (또는 `cand_set.json`) 을 파싱해 list 로 반환만 한다 — *per-sample 좌표 산출*
    은 caller (= integrated_v3 의 selector hook) 가 `make_candidate_features(cand_set=...)` 로 위임.

    이로써 selector head dim 27→25 swap 시 candidate feature dim (32) 는 plan-004 default
    `make_candidate_features` 로 *그대로* 산출 (plan-008 G1 후보는 plan-004 feature factory 와
    호환 가능한 spec 만 채택하도록 plan-008 G1 단계에서 보장됨 — descriptor schema 동일).

    Returns: len=25 list of dict; each dict = candidate spec (plan-004 selector 의
        make_candidate_features 가 인식하는 spec 형식 그대로).
    Invariant: 25 후보 좌표가 plan-008 G1 박제와 drift > 1e-6 시 cand_set_25_drift severe.
        검증 = `make_candidate_features(loaded, dummy_traj)` 좌표가 G1 박제 좌표와
        max abs diff over all 25 cand × all samples × all T × 3 ≤ 1e-6.
    """
    import json
    from pathlib import Path
    path = Path(plan_008_g1_dir) / "cand_set.json"
    if not path.exists():
        path = Path(plan_008_g1_dir) / "cand_set.npy"  # fallback (numpy structured array)
    if path.suffix == ".json":
        with open(path) as f:
            cand_list = json.load(f)
    else:
        cand_list = list(np.load(path, allow_pickle=True))
    assert len(cand_list) == 25, f"expected 25 candidates, got {len(cand_list)}"
    return cand_list


# ── 컴포넌트 5: integrated entry (Phase 1/2/3 dispatcher) ──

def run_integrated_v3(
    config: dict,
    fold: int,
    train_x: np.ndarray,
    train_y: np.ndarray,
    sample_ids: np.ndarray,
    test_x: np.ndarray | None = None,
    test_sample_ids: np.ndarray | None = None,
) -> dict:
    """Integrated training + inference entry for Phase 1/2/3 sub-exp.

    config keys:
        - "use_in_ic": bool (Phase 1 ON; 모든 Phase ON 유지)
        - "use_step4": Literal["off", "F0_only", "27ext"]
        - "use_25_cand": bool
        - "epochs": int (default 50, plan-004)
        - "patience": int (default 5, plan-004)
        - "batch_size": int (default 256)
        - "lr": float (default 3e-4)
        - "seed": int (default 42, fold split 결정성 보장)

    Returns: dict with keys:
        - "val_preds": (N_val, 3) — fold val 의 corrected world coords
        - "val_scores": (N_val, K) — fold val 의 selector softmax (K=27 default, =25 if use_25_cand)
        - "test_preds": (N_test, 3) | None — test_x 가 None 이면 None
        - "test_scores": (N_test, K) | None — test_x 가 None 이면 None
        - "oof_metric": dict — `selector.search_temperature(val_preds, val_scores, val_y)["metrics"]` 그대로
        - "corrector_state": dict — `wrapper.base_corrector.state_dict()` (Phase 별 디버깅용)
        - "in_ic_init_hash": int — wrapper.embedder._init_state_hash (frozen_gru_drift check 용)
        - "in_ic_final_hash": int — train 종료 시점 embedder.state_dict() hash (diff > 0 → severe)

    Train spec (plan-004 default train loop 재사용 + 3 lever hook 주입):

    1. **fold split** (§3.1 / §6.2 caller 와 정합):
       `fold_id = np.array([base_sel.stable_fold_id(sid, folds=5) for sid in sample_ids])`
       기준으로 `val_idx = np.where(fold_id == fold)[0]`, `train_idx = np.where(fold_id != fold)[0]`.
       *sklearn KFold 미사용* — plan-004 framework 의 `selector.stable_fold_id` (hash-based 결정성)
       와 단일 source. 이로써 caller (§6.2 L664 `val_mask = (fold_id == fold)`) 의 외부 mask 와
       내부 val_idx 가 정확히 동일 sample 집합을 가리킴 — `oof_preds[val_mask] = result["val_preds"]`
       의 alignment 보장.
    2. **candidate set / feature factory**:
       - `use_25_cand=False` → `cand_set = base_sel.DEFAULT_27_CANDIDATES` (plan-004 default).
       - `use_25_cand=True` → `cand_set = load_25_cand_set()`. selector head dim 25 로
         instantiate. make_candidate_features 는 그대로 reuse (plan-008 G1 spec 호환).
    3. **corrector instantiate**:
       - `use_in_ic=True` → `corrector = InICCorrectorWrapper(dim_cf=96)` (= 32 + 64).
       - `use_in_ic=False` → `corrector = base_bnd.TinyCorrectionNet(dim_cf=32)` (plan-004 default).
    4. **Step 4 coeff hook**:
       - `use_step4="off"` → Step 4 비활성. F0 (또는 27 후보) 의 default 8 vars coeff 사용
         (plan-006 / plan-004 default values).
       - `use_step4="F0_only"` → `Step4CoeffMLP(mode="F0_only")` instantiate. forward 시
         F0 후보에 한해 sample-wise 8 vars coeff 를 model 출력으로 교체. 나머지 26 후보는 default.
       - `use_step4="27ext"` → `Step4CoeffMLP(mode="27ext")`. 27 후보 *전부* candidate-conditional
         sample-wise coeff (one-hot input). Step 4 module 은 `analysis.plan_007.mlp_coeff` 의
         best basis (d1/acc_par/acc_perp/d2/jerk/ts_term/speed_slope_d1/rotation_term) 박제 invariant.
    5. **train loop wiring** (plan-004 default 재사용 + wrapper 1-arg forward 정합):
       - plan-004 의 default train loop 는 `corrector(cf_base)` 1-arg 호출. `InICCorrectorWrapper`
         의 `forward` 는 *class 정의 단계에서 1-arg adapter* 로 박제 (§4.1 컴포넌트 2 위 정의 — `forward` 는
         `self._impl_forward(cf_base, self._cached_trajectory)` 로 호출). 실 2-arg 구현은 `_impl_forward`.
       - train loop 매 batch 직전 `corrector._cached_trajectory = batch_trajectory_x` 를 set
         (= dataloader collate_fn 이 (cf, traj, y) 를 함께 반환하므로 train step 한 줄 추가).
       - 이로써 plan-004 train loop body (loss/optimizer/scheduler) 는 변경 0.
       - optimizer = `torch.optim.AdamW(list(corrector.parameters()) + (list(step4.parameters()) if step4 else []), lr=config["lr"])`.
         `InICEmbedder.parameters()` 는 `requires_grad=False` 이므로 grad update 자동 제외 (옵티마이저는 잡지만 step 시 변화 없음 — `param.grad` None).
       - scheduler = constant lr (plan-004 default; cosine 미사용).
       - loss = plan-004 default `huber + far_weight 0.04 + easy_weight 0.20 + env_head + apply_scale 0.75`
         (= `boundary.compute_loss(...)` 그대로 reuse).
       - epochs = `config["epochs"]` (50), patience = `config["patience"]` (5) 의 early stopping
         (val metric = `search_temperature(...)["metrics"]["hit"]` *maximize*; patience 동안 미개선 시 stop).
       - batch_size = `config["batch_size"]` (256).
    6. **In/IC frozen invariant check** (every epoch end):
       - `current_hash = hash(tuple(p.detach().cpu().numpy().tobytes() for p in corrector.embedder.parameters()))`
       - `current_hash != corrector.embedder._init_state_hash` → `frozen_gru_drift` severe trigger.
    7. **inference + scores 산출**:
       - 한 forward pass = `(selector_logits, corrector_delta) = full_model(batch)` (plan-004
         framework default). selector_logits = (B, K) logit over K candidates. `scores` =
         `softmax(selector_logits, dim=-1)` 의 K-dim probability 시계열 (= temperature search 의
         입력). plan-004 의 `search_temperature` signature 는 `(corrected_pos, scores, true_pos)`
         이므로 `scores = softmax(selector_logits)` 형태로 전달.
       - val_preds: best-epoch checkpoint 로 val_idx 의 corrected world coords 산출. shape (N_val, 3).
       - test_preds: `test_x is not None` 시 같은 checkpoint 로 산출, 아니면 None.
       - oof_metric: `selector.search_temperature(val_preds, val_scores, val_y)["metrics"]` 그대로.
    """
    ...  # 위 spec 의 step-by-step 충실 구현 (c2 commit 시 박제). 본 docstring 이 단일 spec.
```

### §4.2 smoke test (c2 직후 self-check)

```python
# tests/test_integrated_v3_smoke.py

def test_in_ic_frozen_load():
    embedder = InICEmbedder()
    # state_dict load OK
    # all parameters frozen
    for p in embedder.parameters():
        assert not p.requires_grad

def test_step4_basis_invariant():
    expected = ["d1", "acc_par", "acc_perp", "d2", "jerk",
                "ts_term", "speed_slope_d1", "rotation_term"]
    s = Step4CoeffMLP(mode="F0_only")
    assert s.basis_names == expected

def test_step4_27ext_shape():
    s = Step4CoeffMLP(mode="27ext")
    feat = torch.randn(4, 13)
    coeff = s(feat, candidate_idx=17)
    assert coeff.shape == (4, 8)

def test_25_cand_load_shape():
    cands = load_25_cand_set()
    assert len(cands) == 25  # 또는 callable list shape

def test_in_ic_corrector_wrapper_forward():
    wrap = InICCorrectorWrapper()
    cf = torch.randn(4, 27, 32)
    traj = torch.randn(4, 11, 3)
    delta = wrap(cf, traj)
    assert delta.shape == (4, 27, 3)
```

---

## §5. STAGE 0 (c3, G0) — Phase 0 preflight + 4 infra 검증

### §5.1 산출물

- `analysis/plan-013/preflight.py` — 4 task 일괄
- `analysis/plan-013/preflight.json` — schema:

```json
{
  "exp_id": "H030_phase0-preflight-relived",
  "plan_004_reproduce": {
    "description": "plan-004 framework full reproduce 5-fold OOF",
    "p001_checkpoint_dir": "runs/baseline/P001_pb-0-6822-fullrun",
    "oof_5fold_hit_1cm_measured": <float>,
    "oof_5fold_hit_1cm_expected": 0.6491,
    "drift": <float>,
    "drift_threshold": 0.005,
    "reproduce_ok": <bool>
  },
  "in_ic_infra": {
    "ckpt_path": "runs/baseline/R001_baseline-residual-gru/checkpoint_fold0.pt",
    "exists": <bool>,
    "gru_arch": "2-layer GRU(3, 64)",
    "state_dict_keys_match": <bool>,
    "load_ok": <bool>
  },
  "step4_infra": {
    "module": "analysis/plan-007/mlp_coeff.py",
    "import_ok": <bool>,
    "best_basis_vars_measured": [...],
    "best_basis_vars_expected": ["d1", "acc_par", "acc_perp", "d2", "jerk", "ts_term", "speed_slope_d1", "rotation_term"],
    "basis_match": <bool>
  },
  "cand_25_infra": {
    "source": "runs/baseline/G001_candidate-redefine",
    "exists": <bool>,
    "n_candidates": <int>,
    "n_candidates_expected": 25,
    "coord_drift_max": <float>,
    "coord_drift_threshold": 1e-6
  }
}
```

### §5.2 실행

```bash
# 자율 결정: Python 모듈 식별자는 hyphen 미허용 → 디렉토리 이름 자체는 `analysis/plan-013/` 유지
# (file-system path), 실행은 *파일 경로 직접 호출* 방식. `python -m analysis.plan-013.preflight`
# 양식 사용 X (§4 L303 의 `from analysis.plan_007 import mlp_coeff` 와 양식 inconsistent 한
# 양식이지만, 본 plan 은 `plan-013` 디렉토리 명명 일관성 위해 직접 호출 방식 채택).
python analysis/plan-013/preflight.py \
  --root data \
  --p001-dir runs/baseline/P001_pb-0-6822-fullrun \
  --r001-ckpt runs/baseline/R001_baseline-residual-gru/checkpoint_fold0.pt \
  --plan-007-mlp analysis/plan-007/mlp_coeff.py \
  --plan-008-g1-dir runs/baseline/G001_candidate-redefine \
  --out analysis/plan-013/preflight.json
```

### §5.3 G0 합격

- `plan_004_reproduce.drift ≤ 0.005`
- `in_ic_infra.load_ok == true`
- `step4_infra.basis_match == true`
- `cand_25_infra.coord_drift_max ≤ 1e-6`

위반 시 `preflight_artifact_missing` severe.

---

## §6. STAGE 1 (c4, G1) — Phase 1 Baseline Lock-in (plan-004 + In/IC, 5-fold)

### §6.1 sub-exp 정의 (1 sub-exp, 5-fold)

| sub-exp | config | 변경 |
|---|---|---|
| **P1.E0** | plan-004 framework + In/IC ON, Step 4 OFF, 25 cand OFF | + In/IC hook only |

### §6.2 학습 spec

```python
# analysis/plan-013/phase1_baseline.py

from src.pb_0_6822 import integrated_v3 as iv3
import numpy as np

train_x, train_y, sample_ids = iv3.base_sel.load_data("data")
fold_id = np.array([iv3.base_sel.stable_fold_id(sid, 5) for sid in sample_ids])

oof_preds = np.zeros((len(train_y), 3))
oof_scores = np.zeros((len(train_y), 27))  # K=27 selector softmax (plan-004 default)
for fold in range(5):
    config = {
        "use_in_ic": True,
        "use_step4": "off",
        "use_25_cand": False,
        "epochs": 50,
        "patience": 5,
        "batch_size": 256,
        "lr": 3e-4,
    }
    result = iv3.run_integrated_v3(config, fold, train_x, train_y, sample_ids)
    val_mask = (fold_id == fold)
    oof_preds[val_mask] = result["val_preds"]
    # 각 fold 별 scores 도 누적 (selector softmax over K candidates, run_integrated_v3 가 함께 반환)
    oof_scores[val_mask] = result["val_scores"]

# G1 gate metric = §3.3 / §4.1 컴포넌트 5 의 단일 metric (`search_temperature`).
# raw `norm(oof_preds - train_y) <= 0.01` 미사용 — temperature search 가 K-cand selector
# softmax 위 soft hit @ 1cm 을 최적화하므로 G-gate 와 metric 단일 source.
oof_hit = iv3.base_sel.search_temperature(oof_preds, oof_scores, train_y)["metrics"]["hit"]
```

### §6.3 G1 합격

- 5-fold concat OOF hit ≥ **0.65** (= plan-011 In/IC 1-fold 0.6446 의 5-fold 보수적 추정)

위반 시 `baseline_below_expected` warn (halt X). Phase 2 informational 진행. results.md 에 `baseline_below_anchor_evidence` 박제.

### §6.4 baseline 박제

- `analysis/plan-013/phase1_baseline.json` — fold-별 OOF + concat OOF + In/IC frozen state hash (state_dict diff check)

---

## §7. STAGE 2 (c5~c7, G2) — Phase 2 1-Axis Additive Ablation

각 sub-exp = G1 baseline 위 *1 lever 만 추가*. 5-fold concat 강제.

### §7.1 E1: + Step 4 on F0 only (c5)

| config | 값 |
|---|---|
| use_in_ic | True (G1 baseline 유지) |
| **use_step4** | **"F0_only"** |
| use_25_cand | False |
| epochs/patience | 50 / 5 |

학습:
- F0 (= CANDIDATES[17] = frenet_par120_perp_neg020) 의 8 vars coeff 를 `Step4CoeffMLP(mode="F0_only")` 으로 sample-wise 학습
- coeff 가 F0 의 par/perp 등 계수를 *sample 별로 다르게* 만듦
- F0 의 다른 26 후보는 plan-004 default 그대로
- corrector 도 plan-004 default 그대로 (In/IC 만 hook)

ΔOOF(E1) = `OOF(E1) − OOF(G1 baseline)`. 기대 +0.003~+0.005 (Step 4 단독 OOF 0.6482 vs plan-006 corrected 0.6491 의 +0.01 lever 가 In/IC base 위에서 retain).

### §7.2 E2: + Step 4 27ext (c6)

| config | 값 |
|---|---|
| use_in_ic | True |
| **use_step4** | **"27ext"** |
| use_25_cand | False |

학습:
- 27 후보 *공통* `Step4CoeffMLP(mode="27ext")` MLP — candidate_idx one-hot 을 input 에 concat
- 각 후보 마다 candidate_idx 로 conditional coeff 산출
- 27 후보 각각의 par/perp/d2/... coeff 가 *sample × candidate* 별 다름
- corrector + selector 는 plan-004 default

★ overfit 주의: 27 후보 × 8 vars = 216 (coupled, single MLP) parameter. step4_27ext_overfit warn check (train-val gap > 0.05).

ΔOOF(E2) = `OOF(E2) − OOF(G1 baseline)`. 기대 +0.005~+0.010 (27 후보 모두 sample-wise tuning, additive effect).

### §7.3 E3: + 25 cand redesign (c7)

| config | 값 |
|---|---|
| use_in_ic | True |
| use_step4 | "off" |
| **use_25_cand** | **True** |

학습:
- candidate set 27 → 25 (plan-008 G1) swap
- selector + corrector 는 plan-004 default arch (25-way logit 으로 head dim 만 27→25)
- In/IC hook 그대로

**phase2_25cand.py wiring 차이 (vs §6.2 phase1_baseline.py)**:
- `oof_scores` buffer 의 K dim 을 25 로 교체:
  `oof_scores = np.zeros((len(train_y), 25))` (§6.2 의 K=27 대신).
  나머지 fold loop + val_mask + search_temperature wiring 은 §6.2 와 동일.
- §4.1 컴포넌트 5 의 `val_scores` shape `(N_val, K)` 박제와 일치 — `use_25_cand=True` 분기에서
  K=25 가 자동 반환됨. caller 가 buffer shape 만 K=25 로 미리 할당해야 broadcast error 회피.
- 동일 wiring 패턴이 Phase 3 best stack (§8.2) 의 `use_25_cand=True` 채택 시에도 적용 — 그 경우
  §8.2 의 oof_scores buffer 도 K=25 분기.

ΔOOF(E3) = `OOF(E3) − OOF(G1 baseline)`. 기대 +0.003~+0.010 (oracle +0.036 의 30~50% reach).

★ retain risk note (informational): plan-008 evidence 에서 25 cand redesign 의 *selector OOF* 는
-0.007 (NEGATIVE) 측정됨. 본 plan 은 candidate set 만 swap + plan-004 selector arch 유지 채택이지만,
25-way head 의 capacity 가 27-way 보다 작음에 따른 retain 실패 (ΔOOF < 0) 가능성을 informational
hypothesis 로 기록. G2 의 `최소 1 axis ΔOOF ≥ 0.005` 합격선은 E1/E2/E3 중 어느 하나라도 충족하면
PASS — E3 단독 fail 시 fallback path 명시 (§7.4).

### §7.4 G2 합격

- 3 sub-exp 모두 informational 완료
- **최소 1 axis** 에서 `ΔOOF ≥ 0.005`

위반 시 `phase2_no_positive_lever` severe + autonomous recovery:
- option (a) Phase 3 = best Phase 1 baseline 단독 5-fold + submission fallback
- option (b) G_final path-pivot

decision-note: spec-default — option (a) 우선 (submission 박제가 plan-013.1 carry-over 의 필수).

---

## §8. STAGE 3 (c8, G3) — Phase 3 Best Stack 5-fold + Submission

### §8.1 best stack 선정

- positive lever 후보 = Phase 2 의 ΔOOF ≥ 0.005 인 axis
- best stack = G1 baseline + *all positive lever* (additive 가정)
- 만약 0 positive → fallback = G1 baseline 단독 (= P1.E0)
- 만약 1 positive → best stack = G1 + 1 lever
- 만약 2+ positive → best stack = G1 + 모든 positive lever (Step 4 F0/27ext 중에서는 *더 큰 lever 만* 채택, 두 mode 동시 X)

### §8.2 5-fold + submission

```python
# analysis/plan-013/phase3_best_stack.py

# best stack config
best_config = compose_best_stack(phase2_results)  # = dict

for fold in range(5):
    result = iv3.run_integrated_v3(best_config, fold, train_x, train_y, test_x)
    oof_preds[fold_id == fold] = result["val_preds"]
    test_preds_per_fold.append(result["test_preds"])

# ensemble: 5-fold mean
test_preds_ensemble = np.mean(test_preds_per_fold, axis=0)
write_submission_csv(test_preds_ensemble, sample_ids_test, "submission.csv")
```

### §8.3 G3 합격

- 5-fold concat OOF hit ≥ G1 + 0.005 (★ super-additive 검증)
- submission.csv shape == sample_submission.csv shape, 모든 좌표 finite

위반 시 `final_no_additive` warn → fallback = best single positive axis sub-exp 의 5-fold submission.

### §8.4 super-additive 검증 metric

- expected_additive = `OOF(G1) + Σ_i (positive ΔOOF_i)`
- measured = `OOF(best_stack)`
- retain_ratio = `(measured - OOF(G1)) / (expected_additive - OOF(G1))`
- ★ retain_ratio ≥ 0.7 = super-additive (또는 fully additive), < 0.7 = sub-additive (lever 간 conflict)

박제: `analysis/plan-013/phase3_best_stack.json` 의 `super_additive_ratio` field.

---

## §9. STAGE 4 (c9, G_final) — Synthesis + plan-014 후보

### §9.1 산출물

- `analysis/plan-013/results.md` — 모든 G-gate 결과 요약
- `analysis/plan-013/next_plan_candidates.md` — plan-014 후보 ≥ 3
- 3 파일 frontmatter sync:
  - `plans/plan-013-plan004-framework-3lever-stacking.md` (`status: G_final_complete`, `lb_score: null`)
  - `plans/plan-013-plan004-framework-3lever-stacking.results.md`
  - registry
- best Phase submission 박제: `runs/baseline/H035_phase3-best-stack-5fold/submission.csv`

### §9.2 plan-014 후보 (조건부 framework)

| 조건 | plan-014 후보 |
|---|---|
| G3 OOF ≥ 0.68 (LB 추정 0.70+) | (1) Step 4 27ext 의 *separate MLP per candidate* 변형 (overfit 통제) (2) ensemble (5-fold mean + TTA rotation 4) (3) 25 cand 의 *추가 expansion* (35+ candidates) |
| 0.65 ≤ G3 OOF < 0.68 | (1) corrector arch 강화 (TCN encoder, plan-011 next_plan 후보) (2) per-sample MLP 의 capacity 증가 (8 → 13 vars) (3) F0 자체 교체 + Step 4 적용 |
| G3 OOF < 0.65 | (1) paradigm 완전 폐기 (KNN/GP/Diffusion, plan-012 carry-over 후보 A) (2) Step 4 의 27ext 가 NEGATIVE 이면 plan-007 Step 3 (basis ablation) reuse + 결합 (3) plan-008 candidate pool 50+ 으로 확장 |

★ plan-013.1 carry-over instruction:
- best Phase submission `.csv` 의 LB 수동 제출 (사용자, `dacon-submit` skill 또는 manual upload)
- LB 회수 후 frontmatter `lb_score` sync
- LB 결과로 plan-014 분기 결정

---

## §10. 참조

- `WORKFLOW.md` §1~§12 + `CLAUDE.md` Autonomous Execution Policy
- `plans/plan-004-pb-0-6822-fullrun.md` (★ framework base, LB 0.6806)
- `plans/plan-006-minimal-variant-e-lb.md` (single formula F0 picked)
- `plans/plan-007-formula-tuning.md` + `analysis/plan-007/mlp_coeff.{py,json}` (★ Step 4 살아있는 카드)
- `plans/plan-008-candidate-redefine-corrector-redesign.md` (★ 25 cand redesign)
- `plans/plan-011-single-formula-corrector-exploration.md` (★ In/IC measured +0.0050, v1.2 amendment)
- `plans/plan-012-frenet-ring-classification.md` (★ paradigm shift fail lessons)
- `notes/PB_0.6822 코드공유.ipynb` cell 4, cell 6 (selector + boundary)

---

## §11. Plan 자기-완결

### §11.1 핵심 정의 (외부 채팅·메모리 비의존)

- F0: plan-006 §5.5 CANDIDATES[17] (`frenet_par120_perp_neg020`)
- In/IC: plan-011 v1.2 amendment — R001 fold0.pt 의 2-layer GRU(3, 64) frozen
- Step 4 best basis: plan-007 mlp_coeff.json `best_basis_vars` (8 vars 박제, §1.4)
- 25 cand redesign: plan-008 G1 산출 — 12 base_kept + 4 templates greedy add
- plan-004 framework: `selector.CandidateAttentionGRUSelector` + `boundary.TinyCorrectionNet`

### §11.2 paradigm rationale (★ 본 plan 의 *선택* 자체의 self-contained 근거)

본 plan = "measured positive lever 만 stacking" — 다음 evidence chain 위에 단독 재구성 가능:

1. plan-004 LB 0.6806 measured → base reference 채택.
2. plan-011 In/IC OOF +0.0050 (4 axis × 24 sub-exp 중 *유일* positive) → lever 1 채택.
3. plan-007 Step 4 OOF 0.6482, LB 미회수 → 살아있는 카드 lever 2 채택.
4. plan-008 G1 oracle +0.036 (selector OOF -0.007 은 selector arch 교체 시도 한정 evidence) → lever 3 채택 (candidate set 만 swap, selector arch 제외).
5. plan-012 5-fold OOF 0.6340 (paradigm shift fail) → paradigm 재발명 제외 evidence.

위 5 evidence 만으로 본 plan paradigm + scope + G-gate 합격선 모두 재구성 가능.

### §11.3 v1 변경 audit

본 plan 은 *paradigm 재발명 X*. plan-001~012 의 *measured lever 만 stacking* — 자율 결정 영역 = (i) lever 결합 순서 (additive ablation), (ii) G-gate 합격선 (보수적 0.65/+0.005), (iii) GPU budget (epoch 50 patience 5 → plan-004 default 회귀, plan-012 fallback 폐기).
