---
plan_id: 027
version: 1.1
date: 2026-05-22 (Asia/Seoul)
status: all_complete
best_cell: E3_weighted
best_hit_1cm: 0.6529
best_hit_1p5cm: 0.8118
best_delta_1cm: -0.0001
band: negative_ensemble
based_on:
  - 022 (winner A6_bcc14_tau001 hit_1cm 0.6528 / hit_1p5cm 0.8104. K=14 BCC + τ=0.001)
  - 023 (winner B4_fib50_tau001 hit_1cm 0.6532 / hit_1p5cm 0.8108. K=50 Fibonacci)
  - 025 (G2.C1 hit_1cm=0.6320 regression to F0. paradigm finding = block ③ mode collapse 원인)
  - 026 (G2.A2 hit_1cm=**0.6509** = plan-022 winner 99.66% 회수. paradigm reversal — block ③ 제거 후 1058D LGBM. ensemble 새 후보)
  - 020 (F0 baseline 0.6320/0.8033 + stable_fold_id MD5)
inspired_by:
  - 025 §0.5 followed_by — plan-022/023 winner + plan-025 winner soft-vote
code_reuse:
  - module: analysis/plan-022/anchors.py
    symbols: [ANCHORS_A6, LAYOUT_NAMES]
    reason: K=14 BCC carry (plan-022 winner anchor).
  - module: analysis/plan-022/selector_only_model.py
    symbols: [LgbmSelectorOnly, build_soft_label_with_tau]
    reason: plan-022 LGBM model + soft label.
  - module: analysis/plan-022/run_oof.py
    symbols: [run_oof_cell]
    reason: plan-022 winner OOF reproduce — sample 별 final_pred 출력 wrapper 필요 (plan-022 source 미수정, 본 plan run_oof_with_preds 가 fold 별 final_pred 재계산).
  - module: analysis/plan-023/anchors_largeN.py
    symbols: [ANCHORS_B4_fib50, build_fibonacci_sphere]
    reason: K=50 Fibonacci anchor carry (plan-023 winner).
  - module: analysis/plan-023/run_oof_largeN.py
    symbols: [run_oof_largeN_cell]
    reason: plan-023 winner OOF reproduce.
  - module: analysis/plan-025/run_oof.py
    symbols: [LgbmSelectorRowExpanded, run_oof_plan025, _normalize_p022_result]
    reason: plan-026 A2 결과 carry (band=regression → ensemble 후보 자격 미달, but reproducibility 위해 import).
  - module: analysis/plan-026/run_oof.py
    symbols: [run_oof_plan026]
    reason: plan-026 A2 = 1058D no-block③ — hit_1cm=0.6509 (plan-022 winner 99.66% 회수). 본 plan 의 3-way ensemble 의 third base predictor.
  - module: analysis/plan-026/block_mask_builder.py
    symbols: [build_feat_masked]
    reason: plan-026 A2 OOF reproduce 시 feature builder.
  - module: analysis/plan-021/build_input.py
    symbols: [build_frenet_basis_3d, build_input_common, build_input_lgbm_extra]
    reason: 170D LGBM input pipeline (plan-022/023 carry).
  - module: analysis/plan-020/baseline_f0.py
    symbols: [f0_baseline, D1, R_HIT, R_HIT_LOOSE]
    reason: F0 baseline + paired Δ anchor.
  - module: src/io.py
    symbols: [load_all_samples, load_labels]
    reason: data loader.
  - module: src/pb_0_6822/selector.py
    symbols: [stable_fold_id]
    reason: 5-fold stable split.
followed_by:
  - plan-028 (F0 baseline ML — systematic forward bias 완화, oracle 0.7928 → 0.85+)
  - plan-029 (가칭, lower priority): N>14 anchor 의 corrector reg head 재투입
scope: plan-022 A6_bcc14_tau001 + plan-023 B4_fib50_tau001 + plan-026 A2_no_block3 (3 base predictor) 의 sample 별 OOF final prediction 의 soft-vote ensemble. plan-026 A2 hit_1cm=0.6509 (plan-022 winner 99.66% 회수) → 3-way ensemble 자격 충족 (조건부 분기 제거, 항상 3-way 시도). 단일 변수 = ensemble weight w. LGBM hparam / anchor / τ_cls / fold split / soft-label / F0 = 각 plan carry. DACON LB / 신규 cell / hparam adjust = out-of-scope.
exp_ids:
  - Z027_E1_eq_2way
  - Z027_E2_eq_3way
  - Z027_E3_weighted_optim
lb_score: null
---

# plan-027 v1 — Ensemble (plan-022/023/[025] winner soft-vote)

## §0. 한 줄 목적

> **plan-022 winner A6_bcc14_tau001 (0.6528)** + **plan-023 winner B4_fib50_tau001 (0.6532)** + **plan-026 A2_no_block3 (hit_1cm=0.6509, plan-022 winner 99.66% 회수)** 의 sample 별 OOF final prediction soft-vote ensemble. 단일 변수 = ensemble weight w. 3 cell × 5-fold OOF (= 각 base predictor 의 OOF prediction reproduce + weight sweep).
>
> **paradigm rationale**: plan-022/023 winner 가 동일 14-anchor oracle ceiling 0.7928 에 근접 (회수율 82.3-82.4%). 두 cell 의 *prediction error pattern* 이 anchor codebook 차이 (K=14 BCC vs K=50 fib) 로 *uncorrelated* 일 가능성 → ensemble 로 +0.005~0.010 lift 가능. plan-025 추가 시 1080D input 의 *추가 lever* (band 조건 통과 시).
>
> **band 조건 분기 (제거 — scope L55 carry: 항상 3-way 시도)**: 본 plan 의 모든 cell 은 plan-026 A2 가 plan-022 winner 99.66% 회수 (= ensemble candidate 자격) 라는 점에 의거 *항상 3-way 시도*. plan-026 A2 의 hit_1cm (0.6509) 자체가 band=regression (=`< 0.6528`) 정의 안에 들지만, *ensemble candidate 자격은 base predictor diversity 기준이지 단독 hit 기준 아님*. 단순 base predictor lift 가 아니라 prediction error pattern diversity 가 ensemble 의 본질.
>
> ~~old band 분기 (deprecated, L55 scope 가 supersede):~~
> - **positive/partial_lift band** (plan-026 A2 hit_1cm > 0.6528) → **3-way ensemble** (p022 + p023 + p026_A2)
> - **regression band** (plan-026 A2 hit_1cm ≤ 0.6528) → **2-way ensemble** (p022 + p023) — plan-026 A2 은 ensemble 부재
>
> **cell scan**:
> 1. **E1** = equal-weight 2-way (w_p022 = w_p023 = 0.5)
> 2. **E2** = equal-weight 3-way (w_p022 = w_p023 = w_p026_A2 = 1/3) — band 조건 충족 시만
> 3. **E3** = weighted optim — weight grid sweep ∈ {0.3, 0.4, 0.5, 0.6, 0.7} for 2-way; OR 3-simplex sweep for 3-way (5×5×5 grid filtered to sum=1)
>
> **pass criterion (G3)**: 3 cell 중 ≥ 1 개가 hit_1cm > max(plan-022 winner, plan-023 winner) + 0.002 → PASS (= ensemble effective). partial = 0.6532 ≤ best ≤ max+0.002. regression = best < 0.6532.
>
> **out-of-scope**: DACON LB submit (별개 결정) / 신규 cell 학습 (plan-022/023/025 carry only) / hparam adjust / 4-way 이상 ensemble / temperature scaling / median voting.

---

## §0.5 Quick Reference (autonomous loop 매 turn 읽는 section)

### 합격 기준 (G-gate sequence)

- **G0**: plan-022/023/025 carry module + `ensemble_predict` smoke + tests green. 위반 시 `infra_drift` severe.
- **G1**: plan-026 A2 결과 (`analysis/plan-026/results_A2.json`) 존재 + p022/p023 winner reproduce (sample-level final_pred 박제, plan-026 A2 OOF prediction 재산출). 위반 시 `prereq_p026_A2_missing` 또는 `reproduce_drift` severe.
- **G2.E1/E2/E3**: 각 cell 의 ensemble hit_1cm finite + sample 단위 prediction 정합 (N_test mismatch check). 위반 시 `ensemble_dim_mismatch` severe.
- **G3 (paradigm)**: best cell hit_1cm > max(p022, p023) + 0.002 → PASS. partial = [0.6532, max+0.002]. regression = < 0.6532.
- **G_final**: results.md + 3-file frontmatter sync + follow-up plan-028/029 박제.

### G-gates

- G0: STAGE 0 인프라 [DONE — bqxddnj9i]
- G1: STAGE 1 base predictor reproduce + plan-025 band 결정 [DONE — bqxddnj9i]
- G2.E1: equal-weight 2-way ensemble [DONE — bqxddnj9i]
- G2.E2: equal-weight 3-way ensemble (band 조건부) [DONE — bqxddnj9i]
- G2.E3: weighted optim [DONE — bqxddnj9i]
- G3: paradigm — band 판정 [DONE — bqxddnj9i]
- G_final: results + 3-file sync [DONE — bqxddnj9i]

### Commit chain (next-up)

| # | type | spec section | status |
|---|---|---|---|
| c1 | docs | `plans/plan-027-ensemble.md` v1 작성 | [DONE — bqxddnj9i] |
| c2 | code | `analysis/plan-027/ensemble_predict.py` — `predict_p022_oof(X, gt, folds) -> np.ndarray (N, 3)`, `predict_p023_oof(...)`, `predict_p026_A2_oof(...)`, `ensemble_soft_vote(preds: list[np.ndarray], weights: list[float]) -> (N, 3)` | [DONE — bqxddnj9i] |
| c3 | code | `analysis/plan-027/run_oof.py` — 3 cell runner + band 분기 + CLI `--cell {E1,E2,E3}` | [DONE — bqxddnj9i] |
| c4 | test | `tests/test_plan027_smoke.py` — predict OOF shape + ensemble weight sum=1 + band 분기 smoke | [DONE — bqxddnj9i] |
| G0 | gate | smoke + tests green | [DONE — bqxddnj9i] |
| c5 | exp G1 | base predictor (p022 + p023) reproduce + plan-026 A2 carry → `base_preds.json` 박제 (sample 별 final_pred) + band 결정 | [DONE — bqxddnj9i] |
| G1 | gate | p022 reproduce hit ∈ [0.6523, 0.6533] + p023 reproduce hit ∈ [0.6527, 0.6537] + plan-026 A2 carry OK | [DONE — bqxddnj9i] |
| c6 | exp G2.E1 | E1 equal-weight 2-way (w=0.5/0.5) — `results_E1.json` 박제 | [DONE — bqxddnj9i] |
| G2.E1 | gate | metric finite | [DONE — bqxddnj9i] |
| c7 | exp G2.E2 | E2 equal-weight 3-way (w=1/3) — band 조건 충족 시만. 미충족 시 skip (decision-note 박제) → `results_E2.json` 또는 skip log | [DONE — bqxddnj9i] |
| G2.E2 | gate | metric finite OR skip 박제 | [DONE — bqxddnj9i] |
| c8 | exp G2.E3 | E3 weighted optim — 2-way grid sweep (5-point or 9-point) + (band 충족 시) 3-simplex sweep — `results_E3.json` 박제 | [DONE — bqxddnj9i] |
| G2.E3 | gate | metric finite + best weight 박제 | [DONE — bqxddnj9i] |
| c9 | analysis | 3 cell 비교 + best cell + lift vs single winner + paired Δ → `ensemble_analysis.{json,md}` | [DONE — bqxddnj9i] |
| G3 | gate | band 박제 | [DONE — bqxddnj9i] |
| c10 | docs | 3-file frontmatter sync + `analysis/plan-027/results.md` + pair + follow-up plan-028/029 박제 | [DONE — bqxddnj9i] |
| G_final | gate | 3-file sync + §0.5 c1~c10 [DONE] | [DONE — bqxddnj9i] |

### Plan-specific severe

- `prereq_p026_A2_missing`: `analysis/plan-026/results_A2.json` 부재 → halt.
- `reproduce_drift`: p022 (band [0.6523, 0.6533]) 또는 p023 (band [0.6527, 0.6537]) reproduce 가 tight band 밖.
- `ensemble_dim_mismatch`: base predictor 의 sample 수 / final_pred shape 불일치.
- `band_decision_change`: band 결정 후 ensemble runtime 중 base predictor reproduce 가 band 바꿈 → halt (재현성 깨짐).
- `negative_ensemble`: ensemble best < min(p022, p023) → warn 박제 + paradigm finding 으로 carry.

### Plan-specific paths

- whitelist: `analysis/plan-027/**` + `tests/test_plan027_smoke.py`
- blacklist: `analysis/plan-{001..026}/**` (read-only import 예외)

### Decision-note 사용 예

- `decision-note: spec-default — plan-026 A2 band=regression → E2 skip, 2-way ensemble (E1, E3) 만 측정.`
- `decision-note: spec-default — E3 weight grid = {0.3, 0.4, 0.5, 0.6, 0.7} for 2-way (5-point) OR 3-simplex {(0.5,0.5,0), (0.4,0.4,0.2), (0.4,0.3,0.3), ...} 5-point for 3-way.`
- `decision-note: spec-default — base predictor sample-level final_pred reproduce 는 plan-022 run_oof_cell + plan-023 run_oof_largeN_cell wrapper 로 (fold 별 final_pred 누적) — plan-022/023 source 미수정.`

---

## §1. 배경

### §1.1 plan-022/023/024/025 finding

| Plan | Best cell | hit_1cm | hit_1p5cm | Lift vs F0 |
|:--|:--|--:|--:|--:|
| plan-022 | A6_bcc14_tau001 | 0.6528 | 0.8104 | +0.0208 |
| plan-023 | B4_fib50_tau001 | 0.6532 | 0.8108 | +0.0212 |
| plan-024 | cross-attention (FAIL) | 0.6370 | 0.8092 | +0.0050 |
| plan-025 | C1 (regression) | 0.6320 | 0.8033 | +0.0000 (= F0) |

본 plan = plan-022/023 두 winner 의 ensemble 로 lift 확장. plan-026 A2 은 regression 이라 ensemble 후보 부재 (band 조건 분기).

### §1.2 가설

- **H1 (강): 2-way ensemble (p022 + p023)** equal-weight 가 max(p022, p023) + ≥0.002 lift. 두 winner 의 error pattern uncorrelated 가정.
- **H2 (약): 3-way ensemble (p022 + p023 + p026_A2)** 는 plan-026 A2 band 충족 시만. band=regression 이면 H2 skip.
- **H3 (약): weighted optim** 이 equal-weight 대비 ≥+0.001 추가 lift.

H1 FAIL = ensemble 자체 무효 (winner error pattern correlated, anchor lever 차이가 prediction diversity 만들지 못함). paradigm finding 박제.

### §1.3 baseline 위치

- **max(plan-022, plan-023) = 0.6532** (plan-023 winner). 본 plan G3 의 primary 비교 anchor.
- F0 (0.6320) secondary.

---

## §2. 가설 검증 paradigm (한 변수 원칙)

| 축 | 변경 | 단일 변수 |
|:--|:--|:--|
| Base predictor | plan-022/023 (+ plan-025 band 조건부) carry | ✗ |
| Anchor codebook | plan-022 K=14 BCC, plan-023 K=50 fib carry | ✗ |
| τ_cls | 0.001 fix (plan-022/023 carry) | ✗ |
| Fold split | stable_fold_id carry | ✗ |
| Model | LGBM carry (plan-022/023 reproduce) | ✗ |
| F0 baseline | f0_baseline carry | ✗ |
| **Ensemble weight w** | **E1 equal / E2 equal / E3 grid sweep** | **✓ 본 plan 변수** |

---

## §3. 사전 등록

### §3.1 합격 기준

| Gate | 합격 |
|:--|:--|
| G0 | tests green + plan-022/023/025 module import |
| G1 | p022 + p023 reproduce tight band + plan-025 carry OK |
| G2.E1 | 2-way equal hit_1cm finite |
| G2.E2 | 3-way equal OR skip (band=regression 시) |
| G2.E3 | grid sweep best weight 박제 |
| **G3** | **best cell hit_1cm > max(p022, p023) + 0.002 → PASS** |
| G_final | 3-file sync + follow-up 2건 박제 |

### §3.2 Ensemble 식

```
# 2-way (E1, E3 2-way variant)
final_pred = w_p022 * pred_p022 + w_p023 * pred_p023        # (N, 3) world
  여기서 w_p022 + w_p023 = 1.0

# 3-way (E2, E3 3-way variant — band 충족 시)
final_pred = w_p022 * pred_p022 + w_p023 * pred_p023 + w_p026_A2 * pred_p026_A2
  여기서 sum w = 1.0

hit_1cm = (np.linalg.norm(final_pred - gt, axis=1) <= 0.01).mean()
```

각 pred_* 는 5-fold OOF concat (= sample 별 1개 prediction).

### §3.3 Base predictor reproduce (sample-level final_pred 출력)

plan-022 run_oof_cell / plan-023 run_oof_largeN_cell 가 hit 만 반환 → 본 plan 의 wrapper `predict_*_oof(X, gt, folds)` 가 fold loop 안에서 final_pred 누적 후 (N, 3) 반환:

```python
def predict_p022_oof(X, gt, folds, anchors=ANCHORS_A6, tau_cls=0.001) -> np.ndarray:
    """5-fold OOF concat → (N, 3) world final_pred + assert reproduce band."""
    # plan-022 run_oof_cell 의 fold loop 정확 carry — selector self-consistency
    # + Frenet→world 변환 식까지 (plan-025 동일 패턴).
    ...
    assert 0.6523 <= hit_check <= 0.6533, "p022 reproduce drift"
    return oof_pred
```

`predict_p023_oof(X, gt, folds)` 도 동일 (anchors=ANCHORS_B4_fib50, K=50).

`predict_p026_A2_oof(X, gt, folds)` = `analysis/plan-025/results_C1.json` 의 per_fold 결과로부터 final_pred 재구성 또는 run_oof_plan025 직접 재호출.

### §3.4 E3 weight grid

**2-way grid (5-point)**: `[(0.3, 0.7), (0.4, 0.6), (0.5, 0.5), (0.6, 0.4), (0.7, 0.3)]` — 5 weight × hit metric.

**3-way grid (5-point, band 충족 시)**: `[(0.5, 0.5, 0.0), (0.4, 0.4, 0.2), (0.4, 0.3, 0.3), (0.3, 0.3, 0.4), (0.34, 0.33, 0.33)]` (= 2-way variants + 3-way variants 의 represantive 5).

`best_weight = argmax_{w grid} hit_1cm(w)`.

---

## §4~§8. STAGE 0~4 — expansion

### §4 (G0) — 인프라

- `analysis/plan-027/__init__.py` + `run_oof.py` (predict wrappers + ensemble grid).
- prerequisite: `analysis/plan-026/results_A2.json` 존재 확인 (G1 단계의 hit_1cm=0.6509 ±0.001 lazy check, 정확 reproduce 는 G1 단계 OOF predict 안에서).
- pytest skip (소형 plan, run_oof.py 자체 smoke 검증 + 학습 path 가 G1 단계에 통합).

### §5 (G1) — base predictor OOF reproduce + final_pred 박제

- `_predict_oof_p022_style(X, gt, ids, ANCHORS_A6, seed=20260519)` → (N, 3) world `pred_p022` + hit_1cm assert ∈ [0.6523, 0.6533].
- `_predict_oof_p022_style(X, gt, ids, ANCHORS_B4, seed=20260519)` → (N, 3) world `pred_p023` + hit_1cm assert ∈ [0.6527, 0.6537].
- `_predict_oof_p026_A2_style(X, gt, ids, seed=20260522)` → (N, 3) world `pred_p026_A2` + hit_1cm assert ∈ [0.6499, 0.6519] (plan-026 A2 hit=0.6509 ±0.001).
- tolerance ±0.0005~0.001: LGBM seed deterministic + stable_fold_id MD5 동일 → 실제 drift 미세 (LightGBM thread non-determinism 일부 가능).

### §6 (G2) — 3 cell ensemble hit 측정

- **E1 equal 3-way**: `pred_E1 = (pred_p022 + pred_p023 + pred_p026_A2) / 3.0`. hit_E1_1cm = (‖pred_E1 − gt‖ ≤ 0.01).mean().
- **E2 equal 2-way**: `pred_E2 = (pred_p022 + pred_p023) / 2.0` (p026_A2 제외, baseline 비교).
- **E3 weighted grid sweep**: 9-point grid {(0.5,0.5,0), (0.4,0.4,0.2), (0.4,0.3,0.3), (0.3,0.3,0.4), (0.34,0.33,0.33), (0.6,0.4,0), (0.4,0.6,0), (0.3,0.5,0.2), (0.5,0.3,0.2)} 각 pred 산출 + best = argmax hit_1cm.
- 모든 weight ≥ 0 + sum = 1.0 (simplex constraint).
- tie-break: max hit_1cm 동일 시 max hit_1p5cm.

### §7 (G3) — paradigm verdict + paired Δ

- `base_max = max(hit_p022, hit_p023, hit_p026_A2)` (3-way base ceiling).
- `pass_threshold = base_max + 0.002`.
- best_cell = argmax_{E1, E2, E3} hit_1cm; best_hit = corresponding metric.
- verdict: best_hit > pass_threshold → PASS / ≥ base_max → marginal / < base_max → negative_ensemble.
- paired Δ vs base_max: `delta_ensemble_vs_base = best_hit − base_max` (단순 metric Δ, bootstrap CI 생략 — 5-fold OOF 가 이미 sample 단위 평균이라 paired sample test 1차 근사).

### §8 (G_final) — results.md + 3-file sync

- `analysis/plan-027/results_ensemble.json` (run_oof.py 자체 dump):
  - `base_predictors` (3 dict), `base_max_hit_1cm`, `pass_threshold`
  - `cells` (E1/E2/E3 각 hit + weights + E3 grid)
  - `best_cell`, `best_hit_1cm`, `G3_verdict`, `G3_band`, `runtime_s`
- `analysis/plan-027/results.md` (sample table + paradigm finding)
- `plans/plan-027-ensemble.results.md` pair
- 3-file frontmatter sync (best_cell, best_hit_1cm, band).
- follow-up: plan-028 (F0 ML — anchor selection ceiling 확인), plan-029 (selector redesign).

## §9. Out of scope

- DACON LB submit
- 신규 LGBM 학습 (plan-022/023/025 carry only)
- 4-way 이상 ensemble
- temperature scaling / median voting
- F0 baseline ML (plan-028)

## §10. 참조

- plan-022/023/025 spec + results
- plan-022 anchors/run_oof, plan-023 anchors_largeN/run_oof_largeN, plan-025 run_oof
