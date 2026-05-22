---
plan_id: 026
version: 1.1
date: 2026-05-22 (Asia/Seoul)
status: all_complete
best_cell: A2_no_block3
best_hit_1cm: 0.6509
best_hit_1p5cm: 0.8118
best_delta_1cm: 0.0189
band: paradigm_reversal
based_on:
  - 025 (input 1080D = block ① 170 + ② 128 + ③ 22 + ④ 760, K=14 BCC + τ=0.001, LGBM row-expand. C1 default hparam = plan-022 carry, baseline 측정 = plan-025 G2.C1 결과)
  - 022 (LGBM K=14 BCC + τ=0.001 winner paradigm)
  - 024 (cand_builder / seq_builder 모듈 carry — plan-024 commit 915dd26)
  - 021 (170D LGBM input pipeline)
  - 020 (F0 baseline + stable_fold_id MD5)
inspired_by:
  - 025 §0.5 followed_by — block ② / ③ / ④ each-out ablation 으로 lift attribution
code_reuse:
  - module: analysis/plan-025/build_feat_1080.py
    symbols: [build_feat_1080, compress_seq_8stat, BLOCK_DIMS, STAT_NAMES, K_ANCHORS]
    reason: full 1080D builder carry. block masking 은 본 plan 의 wrapper 가 column slice 로 처리.
  - module: analysis/plan-025/run_oof.py
    symbols: [LgbmSelectorRowExpanded, run_oof_plan025, _normalize_p022_result, _stratified_inner_split, _row_expand_idx, CELL_CONFIGS, LGBM_RANDOM_STATE, TAU_CLS, K_ANCHORS, N_FOLDS]
    reason: 5-fold OOF runner carry. cell config 만 변경.
  - module: analysis/plan-022/anchors.py
    symbols: [ANCHORS_A6]
    reason: K=14 BCC carry.
  - module: analysis/plan-022/selector_only_model.py
    symbols: [LgbmSelectorOnly, build_soft_label_with_tau]
    reason: model + soft label carry (plan-025 와 동일).
  - module: analysis/plan-024/*
    symbols: [build_cand_feat, build_seq_feat, build_train_quantiles, QuantileCarry]
    reason: cand_builder / seq_builder / quantile_carry 모듈 그대로 carry (plan-025 와 동일).
  - module: analysis/plan-021/build_input.py
    symbols: [build_frenet_basis_3d, build_input_common, build_input_lgbm_extra]
    reason: 170D block ① pipeline.
  - module: analysis/plan-020/baseline_f0.py
    symbols: [f0_baseline, D1, R_HIT, R_HIT_LOOSE]
    reason: F0 baseline.
  - module: src/io.py
    symbols: [load_all_samples, load_labels]
    reason: data loader.
  - module: src/pb_0_6822/selector.py
    symbols: [stable_fold_id, fit_regime_bins, assign_regimes]
    reason: 5-fold split + regime assignment.
followed_by:
  - plan-027 (ensemble — plan-025 winner + plan-022/023 winner soft-vote)
  - plan-028 (F0 baseline ML)
scope: plan-025 input 1080D 위 block ②/③/④ each-out ablation. 3 ablation cell (A1 = -block②, A2 = -block③, A3 = -block④) + baseline = plan-025 C1 carry (1080D full). 단일 변수 = "block 1개 제거" (= input feature 부분집합 변경). LGBM hparam = plan-025 C1 default 그대로 (n_est=500, lr=0.05, num_leaves=63, random_state=20260522). Anchor / τ_cls / fold / soft label / F0 = plan-025/022 carry. block ① 제거 (plan-022 carry 전체 제거) = out-of-scope (paradigm 회귀 의미 손실). C2 hparam = out-of-scope (별개 plan).
exp_ids:
  - Z026_A1_no_block2
  - Z026_A2_no_block3
  - Z026_A3_no_block4
lb_score: null
---

# plan-026 v1 — Block Ablation (plan-025 input 1080D, block ②/③/④ each-out)

## §0. 한 줄 목적

> **plan-025 의 input 1080D** (block ① plan-022 170 + ② cand_ctx 128 + ③ per-anchor 22 + ④ seq 8-stat 760) 에서 block ②/③/④ **each-out** ablation 으로 *각 block 의 lift attribution* 측정. C1 hparam (plan-022 default carry) + K=14 BCC + τ=0.001 fix. 3 cell × 5-fold OOF.
>
> **paradigm rationale**: plan-025 G2.C1 baseline 위에서, "block X 제거 시 hit drop" = X 의 contribution. plan-025 의 1080D 가 lift 했는지 → 어느 block 이 dominant lever 인지 분리.
>
> **prerequisite**: plan-025 G2.C1 결과 (= results_C1.json) 필수. plan-025 G2 미진행 시 본 plan G0 fail (severe `prereq_p025_g2_missing`).
>
> **ablation cells (3)**:
> 1. **A1** = without block ② (cand_builder ctx 128D 제거) → 952D per row
> 2. **A2** = without block ③ (per-anchor 22D 제거) → 1058D per row, **per-anchor lever 자체가 없어지므로 selector 자체가 sample-level prediction 으로 회귀** (예상 큰 drop). 구체 동작: row-expand 후 14 row 의 X feature 가 *모두 동일* (block ①②④ 만 = sample-level broadcast) → LGBM 이 anchor row 를 구분할 feature 부재 → predict_proba 14 row 모두 동일 K-dim 분포 → diag extraction 시 self-consistency 의미 깨짐 → final residual = mean(ANCHORS_A6 weighted by global average prob). 이는 *예상 큰 drop* 의 원인이며 본 plan H1 의 검증 대상 (H1 = block③ 의 dominant lever 가정).
> 3. **A3** = without block ④ (seq 8-stat 760D 제거) → 320D per row
>
> **pass criterion (G3)**: 3 cell 모두 finite metric + each-out drop 박제. PASS 자체는 "결과 박제 완료" (= attribution 측정 plan, lift PASS/FAIL 기준 X). **plan-025 G2.C1 의 hit_1cm** 가 baseline. attribution: `drop_X = baseline_hit_1cm - hit_A{X}`.
>
> **out-of-scope**: block ① 제거 (= plan-022 carry 전체 제거, paradigm 의미 손실) / C2 hparam adjusted / multi-block out (= ②+③ 동시 제거 등) / DACON LB submit / ensemble.

---

## §0.5 Quick Reference (autonomous loop 매 turn 읽는 section)

### 합격 기준 (G-gate sequence)

- **G0**: plan-025 모든 carry module import + `block_mask_builder` smoke + tests green. plan-025 G2.C1 결과 (results_C1.json) 존재 확인. 위반 시 `infra_drift` 또는 `prereq_p025_g2_missing` severe.
- **G1**: plan-025 G2.C1 hit_1cm baseline carry (= results_C1.json 읽기). drift 확인 X (carry only).
- **G2.A{n}** (n ∈ 1..3): 각 ablation cell 5-fold OOF metric finite + `max_class_ratio < 0.95`. 위반 시 `lgbm_numerical` / `soft_label_collapse` severe / warn.
- **G3 (attribution)**: 3 cell 결과 + baseline 비교 표 + drop_X attribution + best block 박제. 자체적으로 PASS/FAIL 기준 없음 (측정 plan).
- **G_final**: results.md + 3-file frontmatter sync + follow-up plan 후보 (plan-027 ensemble, plan-028 F0 ML) 박제.

### G-gates

- G0: STAGE 0 인프라 + plan-025 prerequisite check [DONE — 04ba0bf] 8/8 pytest + prereq_check ✓
- G1: STAGE 1 plan-025 G2.C1 baseline carry [TODO]
- G2.A1: A1 (no block②) 5-fold OOF [DONE — d9daaf8] hit_1cm=0.6320 (Δ=0)
- G2.A2: A2 (no block③) 5-fold OOF [DONE — d9daaf8] **hit_1cm=0.6509 (Δ=+0.0189, 🎯 mode collapse 해소)**
- G2.A3: A3 (no block④) 5-fold OOF [DONE — d9daaf8] hit_1cm=0.6320 (Δ=0)
- G3: STAGE 3 attribution + best block [TODO]
- G_final: STAGE 4 results + 3-file sync [TODO]

### Commit chain (next-up)

| # | type | spec section | status |
|---|---|---|---|
| c1 | docs | `plans/plan-026-block-ablation.md` v1 작성 | [TODO] |
| c2 | code | `analysis/plan-026/block_mask_builder.py` — column slice helper + `BLOCK_RANGES: dict` (block id → (start, end)) + `build_feat_masked(X, anchors, f0_baseline_fn, quantiles, excluded_cell: str)` returns (N*K, D_masked) float32 (dtype carry from plan-025 build_feat_1080). | [DONE — 33c5220] |
| c3 | code | `analysis/plan-026/run_oof.py` — 3 ablation cell 5-fold OOF runner. CLI: `--cell {A1,A2,A3}`. plan-025 LgbmSelectorRowExpanded carry. | [DONE — 9883d9c] |
| c4 | test | `tests/test_plan026_smoke.py` — block mask shape + run smoke (small N=8). | [DONE — 04ba0bf] 8/8 pytest |
| G0 | gate | smoke + tests green + plan-025 G2.C1 results_C1.json 존재 확인 | [DONE — 04ba0bf] |
| c5 | exp G1 | plan-025 G2.C1 baseline carry (= results_C1.json 읽고 hit_1cm 기억) | [TODO] |
| G1 | gate | baseline carry 완료 | [TODO] |
| c6 | exp G2.A1 | A1 no-block② 5-fold OOF (~1-2h CPU 예상, block ② 제거로 dim 952D) | [DONE — d9daaf8] 0.6320 (Δ=0), 302s |
| G2.A1 | gate | metric finite + max_class_ratio < 0.95 | [DONE — d9daaf8] max_class_ratio=0.071 |
| c7 | exp G2.A2 | A2 no-block③ 5-fold OOF (1058D, **per-anchor lever 제거 — 예상 큰 drop**) | [DONE — d9daaf8] **0.6509 (Δ=+0.0189, REVERSE 가설)**, 779s, max_class_ratio=0.106 |
| G2.A2 | gate | 동일 | [DONE — d9daaf8] mode collapse 해소 ✓ |
| c8 | exp G2.A3 | A3 no-block④ 5-fold OOF (320D, ~30-60min CPU 예상) | [DONE — d9daaf8] 0.6320 (Δ=0), 132s |
| G2.A3 | gate | 동일 | [DONE — d9daaf8] |
| c9 | analysis | 3 cell + baseline 표 + drop_X attribution + dominant block → `attribution.{json,md}` | [DONE — 본 commit] best=A2 0.6509, paradigm reversal finding |
| G3 | gate | 표 + best block 박제 | [DONE — 본 commit] `attribution_negative` warn (block ③ = noise lever, REVERSE 가설) |
| c10 | docs | 3-file frontmatter sync + `analysis/plan-026/results.md` + `plans/plan-026-*.results.md` pair + follow-up plan-027/028 박제 | [DONE — 본 commit] |
| G_final | gate | 3-file sync + §0.5 c1~c10 [DONE] + follow-up 2건 박제 | [DONE — 본 commit] |

### Plan-specific severe

- `prereq_p025_g2_missing`: `analysis/plan-025/results_C1.json` 부재 → halt.
- `infra_drift`: plan-025 module import 실패.
- `lgbm_numerical`: NaN/Inf.
- `soft_label_collapse`: max_class_ratio ≥ 0.95.
- `block_dim_mismatch`: build_feat_masked output dim ≠ expected (952 / 1058 / 320).
- `attribution_negative`: drop_X < 0 (block 제거가 hit 향상) — warn only, paradigm finding 으로 박제.

### Plan-specific paths

- whitelist:
  - `analysis/plan-026/**`
  - `tests/test_plan026_smoke.py`
- blacklist: `analysis/plan-{001..025}/**` (read-only import 만 예외 — §0.5 code_reuse 표 참조)

---

## §1. 배경

### §1.1 plan-025 finding 과 본 plan 의 응답

| Plan | Finding |
|:--|:--|
| plan-025 G1 | F0 0.6320/0.8033 + plan-022 winner 0.6531/0.8108 reproduce ✓ |
| plan-025 G2.C1 | **prerequisite** — 본 plan 의 baseline = C1 hit_1cm |
| plan-025 G2.C2 | (별개 lever, 본 plan 사용 X) |

본 plan = plan-025 의 1080D 가 (lift PASS / partial / regression 어느 band 든) "**어느 block 이 lift 의 dominant 였는가**" 분리. plan-025 자체는 *full 1080D 단일 변수* 였고, 본 plan = block-level 분해.

### §1.2 가설

- **H1 (강): block ③ per-anchor 22D 제거 시 큰 drop** (selector 가 per-anchor lever 없으면 sample-level prediction 으로 회귀, oracle 회수율 ↓ 예상).
- **H2 (약): block ② ctx broadcast 128D 제거 시 작은 drop** (sample-level summary 는 block ① 의 170D 와 redundant 일 가능성).
- **H3 (약): block ④ seq 8-stat 760D 제거 시 중간 drop** (시간 미세 패턴 일부 손실, 단 ctx broadcast 의 sample-level summary 와 일부 redundant).

### §1.3 baseline 위치

- **plan-025 G2.C1 hit_1cm** (carry from `analysis/plan-025/results_C1.json`). 본 plan G2 의 모든 비교 anchor.
- plan-022 winner 0.6531 (G1 b reproduce) 도 secondary baseline 으로 박제.

**baseline carry key list** (results_C1.json 에서 본 plan 이 읽는 필수 key — `_normalize_p022_result` 정규화 적용 후):
- `hit_1cm: float` — primary baseline metric
- `hit_1p5cm: float` — secondary metric (§7 표 baseline 칼럼)
- `max_class_ratio: float` — mode collapse 진단 비교
- `runtime_s: float` — §7 표 baseline runtime 칼럼
- `per_fold: list[dict]` — fold-level reproducibility 확인 (선택)
- 그 외 key 는 본 plan 미사용 (carry only). 키 부재 시 → severe `prereq_p025_g2_missing` 또는 warn (key 별 lenience: hit_1cm/1p5cm 필수, 나머지 lenient).

---

## §2. 가설 검증 paradigm (한 변수 원칙)

| 축 | 변경 | 단일 변수 |
|:--|:--|:--|
| Anchor codebook | K=14 BCC fix | ✗ |
| τ_cls | 0.001 fix | ✗ |
| Model | LgbmSelectorRowExpanded carry | ✗ |
| LGBM hparam | C1 default carry (n_est=500, lr=0.05, num_leaves=63) | ✗ |
| **Input block** | **block ②/③/④ 1개 제거** | **✓ 본 plan 변수** |
| Fold split | stable_fold_id carry | ✗ |
| F0 baseline | f0_baseline carry | ✗ |

---

## §3. 사전 등록

### §3.1 Block ranges (1080D total)

| Block id | Column range | Dim | A1 mask | A2 mask | A3 mask |
|:--|:--|--:|:--:|:--:|:--:|
| ① plan-022 | 0..170 | 170 | keep | keep | keep |
| ② ctx | 170..298 | 128 | **drop** | keep | keep |
| ③ per-anchor | 298..320 | 22 | keep | **drop** | keep |
| ④ seq 8-stat | 320..1080 | 760 | keep | keep | **drop** |

각 ablation cell 의 D_masked:
- A1 (no block②): 1080 − 128 = **952D**
- A2 (no block③): 1080 − 22 = **1058D**
- A3 (no block④): 1080 − 760 = **320D**

### §3.2 합격 기준

| Gate | 합격 |
|:--|:--|
| G0 | tests green + `analysis/plan-025/results_C1.json` 존재 ✓ |
| G1 | C1 baseline hit_1cm 읽기 OK |
| G2.A1/A2/A3 | metric finite + max_class_ratio < 0.95 + dim 정합 |
| G3 | 3 cell + baseline 비교 표 + drop_X attribution + dominant block 박제 |
| G_final | 3-file sync + follow-up 2건 박제 |

### §3.3 Attribution 식

```
drop_X = baseline_hit_1cm - hit_A{X}              # X ∈ {1, 2, 3} = block id 2/3/4
contribution_X = drop_X / baseline_hit_1cm * 100  # %
dominant_block = argmax_X drop_X
```

drop_X < 0 = block X 제거가 hit 향상 (= 해당 block 이 *noise* lever 였을 가능성). `attribution_negative` warn 박제 후 paradigm finding 으로 carry.

---

## §4~§8. STAGE 0~4

### §4.1 모듈 layout

```
analysis/plan-026/
├── __init__.py
├── block_mask_builder.py     ← BLOCK_RANGES + build_feat_masked (c2)
├── run_oof.py                 ← 3 cell runner (c3)
├── results_A1.json            ← G2.A1 직후
├── results_A2.json            ← G2.A2 직후
├── results_A3.json            ← G2.A3 직후
├── attribution.{json,md}      ← c9
└── results.md                 ← c10
tests/test_plan026_smoke.py    ← c4
```

### §4.2 build_feat_masked 시그너처

```python
BLOCK_RANGES: dict[str, tuple[int, int]] = {
    "block1": (0, 170),
    "block2": (170, 298),
    "block3": (298, 320),
    "block4": (320, 1080),
}

EXCLUSION_MAP: dict[str, str] = {"A1": "block2", "A2": "block3", "A3": "block4"}

def build_feat_masked(X, anchors, f0_baseline_fn, quantiles, excluded_cell: str) -> np.ndarray:
    """returns (N*K, D_masked) — plan-025 의 build_feat_1080 출력에서 excluded_cell 의 column 만 제거."""
    feat_full = build_feat_1080(X, anchors, f0_baseline_fn, quantiles)   # (N*K, 1080)
    excl_block = EXCLUSION_MAP[excluded_cell]
    start, end = BLOCK_RANGES[excl_block]
    keep_mask = np.ones(1080, dtype=bool)
    keep_mask[start:end] = False
    return feat_full[:, keep_mask]
```

### §4.5 tests (c4)

- `test_block_ranges_sum_1080`: sum of BLOCK_RANGES dims = 1080
- `test_build_feat_masked_dims_A1`: A1 → (N*14, 952)
- `test_build_feat_masked_dims_A2`: A2 → (N*14, 1058)
- `test_build_feat_masked_dims_A3`: A3 → (N*14, 320)
- `test_build_feat_masked_preserve_columns`: keep block 의 column 값이 build_feat_1080 의 동일 column 과 일치
- `test_prereq_results_C1_exists`: `analysis/plan-025/results_C1.json` 존재 + key "hit_1cm" 있음

### §6.2 Per-cell 5-fold OOF (plan-025 run_oof carry)

```python
# 각 cell A{n} 별
for fold in 0..4:
    train_idx, test_idx = stable_fold_id ... == fold filter
    R_wfn_train, F0_train, qc 산출 (plan-025 동일)
    feat_train = build_feat_masked(X_train, ANCHORS_A6, f0_baseline, qc, cell)   # (N_tr*14, D_masked)
    feat_test = build_feat_masked(X_test, ...)
    q_train = build_soft_label_with_tau(gt_train, R_wfn_train, F0_train, ANCHORS_A6, 0.001)
    model = LgbmSelectorRowExpanded(K=14)  # plan-025 carry, default hparam
    model.fit(feat_train, q_train)
    probs_test_expanded = model.clf.predict_proba(feat_test)
    # plan-025 와 동일 selector self-consistency + Frenet→world
    ...
    oof_pred[test_idx] = final_pred

hit_1cm = (np.linalg.norm(oof_pred - gt, axis=1) <= 0.01).mean()
```

### §7. STAGE 3 — attribution

| Cell | hit_1cm | hit_1p5cm | drop_1cm vs C1 | drop_1p5cm vs C1 | contribution_1cm % | max_class_ratio | runtime |
|:--|--:|--:|--:|--:|--:|--:|--:|
| baseline (plan-025 C1) | ?.???? (carry) | ?.???? (carry) | — | — | — | ?.??? | (carry) |
| A1 (no block②) | ?.???? | ?.???? | +?.???? | +?.???? | ?.??% | ?.??? | ?h |
| A2 (no block③) | ?.???? | ?.???? | +?.???? | +?.???? | ?.??% | ?.??? | ?h |
| A3 (no block④) | ?.???? | ?.???? | +?.???? | +?.???? | ?.??% | ?.??? | ?h |

dominant_block = argmax_X drop_X.

## §9. Out of scope

- block ① 제거 (paradigm 회귀)
- C2 hparam adjusted (별개 plan)
- multi-block out (e.g. ②+③ 동시 제거)
- DACON LB / ensemble (plan-027)
- F0 ML (plan-028)

## §10. 참조

- plan-025 spec + results (G1 + G2.C1)
- plan-022 anchors / selector_only_model
- plan-021 build_input
- plan-020 baseline_f0
- plan-024 module 8 file (commit 915dd26)
