---
plan_id: 025
version: 1.0
date: 2026-05-22 (Asia/Seoul)
status: written
best_cell: null
best_hit_1cm: null
best_hit_1.5cm: null
best_delta_1cm: null
best_delta_1.5cm: null
based_on:
  - 022 (best A6_bcc14_tau001 → hit@1cm 0.6528 / hit@1.5cm 0.8104. K=14 BCC + τ_cls=0.001 paradigm. 170D LGBM selector-only)
  - 023 (best B4_fib50_tau001 → 0.6532 / 0.8108. anchor large-N lever marginal +0.0008 — anchor 좌표 lever 만으로는 saturation)
  - 024 (cross-attention paradigm FAIL → 0.6370 / 0.8092. 그러나 부산물 = feature engineering 16 lever 풀세트 + 14-anchor oracle 0.7928 ceiling 박제)
  - 021 (selector-only ablation: reg_offset dead, 170D input pipeline)
  - 020 (F0 baseline 0.6320 / 0.8033 + 5-fold stable_fold_id MD5)
inspired_by:
  - 024 (cross-attention model.py 는 폐기하지만 feature engineering module 5개 = cand_builder/seq_builder/torsion_calc/quantile_carry/multiwindow_trim_build 는 LGBM selector 의 input feature 로 *전부 carry*. 2026-05-22 user 한 줄 재정의: "cross-attention 버리고 LGBM + 후보 concat + seq 압축")
  - 022 (LGBM K=14 BCC + τ=0.001 winner paradigm carry — anchor / τ_cls 는 본 plan 변수 X)
code_reuse:
  - module: analysis/plan-024/cand_builder.py
    symbols: [build_cand_feat]
    reason: 묶음①(par/perp/dist 3) + 묶음②(anchor spec 9) + 묶음③(ctx broadcast 128) + 묶음④(interactions 10) = 150D per (sample × anchor). 본 plan input 의 block ②③ 핵심 source. cherry-pick 대상 (c2).
  - module: analysis/plan-024/seq_builder.py
    symbols: [build_seq_feat]
    reason: 7 past step × 95 channel = seq raw. 본 plan 의 block ④ (per-channel 8 stat 압축) input. cherry-pick 대상 (c2).
  - module: analysis/plan-024/torsion_calc.py
    symbols: [build]
    reason: Frenet torsion τ scalar per step (seq_builder 가 internal call). cherry-pick 대상 (c2).
  - module: analysis/plan-024/quantile_carry.py
    symbols: [QuantileCarry, build_train_quantiles, apply_quantiles]
    reason: train fold quantile 박제 (omega_p90, jerk_p90) → test fold 동일 사용. cand_builder 가 A10/S3 threshold 주입에 사용. cherry-pick 대상 (c2).
  - module: analysis/plan-024/multiwindow_trim_build.py
    symbols: [load_trim]
    reason: 144D Multi-window stat → 60D trim list (multiwindow_trim.json carry). cand_builder 묶음③ A2 60D 산출. cherry-pick 대상 (c2). **symbol decision** (c2 cherry-pick 후 확인): `load_trim(json_path) -> np.ndarray[int]` 가 module export 표준. `TRIM_INDICES` 상수가 module top-level 에 있으면 동등 사용 가능 (c2 단계 decision-note 박제).
  - module: analysis/plan-024/anchor_vocab.py
    symbols: [build_anchor_vocab]
    reason: seq_builder 가 internal call (per past step F0 residual → 14-anchor soft assignment F/G/H/F2). cherry-pick 대상 (c2).
  - module: analysis/plan-022/selector_only_model.py
    symbols: [LgbmSelectorOnly, build_soft_label_with_tau]
    reason: row-expand LGBM K-class softmax + soft label 산식. K=14 BCC + τ=0.001 carry. 본 plan model 그대로 (단 init signature 에 LightGBM hparam override 추가 — §4.4 참조).
  - module: analysis/plan-022/anchors.py
    symbols: [ANCHORS_A6, LAYOUT_NAMES]
    reason: K=14 BCC anchor codebook (axis 6 + corner 8, `LAYOUT_NAMES["A6_bcc14"]` 이 같은 array 를 가리킴). plan-022 winner.
  - module: analysis/plan-022/run_oof.py
    symbols: [run_oof_cell]
    reason: per-cell 5-fold OOF runner. 본 plan = 1~2 cell (C1 default, C2 hparam adjust). carry + cell config override.
  - module: analysis/plan-021/build_input.py
    symbols: [build_frenet_basis_3d, to_frenet, build_input_common, build_input_lgbm_extra]
    reason: 170D plan-022 input pipeline. 본 plan input 의 block ① (170D) source 그대로.
  - module: analysis/plan-020/baseline_f0.py
    symbols: [f0_baseline, D1, PAR, PERP, R_HIT, R_HIT_LOOSE]
    reason: F0 baseline injection + paired Δ anchor + hit metric 산식.
  - module: src/io.py
    symbols: [load_all_samples, load_labels]
    reason: data loader.
  - module: src/pb_0_6822/selector.py
    symbols: [stable_fold_id]
    reason: 5-fold stable split (plan-020/021/022/023 carry).
followed_by:
  - plan-026 (가칭): plan-025 결과가 G3 PASS (hit@1cm > 0.67) → block ② / ③ / ④ each-out ablation 으로 lift attribution. fail → hparam grid 확장 또는 anchor radius 0.7cm 시도
  - plan-027 (가칭): plan-025 + plan-022/023 winner ensemble (soft-vote)
  - plan-028 (가칭): F0 baseline 자체를 ML 화 (현재 hand-crafted → systematic forward bias 완화)
scope: plan-022 winner cell (K=14 BCC + τ_cls=0.001) 위 LGBM selector input 확장 — block ①(plan-022 170D carry) + ②(cand_builder ctx broadcast 128D) + ③(per-anchor 22D) + ④(seq_builder 95×7 → 8-stat 압축 760D) = **1080D per row**. 단일 변수 = input feature 확장 (+910D). Anchor / τ_cls / fold split / soft-label 산식 / F0 baseline = plan-022 carry, 변경 X. LGBM hparam = plan-022 default carry (C1) + 1080D 대응 adjusted variant (C2, feature_fraction=0.7 + min_data_in_leaf=50 + n_estimators=2000 + early_stopping_rounds=100 + lr=0.03). corrector reg head / GRU / cross-attention / DACON LB / ensemble / anchor radius ≠ 0.005m / K ≠ 14 = out-of-scope.
exp_ids:
  - Z025_C1_default
  - Z025_C2_adjusted
lb_score: null
band: null
---

# plan-025 v1 — Candidate-concat Input Max (1080D LGBM selector, K=14 BCC, τ=0.001)

## §0. 한 줄 목적

> **plan-022 winner cell (A6_bcc14_tau001, hit@1cm 0.6528)** + **plan-024 cross-attention 폐기 후 부산물 feature engineering 16 lever 전부 carry** 위에서, **LGBM row-expand selector 의 input 을 170D → 1080D 로 확장** (block ① plan-022 170D + ② cand_builder ctx 128D + ③ per-anchor 22D + ④ seq_builder 95×7 → 8-stat 압축 760D) 하여 **hit@1cm > 0.67** lift 측정. anchor (K=14 BCC) / τ_cls (0.001) / fold / soft-label = plan-022 carry, 단일 변수 = input feature concat.
>
> **paradigm rationale**: 14-anchor oracle 0.7928 = ranking lever 의 *수치 ceiling*. plan-022 selector 가 oracle 의 82.3% 회수. selector capacity ↑ (= input dim 6.4× ↑) 로 ceiling 추가 회수 가능성 측정. plan-024 cross-attention 은 CPU under-converged + 다중 lever 동시 bottleneck 분해 불가로 fail → paradigm 회귀 (LGBM) 위에 feature lever 만 누적.
>
> **input block 4 (1080D per row)**:
> 1. **Block ①** plan-022 170D carry — sample-level (`build_input_common` + `build_input_lgbm_extra`)
> 2. **Block ②** cand_builder 묶음③ ctx broadcast 128D — sample-level (regime 18 + Multi-window 60 + STA/LTA 3 + ...)
> 3. **Block ③** cand_builder 묶음①②④ per-anchor 22D — sample × anchor (par/perp/dist 3 + anchor spec 9 + interactions 10)
> 4. **Block ④** seq_builder 95×7 → per-channel 8-stat 압축 760D — sample-level (last / first / mean / std / slope / max / min / range)
>
> **cell scan**: C1 (LGBM hparam plan-022 carry: n_estimators=500, lr=0.05, num_leaves=63) + C2 (1080D 대응 adjusted: n_estimators=2000 + lr=0.03 + num_leaves=63 + feature_fraction=0.7 + min_data_in_leaf=50 + early_stopping_rounds=100). 2 cell total.
>
> **pass criterion (G3)**: 2 cell 중 ≥ 1 개가 hit@1cm > 0.6700 (STRICT) → PASS. partial band = max(hit@1cm) ∈ [0.6528, 0.6700] (= plan-022 winner 이상이지만 stretch goal 미달, **양 끝점 포함** — 0.6700 끝점은 partial 에 속함, > 0.6700 strict 만 PASS). FAIL = max < 0.6528 (baseline 도 못 도달).
>
> **out-of-scope**: corrector reg head 재투입 / GRU sub-exp / cross-attention 재시도 / LB 측정 / DACON submit / ensemble / anchor layout 변경 (K=14 BCC fix) / τ_cls 변경 (0.001 fix) / anchor radius ≠ 0.005m / block ②③④ each-out ablation (G3 PASS 시 plan-026 후보) / F0 baseline ML 화 (plan-028 후보).

---

## §0.5 Quick Reference (autonomous loop 매 turn 읽는 section)

### 합격 기준 (G-gate sequence)

- **G0**: 4 module (build_feat_1080 / run_oof / tests + plan-024 **6 module** + 1 data file cherry-pick 정상) import + smoke + tests green. plan-022 selector_only_model.py + plan-021 build_input.py import 정상. 위반 시 `infra_drift` severe.
- **G1**: F0 baseline 5-fold concat OOF — hit@1cm ∈ [0.6315, 0.6325] AND hit@1.5cm ∈ [0.8028, 0.8038] (plan-020 / 021 / 022 / 023 carry). plan-022 winner reproduce — A6_bcc14 + τ=0.001 cell hit@1cm ∈ [0.6523, 0.6533] AND hit@1.5cm ∈ [0.8099, 0.8109]. 위반 시 `f0_reproduce_drift` / `plan022_reproduce_drift` severe.
- **G2.C1** (LGBM hparam carry): 5-fold OOF metric finite + `max_class_ratio < 0.95` + 1080D input pipeline lint-clean. 위반 시 `lgbm_numerical` severe / `soft_label_collapse` warn.
- **G2.C2** (LGBM hparam adjusted): 위 동일 + early_stopping 정상 trigger (best_iteration ∈ [50, 2000] per fold). 위반 시 동일 severe.
- **G3 (paradigm-level)**: 2 cell (C1, C2) 중 ≥ 1 cell 이 hit@1cm > 0.6700 → PASS. 0.6528 ≤ max < 0.6700 = `partial_lift` warn. max < 0.6528 = `regression` warn 박제 후 G_final.
- **G_final**: results.md + best cell 박제 (C1 / C2 + hparam + 모든 metric) + plan-022 winner 대비 Δ + 14-anchor oracle 대비 회수율 (= best / 0.7928) + follow-up plan 후보 ≥ 2 건 박제 + 3-file frontmatter sync.

### G-gates

- G0: STAGE 0 인프라 [DONE — a3646dd] 12/12 pytest pass (3.33s)
- G1: STAGE 1 F0 + plan-022 winner reproduce [DONE — e262299] F0 0.6320/0.8033 + plan-022 0.6531/0.8108 ✓
- G2.C1: C1 default hparam 5-fold OOF [TODO]
- G2.C2: C2 adjusted hparam 5-fold OOF [TODO]
- G3: STAGE 3 paradigm + best cell [TODO]
- G_final: STAGE 4 results + 3-file sync [TODO]

### Commit chain (next-up)

| # | type | spec section | status |
|---|---|---|---|
| c1 | docs | `plans/plan-025-candidate-concat-input-max.md` v1 작성 (plan-review-master 자동 fix BLOCKER 0 도달) | [TODO] |
| c2 | chore | plan-024 **8 file** cherry-pick (= code 6 module + data 1 + __init__.py) from `worktree-plan-024-combo` (commit 915dd26 이후 최신) → main `analysis/plan-024/` : `__init__.py` + `anchor_vocab.py` + `cand_builder.py` + `seq_builder.py` + `torsion_calc.py` + `quantile_carry.py` + `multiwindow_trim_build.py` + `multiwindow_trim.json`. **code only**, results / log / `model.py` (cross-attention) / `run_oof*.py` / `diagnose_*.py` / `feature_weighted_dropout.py` 는 *cherry-pick 제외*. | [DONE — 91b02a0] |
| c3 | code | `analysis/plan-025/build_feat_1080.py` (block ① + ② + ③ + ④ concat + 8-stat 압축 + 1080D per row 출력 + smoke test) | [DONE — 2eb8485] |
| c4 | code | `analysis/plan-025/run_oof.py` (5-fold OOF runner + C1/C2 cell config + plan-022 `run_oof_cell` carry import + CLI: `--cell {C1, C2}`) | [DONE — 62d138e] |
| c5 | test | `tests/test_plan025_smoke.py` (≥ 8 pytest: import / block dim / 8-stat 산식 / row-expand / LgbmSelectorOnly K=14 + 1080D fit/predict smoke / F0 carry / soft label sum=1 / plan-024 module import) | [DONE — a3646dd] |
| G0 | gate | smoke + tests green — 12/12 pytest pass (3.33s) | [DONE — a3646dd] |
| c6 | exp G1 | F0 baseline reproduce → 0.6320 / 0.8033 + plan-022 winner A6_bcc14_tau001 reproduce → 0.6528 / 0.8104. `analysis/plan-025/baseline_carry.json` 박제 (dataset_hash + plan-022 carry hash) | [DONE — e262299] F0 0.6320/0.8033 + plan-022 0.6531/0.8108 |
| G1 | gate | F0 hit ∈ tight band ✓ AND plan-022 winner hit ∈ tight band ✓ AND dataset_hash 일치 ✓ | [DONE — e262299] |
| c7 | exp G2.C1 | C1 default hparam (n_estimators=500, lr=0.05, num_leaves=63) — 5-fold OOF, K=14 BCC + τ=0.001 fix, 1080D input. `results_C1.json` 박제. 예상 runtime 1.5~3h CPU | [TODO] |
| G2.C1 | gate | C1 metric finite ✓ + max_class_ratio < 0.95 ✓ | [TODO] |
| c8 | exp G2.C2 | C2 adjusted hparam (n_estimators=2000 + lr=0.03 + num_leaves=63 + feature_fraction=0.7 + min_data_in_leaf=50 + early_stopping_rounds=100) — 5-fold OOF, 동일 input. `results_C2.json` 박제. 예상 runtime 2~5h CPU (early_stopping 영향) | [TODO] |
| G2.C2 | gate | C2 metric finite ✓ + max_class_ratio < 0.95 ✓ + best_iteration ∈ [50, 2000] per fold ✓ | [TODO] |
| c9 | analysis | 2 cell hit@1cm/1.5cm 표 + best cell selection + paired Δ vs plan-022 winner + 14-anchor oracle 0.7928 대비 회수율 + (PASS 시) block ②③④ lift attribution 잠재력 박제 → `paradigm_analysis.{json,md}` | [TODO] |
| G3 | gate | best cell hit@1cm > 0.6700 → PASS / ∈ [0.6528, 0.6700] → partial_lift warn / < 0.6528 → regression warn | [TODO] |
| c10 | docs | 3-file frontmatter sync (status=all_complete, band=positive/partial/negative, best_cell) + `analysis/plan-025/results.md` (11 항목) + `plans/plan-025-*.results.md` pair + follow-up plan-026/027/028 박제 | [TODO] |
| G_final | gate | 3-file sync ✓ + §0.5 c1~c10 모두 [DONE] ✓ + follow-up 3건 박제 ✓ | [TODO] |

### Plan-specific severe (WORKFLOW.md §12.3 default 위 추가분)

- `f0_reproduce_drift`: G1 F0 reproduce 가 plan-020/021/022/023 hard evidence 0.6320 / 0.8033 ±0.0005 밖. → halt.
- `plan022_reproduce_drift`: G1 plan-022 winner (A6_bcc14 + τ=0.001) reproduce 가 plan-022 hard evidence 0.6528 / 0.8104 ±0.0005 밖. → halt.
- `lgbm_numerical`: 2 cell 중 어느 LGBM classifier 출력 NaN/Inf. soft label CE / softmax 산출 / 1080D input 의 NaN/Inf propagation 의심. → halt.
- `soft_label_collapse`: 2 cell 중 selector probs 가 단일 anchor 에 95% 이상 mass (= `max_class_ratio = probs_all.mean(axis=0).max() > 0.95`). warn (severe 아님). G3 분모 영향: drop 시 G3 분모 = "(2 − N_drop) cell 중 ≥ 1". 2 cell 모두 drop = `soft_label_collapse_total` severe escalate.
- `frenet_basis_degenerate`: plan-021/022/023 carry — ‖v_last‖ < 1e-9 또는 ‖a_⊥‖ < 1e-9 sample 비율 > 5%. plan-021 fallback (R_wfn ← I_3) 그대로.
- `plan024_module_import_fail`: c2 cherry-pick 후 plan-024 **6 module** (anchor_vocab / cand_builder / seq_builder / torsion_calc / quantile_carry / multiwindow_trim_build) 중 어느 importlib 실패, OR `__init__.py` / `multiwindow_trim.json` 부재. → halt.
- `block_dim_mismatch`: block ① + ② + ③ + ④ concat 결과 dim ≠ 1080 per row. spec 산식 위반. → halt.
- `early_stop_outlier`: C2 의 best_iteration < 50 OR > 2000 per fold. early_stopping 비정상 trigger. warn (severe 아님), partial_metric 박제 후 진행.
- `partial_lift`: G3 best hit@1cm ∈ [0.6528, 0.6700]. plan-022 winner 이상이지만 stretch goal 미달. warn 박제 후 G_final (band=partial).
- `regression`: G3 best hit@1cm < 0.6528. plan-022 winner 도 못 도달. warn 박제 후 G_final (band=negative).

### Plan-specific paths (WORKFLOW.md §12.5/§12.6)

- whitelist 추가:
  - `analysis/plan-025/**`
  - `tests/test_plan025_smoke.py`
  - `analysis/plan-024/{__init__.py, anchor_vocab.py, cand_builder.py, seq_builder.py, torsion_calc.py, quantile_carry.py, multiwindow_trim_build.py, multiwindow_trim.json}` — **c2 cherry-pick 단계의 *유일한* plan-024 path 수정 허용** (file add only, post-c2 수정 금지)
- blacklist (plan-001~024 산출 자동 변경 금지):
  - `runs/baseline/{B,S,R,P,D,E,F,H,Z020,Z021,Z022,Z023,Z024}*/**`
  - `analysis/plan-{001..023}/**` (단, **read-only import** 는 §4.3 의 plan-022 / plan-021 / plan-020 module reuse 만 예외)
  - `analysis/plan-024/**` (단, c2 cherry-pick 으로 *추가* 된 7개 file 의 `analysis/plan-024/` path 는 read-only import 만 허용, 본 plan 의 수정 금지)
- 참조 (read-only):
  - `analysis/plan-024/cand_builder.py:build_cand_feat` — 150D per (sample × anchor) builder
  - `analysis/plan-024/seq_builder.py:build_seq_feat` — 95×7 per past step builder
  - `analysis/plan-024/torsion_calc.py:build` — Frenet torsion τ per step
  - `analysis/plan-024/quantile_carry.py:{QuantileCarry, build_train_quantiles, apply_quantiles}` — train fold quantile carry
  - `analysis/plan-024/multiwindow_trim_build.py:{load_trim 또는 TRIM_INDICES}` — 144→60 trim index
  - `analysis/plan-024/multiwindow_trim.json` — kept_indices 60 data carry
  - `analysis/plan-024/anchor_vocab.py:build_anchor_vocab` — seq_builder internal call
  - `analysis/plan-022/selector_only_model.py:{LgbmSelectorOnly, build_soft_label_with_tau}` — model + soft label carry
  - `analysis/plan-022/anchors.py:{ANCHORS_A6, LAYOUT_NAMES}` — K=14 BCC codebook
  - `analysis/plan-022/run_oof.py:run_oof_cell` — per-cell OOF runner
  - `analysis/plan-022/baseline_carry.json` — dataset hash carry
  - `analysis/plan-021/build_input.py` — 170D input pipeline (block ①)
  - `analysis/plan-020/baseline_oof.json` — F0 0.6320 / 0.8033 hard evidence
  - `analysis/plan-020/baseline_f0.py` — F0 산식
  - `src/pb_0_6822/selector.py:stable_fold_id` — 5-fold split

### Decision-note 사용 예 (자율 결정 시 commit msg 박제)

- `decision-note: spec-default — c2 cherry-pick 은 worktree-plan-024-combo branch 의 latest commit 에서 7 file (code 5 + data 1 + __init__) 만. results/log/model.py/run_oof/diagnose_* 제외. cherry-pick command: git checkout worktree-plan-024-combo -- analysis/plan-024/<file>`
- `decision-note: spec-default — block ① 170D = plan-021 build_input.py 의 build_input_common + build_input_lgbm_extra 그대로 carry (= plan-022/023 baseline 동일).`
- `decision-note: spec-default — block ② ctx broadcast 128D = cand_builder build_cand_feat output 중 묶음③ 부분 slice. 14 anchor row 동일 broadcast 라 row 1개당 128D 만 carry (anchor 무관).`
- `decision-note: spec-default — block ③ per-anchor 22D = cand_builder build_cand_feat output 중 묶음①(3) + 묶음②(9) + 묶음④(10) slice. 14 anchor 각 다른 값.`
- `decision-note: spec-default — block ④ seq 8-stat 압축 760D = seq_builder build_seq_feat (N, 7, 95) → per-channel {last, first, mean, std, slope, max, min, range} stack → 95 × 8 = 760. slope = linear regression coefficient over t=4..10 (closed-form). range = max − min.`
- `decision-note: spec-default — K=14 BCC anchor + τ_cls=0.001 fix (plan-022 winner cell). 변경 X.`
- `decision-note: spec-default — C1 LGBM hparam = plan-022 LgbmSelectorOnly 그대로 (n_estimators=500, lr=0.05, num_leaves=63, random_state=20260522).`
- `decision-note: spec-default — C2 LGBM hparam = C1 위 (lr 0.05→0.03, n_estimators 500→2000, feature_fraction 1.0→0.7, min_data_in_leaf 20→50, early_stopping_rounds=None→100). random_state 동일.`
- `decision-note: spec-default — soft label τ_cls = 0.001 plan-022 carry. build_soft_label_with_tau 그대로.`
- `decision-note: spec-default — quantile_carry 의 omega_p90 / jerk_p90 은 train fold (80% 5-fold rotating) 위 quantile 계산 후 test fold 동일 사용. fold-leakage 차단.`
- `decision-note: spec-default — multiwindow_trim.json (kept_indices 60) 은 plan-024 commit 915dd26 carry, full train (10000) 위 deterministic. fold-leakage 미미 (LANL Singer 1st carry, label 미사용).`
- `decision-note: spec-default — early_stopping_rounds=100 의 val set = inner-fold (train 80% 중 marker random_state=20260522 으로 20% split, K=14 BCC 기준 stratified). 결정: K=14 BCC 의 hard-argmax label 위 stratified split.`

---

## §1. 배경

### §1.1 plan-022 / 023 / 024 finding 과 본 plan 의 응답

| Plan | Best cell | hit@1cm | hit@1.5cm | Finding |
|:--|:--|--:|--:|:--|
| plan-022 | A6_bcc14_tau001 | 0.6528 | 0.8104 | K=14 BCC + sharp τ=0.001 paradigm 박제. selector-only LGBM (170D input) 의 floor. |
| plan-023 | B4_fib50_tau001 | 0.6532 | 0.8108 | anchor large-N (K=50 fib) 의 lift = +0.0004 / +0.0004 (marginal). anchor 좌표 lever 만으로는 saturation. |
| plan-024 | cross-attention | 0.6370 | 0.8092 | paradigm 전환 (cross-attention selector) FAIL. 그러나 14-anchor oracle 0.7928 ceiling 박제 + feature engineering 16 lever 풀세트 5 module 산출. |

본 plan 의 응답:
- **plan-022 winner cell (K=14 BCC + τ=0.001) 위에서** selector model = LGBM (= plan-024 paradigm 전환 폐기).
- plan-022 의 170D input 을 plan-024 의 5 module FE 16 lever 로 확장 → **1080D**.
- **단일 변수 = input feature 확장**. anchor / τ_cls / soft label / fold / F0 baseline 모두 carry.
- **목표**: 14-anchor oracle 0.7928 의 회수율 0.6528/0.7928 = 82.3% → 0.6700/0.7928 = 84.5% 로 +2.2pp ↑.

**plan-024 module 의 task fit 검토 (cross-attention 용 설계 → LGBM row-expand 대응)**:

plan-024 의 5 FE module (`cand_builder`, `seq_builder`, `torsion_calc`, `quantile_carry`, `multiwindow_trim_build`) 는 원래 cross-attention encoder 의 입력 (cand 150D × K=14 + seq 95D × T=7) 으로 설계됨. 본 plan 의 LGBM row-expand 위 task fit:
- **cand_builder (150D × 14)**: cross-attention 에서는 K=14 candidate vector 가 attention query/key 로 사용. LGBM row-expand 에서는 동일한 14 row 가 *별개 sample* 로 처리되며 묶음③ ctx 128D 가 14 row 동일 broadcast (= sample-level feature) — 의미 1:1 mapping 가능. 묶음①②④ = per-anchor → row 식별 lever 로 자연 활용.
- **seq_builder (95D × 7)**: cross-attention 에서는 7 step 의 sequence attention. LGBM 은 step 인접성 정보 못 쓰므로 §6.1 의 8-stat 압축 (760D) 필요. 정보 손실 일부 발생하나 base 12 + macro_stat 8 + EWMA 등 sample-level summary 가 ctx 128D 에 이미 carry → 시간 미세 패턴 손실 영향 제한적.
- **torsion_calc (3D × 7)**: seq_builder internal call. 본 plan 에서는 seq 95D 안에 포함 (torsion 3D = seq channel 87..89 추정, plan-024 §4.5 carry). LGBM 위에서도 8-stat 압축 그대로 적용.
- **quantile_carry (omega_p90 / jerk_p90)**: fold-leakage 차단용 train fold quantile. cross-attention 의 A10 Peak count / S3 saccade threshold 주입에 사용 — LGBM 도 동일 threshold 가 cand_builder 묶음③ A10 / A12 산출에 그대로 사용. task fit OK.
- **multiwindow_trim_build (144 → 60 trim)**: deterministic full-train trim. label 미사용 → fold-leakage 미미 (LANL Singer 1st carry, plan-024 §4.4.1 명시). cross-attention 의 input dim 제어 동기와 LGBM 의 feature pruning 동기 일치. task fit OK.

→ 5 module 모두 *cross-attention 설계 동기 → LGBM row-expand 적합성* 1:1 mapping 검증. 본 plan 에서 추가 wrapper 없이 정확 carry 가능.

### §1.2 사용자 narrative — 2026-05-22 "후보 concat + seq 압축"

2026-05-22 사용자 한 줄 재정의: "cross-attention 버리고 LGBM + 후보 concat + seq 압축". 추가 user spec:
- "input 개수 최대로 설정" → block ④ seq 압축 = per-channel 6 stat (570D) 가 아니라 **8 stat (760D)** 채택 (last/first/mean/std/slope/max/min/range).
- "압축 안 들어가는 feature 도 input 에 들어가도록" → block ② ctx broadcast 128D + block ③ per-anchor 22D 모두 carry.
- 목표 = "OOF 6.7 이기기" → G3 PASS criterion = hit@1cm > 0.6700 (STRICT).

### §1.3 가설

- **H1 (강): 1080D row-expand LGBM** 이 plan-022 170D 대비 hit@1cm + ≥ 0.005 lift. **측정 식**: `max(hit_C1, hit_C2) - hit_p022_reproduce ≥ +0.005`, 여기서 `hit_p022_reproduce` = G1 b 단계의 `result_p022_winner["hit_1cm"]` (= §5.1 reproduce 측정값, hard evidence 0.6528 ±0.0005 band 통과 후 박제된 *실측치*). 단순 상수 0.6528 사용 X — fold split / numerical noise 미세 차이를 absorbed. 측정은 cell 별 OOF metric 의 best 단일 값 (= §7.2 의 `best_cell`).
- **H2 (강): hit@1cm > 0.6700** stretch goal 달성 (14-anchor oracle 회수율 +2.2pp ↑). 측정 식: `max(hit_C1, hit_C2) > 0.6700` (STRICT). 동일 `best_cell` 위.
- **H3 (약): C2 adjusted hparam** 이 C1 default 대비 추가 lift ≥ +0.003. 측정 식: `hit_C2 - hit_C1 ≥ +0.003`. C2 의 hparam 변경은 **5개 동시** (n_estimators / lr / feature_fraction / min_data_in_leaf / early_stopping) → H3 PASS 시 *어느 hparam 이 lift 의 원인인지 attribution 불가능* (단일 변수 위반). 개별 hparam attribution = **plan-026 후보** (block ablation 과 별개 ablation axis). H3 측정의 의도는 "1080D 대응 hparam adjust 의 *합산 효과* 검증" 만, attribution 아님.

H1 FAIL 시 = paradigm-level finding "feature engineering 16 lever 의 LGBM lift 미미" 박제 후 follow-up plan-026 으로 ablation / hparam grid 확장.
H2 FAIL but H1 PASS = `partial_lift` band, lift 잠재력 박제 후 plan-026 (block ablation) + plan-027 (ensemble) 후보.
H3 PASS but H2 FAIL = "hparam adjust 만으론 stretch goal 부족" 박제 → plan-026 hparam grid 확장 priority ↓ + plan-026 block ablation priority ↑ (= input lever 자체 한계 식별).

### §1.4 baseline 두 layer

- **G1 a**: F0 baseline (plan-020 carry) — hit@1cm 0.6320 / 0.8033. 모든 paired Δ 의 anchor.
- **G1 b**: plan-022 winner reproduce — A6_bcc14_tau001, hit@1cm 0.6528 / 0.8104. 본 plan G3 의 1차 비교 anchor.

두 layer 모두 G1 에서 reproduce 검증. 둘 중 하나라도 drift 시 severe halt.

---

## §2. 가설 검증 paradigm (한 변수 원칙)

WORKFLOW.md §9 (#2 한 변수 원칙) 의 본 plan 적용:

| 축 | 변경 | 단일 변수 |
|:--|:--|:--|
| Anchor codebook | K=14 BCC fix (plan-022 winner) | ✗ (carry) |
| τ_cls | 0.001 fix (plan-022 winner) | ✗ (carry) |
| Soft label 산식 | `build_soft_label_with_tau` 그대로 | ✗ (carry) |
| 5-fold split | `stable_fold_id` 그대로 | ✗ (carry) |
| F0 baseline | `f0_baseline` 그대로 | ✗ (carry) |
| Model | `LgbmSelectorOnly` row-expand softmax | ✗ (carry) |
| **Input feature** | **170D → 1080D (+910D)** | **✓ 본 plan 변수** |
| LGBM hparam | C1 carry / C2 adjusted | (별개 sub-cell) |

input feature 확장이 *본 plan 의 핵심 단일 변수*. LGBM hparam 의 C1 / C2 분리는 sub-cell 단위 — 한 cell 안의 단일 변수 원칙은 만족 (cell 간 비교는 hparam 변수만 차이).

---

## §3. 사전 등록 (Pre-registration)

### §3.1 Fold split (plan-020/021/022/023 carry)

- 5-fold rotating, `stable_fold_id(sample_id_int) % 5` (plan-020 carry, MD5 단조).
- dataset = 10000 samples. per-fold test = 2000, train = 8000.
- inner val (C2 early_stopping 용) = train 8000 의 20% (= 1600 sample), `sklearn.model_selection.train_test_split(idx_all, test_size=0.20, stratify=q_train.argmax(axis=1), random_state=20260522)`. **stratify target = sample 단위 hard-argmax label** ((N_train,) shape). split 결과 index 를 *sample 단위* 로 받아 row-expand `row_idx = idx[:, None] * 14 + np.arange(14)[None, :]` 으로 확장 (= sample i 의 14 anchor row 가 train/val 분할에서 *항상 같은 쪽* 으로 묶임). row 단위 split 금지 (label leakage 위험 + selector 의 14-row 자체-일관성 깨짐).
- dataset_hash = `analysis/plan-022/baseline_carry.json` 의 hash 와 일치 (G1 검증).

### §3.2 합격 기준 (정량)

| Gate | 합격 |
|:--|:--|
| G0 | 4 모듈 import + tests green (≥ 8/8 pytest) |
| G1 a | F0 hit@1cm ∈ [0.6315, 0.6325] AND hit@1.5cm ∈ [0.8028, 0.8038] |
| G1 b | plan-022 winner A6_bcc14_tau001 hit@1cm ∈ [0.6523, 0.6533] AND hit@1.5cm ∈ [0.8099, 0.8109] |
| G2.C1 | C1 metric finite + max_class_ratio < 0.95 |
| G2.C2 | C2 metric finite + max_class_ratio < 0.95 + best_iteration ∈ [50, 2000] per fold |
| **G3** | **2 cell 중 ≥ 1 개 hit@1cm > 0.6700** (STRICT, 0.6700 끝점은 PASS 아님 — partial 에 포함) |
| G_final | 3-file sync + §0.5 c1~c10 모두 [DONE] + follow-up 3건 박제 |

### §3.3 평가 점수

- **primary metric**: `hit@1cm` = mean(D1(pred_world, gt_world) ≤ 0.01m). 5-fold concat OOF.
- **secondary metric**: `hit@1.5cm` = mean(D1 ≤ 0.015m).
- **paired Δ**: 동일 sample 위 `hit@1cm_plan025 − hit@1cm_F0` (= G1 a baseline) + `hit@1cm_plan025 − hit@1cm_plan022_winner` (= G1 b baseline). 양쪽 모두 박제 (results.md).
- **secondary**: top1_acc (per-sample selector argmax 정확도), max_class_ratio (mode collapse 진단), oracle 회수율 (= best / 0.7928 = 14-anchor oracle).

### §3.4 K=14 BCC anchor + τ_cls=0.001 (plan-022 carry)

- 좌표 = plan-022 `anchors.py:ANCHORS_A6`. axis 6 (±x, ±y, ±z × 0.005m) + corner 8 (±0.005/√3 each axis). 모든 vertex norm = 0.005m exact.
- τ_cls = 0.001m (anchor radius 의 1/5, plan-022 sharp 측 winner).

### §3.5 LGBM hparam — 2 cell

**Seed layer 분리 (NOTE C — 본 plan 의 모든 seed 의 의미)**:
- **5-fold split seed**: 없음 (= `stable_fold_id` MD5 deterministic, sample_id_int 단조). 모든 fold 분할 = plan-020/021/022/023 carry 와 정확 일치.
- **plan-022 reproduce LGBM `random_state`** = `20260519` (G1 b 전용, plan-022 baseline_carry 확정. §5.1 carry).
- **본 plan 신규 cell C1/C2 LGBM `random_state`** = `20260522` (본 plan 의 작성 date). G1 reproduce 와 *분리된 layer*.
- **C2 inner-val split `random_state`** = `20260522` (C2 LGBM seed 와 동일, early_stopping val set 결정용).

- **C1 (default carry)**: `LgbmSelectorOnly(K=14)` 생성자 그대로 (plan-022). n_estimators=500, lr=0.05, num_leaves=63, random_state=20260522, verbose=-1. 추가 hparam 없음.
- **C2 (adjusted)**: 1080D 대응. C1 위:
  - `n_estimators=2000`, `learning_rate=0.03`, `num_leaves=63`
  - `feature_fraction=0.7` (= LightGBM `colsample_bytree=0.7`)
  - `min_data_in_leaf=50` (= `min_child_samples=50`)
  - `early_stopping_rounds=100` + inner val (§3.1 carry)
  - random_state=20260522

C1 vs C2 비교 = "default LGBM 이 1080D capacity 를 충분히 활용하는가" 검증 (H3).

---

## §4. STAGE 0 — 인프라 (G0)

### §4.1 모듈 layout

```
analysis/plan-025/
├── __init__.py
├── build_feat_1080.py     ← block ①②③④ concat builder (c3)
├── run_oof.py              ← 5-fold OOF runner, C1/C2 cell (c4)
└── baseline_carry.json     ← G1 직후 박제 (c6)

analysis/plan-024/          ← c2 cherry-pick from worktree-plan-024-combo
├── __init__.py
├── anchor_vocab.py
├── cand_builder.py
├── seq_builder.py
├── torsion_calc.py
├── quantile_carry.py
├── multiwindow_trim_build.py
└── multiwindow_trim.json

tests/test_plan025_smoke.py  ← ≥ 8 pytest (c5)
```

### §4.2 module top-level export (smoke test lock-in)

- `analysis/plan-025/build_feat_1080.py`:
  - `BLOCK_DIMS: dict[str, int]` = `{"block1_p022": 170, "block2_ctx": 128, "block3_per_anchor": 22, "block4_seq_stat": 760, "total_per_row": 1080}`
  - `STAT_NAMES: list[str]` = `["last", "first", "mean", "std", "slope", "max", "min", "range"]` (= 8 stat)
  - `build_feat_1080(X: np.ndarray, anchors: np.ndarray, f0_baseline_fn: Callable, quantiles: QuantileCarry) -> np.ndarray`: returns (N*K, 1080) row-expanded. `X` shape (N, 11, 3) float32 world frame, `anchors` (K=14, 3) float32 Frenet 좌표 (= plan-022 `ANCHORS_A6`), `f0_baseline_fn(X, end_idx=10) -> (N, 3)` callable (plan-020 carry), `quantiles: QuantileCarry` train fold carry (plan-024 schema, §6.1 NOTE A 참조).
  - `compress_seq_8stat(seq: np.ndarray) -> np.ndarray`: `seq` shape (N, 7, 95) float32 — per past step t ∈ {4,5,6,7,8,9,10} 의 95-channel seq_builder output. returns (N, 760) float32 — per-channel 8-stat stack, stat-major order `[last_0..last_94, first_0..first_94, mean_0..mean_94, std_0..std_94, slope_0..slope_94, max_0..max_94, min_0..min_94, range_0..range_94]` (§6.1 산식).
  - **내부 call chain (자족 표현)**:
    - `cand_feat_150 = cand_builder.build_cand_feat(X, anchors, quantiles)` — returns (N, K=14, 150). slice: `[:, :, 0:3]` = 묶음① par/perp/dist, `[:, :, 3:12]` = 묶음② anchor spec 9D, `[:, :, 12:140]` = 묶음③ ctx broadcast 128D, `[:, :, 140:150]` = 묶음④ interactions 10D (§6.1 NOTE B 의 slice index 표 참조).
    - `seq_feat = seq_builder.build_seq_feat(X, anchors, quantiles)` — returns (N, 7, 95) float32.
    - `seq_8stat = compress_seq_8stat(seq_feat)` → (N, 760).
    - block ②: `ctx = cand_feat_150[:, 0, 12:140]` (anchor 0 row 의 128D — 14 row 모두 동일 broadcast 이므로 첫 row 사용; smoke test 에서 `np.allclose(cand_feat_150[:, k, 12:140], ctx)` for all k 검증).
    - block ③: `per_anchor_22 = np.concatenate([cand_feat_150[:, :, 0:3], cand_feat_150[:, :, 3:12], cand_feat_150[:, :, 140:150]], axis=2)` → (N, 14, 22).

- `analysis/plan-025/run_oof.py`:
  - `CELL_CONFIGS: dict[str, dict]` = `{"C1": {...}, "C2": {...}}` (§3.5 hparam)
  - `run_oof_plan025(cell_id: str, n_folds: int = 5, seed: int = 20260522) -> dict`: returns `{"hit_1cm": float, "hit_1p5cm": float, "top1_acc": float, "max_class_ratio": float, "per_fold": list[dict], "runtime_s": float, "best_iteration_per_fold": list[int] | None}` (C1: `best_iteration_per_fold=None`, C2: `list[int]` length=5). `per_fold[i]` schema = `{"fold": int (0..4), "n_test": int, "hit_1cm": float, "hit_1p5cm": float, "top1_acc": float, "max_class_ratio_fold": float, "runtime_s_fold": float, "best_iteration": int | None}`. 모든 key string-safe 표기 (dot 없음).
  - `_normalize_p022_result(d: dict) -> dict`: plan-022 `run_oof_cell` 반환 dict 의 key 를 본 plan key 로 정규화. **변환 매핑 표 (lock-in)**:

| plan-022 (legacy) key | 본 plan (canonical) key |
|:--|:--|
| `hit_1cm` | `hit_1cm` (no change) |
| `hit_1.5cm` | `hit_1p5cm` |
| `hit_15mm` | `hit_1p5cm` |
| `hit_at_1cm` | `hit_1cm` |
| `hit_at_1.5cm` | `hit_1p5cm` |
| `top1_acc` | `top1_acc` (no change) |
| `max_class_ratio` | `max_class_ratio` (no change) |

  매핑 표에 없는 key 는 그대로 carry (no-op). dict-in dict-out, side-effect 없음.
  - CLI: `python -m analysis.plan_025.run_oof --cell {C1, C2}` (또는 동등).

### §4.3 plan-022 / 021 / 020 module reuse (importlib)

plan-022 `selector_only_model.py:8~37` 의 importlib pattern 정확 carry. **정정 path 표현 (본 plan 본문 박제, plan-022 pattern 의 path resolve 식 동일)**:
- `_THIS = Path(__file__).resolve().parent`  # = `analysis/plan-025/` 디렉토리 (build_feat_1080.py 가 있는 곳)
- `_REPO = _THIS.parent.parent`  # = repo root
- `_PLAN020 = _THIS.parent / "plan-020"`  # = `analysis/plan-020/`
- `_PLAN021 = _THIS.parent / "plan-021"`  # = `analysis/plan-021/`
- `_PLAN022 = _THIS.parent / "plan-022"`  # = `analysis/plan-022/`
- `_PLAN024 = _THIS.parent / "plan-024"`  # = `analysis/plan-024/` (c2 cherry-pick 후)

추가: plan-024 module 도 동일 패턴으로 import:
- `importlib.util.spec_from_file_location(...)` 으로 `cand_builder`, `seq_builder`, `torsion_calc`, `quantile_carry`, `multiwindow_trim_build`, `anchor_vocab` 6 module 로드.

### §4.4 LgbmSelectorOnly hparam override (C2 대응)

plan-022 `LgbmSelectorOnly.__init__(K: int)` 은 hparam hard-coded → C2 의 adjusted hparam 주입 방식은 **선택 A 단일 default 로 고정**:

- **선택 A (default, 본 plan 결정 박제)**:
  - plan-022 `LgbmSelectorOnly(K=14)` 객체 생성 → `model.clf.set_params(...)` 으로 5 hparam attribute 직접 override.
  - `model.fit(X_tr2, q_tr2, eval_set=[(X_val, q_val)], early_stopping_rounds=100)` 시도.
  - plan-022 source 미수정.
- **fallback (선택 B, 자동 trigger)**:
  - 선택 A 의 `fit(eval_set=...)` 시도가 `TypeError` 발생 (plan-022 `LgbmSelectorOnly.fit` signature 가 eval_set 인자 미지원) 시 → §6.2 의 try/except 블록이 자동 fallback (early_stopping 포기, `model.fit(feat_train, q_train)` default 로 진행).
  - fallback 발생 시 `decision-note: spec-default — C2 early_stop fallback to default fit (plan-022 fit signature 가 eval_set 미지원)` 박제. H3 측정만 영향 (5 hparam 중 early_stopping 1개 drop), G3 영향 X.
  - **plan-022 source read 없이** 자동 분기 — 본 plan 본문 self-contained 유지.

→ 선택 A 가 본 plan 의 default. 선택 B 는 fallback (try/except 으로 자동 결정), 실행자 수동 결정 불필요.

### §4.5 tests (c5)

```python
# tests/test_plan025_smoke.py
import numpy as np

def test_imports():
    """plan-025 + plan-024 cherry-pick + plan-022 + plan-021 + plan-020 module 모두 import."""

def test_block_dims():
    """BLOCK_DIMS sum = 1080."""

def test_compress_seq_8stat_shape():
    """compress_seq_8stat (N=5, 7, 95) → (5, 760). 8 stat 모두 closed-form 검증."""

def test_compress_seq_8stat_invariants():
    """last == seq[:, -1, :], first == seq[:, 0, :], range == max − min."""

def test_build_feat_1080_shape():
    """build_feat_1080 (N=5, 11, 3) → (5*14, 1080)."""

def test_lgbm_K14_fit_predict_smoke():
    """LgbmSelectorOnly(K=14) + 1080D dummy input 위 fit/predict 정상."""

def test_soft_label_sum_one():
    """build_soft_label_with_tau output row-sum=1 (numerical tolerance 1e-6)."""

def test_f0_baseline_carry():
    """plan-020 f0_baseline + plan-022 ANCHORS_A6 import 정상."""

def test_quantile_carry_apply():
    """build_train_quantiles + apply_quantiles roundtrip 정상."""

def test_anchor_vocab_codebook_eq_A6():
    """plan-024 anchor_vocab.ANCHORS 가 plan-022 ANCHORS_A6 와 element-wise 정확 일치
    (NOTE A2 의 codebook consistency invariant 검증)."""
    # assert np.allclose(plan024.anchor_vocab.ANCHORS, plan022.anchors.ANCHORS_A6, atol=1e-6)

def test_block2_subblock_order():
    """cand_builder 묶음③ 128D = base 12 + macro_stat 8 + Bz/Tz 2 + regime 18 + STA/LTA 3 +
    Multi-window 60 + WAP5 + wingbeat 3 + f0_conf 2 + Peak 12 + v_autocorr 3 순서 검증.
    sub-block boundary index = [0, 12, 20, 22, 40, 43, 103, 108, 111, 113, 125, 128].
    각 sub-block index 위 NaN/Inf 부재 + dim 합산 = 128 검증."""
```

≥ 10 test (8 → 10). 모두 green = G0 합격.

---

## §5. STAGE 1 — F0 + plan-022 winner reproduce (c6, G1)

### §5.1 실행

```python
# analysis/plan-025/run_oof.py (G1 mode)
from analysis.plan_020 import baseline_f0
from analysis.plan_022.run_oof import run_oof_cell as run_oof_p022
from analysis.plan_022.anchors import ANCHORS_A6

# G1 a: F0 baseline 5-fold concat OOF
F0_pred = baseline_f0.f0_baseline(X, end_idx=10)
hit_1cm_F0 = (np.linalg.norm(F0_pred - gt, axis=1) <= 0.01).mean()
hit_1.5cm_F0 = (np.linalg.norm(F0_pred - gt, axis=1) <= 0.015).mean()
assert 0.6315 <= hit_1cm_F0 <= 0.6325
assert 0.8028 <= hit_1.5cm_F0 <= 0.8038

# G1 b: plan-022 winner reproduce
# seed layer 명시 (NOTE C): plan-022 reproduce 은 plan-022 의 LGBM `random_state` (= 20260519, plan-022 baseline_carry 확정)
# 만 사용. 5-fold split 자체는 `stable_fold_id(sample_id_int) % 5` deterministic MD5 라 seed 영향 없음.
# 본 plan 의 신규 C1/C2 cell 은 별개 seed = 20260522 (§3.5) — plan-022 reproduce 와 *분리된 layer*.
result_p022_winner = run_oof_p022(
    anchor_name="A6_bcc14",
    tau_cls=0.001,
    n_folds=5,
    seed=20260519,  # plan-022 LGBM random_state carry (G1 reproduce 전용)
)
# **dict key naming convention (본 plan 박제)**: 본 plan 의 모든 metric dict key 는 *string-safe* 표기 사용 —
#   `hit_1cm`, `hit_1p5cm` (dot 대신 `p`), `top1_acc`, `max_class_ratio`, `runtime_s`, `best_iteration_per_fold`.
# plan-022 `run_oof_cell` 의 실제 반환 key 가 다른 표기 (예: `hit_15mm`) 일 경우 wrapper layer 가 본 plan key 로 변환.
# 본 wrapper 는 `analysis/plan-025/run_oof.py` 안에 `_normalize_p022_result(d: dict) -> dict` 로 명시 (c4 작성).
hit_1cm_p022 = result_p022_winner["hit_1cm"]
hit_1p5cm_p022 = result_p022_winner.get("hit_1p5cm", result_p022_winner.get("hit_1.5cm"))
assert 0.6523 <= hit_1cm_p022 <= 0.6533
assert 0.8099 <= hit_1p5cm_p022 <= 0.8109

# baseline_carry.json 박제
json.dump({
    "F0": {"hit_1cm": hit_1cm_F0, "hit_1.5cm": hit_1.5cm_F0},
    "plan022_winner": result_p022_winner,
    "dataset_hash": "<from plan-022 baseline_carry.json>",
    "plan024_module_carry_commit": "915dd26",
}, f, indent=2)
```

### §5.2 G1 합격 (자동)

- F0 hit ∈ tight band ✓
- plan-022 winner hit ∈ tight band ✓
- dataset_hash = plan-022 carry ✓
- 위반 1 = severe halt (f0_reproduce_drift / plan022_reproduce_drift)

---

## §6. STAGE 2 — Sub-exp C1 / C2 (c7~c8, G2.C1 / G2.C2)

### §6.1 Input spec (1080D per row)

| Block | Source | Dim | Sample-level vs Per-anchor |
|:--|:--|--:|:--|
| ① | `build_input_common(X, f0_baseline) + build_input_lgbm_extra(X, L1)` (plan-021) | 170 | sample-level (anchor row 14개 broadcast) |
| ② | `build_cand_feat(X, anchors, quantiles)` 의 묶음③ slice (= base 12 + macro_stat 8 + Bz/Tz 2 + regime 18 + A1 STA/LTA 3 + A2 Multi-window 60 + A5 WAP-5 + A6 wingbeat 3 + A8 f0_conf 2 + A10 Pct-roll+Peak 12 + A12 v_autocorr 3) | 128 | sample-level (anchor row 14개 broadcast) |
| ③ | `build_cand_feat` 의 묶음①(par/perp/dist 3) + 묶음②(anchor spec 9) + 묶음④(interactions 10) | 22 | per-anchor (14 row 각 다름) |
| ④ | `build_seq_feat(X, anchors, quantiles)` → (N, 7, 95) → `compress_seq_8stat` → (N, 760) | 760 | sample-level (anchor row 14개 broadcast) |
| **Total** | | **1080** | sample × anchor row-expand |

**8-stat 산식 (block ④, per-channel c ∈ 0..94)**:
- `last_c = seq[:, -1, c]` (t=10)
- `first_c = seq[:, 0, c]` (t=4)
- `mean_c = seq[:, :, c].mean(axis=1)`
- `std_c = seq[:, :, c].std(axis=1)`
- `slope_c = sum((t - t_mean) * (seq - seq_mean)) / sum((t - t_mean)^2)` — closed-form linear regression coefficient. **t-grid 명시: `t = np.arange(7, dtype=np.float32)`** (= [0, 1, 2, 3, 4, 5, 6], unit-spaced, time-step unit, sec 변환 X). LGBM 은 affine invariant 라 단위 결과 영향 없음 — reproducibility 박제 용도.
- `max_c = seq[:, :, c].max(axis=1)`
- `min_c = seq[:, :, c].min(axis=1)`
- `range_c = max_c - min_c`

→ 95 channel × 8 stat = 760, stack 순서 = `[last_0..last_94, first_0..first_94, ..., range_0..range_94]` (stat-major).

**row-expand 산식**:
- block ① ② ④ = sample-level (N, D_block) → `np.repeat(., K=14, axis=0)` → (N*14, D_block)
- block ③ = per-anchor (N, K=14, 22) → `reshape(N*14, 22)` (row 순서 sample-major: row i*K + k = sample i, anchor k)
- concat axis=1 → (N*14, 1080).

**NOTE A (`quantiles: QuantileCarry` schema)** — plan-024 carry `quantile_carry.py:QuantileCarry`:
- `omega_p90: float` — train fold 위 ‖ω_Frenet‖ (각속도) 의 90% quantile. cand_builder 묶음③ A1 STA/LTA + 묶음③ A12 v_autocorr threshold 주입.
- `jerk_p90: float` — train fold 위 ‖jerk_Frenet‖ 의 90% quantile. cand_builder 묶음③ A10 Pct-rolling+Peak 의 sharp-turn / jerk peak threshold 주입.
- `levy_tail_threshold: float` — train fold 위 ‖Δp‖ 의 95% quantile (plan-024 v1.1-rev2 미사용, 본 plan 도 미사용 — schema 호환성 위해 carry).

산출 식: `quantiles = build_train_quantiles(X_train)` (plan-024 `quantile_carry.py` carry). `X_train` (N_train, 11, 3). 5-fold 의 매 fold 의 train portion (8000 sample) 위에서 새로 산출 후 test fold 의 cand_feat / seq_feat 산출에 그대로 주입 (fold-leakage 차단, plan-024 §3.6 carry).

**NOTE A2 (`seq_builder` 내부 `anchor_vocab` 의 codebook + fold-leakage 정책)**:
- seq_builder 의 channel 26-39 (F: anchor-vocab soft 14D) + 41-54 (H: top1 one-hot 14D) 는 plan-024 `anchor_vocab.build_anchor_vocab` 호출 → past step F0 residual 의 14-anchor soft assignment.
- **codebook = plan-022 `ANCHORS_A6` (K=14 BCC) 와 동일** (= 본 plan §3.4 anchor codebook 정확 동일). plan-024 commit 915dd26 시점에 `anchor_vocab.py` 의 ANCHORS 가 plan-022 A6 BCC 와 일치하도록 carry 됨 (c2 cherry-pick 후 smoke test 에서 `np.allclose(plan024.anchor_vocab.ANCHORS, plan022.anchors.ANCHORS_A6)` 검증 필요 — c5 test_imports 안에 추가).
- **fold-leakage 정책**: anchor_vocab 의 codebook 은 *deterministic constant* (label 미사용) → fold-leakage 무관. soft assignment 산출 식 (softmax(-‖a_k - residual‖ / τ_past)) 의 τ_past=0.003 도 constant (label 미사용). 즉 seq_builder 내부 anchor_vocab 은 train/test fold 위 동일 codebook + 동일 τ → no leakage.
- 단 `quantile_carry.omega_p90 / jerk_p90` 만 train fold 위 산출 (label 미사용이나 fold-dependent 통계량) — 위 산출 식 그대로.

**NOTE B (`build_cand_feat` output slice index 표)** — plan-024 `cand_builder.py:build_cand_feat` returns `(N, K=14, 150)` float32:

| Output dim slice | 묶음 | 내용 | sample-level vs per-anchor |
|:--|:--|:--|:--|
| `[:, :, 0:3]` | 묶음① par/perp/dist | anchor k 와 last F0 residual r_last 의 Frenet 분해 (normalized by speed×horizon) | per-anchor |
| `[:, :, 3:12]` | 묶음② anchor spec 9D | Frenet coord 3 + sign 3 + group 2 (axis vs corner) + idx scalar 1 | anchor-static (sample 무관) |
| `[:, :, 12:140]` | 묶음③ ctx broadcast 128D | base 12 + macro_stat 8 + Bz/Tz 2 + regime 18 + A1 STA/LTA 3 + A2 Multi-window 60 + A5 WAP-5 + A6 wingbeat 3 + A8 f0_conf 2 + A10 Pct-roll+Peak 12 + A12 v_autocorr 3 | sample-level (14 anchor row 동일 broadcast) |
| `[:, :, 140:150]` | 묶음④ interactions 10D | base 8 scalar (anchor·res / anchor·v / anchor·acc / anchor·EWMA / corner×turn / sign-agreement / physics-extrap·anchor / anchor·Δz_world) + A3 BCC adjacency 2 scalar | per-anchor |

→ block ② = `[:, 0, 12:140]` (anchor 0 row 의 128D, sample-major broadcast 후 14× repeat). block ③ = `concat([0:3], [3:12], [140:150], axis=2)` = (N, 14, 22).

smoke test (c5) 에서 `np.allclose(cand_feat_150[:, k, 12:140], cand_feat_150[:, 0, 12:140])` for all k ∈ 0..13 검증 (sample-level broadcast invariant 확인).

### §6.2 Per-cell 5-fold OOF 식

```python
# §6.2 — per cell (C1 or C2)
for fold in 0..4:
    train_idx, test_idx = stable_fold_id(...) == fold filter
    X_train, X_test = X[train_idx], X[test_idx]
    gt_train, gt_test = gt[train_idx], gt[test_idx]

    # block ① + ② + ③ + ④ 모두 train fold 위 산출 (quantile carry 만 train 기반)
    quantiles = build_train_quantiles(X_train)
    feat_train = build_feat_1080(X_train, ANCHORS_A6, f0_baseline, quantiles)
    feat_test = build_feat_1080(X_test, ANCHORS_A6, f0_baseline, quantiles)
    # feat_*: (N_*, 1080) — 내부에서 row-expand 처리 후 (N_* * 14, 1080) 반환

    # Frenet basis + F0 prediction (soft label + predict 모두에 필요)
    # plan-021 build_input.py 의 build_frenet_basis_3d 정확 carry — end_idx = T-1 = 10
    R_wfn_train = build_frenet_basis_3d(X_train, end_idx=10)  # (N_train, 3, 3), columns=[t̂, n̂, b̂]
    R_wfn_test  = build_frenet_basis_3d(X_test,  end_idx=10)  # (N_test, 3, 3)
    F0_train = f0_baseline(X_train, end_idx=10)  # (N_train, 3) world frame, 80ms 미래 예측
    F0_test  = f0_baseline(X_test,  end_idx=10)  # (N_test, 3) world frame

    # soft label (plan-022 carry) — ANCHORS_A6 은 Frenet 좌표, residual_true = gt - F0, soft = softmax(-‖a_k - residual_frenet‖ / τ)
    q_train = build_soft_label_with_tau(gt_train, R_wfn_train, F0_train, ANCHORS_A6, tau_cls=0.001)

    # model — C1 / C2 모두 plan-022 LgbmSelectorOnly carry. C2 만 hparam override + early_stopping.
    from sklearn.model_selection import train_test_split  # standard helper
    assert cell in ("C1", "C2"), f"unsupported cell: {cell}"
    model = LgbmSelectorOnly(K=14)  # plan-022 default constructor (n_estimators=500, lr=0.05, num_leaves=63)
    if cell == "C2":
        # C2 hparam override (§3.5): 5 hparam 동시 변경
        model.clf.set_params(
            n_estimators=2000,
            learning_rate=0.03,
            colsample_bytree=0.7,    # = feature_fraction
            min_child_samples=50,    # = min_data_in_leaf
        )
        # inner val split for early_stopping (§3.1 carry — sample 단위 split 후 row-expand)
        # feat_train 은 이미 (N_train*K, 1080) row-expanded → q_train (N_train, 14) 의 argmax(1) 로 sample 단위 stratify,
        # 결과 train_idx_inner / val_idx_inner 를 *sample index* 로 받아 row-expand 의 (sample_i*K)..(sample_i*K+K-1) 14 row 슬라이스 적용.
        N_train_inner = q_train.shape[0]
        idx_all = np.arange(N_train_inner)
        idx_tr2, idx_val = train_test_split(
            idx_all,
            test_size=0.20,
            stratify=q_train.argmax(axis=1),       # (N_train,) hard-argmax label
            random_state=20260522,                  # §3.5 inner-val seed (= C2 LGBM random_state 동일)
        )
        # row-expand index 확장: sample i → row i*K .. i*K+K-1
        row_idx_tr2 = (idx_tr2[:, None] * 14 + np.arange(14)[None, :]).ravel()
        row_idx_val = (idx_val[:, None] * 14 + np.arange(14)[None, :]).ravel()
        X_tr2, X_val = feat_train[row_idx_tr2], feat_train[row_idx_val]
        q_tr2, q_val = q_train[idx_tr2], q_train[idx_val]
        # plan-022 LgbmSelectorOnly.fit signature 가 eval_set / early_stopping_rounds 미지원 시
        # decision-note "early_stop_fallback" 박제 + C2 도 C1 처럼 default fit 으로 fallback (그 경우 H3 측정만 영향, G3 영향 X).
        try:
            model.fit(X_tr2, q_tr2, eval_set=[(X_val, q_val)], early_stopping_rounds=100)
        except (TypeError, ValueError) as e:
            # fallback: plan-022 fit signature 가 eval_set 미지원 (TypeError) OR LightGBM eval_set 의 soft label
            # (N_val, 14) 가 multiclass 표준 y (1D class index) 와 shape mismatch (ValueError) 인 경우 자동 fallback.
            # 둘 다 §4.4 선택 B 로 자동 전환 (= early_stopping 포기, default fit). H3 측정만 영향, G3 영향 X.
            model.fit(feat_train, q_train)
            warnings.warn(f"C2 early_stopping fallback to default fit ({type(e).__name__}) — decision-note: early_stop_fallback")
    else:
        # C1 — plan-022 default carry, 추가 hparam override 없음
        model.fit(feat_train, q_train)

    # predict — row-expand selector 의 정확한 predict 식 (plan-022 carry, 본 plan 자족 표현):
    #   1) probs_test_expanded = model.predict_proba(feat_test)   # (N_test*14, 14) — row-expanded
    #   2) row order = sample-major (row i*K + k = sample i, anchor k 의 X feature). predict 결과의
    #      "row i*K + k" 의 K-dim 확률 분포 중 **k 번째 element (= 자기 anchor 위 prob)** 만 추출
    #      → selector design 의 self-consistency: sample i 의 anchor k 에 대한 selector score
    #      = probs_test_expanded[i*K + k, k]
    #   3) probs_sel = probs_test_expanded[np.arange(N_test*K), np.tile(np.arange(K), N_test)]
    #      → reshape (N_test, K). 이게 sample 별 14-anchor 의 selector 확률.
    probs_test_expanded = model.predict_proba(feat_test)              # (N_test*14, 14)
    K = 14
    sample_idx = np.repeat(np.arange(len(X_test)), K)                 # (N_test*K,)
    anchor_idx = np.tile(np.arange(K), len(X_test))                   # (N_test*K,)
    probs_sel = probs_test_expanded[np.arange(len(X_test)*K), anchor_idx].reshape(len(X_test), K)
    probs_sel = probs_sel / probs_sel.sum(axis=1, keepdims=True)      # row-normalize (정합성)

    # Frenet → world 변환: anchor (Frenet 좌표) → world residual via R_wfn → world prediction
    #   residual_frenet = Σ_k probs_sel[i, k] * ANCHORS_A6[k]          # (N_test, 3) Frenet
    #   residual_world  = einsum("nij,nj->ni", R_wfn_test, residual_frenet)  # (N_test, 3) world
    #   final_pred      = F0_test + residual_world                     # (N_test, 3) world
    residual_frenet = (probs_sel[:, :, None] * ANCHORS_A6[None, :, :]).sum(axis=1)    # (N_test, 3)
    residual_world  = np.einsum("nij,nj->ni", R_wfn_test, residual_frenet)            # (N_test, 3)
    final_pred      = F0_test + residual_world                                          # (N_test, 3)

    # per-fold 누적 (fold 마다 X_test / gt_test 의 sample 순서를 정확히 기억하여 OOF concat)
    oof_pred[test_idx] = final_pred                              # (N_total=10000, 3) world frame
    oof_probs_sel[test_idx] = probs_sel                          # (N_total, K=14)
    if cell == "C2":
        oof_best_iter[fold] = model.clf.best_iteration_          # int per fold

# ── 5-fold concat OOF metric (§3.3 산식) ──
hit_1cm    = (np.linalg.norm(oof_pred - gt_all, axis=1) <= 0.01).mean()
hit_1p5cm  = (np.linalg.norm(oof_pred - gt_all, axis=1) <= 0.015).mean()
top1_acc   = (oof_probs_sel.argmax(axis=1) == gt_anchor_label_all).mean()  # gt_anchor_label = argmin_k ‖a_k - residual_true_frenet‖
max_class_ratio = oof_probs_sel.mean(axis=0).max()                # mode-collapse 진단
# paired Δ
delta_1cm_vs_F0    = hit_1cm   - hit_1cm_F0_g1a
delta_1p5cm_vs_F0  = hit_1p5cm - hit_1p5cm_F0_g1a
delta_1cm_vs_p022  = hit_1cm   - hit_1cm_p022_g1b
delta_1p5cm_vs_p022 = hit_1p5cm - hit_1p5cm_p022_g1b
```

### §6.3 Per-cell 실행 (c7, c8)

- **c7 G2.C1**: `python -m analysis.plan_025.run_oof --cell C1 --seed 20260522 > c7_run.log 2>&1`. 5-fold OOF, K=14 BCC + τ=0.001, hparam default. 예상 runtime 1.5~3h CPU. 산출 `results_C1.json`.
- **c8 G2.C2**: `--cell C2`. 동일 input. 예상 runtime 2~5h CPU (early_stopping 영향). 산출 `results_C2.json`.

### §6.4 G2.C{n} 합격 (per cell)

- metric finite ✓ (NaN/Inf X)
- `max_class_ratio < 0.95` ✓ (= `probs_all.mean(axis=0).max()`)
- C2 only: `best_iteration ∈ [50, 2000]` per fold ✓ (5 fold 모두)
- 위반 1 = severe (lgbm_numerical / soft_label_collapse / early_stop_outlier)

---

## §7. STAGE 3 — Paradigm analysis (c9, G3)

### §7.1 2 cell 표 산출

| Cell | hit@1cm | hit@1.5cm | Δ_1cm vs F0 | Δ_1.5cm vs F0 | Δ_1cm vs p022 winner | Δ_1.5cm vs p022 winner | max_class_ratio | top1_acc | oracle 회수율 | runtime |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| C1 (default) | ?.???? | ?.???? | +?.???? | +?.???? | +?.???? | +?.???? | ?.??? | ?.???? | ?.??% | ?h |
| C2 (adjusted) | ?.???? | ?.???? | +?.???? | +?.???? | +?.???? | +?.???? | ?.??? | ?.???? | ?.??% | ?h |

### §7.2 Best cell selection

- `best_cell = argmax_{C1, C2} hit@1cm`
- tiebreaker: hit@1.5cm.
- `best_hit_1cm`, `best_hit_1.5cm` 박제 (frontmatter sync 용).

### §7.3 Paradigm finding (G3 판정)

- `best_hit_1cm > 0.6700` → G3 **PASS** (band=positive).
- `0.6528 ≤ best_hit_1cm ≤ 0.6700` → G3 **partial_lift** (band=partial). lift 잠재력 박제 + plan-026 (block ablation) priority.
- `best_hit_1cm < 0.6528` → G3 **regression** (band=negative). H1 FAIL — paradigm-level finding "1080D LGBM 의 selector capacity ↑ 가 plan-022 170D 대비 lift 미미" 박제 + plan-026 hparam grid 확장 priority.

**Attribution boundary 박제 (best_cell == C2 인 경우)**: best_cell 이 C2 이고 G3 PASS / partial_lift 일 때, 그 lift 는 *(input 1080D 확장) + (hparam 5개 adjust)* 의 합산 효과. H1 의 *input 단독 attribution* 측정은 **C1 cell 측정** 만으로 정의됨 (= `hit_C1 - hit_p022_reproduce ≥ +0.005`). 따라서 results.md §7.5 의 H1 결과 표는 best_cell 과 무관하게 *C1 cell 위 lift* 도 함께 박제 (best_cell == C2 이어도 C1 측정 별도 행). 이 분리 박제는 plan-026 후보 결정 시 (input lever 한계 vs hparam tuning 잠재력) attribution lever 가 됨.

### §7.4 14-anchor oracle 회수율

- `oracle_recovery = best_hit_1cm / 0.7928`
- plan-024 측정 carry. 본 plan 의 lift 잠재력 평가 anchor.

### §7.5 H3 검증 (C1 vs C2)

- `Δ_C2_vs_C1 = hit_C2 - hit_C1`
- `Δ ≥ +0.003` → H3 **PASS** (1080D 대응 hparam adjust *합산 효과* 검증). **NOTE**: C2 의 hparam 변경 5개 동시 → 개별 hparam attribution 불가능. 본 plan 의 H3 의도는 "5 hparam 합산 lift 의 PASS/FAIL" 만, "어느 hparam 이 dominant" 분리 검증은 **plan-026 hparam grid 후보**.
- `Δ < +0.003` → H3 **partial** (default hparam 이 1080D 도 충분 capacity, 또는 input lever 가 dominant). 동일 — attribution 분리는 plan-026 후보.

---

## §8. STAGE 4 — G_final (c10)

### §8.1 산출

- `analysis/plan-025/results.md` (11 항목):
  1. plan_id / version / date / status / band / best_cell
  2. G-gate 표 (G0~G_final 모두 [DONE])
  3. 2 cell 결과 표 (§7.1)
  4. Best cell 박제 + paired Δ (vs F0, vs plan-022 winner)
  5. H1 / H2 / H3 검증 결과
  6. 14-anchor oracle 회수율
  7. 1080D input block 분해 표 (G3 PASS 시 잠재력, FAIL 시 한계)
  8. Runtime 박제
  9. max_class_ratio + top1_acc + (C2 만) best_iteration_per_fold
  10. Follow-up plan 후보 (plan-026/027/028)
  11. Cross-refs (plan-022, plan-024, memory)
- `plans/plan-025-candidate-concat-input-max.results.md` pair
- 3-file frontmatter sync (plan-025 spec frontmatter + results.md + analysis/plan-025/results.md)

### §8.2 G_final 합격

- 3-file frontmatter sync ✓
- §0.5 c1~c10 모두 [DONE] ✓
- follow-up plan 후보 ≥ 3 건 박제 ✓

---

## §9. Out of scope (명시적으로 안 함)

- corrector reg head 재투입 (plan-021 dead 결론 carry, plan-023 followed_by 후보 plan-024 가 별도 paradigm 으로 시도 후 FAIL)
- GRU sub-exp / cross-attention 재시도 (plan-024 G_final band=negative carry)
- LB 측정 / DACON submit (G3 PASS 시 plan-026 또는 별도 plan)
- ensemble (plan-027 후보)
- anchor layout 변경 (K=14 BCC fix)
- τ_cls 변경 (0.001 fix)
- anchor radius ≠ 0.005m
- block ②③④ each-out ablation (plan-026 후보, G3 결과 보고 결정)
- F0 baseline ML 화 (plan-028 후보)
- N ≠ 14 anchor (plan-023 sweep 영역)
- soft label τ_loss 분리 (plan-024 carry, 본 plan 미사용)

---

## §10. 참조 (read-only — path blacklist 예외)

- `plans/plan-022-corrector-free-anchor-layout-sweep.md` — winner cell A6_bcc14_tau001 spec
- `plans/plan-022-corrector-free-anchor-layout-sweep.results.md` — winner hit 0.6528 / 0.8104
- `plans/plan-023-large-n-anchor-sweep.md` — anchor large-N marginal lift 결론
- `plans/plan-024-cross-attention-anchor-vocab.md` (worktree-plan-024-spec branch) — cross-attention paradigm FAIL + 16 lever FE 박제
- `analysis/plan-022/{anchors.py, selector_only_model.py, run_oof.py, baseline_carry.json}`
- `analysis/plan-021/{build_input.py, dual_head_model.py}`
- `analysis/plan-020/{baseline_f0.py, baseline_oof.json}`
- `analysis/plan-024/{cand_builder.py, seq_builder.py, torsion_calc.py, quantile_carry.py, multiwindow_trim_build.py, anchor_vocab.py, multiwindow_trim.json}` (c2 cherry-pick 후)
- `src/{io.py, pb_0_6822/selector.py}` (data loader + stable_fold_id)
- memory `project_next_plan_direction.md` (2026-05-22 user 한 줄 재정의 + input 1080D 박제)

---
