---
plan_id: 028
version: 1
date: 2026-05-22 (Asia/Seoul)
status: all_complete
best_cell: B3
best_hit_1cm: 0.6509
best_hit_1p5cm: 0.8118
best_delta_1cm: 0.0189
best_delta_1p5cm: 0.0085
band: partial
g0_completed: true
g1_completed: true
g2a_completed: true
g2b_completed: skip
g3_completed: true
g_final_completed: true
inspired_by:
  - 025 (1080D LGBM C1+C2 모두 hit_1cm=0.6320=F0 mode collapse — best=C1 0.6320/0.8033, band=negative, max_class_ratio≈1/14, oracle 회수율 79.72%. paradigm_analysis §4 4가설 (a) τ_cls / (b) sample-weight expansion / (c) subclass self-consistency / (d) 1058D broadcast / 22D per-anchor 비율 50:1 dominance 박제. (d) most likely.)
  - 022 (best A6_bcc14_tau001 → hit_1cm 0.6531 / hit_1p5cm 0.8108 — K=14 BCC + τ_cls=0.001 + 170D LGBM selector paradigm. 본 plan **이겨야 할 목표**.)
  - 020 (F0 baseline 0.6320 / 0.8033 + 5-fold stable_fold_id MD5. 본 plan 의 lower-bound reference.)
  - 024 (14-anchor oracle 0.7928 ceiling 박제 + feature engineering module 6개 source. plan-025 가 carry 한 동일 module 본 plan 도 carry.)
code_reuse:
  - module: analysis/plan-025/build_feat_1080.py
    symbols: [build_feat_1080, build_block1, build_block2, build_block3, build_block4]
    reason: 1080D feature pipeline 그대로 carry. 본 plan 의 block ablation 은 build_feat_1080 output 의 *index slice* 로 구현 (재계산 불필요). plan-025 worktree (worktree-plan-025-spec, commit @plan-025 G2.C2 시점) 에서 cherry-pick.
  - module: analysis/plan-025/run_oof.py
    symbols: [run_oof_cell_1080]
    reason: 5-fold OOF runner. 본 plan 의 cell config 는 이 runner 위에 (a) input dim subset + (b) sample-weight expansion on/off flag 만 inject. plan-025 worktree 에서 cherry-pick.
  - module: analysis/plan-024/cand_builder.py
    symbols: [build_cand_feat]
    reason: block ②③ (ctx 128D + per-anchor 22D) builder. plan-025 carry 그대로. cherry-pick 대상 (c2 단계, plan-025 와 동일 path).
  - module: analysis/plan-024/seq_builder.py
    symbols: [build_seq_feat]
    reason: block ④ (95×7 raw, 8-stat 압축으로 760D) source. cherry-pick 대상.
  - module: analysis/plan-024/torsion_calc.py
    symbols: [build]
    reason: Frenet torsion τ scalar. seq_builder internal call.
  - module: analysis/plan-024/quantile_carry.py
    symbols: [QuantileCarry, build_train_quantiles, apply_quantiles]
    reason: train fold quantile carry (omega_p90, jerk_p90). cand_builder threshold 주입.
  - module: analysis/plan-024/multiwindow_trim_build.py
    symbols: [load_trim]
    reason: 144→60 Multi-window trim index.
  - module: analysis/plan-024/anchor_vocab.py
    symbols: [build_anchor_vocab]
    reason: seq_builder internal call.
  - module: analysis/plan-022/selector_only_model.py
    symbols: [LgbmSelectorOnly, build_soft_label_with_tau]
    reason: row-expand LGBM K-class softmax + soft label 산식. K=14 BCC + τ=0.001 carry. 본 plan model 그대로 — single 변수 = input dim subset + sample-weight flag.
  - module: analysis/plan-022/anchors.py
    symbols: [ANCHORS_A6, LAYOUT_NAMES]
    reason: K=14 BCC anchor codebook.
  - module: analysis/plan-022/run_oof.py
    symbols: [run_oof_cell]
    reason: per-cell 5-fold OOF runner. plan-022 winner (G1.b reproduce) 용.
  - module: analysis/plan-021/build_input.py
    symbols: [build_frenet_basis_3d, to_frenet, build_input_common, build_input_lgbm_extra]
    reason: 170D plan-022 input pipeline = block ① (170D).
  - module: analysis/plan-020/baseline_f0.py
    symbols: [f0_baseline, D1, PAR, PERP, R_HIT, R_HIT_LOOSE]
    reason: F0 baseline injection + paired Δ anchor + hit metric.
  - module: src/io.py
    symbols: [load_all_samples, load_labels]
    reason: data loader.
  - module: src/pb_0_6822/selector.py
    symbols: [stable_fold_id]
    reason: 5-fold stable split.
followed_by:
  - (gap 채움 사후 결정 — 본 plan G3 결과에 따라 input enrichment 정착 / paradigm 전환 / ensemble / F0 ML 중 어느 방향이 next 인지 박제 예정)
scope: plan-022 carry (K=14 BCC anchor + soft label 산식 + F0 + 5-fold split) + plan-025 carry (1080D feature pipeline = block ① 170D + ② 128D + ③ 22D + ④ 760D) 위, **단일 변수 5 축** = (1) input dim subset (B1/B2/B3, 가설 d) + (2) sample-weight 값 (W1, 가설 b) + (3) τ_cls value (T1/T2 = {0.01, 0.1}, 가설 a) + (4) model wrapping (S1 = base lightgbm.LGBMClassifier multiclass, 가설 c) + (5) block ④ 산식 (R1 = raw seq flatten 665D, 가설 e) = **G2.A 9 cell**. G2.A 결과 branch ((b)+(d) cell 기준) 에 따라 **G2.B = 1~2 추가 cell** (selector/hparam tweak). plan-022 winner 0.6531 lift 가 G3 합격 기준. corrector reg head / GRU / cross-attention / DACON submit / ensemble / anchor (K=14 BCC) 변경 / τ_cls {0.001, 0.01, 0.1} 외 추가 sweep / fold 변경 = out-of-scope.
exp_ids:
  - Z028_B1_anchor22
  - Z028_B2_combo192
  - Z028_B3_no_anchor1058
  - Z028_B4_full1080_ref
  - Z028_W1_weight_off
  - Z028_T1_tau01
  - Z028_T2_tau1
  - Z028_S1_base_lgbm
  - Z028_R1_seq_raw
  - Z028_Bx_branch (G2.B conditional, 1~2 exp_id 추가 — branch 확정 후 박제)
lb_score: null
---

# plan-028 v1 — Per-anchor Isolation × Sample-weight Probe (mode collapse 진단 + plan-022 lift)

## §0. 한 줄 목적

> **plan-025 mode collapse 의 5 가설 — (a) τ_cls=0.001 sharp soft label train/test gap / (b) sample-weight expansion (140k row × 14-class CE) 비효율 / (c) LgbmSelectorOnly subclass self-consistency 약화 / (d) 1058D broadcast / 22D per-anchor 50:1 LGBM split dominance / (e) seq 압축 lossy (block ④ 8-stat) — 전부 본 plan G2.A 9 cell 로 직접 검증한 뒤, winning configuration 위에서 G2.B 1~2 cell 로 plan-022 winner (hit_1cm 0.6531, hit_1p5cm 0.8108) 를 paired Δ > 0 로 lift.**
>
> **분석 축 (G2.A, 9 cell)**: plan-025 1080D feature pipeline 의 block 분해 ① 170D / ② 128D / ③ 22D / ④ 760D 위 — 각 cell 은 B4 baseline (1080D full + weight ON + τ=0.001 + LgbmSelectorOnly + block ④ 8-stat) 대비 single-variable 변경:
> - **B1 (22D = ③ only)** / **B2 (192D = ①+③)** / **B3 (1058D = no ③)** — input dim subset, 가설 (d)
> - **B4 (1080D = plan-025 C1 carry)** — baseline reference, c6 G1 재실행 금지
> - **W1 (1080D + sample-weight 1.0 균등)** — sample_weight 값, 가설 (b)
> - **T1 (1080D + τ_cls=0.01)** / **T2 (1080D + τ_cls=0.1)** — τ_cls (10×/100× softer), 가설 (a)
> - **S1 (1080D + base lightgbm.LGBMClassifier multiclass, LgbmSelectorOnly subclass 우회)** — model wrapping, 가설 (c)
> - **R1 (985D = ①+②+③+raw seq flatten 95×7=665D, block ④ 8-stat 우회)** — block ④ 산식, 가설 (e)
>
> **승부 축 (G2.B)**: G2.A 결과의 4 branch (α / β / γ / δ) 중 1 branch 활성화 → 1~2 cell 추가 실험. branch 결정 함수는 §4.5 박제. branch 활성화 우선순위 α > β > γ > δ (복수 branch 조건 만족 시 우선순위 높은 것 1개만 실행).
>
> **합격 기준 (G3)**: G2.A + G2.B 통합 best cell 의 hit_1cm > 0.6531 (= plan-022 winner) AND paired Δ vs plan-022 winner > 0 → **PASS (band=positive)**. 0.6320 < best ≤ 0.6531 → partial band (F0 초과 but plan-022 미달). best ≤ 0.6320 → negative band (plan-025 와 동일 = mode collapse 잔존).
>
> **plan-025 와 차별점**: plan-025 는 input dim 확장 lever, 본 plan 은 *동일 1080D 안에서의 subset / weight flag* lever. input pipeline 자체는 plan-025 carry (재계산 없음). single 변수 = (dim subset, weight flag, branch hparam) 중 cell 마다 하나만.
>
> **out-of-scope**: anchor layout 변경 (K=14 BCC fix) / τ_cls 변경 (0.001 fix) / fold 변경 / soft-label 산식 / F0 baseline ML화 / cross-attention / GRU / corrector head / 새 feature engineering / 1080D 외 새 dim / DACON submit / ensemble / plan-026 carry (worktree-only block ablation spec 의 carry 금지 — 본 plan self-contained).

---

## §0.5 Quick Reference (autonomous loop 매 turn 읽는 section)

### 합격 기준 (G-gate sequence)

- **G0**: 본 plan path (analysis/plan-028/**, tests/test_plan028_smoke.py) + cherry-pick path (analysis/plan-025/{build_feat_1080.py, run_oof.py}, analysis/plan-024/{6 module + 1 data + __init__}, analysis/plan-022/*, analysis/plan-021/build_input.py, analysis/plan-020/baseline_f0.py) import + smoke + tests green. 위반 시 `infra_drift` severe.
- **G1**: F0 baseline 5-fold concat OOF — hit_1cm ∈ [0.6315, 0.6325] AND hit_1p5cm ∈ [0.8028, 0.8038]. plan-022 winner reproduce — A6_bcc14 + τ=0.001 cell hit_1cm ∈ [0.6526, 0.6536] AND hit_1p5cm ∈ [0.8103, 0.8113]. plan-025 C1 carry — hit_1cm ∈ [0.6315, 0.6325] (= mode collapse reference). 위반 시 `f0_reproduce_drift` / `plan022_reproduce_drift` / `plan025_C1_drift` severe.
- **G2.A (9 cell)**: B1/B2/B3/B4/W1/T1/T2/S1/R1 각 5-fold OOF metric finite + `max_class_ratio` 측정 + soft label sum=1 invariant. T1/T2 는 τ_cls 변경 시 soft label 재계산. S1 은 LgbmSelectorOnly subclass 우회 (base lightgbm.LGBMClassifier 직접 fit). R1 은 block ④ 산식 변경 (8-stat 760D → raw flatten 665D, total 985D). 위반 시 `lgbm_numerical` severe.
- **G2.B (conditional, 1~2 cell)**: §4.5 branch 함수로 α/β/γ/δ 중 1 개 활성화, 해당 branch 의 1~2 cell 실행. branch 미정 (조건 모두 false) → δ default (selector arch MLP per-sample softmax) 1 cell.
- **G3 (paradigm-level)**: best_cell = argmax(hit_1cm over G2.A + G2.B 통합). best_hit_1cm > 0.6531 → PASS (band=positive). 0.6320 < best_hit_1cm ≤ 0.6531 → partial band (warn `partial_lift`). best ≤ 0.6320 → negative band (warn `regression`). 0.6526 ≤ best_hit_1cm ≤ 0.6536 = `tight_band_around_p022` 경계 — paired Δ 부호로 결정.
- **G_final**: results.md (12 항목 = §6 정합) + best cell 박제 (cell_id + hparam + 모든 metric + max_class_ratio + top1_acc + paired Δ vs F0/plan-022/plan-025-C1) + 14-anchor oracle 회수율 (= best / 0.7928) + paradigm_analysis (5가설 (a/b/c/d/e) 각 confirmed/rejected/inconclusive 박제) + follow-up plan 후보 ≥ 2 건 + 3-file frontmatter sync + final band ∈ {positive, partial, negative} (tight 은 §3.2 paired Δ 부호로 positive/partial 으로 resolve 후 final 박제, intermediate state).

### G-gates (commit 단위 milestone)

- G0: STAGE 0 인프라 + cherry-pick + tests [TODO]
- G1: STAGE 1 F0 + plan-022 winner + plan-025 C1 carry reproduce [TODO]
- G2.A: STAGE 2.A 9 cell (B1/B2/B3/B4/W1/T1/T2/S1/R1) [TODO]
- G2.B: STAGE 2.B conditional branch 1~2 cell [TODO]
- G3: STAGE 3 paradigm + best cell + 5가설 (a/b/c/d/e) verdict 박제 [TODO]
- G_final: STAGE 4 results + 3-file sync [TODO]

### Commit chain (next-up)

| # | type | spec section | status |
|---|---|---|---|
| c1 | docs | `plans/plan-028-per-anchor-isolation-weight-probe.md` v1 작성 + plan-review-master 자동 fix iter 1~4 BLOCKER 0 도달 (c1.1~c1.5 patch chain, §8 변경이력 참조) | [DONE — c1 ce55c3f + c1.1 723cdba + c1.2 a7afa55 + c1.3 83c65b6 + c1.4 1a09f5b + c1.5 본 commit] |
| c2 | chore | plan-025 worktree (`worktree-plan-025-spec`) 에서 `analysis/plan-025/{build_feat_1080.py, run_oof.py, __init__.py}` 3 file cherry-pick → 본 worktree `analysis/plan-025/` (post-cherry-pick read-only). plan-024 6 module + 1 data + __init__.py 는 main 에 머지된 상태가 아니므로 동일 worktree 에서 cherry-pick (= plan-025 c2 와 동일 source). | [TODO] |
| c3 | code | `analysis/plan-028/build_feat_subset.py` — `build_feat_1080` output 의 *index slice* 함수 4개: `slice_B1_anchor22(X)` / `slice_B2_combo192(X)` / `slice_B3_no_anchor1058(X)` / `slice_B4_full1080(X)` 그리고 `weight_flag(on=True/False)`. recomputation 금지 — slice only. | [TODO] |
| c4 | code | `analysis/plan-028/run_oof_subset.py` — plan-025 `run_oof_cell_1080` carry 위에 (a) input slice fn 주입 + (b) sample_weight expansion flag (on/off). CLI: `--cell {B1, B2, B3, B4, W1}` + 향후 branch cell 추가. | [TODO] |
| c5 | test | `tests/test_plan028_smoke.py` (≥ 8 pytest: import / 4 slice dim check (22, 192, 1058, 1080) / sample_weight on/off shape / LgbmSelectorOnly K=14 fit/predict smoke (subset dim) / F0 carry / soft label sum=1 / branch decision fn unit test) | [TODO] |
| G0 | gate | smoke + tests green | [TODO] |
| c6 | exp G1 | F0 baseline + plan-022 winner A6_bcc14_tau001 + plan-025 C1 (1080D full carry, sample-weight ON) reproduce. `analysis/plan-028/baseline_carry.json` 박제 (dataset_hash + 3 carry hash). | [TODO] |
| G1 | gate | F0 ∈ tight ✓ AND plan-022 winner ∈ tight ✓ AND plan-025 C1 ∈ tight ✓ | [TODO] |
| c7 | exp G2.A.B1 | Cell B1 (22D=③ only, weight ON, τ=0.001, LgbmSelectorOnly, hparam=p022 default) 5-fold OOF. `results_B1.json`. 예상 ~3min. | [TODO] |
| c8 | exp G2.A.B2 | Cell B2 (192D=①+③, weight ON, τ=0.001, LgbmSelectorOnly, hparam=p022) 5-fold OOF. `results_B2.json`. 예상 ~3min. | [TODO] |
| c9 | exp G2.A.B3 | Cell B3 (1058D=no ③, weight ON, τ=0.001, LgbmSelectorOnly, hparam=p022) 5-fold OOF. `results_B3.json`. 예상 ~5min. | [TODO] |
| c10 | exp G2.A.B4 | Cell B4 (1080D full = plan-025 C1 carry, weight ON, τ=0.001, LgbmSelectorOnly). **c6 G1 carry, 재실행 금지**. `results_B4.json`. | [TODO] |
| c11 | exp G2.A.W1 | Cell W1 (1080D full, **weight=1.0 균등**, τ=0.001, LgbmSelectorOnly, hparam=p022) 5-fold OOF. `results_W1.json`. 예상 ~5min. | [TODO] |
| c12 | exp G2.A.T1 | Cell T1 (1080D full, weight ON, **τ=0.01** soft label 재계산, LgbmSelectorOnly, hparam=p022) 5-fold OOF. `results_T1.json`. 예상 ~5min. | [TODO] |
| c13 | exp G2.A.T2 | Cell T2 (1080D full, weight ON, **τ=0.1** soft label 재계산, LgbmSelectorOnly, hparam=p022) 5-fold OOF. `results_T2.json`. 예상 ~5min. | [TODO] |
| c14 | exp G2.A.S1 | Cell S1 (1080D full, weight ON, τ=0.001, **base lightgbm.LGBMClassifier multiclass** subclass 우회, hparam=p022) 5-fold OOF. `results_S1.json`. 예상 ~5min. | [TODO] |
| c15 | exp G2.A.R1 | Cell R1 (block ④ = **raw seq flatten 95×7=665D** 대체, total 985D=①+②+③+R, weight ON, τ=0.001, LgbmSelectorOnly, hparam=p022) 5-fold OOF. `results_R1.json`. 예상 ~6min. | [TODO] |
| G2.A | gate | 9 cell metric finite ✓ + max_class_ratio 박제 ✓ + paired Δ vs B4 per cell 박제 | [TODO] |
| c16 | analysis | G2.A 9 cell 표 + 5가설 (a/b/c/d/e) verdict (§4.6) + branch 함수 §4.5 실행 결과 = α/β/γ/δ 중 1 activate → `paradigm_analysis_g2a.json` | [TODO] |
| c17 | exp G2.B.cell1 | activated branch 의 cell 1 (§4.4 정의) 5-fold OOF. `results_Bx_1.json` | [TODO] |
| c18 | exp G2.B.cell2 | activated branch 의 cell 2 (있을 시) 5-fold OOF. `results_Bx_2.json`. δ branch = 1 cell only → c18 skip. | [TODO] |
| G2.B | gate | branch cell metric finite ✓ + max_class_ratio 박제 ✓ + paired Δ vs plan-022 winner per cell 박제 | [TODO] |
| c19 | analysis | G2.A + G2.B 통합 best_cell selection (tiebreaker: hit_1cm > paired Δ_p022 > runtime) + paired Δ vs F0/plan-022/plan-025-C1 + 14-anchor oracle 회수율 + 5가설 (a/b/c/d/e) verdict 통합 → `paradigm_analysis.{json,md}` | [TODO] |
| G3 | gate | best_hit_1cm > 0.6531 → PASS / 0.6320 < best ≤ 0.6531 → partial_lift warn / best ≤ 0.6320 → regression warn | [TODO] |
| c20 | docs | 3-file frontmatter sync (status=all_complete, band=positive/partial/negative, best_cell) + `analysis/plan-028/results.md` (12 항목) + `plans/plan-028-*.results.md` pair + follow-up ≥ 2 건 | [TODO] |
| G_final | gate | 3-file sync ✓ + §0.5 c1~c20 모두 [DONE] ✓ + follow-up ≥ 2 건 ✓ | [TODO] |

### Plan-specific severe (WORKFLOW.md §12.3 default 위 추가분)

- `f0_reproduce_drift`: G1 F0 reproduce 가 plan-020/021/022/023/025 hard evidence 0.6320 / 0.8033 ±0.0005 밖. → halt.
- `plan022_reproduce_drift`: G1 plan-022 winner reproduce (A6_bcc14 + τ=0.001) 가 hard evidence 0.6531 / 0.8108 ±0.0005 밖. → halt.
- `plan025_C1_drift`: G1 plan-025 C1 carry (1080D full + sample-weight ON + hparam default) reproduce 가 plan-025 hard evidence 0.6320 / 0.8033 ±0.0005 밖. → halt. (= G2.A.B4 base reference)
- `lgbm_numerical`: G2.A/G2.B 어느 cell LGBM 출력 NaN/Inf. → halt.
- `soft_label_collapse`: cell 의 selector probs 가 단일 anchor 95% 이상 mass (`max_class_ratio > 0.95`). warn (severe 아님). 5+ cell drop = `soft_label_collapse_total` severe escalate.
- `slice_dim_mismatch`: c3 slice fn output dim ≠ {22, 192, 1058, 1080} per 함수. → halt.
- `branch_undefined`: §4.5 `decide_branch` 가 {α, β, γ, δ} 외 literal 또는 None 반환 시 (= invariant violation, 정상 path 에서는 trigger 안 됨 — 의사 코드의 무조건 fallback `return "δ"` 가 invariant 보장). c16 commit msg 에 activated branch 명시 필수 — decision-note 박제. self-check 용 severe.
- `weight_flag_silent`: W1 cell 의 sample_weight 가 실제로 균등 1.0 으로 inject 됐는지 (= 140000 row × `sample_weight=1.0` 균등, NOT 140000 row × per-row soft_label weight) self-check fail. row-expand reshape 자체는 ON/OFF 동일 (§4.3 정합). → halt.
- `tight_band_around_p022`: G3 best_hit_1cm ∈ [0.6526, 0.6536] (= plan-022 winner ±0.0005). paired Δ 부호로 결정 — Δ > 0 → positive, Δ ≤ 0 → partial. warn 박제.
- `partial_lift`: G3 best ∈ (0.6320, 0.6531]. F0 초과 but plan-022 미달. warn 박제 후 G_final (band=partial).
- `regression`: G3 best ≤ 0.6320. plan-025 mode collapse 잔존. warn 박제 후 G_final (band=negative).

### Plan-specific paths (WORKFLOW.md §12.5/§12.6)

- whitelist 추가:
  - `analysis/plan-028/**`
  - `tests/test_plan028_smoke.py`
  - `analysis/plan-025/{build_feat_1080.py, run_oof.py, __init__.py}` — **c2 cherry-pick 단계 유일한 plan-025 path 수정 허용** (file add only, post-c2 수정 금지)
  - `analysis/plan-024/{__init__.py, anchor_vocab.py, cand_builder.py, seq_builder.py, torsion_calc.py, quantile_carry.py, multiwindow_trim_build.py, multiwindow_trim.json}` — c2 cherry-pick 단계 유일한 plan-024 path 수정 허용 (plan-025 와 동일 패턴)
- blacklist:
  - `runs/baseline/**`
  - `analysis/plan-{001..027}/**` (단, c2 cherry-pick 으로 *추가* 된 plan-024/plan-025 path 는 read-only import 만 허용)
  - `plans/plan-{001..027}-*.md` (수정 금지)
- 참조 (read-only):
  - `analysis/plan-025/{build_feat_1080.py, run_oof.py}` (cherry-pick 후 read-only)
  - `analysis/plan-024/{cand_builder.py, seq_builder.py, torsion_calc.py, quantile_carry.py, multiwindow_trim_build.py, multiwindow_trim.json, anchor_vocab.py}` (동)
  - `analysis/plan-022/{selector_only_model.py, anchors.py, run_oof.py, baseline_carry.json}`
  - `analysis/plan-021/build_input.py`
  - `analysis/plan-020/{baseline_f0.py, baseline_oof.json}`
  - `src/{io.py, pb_0_6822/selector.py}`

### Decision-note 사용 예 (자율 결정 시 commit msg 박제)

- `decision-note: spec-default — c2 cherry-pick 은 worktree-plan-025-spec branch 의 latest commit (= G2.C2 시점) 에서 plan-025 3 file + plan-024 8 file 만. cherry-pick command: git checkout worktree-plan-025-spec -- analysis/plan-025/<file> analysis/plan-024/<file>`
- `decision-note: spec-default — block ablation 은 build_feat_1080 output 의 *index slice* 로 구현 (재계산 금지). slice index 는 §4.2 표 정의 따라 hard-coded.`
- `decision-note: spec-default — K=14 BCC anchor + τ_cls=0.001 + soft label + 5-fold + F0 = plan-022/025 carry fix. 변경 X.`
- `decision-note: spec-default — G2.A B4 (1080D full + sample-weight ON) 는 plan-025 C1 carry 와 *완전 동일* 입력/하이퍼파라미터 → c10 단계에서 G1 결과 그대로 박제, 재실행 금지 (runtime 절약).`
- `decision-note: spec-default — G2.A 9 cell hparam = plan-022 default (n_estimators=500, lr=0.05, num_leaves=63, random_state=20260522). 각 cell 의 single 변수 = 5 축 중 1 개: input slice (B1/B2/B3) OR sample_weight 값 (W1) OR τ_cls (T1/T2) OR model wrapping (S1) OR block ④ 산식 (R1). B4 baseline 은 plan-025 C1 carry, c10 instant.`
- `decision-note: spec-default — sample_weight 산식 (§4.3 정합): ON = row-expand X[N×14, D] + sample_weight=soft_label[sample, anchor_idx] + label=anchor_idx + multiclass num_class=14 → 140000 row. OFF (W1 only) = ON 과 reshape/label/objective 모두 동일하되 sample_weight=1.0 균등. LgbmSelectorOnly 의 \`weighted=False\` 옵션 추가 (= weight 만 1.0 균등 inject, row-expand 자체는 plan-025 carry 그대로).`
- `decision-note: spec-default — branch 함수 (§4.5) 가 복수 조건 만족 시 우선순위 α > β > γ > δ 로 1 branch only.`

---

## §1. 배경

### §1.1 plan-025 mode collapse 의 5 가설 (paradigm_analysis §4 carry (a/b/c/d) + 본 plan 가설 (e) 추가)

| 가설 | 정의 | 본 plan 검증 cell | likelihood |
|:--|:--|:--|:--|
| (a) τ_cls=0.001 sharp soft label train/test gap | τ 너무 sharp 라 train fold 의 soft label 이 test fold 에서 generalize 안됨 | **T1 (τ=0.01), T2 (τ=0.1)** — 본 plan G2.A 직접 검증 (soft label 재계산) | 중 |
| (b) sample-weight expansion (140k row × 14-class CE) 비효율 | row-expand 후 row 별 weight=soft_label 로 fit. LightGBM 의 weight 처리가 14-class objective 와 충돌 가능. | **W1 (sample-weight=1.0 균등)** | 중 |
| (c) LgbmSelectorRowExpanded subclass self-consistency 약화 | row-expand subclass 의 fit/predict path 가 14-row 일관성 깨짐 | **S1 (base lightgbm.LGBMClassifier multiclass 직접 사용, subclass 우회)** | 낮 |
| (d) 1058D broadcast / 22D per-anchor 50:1 dominance | broadcast feature 가 LGBM split gain 에서 per-anchor 22D 묻음 → row-discriminative 신호 못 잡음 | **B1 (22D only), B2 (192D = ①+③), B3 (1058D = no ③)** | **높 (most likely)** |
| (e) seq 압축 lossy (block ④ 8-stat) | block ④ = seq_builder 95×7 raw (665 value) → per-channel 8-stat 압축 (last/first/mean/std/slope/max/min/range × 95 = 760D). raw seq 의 row-discriminative / temporal fine-grained signal 이 8-stat 으로 평탄화 → LGBM 이 row 별 best anchor 못 가림. | **R1 (block ④ = raw seq flatten 95×7=665D 대체, total 985D)** — 본 plan G2.A 직접 검증. B1 vs B4 비교로 부분 cross-check (단 (d) 와 confound). | 중 (block ④ 760D 가 1080D 의 70% 차지) |

본 plan 의 분석 축 = **(a) + (b) + (c) + (d) + (e) 5가설 통합 직접 검증** — G2.A 9 cell (B1/B2/B3/B4=baseline/W1/T1/T2/S1/R1). 각 cell 은 B4 baseline 대비 single-variable 변경: input dim subset (B1/B2/B3), sample_weight 값 (W1), τ_cls (T1/T2), model wrapping (S1), block ④ 산식 (R1). 5가설 각각 §4.6 verdict 함수로 confirmed/rejected/inconclusive 판정.

### §1.2 plan-022 winner 가 본 plan 의 *경기 상대*

- plan-022 best A6_bcc14_tau001: hit_1cm = 0.6531, hit_1p5cm = 0.8108
- plan-022 winner 는 block ① 170D 만 사용 (broadcast feature 도 17D 정도 포함 — 즉 170D 안에서도 broadcast/per-anchor 비율이 plan-025 만큼 극단 아님)
- B2 (192D = block ① 170D + block ③ 22D) 가 가장 직접적인 plan-022 winner + ε 후보 cell
- B2 가 0.6531 + ε 이상이면 (d) 가설 확정 (= per-anchor 22D 가 보태질 때 lift), 0.6531 미달이면 broadcast feature 가 도리어 plan-022 winner 보다 noise 였다는 hint

### §1.3 14-anchor oracle ceiling 0.7928 의 의미

- plan-024 박제 (carry): test fold 각 sample 마다 14-anchor 중 *진짜* 최선의 anchor 를 oracle 이 골라줬을 때의 hit_1cm
- plan-022 winner 회수율 = 0.6531 / 0.7928 = 82.38%
- plan-025 C1 회수율 = 0.6320 / 0.7928 = 79.72% (= F0 동일 = mode collapse)
- 본 plan G3 PASS 시 회수율 ≥ 82.38% 진입. stretch (0.6700) = 84.5%. 100% = oracle 자체.

---

## §2. Scope (명시적)

### §2.1 In-scope

| 항목 | 값 |
|:--|:--|
| 입력 feature pipeline | plan-025 1080D carry (build_feat_1080) — 재계산 없음, slice only |
| 입력 dim 변수 (B1/B2/B3, 가설 d) | {22 (B1=③), 192 (B2=①+③), 1058 (B3=no③), 1080 (B4=full baseline)} — 3 cell + 1 baseline |
| sample-weight 변수 (W1, 가설 b) | {ON (soft_label-weighted default), OFF=1.0 균등} — OFF cell 1개 |
| τ_cls 변수 (T1/T2, 가설 a) | {0.001 (default baseline), 0.01 (T1), 0.1 (T2)} — soft label 재계산. 2 cell. |
| model wrapping 변수 (S1, 가설 c) | LgbmSelectorOnly subclass (default) vs base lightgbm.LGBMClassifier multiclass 직접 사용. 1 cell. |
| block ④ 산식 변수 (R1, 가설 e) | 8-stat 760D (default) vs raw seq flatten 95×7=665D. total dim 1080 → 985. 1 cell. |
| **Total G2.A** | **9 cell** (B1/B2/B3/B4=baseline/W1/T1/T2/S1/R1) — 각 cell single-variable vs B4 |
| Anchor | K=14 BCC (A6_bcc14, ANCHORS_A6) fix |
| Soft label 산식 | plan-022 `build_soft_label_with_tau` carry (τ_cls value 만 T1/T2 에서 변경) |
| Model | LgbmSelectorOnly subclass default + S1 cell 만 base `lightgbm.LGBMClassifier(objective='multiclass', num_class=14)` |
| LGBM hparam | plan-022 default (n_est=500, lr=0.05, num_leaves=63, rs=20260522) — G2.A 9 cell 공통 (T1/T2 는 τ_cls value 만 별도, S1 은 LGBM hparam 그대로 carry but model class = base LGBMClassifier, R1 은 input dim 만 다름). G2.B branch α/γ 만 LGBM hparam tweak. |
| Fold | plan-020/021/022 stable_fold_id 5-fold carry |
| F0 baseline | plan-020 carry, paired Δ anchor |
| G2.B branch | §4.5 정의 함수로 α/β/γ/δ 중 1 개 activate, 1~2 cell 추가 |
| Hit metric | hit_1cm (primary, R_HIT=0.01m), hit_1p5cm (secondary, R_HIT_LOOSE=0.015m) |
| 평가 | 5-fold OOF concat (plan-020 carry) |
| 합격 기준 | best_hit_1cm > 0.6531 AND paired Δ vs plan-022 winner > 0 |

### §2.2 Out-of-scope (절대 안 함)

| 항목 | 이유 |
|:--|:--|
| Anchor layout 변경 (K ≠ 14, BCC 외 codebook) | plan-022 winner carry — paradigm fix, 본 plan 변수 X |
| τ_cls 변경 (T1/T2 외 추가 sweep) | T1=0.01 / T2=0.1 외 다른 τ value 는 본 plan out-of-scope. baseline τ=0.001 (plan-022 winner carry). T1/T2 는 가설 (a) 검증 in-scope. |
| Fold 변경 | plan-020 stable_fold_id 5-fold carry |
| Soft label 산식 변경 | plan-022 build_soft_label_with_tau carry |
| F0 baseline 자체 변경 (ML화) | 본 plan 변수 X. 별도 followed_by 후보. |
| 새 dim 추가 (block ⑤ 등) | plan-025 1080D fix. subset 만 허용. |
| Cross-attention / GRU / corrector reg head | plan-024 검증 종료 / plan-021 dead lever |
| DACON submit | LB 측정 본 plan 변수 X (G3 metric = OOF only) |
| Ensemble (plan-022 + plan-028 등) | 별도 plan (ensemble = follow-up 후보) |
| Anchor radius ≠ 0.005m | hit_1cm 기준 fix |
| plan-026 (worktree-only) carry | spec isolation — 본 plan self-contained, plan-026 spec 도 별도 paradigm |

---

## §3. 사전 등록 (Pre-registration)

### §3.1 Fold split

- 5-fold stable_fold_id (`src/pb_0_6822/selector.py:stable_fold_id`, MD5 기반)
- plan-020/021/022/023/025 carry — 변경 X
- inner-fold (early_stopping 등) = 본 plan 사용 안 함 (G2.A 9 cell hparam fix, G2.B branch δ 만 selector arch 변경 시 inner-fold 가능)

### §3.2 합격 기준

| 조건 | 정의 | 결과 band |
|:--|:--|:--|
| A (PASS) | best_hit_1cm > 0.6531 AND paired Δ vs plan-022 winner > 0 | positive |
| B (tight_band) | best_hit_1cm ∈ [0.6526, 0.6536] | tight (paired Δ 부호로 final: Δ > 0 → positive, Δ ≤ 0 → partial) |
| C (partial) | 0.6320 < best_hit_1cm ≤ 0.6531 (B 범위 밖) | partial (warn `partial_lift`) |
| C' (partial fallback) | best_hit_1cm > 0.6531 AND paired Δ ≤ 0 (= A 조건 의 paired Δ 미충족) | partial (warn `partial_lift`, corner case) |
| D (negative) | best_hit_1cm ≤ 0.6320 | negative (warn `regression`) |

paired Δ 정의: per-sample 같은 fold split 위 hit_1cm 차이의 평균. 통계 검정 = paired bootstrap 5000× (§7 caveat + plan-022 carry symbol `analysis/plan-022/run_oof.py:bootstrap_paired_delta` 그대로). 95% CI 0 포함 시 `partial_band` warn 박제.

### §3.3 평가 점수 / median 집계

- per-fold hit_1cm/1p5cm 계산 후 5-fold OOF concat → 단일 hit_1cm
- 5-fold 분산 표시 (std, min, max) but median 아닌 concat 평균
- max_class_ratio = `probs_all.mean(axis=0).max()` (5-fold concat 위)
- top1_acc = `mean(argmax(probs) == hard_label)` (5-fold concat 위, hard label = K=14 BCC argmax of soft label)
- **"mode collapse" 정의 (본 plan 박제)**: 본 plan 의 "mode collapse" 는 *metric-level* (hit_1cm ≤ 0.6320 = F0 baseline 값) 을 일관 의미. plan-025 paradigm_analysis 의 *opposite-direction mode collapse* (= `max_class_ratio ≈ 1/14 = 0.0714`, softmax probs uniform → soft-mean(ANCHORS) ≈ Frenet origin → world prediction = F0 복사 → metric-level F0) 도 본 정의에 해당. softmax-mass 의 single-anchor 95% collapse (= §0.5 `soft_label_collapse > 0.95` warn) 는 *별도 반대 방향* warn 으로, metric-level mode collapse 와 동시 또는 독립 발생 가능. (b)/(d) verdict 함수 (§4.6) 의 기준은 metric-level (hit_1cm 값) 만 사용 — softmax-mass 는 보조 진단 metric.

---

## §4. STAGE 정의

### §4.1 STAGE 0 (G0) — 인프라

c1 (spec) → c2 (cherry-pick) → c3 (slice fn) → c4 (runner) → c5 (tests) → G0 gate.

산출물:
- `analysis/plan-028/__init__.py`
- `analysis/plan-028/build_feat_subset.py` (slice fn 4개 + weight_flag)
- `analysis/plan-028/run_oof_subset.py` (CLI: `--cell {B1,B2,B3,B4,W1,Bx_*}`)
- `tests/test_plan028_smoke.py` (≥ 8 pytest)
- cherry-pick (read-only): plan-025 3 file + plan-024 8 file

G0 종료 조건: pytest ≥ 8 pass + import 정상 (= §0.5 c5 박제 정합).

### §4.2 STAGE 1 (G1) — Baseline carry

c6: F0 + plan-022 winner + plan-025 C1 3 carry reproduce.

baseline_carry.json:
```json
{
  "f0": {"hit_1cm": 0.6320, "hit_1p5cm": 0.8033},
  "plan022_winner": {"hit_1cm": 0.6531, "hit_1p5cm": 0.8108, "cell": "A6_bcc14_tau001"},
  "plan025_C1": {"hit_1cm": 0.6320, "hit_1p5cm": 0.8033, "max_class_ratio": 0.0714, "input_dim": 1080},
  "dataset_hash": "<sha256>",
  "fold_seed": "stable_fold_id MD5"
}
```

G1 tight band:
- F0: hit_1cm ∈ [0.6315, 0.6325] AND hit_1p5cm ∈ [0.8028, 0.8038]
- plan-022 winner: hit_1cm ∈ [0.6526, 0.6536] AND hit_1p5cm ∈ [0.8103, 0.8113]
- plan-025 C1: hit_1cm ∈ [0.6315, 0.6325] AND hit_1p5cm ∈ [0.8028, 0.8038] (= F0 동일)

### §4.3 STAGE 2.A (G2.A) — 9 cell (5가설 single-variable 통합 검증)

| Cell | Input dim | weight | τ_cls | Model | block ④ 산식 | 변경 변수 (vs B4) | 가설 |
|:--|--:|:--|--:|:--|:--|:--|:--|
| **B1** | 22 (③ only) | ON | 0.001 | LgbmSelectorOnly | 8-stat (제외) | input dim | (d) most aggressive — broadcast 완전 제거 |
| **B2** | 192 (①+③) | ON | 0.001 | LgbmSelectorOnly | 8-stat (제외) | input dim | (d) lift most likely — p022 base + per-anchor |
| **B3** | 1058 (no ③) | ON | 0.001 | LgbmSelectorOnly | 8-stat | input dim | (d) — per-anchor 없으면 mode collapse 회복? |
| **B4** | 1080 (full) | ON | 0.001 | LgbmSelectorOnly | 8-stat | (**baseline**) | = plan-025 C1 carry. c6 G1 carry, 재실행 금지. |
| **W1** | 1080 (full) | **OFF=1.0** | 0.001 | LgbmSelectorOnly | 8-stat | sample_weight 값 | (b) — weight expansion 비효율 |
| **T1** | 1080 (full) | ON | **0.01** | LgbmSelectorOnly | 8-stat | τ_cls (10× softer) | (a) — τ sharp gap |
| **T2** | 1080 (full) | ON | **0.1** | LgbmSelectorOnly | 8-stat | τ_cls (100× softer) | (a) — τ sharp gap |
| **S1** | 1080 (full) | ON | 0.001 | **base LGBMClassifier** | 8-stat | model wrapping | (c) — subclass self-consistency |
| **R1** | 985 (①+②+③+R) | ON | 0.001 | LgbmSelectorOnly | **raw flatten 95×7=665D** | block ④ 산식 | (e) — seq 압축 lossy |

각 cell 은 B4 baseline 대비 column "변경 변수" 1개만 ≠ B4 (single-variable 원칙).

block ① / ② / ③ / ④ slice index (`build_feat_subset.py` 박제):
- block ① indices [0:170] (plan-022 build_input_lgbm_extra output)
- block ② indices [170:298] (cand_builder ctx broadcast 128D, 14 row 동일)
- block ③ indices [298:320] (cand_builder per-anchor 22D, 14 row 각 다름)
- block ④ indices [320:1080] (seq_builder 8-stat 760D, 14 row 동일)

slice fn output (B1~B4 동일 carry, R1 별도):
- `slice_B1_anchor22(X[N, 14, 1080]) → X[N, 14, 22]` = X[:, :, 298:320]
- `slice_B2_combo192(X) → X[:, :, np.r_[0:170, 298:320]]`
- `slice_B3_no_anchor1058(X) → X[:, :, np.r_[0:298, 320:1080]]`
- `slice_B4_full1080(X) → X[:, :, :]`
- `build_R1_seq_raw(X[N, 14, 1080], seq_raw[N, 95, 7])`: seq_raw 입력 = `analysis/plan-024/seq_builder.py:build_seq_feat` 의 *raw 95×7 output* (= per-channel 8-stat 압축 *이전* ndarray). c3 `analysis/plan-028/build_feat_subset.py` 에 helper `build_feat_1080_with_raw_seq(samples, ...) -> (X[N, 14, 1080], seq_raw[N, 95, 7])` 추가 — plan-025 `build_feat_1080` 호출 후, plan-024 `seq_builder.build_seq_feat` 를 wrapper 가 별도 호출 (plan-025 file 직접 수정 X, plan-024 carry symbol read-only). **deterministic 보장**: plan-024 `seq_builder.build_seq_feat` 는 plan-022/024/025 carry symbol contract 상 deterministic (= 같은 (samples, quantile_carry) 입력 → 같은 output, random seed / fold leakage 우려 없음). plan-025 `build_feat_1080` 내부의 seq_builder 호출 결과 (8-stat 압축 전) 와 wrapper 의 별도 호출 결과가 동일 sample list / quantile 위에서 일관됨. block ④ slice [320:1080] 제외 후 raw seq flatten (95×7=665D, sample-level broadcast 14 row) concat → 170+128+22+665 = 985D per row. output shape `[N, 14, 985]`.

cell 별 산식 변경 (B4 baseline 산식 carry, 아래 cell 만 변경):
- **T1/T2**: τ_cls value 변경 → `build_soft_label_with_tau(τ_cls=0.01)` (T1) / `(τ_cls=0.1)` (T2). soft label 재계산. label/weight/objective 산식 baseline 동일.
- **S1**: LgbmSelectorOnly subclass 우회 → `lightgbm.LGBMClassifier(objective='multiclass', num_class=14, **plan022_hparam)` 직접 fit. row-expand reshape + sample_weight + label 산식 baseline 동일 (subclass wrapper 만 제거).
- **R1**: block ④ 산식 변경 → `build_R1_seq_raw` 사용 (위 박제). input dim 1080 → 985, slice fn 별도. label/weight/τ_cls/model baseline 동일.

sample-weight ON / OFF 산식 (W1 cell 의 (b) 가설 single-variable isolation):
- **ON** (plan-022/025 default, B1/B2/B3/B4 공통): input `X[N, 14, D]` → `row-expand` reshape `X[N×14, D]` (각 sample 의 14 row 가 dim-D feature vector × 14 anchor 분 각각 1 row 로 펼침). 각 row 의 `sample_weight = soft_label[sample, anchor_idx]`, `label = anchor_idx` (= 0..13). LGBM objective = `multiclass` + `num_class=14`. row 수 = N × 14 = 140,000.
- **OFF** (W1 only): **reshape / label / objective 모두 ON 과 동일** (row-expand → `X[N×14, D]`, label = anchor_idx, objective = multiclass, num_class=14, row 수 = N×14 = 140,000). 차이는 **`sample_weight = 1.0` 균등** (ON 의 soft_label-weighted 대신). → (b) 가설 (LightGBM 의 per-row weight 처리가 14-class objective 와 충돌) 의 *single-variable isolation*: 단일 변수 = weight 값 (ON: soft_label, OFF: 1.0). expand 자체 / row 수 / label 산식 / objective 는 ON 과 동일.

W1 의 fit signature 차이는 `LgbmSelectorOnly.fit(..., weighted=False)` flag 추가 (plan-025 carry runner 에 옵션 inject) — `weighted=False` 시 위 `sample_weight = 1.0` 균등 inject (row-expand reshape 는 ON 과 동일 진행, plan-025 carry 그대로). predict signature 는 변경 X.

**inject 경로 명시** (c2 read-only 박제와 정합): c4 `analysis/plan-028/run_oof_subset.py` 가 plan-025 `run_oof_cell_1080` 호출 wrapper 로, cell 별 hyper-parameter / weight / τ_cls / model class / input slice 를 LgbmSelectorOnly fit 직전 kwargs 에 patch — W1 cell 만 `weighted=False` 키 추가 (LgbmSelectorOnly subclass 가 keyword 미지원 시 `sample_weight=np.ones(N*14)` 명시 inject 으로 fallback). 동일 wrapper 패턴이 T1/T2 의 τ_cls 변경 (= `build_soft_label_with_tau(τ=...)` 호출 직전 inject), S1 의 base `lightgbm.LGBMClassifier` 직접 사용 (= LgbmSelectorOnly subclass 우회), R1 의 build_R1_seq_raw 사용 (= input slice fn 교체) 도 wrapper 내부에서 처리. plan-025 carry file 자체는 cherry-pick 후 read-only 유지 (= c2 박제 §0.5 L123 정합).

c7~c11 commit 단계마다 cell 1개씩 OOF 측정 후 `results_<cell>.json` 박제 (per-fold hit/1p5cm + concat hit + max_class_ratio + top1_acc + runtime + paired Δ vs B4/plan-022).

### §4.4 STAGE 2.B (G2.B) — Conditional branch (1~2 cell)

§4.5 branch 함수로 α/β/γ/δ 중 1 개 activate. **branch 결정 자체는 c16 analysis commit** (§0.5 commit chain 정합), **branch cell 실행은 c17~c18 단계**.

| Branch | 활성 조건 | Cell 구성 (1~2 cell) | 비고 |
|:--|:--|:--|:--|
| **α** (input dim sweet spot) | B2 ≥ 0.6531 OR (B2 > plan-022 winner - 0.005 AND B2 > B4 + 0.005) | α1: B2 (192D) + LGBM hparam tweak (n_est=2000 + lr=0.02 + feature_fraction=0.7), α2: B2 (192D) + num_leaves=127 + min_data_in_leaf=10 | 2 cell |
| **β** (per-anchor 22D 단독 회복) | B1 > B3 + 0.005 AND B1 < 0.6531 (= 22D 가 broadcast 없을 때 살아남지만 plan-022 미달) | β1: B2 (192D) + selector arch = plan-022 LgbmSelectorOnly + 추가 22D feature normalization (z-score per fold), β2: B1 (22D) + LGBM hparam tweak (num_leaves=31, n_est=2000) | 2 cell |
| **γ** (sample-weight 가 진짜 원인) | W1 > B4 + 0.005 | γ1: W1 (1080D + weight OFF) + LGBM hparam tweak (lr=0.02 + n_est=2000), γ2: B2 (192D) + weight OFF | 2 cell |
| **δ** (default — 모든 cell ≤ B4 + 0.003) | 모든 G2.A cell이 B4 baseline 0.6320 + 0.003 = 0.6323 이하 (mode collapse 잔존) | δ1: B2 (192D) + selector arch 변경 — LGBM → 작은 MLP per-sample softmax (hidden=64, depth=2, lr=1e-3, epoch=50, CPU). only 1 cell. | 1 cell |

branch 우선순위: α > β > γ > δ. 복수 branch 조건 만족 시 우선순위 높은 것만 activate.

**G2.B cell 의 multi-variable 변경 허용 (G2.A single-variable 원칙 예외)**: G2.B 는 G2.A single-variable 분석 결과 위 lift optimization 단계 — α1/α2/β1/β2/γ1/γ2 cell 이 동시에 2~3 변수 변경 가능 (예: β1 = B2 192D dim + 22D z-score 정규화 + selector arch wrapper = 3 변수 동시). G2.B 의 목적 = lift cell 발견이고 single-variable causal 분석 아님. WORKFLOW.md §9.2 single-variable 원칙은 G2.A 만 적용, G2.B 는 explicit 예외 박제.

decision-note 박제 의무: c16 commit msg (= G2.A branch analysis) 에 activate 된 branch + 활성 조건 만족 cell 수치 명시.

### §4.5 Branch 결정 함수 (의사 코드)

```python
def decide_branch(B1, B2, B3, B4, W1) -> Literal["α", "β", "γ", "δ"]:
    """
    G2.A 의 (b)+(d) 5 cell (B1, B2, B3, B4, W1) hit_1cm 입력 → 1 branch activate.
    T1/T2/S1/R1 (가설 (a)/(c)/(e)) 결과는 본 함수 input 미포함 — §4.6 verdict 별도 산출.
    우선순위: α > β > γ > δ.
    """
    P022 = 0.6531  # plan-022 winner

    # α: input dim sweet spot
    if B2 >= P022 or (B2 > P022 - 0.005 and B2 > B4 + 0.005):
        return "α"

    # β: per-anchor 22D 단독 회복
    if B1 > B3 + 0.005 and B1 < P022:
        return "β"

    # γ: sample-weight 가 진짜 원인
    if W1 > B4 + 0.005:
        return "γ"

    # δ: default — α/β/γ 모두 미충족 → fallback.
    # §4.4 표 활성 조건 "모든 cell ≤ B4 + 0.003" 은 implicit (= α/β/γ 모두 false
    # 이면 자동으로 lift cell 부재 의미, mode collapse 잔존). explicit check
    # 별도 안 함. (a)/(c)/(e) 가설 cell (T1/T2/S1/R1) 결과는 branch decision 에
    # 미반영 — §4.6 verdict 함수에서만 산출, G3 best_cell selection 에서 통합 argmax.
    return "δ"
```

`tests/test_plan028_smoke.py` 에 4 case unit test 박제 — 각 case 의 input (B1, B2, B3, B4, W1) 과 expected branch:

| case | B1 | B2 | B3 | B4 | W1 | expected |
|:--|--:|--:|--:|--:|--:|:--|
| α activates | 0.55 | **0.66** | 0.60 | 0.6320 | 0.6320 | `"α"` |
| β activates | **0.65** | 0.55 | 0.60 | 0.6320 | 0.6320 | `"β"` |
| γ activates | 0.55 | 0.60 | 0.55 | 0.6320 | **0.66** | `"γ"` |
| δ default | 0.6320 | 0.6320 | 0.6320 | 0.6320 | 0.6320 | `"δ"` |

(α case: B2=0.66 ≥ P022=0.6531 ✓ → α; β case: B1=0.65 > B3=0.60+0.005 AND B1 < P022 ✓ → β (α 미충족); γ case: W1=0.66 > B4=0.6320+0.005 ✓ → γ (α/β 미충족); δ case: 모든 cell = B4 baseline → δ default)

**branch fn 의 5-cell input 정합 박제**: `decide_branch` signature 는 (b)+(d) cell (B1/B2/B3/B4/W1) 만 input — T1/T2/S1/R1 (가설 (a)/(c)/(e)) verdict 는 §4.6 verdict 함수로 별도 박제하며 G2.B branch decision 에 영향 X (의도적 분리: G2.B 의 목적 = plan-022 winner lift cell 발견이고, (a)/(c)/(e) 검증은 진단 성격으로 별도 path). T1/T2/S1/R1 중 어느 cell 이 plan-022 winner 보다 좋으면 G3 best_cell selection (§4.6, c19) 에서 G2.A 9 cell + G2.B 1~2 cell 통합 argmax 의 후보로 자동 포함.

**거동 예시**: 예) T1 (τ=0.01 cell) 의 hit_1cm = 0.66 > plan-022 0.6531 인 가상 case — G2.B 는 (b)+(d) branch 룰만 적용 (α 조건 = B2 기준, β = B1/B3, γ = W1) → 만약 B2/B1/W1 모두 baseline 근처면 δ default 활성. T1 의 lift 는 c19 best_cell argmax 단계에서 picked-up (= positive band 가능). 즉 G2.B branch decision 의 silent drop 은 (a)/(c)/(e) cell 의 G3 final 박제와 *독립*.

### §4.6 STAGE 3 (G3) — Paradigm + best_cell

c19: G2.A 9 cell + G2.B 1~2 cell 통합 best_cell selection (§0.5 commit chain 정합).

tiebreaker:
1. hit_1cm 최대
2. tie 시 paired Δ vs plan-022 winner 최대
3. tie 시 runtime 최소

박제:
```json
{
  "best_cell": "<cell_id>",
  "best_hit_1cm": <float>,
  "best_hit_1p5cm": <float>,
  "paired_delta": {
    "vs_f0": {"hit_1cm": <>, "hit_1p5cm": <>},
    "vs_p022": {"hit_1cm": <>, "hit_1p5cm": <>},
    "vs_p025_C1": {"hit_1cm": <>, "hit_1p5cm": <>}
  },
  "oracle_recovery": <best_hit_1cm / 0.7928>,
  "hypothesis_verdict": {
    "(a) tau_cls_sharp_gap": "<confirmed|rejected|inconclusive>",
    "(b) sample_weight": "<confirmed|rejected|inconclusive>",
    "(c) subclass_self_consistency": "<confirmed|rejected|inconclusive>",
    "(d) broadcast_dominance": "<confirmed|rejected|inconclusive>",
    "(e) seq_compression_lossy": "<confirmed|rejected|inconclusive>"
  },
  "band": "<positive|partial|negative|tight>"
}
```

가설 verdict 함수 (5가설 통합):
- **(a)** confirmed: max(T1, T2) > B4 + 0.005 (τ softer 가 mode collapse 해소)
- **(a)** rejected: max(T1, T2) < B4 - 0.003 (τ softer 가 도리어 악화)
- **(a)** inconclusive: 위 외
- **(b)** confirmed: W1 > B4 + 0.005 (sample_weight 균등화가 lift)
- **(b)** rejected: W1 < B4 - 0.003
- **(b)** inconclusive: 위 외
- **(c)** confirmed: S1 > B4 + 0.005 (subclass 우회가 lift)
- **(c)** rejected: S1 < B4 - 0.003
- **(c)** inconclusive: 위 외
- **(d)** confirmed: B1 > B3 + 0.005 OR B2 > B4 + 0.005 (per-anchor 가 broadcast 보다 lift 줌)
- **(d)** rejected: B2 < B4 - 0.003 AND B1 < B3 - 0.003 (per-anchor 가 도리어 noise)
- **(d)** inconclusive: 위 외
- **(e)** confirmed: R1 > B4 + 0.005 (raw seq 가 8-stat 보다 lift)
- **(e)** rejected: R1 < B4 - 0.003
- **(e)** inconclusive: 위 외

### §4.7 STAGE 4 (G_final) — Results

c20:
- frontmatter sync: spec + results pair + analysis/plan-028/results.md → status=all_complete, band, best_cell, best_hit_1cm, best_hit_1p5cm, best_delta_1cm (vs F0), best_delta_1p5cm (vs F0), g{1,2,3,_final}_completed=true, exp_ids_completed/skipped
- `plans/plan-028-*.results.md` (plan-025 form 11 항목)
- `analysis/plan-028/results.md` (plan-025 form, G2.A 9 cell 표 (B1/B2/B3/B4/W1/T1/T2/S1/R1) + G2.B branch cell 표 + best_cell 박제 + paired Δ + oracle 회수율 + 5가설 (a/b/c/d/e) verdict + block 분해 + Runtime + max_class_ratio/top1_acc + follow-up + cross-refs)
- follow-up ≥ 2건 박제 (예: 가설 (a) τ_cls sweep / ensemble (plan-022 + plan-028 best) / F0 ML / oracle gap 추가 분석)

---

## §5. 작업량 총 회계

| STAGE | Commit 수 | Cell 수 | 예상 runtime (CPU) |
|:--|--:|--:|--:|
| G0 (c1~c5) | 5 | 0 | <10min (setup) |
| G1 (c6) | 1 | 3 carry reproduce | ~5min |
| G2.A (c7~c15) | 9 (B4 재실행 skip → c10 instant) | 8 신규 + B4 carry | ~37min total (B1 ~3 + B2 ~3 + B3 ~5 + B4 instant + W1 ~5 + T1 ~5 + T2 ~5 + S1 ~5 + R1 ~6) |
| G2.A analysis (c16) | 1 | 0 (5가설 verdict + branch 결정) | <1min |
| G2.B (c17~c18) | 1~2 (branch δ = 1 cell, α/β/γ = 2 cell) | 1~2 | ~5~15min |
| G3 (c19) | 1 | 0 (best_cell selection) | <1min |
| G_final (c20) | 1 | 0 (results) | <5min |
| **Total** | **19~20** | **9~11 cell + 3 carry reproduce** | **~50~70min CPU** |

cell 수 / commit 수 / 작업량 plan-025 (2 cell, ~15min) 대비 약 4-5× 증가 — 5가설 통합 grid 가 9 cell 이라 자연스러움. spec 의 G2.A 9 cell 단일 변수 원칙 (WORKFLOW.md §9.2) 준수: 각 cell 은 baseline (= B4) 대비 한 변수만 변경 (input dim / sample_weight / τ_cls / model wrapping / block ④ 산식 5 축 중 1 개).

---

## §6. results.md 필수 항목 (12 항목, plan-025 form 일치)

1. plan_id / version / date / status / band / best_cell
2. G-gate 표 (G0/G1/G2.A/G2.B/G3/G_final 별 status + commit hash + 결과)
3. G2.A 9 cell 결과 표 (B1/B2/B3/B4/W1/T1/T2/S1/R1 — 각 cell / hit_1cm / hit_1p5cm / Δ_1cm vs F0 / Δ_1cm vs p022 / Δ_1cm vs B4 / max_class_ratio / top1_acc / runtime)
4. G2.B branch + cell 결과 표 (활성 branch / cell / 위 동일 metric)
5. Best cell 박제 + paired Δ 3종 (F0, p022, p025-C1)
6. 가설 (a) + (b) + (c) + (d) + (e) 5가설 verdict (각 confirmed / rejected / inconclusive + 수치 근거 — §4.6 verdict 함수 산출)
7. 14-anchor oracle 회수율 (best / 0.7928)
8. 1080D input block 분해 표 (plan-025 form 참고, 본 plan 의 G2.A 결과로 (d) 가설 update)
9. Runtime (G0~G_final per STAGE)
10. max_class_ratio + top1_acc + best_iteration (cell 별)
11. Follow-up plan 후보 ≥ 2건
12. Cross-refs (spec, results pair, baseline_carry, results_<cell>.json 5+ 개, paradigm_analysis, 참조 plan)

---

## §7. 통계 함정 & caveats

- **5-fold OOF concat 의 분산**: plan-022/025 carry — per-fold std 박제. paired Δ 검정은 같은 fold split 위 per-sample 차이의 평균 → fold 분산 영향 적음.
- **W1 cell 의 weight 균등화 isolation**: W1 의 단일 변수 = `sample_weight` (ON: soft_label-weighted, OFF: 1.0 균등). row-expand reshape / label = anchor_idx / objective = multiclass / num_class = 14 모두 ON 동일 (§4.3 sample-weight 산식 정합). (b) 가설 (LightGBM 의 per-row weight 처리가 14-class objective 와 충돌 가능) 의 isolation 명확 — soft_label 의 sharpness (τ=0.001) 가 ON 에서 effective weight 분포 sharpness 와 OFF uniform 의 effective weight 분포 평탄성 차이 = (b) 가설 검정 신호.
- **B1 (22D only) 의 LGBM hparam carry**: 22D 가 너무 작아 plan-022 default `num_leaves=63` + `min_data_in_leaf=20` 이 일부 leaf 에서 invalid 가능. **본 plan 의 결정**: hparam 자동 축소 없음 — plan-022 default 그대로 carry (§2.1 LGBM hparam 행 정합). LGBM 자체 fallback (leaf 부족 시 no split, 트리 early termination) 위임. 결과적으로 B1 의 effective num_leaves < 63 일 수 있으나 본 cell 의 (d) 가설 검증 (= 22D per-anchor signal 존재 여부) 목적상 acceptable.
- **G2.A.B4 의 재현성**: c10 단계에서 plan-025 C1 carry 의 hash 정확성 — plan-025 worktree 의 commit hash 박제 필수 (`decision-note: spec-default — B4 = plan-025 C1 carry hash <commit>`).
- **G2.B branch δ (MLP) 의 CPU 수렴**: plan-024 cross-attention 의 CPU under-converged 교훈 — 본 plan δ branch 의 MLP 는 작은 capacity (hidden=64, depth=2, epoch=50) 로 의도적으로 under-converged 위험 회피. 단 본 cell 이 plan-022 winner 못 이기면 paradigm-level conclusion = "LGBM 자체가 ceiling".
- **paired Δ 의 fold variance**: paired bootstrap 5000× 측정 박제 (plan-022/025 carry 산식, `analysis/plan-022/run_oof.py:bootstrap_paired_delta`). 95% CI 박제. CI 가 0 포함 시 partial band warn.
- **dataset_hash 일치**: G1 baseline_carry.json 의 dataset_hash 가 plan-022/025 baseline_carry.json 의 hash 와 일치 — 데이터 drift 차단.

---

## §8. 변경 이력

- v1 (2026-05-22): 초안 + iter 1~4 plan-review 자동 fix log (frontmatter `version: 1` 유지, 본 §8 만 내부 evolve 추적).
  - c1 (ce55c3f): 초안 작성 — plan-025 mode collapse paradigm_analysis §4 의 가설 (b) + (d) 검증 + plan-022 winner lift 목표 spec.
  - c1.1 (723cdba): plan-review iter 1 자동 fix 9건 (BLOCKER 2 + AMB 5 + FP 1 skip + 가설 e 추가).
  - c1.2 (a7afa55): 사용자 instruction "a, c, e 도 이 plan 안에서 진행" 으로 scope 확장 — 5가설 (a)(b)(c)(d)(e) 통합 본 plan 직접 검증. G2.A 5 cell → 9 cell (T1/T2 τ sweep, S1 base LGBM, R1 seq raw 추가), §2 In-scope 에 τ_cls / model wrapping / block ④ 산식 변수 추가, commit chain c1~c16 → c1~c20, runtime 30-50min → 50-70min CPU, §4.6 verdict 함수 5가설 통합.
  - c1.3 (83c65b6): plan-review iter 2 자동 fix — BLOCKER 4 (branch fn signature, weighted flag inject, 표 title, results.md 항목) + "5 cell" 잔재 3곳 + AMB 4.
  - c1.4 (1a09f5b): plan-review iter 3 자동 fix — commit pointer propagation (c12/c13~14/c15/c16 → c16/c17~18/c19/c20) + JSON template 5가설 추가.
  - c1.5 (f11d359): plan-review iter 4 자동 fix — AMB 8 (results.md 항목 11→12, pytest 12+→≥8, band corner case, band enum, wrapper file path, branch_undefined 재정의, code_reuse count, c1 status sync).
  - c1.6 (본 commit): plan-review iter 5 (max) 자동 fix — BLOCKER 0 유지 + AMB 2 fix (§4.5 branch fn silent drop 거동 예시, §7 B1 22D num_leaves carry 명시) + MINOR 2 fix (§4.3 R1 wrapper deterministic 보장, 본 §8 c1.6 박제). plan-review-master 종료 (BLOCKER 0 + AMB 0 도달, max iter 5/5).

---

## §9. 참조

- `plans/plan-025-candidate-concat-input-max.md` v1 (worktree-plan-025-spec) — mode collapse paradigm + 4 가설 source
- `plans/plan-025-candidate-concat-input-max.results.md` (worktree-plan-025-spec) — 0.6320 mode collapse hard evidence + paradigm_analysis.json
- `plans/plan-022-corrector-free-anchor-layout-sweep.md` + `.results.md` — plan-022 winner A6_bcc14_tau001 paradigm
- `plans/plan-024-…` (worktree-plan-024-combo) — 14-anchor oracle 0.7928 ceiling + feature engineering 6 module
- `plans/plan-020-polyfit-baseline.md` + `.results.md` — F0 baseline + 5-fold split + paired Δ 산식
- `WORKFLOW.md §1~§12` — plan/results/registry/Autonomous Execution Protocol 규약
- `CLAUDE.md` — Autonomous Execution Policy
- `memory/project_next_plan_direction.md` — 2026-05-22 박제: plan-025 ablation 시급 (= 본 plan 가 회수)
- `analysis/plan-024/oracle_ceiling.json` (carry) — 14-anchor oracle 0.7928

---

## §10. plan-028 self-contained 확인 (Spec 자기-완결 invariant, WORKFLOW.md §9.3)

본 plan 은 외부 컨텍스트 (채팅 로그, 메모리) 없이 단독 재구성 가능:

- §1.1 ~ §1.3 = 배경 (plan-025 mode collapse + plan-022 winner + oracle ceiling 수치 박제)
- §2.1 + §2.2 = Scope
- §3.1 ~ §3.3 = Pre-reg (fold, 합격 기준, 평가)
- §4.1 ~ §4.7 = STAGE 정의 (commit chain, cell config, branch 함수)
- §5 = 작업량
- §6 = results.md 필수 항목 (plan-025 form 일치)
- §7 = caveats
- §8 = 변경 이력
- §9 = 참조 (모든 plan / module / data 의 path)

§0.5 = autonomous loop 가 매 turn 읽을 self-updating log + commit chain 16-step + plan-specific severe 9 + paths whitelist/blacklist + decision-note 예시.

frontmatter `code_reuse` = 명시적 carry 모듈 **15 개** 박제 (plan-025: build_feat_1080.py + run_oof.py = 2, plan-024: cand_builder + seq_builder + torsion_calc + quantile_carry + multiwindow_trim_build + anchor_vocab = 6, plan-022: selector_only_model + anchors + run_oof = 3, plan-021: build_input = 1, plan-020: baseline_f0 = 1, src: io + pb_0_6822/selector = 2 → total 15). symbols 단위 박제는 module 안 함수 / class 별 별도 (= read-only import 만).

본 plan G_final 도달 시 plans/plan-028-*.results.md + analysis/plan-028/results.md 2 file 생성, 3-file frontmatter sync 완료.
