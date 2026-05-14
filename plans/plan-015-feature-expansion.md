---
plan_id: 015
version: 2 (spec patch — v1 draft → spec complete. §3 v1 의 "후속 사항" 4개 (순차 ablation / baseline 0.6425 / 합격 기준 Δ+band / exp_id naming) 모두 박제. §0.5 G-gates G1~G_final + commit chain c2~c8 추가. §4~§9 STAGE 본문 추가 (G0 preflight / G1~G4 sub-exp 4 / G5 best stack + submission / G_final synthesis + plan-016 후보). 사용자 결정 (Q1) 순차 ablation + (Q2) Δ ≥ +0.005 + band classification + (Q3) LB 제출 plan-015 결과 후 1회. v1 → v2.)
date: 2026-05-14 (Asia/Seoul)
status: spec
based_on:
  - 014 (band=negative, best_stack 0.6425, oracle ceiling 0.8248, 회수율 5.4%)
followed_by: []
scope: corrector input feature 확장 — 현 9D kinematic 의 표현력 부족이 plan-014 G3 5축 negative 의 root cause 신호. 4 feature (A F0 residual / B binormal split / C multi-scale stride / D pairwise) 순차 ablation 으로 attribution + best stack 결정. plan-014 best_stack (E0c K-Means K=9 + boundary_weight_on, F0 frozen plan-006) 위 input feature 만 swap (corrector arch / loss / lever 모두 plan-014 carry).
exp_ids:
  - H042_g0_preflight
  - H043_g1_e1_feature_A
  - H044_g2_e2_feature_AB
  - H045_g3_e3_feature_ABC
  - H046_g4_e4_feature_ABCD
  - H047_g5_best_stack_5fold
  - H048_g_final_synthesis
lb_score: null
---

# plan-015 v2 — Feature Expansion (순차 ablation A→B→C→D)

## §0. 한 줄 목적

> **plan-014 corrector 의 input 표현력 부족 (oracle 0.82 vs measured 0.64, 회수율 5.4%) 을 직접 닫기 위해 9D → +4~28D feature 확장. 4 feature (A F0 residual / B binormal split / C multi-scale stride / D pairwise) 순차 ablation: A → A+B → A+B+C → A+B+C+D 단계별 Δ 측정 후 best 채택.**
>
> baseline (anchor) = plan-014 G5 best_stack (E0c K-Means K=9 + boundary_weight_on, F0 frozen plan-006). corrector arch / loss / lever 모두 plan-014 carry, input feature 만 swap.

---

## §0.5 Quick Reference

### 본 plan task essence

- **plan-014 measured ceiling = F0 raw + 0.0105 (회수율 5.4%)** → corrector 의 features 가 F0 error 방향 predict 부족.
- **oracle ceiling 0.8248 (E0b Frenet-ortho)** = 가능한 상한. plan-015 = features 만 강화하여 회수율 ↑ 시도.
- **순차 ablation**: A → A+B → A+B+C → A+B+C+D. 각 step 별 ΔOOF measured + attribution.

### plan-014 carry (고정)

- F0 = plan-006 frozen (d1=1.98 / par=1.20 / perp=−0.20 constants).
- Corrector arch = BiGRU h=128 (encoder) + cls head (Linear → K) + reg head (Linear → K*3, tanh × 0.005).
- baseline anchor codebook: **E0c K-Means K=9** (plan-014 G2 winner + Phase 2 best lever).
- baseline lever: **boundary_weight_on** (plan-014 Phase 3 best, E6b).
- 5-fold OOF scheme: SHA256 stable_hash, salt='plan-014-v1' (cross-plan reproducibility).
- monitor=val_hit (ascending), patience=5.

### Feature 정의 (v1 carry)

- **A** F0 prior residual 직접 input — per-step `(obs[t] − F0_pred[t])` 3D concat. +3D (9D → 12D).
- **B** Frenet binormal axis 분리 — `perp_norm/speed` (1D) → `normal_norm/speed` + `binormal_norm/speed` (2D). +1D (12D → 13D when A applied).
- **C** Multi-scale stride — base feature 를 τ ∈ {1, 2, 3} stride 로 3 stream concat. +(현재 dim × 2) (13D → 39D when A+B applied).
- **D** Pairwise cross-step interaction — step t vs t-2 / t-4 의 cosine similarity + Δspeed + Δangle. +6D (or stream 형태).

### G-gates (정량 spec @ §3.3)

- **G0** preflight: baseline (plan-014 best_stack) 5-fold reproduce ± 0.005 + feature dim sanity [TODO]
- **G1** E1 (A): A feature added, ΔOOF vs G0 anchor [TODO]
- **G2** E2 (A+B): + B feature, ΔOOF vs G1 [TODO]
- **G3** E3 (A+B+C): + C feature, ΔOOF vs G2 [TODO]
- **G4** E4 (A+B+C+D): + D feature, ΔOOF vs G3 [TODO]
- **G5** best stack 5-fold + submission: 4 step 중 *cumulative ΔOOF 가 가장 큰 sub-exp* 채택. submission 박제 [TODO]
- **G_final** synthesis: results.md + frontmatter sync + plan-016 후보 (LB carry-over 포함) [TODO]

### 합격 기준 (Q2 결정 — Δ + band)

**per-step Δ threshold** (additive lever 검증):
- Δ ≥ +0.005 → step `positive` (해당 feature 채택, 다음 step 의 cumulative baseline)
- 0 ≤ Δ < +0.005 → step `marginal` (채택은 하되 warn flag 박제)
- Δ < 0 → step `negative` (해당 feature drop, 이전 step 의 best 유지하고 다음 feature 추가는 skip 후 G_final 직행)

**band classification** (G5 cumulative best 의 5-fold OOF 기준):
- ≥ 0.66 → **positive** (plan-015 = polish + LB)
- 0.65 ≤ OOF < 0.66 → **partial** (plan-016 = ensemble / hybrid)
- < 0.65 → **negative** (plan-014 band 와 동일 — feature 확장으로도 ceiling break 실패, deep path-pivot)

### Target (judgement criteria)

- baseline = **plan-014 G5 best_stack OOF = 0.6425** (5-fold concat).
- **plan-015 G5 best stack OOF ≥ baseline + 0.005 = 0.6475** (= G5 pass).
- band classification 의 negative band (< 0.65) = ceiling break 실패 신호 → plan-016 deep path-pivot.

### Commit chain

| # | type | spec section | status |
|---|---|---|---|
| c1 | docs | v1 draft — feature spec (A/B/C/D) 박제 | [DONE] de3131b |
| **c2** | docs | **v2 spec patch — §3 expand: 순차 ablation + Δ+band 합격기준 + exp_id naming + STAGE §4~§9 추가** | [DONE] f195da4 |
| c3 | code+exp | STAGE 0 (G0) — preflight: plan-014 baseline 5-fold reproduce + feature dim sanity | [TODO] |
| c4 | code+exp | STAGE 1 (G1, E1) — feature A only (F0 residual direct), 5-fold OOF | [TODO] |
| c5 | exp | STAGE 2 (G2, E2) — A+B (F0 residual + binormal split), 5-fold OOF | [TODO] |
| c6 | exp | STAGE 3 (G3, E3) — A+B+C (+ multi-scale stride), 5-fold OOF | [TODO] |
| c7 | exp | STAGE 4 (G4, E4) — A+B+C+D (+ pairwise), 5-fold OOF | [TODO] |
| c8 | code+exp | STAGE 5 (G5) — best cumulative + 5-fold concat + submission (Δ + band 판정) | [TODO] |
| c9 | docs+sync | STAGE 6 (G_final) — results.md + frontmatter sync + plan-016 후보 + LB carry-over (plan-014 + plan-015 best 둘 다 dacon-submit 1회) | [TODO] |

---

## §1. Feature 확장 후보 (1, 2 순위, v1 carry)

(v1 본문 carry — A/B/C/D 4 feature 정의)

### 1순위 — 즉시 시도

#### A. F0 prior residual 직접 input ★

- **정의**: 11-step F0 prediction 좌표를 매 step encoder input 에 concat. per-step `(obs[t] − F0_pred[t])` 3D 추가.
- **dim**: +3D (9D → 12D)
- **구현**: `make_seq_features` 에서 step `s` 마다 `F0_pred[s] = F0_function(X[:, :s+1])` 산출 후 `obs[s] − F0_pred[s]` 3D 를 8d turn features + direction(1D) 와 concat → 12D.
- **단**: F0 산식은 horizon=2 미래 위치 예측. step `s` 의 "F0_pred[s]" 는 step `s` 시점 prior 가 아닌 *현재 위치까지의 F0 적용* 으로 의미 명확화 필요. **spec 결정**: step `s` 에서 `F0_pred[s] = X[:, s] + 1.98·(X[:, s] − X[:, s−1]) + ...` (= s-시점 horizon=2 prior). residual = `X[:, s+2] − F0_pred[s]` 가 idealistic 하지만 s+2 가 안 보이는 step (s=10) 도 있음. **간단 spec**: per-step `(F0_pred[s] − X[:, s])` 자체를 3D channel (즉 "현재 F0 가 어디로 prediction 하는지" 의 vector) 로 input. observed vs F0 의 ongoing divergence 신호.

#### B. Frenet binormal axis 분리

- **정의**: 현 `perp_norm/speed` (normal + binormal 합 1D) 을 `normal_norm/speed` + `binormal_norm/speed` (2D) 로 분리.
- **dim**: +1D (12D → 13D when A applied)
- **구현**: `_turn_features_per_step` 에서 `acc_perp_vec` 를 step-local Frenet basis `(t̂_s, n̂_s, b̂_s)` 로 추가 분해. n̂_s 는 acc_perp 자체 방향, b̂_s = t̂_s × n̂_s. `acc_normal = acc_perp · n̂_s` (= ‖acc_perp‖, 정의상), `acc_binormal = acc · b̂_s` (= acc 의 b̂ 성분 magnitude). 둘 다 abs value 로 normalize.

### 2순위 — 표현력 보강

#### C. Multi-scale stride features

- **정의**: 9D base feature 를 stride τ ∈ {1, 2, 3} 으로 3 stream 계산 → concat per-step.
- **dim**: per-step 13D (A+B 적용 후) × 3 stream = 39D
- **구현**: `make_seq_features` 의 step indices 산출 방식 3 set:
  - τ=1: 기존 `range(max(3, end_idx-5), end_idx+1)` (6 step, step gap 1)
  - τ=2: `range(max(3, end_idx-10), end_idx+1, 2)` (≤ 6 step, step gap 2). end_idx=10 에서 [0, 2, 4, 6, 8, 10] 6 step 가능
  - τ=3: `range(max(3, end_idx-15), end_idx+1, 3)` ([1, 4, 7, 10] 4 step → pad first 2 회) — pad rule §2.1.A 응용
  - 각 τ 마다 per-step feature 산출 후 *동일 시점 step* 끼리 concat (단순 concat, 시간 alignment 는 step index '직접 비교' 아닌 'BiGRU 가 알아서')
- **단순화 결정**: τ=1, τ=2 만 (τ=3 step 수 부족) → 2 stream concat = 26D (A+B+C base 13D × 2). **본 spec 의 C** = 2-stride 2 stream.

#### D. Pairwise cross-step interaction

- **정의**: per-step feature 에 cross-step pairwise 추가. step t 와 t-2 / t-4 의 cosine similarity + Δspeed + Δangle.
- **dim**: +6D (3 pair × 2 stat) per-step = 26D + 6D = 32D when A+B+C applied
- **구현**: per-step `s` 에서 `v[s], v[s-2], v[s-4]` 의 3 velocity vector. pair (s, s-2), (s, s-4), (s-2, s-4) 3 pair × cosine + Δspeed = 6D 추가.
- **edge case**: s=3, 4 일 때 s-4 미정의 → 첫 valid step (s=5) 의 값 forward fill.

---

## §2. Scope (명시적)

### §2.1 In-scope (= 4 feature 순차 ablation)

| 항목 | 값 |
|---|---|
| Feature 확장 | A / B / C / D (v1 박제) |
| Ablation strategy | **순차** (A → A+B → A+B+C → A+B+C+D) |
| Baseline (anchor) | plan-014 G5 best_stack (E0c K-Means K=9 + boundary_weight_on) — corrector arch / loss / lever 모두 carry |
| 변경 변수 | input feature 만 (per stage 1 feature add) |
| Validation | 5-fold OOF (plan-014 stable_hash carry, salt='plan-014-v1') |
| 합격 기준 | per-step Δ ≥ +0.005 + final band classification |

### §2.2 Out-of-scope

| 항목 | 이유 |
|---|---|
| Corrector arch 변경 | plan-014 carry (input 만 swap, controlled comparison) |
| Lever ablation (E1~E8) | plan-014 G3/G4 결과 carry (positive_axes=['E6'] only) |
| F0 산식 변경 | plan-006 carry (frozen) |
| 3순위 feature (snap, curvature rate 등) | v1 §2.2 박제 — 후순위 |
| Negative evidence 후보 (FFT, Neural ODE) | notes/new-ideas.md A.2/B.3 negative |
| Ensemble with plan-013/plan-014 | plan-016 후보 (band partial 진입 시) |

---

## §3. 사전 등록 (Pre-registration) — v2 신규

### §3.1 Baseline reference (plan-014 carry)

| metric | value | source |
|---|---|---|
| F0 raw hit@1cm | 0.6320 | plan-014 G0 (H036_g0_preflight) |
| plan-014 G5 anchor 5-fold OOF | 0.6359 | plan-014 G5 (H041) |
| **plan-014 G5 best_stack 5-fold OOF** ★ | **0.6425** | plan-014 G5 (H041), = plan-015 baseline |
| oracle ceiling (E0b Frenet-ortho) | 0.8248 | plan-014 G0 |
| corrector 회수율 | 5.4% | (best − F0) / (oracle − F0) = 0.0105/0.1928 |

### §3.2 Sub-exp matrix (순차 ablation)

| stage | sub-exp | feature config | dim | base |
|---|---|---|---|---|
| G1 | **E1** (A) | F0 residual direct | 12D | plan-014 best_stack |
| G2 | **E2** (A+B) | + binormal split | 13D | G1 (if Δ ≥ +0.005) or G0 (if G1 negative) |
| G3 | **E3** (A+B+C) | + multi-scale stride (τ=1,2 stream) | 26D | G2 (if Δ ≥ +0.005) |
| G4 | **E4** (A+B+C+D) | + pairwise cross-step | 32D | G3 (if Δ ≥ +0.005) |
| G5 | **best** | cumulative best (max ΔOOF over G0/G1/G2/G3/G4) | varies | G_final 의 submission base |

**Drop rule** (v2 patch 결정): 만약 G_n 의 ΔOOF < 0 (negative), 해당 feature drop + 그 feature 까지 cumulative 해서 다음 stage 추가 진행 *skip* → G_final 직행 (= 이전 best stage 가 best_stack 으로 채택).

(예: G2 (A+B) 가 G1 (A) 대비 -0.003 이면 → B drop + C/D 시도 skip → best = G1 (A only).)

### §3.3 G-gate quantitative criteria

#### G0 — preflight

- artifact: `analysis/plan-015/preflight.json`
- **(a) plan-014 baseline 5-fold reproduce**: 동일 config (E0c K-Means K=9 + boundary_weight_on, F0 frozen) 으로 5-fold OOF 재산출 → 0.6425 ± 0.005 일치 확인.
- **(b) feature dim sanity**: 4 feature (A/B/C/D) 각각 단독 적용 시 shape verify (12D / 10D / 27D / 15D 단일 적용 dim 사양).
- fail trigger: (a)/(b) 중 1+ 누락 → `preflight_artifact_missing` severe (plan-014 baseline 재현 불가 = 측정 base 부재).

#### G1~G4 — sub-exp 순차 ablation

각 G_n (n=1..4) 동일 schema:
- artifact: `analysis/plan-015/gN_eN.json` + `runs/baseline/plan015_eN/`
- spec: E_n config (cumulative feature) × 5-fold OOF.
- criterion: ΔOOF vs G_(n-1) anchor ≥ +0.005 → `positive`, 채택 후 다음 stage.
- fail trigger: ΔOOF < 0 → `eN_negative` warn, feature drop + 후속 G_(n+1)~G_4 skip → G_final 직행.
- marginal (0 ≤ Δ < +0.005): 채택 + warn flag 박제 후 continue.

#### G5 — best stack 5-fold + submission

- artifact: `analysis/plan-015/g5_phase4.json` + `runs/baseline/plan015_g5/submission.csv`
- spec: G0~G4 중 max ΔOOF cumulative best config 으로 5-fold concat OOF 박제 (이미 sub-exp 에서 산출됨 — 재학습 불필요) + test 5-fold ensemble submission.
- criterion: **best_stack 5-fold OOF ≥ 0.6475** (= baseline 0.6425 + 0.005) → G5_passed.
- band 분류:
  - best_stack OOF ≥ 0.66 → **positive** (paradigm 회수 성공)
  - 0.65 ≤ OOF < 0.66 → **partial**
  - OOF < 0.65 → **negative** (feature 확장 실패, plan-016 deep path-pivot)
- fail trigger: 모든 stage 가 negative/marginal → best = plan-014 baseline (= submission 동일, plan-015 = no improvement).

#### G_final — synthesis + LB carry-over

- artifact: `plans/plan-015-feature-expansion.results.md` 신규 + frontmatter sync + plan-016 후보
- LB carry-over (Q3 결정): plan-015 best_stack + plan-014 best_stack 두 submission 모두 dacon-submit (1회 each) — 2 LB 값 비교 = paradigm path 의 *measured* 비교 reference.
- content:
  - G0~G5 결과 narrative (각 step Δ + drop event 박제)
  - band 분류 결과
  - feature attribution (각 feature 의 net 기여 measured)
  - LB measured (plan-015 best + plan-014 best 2 값)
  - plan-016 후보 ≥ 3 (band 별 분기)
- fail trigger: 3 파일 sync 누락 → `final_sync_missing` severe

### §3.4 exp_id naming + registry append schema

| exp_id | stage | config_path |
|---|---|---|
| H042_g0_preflight | G0 | `analysis/plan-015/preflight.py` |
| H043_g1_e1_feature_A | G1 | `analysis/plan-015/g1_e1_feature_A.py` |
| H044_g2_e2_feature_AB | G2 | `analysis/plan-015/g2_e2_feature_AB.py` |
| H045_g3_e3_feature_ABC | G3 | `analysis/plan-015/g3_e3_feature_ABC.py` |
| H046_g4_e4_feature_ABCD | G4 | `analysis/plan-015/g4_e4_feature_ABCD.py` |
| H047_g5_best_stack_5fold | G5 | `analysis/plan-015/g5_best_stack.py` |
| H048_g_final_synthesis | G_final | (results.md + sync, no script) |

registry.csv schema = plan-014 §3.4 G_final carry (12 columns: id / plan_id / type / status / started_at / finished_at / duration_sec / run_dir / config_path / baseline_id / corrects / notes). baseline_id chain:
- G0.baseline_id = `H041_g5_phase4_final` (plan-014 last row)
- G1.baseline_id = G0 id, G2.baseline_id = G1 id, ... (chain)
- G_final.baseline_id = G5 id

---

## §4. STAGE 0 (c3, G0) — preflight [TODO]

### §4.1 산출물

- `analysis/plan-015/preflight.py` — 2 task 일괄 실행:
  - (a) plan-014 baseline (E0c K-Means K=9 + boundary_weight_on, F0 frozen) 5-fold OOF 재산출 → 0.6425 ± 0.005 reproduce 확인
  - (b) 4 feature (A/B/C/D) 단독 적용 시 input pipeline shape sanity (no NaN, dim match)
- `analysis/plan-015/preflight.json` — schema = §3.3 G0
- registry row: `H042_g0_preflight`

### §4.2 실행

```bash
python analysis/plan-015/preflight.py \
  --out analysis/plan-015/preflight.json
```

plan-014 module (`src/pb_0_6822/plan014_paradigm.py`) reuse OK — corrector arch / F0 frozen / loss 모두 carry. plan-015 의 새 feature 함수는 별도 module (`src/pb_0_6822/plan015_features.py`) 으로 분리.

### §4.3 G0 합격

- (a) reproduce hit@1cm ∈ [0.6375, 0.6475] (= 0.6425 ± 0.005)
- (b) 4 feature single-apply 시 input shape (N, 6, target_dim) NaN/Inf 0

위반 시 `preflight_artifact_missing` severe.

---

## §5. STAGE 1 (c4, G1, E1) — feature A only (F0 residual direct) [TODO]

### §5.1 산출물

- `src/pb_0_6822/plan015_features.py` — A/B/C/D feature 함수 정의 (plan-014 `make_seq_features` 의 확장 wrapper)
- `analysis/plan-015/g1_e1_feature_A.py` — E1 config × 5-fold OOF
- `analysis/plan-015/g1_e1.json` — schema = §3.3
- registry row: `H043_g1_e1_feature_A`

### §5.2 spec

- input dim = 12D (9D plan-014 + 3D F0 residual)
- corrector arch / loss / lever / F0 frozen 모두 plan-014 G5 best_stack carry
- 5-fold OOF (stable_hash carry)

### §5.3 G1 합격

- ΔOOF(E1 vs G0 baseline) ≥ +0.005 → `positive`, G2 진행
- 0 ≤ Δ < +0.005 → `marginal`, G2 진행 + warn flag
- Δ < 0 → `e1_negative` warn, G2~G4 skip → G_final 직행 (best = baseline)

---

## §6. STAGE 2 (c5, G2, E2) — A+B (binormal split) [TODO]

### §6.1 산출물

- `analysis/plan-015/g2_e2_feature_AB.py`
- `analysis/plan-015/g2_e2.json`
- registry row: `H044_g2_e2_feature_AB`

### §6.2 spec

- input dim = 13D (12D + 1D binormal split)
- 외 plan-014 carry

### §6.3 G2 합격

- ΔOOF(E2 vs G1) ≥ +0.005 → positive, G3 진행
- < 0 → drop B, G3 skip → G_final

---

## §7. STAGE 3 (c6, G3, E3) — A+B+C (multi-scale stride) [TODO]

### §7.1 산출물

- `analysis/plan-015/g3_e3_feature_ABC.py`
- `analysis/plan-015/g3_e3.json`
- registry row: `H045_g3_e3_feature_ABC`

### §7.2 spec

- input dim ≈ 26D (13D × 2 stream τ=1,2)
- BiGRU input_dim 26 으로 변경 (encoder 첫 layer 만 다름, 나머지 carry)
- 5-fold OOF

### §7.3 G3 합격

- ΔOOF(E3 vs G2) ≥ +0.005 → positive, G4 진행
- < 0 → drop C, G4 skip → G_final

---

## §8. STAGE 4 (c7, G4, E4) — A+B+C+D (pairwise) [TODO]

### §8.1 산출물

- `analysis/plan-015/g4_e4_feature_ABCD.py`
- `analysis/plan-015/g4_e4.json`
- registry row: `H046_g4_e4_feature_ABCD`

### §8.2 spec

- input dim ≈ 32D (26D + 6D pairwise)
- 5-fold OOF

### §8.3 G4 합격

- ΔOOF(E4 vs G3) ≥ +0.005 → positive, all features 채택
- < 0 → drop D, best = G3 (A+B+C)

---

## §9. STAGE 5 (c8, G5) — best stack 5-fold + submission [TODO]

### §9.1 best 선정

cumulative ΔOOF 추적:
```
candidates = {
    "baseline": 0.6425,  # G0
    "E1 (A)": G1_oof,
    "E2 (A+B)": G2_oof,
    "E3 (A+B+C)": G3_oof,
    "E4 (A+B+C+D)": G4_oof,
}
best_name = argmax(candidates)
best_oof = candidates[best_name]
```

drop rule (§3.2): negative stage 이후 stages 는 candidates 에서 제외.

### §9.2 산출물

- `analysis/plan-015/g5_best_stack.py` — best config 의 test 5-fold ensemble (이미 G_n 에서 5-fold OOF 산출 → test 만 새로 산출)
- `runs/baseline/plan015_g5/submission_best.csv`
- `analysis/plan-015/g5_phase4.json`
- registry row: `H047_g5_best_stack_5fold`

### §9.3 G5 합격

- best_stack 5-fold OOF ≥ 0.6475 (= 0.6425 + 0.005)
- band 분류 (§3.3)
- 위반 (best == baseline 또는 < 0.6475) → `g5_no_improvement` warn → submission = plan-014 best (baseline 동일)

---

## §10. STAGE 6 (c9, G_final) — synthesis + plan-016 + LB carry-over [TODO]

### §10.1 산출물

- `plans/plan-015-feature-expansion.results.md` 신규 (frontmatter + G0~G5 narrative + band + feature attribution + plan-016 후보)
- plan-015 frontmatter sync (status spec → G_final_complete, exp_ids fill, band, best_stack_5fold_oof, lb_score)
- registry append 7 row (H042~H048) — 이미 incremental 완료 기대
- **LB carry-over (Q3)**: dacon-submit 2회 — plan-014 best (`runs/baseline/plan014_g5_phase4/submission_best.csv`) + plan-015 best (`runs/baseline/plan015_g5/submission_best.csv`). 두 LB 값 frontmatter 박제.

### §10.2 plan-016 후보 (band 별 분기, ≥ 3)

#### 공통 (모든 band)

- **(공통-1) Multi-seed 분산** — plan-015 best 의 5-seed × 5-fold std (single seed = 20260514)
- **(공통-2) Feature attribution full factorial** — 4 feature × 2^4 = 16 sub-exp (순차 ablation 의 interaction 측정)

#### Band positive (≥ 0.66)

- **(positive-1) Polish + ensemble with plan-013** — plan-013 fallback 0.6381 + plan-015 best 좌표 mean
- **(positive-2) Code 제출 / 더 빠른 inference**

#### Band partial (0.65 ≤ OOF < 0.66)

- **(partial-1) plan-013 + plan-015 ensemble** — Candidate C 변형 (plan-014 §1.4 row 5 evolved)
- **(partial-2) Higher-order features** (jerk², snap) 추가

#### Band negative (< 0.65)

- **(negative-1) Deep path-pivot** — KNN-based corrector (plan-014 §10.2 negative-1 carry)
- **(negative-2) Task framing 변경** — 11-step seq2seq transformer / Neural ODE 등
- **(negative-3) DACON 236716 의 framework family ceiling 정량 박제** — 더 이상 ROI 낮음 판단 시 작업 중단

### §10.3 G_final 합격

- 3 파일 sync + plan-016 후보 ≥ 3 + LB carry-over 2회 (plan-014 best + plan-015 best)
- 누락 시 `final_sync_missing` severe

### §10.4 종료

- §0.5 c9 [TODO]→[DONE] sync commit + push
- telegram alert (§12.4): `"plan-015 완료, band=<...>, best_stack=X.XXXX, LB_plan014=X.XXXX, LB_plan015=X.XXXX"`
- `/loop` 자연 종료

---

## §N+4. 변경 이력

- v1 (2026-05-14): 1, 2 순위 feature (A/B/C/D) spec 박제. G-gate / 실험 spec 은 v2 carry.
- v2 (2026-05-14): §3 v1 후속 사항 4개 (순차 ablation / baseline 0.6425 / 합격 기준 Δ+band / exp_id naming) 모두 박제. §4~§10 STAGE 본문 추가 (G0~G_final 6 stage). drop rule 도입 (negative stage 이후 skip). exp_ids H042~H048 예약. LB carry-over 2회 spec (plan-014 best + plan-015 best).

---

## §N+5. 참조

- `plans/plan-014-plan012-failure-inversion.results.md` — band=negative, 회수율 5.4%, oracle 0.8248, baseline 0.6425
- `plans/plan-013-plan004-framework-3lever-stacking.results.md` — LB 0.6381 join row 4
- `plans/plan-005-pb-0-6822-diagnostic.md` — binormal axis error 0.64cm evidence
- `notes/new-ideas.md` — A.2 (FFT N=11 fatal), B.3 (corrector 회수 한계 진단)
- `notes/코드공유-upgrade.md` — C010 frenet-anisotropic-loss / Idea 1 continuous regime
- `src/pb_0_6822/plan014_paradigm.py` — 현 9D feature + corrector 구현부 (plan-015 carry base)
