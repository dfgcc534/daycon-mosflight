---
plan_id: 015
version: 2.3 (spec patch — plan-review-master iter 3 fix 7건. (1) §1 Feature B acc_normal/binormal 산식 단일화 (raw acc · n̂ / b̂, sign 보존 — 이전 ‖acc_perp‖ 와 동일해지는 문제 제거). (2) §5.3/§6.3/§7.3/§8.3 marginal anchor inheritance 명기 (G_(n+1) anchor = G_n cumulative). (3) §1 Feature C τ=2 alignment spec 단일화 (alignment 무시, BiGRU 흡수). (4) §3.3 G0(b) Feature C 단독 18D = 9D base (cumulative 26D 의 13D base 와 다름) 명시. (5) §9.1 candidates baseline = plan-015 G0 재현 OOF (plan-014 hard-coded 0.6425 아님) — fair comparison. (6) §5.1 plan015_features.py signature 박제 (`make_seq_features_v2`, feature_flags dict). (7) §7.2 BiGRU input_dim 변경 시 weight 재초기화 (Kaiming, seed=20260514 carry, plan-014 weight transfer 안 함). v2.2 → v2.3.)
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
- **C** Multi-scale stride — base feature 를 τ ∈ {1, 2} 2 stream concat (τ=3 step 부족으로 제외, §1 v2.1 단일화). 13D × 2 stream = **26D when A+B applied**.
- **D** Pairwise cross-step interaction — step t vs t-2 / t-4 의 cosine similarity + Δspeed (Δangle 은 cosine 의 monotone mapping 이므로 제외, v2.2 단일화). 3 pair × 2 stat = **+6D**.

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
- Δ ≥ +0.005 → step `positive` (해당 feature 채택, 다음 step 의 anchor = 본 step cumulative OOF)
- 0 ≤ Δ < +0.005 → step `marginal` (해당 feature 채택, warn flag 박제, **다음 step anchor = 본 step cumulative OOF** = positive 와 동일 inheritance)
- Δ < 0 → step `negative` (해당 feature drop, **이후 stage 모두 skip → G_final 직행**. best = 이전 step 의 cumulative OOF, 그 stage 의 config 가 best_stack)

**Anchor inheritance (v2 단일화)**: 매 stage `G_n` 의 ΔOOF 비교 anchor = **immediate prior stage G_(n-1) 의 cumulative OOF** (positive/marginal 모두 inherited). G5 best 선정 시 cumulative best = max over {G0, G1, ..., G_n_last} (n_last = negative 직전 또는 G4 끝까지 도달 시 G4).

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
| c2.1 | docs | **v2.1 spec patch — plan-review-master iter 1 fix 6건.** (1) §1 Feature A residual modal switch 단일화 (displacement_F0 sign convention). (2) §1 Feature C dim 모순 제거 (26D cumulative, τ=1,2 2 stream). (3) §3.3 G0 (b) feature dim 도출식 박제. (4) anchor inheritance 단일화 (immediate prior cumulative). (5) §3.1 baseline reduction = 5-fold concat hit. (6) §1 Feature B Frenet basis 산출식 + edge case 박제. v2 → v2.1 | [DONE] 0c53cd9 |
| c2.2 | docs | **v2.2 spec patch — plan-review-master iter 2 fix 6건.** (1) §1 Feature C τ=2 step indices `range(3,11,2)`=[3,5,7,9] 4 step + pad rule 박제. (2) §3.2 sub-exp matrix base column anchor inheritance 정합화 (G_(n-1) cumulative, negative → G_final). (3) §0.5 Quick Ref C feature 39D → 26D 갱신. (4) §1 Feature D stat 정의 명료화 (cosine + Δspeed 2 stat × 3 pair = 6D, Δangle 제외). (5) §1 Feature B edge case 단일화 (world ẑ post-ortho). (6) B/C/D narrative 가설 motivation 박제. v2.1 → v2.2 | [DONE] 612f92e |
| c2.3 | docs | **v2.3 spec patch — plan-review-master iter 3 fix 7건.** (1) §1 Feature B acc_normal/binormal = raw acc · n̂/b̂ sign 보존 (정보 손실 해결). (2) §5~§8 marginal anchor inheritance 명기. (3) Feature C τ=2 alignment 단일화. (4) §3.3 G0(b) C 단독 18D base 9D 명시. (5) §9.1 candidates baseline = G0 재현 OOF. (6) §5.1 plan015_features.py signature 박제. (7) §7.2 weight 재초기화 spec. v2.2 → v2.3 | [DONE] d4cb908 |
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

- **정의 (v2 단일화 spec)**: per-step `displacement_F0[s] = F0_pred[s] − X[:, s]` (= step s 시점에서 F0 가 prediction 하는 위치 − 관측 위치, displacement vector). 3D channel. **부호 convention: F0_pred − X (negative residual 아님)** — "F0 가 향하는 방향" 자체를 encoder 가 보게 함.
- **dim**: +3D (9D → 12D)
- **F0_pred[s] 산출** (step-local horizon=2 prior, plan-014 §A.1 finite-diff carry):
  `F0_pred[s] = X[:, s] + 1.98·v_last_s + 1.20·acc_par_vec_s + (−0.20)·acc_perp_vec_s`
  where `v_last_s = X[:, s] − X[:, s−1]`, `acc_s = X[:, s] − 2·X[:, s−1] + X[:, s−2]`, t̂_s = v_last_s/‖v_last_s‖, `acc_par_vec_s = (acc_s · t̂_s)·t̂_s`, `acc_perp_vec_s = acc_s − acc_par_vec_s`.
- **구현**: `make_seq_features` 에서 step `s` 마다 위 산식으로 `F0_pred[s]` 산출 후 `displacement_F0 = F0_pred[s] − X[:, s]` 3D 를 8d turn features + direction(1D) 와 concat → per-step 12D.
- **edge case** (s ∈ {0, 1, 2}, finite-diff 정의 안 됨): `displacement_F0[s] = (0, 0, 0)` zero-fill. baseline `end_idx=10` + `range(max(3, end_idx-5), end_idx+1)` pad rule 로 사실 step 3~10 만 사용 → edge case 미발생.

#### B. Frenet binormal axis 분리

- **가설 (회수율 sub-failure 매핑)**: 현 `perp_norm/speed` 1D 는 normal + binormal magnitude 의 *RMS-합* 으로 두 방향 정보 손실. plan-005 진단 = binormal axis 의 error 0.64cm (normal axis 4.51cm 의 1/7). **두 방향의 magnitude 가 다른 정보를 담음** → split 시 corrector 가 binormal-driven sample (소수지만 specific 한 회수 후보) 을 정밀 회수 가능. plan-014 G2 oracle ceiling 도 Frenet-orthogonal codebook 이 최고 (0.8248) → Frenet basis 의 정밀한 분해가 회수율 개선의 직접 신호.
- **정의 (v2 명료화 spec)**: 현 `perp_norm/speed` (1D, normal+binormal magnitude 합) 을 `normal_norm/speed` (1D) + `binormal_norm/speed` (1D) 2D 로 split.
- **dim**: +1D net (split 후 9D 의 (5) 자리 1D → 2D 로 늘어남, 9D − 1D + 2D = 10D 단독 / A+B cumulative = 13D)
- **step-local Frenet basis 산출 (n̂_s, b̂_s)**:
  - `v_s = X[:, s] − X[:, s−1]`, `t̂_s = v_s / (‖v_s‖ + ε)` (ε = 1e-12 numeric stability)
  - `acc_s = X[:, s] − 2·X[:, s−1] + X[:, s−2]` (raw acc)
  - `acc_perp_vec_s = acc_s − (acc_s · t̂_s)·t̂_s` (= acc 의 perp plane projection)
  - `n̂_s = acc_perp_vec_s / (‖acc_perp_vec_s‖ + ε)` — normal direction
  - `b̂_s = t̂_s × n̂_s` — binormal direction (오른손 법칙)
  - **edge case** (degenerate motion, v2.2 단일화): `‖v_s‖ < ε_basis = 1e-6` 또는 `‖acc_perp_vec_s‖ < ε_basis` 시 → `n̂_s = world ẑ` post-orthogonalize (`n̂_s ← n̂_s − (n̂_s · t̂_s)·t̂_s`, 재정규화). plan-014 §A.1 carry. (acc_normal/binormal 0 fallback 옵션 제외 — basis 구성 후 정상 산출.)
- **feature split 산출** (v2.3 명료화, sign 보존 단일화):
  - `acc_normal = acc_s · n̂_s` (raw acc 의 n̂_s 성분 scalar projection, sign 보존 — `acc_perp · n̂` 가 아닌 raw acc · n̂)
  - `acc_binormal = acc_s · b̂_s` (raw acc 의 b̂_s 성분 scalar, sign 보존)
  - normalize: `acc_normal/speed`, `acc_binormal/speed` (둘 다 sign 보존 — abs() 적용 X)
  - 두 channel 이 모두 sign 정보 가짐 → "방향 + magnitude" 정보 분리 의미 보존 (이전 `‖acc_perp‖/speed` 1D 는 abs magnitude 만, split 후 sign-aware 2D 로 정보량 ↑).
- **구현**: `_turn_features_per_step` 의 기존 `perp_norm/speed` 1D 자리에 `(acc_normal/speed, |acc_binormal|/speed)` 2D 로 swap. 단독 = 10D, A+B = 13D.

### 2순위 — 표현력 보강

#### C. Multi-scale stride features

- **가설 (회수율 sub-failure 매핑)**: 현 9D feature 는 step gap 1 (40ms) 의 high-frequency kinematic signal 만 추출. 모기 wingbeat 같은 short-period oscillation 은 step gap 1 에서 aliased noise 로 보이지만, step gap 2 (80ms) 에서 *integrated maneuver* signal 로 분리 가능. plan-014 G4 E7 (LastStep MLP −0.005) = 6-step BiGRU 자체는 가치 있지만, *동일 stride* 만 사용 → 시간 scale 분리 안 됨. 2 stride concat = encoder 가 dual-scale 패턴 동시 학습.
- **정의 (v2 단일화 spec)**: A+B 적용 후 13D base feature 를 stride τ ∈ {1, 2} 2 stream 계산 → per-step concat = **26D**. (τ=3 stride 는 11-step trajectory 에서 step 수 부족으로 제외.)
- **dim**: per-step 13D × 2 stream = **26D** (A+B+C cumulative)
- **구현**: `make_seq_features` 에서 step indices 산출 2 set (v2.2 단일화, python `range` 실 결과 박제):
  - **τ=1 (기존)**: `list(range(max(3, end_idx-5), end_idx+1, 1))` → end_idx=10 시 `[5, 6, 7, 8, 9, 10]` (6 step, gap 1)
  - **τ=2**: `list(range(max(3, end_idx-10), end_idx+1, 2))` → end_idx=10 시 `max(3, 0)=3` → `range(3, 11, 2)` = `[3, 5, 7, 9]` (4 step). **pad rule**: 6 step 보다 부족 시 `indices = [indices[0]] * (6 − len(indices)) + indices` (plan-014 §A.1 carry) → `[3, 3, 3, 5, 7, 9]` (6 step).
  - 각 τ stream 의 per-step 13D feature (A+B 적용) 산출 후 *동일 position* (6개 step slot) 끼리 axis=-1 concat → per-step 26D.
  - **alignment spec (v2.3 단일화)**: τ=1 position `p` 의 시각 vs τ=2 position `p` 의 시각은 *다름* (e.g., position 0 = τ=1 의 step 5 vs τ=2 의 step 3 pad). **alignment 무시, BiGRU 의 sequential 학습이 흡수** 채택 (= dual-scale feature 가 position 별 시간 misalignment 와 함께 input). controlled comparison 관점: τ=1 + τ=2 concat 이 단순 stream concat 이지 strict time-align stream 아님. plan-016 후보 = strict alignment 도입 (예: τ=2 의 step 5 와 τ=1 의 step 5 정렬).

#### D. Pairwise cross-step interaction

- **가설 (회수율 sub-failure 매핑)**: BiGRU 은 sequential pairwise 패턴은 학습하지만, *long-range pairwise* (t vs t-4) 의 explicit signal 은 6-step hidden state 안 implicit 표현 → corrector 가 *직접 보기* 어려움. plan-014 G4 E7 (LastStep MLP −0.005, 시계열 가치 입증) carry — BiGRU 가 충분 표현력 가지지만 *long-range pairwise gap* 은 explicit feature 로 보완 시 추가 회수.
- **정의 (v2.2 명료화)**: per-step feature 에 cross-step pairwise 추가. velocity vector pair 3개 × stat 2개 = **6D**. (이전 본문의 "Δangle" 는 cosine similarity 와 중복 정보이므로 *제외* — Δangle = acos(cosine similarity) 이라 단조 mapping.)
- **dim**: +6D (3 pair × 2 stat = 6D) per-step = 26D + 6D = **32D when A+B+C applied**
- **stat 2개** (per pair):
  - `cosine_similarity(v[s_a], v[s_b])` = `v[s_a] · v[s_b] / (‖v[s_a]‖·‖v[s_b]‖ + ε)`
  - `Δspeed = ‖v[s_a]‖ − ‖v[s_b]‖`
- **3 pair** (per-step `s` 의 velocity `v[s] = X[:, s] − X[:, s−1]` 기준):
  - `(s, s-2)`, `(s, s-4)`, `(s-2, s-4)`
- **edge case**: s=3, 4 일 때 s-4 < 0 → 해당 pair stat 0 fill (즉 v[s-4] 미정의 시 cosine=0, Δspeed=0). baseline `end_idx=10` + step indices `[5,6,7,8,9,10]` 기준 모든 step 에서 s-4 ≥ 1 valid → edge case 발생 안 함.

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
| **plan-014 G5 best_stack 5-fold OOF** ★ | **0.6425** | plan-014 G5 (H041), = plan-015 baseline. **reduction = 5-fold concat hit@1cm** (= `mean(‖oof_pred − y_true‖₂ ≤ 0.01m)` over all 10000 samples, 각 sample 이 정확히 1번 val 등장; *fold-mean of fold-means* 아님). |
| oracle ceiling (E0b Frenet-ortho) | 0.8248 | plan-014 G0 |
| corrector 회수율 | 5.4% | (best − F0) / (oracle − F0) = 0.0105/0.1928 |

### §3.2 Sub-exp matrix (순차 ablation)

| stage | sub-exp | feature config | dim | base |
|---|---|---|---|---|
| G1 | **E1** (A) | F0 residual direct | 12D | plan-014 best_stack |
| G2 | **E2** (A+B) | + binormal split | 13D | **G1 cumulative** (positive/marginal 모두 inherited). G1 negative → G2 skip → G_final 직행 |
| G3 | **E3** (A+B+C) | + multi-scale stride (τ=1,2 stream) | 26D | **G2 cumulative** (positive/marginal). G2 negative → G3 skip → G_final 직행 |
| G4 | **E4** (A+B+C+D) | + pairwise cross-step | 32D | **G3 cumulative** (positive/marginal). G3 negative → G4 skip → G_final 직행 |
| G5 | **best** | cumulative best (max ΔOOF over G0/G1/G2/G3/G4) | varies | G_final 의 submission base |

**Drop rule** (v2 patch 결정): 만약 G_n 의 ΔOOF < 0 (negative), 해당 stage feature drop + **G_(n+1)~G_4 모든 후속 stage skip** → G_final 직행. best = G_(n−1) cumulative (해당 stage 직전의 cumulative OOF) — 그 stage 의 config 가 best_stack 으로 채택.

(예: G2 (A+B) 가 G1 (A) 대비 -0.003 이면 → B drop + C/D 시도 skip → best = G1 (A only).)

### §3.3 G-gate quantitative criteria

#### G0 — preflight

- artifact: `analysis/plan-015/preflight.json`
- **(a) plan-014 baseline 5-fold reproduce**: 동일 config (E0c K-Means K=9 + boundary_weight_on, F0 frozen) 으로 5-fold OOF 재산출 → 0.6425 ± 0.005 일치 확인.
- **(b) feature dim sanity**: 4 feature (A/B/C/D) 각각 *단독 적용* 시 shape verify. 단독 dim 도출식:
  - A 단독: 9D base + 3D (displacement_F0) = **12D**
  - B 단독: 9D base 의 `(5) perp_norm/speed` 1D 를 `(5a) normal_norm/speed + (5b) binormal_norm/speed` 2D 로 split → 9D − 1D + 2D = **10D**
  - C 단독: **9D base (A/B 미적용 raw plan-014 feature)** × 2 stream (τ=1, τ=2) = **18D** (cumulative A+B+C 의 26D 와 다름 — cumulative 는 13D base × 2 stream. 단독 sanity check 시 base 가 plan-014 의 9D feature 임을 명시)
  - D 단독: 9D base + 6D pairwise (3 pair × 2 stat) = **15D**
  cumulative 적용 시 dim: A=12D, A+B=13D, A+B+C=26D, A+B+C+D=32D (§3.2 표 carry).
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

- `src/pb_0_6822/plan015_features.py` — A/B/C/D feature 함수 정의 (plan-014 `make_seq_features` 의 확장 wrapper). **signature**: `make_seq_features_v2(X: np.ndarray, end_idx: int = 10, direction: float = 1.0, *, feature_flags: dict[str, bool]) -> np.ndarray` where `feature_flags = {"A": bool, "B": bool, "C": bool, "D": bool}`. plan-014 module *import* (monkey-patch 아님), 별도 함수로 wrap. cumulative E_n 시 `feature_flags = {f: f in {"A", ..., n-th}}` 활성. shape `(N, 6, target_dim)` 반환, target_dim 은 §3.3 G0(b) carry table.
- `analysis/plan-015/g1_e1_feature_A.py` — E1 config × 5-fold OOF
- `analysis/plan-015/g1_e1.json` — schema = §3.3
- registry row: `H043_g1_e1_feature_A`

### §5.2 spec

- input dim = 12D (9D plan-014 + 3D F0 residual)
- corrector arch / loss / lever / F0 frozen 모두 plan-014 G5 best_stack carry
- 5-fold OOF (stable_hash carry)

### §5.3 G1 합격

- ΔOOF(E1 vs G0 baseline) ≥ +0.005 → `positive`, G2 진행 (G2 의 anchor = **G1 cumulative OOF**)
- 0 ≤ Δ < +0.005 → `marginal`, G2 진행 + warn flag (**G2 의 anchor = G1 cumulative OOF**, positive 와 동일 inheritance, §0.5/§3.2 carry)
- Δ < 0 → `e1_negative` warn, G2~G4 skip → G_final 직행 (best = baseline G0)

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

- ΔOOF(E2 vs G1 cumulative) ≥ +0.005 → positive, G3 진행 (G3 anchor = G2 cumulative)
- 0 ≤ Δ < +0.005 → marginal + warn flag, G3 진행 (G3 anchor = G2 cumulative, positive 동일 inheritance)
- Δ < 0 → drop B, G3~G4 skip → G_final (best = G1 cumulative)

---

## §7. STAGE 3 (c6, G3, E3) — A+B+C (multi-scale stride) [TODO]

### §7.1 산출물

- `analysis/plan-015/g3_e3_feature_ABC.py`
- `analysis/plan-015/g3_e3.json`
- registry row: `H045_g3_e3_feature_ABC`

### §7.2 spec

- input dim = **26D** (13D × 2 stream τ=1,2, A+B+C cumulative)
- BiGRU input_dim = 26 (G2 의 13 에서 변경). **encoder 첫 layer (input_proj of GRU) 만 input_dim 다름, 나머지 hyperparam carry** (hidden=128, num_layers=2, bidirectional=True, dropout=0.1). **weight 재초기화 (v2.3 결정)**: input_dim 변경 시 전체 model state 재초기화 (PyTorch default Kaiming init, seed=20260514 carry). plan-014 의 학습된 weight transfer 안 함 — fair comparison 위해 모든 G_n 이 동일 init + 동일 데이터로부터 학습.
- 5-fold OOF

### §7.3 G3 합격

- ΔOOF(E3 vs G2 cumulative) ≥ +0.005 → positive, G4 진행 (G4 anchor = G3 cumulative)
- 0 ≤ Δ < +0.005 → marginal + warn flag, G4 진행 (G4 anchor = G3 cumulative)
- Δ < 0 → drop C, G4 skip → G_final (best = G2 cumulative)

---

## §8. STAGE 4 (c7, G4, E4) — A+B+C+D (pairwise) [TODO]

### §8.1 산출물

- `analysis/plan-015/g4_e4_feature_ABCD.py`
- `analysis/plan-015/g4_e4.json`
- registry row: `H046_g4_e4_feature_ABCD`

### §8.2 spec

- input dim ≈ 32D (26D + 6D pairwise)
- 5-fold OOF

### §8.3 G4 합격 (마지막 stage)

- ΔOOF(E4 vs G3 cumulative) ≥ +0.005 → positive, **all 4 features (A+B+C+D) 채택** → best_stack = G4 cumulative
- 0 ≤ Δ < +0.005 → marginal + warn flag, all 4 features 채택 → best_stack = G4 cumulative (Δ 만 박제)
- Δ < 0 → drop D, best_stack = G3 cumulative (A+B+C)

---

## §9. STAGE 5 (c8, G5) — best stack 5-fold + submission [TODO]

### §9.1 best 선정

cumulative ΔOOF 추적:
```
candidates = {
    "baseline": G0_reproduce_oof,  # plan-015 G0 재현 값 (= 0.6425 ± 0.005)
    "E1 (A)": G1_oof,
    "E2 (A+B)": G2_oof,
    "E3 (A+B+C)": G3_oof,
    "E4 (A+B+C+D)": G4_oof,
}
best_name = argmax(candidates)
best_oof = candidates[best_name]
```

**baseline 값 결정 (v2.3 단일화)**: candidates 의 "baseline" = **plan-015 G0 재현 OOF** (= 0.6425 ± 0.005 reproduce 결과). plan-014 의 hard-coded 0.6425 가 아닌 plan-015 자체 재현값 사용 — fair comparison (G0~G4 모두 동일 5-fold scheme + 동일 seed + 동일 코드 path 산출).

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
