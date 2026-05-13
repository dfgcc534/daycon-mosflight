---
plan_id: 012
version: 2
date: 2026-05-13 (Asia/Seoul)
status: written (v2 spec replacement — codebook bake-off + 3D fix per user direction)
based_on:
  - 004
  - 005
  - 006
  - 007
  - 010
  - 011
  - notes/PB_0.6822 코드공유.ipynb
  - notes/코드공유-upgrade.md
followed_by:
  - 012.1 (LB carry-over; user manual dacon-submit)
scope: 단일공식 + corrector path 의 *paradigm reframe*. plan-005~011 의 residual *regression* path 폐기. *3D Frenet 또는 world frame anchor codebook (7 또는 9 후보) classification + per-mode regression head (~0.5cm 미세조정)* hybrid. 단일공식 (frenet_par120_perp_neg020, plan-006 LB 0.6692) 의 raw hit potential (G0 측정) 위에 *3-way codebook bake-off* (Absolute world-frame 7-Way ↔ Frenet-Orthogonal 7-Way ↔ K-Means data-driven 7-Way) 를 Phase 1 에서 비교 → winner codebook 위에서만 Phase 2~3 ablation 진행 + 5-fold submission. plan-011 처럼 LB carry-over (0 제출).
exp_ids:
  - H019_phase0-preflight-codebook        # G0 — F0 raw hit + oracle ceiling + K-Means anchor 산출 + plan-006 reproduce
  - H020_phase1-codebook-bakeoff          # G1 — 3 sub-exp (E0a Absolute, E0b Frenet-Orthogonal, E0c K-Means) winner 결정
  - H021_phase2-frame                     # G2.E1 — winner codebook 위 frame swap (조건부, codebook frame-dependent 시만)
  - H022_phase2-codebook-K                # G2.E2 — winner codebook 의 K density (5 / 7 / 9 / 13) swap
  - H023_phase2-temperature               # G2.E3 — τ scan (argmax + 5 points)
  - H024_phase2-loss                      # G2.E4 — L7 hinge vs distance regression
  - H025_phase2-reg-head                  # G2.E5 — classification only vs classification + regression head (hybrid)
  - H026_phase3-boundary-weight           # G3.E6 — boundary sample weighting on/off
  - H027_phase3-scorer-arch               # G3.E7 — full Attn-GRU vs last-step MLP (★ 시계열 input 가치 측정)
  - H028_phase3-r0-prior                  # G3.E8 — r=0 logit prior 강도 (0 / +0.5 / +1.0)
  - H029_phase4-final-5fold               # G4 — best stack 5-fold + submission 박제
lb_score: null
---

# plan-012 v2 — Codebook Bake-off Classification + Regression Hybrid (paradigm reframe, 3D)

## §0. 한 줄 목적

> **plan-005~011 의 residual regression path 폐기 → *3-way codebook bake-off* (Phase 1) 으로 anchor 좌표 paradigm 자체를 데이터가 결정하게 함**. 후보:
> - **E0a Absolute-7Way** — world frame ±x, ±y, ±z + center (= "절대 방향 prior 0"; frame inductive bias 없음)
> - **E0b Frenet-Orthogonal-7Way** — Frenet local frame ±t, ±n, ±b + center (= trajectory-aligned prior, 사용자 방법 1)
> - **E0c K-Means-7Way** — train residuals 의 Frenet 변환 위 K-Means K=7 cluster center + center origin (= 데이터-주도 prior, 사용자 방법 2)
>
> 3 codebook 공통 = 7 anchor, 3D shape (data train_y = (x, y, z)), classifier head (7 logit) + regression head (3D offset, scale prior 0.5cm) **hybrid** — classifier 가 mode 선택 + regression 이 cluster 내 미세 offset 회귀. *mean-regression trap* (soft-mean 의 평균 회귀) 회피 + *직접 residual 회귀의 scale 문제* (cluster 내 residual scale ~0.5cm → 학습 trivial) 동시 해결.
>
> **plan-010/011 과의 path 분리** (재확인):
> - plan-010 = corrector path *depth*. 4 후보 marginal.
> - plan-011 = corrector path *breadth*. 24 sub-exp 중 1 positive (In/ID +0.0050).
> - **plan-012 = paradigm 자체 교체** — codebook + classifier + reg head hybrid 로 *residual 회귀가 풀지 못한 destructive band* 직접 우회.
>
> **v1 (2D ring) 폐기 근거**: `read_labels` ([selector.py:172](src/pb_0_6822/selector.py#L172)) 가 `(x, y, z)` 3D 반환 — v1 의 `(B, 2)` 가정 = **z 축 (= binormal) 전체 무시** spec 오류. plan-005 §1.3 의 `binormal_delta_norm_mean = 0.0064 (= parallel 의 1/7)` evidence 와 동일 양상 (corrector 도 binormal 약했는데 v1 ring 은 아예 cover X). v2 = 3D + ±b 포함 + 사용자 제안 두 방법 (Frenet ortho + K-Means) 직접 비교.
>
> **재사용 / 비재사용**:
> - 재사용 = `selector.make_seq_features` (시계열 9-dim × 6-step, plan-004 그대로) + `selector.CandidateAttentionGRUSelector` encoder + plan-006 single formula `frenet_par120_perp_neg020`.
> - 비재사용 = `boundary.py` (corrector path 전체), plan-010 `corrector_redesign.py`, plan-011 `corrector_redesign_v2.py`, plan-008 `candidates_extended.py`.
>
> **Target**: 5-fold OOF ≥ 0.66 (= plan-006 corrected 0.6491 위 +0.011), LB 추정 0.68~0.73. oracle ceiling (G0, anchor-aware) = 0.84 근방.
>
> **LB 제출 정책**: 본 plan 내 LB 제출 **0 회** (plan-009.1+010.1+011.1 carry-over pattern). plan-012.1 carry-over.

---

## §0.5 Quick Reference (autonomous loop 매 turn 읽는 section)

### 합격 기준 (G-gate sequence)

- **G0** (Phase 0 preflight + codebook prep): (a) F0 단일공식의 raw hit@1cm 박제 + hit@1.5cm 박제 (= "64/84 hypothesis" 실측). (b) oracle ceiling per codebook 박제 (= 각 codebook 의 *argmin distance* anchor 선택 시 hit@1cm; 3 codebook = 3 oracle 값). (c) K-Means anchor 산출 + 박제 (cluster center 7개 in Frenet, sample 수 분포, fold-aware fit 검증). (d) plan-006 reproduce drift ≤ 0.005. `analysis/plan-012/preflight.json` 생성. 위반 시 `preflight_artifact_missing` severe.
- **G1** (Phase 1 codebook bake-off, ★ paradigm 결정): 3 sub-exp 동시 학습 (E0a Absolute / E0b Frenet-Orthogonal / E0c K-Means) — *동일 arch + 동일 loss + 동일 τ*, 유일 변수 = anchor 좌표 집합. (a) 3 sub-exp 모두 fold-0 OOF soft hit 박제. (b) winner = `argmax(OOF over E0a, E0b, E0c)`. winner OOF ≥ **0.6450** (plan-011 In/ID anchor). 미달 시 `baseline_below_anchor` warn (Phase 2~3 informational 진행). (c) **tie-break rule** (winner − second < 0.005): 단순성 우선 = `E0a > E0b > E0c` (해석 가능성 + reproduce 용이성 순). winner → §3.4 anchor combo 갱신 + Phase 2/3 의 모든 ablation 의 base.
- **G2** (Phase 2 core ablation on winner, 5 axis × ~14 sub-exp 총합): (a) 5 axis 모두 informational 완료. (b) 5 axis 중 *최소 1 axis* 에서 `max(ΔOOF) ≥ 0.005`. 위반 시 `phase2_no_positive_lever` severe — autonomous recovery (a) Phase 3 진행 후 G_final path-pivot 또는 (b) G1 winner 단독으로 G4 직진.
- **G3** (Phase 3 aux ablation, 3 axis × ~7 sub-exp): 3 axis informational 완료. positive lever 권장 (hard fail 없음).
- **G4** (Phase 4 final 5-fold): best stack (G1 winner + G2 best lever + G3 best lever) 의 5-fold concat OOF ≥ G1 winner + 0.005. submission.csv 생성. 위반 시 `final_no_additive` warn → fallback = G1 winner 단독 5-fold submission.
- **G_final**: synthesis + plan-013 후보 ≥ 3 + 3 파일 frontmatter sync (`lb_score: null` carry-over) + best Phase submission 박제 + plan-012.1 carry-over instruction.

### G-gates

- G0: preflight + 3 codebook ready + K-Means cluster 박제 [TODO]
- G1: 3-way bake-off (E0a/E0b/E0c) — winner 결정 + winner OOF ≥ 0.6450 [TODO]
- G2: Phase 2 core (5 axis × ~14 sub-exp on winner) — 최소 1 axis +0.005 [TODO]
- G3: Phase 3 aux (3 axis × ~7 sub-exp) — informational [TODO]
- G4: best stack 5-fold ≥ G1 winner + 0.005 + submission 박제 [TODO]
- G_final: synthesis + plan-013 후보 + 3 파일 sync + plan-012.1 instruction [TODO]

### Commit chain (next-up)

| # | type | spec section | status |
|---|---|---|---|
| c1 | docs | `plans/plan-012-frenet-ring-classification.md` v1 작성 | [DONE] (e1f08eb — 2D ring spec, 후속 v2 로 대체) |
| c1.1 | docs | v2 spec replacement — codebook bake-off + 3D + classifier+reg head hybrid (★ paradigm shift per user direction) | [TODO] (본 commit) |
| c2 | code | `src/pb_0_6822/ring_classifier.py` — 3 codebook generator + 3D shape + classifier head (7 logit) + regression head (3D, per-mode offset) + L7 hinge loss. spec @ §4 | [TODO] |
| c3 | code+exp | `analysis/plan-012/preflight.py` — F0 raw hit + oracle ceiling per codebook + K-Means anchor 산출 + plan-006 reproduce. spec @ §5 | [TODO] |
| G0 | gate | `preflight.json` 생성 + K-Means cluster 박제 + 3 oracle ceilings + reproduce drift ≤ 0.005 | [TODO] |
| c4 | code+exp | `analysis/plan-012/phase1_bakeoff.py` — 3 sub-exp E0a/E0b/E0c 학습 + winner 결정 + winner_id 박제. spec @ §6 | [TODO] |
| G1 | gate | 3 sub-exp 박제 + winner OOF ≥ 0.6450 (또는 baseline_below_anchor warn) | [TODO] |
| c5 | code | `analysis/plan-012/phase2_core.py` — wrapper for E1~E5 on winner. spec @ §7 | [TODO] |
| c6 | exp | Phase 2.E1 — Frame swap (조건부: winner ∈ {E0b Frenet, E0c K-Means} 이면 Frenet vs world; winner = E0a Absolute 이면 skip per `frame_axis_n/a`) | [TODO] |
| c7 | exp | Phase 2.E2 — Codebook K density swap (winner 의 K=5/7/9/13, 4 sub-exp) | [TODO] |
| c8 | exp | Phase 2.E3 — Temperature scan (argmax + {0.01, 0.03, 0.1, 0.3, 1.0}, 6 sub-exp) | [TODO] |
| c9 | exp | Phase 2.E4 — Loss swap (L7 hinge vs distance regression, 2 sub-exp) | [TODO] |
| c10 | exp | Phase 2.E5 — Reg head on/off (classifier only soft-mean vs classifier + reg head, 2 sub-exp). ★ hybrid 가치 측정 | [TODO] |
| G2 | gate | 5 axis informational 완료 + 최소 1 axis +0.005 ΔOOF | [TODO] |
| c11 | code | `analysis/plan-012/phase3_aux.py` — wrapper for E6~E8. spec @ §8 | [TODO] |
| c12 | exp | Phase 3.E6 — Boundary sample weighting on/off (2 sub-exp) | [TODO] |
| c13 | exp | Phase 3.E7 — Scorer arch (full Attn-GRU vs last-step MLP, 2 sub-exp) ★ 시계열 input 가치 | [TODO] |
| c14 | exp | Phase 3.E8 — r=0 logit prior 강도 (0 / +0.5 / +1.0, 3 sub-exp) | [TODO] |
| G3 | gate | aux ablation 완료 (informational) | [TODO] |
| c15 | code+exp | `analysis/plan-012/phase4_final.py` — best stack 5-fold + submission. spec @ §9 | [TODO] |
| G4 | gate | 5-fold OOF ≥ G1 winner + 0.005 + submission.csv 박제 | [TODO] |
| c16 | analysis | `analysis/plan-012/results.md` + `next_plan_candidates.md` (≥ 3 후보) + 3 파일 frontmatter sync + plan-012.1 instruction. spec @ §10 | [TODO] |
| G_final | gate | synthesis + plan-013 후보 + 3 파일 sync + plan-012.1 instruction | [TODO] |

### Plan-specific severe (WORKFLOW.md §12.3 default 위 추가분)

- `preflight_artifact_missing` — G0 의 `preflight.json` 미생성 또는 plan-006 reproduce drift > 0.005 또는 K-Means fit 실패 (cluster count < 7 또는 silhouette < 0). severity=**severe**.
- `phase2_no_positive_lever` — G2 의 5 axis 모두 negative. severity=**severe** + autonomous path-pivot 옵션.
- `final_no_additive` — G4 best stack 5-fold OOF < G1 winner + 0.005. severity=**warn** + fallback submission.
- `codebook_geometry_drift` — E0a/E0b/E0c anchor 좌표 invariant 위반:
  - E0a: 7 anchors, ‖anchor 0‖ < 1e-6, ‖anchor 1~6‖ = 0.005m ± 2%, anchor 1~6 의 *world-frame basis* (= ±e_x, ±e_y, ±e_z) 와의 cosine ≥ 0.99
  - E0b: 동일 norm 조건 + *Frenet basis* (= ±t̂, ±n̂, ±b̂) 와의 cosine ≥ 0.99
  - E0c: 7 anchors, ‖anchor 0‖ < 1e-6, anchor 1~6 = K-Means center on residual_frenet (cluster size > 100 per cluster)
- `dilution_collapse` — E3 의 τ ≥ 0.3 sub-exp 에서 `mean(‖softmax-weighted center − origin‖₂) < 0.001m` (= 1mm). warn (sub-exp 무효, axis 진행).
- `frozen_gru_drift` — E7 sub-exp B (frozen GRU) 에서 plan-004 GRU encoder weight state_dict diff > 0. severe.
- `kmeans_fold_leakage` — E0c K-Means 가 fold-aware fit 안 됨 (= 학습 전체 data 로 fit 했는데 OOF 측정 시 val fold residual 이 cluster center 산출에 포함됨). severe — fold-별 refit 필수.

### Plan-specific paths (WORKFLOW.md §12.5/§12.6 default 위 추가/제외)

- whitelist 추가:
  - `src/pb_0_6822/ring_classifier.py` (신규 모듈)
  - `analysis/plan-012/**`
  - `runs/baseline/H019_phase0-preflight-codebook/**` ~ `runs/baseline/H029_phase4-final-5fold/**`
- whitelist 제외 (blacklist):
  - `src/pb_0_6822/boundary.py` (corrector path 전체 폐기)
  - `src/pb_0_6822/selector.py` (frozen reuse only)
  - `src/pb_0_6822/corrector_redesign{,_v2}.py` (plan-010/011 산출)
  - `src/pb_0_6822/candidates_extended.py` (plan-008 산출)
- 참조 (read-only):
  - `runs/baseline/F001_variant-e/**` (plan-006 산출, single formula baseline)
  - `runs/baseline/P001_pb-0-6822-fullrun/**` (plan-004 산출, GRU pretrained)
  - `analysis/plan-005/corrector_decomp.{md,json}` (★ destructive band + direction breakdown — binormal 약점 evidence)
  - `analysis/plan-007/per_candidate_hit.{md,json}` (raw single formula ranking)
  - `analysis/plan-011/results.md` (In/ID +0.0050 anchor)
  - `notes/PB_0.6822 코드공유.ipynb` cell 4 (`CandidateAttentionGRUSelector` 원본)

### Decision-note 사용 예 (자율 결정 시 commit msg 박제)

- `decision-note: spec-default — Phase 1 bake-off fold=0 (~5min × 3 sub-exp = 15min), Phase 4 5-fold 강제`
- `decision-note: spec-default — codebook anchor radius = 0.005m (= 0.5cm, regression head scale prior 와 동일)`
- `decision-note: spec-default — K-Means K=7 (E0c), n_init=10, random_state=20260606. fold-aware refit (= 매 fold 의 학습 partition residual 으로 cluster fit, val 은 학습된 cluster 에 assign only)`
- `decision-note: spec-default — classifier loss = CE on argmin-by-distance hard label, regression loss = L7 hinge on (classifier-selected anchor + reg offset). 두 loss 등가중치 결합 (1.0 + 1.0)`
- `decision-note: spec-default — winner_id 박제 후 Phase 2/3 의 모든 sub-exp = winner anchor 그대로 reuse (= G1 winner_id 의 anchor 좌표가 §3.4 anchor 의 source of truth)`
- `decision-note: tie-break — G1 winner gap < 0.005 시 단순성 우선 (E0a > E0b > E0c)`
- `decision-note: conditional-skip — winner = E0a Absolute 시 G2.E1 Frame swap skip (frame_axis_n/a, frame 가 anchor 좌표에 영향 없음)`
- `decision-note: conditional-skip — G1 winner OOF < 0.6450 시 baseline_below_anchor warn + Phase 2 informational 진행`

---

## §1. 배경 / 이전 plan 인계 + v1 → v2 paradigm shift

### §1.1 plan-005~011 의 residual regression path 한계 (★ paradigm 폐기 evidence)

| plan | path | best OOF | best LB | evidence |
|---|---|---|---|---|
| plan-004 | 27-cand selector + boundary corrector | 0.6491 | 0.6822 (notebook) / 0.6692 (우리 data) | full-stack anchor |
| plan-005 | corrector_decomp 분해 | 0.6524 (5-fold soft) | — | ★ destructive band [0.5, 1cm) -7.83pp + binormal 0.0064 (= parallel 1/7) |
| plan-006 | single formula + plan-004 corrector | 0.6491 (corrected) | **0.6692** | single formula baseline |
| plan-007 | single formula 4-step 개선 | 0.6482 (Step 4) | carry-over 미회수 | regression path |
| plan-008 | candidate redefine + corrector redesign | — | marginal | 27→34 후보 확장 |
| plan-010 | corrector_redesign 4 후보 depth | 0.6320 (Z1 anchor) | — | 4 후보 모두 marginal/NEGATIVE |
| plan-011 | corrector_redesign_v2 4 axis breadth | 0.6450 (In/ID best) | — | ★ 1/4 axis positive (In/ID +0.0050) |

**→ 결론**: 4 plan (006/008/010/011) 의 corrector path 모두 plan-006 LB 0.6692 위로 못 올라옴. 24 sub-exp 중 단 1개만 +0.005 marginal. paradigm 자체의 한계 강함.

### §1.2 plan-005 의 두 evidence — paradigm shift 핵심 근거

#### §1.2.1 Destructive band [0.5, 1cm) -7.83pp (★ mean-regression trap evidence)

- 2594 sample 중 -203 hits lost (= 7.83pp).
- 해석: residual *회귀* 모델이 "보정 안 했으면 hit 였을 sample 을 *멀리 밀어냄*". 회귀 head 의 *평균 예측 경향* 때문 — confident sample 도 평균 쪽으로 끌려감.
- plan-011 C008/L2 의 gate-asymmetric loss 도 이 band 완전히 못 회복 (gate output collapse 위험).
- **plan-012 v2 의 해결**: classifier (argmax) 가 mode k 선택 → cluster k 내 *작은 scale* regression. mean-regression trap = 회귀 scale 이 *전체 residual 분포* 일 때 발생; cluster 내 회귀는 scale ~0.5cm 로 trivial.

#### §1.2.2 Binormal direction 0.0064 (= parallel 의 1/7)

| direction | corrector delta norm 평균 (m) | plan-012 v1 cover |
|---|---|---|
| parallel (t-axis) | 0.0451 | ✓ (2D ring) |
| perpendicular (n-axis) | 0.0214 | ✓ (2D ring) |
| **binormal (b-axis)** | **0.0064** | **✗ (v1 2D 누락!)** |

★ **v1 의 z 축 누락 spec 오류** = data 가 3D `(x, y, z)` ([selector.py:172](src/pb_0_6822/selector.py#L172)) 인데 v1 ring 이 2D 만 cover → binormal 축 0.6cm scale 의 residual 을 *학습 자체에서 누락*. v2 = 3D + ±b 포함 으로 직접 fix.

### §1.3 사용자 제안의 직관 (★ paradigm-level 통찰)

사용자 conversation:
- "단일 공식 - 변수 3개인 선형 방정식으로만 해도 정답의 64% cover, 1.5cm hit 확장하면 84% cover"
- "MLP residual 직접 예측 매우 어려웠다" (= plan-010/011 evidence 와 일치)
- "0.5cm 정도만 이동시켜서 남은 20% 샘플을 공식 안으로 끌어당기는 것"
- "분류 + 회귀 hybrid: classifier 가 mode 선택, regression 이 0.1~0.5cm 미세 조정"

**세 통찰의 곱셈**:
1. 공식이 이미 64% hit → 학습 task = 36% 의 *방향성 commit*
2. 1.5cm 확장 시 84% → 0.5~1cm shift 로 *20% 회수 가능*
3. 직접 residual 회귀 실패 → *작은 scale* 의 회귀로 분할 (cluster 내)

→ codebook (anchor 좌표 집합) + classifier (mode 선택) + regression head (cluster 내 미세 offset) = **mean-regression trap 회피 + scale 문제 해결**.

### §1.4 plan-004 selector arch 의 재사용

| 컴포넌트 | 위치 | 변경 |
|---|---|---|
| `selector.make_seq_features(x, end_idx, direction=1.0)` | [selector.py:406-449](src/pb_0_6822/selector.py#L406-L449) | **1:1 reuse**, shape `(N, 6, 9)` |
| `selector.CandidateAttentionGRUSelector` encoder | [selector.py:697-720](src/pb_0_6822/selector.py#L697-L720) | encoder 1:1 reuse, head 만 swap (cand_count 27→7, cand_dim 32→**11** (3D feature)) |
| GRU pretrained weight | `runs/baseline/P001_pb-0-6822-fullrun/**` | E7 sub-exp B (frozen GRU) 에서 load + frozen forward |
| single formula F0 | `frenet_par120_perp_neg020` (CANDIDATES[17]) | plan-006 그대로 |

### §1.5 v1 → v2 변경 요약 (audit trail)

| 영역 | v1 (e1f08eb) | **v2 (본 commit c1.1)** | 사유 |
|---|---|---|---|
| 후보 차원 | 2D | **3D** | data `(x, y, z)` 확인, plan-005 binormal evidence |
| 후보 개수 | 9 (1+4+4 magnitude×angle) | **7** (1+6 orthogonal 또는 K-Means) | mean-regression trap 회피 = 격자보다 mode 분리 |
| codebook | single (Frenet 2D ring) | **3-way bake-off** (Absolute / Frenet-ortho / K-Means) | 사용자 결정: data 가 paradigm 선택 |
| inference | soft-mean only | **classifier + regression head** (hybrid) | mean-regression trap + scale 문제 동시 해결 |
| Phase 1 | E0 single baseline | **E0a/E0b/E0c bake-off** + winner 결정 | paradigm-level decision 을 측정으로 |
| Phase 2 axis | 4 (Frame/Density/Temp/Loss) | **5** (+ Reg head on/off) | hybrid 가치 직접 측정 |
| Phase 3 axis | 3 (BoundaryWeight/ScorerArch/r=0) | 3 (동일) | — |

---

## §2. Scope (명시적)

### §2.1 In-scope

| 항목 | 값 |
|---|---|
| codebook (Phase 1 bake-off) | **3-way**: E0a Absolute-7Way (world), E0b Frenet-Orthogonal-7Way, E0c K-Means-7Way (Frenet residual) |
| selector encoder | plan-004 `CandidateAttentionGRUSelector` encoder, full fine-tune (E7 sub-exp B 에서 frozen) |
| single formula F0 | `frenet_par120_perp_neg020` (CANDIDATES[17]) |
| inference arch | classifier (7 logit) + regression head (3D offset, scale prior 0.5cm) **hybrid** |
| target dim | 3 (= `train_y` (x, y, z)) |
| LB 제출 | **0 회** (할당량 carry-over) |
| Validation | Phase 1~3: 1-fold (fold=0, N_val≈2020). Phase 4: 5-fold concat. |
| GPU | server cuda:1 |
| Loss | L7 hit-aware smooth hinge (plan-011 §5.1 reuse, 3D 적용) + classification CE |

### §2.2 Out-of-scope (절대 안 함)

| 항목 | 이유 |
|---|---|
| corrector path (residual regression on raw 좌표) | plan-005~011 폐기. `boundary.py` / `corrector_redesign*` import X |
| 27 후보 physics candidate | plan-008 산출, scope X |
| 2D 후보 (= v1) | spec 오류 확인 후 폐기 |
| K > 9 codebook | over-parametrization 위험, Phase 2.E2 density swap 의 sweet-spot 측정 후 plan-013 carry-over |
| TTA / multi-parse inference | plan-013 후보 |
| iterative refinement | residual regression 의 iterative 변형 — paradigm 분리 |
| LB 제출 | 할당량 소진 carry-over |

---

## §3. 사전 등록 (Pre-registration)

### §3.1 Fold split

- 5-fold OOF: `selector.stable_fold_id(sample_id, folds=5)` (plan-004 동일)
- Phase 1~3 fold=0 only (N_val ≈ 2020, binomial std ≤ 0.005)
- Phase 4 5-fold concat 강제

### §3.2 평가 metric

- main: **5-fold concat OOF soft hit @ 1cm** (Phase 4) 또는 **1-fold OOF soft hit** (Phase 1~3)
- soft hit = `‖hybrid_pred_pos − true_pos‖₂ ≤ 0.01m`. hybrid_pred_pos 산출:
  - mode_k = argmax(classifier_logits, dim=-1)  # hard mode 선택
  - hybrid_pred_pos = F0_pred + anchor[mode_k] + reg_head(mode_k)  # cluster 내 offset 가산
  - τ=0 case (E3 argmax) 와 동일 식; soft-mean case (reg head off, E5 sub-exp A) 에서는 `softmax × anchor` 합 사용
- ΔOOF (axis attribution) = `OOF_with_lever − OOF_anchor` per sub-exp
- `directional_commit_magnitude` = `mean(‖hybrid_pred_pos − F0_pred‖₂)` (= origin 으로부터 평균 이탈 크기, dilution_collapse warn 의 척도)

### §3.3 Anchor 정의 (Phase 2/3 의 모든 ablation 의 기준점)

★ **Anchor combo = G1 winner** (Phase 1 후 확정):

- codebook = winner ∈ {E0a Absolute, E0b Frenet-Orthogonal, E0c K-Means}
- K = 7 (winner 의 default; E2 에서 K density swap)
- frame = winner-dependent (E0a → world; E0b/E0c → Frenet local @ F0 prediction point)
- Scorer encoder = full `CandidateAttentionGRUSelector` on `make_seq_features` (frozen plan-004 GRU pretrained 으로 init, full fine-tune)
- Scorer head = classifier (7 logit) + regression head (3D offset per mode, scale prior 0.5cm)
- Loss = L7 hit-aware hinge on hybrid_pred_pos + CE on hard-label argmin-by-distance
- Inference τ = 0.03 (classifier softmax 후 reg head 적용 시; argmax = E3 sub-exp)
- Boundary sample weight = uniform (E6 axis 에서 on swap)
- r=0 anchor (= mode 0 = center) logit prior = **+0.0** (E8 axis 에서 swap)

### §3.4 Plan-011 anchor 비교

| measure | plan-011 In/ID | plan-012 v2 G1 winner (예상) | gap |
|---|---|---|---|
| 1-fold OOF (fold=0) | 0.6450 | ≥ 0.6450 (G1 lock) | 0~? |
| 5-fold OOF (concat) | TBD | TBD (G4) | TBD |
| oracle ceiling per codebook | N/A | ~0.78~0.86 (G0 preflight, codebook-dependent) | N/A |

---

## §4. STAGE 0 (c2) — `src/pb_0_6822/ring_classifier.py` 신규 모듈 (v2)

### §4.1 모듈 책임 (7 컴포넌트, self-contained, 3D)

```python
# src/pb_0_6822/ring_classifier.py

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from src.pb_0_6822 import selector as base


# ── 컴포넌트 1: 3 codebook generator ──

def compute_anchors_absolute(radius_m: float = 0.005) -> np.ndarray:
    """E0a Absolute-7Way: world frame ±x, ±y, ±z + center.

    Returns: (7, 3) — anchors in world frame.
        [0] origin (0, 0, 0)
        [1] (+radius_m, 0, 0)    +x
        [2] (-radius_m, 0, 0)    -x
        [3] (0, +radius_m, 0)    +y
        [4] (0, -radius_m, 0)    -y
        [5] (0, 0, +radius_m)    +z
        [6] (0, 0, -radius_m)    -z

    Invariants (codebook_geometry_drift severe check):
        - shape == (7, 3)
        - ‖anchors[0]‖ < 1e-6
        - ‖anchors[1:7]‖ ≈ radius_m ± 2%
        - anchors[1:7] basis cosine ≥ 0.99 vs ±e_x/±e_y/±e_z (in fixed order)
    """


def compute_anchors_frenet_orthogonal(radius_m: float = 0.005) -> np.ndarray:
    """E0b Frenet-Orthogonal-7Way: Frenet local frame ±t̂, ±n̂, ±b̂ + center.

    Returns: (7, 3) — anchors in *Frenet local frame coords* (= caller 측에서 sample 별 basis
        (R_world_from_frenet: (B, 3, 3)) 로 회전하여 world 변환).
        [0] (0, 0, 0)            center
        [1] (+radius_m, 0, 0)    +t̂
        [2] (-radius_m, 0, 0)    -t̂
        [3] (0, +radius_m, 0)    +n̂
        [4] (0, -radius_m, 0)    -n̂
        [5] (0, 0, +radius_m)    +b̂
        [6] (0, 0, -radius_m)    -b̂

    NOTE: 좌표 *형식* 은 E0a 와 동일 (= 7 × 3 직교 set), basis 가 trajectory-aligned 인 점만 차이.
    """


def compute_anchors_kmeans(
    train_residuals_world: np.ndarray,    # (N_train, 3) = true_y - F0_pred (world frame)
    fold_id: np.ndarray,                  # (N_train,) — fold-aware fit 용
    K: int = 7,
    radius_clip_m: float = 0.020,        # K-Means cluster center ‖·‖ clip threshold (= 2cm)
    n_init: int = 10,
    random_state: int = 20260606,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """E0c K-Means-7Way: train residuals 의 Frenet 변환 위 K-Means K cluster + center origin.

    fold-aware fit (★ kmeans_fold_leakage severe 방지):
        - 각 fold k = 0..4 마다, `fold_id != k` partition 의 residuals 로 K-Means fit.
        - val partition (fold_id == k) sample 은 학습된 cluster center 에 *assign only* (fit X).
        - 본 fn 은 *5 set of cluster centers* (per-fold) 반환 — caller 가 fold-별 사용.

    Frenet 변환 (caller helper 사용):
        residuals_frenet[i] = R_world_from_frenet[i].T @ residuals_world[i]   # (3,)
        # R_world_from_frenet = build_frenet_basis_3d(train_x, end_idx) (컴포넌트 2)

    Procedure:
        for k in range(5):  # 5-fold
            train_mask = (fold_id != k)
            residuals_frenet_train = R_world_from_frenet[train_mask].transpose(0, 2, 1) @ residuals_world[train_mask, :, None]
            km = KMeans(n_clusters=K-1, n_init=n_init, random_state=random_state).fit(residuals_frenet_train)
            centers_per_fold[k, 0] = (0, 0, 0)          # explicit center anchor
            centers_per_fold[k, 1:K] = km.cluster_centers_
            # ‖center‖ > radius_clip_m 시 clip (outlier mosquito 의 cluster center exceeds 2cm = noise)
            centers_per_fold[k, 1:K] = clip_norm(centers_per_fold[k, 1:K], max_norm=radius_clip_m)

    Returns:
        centers_per_fold: (5, K, 3) — fold-별 cluster centers in Frenet frame
        cluster_sizes_per_fold: (5, K) int — cluster 별 sample 수 (G0 박제용; min 100 per cluster 권장)
        fit_meta: dict — inertia, silhouette (sklearn) per fold

    Invariants (codebook_geometry_drift / kmeans_fold_leakage severe):
        - centers_per_fold.shape == (5, K, 3)
        - all(‖centers_per_fold[:, 0, :]‖ < 1e-6)   # center origin
        - min(cluster_sizes_per_fold[:, 1:]) > 100  # 모든 cluster > 100 sample
    """


# ── 컴포넌트 2: 3D Frenet basis @ F0 prediction point ──

def build_frenet_basis_3d(
    trajectory_x: np.ndarray,   # (N, T, 3) world coords
    end_idx: int,
) -> np.ndarray:
    """Compute 3D Frenet basis (t̂, n̂, b̂) per sample.

    Procedure (plan-011 §5.1 build_frenet_basis):
        v = x[:, end_idx] - x[:, end_idx-1]                  # (N, 3) velocity
        a = v - (x[:, end_idx-1] - x[:, end_idx-2])           # (N, 3) acceleration
        t_hat = v / ‖v‖
        n_unnorm = a - (a · t_hat) * t_hat                   # (N, 3) perp component
        n_hat = n_unnorm / ‖n_unnorm‖
        b_hat = cross(t_hat, n_hat)                          # (N, 3) binormal

    Degenerate (‖v‖ < 1e-6 or ‖n_unnorm‖ < 1e-6) → fallback (identity world basis).
    fallback count 박제 (informational, severe X).

    Returns: R_world_from_frenet — shape (N, 3, 3), columns = (t_hat, n_hat, b_hat).
        Frenet-frame coord → world: world_vec = R_world_from_frenet @ frenet_vec
        World → Frenet: frenet_vec = R_world_from_frenet.transpose() @ world_vec
    """


# ── 컴포넌트 3: anchor → world frame candidate positions ──

def anchors_to_world(
    anchors_local: np.ndarray,    # (K, 3) in codebook-native frame (world for E0a; Frenet for E0b/E0c)
    R_world_from_frenet: np.ndarray | None,  # (N, 3, 3) — None for E0a (frame = world); set for E0b/E0c
    F0_pred_world: np.ndarray,    # (N, 3) — single formula prediction in world
) -> np.ndarray:
    """Returns: (N, K, 3) — anchor positions in world frame.

    E0a (Absolute): R_world_from_frenet=None → anchors_world = anchors_local broadcast.
        cand_pos[i, k] = F0_pred_world[i] + anchors_local[k]
    E0b/E0c (Frenet-based): R_world_from_frenet ≠ None.
        cand_pos[i, k] = F0_pred_world[i] + R_world_from_frenet[i] @ anchors_local[k]
    """


# ── 컴포넌트 4: candidate feature builder (cand_dim = 11) ──

def make_codebook_candidate_features(
    cand_pos_world: np.ndarray,           # (N, K, 3)
    anchors_local: np.ndarray,             # (K, 3)
    codebook_id: str,                      # "absolute" / "frenet_orthogonal" / "kmeans"
    R_world_from_frenet: np.ndarray | None,
    F0_pred_world: np.ndarray,             # (N, 3)
) -> np.ndarray:
    """Build candidate feature tensor for hybrid scorer head. Returns (N, K, 11) float32.

    Per-candidate features (11 dim):
        [0:3]  anchors_local (par, perp/n, b/z)         — codebook-native coords
        [3]    radius_local  = ‖anchors_local[k]‖        — meters
        [4]    is_origin     = 1 if radius_local < 1e-6 else 0
        [5:8]  anchor offset 정규화 (anchors_local / 0.005)  # unit-scaled
        [8]    codebook_id_absolute  = 1 if codebook == absolute else 0
        [9]    codebook_id_frenet_ortho = 1 if codebook == frenet_orthogonal else 0
        [10]   codebook_id_kmeans       = 1 if codebook == kmeans else 0
    """


# ── 컴포넌트 5: hybrid scorer head (classifier + regression head) ──

class HybridScorerHead(nn.Module):
    """plan-004 CandidateAttentionGRUSelector + classifier (K logit) + regression head (3D offset).

    encoder + cand_attn = base.CandidateAttentionGRUSelector reuse (cand_count=K, cand_dim=11).
    head = 2 path:
        classifier_head: (B, hidden) → (B, K) logits
        regression_head: (B, hidden, K) → (B, K, 3) per-mode offset (cluster 내 미세조정, scale prior ~0.5cm)
    """

    def __init__(
        self,
        K: int = 7,
        hidden: int = 64,
        cand_dim: int = 11,
        reg_head_scale_m: float = 0.005,
        encoder_pretrained_path: str | None = None,
    ):
        super().__init__()
        self.K = K
        self.reg_head_scale_m = reg_head_scale_m
        # encoder
        self.scorer = base.CandidateAttentionGRUSelector(
            seq_dim=9, cand_dim=cand_dim, hidden=hidden, cand_count=K,
        )
        # classifier head = scorer 의 기존 head 그대로 (logit per candidate)
        # regression head = 새 mlp (cand_dim → 3 offset per mode)
        self.reg_head = nn.Sequential(
            nn.Linear(cand_dim, hidden), nn.GELU(),
            nn.Linear(hidden, 3),
        )
        if encoder_pretrained_path is not None:
            self._load_encoder_weights(encoder_pretrained_path)

    def forward(
        self, seq: torch.Tensor, cand_feat: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        seq: (B, 6, 9)
        cand_feat: (B, K, 11)
        returns:
            logits: (B, K)
            reg_offset: (B, K, 3) — per-mode regression offset (scaled by reg_head_scale_m via tanh)
        """
        logits = self.scorer(seq, cand_feat)                         # (B, K)
        # reg_head applied per-candidate: tanh-bounded offset
        reg_raw = self.reg_head(cand_feat)                            # (B, K, 3)
        reg_offset = torch.tanh(reg_raw) * self.reg_head_scale_m * 2  # bounded to ±1cm
        return logits, reg_offset

    def freeze_encoder(self):
        """E7 sub-exp B (frozen GRU) 용. encoder GRU + cand_attn 만 freeze, head 는 trainable."""


# ── 컴포넌트 6: L7 hit-aware hinge loss (3D 확장) ──

def hit_aware_hinge(
    corrected_pos: torch.Tensor,  # (B, 3) hybrid_pred_pos
    target: torch.Tensor,         # (B, 3) true_y
    R_HIT: float = 0.01,
    smooth: float = 0.005,
) -> torch.Tensor:
    """plan-011 §5.1 hit_aware_hinge, 3D 적용. (B,) per-sample."""
    excess = torch.norm(corrected_pos - target, dim=1) - R_HIT
    linear_hinge = F.softplus(excess / smooth) * smooth
    return linear_hinge ** 2


def huber_loss_3d(pred, target, beta=0.005):
    """3D huber. (B, 3) → (B,)."""
    return F.smooth_l1_loss(pred, target, beta=beta, reduction='none').sum(dim=1)


def classifier_ce_loss(
    logits: torch.Tensor,         # (B, K)
    F0_pred_world: torch.Tensor,  # (B, 3)
    anchors_world: torch.Tensor,  # (B, K, 3)
    target: torch.Tensor,         # (B, 3)
) -> torch.Tensor:
    """Hard-label CE: target_mode = argmin(‖anchor[k] + F0_pred − target‖₂).
    classifier 가 이 hard label 을 학습.
    """
    cand_pos = F0_pred_world[:, None, :] + anchors_world           # (B, K, 3)
    distances = torch.norm(cand_pos - target[:, None, :], dim=-1)  # (B, K)
    hard_label = distances.argmin(dim=-1)                          # (B,)
    return F.cross_entropy(logits, hard_label)


def hybrid_combined_loss(
    logits: torch.Tensor,         # (B, K)
    reg_offset: torch.Tensor,     # (B, K, 3)
    anchors_world: torch.Tensor,  # (B, K, 3)
    F0_pred_world: torch.Tensor,  # (B, 3)
    target: torch.Tensor,         # (B, 3)
    temperature: float = 0.03,
    R_HIT: float = 0.01,
    smooth: float = 0.005,
    lambda_ce: float = 1.0,
    lambda_hinge: float = 1.0,
    use_reg_head: bool = True,
) -> torch.Tensor:
    """E0 baseline loss = lambda_ce * classifier_CE + lambda_hinge * (huber + hinge on hybrid_pred_pos).

    use_reg_head=False (E5 sub-exp A): hybrid_pred_pos = soft-mean(F0 + anchors) ignoring reg_offset.
    use_reg_head=True  (E5 sub-exp B = anchor): hybrid_pred_pos = F0 + (selected_anchor + selected_reg_offset).
        Selection method:
            τ → 0 (argmax): hard mode_k, hybrid_pos = F0 + anchor[mode_k] + reg_offset[mode_k]
            τ > 0:         soft prob, hybrid_pos = F0 + Σ_k prob[k] × (anchor[k] + reg_offset[k])
    """
    ce_loss = classifier_ce_loss(logits, F0_pred_world, anchors_world, target)

    if temperature <= 1e-8:
        # argmax (E3 sub-exp)
        mode_k = logits.argmax(dim=-1)                                          # (B,)
        selected_anchor = anchors_world.gather(1, mode_k[:, None, None].expand(-1, -1, 3)).squeeze(1)
        selected_reg = reg_offset.gather(1, mode_k[:, None, None].expand(-1, -1, 3)).squeeze(1) if use_reg_head else 0
        hybrid_pos = F0_pred_world + selected_anchor + selected_reg              # (B, 3)
    else:
        prob = F.softmax(logits / temperature, dim=-1)                          # (B, K)
        anchor_blend = (prob[:, :, None] * anchors_world).sum(dim=1)             # (B, 3)
        reg_blend = (prob[:, :, None] * reg_offset).sum(dim=1) if use_reg_head else 0
        hybrid_pos = F0_pred_world + anchor_blend + reg_blend                    # (B, 3)

    huber = huber_loss_3d(hybrid_pos, target)
    hinge = hit_aware_hinge(hybrid_pos, target, R_HIT=R_HIT, smooth=smooth)
    pos_loss = (0.5 * huber + 0.5 * hinge).mean()

    return lambda_ce * ce_loss + lambda_hinge * pos_loss


# ── 컴포넌트 7: hybrid inference ──

def hybrid_predict(
    logits: torch.Tensor,         # (B, K)
    reg_offset: torch.Tensor,     # (B, K, 3)
    anchors_world: torch.Tensor,  # (B, K, 3)
    F0_pred_world: torch.Tensor,  # (B, 3)
    temperature: float = 0.03,
    use_reg_head: bool = True,
    r0_logit_prior: float = 0.0,
) -> torch.Tensor:
    """Inference: returns (B, 3) predicted world position.
    r0_logit_prior applied at mode 0 (= center anchor).
    """
    prior = torch.zeros(logits.shape[-1], device=logits.device)
    prior[0] = r0_logit_prior
    logits_prior = logits + prior

    if temperature <= 1e-8:
        mode_k = logits_prior.argmax(dim=-1)
        selected_anchor = anchors_world.gather(1, mode_k[:, None, None].expand(-1, -1, 3)).squeeze(1)
        selected_reg = reg_offset.gather(1, mode_k[:, None, None].expand(-1, -1, 3)).squeeze(1) if use_reg_head else 0
        return F0_pred_world + selected_anchor + selected_reg
    prob = F.softmax(logits_prior / temperature, dim=-1)
    anchor_blend = (prob[:, :, None] * anchors_world).sum(dim=1)
    reg_blend = (prob[:, :, None] * reg_offset).sum(dim=1) if use_reg_head else 0
    return F0_pred_world + anchor_blend + reg_blend
```

### §4.2 smoke test (c2 직후 self-check)

```python
# tests/test_ring_classifier_smoke.py
def test_codebook_geometry_absolute():
    a = compute_anchors_absolute()
    assert a.shape == (7, 3)
    assert np.linalg.norm(a[0]) < 1e-6
    assert all(0.0049 <= np.linalg.norm(a[i]) <= 0.0051 for i in range(1, 7))
    # ±x, ±y, ±z basis
    expected = np.array([[0,0,0],[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]]) * 0.005
    assert np.allclose(a, expected, atol=1e-6)

def test_codebook_geometry_frenet():
    a = compute_anchors_frenet_orthogonal()
    assert a.shape == (7, 3)
    # 동일 invariants as absolute, frame 만 다름 (값은 동일)
    assert np.linalg.norm(a[0]) < 1e-6
    assert all(0.0049 <= np.linalg.norm(a[i]) <= 0.0051 for i in range(1, 7))

def test_hybrid_forward_shape():
    head = HybridScorerHead(K=7, hidden=64, cand_dim=11)
    seq = torch.randn(4, 6, 9)
    cand_feat = torch.randn(4, 7, 11)
    logits, reg_offset = head(seq, cand_feat)
    assert logits.shape == (4, 7)
    assert reg_offset.shape == (4, 7, 3)
    assert reg_offset.abs().max() < 0.011  # tanh-bounded to ±1cm

def test_kmeans_fold_aware():
    # synthetic 1000 sample 5-fold, K=7
    residuals = np.random.randn(1000, 3) * 0.005
    fold_id = np.arange(1000) % 5
    centers, sizes, meta = compute_anchors_kmeans(residuals, fold_id, K=7)
    assert centers.shape == (5, 7, 3)
    assert all(np.linalg.norm(centers[k, 0]) < 1e-6 for k in range(5))
    assert sizes[:, 1:].min() > 50  # synthetic small data, relaxed
```

---

## §5. STAGE 0 (c3, G0) — Phase 0 preflight + codebook prep

### §5.1 산출물

- `analysis/plan-012/preflight.py` — 4 task 일괄 실행
- `analysis/plan-012/preflight.json` — schema:

```json
{
  "exp_id": "H019_phase0-preflight-codebook",
  "f0_raw_hit_measure": {
    "description": "F0 단일공식 의 raw hit@1cm + hit@1.5cm — 학습 무관 측정",
    "single_formula": "frenet_par120_perp_neg020",
    "candidate_idx": 17,
    "n_train": 10000,
    "hit_at_1cm": {"hit_rate": <float>, "expected_range": [0.60, 0.68]},
    "hit_at_1_5cm": {"hit_rate": <float>, "expected_range": [0.80, 0.88]}
  },
  "codebook_oracle_ceilings": {
    "description": "각 codebook 의 oracle scorer (label-aware argmin) 시 hit@1cm. 3 codebook 동시 박제.",
    "n_train": 10000,
    "absolute_7way": {"oracle_hit_1cm": <float>, "anchors": [[...]]},
    "frenet_orthogonal_7way": {"oracle_hit_1cm": <float>, "anchors": [[...]]},
    "kmeans_7way": {"oracle_hit_1cm": <float>, "anchors_per_fold": [[[...]], ...]}
  },
  "kmeans_fit_meta": {
    "K": 7,
    "fold_count": 5,
    "centers_per_fold": [[[...]]],
    "cluster_sizes_per_fold": [[...]],
    "inertia_per_fold": [<float>],
    "silhouette_per_fold": [<float>],
    "min_cluster_size": <int>,
    "min_cluster_size_threshold": 100
  },
  "plan_006_reproduce": {
    "single_formula": "frenet_par120_perp_neg020",
    "oof_argmax_hit_corrected_measured": <float>,
    "oof_argmax_hit_corrected_expected": 0.6491,
    "drift": <float>,
    "drift_threshold": 0.005,
    "reproduce_ok": <bool>
  }
}
```

### §5.2 실행

```bash
python -m analysis.plan-012.preflight \
  --root data \
  --plan-006-checkpoint runs/baseline/F001_variant-e/checkpoint_best.pt \
  --out                 analysis/plan-012/preflight.json
```

### §5.3 G0 합격

- `f0_raw_hit_measure.hit_at_1cm` ∈ [0.60, 0.68]
- `f0_raw_hit_measure.hit_at_1_5cm` ∈ [0.80, 0.88]
- 3 oracle ceilings 모두 ∈ [0.70, 0.90] (= "각 codebook 의 7 anchor 위에서 oracle 이 hit@1cm ≥ raw hit@1cm")
- `kmeans_fit_meta.min_cluster_size > 100`
- `plan_006_reproduce.drift ≤ 0.005`

위반 시 `preflight_artifact_missing` severe.

---

## §6. STAGE 1 (c4, G1) — Phase 1 3-Way Codebook Bake-off (★ paradigm 결정)

### §6.1 sub-exp matrix (3 sub-exp)

| sub-exp | codebook | frame | anchor source | K |
|---|---|---|---|---|
| **P1.E0a** | Absolute-7Way | world | `compute_anchors_absolute()` | 7 |
| **P1.E0b** | Frenet-Orthogonal-7Way | Frenet local @ F0 prediction | `compute_anchors_frenet_orthogonal()` | 7 |
| **P1.E0c** | K-Means-7Way | Frenet local @ F0 prediction | `compute_anchors_kmeans(...)` (per-fold) | 7 |

3 sub-exp **공통 (paradigm-clean comparison)**:
- arch = `HybridScorerHead(K=7, hidden=64, cand_dim=11)` with GRU pretrained init
- loss = `hybrid_combined_loss(..., lambda_ce=1.0, lambda_hinge=1.0, use_reg_head=True)` (★ E0 baseline = hybrid on)
- temperature = 0.03 (inference)
- r0_logit_prior = +0.0
- boundary sample weight = uniform
- optimizer = AdamW(lr=3e-4, weight_decay=1e-4)
- epochs = 50 (plan-004 default), patience = 5
- batch = 256

### §6.2 winner 결정 + tie-break

```python
winners = {"E0a": oof_E0a, "E0b": oof_E0b, "E0c": oof_E0c}
winner_id = max(winners, key=winners.get)
winner_oof = winners[winner_id]
second_oof = sorted(winners.values(), reverse=True)[1]
gap = winner_oof - second_oof

if gap < 0.005:
    # tie-break: 단순성 우선 (해석 가능성 + reproduce 용이성)
    priority = ["E0a", "E0b", "E0c"]
    tied = [k for k, v in winners.items() if v >= winner_oof - 0.005]
    winner_id = next(k for k in priority if k in tied)
    # decision-note: tie-break — winner=<id> (gap=<float>), priority=Absolute>Frenet-ortho>K-Means
```

### §6.3 G1 합격

- 3 sub-exp 모두 fold-0 OOF 박제 완료
- `winner_oof ≥ 0.6450` (= plan-011 In/ID anchor)
- `directional_commit_magnitude ≥ 0.002 m` for winner (dilution_collapse sanity)

위반 시 `baseline_below_anchor` warn (halt X). Phase 2 informational 진행. results.md 에 `paradigm_below_plan_011_evidence` 박제.

### §6.4 winner 박제

- `analysis/plan-012/phase1_winner.json`:
```json
{
  "winner_id": "E0a" | "E0b" | "E0c",
  "winner_oof": <float>,
  "winner_anchor_source": "compute_anchors_absolute" | "compute_anchors_frenet_orthogonal" | "compute_anchors_kmeans",
  "winner_K": 7,
  "winner_frame": "world" | "frenet",
  "second_id": <str>,
  "gap": <float>,
  "tie_break_applied": <bool>,
  "all_sub_exp_oof": {"E0a": <float>, "E0b": <float>, "E0c": <float>},
  "directional_commit_magnitudes": {"E0a": <float>, "E0b": <float>, "E0c": <float>}
}
```

★ §3.4 anchor combo = winner_id 의 config 으로 갱신. Phase 2/3 의 모든 sub-exp 가 이 source 만 reuse.

---

## §7. STAGE 2 (c5~c10, G2) — Phase 2 Core Ablation on Winner (5 axis)

### §7.1 anchor 위 1-axis swap matrix

각 sub-exp = G1 winner config 에서 *지정 axis 1개만 변경*. fold=0.

#### E1 — Frame swap (c6, 조건부)

| winner | sub-exp | 변경 |
|---|---|---|
| winner = E0a Absolute | (skip) | frame_axis_n/a — Absolute 는 world fixed, swap 무의미 |
| winner = E0b Frenet-Orthogonal | P2.E1a / P2.E1b | Frenet (anchor) / world (= ±t̂ → ±e_x 강제 회전) |
| winner = E0c K-Means | P2.E1a / P2.E1b | Frenet (anchor) / world (= K-Means cluster 좌표를 world 로 회전) |

ΔOOF(E1) = OOF(E1b) − OOF(E1a). winner = E0a 시 skip + decision-note `conditional-skip — frame_axis_n/a`.

#### E2 — Codebook K density swap (c7, 4 sub-exp)

| sub-exp | K | 직관 |
|---|---|---|
| P2.E2a | 5 | sparse (winner type 의 5 anchor variant) |
| **P2.E2b** | 7 (anchor) | winner |
| P2.E2c | 9 | slight dense |
| P2.E2d | 13 | dense — over-parametrization risk |

K-Means 의 경우 K=5/9/13 으로 K-Means 재fit (per-fold, G0 preflight 의 K=7 fit 과는 별도 산출). Absolute/Frenet-ortho 의 경우 K 변경 시 anchor 정의도 변경:
- K=5: center + ±dominant 1 axis (= winner 의 가장 informative axis, G0 oracle ceiling 기준) + ±second
- K=9: 7 + 2 (= dominant diagonal 2개, 예: +t+n / -t-n)
- K=13: 7 + 6 (= 모든 diagonal)

자율 결정 박제: `decision-note: spec-default — K density 의 추가 anchor 는 G0 oracle ceiling 의 marginal gain 기준 ordering`

ΔOOF(E2) = `max(OOF over K∈{5,9,13}) − OOF(K=7)`.

#### E3 — Temperature scan (c8, 6 sub-exp)

| τ | sub-exp |
|---|---|
| 0.0 (argmax) | P2.E3a |
| 0.01 | P2.E3b |
| **0.03 (anchor)** | P2.E3c |
| 0.1 | P2.E3d |
| 0.3 | P2.E3e |
| 1.0 | P2.E3f |

**dilution_collapse warn** (τ ≥ 0.3 sub-exp): `directional_commit_magnitude < 0.001m` 시 sub-exp 단독 무효 (axis 진행).

ΔOOF(E3) = `max(OOF over τ ≠ 0.03) − OOF(0.03)`.

#### E4 — Loss swap (c9, 2 sub-exp)

| sub-exp | loss |
|---|---|
| **P2.E4a** | hybrid (CE + L7 hinge) — anchor |
| P2.E4b | CE + distance regression only (hinge 제거, huber on hybrid_pos만) |

ΔOOF(E4) = OOF(E4b) − OOF(E4a). negative → hinge 의 metric alignment 결정적.

#### E5 — Reg head on/off (c10, 2 sub-exp, ★ hybrid 가치 측정)

| sub-exp | reg head | 직관 |
|---|---|---|
| P2.E5a | off (= classifier only, soft-mean blending) | "분류만으로 충분한가, regression head 가 진짜 필요한가" |
| **P2.E5b** | on (anchor) | E0 baseline hybrid |

ΔOOF(E5) = OOF(E5a) − OOF(E5b). **negative** (= E5b 가 더 좋음) 이면 사용자 제안 hybrid 의 가치 직접 입증. **positive** (= E5a 더 좋음) 이면 reg head 가 overfit/dilution 야기 → 단순화.

### §7.2 G2 합격

- 5 axis 모두 informational 완료 (E1 conditional skip 포함)
- **최소 1 axis** 에서 `max(ΔOOF) ≥ 0.005`

위반 시 `phase2_no_positive_lever` severe + autonomous path-pivot.

---

## §8. STAGE 3 (c11~c14, G3) — Phase 3 Aux Ablation (3 axis)

### §8.1 sub-exp matrix

#### E6 — Boundary sample weighting (c12, 2 sub-exp)

| sub-exp | weight | 직관 |
|---|---|---|
| **P3.E6a** | uniform (anchor) | E0 |
| P3.E6b | `1cm 근처 sample ×3` (= `0.005 < ‖F0_pred − true‖ < 0.015` 인 sample weight=3) | 20% 회수 대상 sample 집중 |

#### E7 — Scorer arch (c13, 2 sub-exp, ★ 시계열 input 가치)

| sub-exp | scorer | 직관 |
|---|---|---|
| **P3.E7a** | full `CandidateAttentionGRUSelector` (anchor) | E0 — 6-step kinematic context |
| P3.E7b | last-step MLP (= `make_seq_features[:, -1, :]` 만, GRU 우회) | 시계열 input 의 부가가치 측정 |

#### E8 — r=0 anchor logit prior (c14, 3 sub-exp)

| sub-exp | prior on mode 0 (= center) | 직관 |
|---|---|---|
| **P3.E8a** | +0.0 (anchor) | no prior |
| P3.E8b | +0.5 | mild — F0 prediction 보호 강화 |
| P3.E8c | +1.0 | strong — 64% 안전 sample 보호 / 36% 회수 손실 risk |

### §8.2 G3 합격

- 3 axis informational 완료. positive lever 권장.

---

## §9. STAGE 4 (c15, G4) — Phase 4 Best Stack 5-fold + Submission

### §9.1 best stack 선정

- Phase 2 best axis = `argmax(ΔOOF over E1, E2, E3, E4, E5)`
- Phase 3 best axis = `argmax(ΔOOF over E6, E7, E8)` (ΔOOF < 0 인 axis 는 anchor 유지)
- best stack = G1 winner + Phase 2 best lever + Phase 3 best lever (additive 가정)

### §9.2 5-fold + submission

```python
# analysis/plan-012/phase4_final.py
for fold in range(5):
    model = HybridScorerHead(K=K_best, hidden=64, cand_dim=11,
                              encoder_pretrained_path="runs/baseline/P001_pb-0-6822-fullrun/checkpoint_best.pt")
    # apply best stack levers
    train_subset_x = train_x[fold_id != fold]
    val_subset_x   = train_x[fold_id == fold]
    train(model, train_subset_x, ...)
    oof_preds[fold_id == fold] = predict(model, val_subset_x)

oof_soft_hit_5fold = compute_hit(oof_preds, train_y, R_HIT=0.01)

# test inference (5-fold ensemble: mean of hybrid predictions)
test_preds_ensemble = mean over folds of hybrid_predict(model_fold, test_x)
write_submission_csv(test_preds_ensemble, sample_ids_test, "submission.csv")
```

### §9.3 G4 합격

- `5-fold concat OOF ≥ G1 winner_oof + 0.005`
- `submission.csv` shape == `data/sample_submission.csv` shape, 좌표 finite

위반 시 `final_no_additive` warn → fallback = G1 winner 단독 5-fold submission.

---

## §10. STAGE 5 (c16, G_final) — Synthesis + plan-013 후보

### §10.1 산출물

- `analysis/plan-012/results.md` — 모든 G-gate 결과 요약 (★ winner_id + Phase 2 best axis + Phase 3 best lever + 5-fold OOF + oracle gap)
- `analysis/plan-012/next_plan_candidates.md` — plan-013 후보 ≥ 3개
- 3 파일 frontmatter sync:
  - `plans/plan-012-frenet-ring-classification.md` (`status: G_final_complete`)
  - `plans/plan-012-frenet-ring-classification.results.md`
  - registry (있을 경우)
- best Phase submission 박제: `runs/baseline/H029_phase4-final-5fold/submission.csv`

### §10.2 plan-013 후보 (조건부 framework)

| 조건 | plan-013 후보 |
|---|---|
| G4 best stack OOF ≥ 0.70 | (1) winner codebook 위 finer K-density tuning (CMA-ES on anchor coords) (2) regression head capacity 강화 (per-mode MLP depth) (3) per-sample F0 selection (F0 외 다른 single formula sample-wise) |
| 0.65 ≤ OOF < 0.70 | (1) MoE over multiple formulas (F0 + F1 + ...) (2) TTA rotation 4 + ensemble (3) classifier Transformer 교체 |
| OOF < 0.65 | (1) paradigm 재폐기 → KNN / GP / Diffusion (plan-011 §2.2 carry-over) (2) F0 자체 교체 (G0 oracle 박제 기반 best codebook re-selection) (3) corrector + hybrid 합체 (2-stage selector → hybrid) |

★ **plan-012.1 carry-over instruction**: best Phase submission `.csv` 의 LB 수동 제출 (`dacon-submit` skill) + frontmatter `lb_score` 후속 박제 + plan-013 분기 결정.

---

## §11. 참조

- `WORKFLOW.md` §1~§12 + `CLAUDE.md` Autonomous Execution Policy
- `plans/plan-004-pb-0-6822-fullrun.md` (selector arch + make_seq_features anchor)
- `plans/plan-006-minimal-variant-e-lb.md` (single formula F0 baseline)
- `plans/plan-011-single-formula-corrector-exploration.md` (★ residual regression path 결과 + L7 hinge 정의 + In/ID +0.0050)
- `analysis/plan-005/corrector_decomp.{md,json}` (★ destructive band + binormal evidence)
- `analysis/plan-007/per_candidate_hit.{md,json}` (raw single formula ranking)
- `notes/PB_0.6822 코드공유.ipynb` cell 4 (CandidateAttentionGRUSelector 원본)

---

## §12. Plan 자기-완결 + v2 변경 audit

### §12.1 핵심 정의 (외부 채팅·메모리 비의존)

- F0: plan-006 §5.5 CANDIDATES[17] (`frenet_par120_perp_neg020`)
- 3 codebook: §4.1 컴포넌트 1 (`compute_anchors_{absolute, frenet_orthogonal, kmeans}`)
- Frenet basis 3D: §4.1 컴포넌트 2 (`build_frenet_basis_3d`)
- L7 loss: §4.1 컴포넌트 6 (`hit_aware_hinge` + `hybrid_combined_loss`)
- Scorer encoder: [selector.py:697-720](src/pb_0_6822/selector.py#L697-L720) (`CandidateAttentionGRUSelector`)
- make_seq_features: [selector.py:406-449](src/pb_0_6822/selector.py#L406-L449)

### §12.2 v1 → v2 변경 audit (user-directed paradigm shift)

| § | v1 spec | v2 spec | 사유 |
|---|---|---|---|
| §0 (목적) | 2D 9-Way ring soft-mean | 3D 7-Way codebook bake-off + hybrid | data 3D 확인 + 사용자 제안 paradigm |
| §0.5 G1 | E0 single baseline | 3-way bake-off + winner 결정 | paradigm-level decision via measurement |
| §3.4 anchor | fixed | G1 winner 동적 결정 | 위 G1 변경의 결과 |
| §4.1 ring_classifier | 9 ring + soft-mean | 7 anchor codebook + classifier + reg head | hybrid paradigm |
| §5 G0 preflight | F0 raw + oracle 1개 | F0 raw + oracle 3개 + K-Means fit | 3 codebook prep |
| §6 G1 | 1 sub-exp lock | 3 sub-exp bake-off | bake-off |
| §7 G2 | 4 axis | 5 axis (+ E5 reg head on/off) | hybrid 가치 측정 |
| §8 G3 | 3 axis | 3 axis (동일) | — |

v1 의 모든 spec 은 본 v2 가 통째 대체. 본 commit 이후 v1 reference 시 git history `e1f08eb` 만 사용.
