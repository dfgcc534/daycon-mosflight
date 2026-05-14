---
plan_id: 014
version: 3.3 (spec patch — Dataset / IO 상세 박제. §2.1.C 새 sub-section: train/test/labels CSV path (`data/train|test/{sample_id}.csv` shape `(11, 3)` + `data/train_labels.csv` columns id/x/y/z) / timestep grid `[-400..0]` ms step 40 / T_TARGET_MS=80 / end_idx=10 / IO utility = `src/io.py` (plan-001, import OK).)
date: 2026-05-14 (Asia/Seoul)
status: spec
based_on:
  - 012
followed_by: []
scope: plan-012 (codebook bake-off + hybrid) 의 5-fold OOF plateau 의 root cause 는 *plan-004 selector + plan-006 F0 의 재사용 강박* — premise 채택 (검증 안 함). 그 premise 위에서 plan-012 의 5-Phase 실험 프로세스 (preflight → bake-off → axis ablation → aux ablation → final 5-fold) 를 **재사용 끊은 새 module 위에서 그대로 재실행**. baseline = 4 컴포넌트 from-scratch (BiGRU / learnable F0 / anchor 0.01m / Gaussian soft) fixed, 그 위에서 plan-012 의 8 ablation lever + 3 codebook bake-off 진행. baseline reproduce 없음 + plan-012 measured 값 reference 없음 (plan-012 result.md = INVALID_REFERENCE 박제 fd64f6c 후 §Target band absolute ≥0.66 / 0.65~0.66 / <0.65).
exp_ids: []
lb_score: null
---

# plan-014 v3 — plan-012 5-Phase 실험을 재사용 끊은 새 baseline 위에서 재실행

## §0. 한 줄 목적

> **F0 단일 선형 공식 (plan-006 `frenet_par120_perp_neg020`) 만으로 hit@1cm = 64%, hit@1.5cm = 84%.** 84% 의 sample 은 F0 근방 1.5cm 안 → 그 중 20% 만 1cm 밖. 본 plan 의 task essence = **20% sample 을 평균 0.5cm 적절한 방향으로 끌어당기기**. residual *vector regression* 직접 회귀 (plan-005~007) 는 어렵다고 입증됨 → residual *direction* 만 classification, magnitude = anchor scale prior + small offset. 이게 plan-012 의 codebook + classifier + regression hybrid paradigm 의 본질.
>
> 본 plan = **plan-012 의 5-Phase 실험 프로세스 (preflight / codebook bake-off / axis 5 ablation / aux 3 ablation / final 5-fold + best stack) 를 plan-004 코드 재사용 끊은 새 module 위에서 그대로 재실행**. plan-012 의 G-gate spec frame carry over (단 threshold 는 v2.3 absolute sync 결과 적용).
>
> **plan-004 참조 범위** (2가지만): (a) input feature 가공 방식 (시계열 9d × 6step 형식), (b) F0 sample cover 입증 (64%/84%, plan-006 hard evidence). 그 외 = 새 module `src/pb_0_6822/plan014_paradigm.py` 안에서 from-scratch.
>
> 본 plan 의 best stack 5-fold OOF 가 §Target band (≥0.66 / 0.65~0.66 / <0.65, absolute) 의 어느 위치에 들어가는지가 paradigm 의 진짜 잠재력 측정. plan-012 result.md = INVALID_REFERENCE 박제 (fd64f6c) → measured 값 비교 reference 없음.

---

## §0.5 Quick Reference

### 본 plan 의 task essence — "F0 64% cover + 남은 20% 끌어당김" (★ narrative anchor)

- **F0 단일 선형 공식** (plan-006 `frenet_par120_perp_neg020`): hit@1cm = **0.6320** (plan-006 F0 공식 자체의 hard evidence — plan-012 result.md INVALID 박제와 무관), hit@1.5cm = **0.8033**.
- **남은 20%** (= 84% − 64%): F0 근방 1.5cm 안이지만 1cm 밖. 평균 0.5cm 적절한 방향 이동 시 hit@1cm.
- **residual 직접 회귀 사망 진단** (plan-005~007): residual vector regression 어려움 → direction classification + small magnitude offset.
- **방향 후보 = Frenet local frame 7 방향** (±t / ±n / ±b / center): trajectory-aligned 방향 분리 = "어느 방향으로 0.5cm 이동" task 직관 일치.

### plan-004 참조 범위 — 2가지만

- **(a) input feature 가공 방식** — 시계열 9d × 6step 형식·전처리.
- **(b) F0 sample cover 입증** — 64%/84% cover (plan-006 hard evidence).

**위 2가지 외 = 전부 새로 build** — 새 module `src/pb_0_6822/plan014_paradigm.py` 안에서 from-scratch (시계열 모듈도 `nn.GRU` 등 표준 layer 직접 생성, plan-004 모듈 import 0).

### 본 plan 의 premise

- **Premise**: plan-012 의 plateau root cause = "plan-004 candidate-attention selector + plan-006 numpy F0 의 재사용 강박" → task essence 와 mismatch. 재사용 끊으면 paradigm 잠재력 0.66+ 까지 살아있음.
- 본 plan 은 premise 검증 안 함. paradigm 의 measured 잠재력만 박제.
- premise 가 *틀렸을 경우* 표식 = §Target negative band — "premise 오류" vs "paradigm 자체 한계" 분리 = plan-013 join interpretation 으로만 (§1.4).

### plan-012 가 "제대로" 가 아니었던 이유 (재사용 7 증상 → premise 근거)

plan-012 ring_classifier.py 는 paradigm shift 라고 self-label 했지만 실제 코드는 다음 *minimal patch*:

```
self.scorer  = base.CandidateAttentionGRUSelector(...)   # ← plan-004 의 27-way selector 그대로
self.reg_head = nn.Sequential(...)                       # ← 위에 작은 MLP 추가
F0 = f0_predict_frenet_par120_perp_neg020(...)           # ← plan-006 numpy 함수 그대로 (no grad)
anchor radius = 0.005m                                   # ← plan-004/006 era scale hardcode
```

7 observable failure mode → 단일 root cause:

| failure mode | 재사용 강박과의 인과 |
|---|---|
| F4 candidate-attention inductive-bias mismatch | 본 plan task = "20% sample 0.5cm 끌어당김" ≠ plan-004 task = "sample-별 27 후보 비교". `CandidateAttentionGRUSelector` = (a) GRU [task-neutral] + (b) candidate-attention head [plan-004 fit]. ring_classifier 의 classifier path 가 (b) 까지 같이 호출 → plan-012 의 fixed 7 anchor 와 mismatch. |
| F1 DCM collapse | F4 결과 — encoder 신호 부족 → classifier head 가 safe minimum (center mode) 수렴 |
| F7 frozen F0 path | plan-006 numpy F0 재사용 → gradient 없음, F0 학습 불가 |
| F3 F0 trivial dominance | F7 결과 — F0 hit 63% sample 학습 signal 무의미 |
| F2 anchor scale mismatch | plan-004/006 의 0.005m 답습, task fit 재검토 없음 |
| F5 hard label CE noise | F2 결과 — anchor 가 hit zone 내부 갇혀 argmin label = noise |
| F6 codebook geometry uniformity | F1 결과 — encoder center 만 고르니 anchor 위치 무관 |

→ 7 증상 = 1 root cause. 본 plan = 그 root cause 를 *제거한* baseline 위에서 plan-012 의 ablation 들이 살아 있는지 측정 (= falsify 아닌 *재실험*).

### 본 plan 의 multi-path 설계 — plan-012 5-Phase frame (v3 reframe, v2.x single-path 폐기)

```
G0 preflight  →  G1 module + smoke  →  G2 Phase 1 bake-off  →  G3 Phase 2 axis 5  →  G4 Phase 3 aux 3  →  G5 Phase 4 final 5-fold  →  G_final synthesis
```

- **Baseline = 4 컴포넌트 from-scratch fixed** (재사용 끊기 spirit 보존, ablation 대상 *아님*):
  - **C1 encoder**: 새 module-local 2-layer BiGRU (hidden=128) + cls head (7-logit) + reg head (7×3 offset). shared encoder.
  - **C2 F0**: learnable `(d1, par, perp)` (3 scalar), init = (1.98, 1.20, −0.20). 산식 = plan-006 source-of-truth (`ring_classifier.py:512-565`).
  - **C3 anchor radius**: 0.01m fixed.
  - **C4 soft label**: Gaussian σ=0.01m kernel.
  - 컴포넌트별 attribution = plan-015 후속 과제.
- **G2 Phase 1 codebook bake-off**: E0a Absolute / E0b Frenet / E0c K-Means 3-way 학습 → winner 1개 결정 (tie-break = 단순성 우선 E0a > E0b > E0c, plan-012 G1 rule carry).
- **G3 Phase 2 axis ablation 5** (winner codebook 위): E1 frame swap (conditional) / E2 K density (K=5/7/9/13) / E3 τ scan / E4 loss swap (L7 hinge vs distance reg) / E5 reg head on/off.
- **G4 Phase 3 aux ablation 3**: E6 boundary sample weighting / E7 scorer arch (BiGRU vs last-step MLP) / E8 r=0 logit prior (0/+0.5/+1.0).
- **G5 Phase 4 final 5-fold + best stack + submission**: winner config + best lever 들 stack 으로 5-fold concat OOF + submission 생성.

### Target (judgement criteria) — absolute (v2.3 sync 유지)

- **OOF ≥ 0.66** ★ positive (paradigm 부활). plan-015 = polish + LB. decision-note: 0.66 = competition-level paradigm target (plan-014 original aspiration, plan-012 INVALID 무관).
- **0.65 ≤ OOF < 0.66** partial 회복. plan-015 = corrector + 본 plan hybrid (plan-013 Candidate C 변형). decision-note: 0.65 = F0 raw 0.6320 + ~0.018 round absolute margin.
- **OOF < 0.65** negative. plan-015 = deep path pivot (`notes/new-ideas.md` KNN/GP/Diffusion).
- best stack 의 5-fold concat OOF 가 band 판정 대상.

### G-gates (정량 spec @ §3.4)

- **G0** preflight: F0 init reproduce ±0.005 / anchor 0.01m / soft entropy ≥0.5 nat / plan-012 disclaimer verify [TODO]
- **G1** module build: `plan014_paradigm.py` + smoke + 재사용 끊김 4가지 (selector import 0 / F0 grad≠0 / anchor 0.01m / soft entropy ≥0.5) [TODO]
- **G2** Phase 1 bake-off: winner_OOF ≥ 0.60 + DCM ≥ 0.002 (plan-012 G1 spec carry) [TODO]
- **G3** Phase 2 axis 5: 5 axis 중 1+ ΔOOF ≥ 0.005 (plan-012 G2 spec) [TODO]
- **G4** Phase 3 aux 3: informational [TODO]
- **G5** Phase 4 final: best_stack ≥ anchor_5fold + 0.005 (plan-012 G4 spec) + band 분류 [TODO]
- **G_final** synthesis: results.md 신규 + registry append + frontmatter sync + plan-015 후보 [TODO]

### Commit chain

| # | type | spec section | status |
|---|---|---|---|
| c1 ~ c2.3 | docs | v0~v2.3 narrative + spec drop + sync (git log authoritative) | [DONE] 4657ff7~b6bf927 (c1: 4657ff7+2a0f755 / c1.1: c7cf5c8 / c1.2: 5e98d6d / c1.3: c7fa9c8 / c1.4: 3a7a26c / c1.5: ab50cce / c2: ad051e2 / c2.1: 0a3c317 / c2.2: 90d9e0d / c2.3: b6bf927) |
| **c3** | docs | **v3 spec replacement — plan-012 5-Phase frame import.** v2.x single-path 폐기, 4 컴포넌트 baseline (fixed) + plan-012 5-Phase ablation frame 으로 재작성. frontmatter version 2.3→3 / title v1→v3 / §0.5 multi-path / §1.1 evolution 표 / §1.5 정직성 reframe / §2.1 baseline+ablation / §3.3 The Configs multi / §3.4 7-stage G-gate / §4~§10 stub | [DONE] 5f6750b |
| c3.1 | docs | **v3.1 spec patch — F0 산식 정정.** v3 박제 3 변수 `(α_par, α_perp_t, α_perp_n)` = plan-006 source-of-truth (`ring_classifier.py:512-565`) 와 mismatch (perp 잘못 t̂/n̂ 분리 + `d1·v_last` baseline 누락) → option A 채택 (사용자 confirm): 3 scalar `(d1, par, perp)` learnable, init (1.98, 1.20, −0.20), 산식 `F0 = p0 + d1·v_scale·v_last + par·acc_scale·acc_par_vec + perp·acc_scale·acc_perp_vec` (d2=0 / jerk=0 fixed, v_scale=acc_scale=1 with horizon=2 / time_scale=1). §0.5 C2 F0 bullet / §2.1.A C2 row / §3.2 F0_pred 산식 / §3.4 G0 (a) / frontmatter version 3→3.1 sync | [DONE] ba9e994 |
| c3.2 | docs | **v3.2 spec patch — Input feature 상세.** `make_seq_features` 정의 박제: shape `(N, 6, 9)`, 6 step indices `range(max(3, end_idx-5), end_idx+1)` (pad if <6), per-step 9 dim = `turn_model_features_from_context` 8 dim (speed / prev_speed-ratio / acc_norm-ratio / acc_par-ratio / perp_norm-ratio / jerk_norm-ratio / turn_cos / curvature) + direction 1 dim. source = `src/pb_0_6822/selector.py:280-294 + 406-449` (형식만 reuse, `selector.py` import X). §2.1.A Input pipeline row sync. frontmatter version 3.1→3.2 | [DONE] eed3c6e |
| c3.3 | docs | **v3.3 spec patch — Dataset / IO 상세.** §2.1 에 새 sub-section C Dataset/IO 추가: train/test CSV path (`data/train|test/{sample_id}.csv` shape `(11, 3)`), labels (`data/train_labels.csv` columns id/x/y/z), timestep grid (`[-400..0]` ms step 40), T_TARGET_MS=80, end_idx=10, IO utility = `src/io.py` (plan-001, import OK). frontmatter version 3.2→3.3 | [DONE] b1645b2 |
| c4 | code+exp | STAGE 0 (G0) — preflight artifact. spec @ §4 | [TODO] |
| c5 | code | STAGE 1 (G1) — `src/pb_0_6822/plan014_paradigm.py` 새 module + smoke + 재사용 끊김. spec @ §5 | [TODO] |
| c6 | code+exp | STAGE 2 (G2) — Phase 1 codebook bake-off (E0a/E0b/E0c 3 sub-exp → winner). spec @ §6 | [TODO] |
| c7 | exp | STAGE 3 (G3) — Phase 2 axis ablation 5 (E1~E5). spec @ §7 | [TODO] |
| c8 | exp | STAGE 4 (G4) — Phase 3 aux ablation 3 (E6~E8). spec @ §8 | [TODO] |
| c9 | exp | STAGE 5 (G5) — Phase 4 final 5-fold + best stack + submission. spec @ §9 | [TODO] |
| c10 | docs+sync | STAGE 6 (G_final) — results.md + registry + frontmatter sync + plan-015 후보. spec @ §10 | [TODO] |

---

## §1. 배경 / 동기 (narrative)

### §1.1 v0 → v1~v2.3 → v3 narrative evolution

| 축 | v0 (검증, 폐기) | v1~v2.3 (single-path, 폐기) | **v3 (현재, 본 plan)** |
|---|---|---|---|
| premise 위치 | hypothesis | assumed | **assumed (보존)** |
| 실험 path | A+B 2 path (head-to-head) | B 단독 1 path (4 컴포넌트 동시 swap) | **plan-012 5-Phase + plan-014 module-build (multi-config, baseline + 11 ablation)** |
| baseline | A reproduce (in-plan) | plan-012 measured 외부 ref | **4 컴포넌트 from-scratch fixed (보존), plan-012 reference X (INVALID 박제 fd64f6c)** |
| Target | B − A gap 기반 | OOF 절대값 (≥0.66 / 0.65~0.66 / <0.65) | **OOF 절대값 동일, best stack 5-fold OOF 기준** |
| ablation 정책 | 해당 없음 | lever 마진 줍기 회피 (4 컴포넌트 동시) | **plan-012 ablation 그대로 재실행 (재사용 끊은 환경에서 lever 살아 있는지 측정)** |
| 실험 의도 | falsify | 잠재력 baseline 측정 | **잠재력 + lever 마진 동시 측정** |

→ v3 = "재사용 끊은 baseline 위에서 plan-012 5-Phase 그대로 재실행". v2.x single-path 의 "lever 마진 줍기 회피" = **의식적 reverse** (사용자 명시 지시).

### §1.2 plan-012 의 사망 진단 — premise 의 근거

plan-012 results.md = "paradigm reframe 은 F0 raw hit 위 +0.002~0.003 만 추가 — paradigm 자체의 limit 확인". 그러나 plan-012 의 코드 = "plan-004 selector + plan-006 F0 위에 hybrid head 만 얹은 minimal patch". → plan-012 가 measured limit = **"minimal patch 의 limit"** 일 뿐, **"paradigm 의 limit"** 은 아직 측정 안 됨.

> **plan-012 의 7 failure mode 는 7개 독립 문제가 아니라 1개 root cause (재사용 강박) 의 7가지 증상.**

- 7-lever ablation (plan-012 의 G2/G3) = 증상 치료 ≠ 원인 치료 → 모든 lever 가 marginal 이었던 것은 합당.
- 본 plan = 원인 (재사용) 제거 후 paradigm 의 *제대로 된* baseline 측정 + 그 위에서 lever 들 재측정.

### §1.3 재사용 강박의 trap chain

```
"plan-004 selector reuse"  ─┐
                            ▶ candidate-attention inductive bias mismatch (F4). `CandidateAttentionGRUSelector` = (a) GRU [task-neutral] + (b) candidate-attention [plan-004 fit]. ring_classifier 의 classifier path 가 (b) 까지 호출 → plan-012 의 fixed 7 anchor (sample-invariant) 와 mismatch.
                            ▶ classifier head 가 신호 없이 학습 = safe minimum (mode 0 center) collapse (F1)
                            ▶ codebook geometry 가 결과에 무관 (F6)

"plan-006 numpy F0 reuse" ──┐
                            ▶ F0 gradient 없음 (F7)
                            ▶ F0 정확 prediction 이 hit 63% 에서 이미 답 = head 가 학습할 거리 없음
                            ▶ F0 hit 63% sample 학습 signal 무의미 (F3)
                            ▶ hard label = noise (F5)

"plan-004/006 era scale"  ──┐
                            ▶ anchor 0.005m 답습 → hit zone 내부 갇힘 (F2)
```

→ 7 mode 의 *원인이 1개* 라는 진단이 옳다면, 재사용 끊은 baseline 위에서 7 mode 가 동시에 풀리고 *추가로* plan-012 lever 들 (E0~E8) 의 마진도 살아 있어야 함. 본 plan 의 outcome 이 그 outcome-level 신호.

### §1.4 plan-013 과의 path 분기 — join interpretation

- **plan-013** (직전): paradigm 폐기 + plan-004 framework 회귀 → G2 0/3 axis FAIL, G1 0.6381 fallback submission
- **plan-014** (본 plan): paradigm 부활 시도 — 재사용 끊고 plan-012 5-Phase 재실행

| plan-013 LB | plan-014 best stack 5-fold OOF | 결합 해석 |
|---|---|---|
| ≥ 0.68 | < 0.65 (negative) | paradigm 폐기 정당화 — plan-004 framework path 가 정답 |
| < 0.68 | ≥ 0.66 (positive) | paradigm 부활 — from-scratch redesign 이 정답, premise 옳음 |
| ≥ 0.68 | ≥ 0.66 (positive) | 둘 다 작동 — plan-015 = 두 path 의 ensemble/stacking |
| < 0.68 | < 0.65 (negative) | 둘 다 실패 — 더 deep path-pivot (`notes/new-ideas.md` KNN/GP/Diffusion) |
| 임의 | 0.65 ≤ OOF < 0.66 (partial) | plan-013 corrector + 본 plan hybrid 합체 (plan-013 Candidate C 변형) — plan-015 default |

### §1.5 본 plan 의 정직성 원칙 (v3 reframe)

- **재실험 frame 명시**: premise (재사용=원인) 검증 안 함, 옳다는 가정 아래 paradigm 잠재력 측정.
- **negative band 해석의 한계**: §Target negative band 단독 해석 불가 — plan-013 join 필수 (§1.4).
- **컴포넌트별 attribution 회피**: 4 컴포넌트 (C1~C4) 동시 fixed baseline — 본 plan 의 outcome 으로는 어느 컴포넌트가 결정적이었는지 알 수 없음. attribution = plan-015 후속.
- **plan-012 ablation 재실행 (v3 reframe)**: plan-012 의 8 ablation lever + 3 codebook bake-off 를 *재사용 끊은 baseline 위에서* 다시 측정. 재사용 환경 위 marginal 이었던 lever 들이 재사용 끊은 환경에서 살아 있는지 측정. (v2.x 의 "lever 마진 줍기 회피" 폐기 — 사용자 명시 reverse.)
- **외부 reference 정책**: plan-012 result.md = INVALID_REFERENCE 박제 (fd64f6c) → measured 값 reference 없음. F0 raw 0.6320 (plan-006 hard evidence) + plan-013 G1 fallback 0.6381 (ref-only) 만 외부 reference.
- **plan-004 참조 범위 = 2가지만**: (a) input feature 가공 방식, (b) F0 64%/84% sample cover 입증. 그 외 = 새 module 안에서 from-scratch.

---

## §2. Scope (명시적)

### §2.1 In-scope (= Baseline 고정 + Ablation lever)

#### A. Baseline (4 컴포넌트 fixed, 모든 ablation 의 기준)

| 항목 | 값 |
|---|---|
| paradigm | codebook + classifier + regression hybrid |
| K | 7 (G3.E2 ablation 시 5/9/13 sub-exp) |
| **C1 encoder** | 새 module-local 2-layer BiGRU (hidden=128, input=9, output=256 via concat bidir), shared encoder + 2 head |
| Classifier head | linear → 7 logit |
| Regression head | linear → 7 × 3D offset (bound ±0.005m via tanh × scale) |
| **C2 F0** | learnable `(d1, par, perp)` (3 scalar), init = (1.98, 1.20, −0.20), Adam grad enabled. 산식 (plan-006 source-of-truth, `ring_classifier.py:512-565`): `F0 = p0 + d1·v_scale·v_last + par·acc_scale·acc_par_vec + perp·acc_scale·acc_perp_vec`. d2=0 / jerk=0 fixed (= plan-006 default). horizon=2, time_scale=1 → v_scale=acc_scale=1. `acc_par_vec` = acc·t̂×t̂ (acc 의 t̂ 성분), `acc_perp_vec` = acc − acc_par (perp plane 전체) |
| **C3 anchor radius** | 0.01m fixed scalar |
| **C4 soft label** | Gaussian σ=0.01m, `w_k ∝ exp(−d_k² / (2σ²))`, normalized over k=0..6 |
| Loss | `L = α × CE(logits, soft_label) + β × Huber(reg_offset, residual)`, (α=β=1.0) — G3.E4 swap 의 base |
| Inference | soft blend, τ=0.03 — G3.E3 τ scan 의 base |
| Input pipeline | shape `(N, 6, 9)` 시계열. 6 step indices = `range(max(3, end_idx-5), end_idx+1)` (pad first if <6). per-step 9 dim = `turn_model_features_from_context` output 8 dim ((1) `speed = ‖v_last‖`, (2) `prev_speed / speed`, (3) `acc_norm / speed`, (4) `acc_par_scalar / speed`, (5) `perp_norm / speed`, (6) `jerk_norm / speed`, (7) `turn_cos`, (8) `curvature`) + direction 1 dim. source-of-truth = `src/pb_0_6822/selector.py:280-294 + 406-449` (형식만 reuse, `selector.py` import X — 본 module 안에서 직접 재구현) |
| Validation | 5-fold OOF, fold = `hash(sample_id, salt='plan-014-v1') % 5` (새 module 내 재구현) |
| Training | Adam lr=1e-3, batch=256, epochs=50, patience=5, seed=20260514, **device=cuda** (plan-012 c18 ff1e578 GPU rerun 인프라 재사용) |
| Multi-seed | single seed (= 20260514). 분산 측정은 plan-015 후보 |
| New module | `src/pb_0_6822/plan014_paradigm.py` (`selector.py` import 0) |

#### B. Ablation lever (plan-012 5-Phase 그대로 carry, baseline 위에서 single-variable swap)

| Stage | lever | variants | base |
|---|---|---|---|
| **G2.Phase 1** | **E0 codebook bake-off** ★ | E0a Absolute-7Way (world ±x/±y/±z + center) / E0b Frenet-Orthogonal-7Way (±t̂/±n̂/±b̂ + center) / E0c K-Means-7Way (Frenet residual cluster) | 동일 arch + loss + τ + seed, 유일 변수 = anchor 좌표 집합 |
| **G3.Phase 2** | **E1 frame swap** (conditional) | Frenet vs world | winner ∈ {E0b, E0c} 만, E0a winner 면 SKIP |
| **G3.Phase 2** | **E2 K density** | K=5 / 7 / 9 / 13 | winner codebook |
| **G3.Phase 2** | **E3 τ scan** | argmax + τ ∈ {0.01, 0.03, 0.1, 0.3, 1.0} | inference-time hyperparam |
| **G3.Phase 2** | **E4 loss swap** | L7 hinge vs distance regression | baseline CE soft + Huber 의 cls loss form swap |
| **G3.Phase 2** | **E5 reg head on/off** | cls only / cls+reg hybrid | reg head 사용 여부 |
| **G4.Phase 3** | **E6 boundary weight** | on/off | boundary sample weighting |
| **G4.Phase 3** | **E7 scorer arch** | full BiGRU vs last-step MLP | C1 encoder variant |
| **G4.Phase 3** | **E8 r=0 logit prior** | 0 / +0.5 / +1.0 | center mode logit bias |

→ 총 11 ablation sub-experiment (E0 3-way + E1~E5 5 axis + E6~E8 3 axis). G5 에서 winner + best lever stack 으로 final 5-fold.

#### C. Dataset / IO

| 항목 | 값 |
|---|---|
| Train data | `data/train/{sample_id}.csv` (shape `(11, 3)`) — 11 timesteps × 3 axes |
| Train labels | `data/train_labels.csv` (columns: `id` / `x` / `y` / `z`) — y_true = position at +80ms from observation end |
| Test data | `data/test/{sample_id}.csv` (same shape `(11, 3)`) |
| Timestep grid | `[-400, -360, ..., -40, 0]` ms (step=40ms, `N_TIMESTEPS = 11`) |
| Target horizon | `T_TARGET_MS = 80` (관측 종료 후 +80ms 의 position) |
| `end_idx` (for `make_seq_features` / F0) | `N_TIMESTEPS − 1 = 10` (last observation index) |
| IO utility | `src/io.py` — `load_all_samples(split)` → `(ids, X (N, 11, 3))`, `load_labels()` → `(ids, Y (N, 3))`. plan-001 utility, import OK (= `selector.py` 와 별개 file, plan-004 module 재사용 정책과 무관) |

### §2.2 Out-of-scope

| 항목 | 이유 |
|---|---|
| plan-004/006/012 의 module·weight 재사용 일체 (`selector.py`, `CandidateAttentionGRUSelector`, plan-004 weight, plan-006 numpy F0, `ring_classifier.py`) | §0.5 plan-004 참조 (a)/(b) *외* totality 제거 — 본 plan = 새 module 안 from-scratch |
| 4 컴포넌트 baseline 의 ablation (C1 / C2 / C3 / C4 alone) | baseline 의 일부, ablation 대상 아님. 컴포넌트별 attribution = plan-015 |
| Corrector path / `boundary.py` / `corrector_redesign*` | plan-005~011 / plan-013 path 분리 |
| 27 후보 physics candidate / `candidates_extended.py` | scope X (= plan-008 산출) |
| TTA / multi-parse inference | plan-015 후보 |
| Ensemble (with plan-013 fallback or plan-012 ring) | plan-015 후보 (band 별 분기) |
| Baseline reproduce (plan-012 minimal-patch) | plan-012 INVALID 박제 (fd64f6c) 후 reference 없음 |
| LB 제출 | band 결과 따라 plan-015 결정 |

---

## §3. 사전 등록 (Pre-registration)

### §3.1 Fold split

- **5-fold OOF**: `fold_id = stable_hash(sample_id, salt='plan-014-v1') % 5` (새 module 내 재구현; SHA256(f"{salt}::{sample_id}").digest() → int.from_bytes([:8]) % 5)
- **G2/G3/G4 sub-exp**: 5-fold OOF (각 sub-exp 5 fold 학습)
- **G5 final**: 5-fold concat (모든 sample 이 정확히 1번 val 등장)
- decision-note: plan-004 `stable_fold_id` 와 분할 다를 수 있음 — plan-014 measurement 는 plan-014 scheme 내 self-consistent (외부 비교 = §3.5 reference only)

### §3.2 평가 metric

- **main metric (G5 band 판정)**: `best_stack_5fold_hit_1cm = mean(‖hybrid_pred − y_true‖₂ ≤ 0.01m)` (5-fold concat OOF)
- `hybrid_pred = F0_pred + Σ_{k=0..K−1} prob_k × (a_k + reg_offset_k)`
  - `F0_pred = p0 + d1·v_scale·v_last + par·acc_scale·acc_par_vec + perp·acc_scale·acc_perp_vec` (3 learnable scalar `(d1, par, perp)`, init (1.98, 1.20, −0.20); d2=0 / jerk=0 fixed; horizon=2 / time_scale=1 → v_scale=acc_scale=1; source = `ring_classifier.py:512-565`)
  - `prob = softmax(logits / τ)`, baseline τ = 0.03
- **secondary**: `hit@1.5cm`
- **diagnostic**: `directional_commit_magnitude (DCM) = mean(‖hybrid_pred − F0_pred‖₂)` — encoder 신호 살아있는지 측정 (G2 criterion)
- **band classifier** (§0.5 박제):
  - best stack 5-fold OOF ≥ 0.66 → **positive** (paradigm 부활)
  - 0.65 ≤ OOF < 0.66 → **partial** 회복
  - OOF < 0.65 → **negative** (premise 의심)

### §3.3 The Configs

#### Baseline config (G2 의 base, 모든 ablation 의 zero-modification reference)

§2.1.A 박제 그대로. = 9d×6step input → BiGRU(128) → cls(7) + reg(7×3) → soft blend τ=0.03 → CE soft + Huber loss.

#### Ablation variants (G2.E0 / G3.E1~E5 / G4.E6~E8)

각 lever 별 sub-exp = baseline 위에서 단 1 변수 swap. 상세 spec = §6~§9 STAGE 별 stub.

### §3.4 G-gate quantitative criteria

#### G0 — preflight artifact

- artifact: `analysis/plan-014/preflight.json`
- (a) F0 init reproduce: `(d1, par, perp) = (1.98, 1.20, −0.20)` 으로 init 한 모든 train sample 의 hit@1cm 측정 → plan-006 reference (0.6320) ± 0.005 일치. 산식 = `ring_classifier.py:512-565` 그대로 (numpy 함수 import 는 X — 새 module 안에서 동일 산식 직접 재구현)
- (b) anchor scale 박제: radius=0.01m, ±t̂/±n̂/±b̂/center 7 anchor Frenet local coord (= ±0.01 직교 set + 원점)
- (c) soft label entropy: σ=0.01m Gaussian → sample-별 entropy 평균 ≥ 0.5 nat
- (d) plan-012 disclaimer verify: `INVALID_REFERENCE` status + `disclaimer:` field 박제 grep
- fail trigger: (a)~(d) 중 1+ 누락 → `preflight_artifact_missing` severe

#### G1 — 새 module 구현 + 재사용 끊김

- artifact: `src/pb_0_6822/plan014_paradigm.py` + smoke test (`tests/test_plan014_smoke.py`)
- (a) smoke train: 1-fold 1-epoch — no NaN, val_loss < initial val_loss
- (b) 재사용 끊김 4가지:
  1. `selector` / `ring_classifier` / `boundary` import 0
  2. F0 `α_*.grad` ≠ None ≠ 0 (1 backward 후)
  3. anchor `‖a_k‖ = 0.01 ± 1e-6` (NOT 0.005m)
  4. soft label entropy ≥ 0.5 nat
- fail trigger: 1+ fail → `reuse_cut_violation` severe (premise 위배)

#### G2 — Phase 1 codebook bake-off

- artifact: `analysis/plan-014/g2_phase1_bakeoff.py` + `g2_phase1.json` + 3 sub-exp `runs/baseline/plan014_g2_E0{a,b,c}/`
- spec: E0a/E0b/E0c 3 sub-exp 5-fold OOF hit@1cm 측정 (동일 arch + loss + τ + seed)
- winner: `argmax(OOF over E0a, E0b, E0c)`. tie-break (gap < 0.005): 단순성 우선 E0a > E0b > E0c
- criterion:
  - winner_OOF ≥ **0.60** (= encoder 가 forward path 망가지지 않음 verify)
  - winner DCM ≥ **0.002** (= encoder mode 신호 살아있음 verify, F1 collapse 회피)
- fail trigger:
  - winner_OOF < 0.60 → `g2_severe_underperform` severe
  - winner DCM < 0.002 → `dcm_collapse` warn (Phase 2~4 informational 진행)

#### G3 — Phase 2 axis ablation 5

- artifact: `analysis/plan-014/g3_phase2_axis.py` + `g3_phase2.json` + 13~14 sub-exp runs (E1 conditional + E2 4 + E3 6 + E4 2 + E5 2)
- spec: 5 axis (E1~E5) winner codebook 위에서, 각 axis 의 sub-exp 5-fold OOF ΔOOF = OOF_variant − OOF_winner
- criterion: 5 axis 중 1+ axis 의 max(ΔOOF) ≥ **+0.005** → paradigm lever 마진 살아있음
- fail trigger: 모든 axis ΔOOF < 0.005 → `g3_marginal_only` warn (Phase 3~4 informational, G5 anchor fallback 후보)

#### G4 — Phase 3 aux ablation 3

- artifact: `analysis/plan-014/g4_phase3_aux.py` + `g4_phase3.json` + sub-exp runs
- spec: 3 axis (E6/E7/E8) winner config 위
- criterion: informational only (G3 pass 했다면 추가 lever 줍기, G3 warn 이면 fallback evidence)

#### G5 — Phase 4 final 5-fold + best stack + submission

- artifact: `analysis/plan-014/g5_phase4_final.py` + `g5_phase4.json` + `runs/baseline/plan014_g5_phase4/submission.csv`
- spec: G2/G3/G4 winner + best lever stack 으로 5-fold concat OOF + submission 생성
- best_stack 정의: anchor config + G3/G4 의 max(ΔOOF) > 0 인 lever 들의 combined config
- criterion: **best_stack 5-fold OOF ≥ anchor_5fold + 0.005** (= lever 마진 진짜 살아있음 verify)
- band 분류 (§3.2): best_stack OOF 기준 ≥0.66 / 0.65~0.66 / <0.65
- fail trigger: best_stack < anchor + 0.005 → `g5_no_additive` warn (anchor fallback submission)

#### G_final — synthesis

- artifact: `plans/plan-014-plan012-failure-inversion.results.md` 신규 + `registry.csv` append + plan-014 frontmatter sync (`lb_score` / `exp_ids` / `status: spec → completed`)
- content:
  - G0~G5 결과 narrative
  - band 분류 결과 (positive / partial / negative)
  - plan-013 join interpretation table (§1.4) 의 activated row 박제
  - plan-015 후보 ≥ 3 (band 별 분기)
- fail trigger: 3 파일 sync 누락 → `final_sync_missing` severe

### §3.5 External reference (plan-006 / plan-013 — plan-012 INVALID 박제 후)

| measure | plan-006 / plan-013 (외부 ref) | plan-014 target |
|---|---|---|
| F0 raw hit@1cm | 0.6320 (plan-006 F0 공식 hard evidence — §1.5 plan-004 참조 (b)) | G0 (a) reproduce (±0.005) |
| 5-fold OOF hit@1cm | plan-013 G1 fallback 0.6381 (paradigm 폐기 path best, ref-only) | G5 best_stack band 측정 (§3.2): ≥0.66 / 0.65~0.66 / <0.65 |
| G1 winner OOF (Phase 1) | — | G2 winner_OOF ≥ 0.60 + DCM ≥ 0.002 |
| Phase 2 axis ΔOOF | — | G3 1+ axis ≥ +0.005 |
| best_stack vs anchor | — | G5 ≥ +0.005 (best_stack additive verify) |

decision-note: plan-012 measured 값 (0.6350 / 0.6411) 은 INVALID_REFERENCE 박제 (fd64f6c) 후 reference 제거. F0 raw 0.6320 는 plan-006 의 F0 공식 자체의 hard evidence 로서 reference 정당.

---

## §4. STAGE 0 (c4, G0) — preflight artifact [TODO]

(c3 v3 spec replacement 단계 — c4 commit 시 채움)

## §5. STAGE 1 (c5, G1) — `plan014_paradigm.py` 새 module + smoke + 재사용 끊김 [TODO]

(c3 v3 spec replacement 단계 — c5 commit 시 채움)

## §6. STAGE 2 (c6, G2) — Phase 1 codebook bake-off (E0a/E0b/E0c) [TODO]

(c3 v3 spec replacement 단계 — c6 commit 시 채움)

## §7. STAGE 3 (c7, G3) — Phase 2 axis ablation 5 (E1~E5) [TODO]

(c3 v3 spec replacement 단계 — c7 commit 시 채움)

## §8. STAGE 4 (c8, G4) — Phase 3 aux ablation 3 (E6~E8) [TODO]

(c3 v3 spec replacement 단계 — c8 commit 시 채움)

## §9. STAGE 5 (c9, G5) — Phase 4 final 5-fold + best stack + submission [TODO]

(c3 v3 spec replacement 단계 — c9 commit 시 채움)

## §10. STAGE 6 (c10, G_final) — synthesis + plan-015 후보 + 3 파일 sync [TODO]

(c3 v3 spec replacement 단계 — c10 commit 시 채움)
