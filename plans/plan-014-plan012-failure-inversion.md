---
plan_id: 014
version: 4.5 (spec patch — plan-review-master iter 5 (max) fix 4건 → loop 종료. (1) §3.4 G5 best_stack 정의 wording sync to §9.1: "Phase 2 best 1 + Phase 3 best 1, ΔOOF > 0 인 경우 채택, max 3 elements stack". (2) §3.4 G1 (a) val_hit threshold sync to §5.4: strict `>` → `≥ initial − 0.05` margin (random-init variance 흡수). (3) §3.4 G1 (b)(ii) F0 range 4-digit 통일 `[0.6270, 0.6370]`. (4) §2.1.B.1 E6 boundary weight `weighted_mean = Σ(w·loss)/Σw` 분모 = Σw (plan-012 convention carry). plan-review-master loop max iter 도달, BLOCKER 0. v4.4 → v4.5.)
date: 2026-05-14 (Asia/Seoul)
status: spec
based_on:
  - 012
followed_by: []
scope: plan-012 (codebook bake-off + hybrid) 의 5-fold OOF plateau 의 root cause = *plan-004 selector + plan-012 corrector 코드의 재사용 강박* — premise 채택 (검증 안 함). 그 premise 위에서 plan-012 의 5-Phase 실험 프로세스 (preflight → bake-off → axis ablation → aux ablation → final 5-fold) 를 **F0 (plan-006 frenet_par120_perp_neg020) frozen prior + corrector from-scratch 재구현 위에서 그대로 재실행**. baseline = (C1 from-scratch BiGRU corrector + C2 frozen plan-006 F0 + C3 anchor 0.01m + C4 Gaussian soft σ=0.01m) fixed, 그 위에서 plan-012 의 8 ablation lever + 3 codebook bake-off 진행. baseline reproduce 없음 + plan-012 measured 값 reference 없음 (plan-012 result.md = INVALID_REFERENCE 박제 fd64f6c 후 §Target band absolute ≥0.66 / 0.65~0.66 / <0.65).
exp_ids: []
lb_score: null
---

# plan-014 v4 — F0 (plan-006) frozen prior + corrector from-scratch 정밀화

## §0. 한 줄 목적

> **F0 단일 공식 (plan-006 `frenet_par120_perp_neg020`, d1=1.98 / par=1.20 / perp=−0.20) 만으로 hit@1cm = 64%, hit@1.5cm = 84%.** 84% 의 sample 은 F0 근방 1.5cm 안 → 그 중 20% 만 1cm 밖. 본 plan 의 task essence = **F0 prior 를 고정한 채, 그 위에 corrector (encoder + cls + reg head + anchor codebook) 를 from-scratch 정밀화 해서 1.5cm 안 20% sample 을 평균 0.5cm 적절한 방향으로 끌어당기기**. residual *vector regression* 직접 회귀 (plan-005~007) 는 어렵다고 입증됨 → residual *direction* 만 classification, magnitude = anchor scale prior + small offset. 이게 plan-012 의 codebook + classifier + regression hybrid paradigm 의 본질.
>
> 본 plan = **plan-012 의 5-Phase 실험 프로세스 (preflight / codebook bake-off / axis 5 ablation / aux 3 ablation / final 5-fold + best stack) 를 plan-004/012 corrector 코드 재사용 끊은 새 module 위에서 그대로 재실행**. **F0 자체는 재사용 끊기 대상 아님** — plan-006 hard evidence 0.6320 의 산식을 본 module 안에서 직접 재구현 (frozen, 학습 안 함). plan-012 의 G-gate spec frame carry over (단 threshold 는 v2.3 absolute sync 결과 적용).
>
> **참조 범위** (3가지만): (a) input feature 가공 방식 (시계열 9d × 6step 형식, plan-004 selector.py carry), (b) F0 sample cover 입증 (64%/84%, plan-006 hard evidence), (c) **F0 단일 공식 산식 자체** (plan-006 `frenet_par120_perp_neg020`, frozen prior). 그 외 = 새 module `src/pb_0_6822/plan014_paradigm.py` 안에서 from-scratch.
>
> 본 plan 의 best stack 5-fold OOF 가 §Target band (≥0.66 / 0.65~0.66 / <0.65, absolute) 의 어느 위치에 들어가는지가 corrector paradigm 의 진짜 잠재력 측정. plan-012 result.md = INVALID_REFERENCE 박제 (fd64f6c) → measured 값 비교 reference 없음.

---

## §0.5 Quick Reference

### 본 plan 의 task essence — "F0 64% cover + corrector 가 1.5cm 안 20% 끌어당김" (★ narrative anchor)

- **F0 단일 공식** (plan-006 `frenet_par120_perp_neg020`, frozen): hit@1cm = **0.6320** (plan-006 hard evidence), hit@1.5cm = **0.8033**. 산식 = `F0 = p0 + 1.98·v_last + 1.20·acc_par_vec + (−0.20)·acc_perp_vec` (d1/par/perp = constants, 학습 안 함).
- **남은 20%** (= 84% − 64%): F0 근방 1.5cm 안이지만 1cm 밖. corrector 가 평균 0.5cm 적절한 방향으로 이동 시 hit@1cm.
- **residual 직접 회귀 사망 진단** (plan-005~007): residual vector regression 어려움 → direction classification + small magnitude offset.
- **방향 후보 = Frenet local frame 7 방향** (±t / ±n / ±b / center): trajectory-aligned 방향 분리 = "어느 방향으로 0.5cm 이동" task 직관 일치.

### 참조 범위 — 3가지

- **(a) input feature 가공 방식** — 시계열 9d × 6step 형식·전처리 (plan-004 `selector.py:280-294 + 406-449` carry naming, 산식 재구현).
- **(b) F0 sample cover 입증** — 64%/84% cover (plan-006 hard evidence).
- **(c) F0 단일 공식 산식 자체** — plan-006 `frenet_par120_perp_neg020` 의 d1=1.98 / par=1.20 / perp=−0.20 constants + Frenet finite-diff 식 (plan-012 `ring_classifier.py:512-565` carry, 산식 재구현). **frozen, 학습 안 함** (= plan-012 의 frozen numpy F0 의 *의도* 만 carry, *import* 는 끊음).

**위 3가지 외 = 전부 새로 build** — 새 module `src/pb_0_6822/plan014_paradigm.py` 안에서 from-scratch (시계열 모듈도 `nn.GRU` 등 표준 layer 직접 생성, plan-004/012 corrector 모듈 import 0).

### 본 plan 의 premise

- **Premise**: plan-012 의 plateau root cause = "plan-004 `CandidateAttentionGRUSelector` + plan-012 `ring_classifier.py` corrector 코드의 재사용 강박" → task essence (corrector 정밀화) 와 mismatch. 재사용 끊고 corrector 만 from-scratch 재설계 하면 F0 (frozen) 위 +0.03~0.04 회수 가능 (band ≥ 0.66).
- 본 plan 은 premise 검증 안 함. **corrector paradigm 의 measured 잠재력만 박제**.
- premise 가 *틀렸을 경우* 표식 = §Target negative band — "premise 오류" vs "corrector paradigm 자체 한계" 분리 = plan-013 join interpretation 으로만 (§1.4).
- **F0 자체는 ablation 대상 아님** (= baseline 의 일부, frozen). F0 attribution = plan-015 후속 (corrector OOF measured 후 평가).

### plan-012 가 "제대로" 가 아니었던 이유 (재사용 6 증상 → premise 근거)

plan-012 ring_classifier.py 는 paradigm shift 라고 self-label 했지만 실제 corrector 코드는 다음 *minimal patch*:

```
self.scorer  = base.CandidateAttentionGRUSelector(...)   # ← plan-004 의 27-way selector 그대로 (corrector)
self.reg_head = nn.Sequential(...)                       # ← 위에 작은 MLP 추가 (corrector)
F0 = f0_predict_frenet_par120_perp_neg020(...)           # ← plan-006 numpy 함수 그대로 — frozen prior, 의도된 설계 (issue 아님)
anchor radius = 0.005m                                   # ← plan-004/006 era scale hardcode (corrector)
```

6 observable failure mode → 단일 root cause (corrector 재사용 강박. F0 frozen 은 본 plan 도 *carry* 하는 의도):

| failure mode | corrector 재사용 강박과의 인과 |
|---|---|
| F4 candidate-attention inductive-bias mismatch | 본 plan task = "20% sample 0.5cm 끌어당김" ≠ plan-004 task = "sample-별 27 후보 비교". `CandidateAttentionGRUSelector` = (a) GRU [task-neutral] + (b) candidate-attention head [plan-004 fit]. ring_classifier 의 classifier path 가 (b) 까지 같이 호출 → plan-012 의 fixed 7 anchor 와 mismatch. |
| F1 DCM collapse | F4 결과 — encoder 신호 부족 → classifier head 가 safe minimum (center mode) 수렴 |
| F3 F0 trivial dominance | F0 hit 63% sample 학습 signal 무의미 → corrector head 가 학습할 거리 없음 → mode 0 collapse |
| F2 anchor scale mismatch | plan-004/006 의 0.005m 답습, task fit 재검토 없음 (corrector 의 anchor scale 결정) |
| F5 hard label CE noise | F2 결과 — anchor 가 hit zone 내부 갇혀 argmin label = noise (corrector loss) |
| F6 codebook geometry uniformity | F1 결과 — corrector encoder 가 center 만 고르니 anchor 위치 무관 |

→ 6 증상 = 1 root cause (corrector 재사용). 본 plan = 그 root cause 를 *제거한* corrector 위에서 plan-012 의 ablation 들이 살아 있는지 측정 (= falsify 아닌 *재실험*).

**(v3.6 의 F7 row "frozen F0 path → gradient 없음 = failure" 제거)** — frozen F0 은 plan-012 의 의도된 설계 (plan-006 hard evidence prior). 본 plan 도 동일 carry. F0 학습 가능성은 plan-015 후속 attribution 과제.

### 본 plan 의 multi-path 설계 — plan-012 5-Phase frame (v4)

```
G0 preflight  →  G1 module + smoke  →  G2 Phase 1 bake-off  →  G3 Phase 2 axis 5  →  G4 Phase 3 aux 3  →  G5 Phase 4 final 5-fold  →  G_final synthesis
```

- **Baseline = 4 컴포넌트 fixed** (재사용 끊기 spirit 보존, ablation 대상 *아님*):
  - **C1 corrector (encoder + heads)**: 새 module-local 2-layer BiGRU (hidden=128) + cls head (7-logit) + reg head (7×3 offset, tanh×0.005). shared encoder. **from-scratch, learnable** (= corrector 정밀화 main lever).
  - **C2 F0**: plan-006 `frenet_par120_perp_neg020` 산식 본 module 안 재구현. d1=1.98 / par=1.20 / perp=−0.20 **frozen constants** (학습 안 함, nn.Buffer 또는 plain function). `F0 = p0 + 1.98·v_last + 1.20·acc_par_vec + (−0.20)·acc_perp_vec` (Frenet finite-diff, horizon=2, time_scale=1).
  - **C3 anchor radius**: 0.01m fixed.
  - **C4 soft label**: Gaussian σ=0.01m kernel.
  - 컴포넌트별 attribution = plan-015 후속 과제.
- **G2 Phase 1 codebook bake-off**: E0a Absolute / E0b Frenet / E0c K-Means 3-way 학습 → winner 1개 결정 (tie-break = 단순성 우선 E0a > E0b > E0c, plan-012 G1 rule carry).
- **G3 Phase 2 axis ablation 5** (winner codebook 위): E1 frame swap (conditional) / E2 K density (K=5/7/9/13) / E3 τ scan / E4 loss swap (L7 hinge vs distance reg) / E5 reg head on/off.
- **G4 Phase 3 aux ablation 3**: E6 boundary sample weighting / E7 scorer arch (BiGRU vs last-step MLP) / E8 r=0 logit prior (0/+0.5/+1.0).
- **G5 Phase 4 final 5-fold + best stack + submission**: winner config + best lever 들 stack 으로 5-fold concat OOF + submission 생성.

### Target (judgement criteria) — absolute (v2.3 sync 유지)

- **OOF ≥ 0.66** ★ positive (corrector paradigm 부활). plan-015 = polish + LB. decision-note: 0.66 = competition-level paradigm target.
- **0.65 ≤ OOF < 0.66** partial 회복. plan-015 = corrector + 본 plan hybrid (plan-013 Candidate C 변형). decision-note: 0.65 = F0 raw 0.6320 + ~0.018 round absolute margin.
- **OOF < 0.65** negative. plan-015 = deep path pivot (`notes/new-ideas.md` KNN/GP/Diffusion).
- best stack 의 5-fold concat OOF 가 band 판정 대상.

### G-gates (정량 spec @ §3.4)

- **G0** preflight: F0 frozen reproduce ±0.005 (= 0.6320) / anchor 0.01m / soft entropy ≥0.5 nat / plan-012 disclaimer verify [DONE 6c56dd1: F0=0.6320 / oracle (E0a/E0b/E0c)=(0.8203/0.8248/0.7625) / soft=1.707 nat / ranking abs=[y,x,z] fro=[n,b,t]]
- **G1** module build: `plan014_paradigm.py` + smoke + 재사용 끊김 4가지 (i) `selector` / `ring_classifier` / `boundary` / plan-006 numpy F0 함수 import 0 (ii) F0 forward = (1.98, 1.20, −0.20) reproduce ±0.005 + grad path 끊김 (= F0 는 nn.Buffer 또는 plain function, requires_grad 없음) (iii) anchor ‖·‖ = 0.01m ± 1e-6 (iv) soft label `w_k` (target Gaussian 분포) 의 sample-별 entropy 평균 ≥0.5 nat — **G0 (c) 와 같은 분석적 산출 (학습 전, model output prob_k 와 별개)** [DONE 8ef0c3c: 5/5 pytest PASS, F0 = Plan014F0Function plain class]
- **G2** Phase 1 bake-off: winner_OOF ≥ 0.60 + DCM ≥ 0.002 (plan-012 G1 spec carry) [DONE 4783ae8: winner=E0c K-Means (OOF=0.6359, DCM=0.0026) — F0 raw 위 +0.0039 회수]
- **G3** Phase 2 axis 5: 5 axis 중 1+ ΔOOF ≥ 0.005 (plan-012 G2 spec) [DONE 76a4e52: `g3_marginal_only` warn, positive_axes=[]. max ΔOOF E2 +0.003 (K=9 best, marginal)]
- **G4** Phase 3 aux 3: informational [DONE 92417dd: E6 +0.0015, E7 -0.0050, E8 -0.0040. positive_axes=['E6']]
- **G5** Phase 4 final: best_stack ≥ anchor_5fold + 0.005 (plan-012 G4 spec) + band 분류 [TODO]
- **G_final** synthesis: results.md 신규 + registry append + frontmatter sync + plan-015 후보 [TODO]

### Commit chain

| # | type | spec section | status |
|---|---|---|---|
| c1 ~ c2.3 | docs | v0~v2.3 narrative + spec drop + sync (git log authoritative) | [DONE] 4657ff7~b6bf927 (c1: 4657ff7+2a0f755 / c1.1: c7cf5c8 / c1.2: 5e98d6d / c1.3: c7fa9c8 / c1.4: 3a7a26c / c1.5: ab50cce / c2: ad051e2 / c2.1: 0a3c317 / c2.2: 90d9e0d / c2.3: b6bf927) |
| **c3** | docs | **v3 spec replacement — plan-012 5-Phase frame import.** v2.x single-path 폐기, 4 컴포넌트 baseline (fixed) + plan-012 5-Phase ablation frame 으로 재작성. (= v3.6 까지 c3.1~c3.6 sub-patches) | [DONE] 5f6750b ~ f304804 (git log authoritative) |
| c3.1 ~ c3.6 | docs | v3.1~v3.6 spec patches (F0 산식 정정 / input feature / dataset IO / lever source / submission format / JSON schema) | [DONE] ba9e994~f304804 |
| **c1.v4** | docs | **v4 spec replacement — F0 frozen prior narrative fix.** v3.6 의 §0.5 "F7 frozen F0 path = failure mode" misframing 제거 + §2.1.A C2 learnable → frozen plan-006 frenet_par120_perp_neg020 (d1=1.98 / par=1.20 / perp=−0.20 fixed constants). "재사용 끊기" 범위 = corrector 코드만 (F0 산식 자체는 carry). §0.5 failure mode 표 7→6 row, §1.3 trap chain F7 분기 제거, §3.4 G1 4 check 의 F0 grad 항목 → F0 함수 재구현 verify. plan-014 task essence = "F0 0.6320 prior 위에 corrector 가 1.5cm 안 20% sample 을 0.5cm 끌어당기기" 명시. v3.6 → v4 frontmatter version. (rollback to 778198f 시점 후 plan 재정의, 사용자 명시 지시) | [DONE] b83a736 |
| c1.v4.1 | docs | **v4.1 spec patch — plan-review-master iter 1 fix.** (1) §4~§10 STAGE spec fill (v3.7 carry, v4 narrative align: F0 frozen 공통 명시 + G5 best_stack stacking rule disambiguation). (2) §0.5 G1 (iv) entropy 측정 명세 (target w_k 분포 학습 전 분석적 산출). (3) §2.1.A Input pipeline feature (8) curvature placeholder 의 (5) 중복이 plan-004 carry 의 의도된 redundancy 명시. v4 → v4.1 | [DONE] 17c2071 |
| c1.v4.2 | docs | **v4.2 spec patch — plan-review-master iter 2 fix 8건.** (1) §2.1.A C1: BiGRU output reduction = last-step bidir concat. (2) §5.2: Plan014HybridHead forward/hybrid_predict 분리 + F0 detach caller-side 명시. (3) §5.4 G1 smoke initial_val_hit 측정 spec + threshold − 0.05 margin. (4) §2.1.B.1 E2 K density frame 선택 룰 (winner-codebook 별 axis_family frame). (5) §7.1 E1 anchor 변수 표식. (6) §7.1 E4 2-variable composition 명시. (7) §3.4 G_final registry 6 row 의 12 column 값 룰. (8) §5.2 run_kfold_oof config dict key set + E7 MLP GELU 위치. v4.1 → v4.2 | [DONE] 906eb67 |
| c1.v4.3 | docs | **v4.3 spec patch — plan-review-master iter 3 fix 5건.** (1) §2.1.B.1 E2 K=9/13 anchor `±(a+b)/√2` norm 보장 가정 박제. (2) §9.1 E4+E5 동시 채택 final loss formula 명시 (`use_reg_head=False, use_hinge=True` 분기). (3) §2.1.A.1 F0 1.98 dimensionless hard-coded constant (plan-006 carry, dt 흡수) + 단위 일관. (4) §5.4 G1 smoke val_hit_after path = `model.eval()` + `hybrid_predict()` (inference path 동일) 명시. (5) §3.1 single→5-fold sign mismatch sub-exp 제외 룰 박제. v4.2 → v4.3 | [DONE] 28235c3 |
| c1.v4.4 | docs | **v4.4 spec patch — plan-review-master iter 4 fix 6건.** (1) §5.2 train_one_fold explicit signature. (2) §9.2 3 helper signature + 반환 spec. (3) §7.1 E5 ΔOOF 부호 convention 통일 (variant − anchor, 다른 axis 동일). (4) §3.4 G5 submission keys 4 explicit. (5) §2.1.A Loss batch reduction = mean 명시. (6) §3.2 hit@1.5cm 정의식 박제. v4.3 → v4.4 | [DONE] 56f31c7 |
| c1.v4.5 | docs | **v4.5 spec patch — plan-review-master iter 5 (max) fix 4건 → loop 종료.** (1) §3.4 G5 best_stack 정의 wording sync to §9.1 (Phase 2 best 1 + Phase 3 best 1, max 3 elements). (2) §3.4 G1 (a) threshold sync to §5.4 (strict > → ≥ initial − 0.05 margin). (3) §3.4 G1 F0 range 4-digit 통일. (4) §2.1.B.1 E6 boundary weighted_mean 분모 = Σw 명시. plan-review-master loop 최종 종료. v4.4 → v4.5 | [DONE] c7f4d31 |
| c4 | code+exp | STAGE 0 (G0) — preflight artifact (F0 frozen reproduce 0.6320 ±0.005). spec @ §4. PASS 4/4: F0 frozen hit@1cm=0.6320 (정확 reproduce). oracle (E0a/E0b/E0c)=(0.8203/0.8248/0.7625). marginal ranking abs=[y,x,z] fro=[n,b,t]. min_cluster=338 | [DONE] 6c56dd1 |
| c5 | code | STAGE 1 (G1) — `src/pb_0_6822/plan014_paradigm.py` v4 module + smoke + 재사용 끊김. spec @ §5. 5/5 PASS: AST import 0 (selector/ring_classifier/boundary/f0_predict_*) / F0 frozen plain class (no nn.Parameter, reproduce 0.6320) / anchor 0.01m / soft entropy ≥0.5 / smoke val_hit_after >= initial − 0.05 | [DONE] 8ef0c3c |
| c6 | code+exp | STAGE 2 (G2) — Phase 1 codebook bake-off (E0a/E0b/E0c 3 sub-exp → winner). spec @ §6. **PASS** (v4 F0 frozen): winner=E0c K-Means (OOF=**0.6359** / DCM=0.0026 / gap=0.0066). E0a=0.6293 (DCM=0.0014) / E0b=0.6239. F0 raw 0.6320 위 +0.0039 회수 — v3.x cascade failure 회피 confirmed | [DONE] 4783ae8 |
| c7 | exp | STAGE 3 (G3) — Phase 2 axis ablation 5 (E1~E5). spec @ §7. **`g3_marginal_only` warn**: positive_axes=[]. anchor fold-0=0.6573. max ΔOOF: E1=-0.0055 / E2=+0.0030 (E2c K=9 best) / E3=-0.0005 / E4=-0.0050 / E5=-0.0015. paradigm 위 의미 있는 lever 0개 — F0 frozen ceiling 신호 | [DONE] 76a4e52 |
| c8 | exp | STAGE 4 (G4) — Phase 3 aux ablation 3 (E6~E8). spec @ §8. informational mode (G3 carry). E6 boundary_weight=+0.0015 (marginal positive) / E7 LastStep MLP=-0.0050 (BiGRU 시계열 가치 입증) / E8 prior=-0.0040. positive_axes=['E6'] | [DONE] 92417dd |
| c9 | exp | STAGE 5 (G5) — Phase 4 final 5-fold + best stack + submission. spec @ §9 | [TODO] |
| c10 | docs+sync | STAGE 6 (G_final) — results.md + registry + frontmatter sync + plan-015 후보. spec @ §10 | [TODO] |

---

## §1. 배경 / 동기 (narrative)

### §1.1 v0 → v1~v2.3 → v3.x → v4 narrative evolution

| 축 | v0 (검증, 폐기) | v1~v2.3 (single-path, 폐기) | v3~v3.6 (잘못된 framing) | **v4 (현재, 본 plan)** |
|---|---|---|---|---|
| premise 위치 | hypothesis | assumed | assumed | **assumed (보존)** |
| 실험 path | A+B 2 path (head-to-head) | B 단독 1 path (4 컴포넌트 동시 swap) | plan-012 5-Phase multi-config | **plan-012 5-Phase multi-config (보존)** |
| baseline | A reproduce (in-plan) | plan-012 measured 외부 ref | 4 컴포넌트 from-scratch incl. **learnable F0** (= 잘못) | **4 컴포넌트, F0 frozen + corrector from-scratch + learnable** |
| Target | B − A gap 기반 | OOF 절대값 (≥0.66 / 0.65~0.66 / <0.65) | OOF 절대값 동일 | **OOF 절대값 동일 (변경 없음)** |
| ablation 정책 | 해당 없음 | lever 마진 줍기 회피 (4 컴포넌트 동시) | plan-012 ablation 재실행 | **plan-012 ablation 재실행 (corrector 만, F0 frozen 위)** |
| 실험 의도 | falsify | 잠재력 baseline 측정 | 잠재력 + lever 마진 동시 측정 | **corrector 잠재력 + lever 마진 (F0 frozen 위) 동시 측정** |
| "재사용 끊기" 범위 | (해당 없음) | (해당 없음) | corrector + F0 둘 다 (= 잘못) | **corrector 만 (F0 산식은 carry, import 만 끊음)** |

→ v4 = v3.x 의 F0 learnable misframing 정정. 핵심 narrative = "F0 plan-006 frozen prior + corrector from-scratch 정밀화". v3.x 의 "F7 frozen F0 = failure mode" 진단은 잘못된 framing (= plan-014 narrative 위반, plan-012 carry 위반). v4 = narrative 정직성 회복.

### §1.2 plan-012 의 사망 진단 — premise 의 근거

plan-012 results.md = "paradigm reframe 은 F0 raw hit 위 +0.002~0.003 만 추가 — paradigm 자체의 limit 확인". 그러나 plan-012 의 코드 = "plan-004 selector + plan-012 corrector head 위에 hybrid head 만 얹은 minimal patch" (plan-006 numpy F0 는 frozen, plan-012 의 의도된 설계 — issue 아님). → plan-012 가 measured limit = **"corrector minimal patch 의 limit"** 일 뿐, **"corrector paradigm 의 limit"** 은 아직 측정 안 됨.

> **plan-012 의 6 failure mode 는 6개 독립 문제가 아니라 1개 root cause (corrector 재사용 강박) 의 6가지 증상.**

- 6-lever ablation (plan-012 의 G2/G3) = 증상 치료 ≠ 원인 치료 → 모든 lever 가 marginal 이었던 것은 합당.
- 본 plan = 원인 (corrector 재사용) 제거 후 corrector paradigm 의 *제대로 된* baseline 측정 + 그 위에서 lever 들 재측정. **F0 frozen 은 plan-012 의 의도된 carry — 끊기 대상 아님**.

### §1.3 corrector 재사용 강박의 trap chain (v4 정정)

```
"plan-004 selector reuse"  ─┐
                            ▶ candidate-attention inductive bias mismatch (F4). `CandidateAttentionGRUSelector` = (a) GRU [task-neutral] + (b) candidate-attention [plan-004 fit]. ring_classifier 의 classifier path 가 (b) 까지 호출 → plan-012 의 fixed 7 anchor (sample-invariant) 와 mismatch.
                            ▶ classifier head 가 신호 없이 학습 = safe minimum (mode 0 center) collapse (F1)
                            ▶ codebook geometry 가 결과에 무관 (F6)

"plan-012 corrector head reuse" ──┐
                            ▶ F0 hit 63% sample 학습 signal 무의미 (F3) — corrector head 가 학습할 거리 없음
                            ▶ F0 trivial dominance — corrector 가 F0 위 marginal-only 학습

"plan-004/006 era scale"  ──┐
                            ▶ anchor 0.005m 답습 → hit zone 내부 갇힘 (F2)
                            ▶ hard label = noise (F5) — F2 의 anchor 위치가 hit zone 내부라 argmin label 신뢰성 없음
```

→ 6 mode 의 *원인이 1개* 라는 진단이 옳다면, **corrector 재사용 끊은 baseline** 위에서 6 mode 가 동시에 풀리고 *추가로* plan-012 lever 들 (E0~E8) 의 마진도 살아 있어야 함. 본 plan 의 outcome 이 그 outcome-level 신호.

**(v3.x trap chain 의 "plan-006 numpy F0 reuse → F7 frozen path" 분기 제거)** — frozen F0 은 plan-012 의 의도된 설계 (plan-006 hard evidence). 본 plan 도 동일 carry, 끊기 대상 아님. trap chain 의 root cause 는 corrector 재사용 (selector + head + scale) 만.

### §1.4 plan-013 과의 path 분기 — join interpretation

- **plan-013** (직전): paradigm 폐기 + plan-004 framework 회귀 → G2 0/3 axis FAIL, G1 0.6381 fallback submission
- **plan-014** (본 plan): corrector paradigm 부활 시도 — corrector 재사용 끊고 F0 frozen + plan-012 5-Phase 재실행

| plan-013 LB | plan-014 best stack 5-fold OOF | 결합 해석 |
|---|---|---|
| ≥ 0.68 | < 0.65 (negative) | corrector paradigm 폐기 정당화 — plan-004 framework path 가 정답 |
| < 0.68 | ≥ 0.66 (positive) | corrector paradigm 부활 — from-scratch redesign 이 정답, premise 옳음 |
| ≥ 0.68 | ≥ 0.66 (positive) | 둘 다 작동 — plan-015 = 두 path 의 ensemble/stacking |
| < 0.68 | < 0.65 (negative) | 둘 다 실패 — 더 deep path-pivot (`notes/new-ideas.md` KNN/GP/Diffusion) |
| 임의 | 0.65 ≤ OOF < 0.66 (partial) | plan-013 corrector + 본 plan hybrid 합체 (plan-013 Candidate C 변형) — plan-015 default |

### §1.5 본 plan 의 정직성 원칙 (v4 reframe)

- **재실험 frame 명시**: premise (corrector 재사용 = paradigm 한계 원인) 검증 안 함, 옳다는 가정 아래 corrector paradigm 잠재력 측정.
- **F0 = frozen prior, ablation 대상 아님 (v4 정정)**: F0 산식 (plan-006 `frenet_par120_perp_neg020`) 은 본 module 안에서 재구현하되 *frozen* (학습 안 함, requires_grad 없음). v3.x 가 F0 도 learnable 로 만든 것은 narrative 위반 (= plan-012 carry 위반, "재사용 끊기" 범위 오해). v4 가 정정.
- **negative band 해석의 한계**: §Target negative band 단독 해석 불가 — plan-013 join 필수 (§1.4).
- **컴포넌트별 attribution 회피**: 4 컴포넌트 (C1~C4) 동시 fixed baseline — 본 plan 의 outcome 으로는 어느 컴포넌트가 결정적이었는지 알 수 없음. F0 frozen vs learnable 비교 = plan-015 후속 attribution.
- **plan-012 ablation 재실행 (v4 reframe)**: plan-012 의 8 ablation lever + 3 codebook bake-off 를 *corrector 재사용 끊은 baseline 위에서* 다시 측정. 재사용 환경 위 marginal 이었던 lever 들이 재사용 끊은 환경에서 살아 있는지 측정.
- **외부 reference 정책**: plan-012 result.md = INVALID_REFERENCE 박제 (fd64f6c) → measured 값 reference 없음. F0 raw 0.6320 (plan-006 hard evidence) + plan-013 G1 fallback 0.6381 (ref-only) 만 외부 reference.
- **참조 범위 = 3가지만**: (a) input feature 가공 방식, (b) F0 64%/84% sample cover 입증, (c) F0 단일 공식 산식 (= frozen prior). 그 외 = 새 module 안 from-scratch.

---

## §2. Scope (명시적)

### §2.1 In-scope (= Baseline 고정 + Ablation lever)

#### A. Baseline (4 컴포넌트 fixed, 모든 ablation 의 기준)

| 항목 | 값 |
|---|---|
| paradigm | codebook + classifier + regression hybrid (corrector 영역) |
| K | 7 (G3.E2 ablation 시 5/9/13 sub-exp) |
| **C1 corrector (encoder + heads)** | 새 module-local 2-layer BiGRU (input=9, hidden=128, bidirectional=True, dropout=0.1 between layers). forward output reduction = **last-step bidirectional concat**: GRU output `(B, T=6, 2*hidden=256)` 의 마지막 step `output[:, -1, :]` → `(B, 256)`. shared encoder + 2 head. Classifier head: `Linear(256, 7)` → 7 logit. Regression head: `Linear(256, 7*3)` → reshape `(B, 7, 3)` → `tanh × 0.005` (bound ±0.005m). **from-scratch, learnable** (= corrector 정밀화 main lever) |
| **C2 F0** | plan-006 `frenet_par120_perp_neg020` 산식 본 module 안 재구현. **frozen constants** d1=1.98 / par=1.20 / perp=−0.20 (= plan-006 hard evidence carry). 학습 안 함 — nn.Buffer 또는 plain numpy/torch function. 산식 = `F0 = p0 + 1.98·v_last + 1.20·acc_par_vec + (−0.20)·acc_perp_vec` where v_last/acc/acc_par_vec/acc_perp_vec = §A.1 Frenet finite-diff (horizon=2, time_scale=1 → v_scale=acc_scale=1). d2=0 / jerk=0 fixed (plan-006 default). p0 = X[:, end_idx] (last observed point). `acc_par_vec` = `(acc · t̂) · t̂` (acc 의 t̂ 성분 vector projection), `acc_perp_vec` = `acc − acc_par_vec`. 산식 reference = plan-012 `ring_classifier.py:512-565` (carry, import X) |
| **C3 anchor radius** | 0.01m fixed scalar |
| **C4 soft label** | Gaussian σ=0.01m, `w_k ∝ exp(−d_k² / (2σ²))`, normalized over k=0..6. d_k = `‖y_true − (F0 + a_k_world)‖₂` |
| Loss | `L = α × CE(logits, soft_label) + β × Huber(reg_offset, residual_k)`, (α=β=1.0). residual_k = `y_true − F0 − a_k_world`. Huber δ = 0.005m. **Batch reduction = `mean` over batch** (= 전체 sample loss 의 평균, `reduction="mean"`). — G3.E4 swap 의 base |
| Inference | soft blend, τ=0.03 — G3.E3 τ scan 의 base. `hybrid_pred = F0 + Σ_k prob_k × (a_k_world + reg_offset_k)`, prob_k = softmax(logits / τ) |
| Input pipeline | shape `(N, 6, 9)` 시계열. 6 step indices = `range(max(3, end_idx-5), end_idx+1)` (pad first if <6 → indices[0] 반복 prepend). per-step 9 dim: 8 dim = step-local finite-diff 위 `turn_features` ((1) speed (2) prev_speed/speed (3) acc_norm/speed (4) acc_par_scalar/speed (5) perp_norm/speed (6) jerk_norm/speed (7) turn_cos (8) **curvature placeholder = `perp_norm/(speed+ε)` — feature (5) 와 ε 한 항만 다른 *의도된 중복* (plan-004 `selector.py:280-294 + 406-449` carry naming, 형식 reuse 차원 의도된 redundancy. 표준 curvature κ=‖a_perp‖/‖v‖² 와는 산식 다름)**) + (9) direction = const 1.0 (per-sample broadcast `(N, 1)` per step, plan-004 mirror augmentation infra carry placeholder, plan-014 의미 신호 0). 6 step concat → `(N, 6, 9)`. source-of-truth = `src/pb_0_6822/selector.py:280-294 + 406-449` (형식만 reuse, import X, 본 module 안 재구현) |
| Validation | 5-fold OOF, fold = `stable_hash_fold(sample_id, salt='plan-014-v1')`: SHA256(f"{salt}::{sample_id}") → int.from_bytes([:8]) % 5 (새 module 내 재구현) |
| Training | Adam lr=1e-3, batch=256, epochs=50, **early stopping**: `monitor = val_hit@1cm` (ascending), `patience = 5`, `best_epoch = argmax val_hit`. F0 frozen 이므로 optimizer 의 param_set = corrector params 만. seed=20260514, **device=cuda** (plan-012 c18 ff1e578 GPU rerun 인프라 재사용) |
| Multi-seed | single seed (= 20260514). 분산 측정은 plan-015 후보 |
| New module | `src/pb_0_6822/plan014_paradigm.py` (`selector` / `ring_classifier` / `boundary` / plan-006 numpy F0 함수 import 0) |

##### A.1 Finite-difference + Frenet basis (본 module 안 자족 재구현 spec, 외부 import 0)

dt = 40ms (= §2.1.C 의 timestep grid step). 본 plan 은 **시간 단위 무차원화 (time_scale=1)** 채택 — 모든 미분 결과는 m/step 단위 (m/s 환산 없음). F0 산식의 1.98 / 1.20 / −0.20 모두 **dimensionless hard-coded constants** (plan-006 carry): dt scaling 이 이미 계수 안에 흡수됨 (plan-006 가 train data 위 grid-search 로 산출한 값). 즉 v_last 는 m/step, acc 는 m/step², F0 산식 좌변/우변 단위 일관 (모두 m). horizon=2 = T_TARGET_MS/dt = 80/40 = 2 step → d1≈1.98 ≈ 2 step (horizon-2 prior + 평균 감속 효과로 1.98).

- `v[s] = p[s] − p[s−1]` (per-step velocity, m/step, shape (N, 3))
- `v_last = v[end_idx]`, `v_prev = v[end_idx−1]`
- `acc = v[end_idx] − v[end_idx−1] = p[10] − 2·p[9] + p[8]`
- `prev_acc = v[end_idx−1] − v[end_idx−2]`, `jerk = acc − prev_acc`

Frenet basis @ end_idx (sample-별 (3, 3) orthonormal matrix R_world_from_frenet = `[t̂ | n̂ | b̂]` columns):

- `t̂ = v_last / (‖v_last‖ + ε)` — unit tangent
- `acc_par_scalar = acc · t̂`, `acc_par_vec = acc_par_scalar · t̂`
- `acc_perp_vec = acc − acc_par_vec`
- `n̂ = acc_perp_vec / (‖acc_perp_vec‖ + ε)` — degenerate fallback `‖acc_perp_vec‖ < ε_basis=1e-6` 시 `n̂ = world ẑ` post-orthogonalize (n̂ ← n̂ − (n̂·t̂)·t̂, 재정규화)
- `b̂ = t̂ × n̂`
- ε = 1e-12, ε_basis = 1e-6

→ `build_frenet_basis_3d(trajectory_x: np.ndarray, end_idx: int = 10) -> np.ndarray` returns (N, 3, 3). 산식 출처 reference = plan-012 `ring_classifier.py:136~` (carry, import X).

##### A.2 Soft label + Huber + hybrid_pred (frozen F0 위에서)

- F0 는 **frozen** — torch.no_grad context 또는 plain numpy. 본 module 에서는 `F0_pred(X)` 함수 (학습 불가) 또는 nn.Module 의 buffer 로 store. corrector forward 시 F0_pred 는 detach() 상태로 입력 (gradient 안 흐름).
- `d_k = ‖y_true − (F0_pred + a_k_world)‖₂` (per-sample, per-anchor). F0_pred = frozen 산출 (gradient 없음).
- `w_k ∝ exp(−d_k² / (2σ²))`, σ=0.01m, normalize over k.
- `residual_k = y_true − F0_pred − a_k_world` (gradient 안 흐름 in F0).
- Huber δ = 0.005m, `Huber_offset = Σ_k w_k · huber(reg_offset_k − residual_k, δ)`.
- `hybrid_pred = F0_pred + Σ_k prob_k · (a_k_world + reg_offset_k)`. F0_pred gradient X (frozen).
- a_k_world = R_wfn @ a_k_local (Frenet anchor 의 경우, sample-별 회전), 또는 a_k (Absolute).

#### B. Ablation lever (plan-012 5-Phase 그대로 carry, baseline 위에서 single-variable swap)

| Stage | lever | variants | base |
|---|---|---|---|
| **G2.Phase 1** | **E0 codebook bake-off** ★ | E0a Absolute-7Way (world ±x/±y/±z + center) / E0b Frenet-Orthogonal-7Way (±t̂/±n̂/±b̂ + center) / E0c K-Means-7Way (Frenet residual cluster) | 동일 corrector arch + loss + τ + seed, F0 frozen, 유일 변수 = anchor 좌표 집합 |
| **G3.Phase 2** | **E1 frame swap** (conditional) | Frenet vs world | winner ∈ {E0b, E0c} 만, E0a winner 면 SKIP |
| **G3.Phase 2** | **E2 K density** | K=5 / 7 / 9 / 13 | winner codebook |
| **G3.Phase 2** | **E3 τ scan** | argmax + τ ∈ {0.01, 0.03, 0.1, 0.3, 1.0} | inference-time hyperparam |
| **G3.Phase 2** | **E4 loss swap** | L7 hinge vs distance regression | baseline CE soft + Huber 의 cls loss form swap |
| **G3.Phase 2** | **E5 reg head on/off** | cls only / cls+reg hybrid | reg head 사용 여부 |
| **G4.Phase 3** | **E6 boundary weight** | on/off | boundary sample weighting |
| **G4.Phase 3** | **E7 scorer arch** | full BiGRU vs last-step MLP | C1 encoder variant |
| **G4.Phase 3** | **E8 r=0 logit prior** | 0 / +0.5 / +1.0 | center mode logit bias |

→ 총 11 ablation sub-experiment (E0 3-way + E1~E5 5 axis + E6~E8 3 axis). G5 에서 winner + best lever stack 으로 final 5-fold. **모든 lever 는 corrector 영역 만 swap — F0 frozen 위에서 측정**.

#### B.1 Ablation lever source-of-truth + plan-014 baseline 위 적용

각 lever 의 source line (plan-012 ring_classifier.py / phase3_aux.py) + plan-014 baseline 위 적용 방식:

| lever | source-of-truth | plan-014 baseline 위 적용 |
|---|---|---|
| E0a Absolute | `ring_classifier.py:39-54` `compute_anchors_absolute(radius_m=0.005)` — (7, 3) world frame ±x/±y/±z + center | `radius_m=0.01` (plan-014 C3) |
| E0b Frenet | `ring_classifier.py:57-62` `compute_anchors_frenet_orthogonal(radius_m=0.005)`. 좌표 형식 = E0a 동일, basis 회전은 caller (`R_wfn @ anchor_local`) | `radius_m=0.01`, basis = `build_frenet_basis_3d(trajectory_x, end_idx=10)` (`ring_classifier.py:136~`) — 산식 본 module 안 재구현 |
| E0c K-Means | `ring_classifier.py:65-128` `compute_anchors_kmeans(train_residuals_world, R_world_from_frenet, fold_id, K=7, radius_clip_m=0.020, n_init=10, random_state=20260606)`. fold-aware, K−1 cluster + center. anchor index convention: k=0=center (prepend), k=1..K−1=sklearn cluster output 순서 | train_residuals = `y_true − F0_pred_frozen` (F0 frozen 산출). `radius_clip_m=0.020` (= plan-012 그대로) |
| E1 frame swap (conditional) | (winner ∈ {E0b, E0c} 만) world vs Frenet | winner=E0b 시 anchor coord를 world 좌표로 *해석만 변경* (E0a 와 동치). winner=E0c 시 train_residuals_world 자체가 world → inverse rotation 불필요, K-Means centroid 자체를 world anchor. winner=E0a 면 SKIP (= frame_axis_n/a) |
| E2 K density | `compute_anchors_*(K=5/9/13)`. K=5/9/13 anchor 공식 inline 박제 (= 모든 anchor 의 ‖·‖ = 0.01m): **K=5** = `[center, +dom, −dom, +second, −second]`. **K=9** = K=7 + `±(dom + second)/√2`. **K=13** = K=7 + 6 unique vector (3 axis-pair × 2 sign, `±(a+b)/√2` for each pair ∈ {(dom,second), (dom,third), (second,third)}). **norm 0.01m 보장 가정**: dom/second/third 는 unit-axis 직교 vector (Frenet ortho t̂/n̂/b̂ 또는 world x/y/z 단위벡터 × 0.01m). 직교 `dom·second = 0` → `‖dom + second‖ = √(0.01² + 0.01²) = 0.01√2`, `/√2` 로 정확히 norm 0.01m. K-Means winner 시 본 공식 미적용 (cluster centroid 비직교, 산식 위반). dom/second/third = G0 task (e) axis_family_ranking top-1/2/3. **frame 선택 룰**: winner codebook 에 따라 — E0a Absolute winner 시 `axis_family_ranking_absolute` ∈ {x, y, z} (world frame), E0b Frenet-ortho winner 시 `axis_family_ranking_frenet` ∈ {t, n, b} (Frenet local + R_wfn 회전), E0c K-Means winner 시 본 E2 공식 미사용 — K-Means 재fit (frame 무관, cluster 위치 자체 변동) | winner codebook 의 K 변형. E0c 의 경우 K-Means 재fit per fold (radius_clip=0.020, random_state=20260606 carry). E2 의 단일 변수 = "anchor 수 K" 이지만 K-Means 의 경우 cluster 위치도 K 와 함께 변동 (의도된 coupling, plan-012 carry) |
| E3 τ scan | inference time `temperature` 변경. variants: argmax (τ≤1e-8) + {0.01, 0.03, 0.1, 0.3, 1.0} | 학습 = baseline τ=0.03, eval 만 변경 (same model checkpoint) |
| E4 loss swap | `ring_classifier.py:410-454` `hybrid_combined_loss(use_hinge)`. L7 hinge = `ring_classifier.py:380-389` `hit_aware_hinge(corrected_pos, target, R_HIT=0.01, smooth=0.005)` — `(softplus(excess / smooth) · smooth)²` where `excess = ‖pred − target‖ − R_HIT` | sub-exp A: baseline `L = α·CE(logits, w_k) + β·Huber_offset`. sub-exp B: `L = α·CE + 0.5·Huber_offset + 0.5·Hinge_pred` (domain 다른 두 항 weighted sum, Huber_offset = per-anchor offset domain, Hinge_pred = sample-level final position) |
| E5 reg head on/off | `hybrid_combined_loss(use_reg_head)` flag | off variant: reg_offset 항 무시, `hybrid_pred = F0 + anchor_blend` 만 |
| E6 boundary weight | `phase3_aux.py:57-61`: `boundary_mask = (err_F0 > 0.005) & (err_F0 < 0.015)`, `sw = where(mask, 3.0, 1.0)`. err_F0 = `‖F0_pred_frozen − y_true‖` (frozen F0 산출) | loss batch reduction 에 sample weight 곱셈 = `weighted_mean = Σ_i (w_i × loss_i) / Σ_i w_i` (분모는 Σw, 단순 N 아님 — plan-012 convention carry). on/off 2 sub-exp |
| E7 scorer arch | `ring_classifier.py:342-372` `LastStepMLPScorer(seq_dim=9, cand_dim=11, hidden=64, cand_count=7)` — GRU 우회, last-step seq → 2-layer GELU MLP | plan-014 baseline (BiGRU h=128) vs LastStep MLP variant. cand_feat = anchor coord (B, K, 3) — plan-014 의 K=7 (not 11). seq[:, -1, :] (last step 9 dim) → `MLP_seq` (2-layer GELU, 9→64) → `h`. anchor → `MLP_cand` (2-layer GELU, 3→64) → `cand_h`. logits = `(cand_h * h[:, None, :]).sum(-1)` → (B, K). seq MLP 와 cand MLP 별도 weight |
| E8 r=0 logit prior | `ring_classifier.py:464-490` `hybrid_predict(r0_logit_prior=0.0/0.5/1.0)`. `prior[0] = r0_logit_prior` (center mode k=0 만 bias) | inference 시만 적용 (학습은 baseline 동일). variants: 0 / +0.5 / +1.0 |

decision-note: E0c K-Means 의 `random_state=20260606` 은 plan-012 그대로 carry (= reproducibility). 본 plan 의 seed (= 20260514) 와 별개 — K-Means 의 init 결정에만 영향.

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
| Submission output | `runs/baseline/<exp_id>/submission.csv` — columns `id` / `x` / `y` / `z`, id order = `data/sample_submission.csv` 의 id column, precision = `f"{val:.6f}"` (6 decimals), float64 dtype, NaN/Inf 금지. source = `src/submit.py:204-231` `write_submission(run_dir, pred, test_ids)` (utility, import OK or 직접 재구현) |

### §2.2 Out-of-scope

| 항목 | 이유 |
|---|---|
| plan-004 corrector 모듈·weight 재사용 (`selector.py`, `CandidateAttentionGRUSelector`, plan-004 weight) | 본 plan 의 "재사용 끊기" 의 주 대상 — premise 검증 위해 끊음 |
| plan-012 corrector 모듈 재사용 (`ring_classifier.py`) | 동일 (corrector 재사용 끊기) |
| plan-006 numpy F0 *함수 import* (`f0_predict_frenet_par120_perp_neg020`) | F0 산식 자체는 carry (재구현) 이나, *함수 import* 는 의도적 끊음 — 본 module 자족성 보존 |
| F0 자체의 learnable variant / F0 attribution / F0 학습 가능성 | baseline 의 일부 (frozen), ablation 대상 아님. plan-015 후속 |
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
- **G2/G3/G4 sub-exp**: G2 = 5-fold OOF (각 sub-exp 5 fold 학습). G3/G4 = fold=0 single-fold (빠른 iteration). 가정: single-fold ΔOOF sign 이 5-fold concat ΔOOF sign 과 일관. 미align 시 G5 `g5_no_additive` warn 으로 anchor fallback (§1 outcome 인과 chain 끊김 X). **sub-exp best_stack 제외 결정 룰**: G5 의 best_stack 5-fold 학습 시 anchor_5fold + best_lever_5fold 차이가 +0.005 미만이면 → 해당 best_lever 가 single→5-fold sign-reversal 한 후보. 이 경우 G5 의 `g5_no_additive` warn 박제 + submission = anchor 5-fold (= best_lever 제외 fallback). G3/G4 의 모든 positive_axes 중 best_stack 진입은 max ΔOOF 1개만 (Phase 2 best + Phase 3 best, §9.1) — 다른 marginal positive 는 candidate 에서 자동 제외.
- **G5 final**: 5-fold concat (모든 sample 이 정확히 1번 val 등장)
- decision-note: plan-004 `stable_fold_id` 와 분할 다를 수 있음 — plan-014 measurement 는 plan-014 scheme 내 self-consistent (외부 비교 = §3.5 reference only)

### §3.2 평가 metric

- **main metric (G5 band 판정)**: `best_stack_5fold_hit_1cm = mean(‖hybrid_pred − y_true‖₂ ≤ 0.01m)` (5-fold concat OOF)
- 두 분포의 역할 분리 (cross-section sync):
  - `w_k` (= **soft label target**, §2.1.A.2 박제) — Gaussian kernel 정규화. **학습 loss CE target** 으로만 사용
  - `prob_k` (= **model output blend weight**) — softmax over model logits, τ=0.03. **inference 시 hybrid_pred 의 anchor blending** 에만 사용
- `hybrid_pred = F0_pred + Σ_{k=0..K−1} prob_k × (a_k_world + reg_offset_k)` — F0_pred = frozen (gradient 없음). anchor 는 world frame 변환 후 합산 (Frenet local anchor 의 경우 `a_k_world = R_wfn @ a_k_local`). soft label 거리 + Huber residual 정의 = §2.1.A.2 박제
- **F0 산식 reference**: `F0_pred = p0 + 1.98·v_last + 1.20·acc_par_vec + (−0.20)·acc_perp_vec` (constants, §2.1.A.1 박제, source = `ring_classifier.py:512-565`)
- **secondary**: `hit@1.5cm = mean(‖hybrid_pred − y_true‖₂ ≤ 0.015m)` (5-fold concat OOF, 1cm 와 대칭 정의)
- **diagnostic**: `directional_commit_magnitude (DCM) = mean(‖hybrid_pred − F0_pred‖₂)` — corrector 신호 살아있는지 측정 (G2 criterion)
- **band classifier** (§0.5 박제):
  - best stack 5-fold OOF ≥ 0.66 → **positive** (corrector paradigm 부활)
  - 0.65 ≤ OOF < 0.66 → **partial** 회복
  - OOF < 0.65 → **negative** (premise 의심)

### §3.3 The Configs

#### Baseline config (G2 의 base, 모든 ablation 의 zero-modification reference)

§2.1.A 박제 그대로. = F0 frozen (plan-006 frenet_par120_perp_neg020) + 9d×6step input → BiGRU(128) → cls(7) + reg(7×3) → soft blend τ=0.03 → CE soft + Huber loss. **F0 는 optimizer param_set 에서 제외** (corrector params 만).

#### Ablation variants (G2.E0 / G3.E1~E5 / G4.E6~E8)

각 lever 별 sub-exp = baseline 위에서 단 1 변수 swap. F0 frozen 은 모든 sub-exp 공통. 상세 spec = §6~§9 STAGE 별.

### §3.4 G-gate quantitative criteria

#### G0 — preflight artifact

- artifact: `analysis/plan-014/preflight.json`
- (a) F0 frozen reproduce: d1=1.98 / par=1.20 / perp=−0.20 constants 으로 모든 train sample hit@1cm 측정 → plan-006 reference (0.6320) ± 0.005 일치. 산식 = §2.1.A.1 + `ring_classifier.py:512-565` 그대로 (numpy 함수 import X — 새 module 안에서 동일 산식 직접 재구현)
- (b) anchor scale 박제: radius=0.01m, ±t̂/±n̂/±b̂/center 7 anchor Frenet local coord
- (c) soft label entropy: σ=0.01m Gaussian → sample-별 entropy 평균 ≥ 0.5 nat
- (d) plan-012 disclaimer verify: `INVALID_REFERENCE` token + `disclaimer:` field 박제 grep
- (e) per-axis marginal oracle ordering (= §7.1 E2 K=5/9/13 anchor source): 각 ±axis 의 2-anchor codebook `[center, ±axis_vector_0.01m]` 의 oracle hit@1cm (hindsight argmin). axis_family_ranking by max(+sign, −sign), tie-break priority `x>y>z` / `t>n>b`
- fail trigger: (a)~(e) 중 1+ 누락 → `preflight_artifact_missing` severe
- **artifact schema** (`preflight.json`): `exp_id` / `n_train` `trajectory_T` `end_idx` / `f0_raw_hit_measure` (dict: `single_formula`, `hit_at_1cm`, `hit_at_1_5cm`) / `codebook_oracle_ceilings` (dict per E0a/E0b/E0c: `oracle_hit_1cm` = hindsight label-aware oracle, `anchors`) / `per_axis_marginal_oracle` / `kmeans_fit_meta` (K=7, centers/sizes/inertia/silhouette per fold, `min_cluster_size_threshold=100`, `min_cluster_size_pass` bool) / `g0_checks` (4 bool: f0_reproduce / anchor_scale / soft_entropy / disclaimer) / `g0_essential_passed` bool

#### G1 — 새 module 구현 + 재사용 끊김

- artifact: `src/pb_0_6822/plan014_paradigm.py` + smoke test (`tests/test_plan014_smoke.py`)
- (a) smoke train: 1-fold 1-epoch — no NaN, `val_hit_after_epoch_1 >= initial_val_hit − 0.05` (= random-init variance 흡수 margin, false-positive smoke fail 방지). 상세 spec = §5.4 (a) (initial_val_hit / val_hit_after 산출 path 박제)
- (b) 재사용 끊김 4가지:
  1. AST import 0: `selector` / `ring_classifier` / `boundary` / plan-006 numpy F0 함수 import 0 (= `f0_predict_frenet_par120_perp_neg020` 류)
  2. **F0 함수 본 module 안 재구현 verify**: F0 forward 가 (1.98, 1.20, −0.20) constants 으로 reproduce, `F0 params 의 requires_grad = False` (= frozen, optimizer 가 학습 안 함). F0 산식 reproduce hit@1cm ∈ [0.6270, 0.6370] (G0 (a) carry)
  3. anchor `‖a_k‖ = 0.01 ± 1e-6` for E0a/E0b (E0c: anchor[0] = center origin)
  4. soft label entropy ≥ 0.5 nat
- fail trigger: 1+ fail → `reuse_cut_violation` severe (premise 위배)

#### G2 — Phase 1 codebook bake-off

- artifact: `analysis/plan-014/g2_phase1_bakeoff.py` + `g2_phase1.json` + 3 sub-exp `runs/baseline/plan014_g2_E0{a,b,c}/`
- spec: E0a/E0b/E0c 3 sub-exp 5-fold OOF hit@1cm 측정 (동일 corrector arch + loss + τ + seed, F0 frozen)
- winner: `argmax(OOF over E0a, E0b, E0c)`. tie-break (gap < 0.005): 단순성 우선 E0a > E0b > E0c
- criterion: winner_OOF ≥ **0.60** + winner DCM ≥ **0.002**
- fail trigger:
  - winner_OOF < 0.60 → `g2_severe_underperform` severe (autonomous path-pivot)
  - winner DCM < 0.002 → `dcm_collapse` warn (Phase 2~4 informational 진행)
- **artifact schema** (`g2_phase1.json`): plan-012 phase1_winner.json carry + `winner_id` / `winner_codebook` / `winner_frame` / `winner_K` / `winner_oof` / `winner_dcm` / `all_sub_exp_oof` / `directional_commit_magnitudes` / `G2_passed` / `G2_warn` / `tie_break_applied` / `results_per_sub_exp` (per-fold log).

#### G3 — Phase 2 axis ablation 5

- artifact: `analysis/plan-014/g3_phase2_axis.py` + `g3_phase2.json` + sub-exp runs
- spec: 5 axis (E1~E5) winner codebook 위, fold=0 single-fold, ΔOOF = OOF_variant − OOF_anchor
- criterion: 1+ axis 의 max(ΔOOF) ≥ **+0.005** → `G3_passed = true`
- fail trigger: 모든 axis ΔOOF < 0.005 → `g3_marginal_only` warn (Phase 3~4 informational, G5 anchor fallback 후보)
- **artifact schema** (`g3_phase2.json`): `winner_id` / `anchor_oof` / `axis_summary` (per E1~E5) / `positive_axes` / `G3_passed` / `G3_warn` / `results_per_sub_exp`

#### G4 — Phase 3 aux ablation 3

- artifact: `analysis/plan-014/g4_phase3_aux.py` + `g4_phase3.json`
- spec: 3 axis (E6/E7/E8) winner config 위, fold=0 single-fold
- criterion: informational only
- **artifact schema**: G3 schema 동일 (axis = E6/E7/E8)

#### G5 — Phase 4 final 5-fold + best stack + submission

- artifact: `analysis/plan-014/g5_phase4_final.py` + `g5_phase4.json` + `runs/baseline/plan014_g5_phase4/submission.csv`
- spec: G2 winner + G3/G4 best lever stack 으로 5-fold concat OOF + submission
- best_stack 정의: anchor config + Phase 2 best lever (E1~E5 중 max ΔOOF 단일 axis, ΔOOF > 0 인 경우) + Phase 3 best lever (E6~E8 중 max ΔOOF 단일 axis, ΔOOF > 0 인 경우) = 최대 3 elements stack. §9.1 stacking rule sync.
- criterion: **best_stack 5-fold OOF ≥ anchor_5fold + 0.005**
- band 분류 (§3.2): best_stack OOF 기준 ≥0.66 / 0.65~0.66 / <0.65
- fail trigger: best_stack < anchor + 0.005 → `g5_no_additive` warn (anchor fallback submission)
- **artifact schema** (`g5_phase4.json`): `config_anchor` `config_best` (lever key 전부: codebook/K/frame/temperature/use_reg_head/use_hinge/boundary_weight_on/scorer_arch/r0_logit_prior. F0 frozen 항상 동일) / `anchor_5fold_oof_hit_1cm` `best_5fold_oof_hit_1cm` `delta_oof` / `G5_passed` / `band` / `fold_results` (dict per fold: train_hit/val_hit/dcm/best_epoch) / **submission keys** (4 explicit): `submission_best_path` (str, `runs/baseline/plan014_g5_phase4/submission_best.csv`) / `submission_anchor_fallback_path` (str, `..._anchor_fallback.csv`) / `submission_used_for_LB` (str, = `submission_best_path` if G5_passed else `submission_anchor_fallback_path`) / `submission_n_rows` (int, = `len(test_ids)`)

#### G_final — synthesis

- artifact: `plans/plan-014-plan012-failure-inversion.results.md` 신규 + `registry.csv` append + plan-014 frontmatter sync (`lb_score` / `exp_ids` / `status: spec → completed`)
- content: G0~G5 결과 narrative + band 분류 + plan-013 join interpretation activated row + plan-015 후보 ≥ 3
- fail trigger: 3 파일 sync 누락 → `final_sync_missing` severe
- **registry.csv schema** (12 columns): id / plan_id / type / status / started_at / finished_at / duration_sec / run_dir / config_path / baseline_id / corrects / notes
- **registry append spec** (per G-stage 6 row): 각 row 의 column 값 룰:
  - `id` = `H036_g0_preflight` / `H037_g1_module_smoke` / `H038_g2_phase1_bakeoff` / `H039_g3_phase2_axis5` / `H040_g4_phase3_aux3` / `H041_g5_phase4_final`
  - `plan_id` = `014`
  - `type` = `baseline` (모든 row)
  - `status` = `complete` (G-stage pass) or `deferred` (g2_severe_underperform 시 G3~G5)
  - `started_at` / `finished_at` = ISO 8601 KST (`+09:00`)
  - `duration_sec` = float (실행 시간)
  - `run_dir` = `analysis/plan-014` (G0/G1) or `runs/baseline` (G2~G5)
  - `config_path` = script path: `analysis/plan-014/preflight.py` (G0) / `src/pb_0_6822/plan014_paradigm.py` (G1) / `analysis/plan-014/g2_phase1_bakeoff.py` (G2) / `analysis/plan-014/g3_phase2_axis.py` (G3) / `analysis/plan-014/g4_phase3_aux.py` (G4) / `analysis/plan-014/g5_phase4_final.py` (G5)
  - `baseline_id` = 직전 G-stage row id (예: G1.baseline_id = `H036_g0_preflight`, G2.baseline_id = `H037_g1_module_smoke`, ...). G0.baseline_id = "" (empty, 시작)
  - `corrects` = "" (보통 공란, paradigm-shift 안 함)
  - `notes` = 한 줄 metric summary (예: G2 row = `winner=E0c, OOF=X.XXXX, DCM=X.XXXX, G2_passed=True`)

### §3.5 External reference

| measure | plan-006 / plan-013 (외부 ref) | plan-014 target |
|---|---|---|
| F0 raw hit@1cm | 0.6320 (plan-006 hard evidence — §1.5 참조 (b)/(c)) | G0 (a) reproduce (±0.005) |
| 5-fold OOF hit@1cm | plan-013 G1 fallback 0.6381 (paradigm 폐기 path best, ref-only) | G5 best_stack band 측정 (§3.2) |
| G2 winner OOF | — | ≥ 0.60 + DCM ≥ 0.002 |
| Phase 2 axis ΔOOF | — | G3 1+ axis ≥ +0.005 |
| best_stack vs anchor | — | G5 ≥ +0.005 |

decision-note: plan-012 measured 값 (0.6350 / 0.6411) 은 INVALID_REFERENCE 박제 후 reference 제거. F0 raw 0.6320 는 plan-006 F0 공식 자체의 hard evidence — reference 정당.

---

## §4. STAGE 0 (c4, G0) — preflight artifact [TODO]

### §4.1 산출물

- `analysis/plan-014/preflight.py` — 5 task 일괄 실행 (v4):
  - (a) **F0 frozen reproduce** (d1=1.98 / par=1.20 / perp=−0.20 constants 으로 전체 train sample hit@1cm 측정) → plan-006 reference 0.6320 ± 0.005 일치
  - (b) 3 codebook oracle ceiling 측정 (E0a Absolute / E0b Frenet-Orthogonal / E0c K-Means, radius=0.01m fixed). oracle 정의: `oracle_hit_1cm = mean(‖F0 + anchor[argmin_k ‖F0+anchor[k]−y‖] − y‖ ≤ 0.01)` (hindsight label-aware)
  - (c) Gaussian σ=0.01m soft label entropy 평균 측정 (target w_k 분포의 entropy, 학습 전 분석적 산출)
  - (d) plan-012 results.md INVALID_REFERENCE disclaimer grep 검증
  - (e) per-axis marginal oracle ordering: 각 ±axis 의 2-anchor codebook `[center, ±axis_vector_0.01m]` oracle hit@1cm. axis_family_ranking by max(+sign, −sign), tie-break priority `x>y>z` / `t>n>b`
- `analysis/plan-014/preflight.json` — schema = §3.4 G0 박제
- registry row: `H036_g0_preflight`

### §4.2 실행

```bash
python analysis/plan-014/preflight.py \
  --root            data \
  --out             analysis/plan-014/preflight.json \
  --plan-012-ref    plans/plan-012-frenet-ring-classification.results.md
```

`src/io.py` (plan-001 utility) import. F0 산식 / Frenet basis / anchor / K-Means 본 스크립트 안 재구현 (= plan-004/006/012 module import 0).

### §4.3 G0 합격 (§3.4 carry)

- (a) F0 frozen reproduce hit@1cm ∈ [0.6270, 0.6370]
- (b) anchor scale 3 codebook 의 ‖non-center anchor‖ = 0.01m ± 1e-6
- (c) soft entropy 평균 ≥ 0.5 nat (target w_k 분포)
- (d) plan-012 disclaimer grep: `INVALID_REFERENCE` + `disclaimer:` 양쪽 hit
- `g0_checks` 4 bool 모두 true → `g0_essential_passed = true`

### §4.4 decision-note 후보

- K-Means random_state = 20260606 (plan-012 carry)
- E0c oracle 측정 시 `radius_clip_m = 0.020` (Frenet residual scale)

---

## §5. STAGE 1 (c5, G1) — `plan014_paradigm.py` 새 module + smoke + 재사용 끊김 [TODO]

### §5.1 산출물

- `src/pb_0_6822/plan014_paradigm.py` — 새 module, 외부 import = `torch`, `torch.nn`, `numpy`, `sklearn`, `src.io` 만. `src.pb_0_6822.{selector,ring_classifier,boundary}` + plan-006 numpy F0 함수 import 0
- `tests/test_plan014_smoke.py` — pytest 1-fold 1-epoch smoke + 재사용 끊김 4 assert
- registry row: `H037_g1_module_smoke`

### §5.2 module 구조 (§2.1.A baseline + §2.1.B.1 lever interface)

```python
# src/pb_0_6822/plan014_paradigm.py
class Plan014BiGRUEncoder(nn.Module):
    """2-layer BiGRU (input=9, hidden=128) shared encoder. forward(seq: (B, 6, 9)) → (B, 256)."""

class Plan014F0Function:  # NOT nn.Module — frozen, no nn.Parameter
    """plan-006 frenet_par120_perp_neg020 산식 본 module 안 재구현.
    constants: d1=1.98, par=1.20, perp=-0.20.
    __call__(X_raw: (B, T, 3), end_idx=10) → F0_pred (B, 3). gradient 없음 (numpy 또는 torch with no_grad).
    """

class Plan014HybridHead(nn.Module):
    """Encoder + cls head (Linear → K) + reg head (Linear → K*3, tanh × 0.005).

    Two-method design (분리 의도):
      - forward(seq, anchors): raw logits + reg_offset 산출. training loop 가 호출.
        F0_pred 미사용 (loss 가 F0_pred + anchors_world 받아 residual_k 계산).
      - hybrid_predict(seq, anchors, R_wfn, F0_pred_detached, temperature, use_reg_head,
        r0_logit_prior): 후처리 — softmax + anchor blending + F0 합산 → final pred.
        inference + val 계산 시 호출.

    F0_pred detach() 책임 = **caller (train_one_fold)**. caller 가 `F0_pred = F0_function(X_raw).detach()`
    호출 후 (또는 plain function 산출 시 자연스럽게 grad 없음) Head 에 전달. Head 안에서는 detach
    재호출 안 함 — caller-side detach 가 단일 source-of-truth.

    forward(seq: (B, 6, 9), anchors: (K, 3)) → (logits: (B, K), reg_offset: (B, K, 3))
    hybrid_predict(seq: (B, 6, 9), anchors: (K, 3), R_wfn: (B, 3, 3) | None,
                   F0_pred_detached: (B, 3), temperature: float, use_reg_head: bool,
                   r0_logit_prior: float) → pred (B, 3)
    """

# anchor 함수 (numpy)
def compute_anchors_absolute(radius_m=0.01) -> np.ndarray:  # (7, 3)
def compute_anchors_frenet_orthogonal(radius_m=0.01) -> np.ndarray:  # (7, 3) Frenet local
def compute_anchors_kmeans(train_residuals_world, R_world_from_frenet, fold_id,
                            K=7, radius_clip_m=0.020, n_init=10, random_state=20260606) -> np.ndarray  # (K, 3)

# loss + train loop
def hybrid_combined_loss(logits, reg_offset, F0_pred_detached, anchors_world, target,
                          use_hinge=False, use_reg_head=True, alpha=1.0, beta=1.0,
                          temperature=0.03) -> torch.Tensor:
    """Returns scalar loss (mean over batch). batch reduction = `mean` (= 전체 sample 평균)."""

def train_one_fold(
    config: dict,                              # §5.2 run_kfold_oof config dict key set
    fold_id: int,                              # 0..N_FOLDS-1
    train_loader: DataLoader,                  # batched (X_raw, seq, anchors_or_R, target)
    val_loader: DataLoader,                    # same
    F0_function: Plan014F0Function,            # frozen, plain function
    anchors_local: np.ndarray,                 # (K, 3) — 또는 fold-aware (K, 3) for E0c
    *, seed: int = DEFAULT_SEED, device: str = "cuda"
) -> dict:
    """Returns {best_val_hit, best_val_loss, best_epoch, dcm, initial_val_hit, val_hit_per_epoch, ...}.
    F0_function 호출 결과는 caller 가 .detach() 해서 Head 에 전달."""

# K-fold OOF runner
def run_kfold_oof(config: dict) -> dict
    # config dict required keys (§3.4 G5 schema cross-ref):
    #   "name": str (= sub_exp_id)
    #   "codebook": str ∈ {"absolute", "frenet_orthogonal", "kmeans"}
    #   "K": int ∈ {5, 7, 9, 13}
    #   "frame": str ∈ {"world", "frenet"}  (E1 frame swap; default = winner frame)
    #   "temperature": float (inference τ, default 0.03)
    #   "use_reg_head": bool (default True; E5 lever)
    #   "use_hinge": bool (default False; E4 lever)
    #   "boundary_weight_on": bool (default False; E6 lever)
    #   "scorer_arch": str ∈ {"bigru", "laststep_mlp"} (E7 lever)
    #   "r0_logit_prior": float (E8 lever)
    # 모든 sub-exp 가 동일 schema 사용, lever 별 단일 key 변경 (single-variable swap principle)
    # Returns: {oof_pred: (N, 3), fold_oof_hit_1cm: list[float], overall_oof_hit_1cm: float, ...}
```

### §5.3 실행

```bash
pytest tests/test_plan014_smoke.py -x -v
```

### §5.4 G1 합격 (§3.4 carry)

- (a) smoke train: 1-fold 1-epoch — no NaN, `val_hit_after_epoch_1 > initial_val_hit − 0.05` (= improvement 또는 minor degradation 허용; threshold 는 random-init variance 흡수용). **initial_val_hit 정의**: `model.eval()` 상태에서 학습 *전* (Adam.step() 0회) random-init weights 으로 val fold forward → `hybrid_predict(seq, anchors, R_wfn, F0_pred_detached, temperature=0.03, use_reg_head=True, r0_logit_prior=0.0)` → hit@1cm 측정. 동일 seed (=20260514) 보장. **val_hit_after 산출 path = identical** — 1 epoch 학습 후 다시 `model.eval()` + `hybrid_predict()` 동일 인자, val fold 동일 sample 위 hit@1cm 측정. 두 산출 path 가 inference path 와 동일 (= G2~G5 의 OOF 산출과 일관). assert: `val_hit_after >= val_hit_initial − 0.05` (절대 hit rate scale, false-positive 안전 margin)
- (b) 재사용 끊김 4 assert:
  1. AST import 0: `selector` / `ring_classifier` / `boundary` / plan-006 numpy F0 함수
  2. **F0 함수 본 module 안 재구현 verify**: F0_pred 산출 hit@1cm ∈ [0.6270, 0.6370] (G0 (a) carry) + F0 관련 attribute 의 `requires_grad = False` (= frozen)
  3. anchor `‖a_k‖ = 0.01 ± 1e-6` for E0a/E0b
  4. soft label `w_k = exp(−d_k² / (2 × 0.01²))` 정규화 후 평균 entropy ≥ 0.5 nat (target 분포, learning-independent)

### §5.5 decision-note 후보

- BiGRU PyTorch default init
- batch_size=256 / lr=1e-3 / Adam default
- seed = 20260514
- F0 = plain function (no nn.Parameter) — optimizer 가 F0 관련 grad 자체 없음, 자연스럽게 frozen

---

## §6. STAGE 2 (c6, G2) — Phase 1 codebook bake-off (E0a/E0b/E0c) [TODO]

### §6.1 sub-exp matrix (3 sub-exp, 모두 5-fold OOF)

| sub-exp | codebook | frame | anchor source | radius_m | K |
|---|---|---|---|---|---|
| **E0a** Absolute-7Way | world | `compute_anchors_absolute()` | 0.01 | 7 |
| **E0b** Frenet-Orthogonal-7Way | Frenet local @ F0 prediction | `compute_anchors_frenet_orthogonal()` | 0.01 | 7 |
| **E0c** K-Means-7Way | Frenet local @ F0 prediction (per-fold cluster) | `compute_anchors_kmeans(..., K=7, radius_clip_m=0.020)` | 0.020 clip | 7 |

3 sub-exp **공통 (paradigm-clean comparison)**:
- F0 frozen (plan-006 frenet_par120_perp_neg020, d1=1.98 / par=1.20 / perp=−0.20 constants) — 모든 sub-exp 동일
- arch = `Plan014HybridHead` (cls + reg head on, BiGRU encoder)
- loss = `hybrid_combined_loss(use_hinge=False, use_reg_head=True)` (CE soft + Huber)
- temperature = 0.03 / r0_logit_prior = 0 / boundary weight uniform
- optimizer = Adam(lr=1e-3) / batch=256 / epochs=50 / patience=5 (monitor=val_hit) / seed=20260514

### §6.2 산출물

- `analysis/plan-014/g2_phase1_bakeoff.py` — 3 sub-exp 5-fold OOF + winner 결정
- `analysis/plan-014/g2_phase1.json` — §3.4 G2 schema 박제
- `runs/baseline/plan014_g2_E0{a,b,c}/`
- registry row: `H038_g2_phase1_bakeoff`

### §6.3 winner 결정 + tie-break (§2.1.B carry)

```python
winners = {"E0a": oof_E0a, "E0b": oof_E0b, "E0c": oof_E0c}
winner_id = max(winners, key=winners.get)
winner_oof, second_oof = sorted(winners.values(), reverse=True)[:2]
gap = winner_oof - second_oof

if gap < 0.005:
    # tie-break: 단순성 우선 E0a > E0b > E0c
    priority = ["E0a", "E0b", "E0c"]
    tied = [k for k, v in winners.items() if v >= winner_oof - 0.005]
    winner_id = next(k for k in priority if k in tied)
```

### §6.4 G2 합격 (§3.4 carry)

- `winner_oof ≥ 0.60` + `winner_dcm ≥ 0.002`
- 위반: winner_OOF < 0.60 → `g2_severe_underperform` severe (autonomous path-pivot to negative band G_final). winner_DCM < 0.002 → `dcm_collapse` warn (Phase 2~4 informational 진행)

### §6.5 winner 박제 → Phase 2 carry

`g2_phase1.json` 의 winner config = §7~§9 의 anchor freeze. Phase 2~4 모든 sub-exp 가 이 source 만 reuse.

---

## §7. STAGE 3 (c7, G3) — Phase 2 axis ablation 5 (E1~E5) [TODO]

### §7.1 anchor 위 1-axis swap matrix (§2.1.B.1 lever spec carry)

각 sub-exp = G2 winner config 에서 *지정 axis 1개만 변경*. fold=0 (single-fold, 빠른 iteration). **F0 frozen 모든 sub-exp 공통**.

#### E1 — Frame swap (conditional)

| winner | sub-exp | 변경 |
|---|---|---|
| E0a | (skip) | frame_axis_n/a |
| E0b | **E1a (anchor)** / E1b | E1a = Frenet local (= G2 winner config, anchor reuse) / E1b = world (= 좌표 해석만 변경, E0a 와 의도적 동치). **E0b winner 시 E1 의 ΔOOF 는 E0b 와 E0a 의 OOF 차이로 거의 동일 — 별도 lever 가치 없음, informational only**. E1 은 best_stack 후보에서 *제외* (informational) |
| E0c | **E1a (anchor)** / E1b | E1a = Frenet (= G2 winner config) / E1b = world (= K-Means centroid 그대로, train_residuals_world 이미 world) |

ΔOOF(E1) = OOF(E1b) − OOF(E1a). E1a = G2 winner reuse 이므로 새 학습 0 (anchor 재실행). best_stack 채택 여부 = E0b winner 시 informational only (제외), E0c winner 시 ΔOOF > 0 면 채택 가능.

#### E2 — Codebook K density (4 sub-exp)

| sub-exp | K |
|---|---|
| E2a | 5 |
| **E2b** | 7 (anchor) |
| E2c | 9 |
| E2d | 13 |

K-Means winner 시 K=5/9/13 으로 K-Means 재fit (per-fold, `radius_clip_m=0.020`). Absolute/Frenet-ortho winner 시 §2.1.B.1 박제된 K=5/9/13 공식 적용 (dom/second/third = G0 task (e) axis_family_ranking top-1/2/3).

ΔOOF(E2) = `max(OOF over K∈{5,9,13}) − OOF(K=7)`.

#### E3 — Temperature scan (6 sub-exp, inference-only)

| τ | sub-exp |
|---|---|
| 0.0 (argmax) | E3a |
| 0.01 | E3b |
| **0.03 (anchor)** | E3c |
| 0.1 | E3d |
| 0.3 | E3e |
| 1.0 | E3f |

학습 = anchor τ=0.03, eval 만 변경 (same checkpoint reuse). `dilution_collapse warn` (τ ≥ 0.3 sub-exp): `directional_commit_magnitude < 0.001m` 시 sub-exp 단독 무효 (= max(ΔOOF) 계산에서 제외, warn flag 박제).

ΔOOF(E3) = `max(OOF over τ ≠ 0.03 excluding dilution_collapse) − OOF(0.03)`.

#### E4 — Loss swap (2 sub-exp)

| sub-exp | loss |
|---|---|
| **E4a** | baseline `α·CE + β·Huber_offset` (α=β=1.0) |
| E4b | swap to `α·CE + 0.5·Huber_offset + 0.5·Hinge_pred` |

ΔOOF(E4) = OOF(E4b) − OOF(E4a).

**E4 2-variable composition 명시** (단일 변수 swap 의 의도된 예외): E4b 는 (i) Huber_offset weight 1.0→0.5 변경 + (ii) Hinge_pred term 0→0.5 추가의 *2-variable swap* — 단순 hinge 추가 시 reg head 학습 신호 약화 risk, weight balance 유지 위함. plan-012 `hybrid_combined_loss(use_hinge=True)` 의 `0.5·Huber + 0.5·hinge` paradigm carry. 의도된 2-variable.

#### E5 — Reg head on/off (2 sub-exp)

| sub-exp | reg head |
|---|---|
| E5a | off (= cls only, `hybrid_pred = F0 + anchor_blend`) |
| **E5b** | on (anchor) |

ΔOOF(E5) = OOF(E5a) − OOF(E5b) = OOF(variant) − OOF(anchor) — **다른 axis (variant − anchor) 와 동일 부호 convention**. positive ΔOOF → reg off (E5a) 가 더 좋음, best_stack 채택. negative ΔOOF → reg on (E5b) 유지 (anchor). §9.1 `argmax ΔOOF` 로직 정합.

### §7.2 산출물

- `analysis/plan-014/g3_phase2_axis.py` — 5 axis sub-exp 일괄 실행
- `analysis/plan-014/g3_phase2.json` — §3.4 G3 schema 박제
- `runs/baseline/plan014_g3_E*/` (net 새 학습 = anchor 제외 ~11 sub-exp)
- registry row: `H039_g3_phase2_axis5`

### §7.3 G3 합격 (§3.4 carry)

- 5 axis 모두 informational 완료 (E1 conditional skip 허용)
- **최소 1 axis** 의 `max(ΔOOF) ≥ +0.005` → `G3_passed = true`
- 모든 axis ΔOOF < 0.005 → `g3_marginal_only` warn

### §7.4 decision-note 후보

- E2 K-Means 재fit 시 random_state=20260606 carry
- E3 τ scan model checkpoint = E2b (= G2 winner) reuse
- E4 hinge weight = 0.5 (plan-012 carry)

---

## §8. STAGE 4 (c8, G4) — Phase 3 aux ablation 3 (E6~E8) [TODO]

### §8.1 sub-exp matrix

각 sub-exp = G2 winner config + G3 anchor (E2b/E3c/E4a/E5b) 위 *지정 axis 1개 변경*. fold=0. F0 frozen 공통.

#### E6 — Boundary sample weighting (2 sub-exp)

| sub-exp | weight |
|---|---|
| **E6a** | uniform 1.0 (anchor) |
| E6b | `boundary_mask = (err_F0 > 0.005) & (err_F0 < 0.015)`, `sw = where(mask, 3.0, 1.0)`. err_F0 = `‖F0_pred_frozen − y_true‖` |

ΔOOF(E6) = OOF(E6b) − OOF(E6a).

#### E7 — Scorer arch (2 sub-exp)

| sub-exp | scorer |
|---|---|
| **E7a** | BiGRU h=128 (anchor) |
| E7b | LastStep MLP — `MLP_seq` (9→64, **2-layer**: `Linear(9, 64) → GELU → Linear(64, 64) → GELU`) + `MLP_cand` (3→64, **2-layer**: `Linear(3, 64) → GELU → Linear(64, 64)` — 마지막 GELU 없음, dot-product 전 raw representation) → dot-product `logits = (cand_h × h).sum(-1)`. seq MLP / cand MLP 별도 weight |

ΔOOF(E7) = OOF(E7b) − OOF(E7a).

#### E8 — r=0 logit prior (3 sub-exp, inference-only)

| sub-exp | r0_logit_prior |
|---|---|
| **E8a** | 0.0 (anchor) |
| E8b | +0.5 |
| E8c | +1.0 |

학습 = anchor config (prior=0), eval 만 `prior[0] = r0_logit_prior`. ΔOOF(E8) = `max(OOF over prior ∈ {+0.5, +1.0}) − OOF(0.0)`.

### §8.2 산출물

- `analysis/plan-014/g4_phase3_aux.py` + `g4_phase3.json` + sub-exp runs
- registry row: `H040_g4_phase3_aux3`

### §8.3 G4 합격

- informational only — `G4_passed` 항상 true
- positive lever (ΔOOF > 0) 발견 시 `positive_axes` 박제 → G5 best stack 후보

---

## §9. STAGE 5 (c9, G5) — Phase 4 final 5-fold + best stack + submission [TODO]

### §9.1 best_stack 선정 알고리즘 (§3.4 G5 carry, AMBIGUITY fix)

**Phase 2 best lever** = `argmax(ΔOOF over E1, E2, E3, E4, E5)`. ΔOOF ≤ 0 면 anchor 유지.
**Phase 3 best lever** = `argmax(ΔOOF over E6, E7, E8)`. ΔOOF ≤ 0 면 anchor 유지.

**best_stack stacking rule** (단일 lever 분리, conditional resolve):
1. **categorical variant winner**: 각 lever 의 sub-exp variant 중 ΔOOF 최대 인 *단일 variant* 만 채택. 예: E2 K density → K∈{5,9,13} 중 max ΔOOF 의 K 만 (= 단일 K 값). E3 τ scan → 단일 τ 값. categorical variant 끼리 combination 안 함.
2. **lever-level pooling**: Phase 2 axis 5 중 *단일 best axis 만 채택*. Phase 3 axis 3 중 *단일 best axis 만 채택*. → best_stack = (G2 anchor) + (Phase 2 best axis 의 winning variant) + (Phase 3 best axis 의 winning variant) 의 3 elements stack.
3. **conditional lever resolve**:
   - E1 winner=E0a 시 skip → Phase 2 best 는 {E2, E3, E4, E5} 중에서
   - E4=E4b (hinge) + E5=E5a (reg off) 동시 채택 시 hinge_pred 항이 reg_offset 미사용 → Huber_offset 항 자동 0. **explicit final loss**: `hybrid_combined_loss(use_reg_head=False, use_hinge=True)` 분기에서 `loss = α·CE(logits, w_k) + 0.5·hit_aware_hinge(hybrid_pred, target)` (Huber term 완전 제거, hinge_pred 항만 reg term 자리에 채택). hybrid_pred = F0 + Σ_k prob_k · a_k_world (reg_offset 무시). 충돌 없음
   - E7=E7b (LastStep MLP) + E8=E8b/c (r0 prior > 0) 동시 채택 시 prior 적용 path 동일 (inference 후처리), 충돌 없음
   - E7=E7b 시 scorer arch 변경 → E2 K density 의 K 값 변경 영향 없음 (anchor 좌표만 의존), 충돌 없음
4. **conflict 발생 시** (e.g., 동시 채택이 inference path 끊기): conflict 박제 + anchor fallback. fail trigger `g5_lever_conflict`.

decision-note: spec-default — best_stack 은 "Phase 2 best 1 lever + Phase 3 best 1 lever" 의 최대 3 lever combination (G2 anchor 포함). multi-lever interaction 검정은 plan-015 grid search 후속.

### §9.2 5-fold concat OOF + submission

```python
config_anchor = {"name": "anchor", **g2_winner_config}
config_best   = {"name": "best", **g2_winner_config, **phase2_best_lever, **phase3_best_lever}

for cfg in (config_anchor, config_best):
    for fold in range(5):
        model = build_plan014_model(cfg, fold)
        # build_plan014_model(cfg: dict, fold_id: int) -> Plan014Model. F0 frozen 공통,
        # encoder (BiGRU or LastStep MLP per cfg.scorer_arch) + cls/reg head + F0_function attach.
        oof_preds[cfg["name"]][fold_id == fold] = train_and_predict(model, fold, cfg)
        # train_and_predict(model: Plan014Model, fold_id: int, cfg: dict) -> np.ndarray:
        # returns val_pred (Nv, 3) — train + val_predict 통합, model state mutate (학습 후 best
        # checkpoint restore), returned array 는 detach + cpu + numpy.

oof_anchor = compute_hit(oof_preds["anchor"], train_y, R_HIT=0.01)
oof_best   = compute_hit(oof_preds["best"],   train_y, R_HIT=0.01)
delta_oof  = oof_best - oof_anchor

# test 5-fold ensemble = 좌표 mean (plan-012 §9.2 carry)
fold_preds = np.stack([predict(model_fold[k], test_x, cfg=config_best) for k in range(5)], axis=0)
# predict(model: Plan014Model, X_test: np.ndarray, *, cfg: dict) -> np.ndarray:
# returns test_pred (N_test, 3) — model.eval() + hybrid_predict, detach + cpu + numpy.
test_preds_ensemble = fold_preds.mean(axis=0)
write_submission(run_dir, test_preds_ensemble, test_ids)
```

### §9.3 산출물

- `analysis/plan-014/g5_phase4_final.py` + `g5_phase4.json`
- `runs/baseline/plan014_g5_phase4/submission.csv`
- registry row: `H041_g5_phase4_final` (★ 핵심 row)

### §9.4 G5 합격 (§3.4 carry)

- `best_stack_5fold_oof ≥ anchor_5fold_oof + 0.005` → `G5_passed = true`
- 위반 시 `g5_no_additive` warn → fallback = anchor 5-fold submission
- band 분류 (§3.2):
  - ≥ 0.66 → **positive** (corrector paradigm 부활)
  - 0.65 ≤ OOF < 0.66 → **partial**
  - < 0.65 → **negative** (premise 의심)

### §9.5 decision-note 후보

- test 5-fold ensemble = 좌표 mean (plan-012 carry)
- best_stack additive 가정 — interaction 측정은 plan-015
- band borderline 시 mechanical 적용 (e.g., 0.6498 → negative)

---

## §10. STAGE 6 (c10, G_final) — synthesis + plan-015 후보 + 3 파일 sync [TODO]

### §10.1 산출물

- `plans/plan-014-plan012-failure-inversion.results.md` 신규 — frontmatter (status `G_final_complete` / lb_score null / exp_ids = H036~H041) + body:
  - §1 G0~G5 결과 narrative (band 박제)
  - §2 plan-013 join interpretation (§1.4 activated row 1개)
  - §3 premise verdict (corrector 재사용 = root cause 검증 신호)
  - §4 plan-015 후보 ≥ 3 (band 별 분기)
- `registry.csv` append — 6 row (H036~H041)
- `plans/plan-014-plan012-failure-inversion.md` frontmatter sync (`status: spec → G_final_complete`)

### §10.2 plan-015 후보 (band 별 분기, ≥ 3)

#### 공통 (모든 band)

- **(공통-1) F0 frozen vs learnable attribution** — plan-014 의 F0 frozen baseline 결과 + F0 learnable variant 측정 → F0 component 의 paradigm 기여 attribution. (plan-014 §1.5 정직성 원칙 carry-over)
- **(공통-2) Multi-seed 분산 측정** — single seed (20260514) → 5-seed × 5-fold + std

#### Band positive (≥ 0.66)

- **(positive-1) LB carry-over** — plan-014 best submission LB submit
- **(positive-2) plan-013 ensemble** — plan-013 fallback (0.6381) + plan-014 best 좌표 mean ensemble (§1.4 row 3 매핑)

#### Band partial (0.65 ≤ OOF < 0.66)

- **(partial-1) plan-013 corrector + plan-014 hybrid** — Candidate C 변형 (§1.4 row 5)
- **(partial-2) Inter-lever interaction grid** — Phase 2 + Phase 3 full 2x2 grid (additive 가정 검증)

#### Band negative (< 0.65)

- **(negative-1) plan-013 join row 1/4 매핑** — corrector paradigm 폐기 정당화 (row 1) 또는 deep path-pivot (row 4)
- **(negative-2) `notes/new-ideas.md` 12종 후보 batch 조사** (KNN / GP / Diffusion 등 paradigm-shift 사전 조사)

### §10.3 G_final 합격 (§3.4 carry)

- 3 파일 sync 완료 + plan-015 후보 ≥ 3 박제 + band 분류
- 누락 시 `final_sync_missing` severe

### §10.4 종료

- §0.5 c10 [TODO]→[DONE] sync commit + push
- telegram alert: `"plan-014 완료, band=<...>, best_stack=X.XXXX"`
- `/loop` 자연 종료 (§12.10 carry)
