---
plan_id: 019
version: 1.1 (G_final synthesis)
date: 2026-05-15 (Asia/Seoul)
status: G_final_complete
based_on:
  - 007 (A0 baseline, step 4 MLP OOF 0.6482, LB 0.6598)
  - 005 (oracle 0.7188 — single-stack 회수 불가능 ceiling)
  - 004 (LB 0.6822 — multi-stage ensemble reference)
  - brainstorm-iter-1~5 (6 candidates × ranking)
  - 018 (single-stack architecture lever falsified — paradigm-shift 동기)
followed_by:
  - plan-004 upgrade direction (plan-020 candidates A/B/C/D 폐기, 사용자 재평가 결과)
finished_at: 2026-05-15T12:55+09:00
exp_ids_completed:
  - F013_a0
  - F014_ebip-base
  - F015_ebip-icnn
  - F016_meta-ebip-icnn
lb_exp_id: null
lb_score: null
lb_submitted_at: null
exception_policy: plan-007 §2.2 end-to-end 통합 예외 — 본 plan 의 EBIP variant 가 single-stack 확인 (multi-stack 아님)
---

# plan-019 G_final — Results

## §1. 본 plan 의 핵심 발견 (1-sentence)

> "energy-based implicit prediction" paradigm 의 *실측 ceiling* = A0 + **0.007** (single-stack 의 데이터-제한 ceiling). ICNN convex / FOMAML meta adaptation 의 추가 component 는 marginal/없음.

## §2. Progressive ablation 결과 table

| stage   | model                          | params | unroll_T | OOF    | Δ vs A0  | Δ vs prev | gate result                        |
|---------|--------------------------------|--------|----------|--------|----------|-----------|------------------------------------|
| A0      | CoeffMLP (13→32→8)             |   297  | —        | 0.6482 | —        | —         | G0 PASS (target ∈ [0.6479, 0.6485])|
| **S1**  | EBIPBase + unrolled GD T=5     | 25,322 | 5        | **0.6552 ★** | **+0.0070** | +0.0070   | G1 WARN (target ≥ 0.66, -0.0048)   |
| S2      | EBIPICNN + softplus + ICNN     | 31,674 | 3        | 0.6520 | +0.0038  | -0.0032   | G2 WARN (target ≥ 0.68, -0.0280)   |
| S3      | MetaEBIPICNN + FOMAML 1-step   | 31,674 | 3        | 0.6538 | +0.0056  | +0.0018   | G3 WARN (target ≥ 0.70, -0.0462)   |

→ best = **S1 EBIP base** (OOF=0.6552). S2/S3 의 추가 component 가 천장 회복 불가.

### Per-fold breakdown

| fold | A0     | S1     | S2     | S3     |
|------|--------|--------|--------|--------|
| 0    | 0.6619 | 0.6713 | 0.6728 | 0.6703 |
| 1    | 0.6453 | 0.6532 | 0.6463 | 0.6507 |
| 2    | 0.6476 | 0.6517 | 0.6497 | 0.6497 |
| 3    | 0.6510 | 0.6559 | 0.6495 | 0.6545 |
| 4    | 0.6350 | 0.6436 | 0.6416 | 0.6436 |

fold 0 이 모든 stage 에서 highest, fold 4 가 lowest — sample_id stable_fold_id 의 *데이터 분포 자체* effect. stage 간 비교는 valid.

## §3. 각 component 의 marginal gain 분석

### §3.1 S1 vs A0 — explicit → implicit reformulation 의 gain (+0.0070)

- A0 (explicit): `pred = p0 + Σ c(τ)·B(τ)`, c = 13-d stats → 8 coeff MLP.
- S1 (implicit): `pred = argmin_p [||p - anchor||² + λ·g_θ(p, cond_77d)]`, anchor = A0 식, unrolled GD T=5.
- gain 의 source:
  - **cond_77d** (= 13 handcrafted ⊕ 64 cnn_encoded) 의 *encoder bottleneck 완화* (plan-018 의 A1/A2/A6 encoder 강화 시도 와 동일 효과).
  - **energy correction term** (g_θ) 의 *universal residual*.
- gain 의 ceiling: +0.0070 (vs target +0.0118 = 0.66 - 0.6482). brainstorm 추정 (0.70~0.72) 의 약 60% 도달.
- *measured*: implicit reformulation 자체는 single-stack 의 ceiling 추가 push 가능 (plan-018 의 A3 +0.0003 보다 23× higher), 그러나 plan-007 §9.2 의 0.6491 ceiling 예측 위 약간 push 에 그침.

### §3.2 S2 vs S1 — ICNN convex 의 trade-off (-0.0032)

- S1 (non-convex MLP energy) → S2 (ICNN convex energy).
- 가설 H2: "convex 제약으로 stability ↑, 천장 약간 하향" — *directionally confirmed*:
  - 4 stability fix 적용 후 모든 fold 학습 안정 (NaN 없음, divergence 없음).
  - 천장 -0.0032 — convex 제약이 capacity 감소시킴.
- icnn_min_softplus = 5 fold 모두 ~0.0075 (= init -5 의 softplus 값 보존) → **W_z 학습 거의 미발현**. 즉 ICNN 은 init-time near-linear 에 머묾, energy correction 의 표현력 자체가 부족.

### §3.3 S3 vs S2 — FOMAML meta adaptation 의 gain (+0.0018)

- S2 + per-sample c_τ adaptation (1-step inner SGD on self-supervised pretext).
- 가설 H3: "per-sample adaptation 이 oracle 근접" — **falsified** (target +0.020 의 9% 도달).
- pretext task (window[-2] → window[-1], +40ms) 와 main task (window[-1] → +80ms) 의 *implicit transfer* 가 약함 — basis 의 비선형 (t², t³) scaling 이 horizon=1 vs horizon=2 에서 다른 형태.
- c_inner_lr=0.01 × 1 step = 매우 작은 c_τ adaptation, c_meta 에서 거의 안 움직임.
- 가설 의도: oracle (best of 27 candidates per-sample) 의 *per-sample c* 를 학습 — 그러나 본 구현은 *implicit metric* (reconstruction error) 위에서만 작동, oracle 의 *task-specific c* 와 mismatch.

### §3.4 S3 vs S1 — combined effect (-0.0014)

- S3 = S1 + ICNN + meta — *둘 다 추가했는데* S1 보다 낮음.
- ICNN 의 capacity loss (-0.003) 가 meta 의 marginal gain (+0.002) 을 상쇄.
- **combined paradigm 가설 falsified**: ICNN + meta 는 *상호 보강* 이 아닌 *상호 cancel*.

## §4. 4 가설 verdict

| 가설 | 검증 | 측정 | Verdict |
|---|---|---|---|
| H1 implicit reformulation gain | S1 OOF ≥ 0.66 | +0.0070 (≈ 58% of target) | **partial** |
| H2 ICNN convex stability + 천장 소량 하향 | S2 OOF ≥ 0.68 | -0.0032 vs S1, 0.6520 | **directionally OK, threshold miss** |
| H3 FOMAML per-sample adaptation | S3 OOF ≥ 0.70 ⭐ | +0.0018 vs S2 → 0.6538 | **marginal positive, target far miss** |
| H4 LB > 0.70 | dacon-submit | SKIP | **inconclusive (OOF estimate likely fail)** |

## §5. plan-007 / plan-018 와의 일관성

| paradigm | 실측 OOF gain vs A0 | source |
|---|---|---|
| plan-007 §9.2 추측 ceiling (단일 공식) | ~0.0009 (= 0.6491 - 0.6482) | plan-007 |
| plan-018 G1 best A3 MoLE (단일 stack head 강화) | +0.0003 | plan-018 |
| **plan-019 G1 S1 EBIP base (implicit reformulation)** | **+0.0070** | 본 plan |
| plan-019 G3 S3 Meta-EBIP+ICNN (combined) | +0.0056 | 본 plan |

→ single-stack 의 *실측 ceiling 의 분포* ≈ **A0 + 0.005~0.010**. plan-007 의 추측 0.6491 위 약간 push 가능하나, plan-005 oracle 0.7188 / plan-004 LB 0.6822 도달은 single-stack 으로 *측정상* 불가능.

## §6. 후속 방향 — plan-004 upgrade (plan-020 candidates 폐기)

본 §6 의 직전 버전 (plan-020 후보 A/B/C/D — corrector 결합 / multi-stack / learnable basis / DEQ) 은 *plan-019 측정 후 사용자 재평가 결과 폐기*.

### 폐기 사유

- 후보 A (S1 + corrector ensemble): S1 OOF=0.6552 < plan-005 D001 (~0.69) — ensemble member 로 약함. revised expected gain +0.000~+0.005.
- 후보 B (multi-stack with S1 corrector): S1 자체가 plan-007 step 4 와 동급 → corrector 가치 약함.
- 후보 C (learnable basis + MoLE): plan-018/019 결합 결론 — basis/head capacity 가 main lever 아님 measured (encoder dim 이 lever).
- 후보 D (DEQ): plan-019 S1 의 energy_mlp 학습 미발현 measured — DEQ 도 동일 ceiling.

본 후보들의 공통점: **plan-019 S1 의 measured contribution 을 활용** 하려는 path. 그러나 S1 의 +0.007 gain source 가 *CNN encoder 64d* 으로 measured → S1 module 자체보다 **encoder lever 직접 활용** 이 더 효과적.

### 재정의된 후속 방향 — **plan-004 upgrade**

- plan-004 LB 0.6822 (27-candidate full ensemble selector + GRU) 의 직접 upgrade path.
- plan-019 의 *encoder dim* (CNN 64d) measured lever 와 plan-018 의 *head ablation* 결합 — 그러나 *paradigm 은 plan-004 carry* (27 candidates + selector).
- 본 plan-019 의 *single-stack ceiling 박제 결론* 으로, paradigm-shift 가 *plan-005/004 paradigm 직접 upgrade* 방향으로 결정.

상세 plan-NNN 작성은 별도 commit (본 plan-019 scope 외).

## §7. 종료 checklist (§13 mirror)

- [x] G0: A0 reproduce OOF=0.6482 ∈ [0.6479, 0.6485]
- [x] G1: S1 EBIP base 5-fold OOF=0.6552 (target 0.66, WARN -0.0048)
- [x] G2: S2 + ICNN convex 5-fold OOF=0.6520 (target 0.68, WARN -0.0280)
- [x] G3: S3 + meta adaptation 5-fold OOF=0.6538 (target 0.70 ⭐, WARN -0.0462)
- [x] G4: best variant LB SKIP per user decision (quota 보존)
- [x] G_final: results.md (본 파일) + 후속 방향 박제 + 3 파일 frontmatter sync
- [x] 모든 commit + push 완료 (CLAUDE.md ⚠️ Commit · Push 의무 답습)

## §8. brainstorm carry verdict (revised)

본 plan G1/G2/G3 모두 WARN, single-stack 의 *실측 ceiling* 박제 → brainstorm 의 나머지 5 candidates 도 single-stack 한정, 모두 fail 예상. **후속 = plan-004 upgrade direction** (paradigm 자체를 plan-005/004 의 multi-formula 27-candidate path 위에서 재구성).

decision-note: plan-020 후보 4종 (A/B/C/D, next_plan_candidates.md) 폐기 — plan-019 측정 후 재평가 결과, 본 후보들 모두 S1 module 활용 path 이나 *plan-004 paradigm 직접 upgrade* 가 더 ROI 큰 path 로 판단. 본 plan-019 의 *실측 ceiling* ≈ A0 + 0.007 박제 후 후속 = plan-004 upgrade.
