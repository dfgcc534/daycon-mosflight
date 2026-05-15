---
plan_id: 010
version: 1 (final, superseded)
date: 2026-05-13 (Asia/Seoul)
status: superseded by plan-011 — 미진행 (NOT EXECUTED)
based_on:
  - 004
  - 005
  - 006
  - 007
  - 008
  - 009
superseded_by: 011
supersession_reason: |
  plan-010 v1 draft (8319b5f) 박제 후, server agent 가 동일 narrative 를 4-axis breadth ablation (L/In/M/F 24 sub-exp) 로 확장 → plan-011 로 신규 생성·실행. plan-011 의 실측 evidence (G1 (b) FAIL — 0/4 axis strict +0.005 통과) 가 본 plan v1 의 핵심 가설 (Z1 결함 fix, iterative refinement, frozen GRU reuse) 을 large falsified. 본 plan 의 narrative ("plan-004 결함 7 fix") 폐기, plan-011 의 narrative ("plan-004 default 위 input 확장") 로 인계.
exp_ids: []  # 미진행
lb_score: null  # 미회수 (실행 안 됨)
---

# plan-010 v1 — Single-Formula Anchor + Corrector Redesign Exploration (★ SUPERSEDED · 미진행)

> ⚠️ **본 plan 은 실행되지 않았음 (NOT EXECUTED).** plan-011 이 동일 narrative 의 더 정밀한 ablation 으로 전면 실행 + 박제. 본 spec 은 *historical artifact* 로 git 박제 유지. **모든 G-gate / commit 미진행.**

## §0. 압축 요약 (원 narrative + 폐기 사유)

### §0.1 원 narrative (v1, 폐기)

> plan-006 단일공식 `frenet_par120_perp_neg020` (LB 0.6692 anchor) 위에 plan-004 corrector 의 *7 결함* (target cap-truncation, MSE-hit misalign, far·easy weight 비대칭, env head 낭비, apply_scale hack, hardcoded band) 을 *4 후보* 로 fix:
>
> 1. **Z1+G2** — Minimum Viable Redesign + frozen plan-004 GRU reuse (★ cheap)
> 2. **Z1+G1** — Z1 + CNN encoder learnable
> 3. **Z3+G2** — Z1 + iterative refinement 3-step + frozen GRU (cap 한계 우회)
> 4. **Z6** — end-to-end GRU + corrector 통합 (조건부)
>
> Target LB: 0.6692 → **0.70~0.72**.

### §0.2 폐기 사유 (plan-011 evidence 기반)

| narrative | plan-011 evidence | 판정 |
|---|---|---|
| H1 — "7 결함 fix" framing | L axis NEGATIVE — plan-004 default 가 small-data 에 best tuned (결함 아닌 hyperparam) | **falsified** |
| H2 — Z1 minimum viable redesign | L1 (Z1 minimum) = **−0.0124** vs L0 anchor | **falsified** |
| H3 — Z3 iterative refinement | M4 (Iterative) = **−0.0693** catastrophic | **falsified** |
| H4 — frozen plan-004 GRU reuse | IC SKIP (selector checkpoint 미존재, loadability infeasible) | **infeasible** |
| H5 — 0.70~0.72 LB target | plan-011 best fold-0 OOF = 0.6450 → LB 추정 ~0.665 ≪ plan-006 0.6692 baseline | **falsified** |

### §0.3 살아남은 부분 (plan-011 이 계승)

- G0 preflight schema (7결함 verify + plan-006 reproduce + corrector_decomp 재측정) → plan-011 c2
- 단일공식 path (K=1, no ranking) → plan-011 base
- LB 0회 정책 (carry-over 패턴) → plan-011 답습
- plan-012 후보로 paradigm 교체 (KNN/GP/Diffusion) 예측 → plan-011 §9 conclusion 확정

### §0.4 plan-011 의 실측 결과 (anchor)

- G1 (b) FAIL — 0/4 axis strict +0.005 통과
- best lever = In axis ID (CNN encoder 64-dim) = +0.00495 (just shy of strict)
- F4 (LearnableSingleCandidate, v1.1 post-fix) = +0.0030 (informational positive)
- L/M axis NEGATIVE 또는 flat
- Phase 3+ autonomous skip → plan-012 paradigm 교체 carry-over

## §1. 미진행 명시

| 항목 | 상태 |
|---|---|
| G0 preflight | ✗ 미실행 |
| G1 H006_Z1G2 (Minimum Viable Redesign + frozen GRU) | ✗ 미실행 |
| G2 H007_Z1G1 (Z1 + CNN encoder learnable) | ✗ 미실행 |
| G3 H008_Z3G2 (iterative refinement + frozen GRU) | ✗ 미실행 |
| G4 H009_Z6 (e2e GRU + corrector, 조건부) | ✗ 미실행 |
| G_final synthesis | ✗ 미실행 |
| LB 제출 / carry-over | ✗ 없음 |
| 산출 (`analysis/plan-010/**`, `src/pb_0_6822/corrector_redesign.py`, `runs/baseline/H006~H009/**`) | ✗ 미생성 |

## §2. 인계 (plan-011)

- spec: `plans/plan-011-single-formula-corrector-exploration.md`
- results: `analysis/plan-011/results.md`
- attribution: `analysis/plan-011/phase1_attribution.md`
- carry-over candidates: `analysis/plan-011/next_plan_candidates.md`
- best Phase submission: `runs/baseline/H012_phase1-input-ablation/sub_ID/submission.csv`

## §3. 변경 이력

- v1 (2026-05-13, 8319b5f): 초안 — 단일공식 + corrector 재설계 4 후보 narrative (Z1+G2 / Z1+G1 / Z3+G2 / Z6). LB 제출 0회 (plan-010.1 carry-over).
- v1 final (2026-05-13, 본 commit): **superseded by plan-011 — 압축 + 미진행 명시.** 원 narrative 와 폐기 사유 + plan-011 인계 reference 만 보존. 상세 spec (§4~§N+5, 원 v1 의 ~600 line) 폐기.
