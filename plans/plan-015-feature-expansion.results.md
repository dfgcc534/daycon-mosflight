---
plan_id: 015
version: 2.4 (results synthesis)
date: 2026-05-14 (Asia/Seoul)
status: G_final_complete (G1 negative drop rule, G2~G4 skipped, G5 best=baseline carry)
based_on:
  - 014 (band=negative, baseline 0.6425)
followed_by:
  - 016 (negative branch — deep path-pivot, plan-015 feature 확장 paradigm 한계 confirm)
scope: corrector input feature 확장 (A/B/C/D 순차 ablation) 결과 박제. Feature A (F0 residual direct) 단독 적용 시 ΔOOF=−0.0010 negative → drop rule 발동, G2~G4 skip. best_stack = G0 baseline (= plan-014 best_stack 0.6425).
exp_ids:
  - H042_g0_preflight
  - H043_g1_e1_feature_A
  - H047_g5_best_stack_5fold
  - H048_g_final_synthesis
lb_score: null
band: negative
best_5fold_oof: 0.6425
delta_oof: 0.0000
---

# plan-015 v2.4 — Results (band=negative, drop rule 발동)

## §1. G0~G5 결과 narrative

### G0 — preflight (H042_g0_preflight)

- **plan-014 baseline 5-fold OOF reproduce = 0.6425** (target 0.6425 ± 0.005, in_range=True). plan-014 G5 best_stack (E0c K-Means K=9 + boundary_weight_on, F0 frozen) 정확 reproduce. deterministic 일치 — plan-014 spec 의 모든 carry 항목 정상 동작 confirmed.
- 4 feature single-apply dim sanity: A=12 / B=10 / C=18 / D=15 ✓
- cumulative dim sanity: A=12 / A+B=13 / A+B+C=26 / A+B+C+D=32 ✓
- **g0_passed = True (2/2)**

### G1 — E1 (Feature A only, F0 residual direct, 12D) — **NEGATIVE** (H043_g1_e1_feature_A)

5-fold OOF (plan-014 base config + Feature A only):

| metric | value |
|---|---|
| baseline (G0 reproduce) | 0.6425 |
| **E1 (A, 12D) OOF** | **0.6415** |
| Δ vs baseline | **−0.0010** |
| status | **negative** |
| fold-wise OOF | [0.6638, 0.6412, 0.6482, 0.6313, 0.6230] |
| mean DCM | 0.0025 |

**Feature A 가설 falsified** (plan-014 §10.2 negative-2 / plan-015 v1 §0 박제된 narrative):
- "회수율 5.4% 의 가장 큰 누락 신호 = encoder 가 F0_pred 자체를 못 봄" 가설이 *measured 틀림*.
- F0 residual 정보 추가가 oracle gap (0.18) 회수에 기여 0 (오히려 −0.001 marginal degradation).

가능 해석 (plan-016 후보 hypothesis):
1. F0 residual 정보가 이미 9D base 의 `acc_par/speed`, `acc_perp/speed` 등에 implicit 포함 (redundant feature 추가, model capacity 분산).
2. epoch=20 fixed 가 12D 증가에 대해 underfit (longer training 또는 lr schedule 필요).
3. Feature dim 증가가 small training data (8000 train) 위 overfit risk.

### G2, G3, G4 — **SKIPPED** (drop rule 발동)

§3.2 drop rule (v2.2 spec): G_n ΔOOF < 0 (negative) → G_(n+1)..G_4 모든 후속 stage skip → G_final 직행. best = G_(n−1) cumulative.

G1 negative → G2 (A+B) / G3 (A+B+C) / G4 (A+B+C+D) 모두 skip. best = G0 baseline.

### G5 — best stack 선정 (H047_g5_best_stack_5fold)

- candidates = `{baseline: 0.6425, E1 (A): 0.6415}` (drop rule per E2/E3/E4 제외)
- **best_name = "baseline"** (argmax)
- **best_oof = 0.6425** (= plan-014 G5 best_stack carry)
- delta_oof = +0.0000 vs baseline
- **G5_passed = False** (g5_no_improvement warn, < +0.005 threshold)
- **band = negative** (< 0.65)
- submission = plan-014 best_stack carry (deterministic same config, plan-014/plan-015 의 best config 동일).

## §2. plan-013/plan-014 join interpretation

plan-014 §1.4 carry table:
- plan-013 LB 0.6381 (fallback, < 0.68)
- plan-014 best_stack 0.6425 (< 0.65 negative)
- **plan-015 best_stack 0.6425** (= plan-014 carry, < 0.65 negative)

→ **row 4 활성** (plan-014 G_final 박제와 동일): "둘 다 실패 — 더 deep path-pivot (`notes/new-ideas.md` KNN/GP/Diffusion)"

plan-015 의 input feature 확장 시도가 *추가 회수 0* 으로 confirmed → **DACON 236716 muflight 의 현 framework family (F0 prior + corrector classifier+regression hybrid) 의 measured ceiling 이 F0 raw + ~0.01 (= 0.6425)**.

## §3. Premise verdict — plan-015 narrative 부분 falsified

**plan-015 premise (v1 §0)**:
> plan-014 corrector 의 input 표현력 부족 (oracle 0.82 vs measured 0.64, 회수율 5.4%) 을 *직접 닫기 위해* 9D → +4~28D feature 확장. A/B 1순위 (F0 residual + binormal split) 는 plan-014/005 의 measured evidence 가 직접 가리키는 누락 신호.

**Verdict**: **falsified for A** (1순위 highest), **untested for B/C/D**.

| 측면 | 결과 | 해석 |
|---|---|---|
| Feature A (F0 residual direct) | ΔOOF=−0.001 | A 가설 *measured 틀림*. encoder 가 F0_pred 의 정보 부재 가 회수율 부족 root cause 아님. |
| Feature B/C/D | drop rule per spec | A 부재 시 B/C/D 단독 또는 B+C+D cumulative 도 검정 안 됨. plan-016 후속. |
| 회수율 5.4% root cause | 미확정 | input feature 확장 paradigm 자체로는 회수 불가 (A 시도 부터 0). features 외 root cause 가능성: corrector capacity / loss 함수 / anchor codebook 의 더 fundamental 한 limit. |

→ **plan-015 의 measured 결론**: input feature 차원 확장 만으로는 corrector paradigm ceiling break 불가. plan-016 = **task-level paradigm shift** 필요.

## §4. plan-016 후보 (negative branch — deep path-pivot)

### 공통 (모든 band)

- **(공통-1) Feature B/C/D 단독 측정** — A 부재 시 B (binormal split, 10D), C (multi-scale stride, 18D), D (pairwise, 15D) 각 단독의 ΔOOF measured. A 가 redundant 여서 B/C/D 가 cumulative chain 에서 부당하게 차단된 가능성.
- **(공통-2) Multi-seed 분산** — plan-015 best 0.6425 의 5-seed × 5-fold std (현재 single seed=20260514).

### Band negative 분기 (활성, ≥ 3 후보)

- **(negative-1) KNN-based corrector** — plan-014 §10.2 negative-1 carry. F0 frozen + Frenet local residual KNN-vote. parametric corrector 의 한계 회피.
- **(negative-2) Task framing 변경** — 11-step → direct seq2seq transformer (positional embedding + cross-attention 으로 long-range pattern 직접 학습). plan-015 D feature (pairwise) 의 motivation 을 architecture level 로 격상.
- **(negative-3) DACON 236716 ceiling 정량 박제 + 작업 중단 판단** — plan-013 / plan-014 / plan-015 모두 F0 raw + ~0.01 ceiling confirm. 더 이상 ROI 낮음. 다른 dataset / problem 으로 path-pivot.

### 가설 검증 우선순위 (cost-ascending)

1. **공통-1 (B/C/D 단독)** — low cost (각 5-fold OOF ~15s × 3 sub-exp). plan-014/015 의 *feature space 가 진짜 redundant 인지* 1차 확인.
2. **negative-1 (KNN corrector)** — medium cost, F0 frozen + non-parametric. simple baseline.
3. **negative-2 (transformer)** — high cost, paradigm shift.
4. **negative-3 (작업 중단)** — 위 모두 fail 시.

## §5. measured 값 박제 (외부 reference)

| measure | value | source |
|---|---|---|
| F0 raw hit@1cm (plan-006 frozen) | 0.6320 | plan-014 G0 |
| plan-014 G5 best_stack 5-fold OOF | 0.6425 | plan-014 G5 |
| **plan-015 G0 baseline reproduce** | **0.6425** (정확 일치) | plan-015 G0 (H042) |
| **plan-015 G1 E1 (A, 12D)** | **0.6415** (Δ=−0.001) | plan-015 G1 (H043) ★ A feature falsified |
| **plan-015 G5 best_stack** | **0.6425** (= baseline carry) | plan-015 G5 (H047) |
| oracle ceiling (E0b Frenet-ortho) | 0.8248 | plan-014 G0 |
| corrector 회수율 | 5.4% | plan-014 carry, plan-015 변동 없음 |

→ **plan-014 & plan-015 의 measured ceiling 동일 = 0.6425**. corrector paradigm + input feature 확장 paradigm 모두 F0 raw + ~0.01 limit.

## §6. LB carry-over (Q3 결정 carry, 사용자 confirm 필요)

- plan-015 best_stack submission = plan-014 best_stack submission *동일* (deterministic same config, drop rule per).
- dacon-submit 시 1회만 필요 (plan-014 best ≡ plan-015 best). LB 값 박제 후 frontmatter `lb_score` 채움.
- **사용자 confirm 후 1회 dacon-submit** (DACON daily limit + 동일 submission 중복 회피 차원).

## §7. 종료

- G_final 합격 (3 파일 sync 완료): ✓
  - results.md 신규 (본 파일) ✓
  - plan-015 frontmatter sync (status / band / best / followed_by [016]) ★ 별도 commit
  - registry append 4 row (H042/H043/H047/H048) — incremental ✓
- plan-016 후보 ≥ 3 박제: ✓ (공통 2 + negative 3)
- band 분류: **negative**
- LB carry-over (Q3 결정): dacon-submit 1회 pending (사용자 confirm)
- §0.5 c9 [TODO]→[DONE] sync 별도 commit
- `/loop` 자연 종료
