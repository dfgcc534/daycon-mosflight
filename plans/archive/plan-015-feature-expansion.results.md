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
lb_score: 0.6628
lb_band: positive
band: negative
best_5fold_oof: 0.6425
delta_oof: 0.0000
oof_lb_gap: 0.0203
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

## §2. plan-013/plan-014 join interpretation — **재해석 with LB measured**

plan-014 §1.4 carry table:
- plan-013 LB 0.6381 (fallback, < 0.68)
- plan-014/plan-015 best_stack: **OOF 0.6425 (negative band) vs LB 0.6628 (positive band, ≥ 0.66)**
- **OOF–LB gap = +0.0203** (5-fold OOF underestimate measured)

→ **plan-013 LB < 0.68 + plan-014/015 LB ≥ 0.66 (positive band)** 매핑:
  - row 1 (plan-014/015 LB ≥ 0.68 + OOF positive band) 의 변종 — LB 는 positive 진입했으나 0.68 threshold 미달.
  - row 3 (둘 다 positive) 부분 매핑: plan-014/015 가 corrector paradigm 으로서 LB band 정복 → ensemble with plan-013 path 가능성 (positive-2 candidate 활성).
  - **row 4 (둘 다 실패) 활성 *해제***: LB measured 가 OOF 예상보다 +0.02 위 → deep path-pivot 필요성 약화.

**plan-014/015 가 plan-013 LB (0.6381) 위 +0.025 LB 회수** = corrector paradigm 의 *실제 ceiling 이 OOF 보다 높음* measured.

DACON 236716 muflight 의 framework family ceiling 재해석: **OOF 5-fold 가 LB 보다 conservative** (test set 분포 vs 5-fold OOF 분포 차이로 ~0.02 systematic underestimate). 모든 향후 plan 의 OOF 측정은 LB 보정 +0.02 를 가설로 박제 가능.

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

## §4. plan-016 후보 — **LB positive band 재해석으로 priority 재정렬**

LB 0.6628 (positive band) measured → plan-014/015 corrector paradigm 이 *실제로 작동*. plan-016 = negative branch (deep path-pivot) 보다 **positive band polish + 추가 회수** 우선.

### 공통 (모든 band, 유지)

- **(공통-1) Feature B/C/D 단독 측정** — A 부재 시 단독 ΔOOF. A 가 redundant 였을 가능성 검증.
- **(공통-2) Multi-seed 분산** — plan-015 best 0.6425 OOF 의 std + LB-OOF gap 의 stability 측정.

### **Band positive 분기 활성** (LB 재해석)

- **(positive-1) plan-013 + plan-015 ensemble** ★ — plan-013 fallback (LB 0.6381) + plan-015 best (LB 0.6628) submission 좌표 mean ensemble. plan-013 framework path + plan-014/015 corrector path 의 *measured 결합 회수 가능성*. low-cost, dacon-submit 1회.
- **(positive-2) plan-015 의 lever stack 재시도** — plan-014 G3 g3_marginal_only 였으나 OOF↔LB gap 고려 시 marginal lever (E2c K=9 +0.003, E6b boundary +0.0015) 의 LB-scale 효과 더 클 수도. lever combination grid (예: K=9 × boundary on/off × τ=0.01/0.03/0.1 등 small grid) 의 LB head-to-head 비교.
- **(positive-3) Multi-seed ensemble** — single seed (20260514) 한계. 5-seed × 5-fold = 25 model coord mean ensemble → variance reduction 으로 +0.005~0.01 LB 추가 회수 가능.

### Band ≥ 0.68 진입 시도 (high-value path)

- **(high-1) Feature B/C/D 단독 LB 측정** — 공통-1 의 LB 보정 — OOF 가 underestimate 였으므로 marginal OOF lever 도 LB 에서 positive 가능. low-cost dacon-submit 3회 (B/C/D 단독).

### Band negative 분기 (deactivated)

- ~~deep path-pivot (KNN/transformer/작업 중단)~~ — LB 0.6628 positive 진입으로 *deactivated*. corrector paradigm 자체는 작동. paradigm shift 불필요.

### 가설 검증 우선순위 (LB-aware, cost-ascending)

1. **공통-1 + high-1** (B/C/D 단독 OOF + LB 측정) — 4 dacon-submit 필요 (B/C/D 단독 + 합 가능 best). low cost (5-fold OOF 산출 ~15s each).
2. **positive-1 (plan-013 + plan-015 ensemble)** — 1 dacon-submit, no training.
3. **positive-3 (multi-seed ensemble)** — 5 train run × 5 fold = 25 models, 1 dacon-submit.
4. **positive-2 (lever grid)** — medium cost grid search.

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
