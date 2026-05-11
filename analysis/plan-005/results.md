---
plan_id: 005
based_on:
  - 004
  - notes/PB_0.6822 코드공유.ipynb
finished_at: 2026-05-12 (Asia/Seoul)
status: all_complete
exp_ids_completed:
  - D001_pb-0-6822-diagnostic
lb_score: null
---

# plan-005 results — PB_0.6822 framework diagnostic

## 한 줄 요약

plan-004 가 적용한 PB_0.6822 framework 의 ceiling 은 raw oracle **0.7188**, post-corr oracle **0.7111** (corrector 가 oracle 을 *떨어뜨림*). selector hit (soft) 는 **0.6599** — oracle 까지 5.1pp gap. selector 의 top-1 ranking 정확도는 **12.6%** 로 매우 약함 (soft 평균이 noise 를 cover 하고 있음). plan-006 anchor 후보 4개 도출.

## STAGE 0 — 인프라 + 재현성

- corrector full-fit 재실행 (~5 min, cuda:1, seed=20260606): seed_drift RMSE = **0.000814 m** (< 0.001m threshold → status=ok).
- corrected_oof / corrected_test shape (10000, 27, 3) finite.
- regime_histogram drift (local fit vs plan-004 박제) = 0 (perfect match).

## STAGE 1 — Oracle 4-tier

- raw oracle (best of 27 raw cands)        : **0.7188**
- post-corr oracle (best of 27 corrected)  : **0.7111** ⚠️ corrector 가 oracle 을 0.77pp 떨어뜨림
- 14/18 regime 에서 corrector 가 oracle 떨어뜨림 (regime 4/13/16/17 만 +).
- per-family oracle: `analysis/plan-005/oracle_summary.{json,md}` 참조.
- **finding**: corrector 의 loss 가 *best-cand* 가 아닌 *family-aware* 라 best 후보를 옮길 수 있음. selector-soft 효과 (+0.89pp) 와 trade-off.

## STAGE 2 — Selector decomposition

- hit (argmax / soft) = **0.6595** / **0.6599**
- top-K accuracy: K=1 **0.1262**, K=3 **0.2183**, K=5 **0.2815** ★
- margin (top1 − top2) percentiles: p10=0.001 / p50=0.008 / p90=0.056
- **finding**: selector 가 *진짜 best candidate* 를 1순위로 picking 12.6% — soft 평균이 cover 하지만 ranking 자체는 약함. arch 교체 ROI 큼.

## STAGE 3 — Corrector decomposition

- cap saturation overall = **0.0358** (cap=0.006 의 95% 임계값 기준; cap 충분, 더 큰 cap 불필요).
- direction breakdown |delta|/scale (Frenet local frame):
    - parallel  mean ± std = **0.0451** ± 0.0872
    - perp      mean ± std = **0.0214** ± 0.0602
    - binormal  mean ± std = **0.0064** ± 0.0272
  → binormal 은 parallel 의 1/7 (z 방향 보정 거의 0). plan-006 의 binormal family 추가 효과 작을 가능성.

## STAGE 4 — Selector component contribution

- 3 variant hit (overall):
    - full         (gru + physics + regime) : argmax 0.6595 / soft 0.6599
    - A_no_regime  (gru + physics, regime=0): argmax 0.6553 / soft 0.6570
    - B_no_gru     (physics + regime only) : argmax 0.6566 / soft 0.6547
- marginal contribution (full − variant):
    - gru contribution (full − B)    : argmax +0.0029 / soft +0.0052
    - regime contribution (full − A) : argmax +0.0042 / soft +0.0029
- intervention (gru: B↔full): rate=0.7659, helped/hurt=0.565/0.435, Δhit when changed=+0.0038
- intervention (regime: A↔full): rate=0.5530, helped/hurt=0.509/0.491, Δhit when changed=+0.0076
- family-change: gru_intervention cross_family_pct=0.634, regime_intervention cross_family_pct=0.547

## STAGE 5 — Failure analysis + B001 비교

- worst-100 의 regime 빈도 (top-3): regime 10=8, regime 11=6, regime 12=5
- B001 baseline 비교: PB hit **0.6599** vs B001 **0.5787** (delta +0.0812)
- per-sample win/loss: PB win **965** / PB loss **153** (ratio 6.3:1)
- PB − B001 mean err = -1.149 mm

## decision-note 박제 list (commit msg 자율 결정)

1. spec-default (c2): plan §3.1 의 형식적 spec 4건 (3-D coords / ens_scores post-bias / regimes 재계산 / bias 항 cache) 자율 해소.
2. spec-default (c3): corrected_*.npz 등 heavy intermediate 모두 .gitignore 추가; STAGE 1~5 code 를 c3 commit 에 같이 박제.
3. spec-default (c4 / G1): plan §5.3 의 hard gate `post_corr_oracle ≥ raw_oracle - 0.001` 를 warn-only flag (corrector_hurts_oracle) 로 softening — finding 자체가 진단 대상.
4. spec-default (c4-c6): c4/c5/c6 단일 commit 묶음 (코드 변경 0, rendering 만).
5. spec-default (c8): Variant A retrain (c7a) 가 background 진행 중이지만 stage5 독립 실행 → audit 진행도 가속.

## 참조

- 본 plan: `plans/plan-005-pb-0-6822-diagnostic.md`
- 후속 plan 후보: `analysis/plan-005/next_plan_candidates.md`
- 산출 raw json/md: `analysis/plan-005/{oracle_summary,selector_decomp,corrector_decomp,component_contribution,failure_b001}.{json,md}`
