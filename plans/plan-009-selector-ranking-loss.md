---
plan_id: 009
version: 1.3
date: 2026-05-12 (Asia/Seoul)
status: draft
based_on:
  - 004
  - 005
  - 006
  - 007
  - 008
  - notes/PB_0.6822 코드공유.ipynb
scope: ranking loss main (G1, cheap+robust on selector.py partial) + corrector 강화 main 추가 push (G2, additive ablation on boundary.py partial) + multi-stage filter (조건부) (on Variant A + extended pool from plan-008). v1.2 의 oracle 1.5cm = 0.8478 ceiling 발견은 *전략적 anchor* 유지 — target LB 0.74~0.80. v1.2 의 "corrector main" framing 은 cap_saturation 3.58% 실측과 충돌 → v1.3 재배치: fragile lever (corrector — boundary.py + arch + cap + loss 4 곳 동시 변경) 를 G2 secondary 로, robust lever (ranking — selector.py partial only) 를 G1 main 으로. corrector sub-exp 는 cumulative 3 → additive 4 (cap만 / band만 / arch만 / all) 로 attribution 측정. G0 에 cap_saturation_extended 신설 — extended 25 cands 의 saturation rate 재측정으로 corrector main framing 의 evidence anchor. Phase 1 (ranking, robust) → Phase 2 (corrector, additive ablation) → Phase 3 (top-K + multi-stage, 조건부). LB 제출 0 회 (v1.1 유지, 할당량 소진 인계) — plan-008.1 + plan-009.1 carry-over 묶음.
exp_ids:
  - H001_ranking-loss          # ★ G1 main (NDCG@1 + pairwise + ListMLE, plan-008 §10.2.1 직접 후속)
  - H002_corrector-strengthen  # ★ G2 main (additive ablation: cap / band / arch / all, plan-008 §7 carry-over 회수)
  - H003_topk-filter           # cheap booster
  - H004_coarse-to-fine        # 조건부, Phase 1+2 OOF < 0.78 일 때만
  - H005_set-transformer       # 조건부, 위까지 OOF < 0.75 일 때만
lb_score: null
---

# plan-009 v1.3 — Ranking Loss (G1 main, robust) + Corrector Strengthening (G2 main, additive ablation) + Multi-stage Filter (on Variant A + extended pool)

## §0. 한 줄 목적

> **plan-008 의 main_bottleneck="ranking" 진단 + v1.2 의 oracle 1.5cm = 0.8478 ceiling 발견을 *둘 다 main lever 후보* 로 인정하되, fragile/robust 위험도로 순서 재배치 (v1.3):**
>
> 1. **★ G1 main — Ranking-specific loss (robust, selector.py partial only)** — plan-008 §10.2.1 의 ROI 표 직접 후속. NDCG@1 differentiable + pairwise margin × 2.0 + listwise ListMLE. arch 보존. plan-008 의 gap_ranking 0.1119 → ≤ 0.09 회복 + top1_acc 0.172 → ≥ 0.22. **+0.02~0.04 OOF**. *fragile lever 진입 前에 baseline 보장*.
>
> 2. **★ G2 main 추가 push — Corrector 강화 (additive ablation, boundary.py + arch + cap + loss 4 곳 변경, on G1 selector)** — plan-005 corrector_decomp 의 [1, 1.5cm) 1290 sample 회수 + [0.5, 1cm) 깎는 부작용 (−7.83pp) 방어. **additive 4 sub-exp** (a=cap만 / b=band만 / c=arch만 / d=all) 로 각 lever 의 attribution 측정 — v1.2 cumulative 3 sub-exp 의 정보량 부족 fix. **+0.03~0.06 OOF on G1**. fragile lever 이므로 G1 baseline +0.02 확보 후 진입.
>
> 3. **(cheap booster) Multi-stage filter** — Hard top-K filter (test-time, 1 줄) + (조건부) Coarse-to-fine 2-stage / Set Transformer (Phase 1+2 미흡 시).
>
> 4. **(전제 조건) plan-008 §7 carry-over** — `boundary.py` 에 `compute_corrector_loss(pred, target, raw=None, weight=None)` module-level hook 신설. **G2 의 전제** — plan-008 G3 DEFERRED 의 직접 회수. v1.2 와 달리 G1 에서는 boundary.py touch X — G1 robustness 확보.
>
> **v1.3 재배치 근거 (사용자 challenge 후속, 2026-05-12)**:
> - plan-005 cap_saturation overall_rate = **3.58%** (cap=0.006 에서 cap 도달 sample 비율). *cap 이 거의 binding 아님*. cap 확장 (0.006 → 0.012) 의 직접 효과 = saturation rate 만큼만 → v1.2 의 "cap 확장이 main lever" framing 의 실측 신호와 위배.
> - 진짜 corrector main lever 후보: **band-specific loss + arch capacity** (cap 은 sub-component). G0 의 cap_saturation_extended 재측정으로 evidence 확보 후 G2 sub-exp attribution 측정.
> - fragile lever (corrector) 를 main 1순위로 두면 G1 fail 시 baseline 손실. robust lever (ranking) 를 1순위로 두면 G1 fail 해도 +0 retention (selector.py partial 의 fallback 용이).
>
> **Baseline 확정**: plan-008 c7 (`G001-candidate-redefine`, EXTENDED 25 cands). OOF baseline = **0.6503**, oracle 1cm = **0.7562**, oracle 1.5cm (raw 27 실측 / extended 25 G0 추정) = **0.8478 / ~0.875**.
>
> **Variant A 유지**: `regime_prior_strength=0`. regime infra 재도입 X (plan-005 STAGE 6 입증 + plan-008 검증 결과).
>
> **LB 제출 정책 (v1.1 유지)**: **본 plan 내 LB 제출 0 회** (할당량 소진 상태 인계). 모든 Phase 의 submission.csv 는 *생성·박제만*, LB 회수는 carry-over:
> - plan-008.1 carry-over (plan-008 의 `submission_step3.csv`) — 다음 날 사용자 수동 dacon-submit
> - plan-009.1 carry-over (본 plan 의 best Phase submission) — 다음 날 사용자 수동 dacon-submit
>
> **Target LB (carry-over 회수 후 추정)**: **0.74~0.80** (G1 ranking +0.02~0.04 → 0.69 LB, G2 corrector +0.03~0.06 additive → 0.74~0.80). v1.2 의 0.75~0.82 보다 *보수 조정* (fragile lever 의 expected variance 반영). OOF→LB gap +0.022 (plan-005/008 trajectory) 로 derive.

---

## §0.5 Quick Reference (autonomous loop 매 turn 읽는 section)

### 합격 기준 (G-gate sequence)

- G0 (preflight + oracle_decomp + cap_saturation_extended, v1.3 확장): plan-008 산출 verify + `lb_baseline.json` 신설 + **`oracle_decomp.json`** (extended 25 cands best-raw-err 8-bin + 1cm/1.5cm/2cm ceiling) + **`cap_saturation_extended.json`** (cap=0.006 saturation rate 재측정, plan-005 의 3.58% 재현 여부). 위반 시 severe `oracle_decomp_artifact_missing`.
- G1 (Phase 1 — ★ Ranking loss, main 1순위 robust): NDCG@1 differentiable + pairwise margin × 2.0 + listwise ListMLE 의 3 loss component 추가. `selector.py` partial only (arch 보존, boundary.py touch X). (a) `oof_soft_hit ≥ 0.6503 + 0.02 = 0.6703` minimum, stretch **0.69**. (b) `top1_ranking_acc ≥ 0.22` (plan-008 0.172 → +5pp). (c) `gap_ranking ≤ 0.09` (plan-008 0.1119 → -0.02). **LB 미제출** — submission.csv 생성만. 위반 시 `ranking_loss_failure` severe.
- G2 (Phase 2 — ★ Corrector 강화, main 2순위 추가 push, additive ablation): boundary.py hook 신설 + 4 sub-experiments (a=cap만 / b=band만 / c=arch만 / d=all) on G1 selector. (a) `oof_soft_hit ≥ G1 OOF + 0.03` minimum, stretch G1 + 0.06. (b) per-band: `[0.5, 1cm) hit_after ≥ 0.95` (깎는 부작용 방어) ∧ **`[1, 1.5cm) hit_after ≥ 0.30`** (plan-005 의 9.77% → 3x 회복, target). (c) `corrector_oracle_gain ≥ 0` (plan-005 의 −0.0077 → 양수). **LB 미제출** — submission.csv 생성만. 위반 시 `corrector_strengthen_marginal` warn-only (G1 OOF ≥ 0.70 시) / severe (G1 OOF < 0.70 시).
- G3 (Phase 3a — Hard top-K filter, cheap): test-time only, 1 줄. K ∈ {3, 5, 7} grid. (a) `oof_soft_hit (best K) ≥ G2 OOF + 0.005` marginal. **LB 미제출**. 위반 시 warn-only `topk_marginal`.
- G4 (Phase 3b — Coarse-to-fine 2-stage, 조건부): G1+G2+G3 누적 OOF < **0.78** 일 때만. Stage 1 cheap filter 25→top-5 + Stage 2 selector rerank. (a) OOF ≥ Phase 1+2 + **0.02**. **LB 미제출**. 위반 시 `coarse_to_fine_failure` severe.
- G5 (Phase 3c — Set Transformer, 조건부): G1~G4 누적 OOF < **0.75** 일 때만. selector.py partial: GRU + Set Transformer 1 layer fusion. (a) OOF ≥ G4 + **0.03**. (b) `top1_ranking_acc ≥ 0.30`. **LB 미제출**. 위반 시 `arch_swap_failure` severe.
- G_final: STAGE N synthesis + plan-010 후보 + 3 파일 frontmatter 동시 박제 (`lb_score: TBD` — carry-over). **best Phase submission 박제** (path: `runs/baseline/<best_H_exp_id>/submission_*.csv`) + plan-009.1 carry-over instruction 박제 (다음 날 사용자 수동 dacon-submit).
- **LB 제출 정책 (v1.1 유지)**: 본 plan 내 LB 제출 **0 회**. 모든 Phase 의 submission.csv 는 생성·박제만. LB 회수는 plan-009.1 carry-over (plan-008.1 carry-over 와 묶음, 다음 날 사용자 수동 호출). plan-004/006/007/008 의 carry-over 패턴 답습.

### G-gates

- G0: preflight + oracle_decomp + **cap_saturation_extended** (v1.3 확장) — plan-008 산출 verify + `lb_baseline.json` + `oracle_decomp.json` (1cm/1.5cm/2cm ceiling) + **`cap_saturation_extended.json` (cap=0.006 binding rate, 3.58% 재현 여부)** [TODO]
- G1: Phase 1 ★ Ranking loss (main robust) — NDCG@1 + pairwise + ListMLE → OOF ≥ 0.6703 + top1_acc ≥ 0.22 + gap_ranking ≤ 0.09 (LB 미제출) [TODO]
- G2: Phase 2 ★ Corrector 강화 (main 추가 push, additive ablation) — hook + cap/band/arch/all 4 sub-exp → OOF ≥ G1 + 0.03 + [1,1.5cm) hit ≥ 0.30 + [0.5,1cm) hit ≥ 0.95 + corrector_oracle_gain ≥ 0 (LB 미제출) [TODO]
- G3: Phase 3a Hard top-K filter (cheap) → OOF ≥ G2 + 0.005 (LB 미제출) [TODO]
- G4: Phase 3b (조건부, < 0.78) Coarse-to-fine 2-stage → OOF ≥ Phase 1+2 + 0.02 (LB 미제출) [TODO]
- G5: Phase 3c (조건부, < 0.75) Set Transformer arch swap → OOF ≥ G4 + 0.03 + top1_acc ≥ 0.30 (LB 미제출) [TODO]
- G_final: synthesis + plan-010 후보 ≥ 2 + 3 파일 frontmatter sync (`lb_score: TBD` carry-over) + best Phase submission 경로 박제 [TODO]

### Commit chain (next-up)

| # | type | spec section | status |
|---|---|---|---|
| c1 | docs | `plans/plan-009-selector-ranking-loss.md` v1 작성 | [DONE in v1] |
| c1.1 | docs | v1.1 spec 갱신 — LB 제출 정책 *0 회* 로 변경 (할당량 소진 인계) | [DONE in v1.1] |
| c1.2 | docs | v1.2 spec 갱신 — main lever 재정렬: corrector 강화 가 main, ranking loss 가 secondary | [SUPERSEDED by v1.3] |
| c1.3 | docs | v1.3 spec 갱신 — Phase 순서 재배치 (B 안): ranking G1 main robust, corrector G2 main 추가 push additive ablation. cap_saturation_extended G0 추가. caveat #7 통일 / #13 G0 격상 / #16 삭제. spec @ §0/§0.5/§1.4/§2/§4/§5/§6/§N+1/§N+3/§N+4 | [TODO] |
| c2 | code | `analysis/plan-009/preflight.py` — plan-008 산출 verify + `lb_baseline.json` 신설 | [TODO] |
| c2.1 | code | `analysis/plan-009/oracle_decomp.py` — extended 25 cands best-raw-err 8-bin 분포 + 1cm/1.5cm/2cm ceiling 측정 + **cap_saturation_extended (cap=0.006 binding rate 재측정)**. spec @ §4.2 | [TODO] |
| G0 | gate | `lb_baseline.json` + `oracle_decomp.json` + `cap_saturation_extended.json` 생성 + plan-008 산출 5 submission variant 존재 확인 | [TODO] |
| c3 | code | `src/pb_0_6822/selector.py` partial — ranking loss 3 component 추가 (NDCG@1 / pairwise / ListMLE). spec @ §5.1 | [TODO] |
| c4 | code | `analysis/plan-009/ranking_loss_train.py` — Phase 1 학습 wrapper (5-fold OOF, Variant A path). spec @ §5.2 | [TODO] |
| c5 | exp | H001_ranking-loss: 5-fold selector retrain + submission 생성 (LB 미제출). spec @ §5 | [TODO] |
| G1 | gate | OOF ≥ 0.6703 + top1_acc ≥ 0.22 + gap_ranking ≤ 0.09 | [TODO] |
| c6 | code | `src/pb_0_6822/boundary.py` partial — `compute_corrector_loss` module-level hook 신설 (cap 인자화, default 0.006 보존 + wrapper override). spec @ §6.1 | [TODO] |
| c7 | code | `analysis/plan-009/corrector_strengthen.py` — additive 4 sub-experiments (a=cap만 / b=band만 / c=arch만 / d=all). spec @ §6.2 | [TODO] |
| c8 | exp | H002_corrector-strengthen: 4 sub-exp on G1 selector + best 채택 + submission 생성 (LB 미제출). spec @ §6 | [TODO] |
| G2 | gate | OOF ≥ G1 + 0.03 + [1,1.5cm) hit ≥ 0.30 + [0.5,1cm) hit ≥ 0.95 + corrector_oracle_gain ≥ 0 | [TODO] |
| c9 | code | `analysis/plan-009/topk_filter.py` — test-time top-K filter K ∈ {3,5,7} grid. spec @ §7 | [TODO] |
| c10 | exp | H003_topk-filter: 3 K 측정 + best 채택 + submission 생성. spec @ §7 | [TODO] |
| G3 | gate | OOF ≥ G2 + 0.005 + best K 박제 | [TODO] |
| c11 | code | (조건부) `analysis/plan-009/coarse_to_fine.py` — 2-stage filter (Phase 1+2+3a OOF < 0.78 일 때만). spec @ §8 | [TODO] |
| c12 | exp | (조건부) H004_coarse-to-fine: 2-stage 측정 + submission 생성. spec @ §8 | [TODO] |
| G4 | gate | (조건부) OOF ≥ Phase 1+2 + 0.02 | [TODO] |
| c13 | code | (조건부) `src/pb_0_6822/selector.py` partial — Set Transformer 1 layer. spec @ §9 | [TODO] |
| c14 | exp | (조건부) H005_set-transformer: arch swap 측정 + submission 생성. spec @ §9 | [TODO] |
| G5 | gate | (조건부) OOF ≥ G4 + 0.03 + top1_acc ≥ 0.30 | [TODO] |
| ~~c15~~ | ~~sub-lb~~ | **본 plan 내 미수행** (LB 할당량 소진). plan-009.1 carry-over (다음 날 사용자 수동 dacon-submit). spec @ §10 | [DEFERRED] |
| c16 | synthesis | `analysis/plan-009/results.md` + `next_plan_candidates.md` (≥ 2 후보) + best Phase submission path 박제 + plan-009.1 carry-over instruction. spec @ §10 | [TODO] |
| G_final | gate | results.md + next plan 후보 ≥ 2 + 3 파일 frontmatter 동시 박제 (`lb_score: TBD` carry-over) + plan-009.1 instruction | [TODO] |

### Plan-specific severe (WORKFLOW.md §12.3 default 위 추가분)

- `ranking_loss_failure` — G1 OOF < 0.6703 또는 top1_acc < 0.22 또는 gap_ranking > 0.09 (v1.3 신설, main 1순위 fail)
- `corrector_strengthen_marginal` — G2 OOF < G1+0.03. (a) G1 OOF ≥ 0.70 시 *warn-only* (main robust 가 already strong 시 corrector 의 marginal 손실 허용). (b) G1 OOF < 0.70 시 *severe* (compound 효과 의존)
- `coarse_to_fine_failure` — G4 진입 시 OOF < Phase 1+2 + 0.02
- `arch_swap_failure` — G5 진입 시 OOF < G4 + 0.03 또는 top1_acc < 0.30
- `variant_a_residue` — selector report 의 `regime_bias_table` variance > 1e-10 (regime infra 부활 방지)
- `oracle_decomp_artifact_missing` — G0 의 `oracle_decomp.json` 또는 **`cap_saturation_extended.json`** 생성 실패 또는 schema 불일치
- (v1.1 제거 유지) `lb_quota_exhausted` — LB 제출 0 회 정책으로 trigger 부재
- (v1.3 변경) v1.2 의 `corrector_strengthen_failure` → v1.3 에서 `corrector_strengthen_marginal` 로 격하 (G1 robust 가 main 1순위 가 됨)
- (v1.3 변경) v1.2 의 `ranking_loss_marginal` warn-only → v1.3 에서 `ranking_loss_failure` severe 로 격상 (ranking 이 main 1순위)

### Plan-specific paths (WORKFLOW.md §12.5/§12.6 default 위 추가/제외)

- whitelist 추가:
  - `src/pb_0_6822/selector.py` (ranking loss section + 조건부 Set Transformer arch swap, partial — G1 main, G5 조건부)
  - `src/pb_0_6822/boundary.py` (compute_corrector_loss hook 신설 + cap 인자화, partial — G2 main. `train_net()` 본문의 reg 계산 부분 1 줄 + cap 인자화 1 곳)
- whitelist 제외 (blacklist 추가):
  - `src/pb_0_6822/candidates_extended.py` (plan-008 산출 — 본 plan scope X)
  - `src/pb_0_6822/boundary.py` 의 `train_net()` 본문 *외* 영역 + `TinyCorrectionNet` class 구조 변경은 §6.2 의 *arch capacity* sub-experiment 만 (depth/hidden) — class 자체 교체 X
  - **boundary.py 의 `CORRECTOR_CAP` 직접 정수 교체 (v1.3 신규 blacklist, caveat #7 통일) — cap 은 `train_net(corrector_cap: float = 0.006)` 인자화만 허용**
- 참조 (read-only): `runs/baseline/G001_candidate-redefine/**` (plan-008 산출, baseline), `analysis/plan-008/**` (carry-over reference), `analysis/plan-005/corrector_decomp.{md,json}` (★ G2 추가 push 근거)

### Decision-note 사용 예 (자율 결정 시 commit msg 박제)

- `decision-note: spec-default — NDCG@1 의 temperature=0.5 default 채택`
- `decision-note: spec-default — band_weight [0.5cm, 1cm) = 2.0, [1cm, 1.5cm) = 3.0 채택 (plan-008 §7.2 그대로)`
- `decision-note: spec-default — corrector arch capacity sub-exp = depth +1 (2→3 layers) 채택 (hidden 2x 보다 conservative)`
- `decision-note: G0 evidence — cap_saturation_extended = 0.0X (vs plan-005 raw 27 의 3.58%) → corrector cap 확장 의 expected gain anchor 박제`
- `decision-note: conditional-skip — Phase 1+2 OOF=0.79 → Phase 3b/3c (Coarse-to-fine, Set Transformer) skip, G_final 직접 진입`

---

## §1. 배경 / 이전 plan 인계

### §1.1 plan-008 의 핵심 finding (carry-over) + v1.3 재해석

| 항목 | 값 | v1.2 해석 | **v1.3 재해석** |
|---|---|---|---|
| n_oracle_miss (raw err.min > 1cm) | **2812** (28.12%) | [1,1.5cm) 1290 sample 회수 가능 | 동일 — *단* 회수율은 실측 attribution 필요 |
| main_bottleneck (diag c2) | **"ranking"** (gap_ranking 0.0516 ≫ gap_drift -0.0004) | partial truth, corrector 가 진짜 main | **G1 main lever 의 직접 근거**. corrector 와 *둘 다 main* 후보 — fragile/robust 순서로 G1/G2 배치 |
| oracle 1cm (base 27) | 0.7188 | - | - |
| oracle 1cm (extended 25) | **0.7562** | plan-008 산출 | G1 ranking 의 *천장* (G1 만 으로 도달 가능한 OOF 상한) |
| **oracle 1.5cm (raw 27, 실측)** | **0.8478** | ★ v1.2 main lever ceiling | **G2 추가 push 의 ceiling** (G1 위에 corrector 가 더 push) |
| oracle 1.5cm (extended 25, 추정) | ~0.875 추정 (G0 실측) | extended pool +3pp 보정 | G2 의 stretch ceiling |
| OOF (extended pool, plan-008 c7) | **0.6503** | 본 plan 의 baseline | 동일 |
| top1_ranking_acc (extended) | 0.172 | secondary metric | **G1 main metric** |
| gap_ranking (extended) | 0.1119 | secondary lever target | **G1 main target** (≤ 0.09) |
| corrector_oracle_gain (plan-005) | **−0.0077** | main lever 의 직접 fix target | G2 의 direct fix target |
| [0.5, 1cm) corrector hit | 100% → **92.17%** (−7.83pp) | band-specific 으로 fix | G2 의 band-specific 방어 target |
| [1, 1.5cm) corrector hit | 0% → **9.77%** (+9.77pp) | 50%+ 회복 target | **G2 의 30% 회복 target** (v1.2 의 40% → v1.3 보수, cap_saturation 3.58% 반영) |
| **cap_saturation overall_rate** (plan-005) | **0.0358 (3.58%)** | (v1.2 caveat #13 측정 권장) | **★ v1.3 G0 acceptance criterion 격상** — cap 이 binding 아님 → cap 확장 main 가설 약화 → corrector main lever 는 band+arch (additive sub-exp 로 attribution) |

### §1.2 plan-008 의 본질적 결론 + v1.3 재해석

> **v1 framing**: 후보 풀 확장은 oracle 천장만 회복, ranking 부족 (caveat #13) 가 hit follow 막음 → ranking loss main.
>
> **v1.2 framing**: oracle 1cm 기준만 — 1.5cm 기준 ceiling 0.8478 미고려. corrector 가 진짜 main. ranking 은 secondary.
>
> **v1.3 framing (재배치)**: v1.1 과 v1.2 의 main lever 가 *둘 다 valid* 후보 — 어느 게 main 인지는 *cap_saturation_extended G0 실측* + *G1/G2 attribution* 으로 사후 확정. 본 plan 의 *commit chain 자원* 은:
> - **fragile/robust 위험도로 순서 배치**: ranking (selector.py partial only, robust) → G1, corrector (boundary.py + arch + cap + loss 4 곳, fragile) → G2.
> - G1 fail 시도 baseline 0.6503 유지 (selector.py revert).
> - G2 fail 시 G1 의 +0.02~0.04 retention 보장 (G2 가 G1 위에서 측정되므로).
> - corrector sub-exp 는 *additive ablation* (cap만 / band만 / arch만 / all) — v1.2 의 cumulative 보다 informative (cap_saturation 3.58% 실측의 직접 검증 가능).
> - v1.2 의 oracle 1.5cm = 0.8478 발견은 **유지** — *target LB 0.74~0.80 의 상한 anchor* 로 사용.

### §1.3 plan-008 의 carry-over 2 항목 (v1.3 배치)

1. **plan-008.1 LB 회수** — `submission_step3.csv` (G001-candidate-redefine 의 Variant A path 산출). **본 plan 내 미수행** (할당량 소진), plan-009.1 carry-over 와 묶음 (다음 날 사용자 수동 dacon-submit 호출). 본 plan G0 = preflight 만 (산출 verify + LB 추정 anchor 박제).
2. **boundary.py compute_corrector_loss hook 신설** — plan-008 c9 진입 시 `LOSS_ATTR` 부재 확정. plan-008 G3 DEFERRED 의 직접 회수. **v1.3 배치**: 본 plan G2 의 *전제 조건* (c6 commit). G1 (ranking) 은 boundary.py touch X — G1 robustness 확보.

### §1.4 가설 (H1~H5, v1.3 재배치)

| ID | 가설 | 검증 metric | 합격 기준 |
|---|---|---|---|
| **H1 ★ G1 main robust** | **Ranking-specific loss (NDCG@1 + pairwise + ListMLE) 가 plan-008 의 gap_ranking 0.1119 → ≤ 0.09 + top1_acc 0.172 → ≥ 0.22 → OOF +0.02~0.04. selector.py partial only, arch 보존, boundary.py touch X.** | OOF + top1_acc + gap_ranking | **G1 (OOF ≥ 0.6703, top1 ≥ 0.22, gap ≤ 0.09)** |
| **H2 ★ G2 main 추가 push** | **Corrector 강화 (cap 인자화 + band-specific loss + arch capacity, additive 4 sub-exp) 가 G1 selector 위에서 [1,1.5cm) band 의 30%+ 회수 + [0.5,1cm) 깎는 부작용 방어 → OOF +0.03~0.06. additive ablation 으로 각 lever (cap/band/arch) attribution 측정.** | OOF + per-band hit + corrector_oracle_gain | **G2 (OOF ≥ G1+0.03, [1,1.5cm) ≥ 0.30, [0.5,1cm) ≥ 0.95, gain ≥ 0)** |
| H3 cheap | Hard top-K filter (test-time, 1 줄) 가 softmax centroid drift 직접 fix → OOF + 0.005 marginal | OOF | G3 (OOF ≥ G2 + 0.005) |
| H4 조건부 | Coarse-to-fine 2-stage 가 search space 5 로 축소 → ranking 정확도 ↑ → OOF + 0.02 | OOF | G4 (OOF ≥ Phase 1+2 + 0.02) |
| H5 조건부 | Set Transformer (cand_i ↔ cand_j attention) 가 GRU 한계 우회 → OOF + 0.03 | OOF + top1_acc | G5 (OOF ≥ G4 + 0.03, top1 ≥ 0.30) |

---

## §2. Scope (명시적)

### §2.1 In-scope

| 항목 | 값 | 출처 |
|---|---|---|
| Baseline candidate pool | EXTENDED 25 (plan-008 c7 산출 그대로) | plan-008 |
| Baseline selector hyperparam | plan-008 c7 의 Variant A path (regime_prior_strength=0) | plan-008 |
| **★ G1 main — Loss component 추가** | NDCG@1 differentiable + pairwise margin × 2.0 + ListMLE on selector.py partial | 본 plan §5 |
| **★ G2 main — boundary.py hook 신설** | `compute_corrector_loss(pred, target, raw, weight)` module-level callable, **cap 인자화 (default 0.006 보존)** | plan-008 §7 carry-over, v1.3 통일 |
| **★ G2 main — Corrector cap override** | wrapper 에서 0.006 → **0.012** (1.2cm shift, [1,1.5cm) band cover) — G2 sub-exp a/d 에서만 | v1.3 main |
| **★ G2 main — Band-specific loss** | weight [0,0.5cm)=1.0, [0.5,1cm)=2.0, [1,1.5cm)=3.0, [1.5cm,∞)=0.5 — G2 sub-exp b/d 에서만 | plan-008 §7.2 재사용 |
| **★ G2 main — Corrector arch capacity** | TinyCorrectionNet depth +1 (2→3 layers) — G2 sub-exp c/d 에서만 | v1.3 신규 |
| Test-time filter (cheap) | Hard top-K (K ∈ {3,5,7} grid) | 본 plan §7 |
| (조건부) Multi-stage | Coarse-to-fine 2-stage | 본 plan §8 |
| (조건부) Arch swap | Set Transformer 1 layer (cand_i ↔ cand_j) | 본 plan §9 |
| **G0 oracle_decomp** | extended 25 cands best-raw-err 8-bin + 1cm/1.5cm/2cm ceiling 측정 | v1.2 신규 유지 |
| **G0 cap_saturation_extended (v1.3 신규)** | extended 25 cands 의 cap=0.006 binding rate 측정 → plan-005 의 3.58% 재현 여부 → corrector cap 확장 의 expected gain anchor | v1.3 신규 |
| LB 제출 | **0 회** (v1.1 유지). submission 박제만, plan-009.1 carry-over | 본 plan §10 |

### §2.2 Out-of-scope (절대 안 함)

| 항목 | 이유 |
|---|---|
| 후보 풀 재정의 (greedy set-cover 재시도, 새 family 추가) | plan-008 산출 그대로 사용 |
| Regime infra 재도입 (`regime_prior_strength > 0`, regime_bias_table) | plan-005 STAGE 6 + plan-008 검증 결과 무용 |
| GRU hidden 변경 (32→64, layer 추가) | plan-008 §6.5 fallback skip 결정 + 본 plan H1 (ranking 효과) 분리 |
| 후보 좌표 보정 (CMA-ES 단일 공식 재시도, plan-007 후속) | plan-007 framework 대체 시도 실패 |
| boundary.py 의 `train_net()` 본문 *외* 수정 + `TinyCorrectionNet` class 자체 교체 | partial 수정 영역 whitelist 명시 — depth/hidden 변경만 |
| Corrector cap > 0.015 (1.5cm 이상 shift) | oracle 1.5cm ceiling *너머* 회수는 [1.5, 2cm) band 의 384 sample (3.84pp) 만 — ROI 낮음 + cap 1.5cm 는 좌표 *오버슛* 위험 |
| **boundary.py CORRECTOR_CAP 직접 정수 교체** (v1.3 신규) | caveat #7 통일 — cap 은 *인자화* (default 0.006 보존 + wrapper override). 기존 plan-004/005/008 backward compat 확보. |
| LB 제출 (v1.1 유지) | **본 plan 내 0 회** (할당량 소진 인계). submission.csv 박제만, plan-009.1 carry-over (plan-008.1 와 묶음) |

---

## §3. 사전 등록 (Pre-registration)

### §3.1 Fold split

- plan-008 c7 의 5-fold split 그대로 (plan-004 default). seed=42.
- OOF 정의: `np.concatenate([fold_predictions for fold in 5])` 의 hit_rate.

### §3.2 합격 기준 (§0.5 의 G-gate 와 동일)

(§0.5 참조 — 본 §3.2 는 후속 검증 시 anchor)

### §3.3 평가 점수 (v1.3 재배치)

- **1 차 metric**: `oof_soft_hit` (boundary.softmax_temperature=0.03 적용 후 hit @ 1cm)
- **2 차 metric (G1 main)**: `top1_ranking_acc` (selector argmax pick = oracle best 일치 비율) + `gap_ranking` (oracle 1cm hit − oof_soft_hit)
- **3 차 metric (G2 main)**: per-band hit — `[0.5, 1cm) hit_after`, `[1, 1.5cm) hit_after`, `corrector_oracle_gain` (plan-005 corrector_decomp 정합)
- **G0 신규 metric**: `oracle_1cm`, `oracle_1.5cm`, `oracle_2cm`, `n_in_band_[1,1.5cm)`, **`cap_saturation_extended` (cap=0.006 binding rate, plan-005 의 3.58% 재현 여부)**

---

## §4. STAGE 0 — Preflight + Oracle Decomp + Cap Saturation Extended (G0, v1.3 확장)

> **v1.3 변경**: 기존 v1.2 의 oracle_decomp 위에 **cap_saturation_extended.py 신설** — extended 25 cands 의 cap=0.006 binding rate 재측정. corrector main framing 의 evidence anchor.

### §4.1 Preflight 작업

1. plan-008 산출 5 submission variant 존재 확인:
   - `runs/baseline/G001_candidate-redefine/submission_step3.csv`
   - `runs/baseline/G001_candidate-redefine/submission_attn_gru_selector_soft.csv`
   - `runs/baseline/G001_candidate-redefine/submission_boundary_tiny_{argmax,soft}.csv`
   - `runs/baseline/G001_candidate-redefine/submission_selector_ensemble_{argmax,soft}.csv`
2. plan-008 metric 4 항목 verify (analysis/plan-008/selector_retrain.json 참조).
3. `analysis/plan-009/lb_baseline.json` 신설 (v1.1 spec 그대로).

### §4.2 Oracle Decomp + Cap Saturation Extended 작업 (v1.3)

`analysis/plan-009/oracle_decomp.py`:

```python
# extended 25 cands 의 best-raw-cand error 8-bin 분포 측정 (plan-005 패턴)
err_per_cand = np.linalg.norm(cands - target[:, None, :], axis=-1)  # (N, 25)
best_err = err_per_cand.min(axis=1)  # (N,)

bins = [0.0, 0.005, 0.01, 0.015, 0.02, 0.03, 0.05, 0.10, np.inf]
hist = np.histogram(best_err, bins=bins)[0]

oracle_1cm   = float((best_err <= 0.01).mean())   # plan-008 의 0.7562 재현
oracle_1_5cm = float((best_err <= 0.015).mean())  # 진짜 ceiling
oracle_2cm   = float((best_err <= 0.02).mean())
n_in_band_1_15 = int(((best_err > 0.01) & (best_err <= 0.015)).sum())
```

`analysis/plan-009/cap_saturation_extended.py` (v1.3 신규):

```python
# extended 25 cands 의 corrector 학습 후 cap=0.006 binding rate 재측정
# plan-005 raw 27 cands 의 0.0358 재현 여부 확인 — corrector main framing 의 evidence
# 1) baseline corrector 학습 (cap=0.006, band=off, arch=default — plan-005 spec 그대로)
# 2) train 후 corrector shift magnitude 측정: shift = ||pred - raw|| per (sample, cand)
# 3) cap_saturation = (shift >= 0.0057).mean()  # saturation threshold 0.0057 = cap × 0.95
# 4) per-candidate breakdown
# 5) plan-005 의 overall 0.0358 와 비교 →
#    - "재현 ±0.01 (≤0.05)" = corrector cap 확장 의 expected gain *약* (band/arch 가 main lever)
#    - "재현 +0.05+ (≥0.08)" = cap 확장 의 expected gain *강* (cap 도 main lever 후보)
```

### §4.3 합격 기준

- plan-008 산출 5 submission 모두 존재 (1 개라도 부재 → severe `plan_008_artifact_missing`)
- plan-008 metric 4 항목 모두 산출 verify
- `oracle_decomp.json` 생성 + `oracle_1cm` ≈ 0.7562 ± 0.002 (plan-008 재현) + `oracle_1.5cm ≥ 0.85` (1.5cm ceiling 확인)
- **`cap_saturation_extended.json` 생성** + overall_rate 측정값 박제 (plan-005 의 0.0358 와 비교 anchor)
- `lb_baseline.json` 생성
- 위반 시 `oracle_decomp_artifact_missing` severe

### §4.4 산출물

- `analysis/plan-009/preflight.py` + `lb_baseline.json`
- `analysis/plan-009/oracle_decomp.py` + `oracle_decomp.json` + `oracle_decomp.md` (8-bin 표 + ceiling 박제)
- **`analysis/plan-009/cap_saturation_extended.py` + `cap_saturation_extended.json`** (v1.3 신규, corrector main framing evidence anchor)

---

## §5. STAGE 1 ★ Phase 1 — Ranking loss (G1 main robust)

> **v1.3 격상**: 기존 v1.2 §6 (secondary, on corrector) → v1.3 §5 (G1 main, baseline 위에서 직접). selector.py partial only — arch 보존, boundary.py touch X. Robust lever 진입으로 fragile corrector 의 G2 baseline 보장.

### §5.1 Loss 3 component 정의

`src/pb_0_6822/selector.py` 의 학습 loop 에 다음 3 loss term 을 *추가* (existing CE loss 와 weighted sum):

**1. NDCG@1 differentiable** (★ 최고 ROI 카테고리):
```python
soft = F.softmax(score / temperature, dim=-1)  # temperature=0.5
loss_ndcg1 = (1.0 - soft.gather(-1, oracle_best_idx.unsqueeze(-1))).mean()
```

**2. Pairwise margin × 2.0**:
```python
pair_score_diff = score.gather(-1, sorted_pair[..., 0]) - score.gather(-1, sorted_pair[..., 1])
loss_pair = F.relu(0.1 - pair_score_diff).mean() * 2.0
```

**3. Listwise ListMLE**:
```python
log_probs = []
score_sorted = score.gather(-1, permutation)  # err ascending 정렬
for k in range(C):
    log_norm = torch.logsumexp(score_sorted[:, k:], dim=-1)
    log_probs.append(score_sorted[:, k] - log_norm)
loss_listmle = -torch.stack(log_probs, dim=-1).sum(-1).mean()
```

**Total loss**:
```python
loss_total = (
    loss_ce * 1.0 + loss_ndcg1 * 1.0 + loss_pair * 1.0 + loss_listmle * 0.5
)
```

### §5.2 학습 wrapper (Variant A baseline)

`analysis/plan-009/ranking_loss_train.py`:
- plan-008 c7 의 baseline corrector (cap=0.006, band=off, arch=default) 위에서 selector 만 retrain.
- `selector.SELECTOR_MAIN([..., '--regime-prior-strength', '0', '--loss-components', 'ndcg1,pair2x,listmle', '--K-pairs', '10', ...])`

### §5.3 합격 기준 (G1)

| 측정 | spec | 위반 시 |
|---|---|---|
| `oof_soft_hit ≥ 0.6703` | plan-008 c7 0.6503 + 0.02 minimum, stretch 0.69 | severe `ranking_loss_failure` |
| `top1_ranking_acc ≥ 0.22` | plan-008 0.172 + 0.05 회복 | severe (위와 동일) |
| `gap_ranking ≤ 0.09` | plan-008 0.1119 − 0.02 회복 | severe (위와 동일) |
| `variant_a_safe` (regime_bias variance < 1e-10) | plan-008 c7 assert 그대로 | severe `variant_a_residue` |

### §5.4 산출물

- `src/pb_0_6822/selector.py` partial (ranking loss 3 component)
- `analysis/plan-009/ranking_loss_summary.json`
- `runs/baseline/H001_ranking-loss/` (5 submission variant)

### §5.5 Fallback (G1 미달 시)

severe 발동 → 멈춤. 사용자 결정 후 재개:
- 옵션 a: weight 조정 (NDCG@1 × 2.0, pair × 1.0)
- 옵션 b: ListMLE drop (gradient 불안정 의심)
- 옵션 c: temperature 0.5 → 0.3
- 옵션 d: G1 skip, G2 (corrector 강화) 만 진행 — baseline 위에서 corrector main lever 단독 측정
- 옵션 e: plan-010 carry-over

---

## §6. STAGE 2 ★ Phase 2 — Corrector 강화 (G2 main 추가 push, additive ablation)

> **v1.3 변경**: 기존 v1.2 §5 (cumulative 3 sub-exp, main) → v1.3 §6 (additive 4 sub-exp, G2 main 추가 push, on G1 selector). cumulative ablation 의 정보량 부족 fix — 각 lever (cap/band/arch) 의 attribution 측정 가능.

### §6.1 boundary.py hook 신설 + cap 인자화 (`src/pb_0_6822/boundary.py` partial)

```python
# 신규 module-level 함수 (1 함수 신설)
def compute_corrector_loss(pred, target, raw=None, weight=None):
    """Default L2 loss. monkey-patch 가능 hook. plan-008 §7 carry-over."""
    reg = ((pred - target) ** 2).sum(dim=1)
    if weight is not None:
        reg = reg * weight
    return reg.mean()

# train_net() 내부 reg 계산 부분 1 줄 교체
# 기존: reg = ((pred - yb) ** 2).sum(dim=1).mean()
reg = compute_corrector_loss(pred, yb)

# Cap 인자화 (v1.3 통일, caveat #7 binding)
# 기존: CORRECTOR_CAP = 0.006 (module constant) → 직접 정수 보존 (default backward compat)
# train_net() 가 cap 인자 받도록 signature 확장:
def train_net(..., corrector_cap: float = 0.006):
    ...
    # cap 사용처에서 CORRECTOR_CAP 대신 corrector_cap 변수 사용
```

→ plan-009 wrapper 에서 `corrector_cap=0.012` override. 기존 plan-004/005/008 invocation 은 default 0.006 으로 backward compat.

### §6.2 Corrector 강화 — additive 4 sub-experiments

`analysis/plan-009/corrector_strengthen.py`:

| sub-exp | cap | band-specific | arch capacity | 측정 |
|---|---|---|---|---|
| **a** (cap만) | 0.012 | off (default L2) | default (hidden=16, depth=2) | cap 확장 단독 효과 → cap_saturation 검증 |
| **b** (band만) | 0.006 (default) | on (weight 1/2/3/0.5) | default | band-specific 단독 효과 |
| **c** (arch만) | 0.006 (default) | off | depth +1 (2→3 layers) | arch capacity 단독 효과 |
| **d** (all) | 0.012 | on | depth +1 | 3 lever 의 compound 효과 |

각 sub-exp 의 attribution = `OOF_x − OOF_baseline` (x ∈ {a, b, c, d}). compound vs sum 비교: `OOF_d` vs `OOF_a + OOF_b + OOF_c − 2 × OOF_baseline` → super-additive (compound > sum) / sub-additive / additive 분류.

**Band-specific loss spec** (b/d 에서 사용):

```python
def band_specific_corrector_loss(pred, target, raw=None, weight=None):
    err = torch.linalg.norm(target, dim=-1)  # (B,)
    band_weight = torch.where(
        err < 0.005, 1.0,              # [0, 0.5cm) baseline
        torch.where(err < 0.010, 2.0,  # [0.5, 1cm) 2x — plan-005 corrector 가 깎은 영역
        torch.where(err < 0.015, 3.0, 0.5))  # [1, 1.5cm) 3x — corrector 회수 target
    )
    reg = ((pred - target) ** 2).sum(dim=-1) * band_weight
    return reg.mean()

# monkey-patch
import src.pb_0_6822.boundary as boundary
boundary.compute_corrector_loss = band_specific_corrector_loss
```

**Best sub-exp 채택 기준**: OOF max (G1 selector 위에서). 단 attribution 정보 박제 (`analysis/plan-009/corrector_attribution.json`) — plan-010 의 sub-lever 선정 anchor.

### §6.3 합격 기준 (G2)

| 측정 | spec | 위반 시 |
|---|---|---|
| `oof_soft_hit (best sub-exp) ≥ G1 OOF + 0.03` | additive minimum, stretch G1 + 0.06 | warn-only `corrector_strengthen_marginal` (G1 OOF ≥ 0.70 시) / severe (G1 OOF < 0.70 시) |
| `[1, 1.5cm) hit_after ≥ 0.30` | plan-005 의 9.77% → 3x 회복 (v1.2 의 40% 보다 보수, cap_saturation 3.58% 반영) | warn-only / severe (위와 동일 분기) |
| `[0.5, 1cm) hit_after ≥ 0.95` | plan-005 의 92.17% → 95%+ (깎는 부작용 방어) | warn-only / severe (위와 동일 분기) |
| `corrector_oracle_gain ≥ 0` | plan-005 의 −0.0077 → 양수 회복 | warn-only / severe (위와 동일 분기) |
| `variant_a_safe` (regime_bias variance < 1e-10) | plan-008 c7 assert 그대로 | severe `variant_a_residue` |
| **attribution 박제** (cap/band/arch 각 ΔOOF) | informativeness 확보 | (필수, severe 아님 — best 선정 무관 박제) |

### §6.4 산출물

- `src/pb_0_6822/boundary.py` partial (compute_corrector_loss hook + cap 인자화)
- `analysis/plan-009/corrector_strengthen.{py,json}` (4 sub-exp 비교 + best 박제)
- `analysis/plan-009/corrector_attribution.json` (sub-lever 별 ΔOOF, plan-010 anchor)
- `runs/baseline/H002_corrector-strengthen/` (best sub-exp 의 5 submission variant)

### §6.5 Fallback (G2 미달, severe path — G1 OOF < 0.70 일 때)

- 옵션 a: band_weight 조정 ([1,1.5cm)=3.0 → 5.0, [0.5,1cm)=2.0 → 1.5)
- 옵션 b: cap 0.012 → 0.015 (1.5cm shift, [1.5, 2cm) band 추가 cover, 단 over-shoot 위험)
- 옵션 c: arch 추가 강화 (depth 3 → 4, hidden 16 → 32)
- 옵션 d: arch 자체 교체 (TinyCorrectionNet → ResNet block, scope 외 → plan-010 carry-over)

### §6.6 Fallback (G2 미달, warn-only path — G1 OOF ≥ 0.70 일 때)

G1 robust 가 이미 strong → G2 marginal 손실 허용. best sub-exp (있다면 OOF ≥ G1) submission 박제 + G3 진입. attribution 정보는 plan-010 의 corrector sub-lever 선정 anchor 로 carry-over.

---

## §7. STAGE 3a Phase 3a — Hard top-K filter (cheap, G3)

### §7.1 구현

test-time only, 1 줄 추가:
```python
if top_k_filter is not None:
    topk_vals, topk_idx = score.topk(top_k_filter, dim=-1)
    mask = torch.full_like(score, float('-inf'))
    mask.scatter_(-1, topk_idx, topk_vals)
    score = mask
```

### §7.2 Grid search

K ∈ {3, 5, 7}. G2 산출 모델 (H002) 의 5-fold OOF prediction 재사용 → 3 회 inference 만.

### §7.3 합격 기준 (G3)

| 측정 | spec | 위반 시 |
|---|---|---|
| `oof_soft_hit (best K) ≥ G2 OOF + 0.005` | marginal booster | warn-only `topk_marginal` |
| best K 박제 | 3/5/7 중 OOF max | (필수) |

### §7.4 산출물

- `analysis/plan-009/topk_filter_grid.json`
- `runs/baseline/H003_topk-filter/submission_*.csv` (best K)

---

## §8. STAGE 3b Phase 3b — Coarse-to-fine 2-stage (G4, 조건부)

### §8.1 진입 조건

Phase 1+2+3a 누적 OOF < **0.78** 일 때만. 0.78 이상이면 G4 skip (decision-note: `conditional-skip — Phase 1+2 saturation reached`).

### §8.2 구현

`analysis/plan-009/coarse_to_fine.py`:
- **Stage 1 cheap filter** (학습 X): cosine sim 기반 25 → top-5
- **Stage 2 expensive rerank** (학습 O): top-5 만 input 으로 selector retrain

### §8.3 합격 기준 (G4)

| 측정 | spec | 위반 시 |
|---|---|---|
| `oof_soft_hit ≥ Phase 1+2 OOF + 0.02` | 2-stage 효과 minimum | severe `coarse_to_fine_failure` |
| `top1_ranking_acc (Stage 2 내)` ≥ 0.40 | search space 5 효과 | severe (위와 동일) |
| Stage 1 hit-rate ≥ 0.95 | top-5 안에 oracle best 포함 비율 | severe `stage1_recall_loss` |

### §8.4 산출물

- `analysis/plan-009/coarse_to_fine_summary.json`
- `runs/baseline/H004_coarse-to-fine/`

---

## §9. STAGE 3c Phase 3c — Set Transformer arch swap (G5, 조건부)

### §9.1 진입 조건

G1+G2+G3+G4 누적 OOF < **0.75** 일 때만. 미달 시만 — *high risk* (overfit, data 10K).

### §9.2 구현

`src/pb_0_6822/selector.py` partial:
- GRU hidden 32 + Set Transformer 1 layer (cand_i ↔ cand_j) fusion
- 신규 파라미터: `--use-set-transformer True`, `--st-num-heads 4`, `--st-hidden 32`

### §9.3 합격 기준 (G5)

| 측정 | spec | 위반 시 |
|---|---|---|
| `oof_soft_hit ≥ G4 OOF + 0.03` | arch swap big lever | severe `arch_swap_failure` |
| `top1_ranking_acc ≥ 0.30` | ranking 직접 회복 | severe (위와 동일) |
| `variant_a_safe` | regime_bias 부재 | severe `variant_a_residue` |

### §9.4 산출물

- `runs/baseline/H005_set-transformer/`
- `analysis/plan-009/set_transformer_summary.json`

---

## §10. STAGE N — Synthesis + best Phase submission 박제 + plan-009.1 carry-over (G_final)

> **v1.1 유지**: LB 제출 0 회. best Phase submission 의 *경로* 만 박제 + plan-009.1 carry-over instruction (plan-008.1 와 묶음).

### §10.1 best Phase 선정 + submission 박제 (LB 미제출)

- 후보 submission: G1 (Phase 1 ranking, **★ G1 main**), G2 (Phase 2 corrector additive best on G1, **★ G2 main**), G3 (Phase 3a top-K), G4 (Phase 3b, 조건부), G5 (Phase 3c, 조건부)
- best OOF 산출 1 개 선정 (5-fold soft hit max).
- best submission path 를 `analysis/plan-009/results.md` 의 frontmatter 필드로 박제 (예: `best_submission: runs/baseline/H00X_<exp>/submission_<variant>.csv`).
- plan-009.1 carry-over instruction 박제 (다음 날 사용자 수동 dacon-submit 1~2 회 호출):
  - 1st: plan-008 의 `submission_step3.csv` (plan-008.1)
  - 2nd: plan-009 의 best submission (plan-009.1)

### §10.2 시나리오 분기 (LB 추정 = OOF + 0.022 gap, v1.3 보수 조정)

| 시나리오 | OOF | LB 추정 | 다음 plan 권장 |
|---|---|---|---|
| **A+** (상위) | ≥ 0.78 | ≥ 0.80 | G1 ranking + G2 corrector *모두* compound → plan-010 main = **arch swap 또는 non-parametric** (Set Transformer / KNN / GP) 으로 추가 push |
| A (목표) | 0.74~0.78 | 0.76~0.80 | G1+G2 compound 정상 → plan-010 main = corrector arch 추가 강화 (depth 3→4) + ranking compound |
| B (보통) | 0.70~0.74 | 0.72~0.76 | G1 만 효과 (G2 marginal) → plan-010 main = corrector arch 자체 교체 (TinyCorrectionNet → ResNet block) |
| C (낮음) | 0.67~0.70 | 0.69~0.72 | G1 marginal → plan-010 main = framework 교체 (plan-006 회귀 또는 KNN/GP 단독) |
| D (실패) | < 0.67 | < 0.69 | G1 ranking 한계 + G2 corrector 한계 → plan-010 main = data augmentation / feature 추가 |

본 plan 내 LB 미회수 → 시나리오 *확정* 은 plan-009.1 carry-over 회수 후. carry-over 시점에 OOF→LB gap actual 측정 + 시나리오 anchor 갱신.

### §10.3 results.md 필수 항목

- §1 요약 + §2 OOF 표 (LB 는 *추정* + carry-over TBD) + §3 per-Phase contribution (Δ OOF) + §4 G2 corrector attribution (a/b/c/d 각 ΔOOF, **plan-010 sub-lever 선정 anchor**) + §5 per-band Δ table (plan-005 corrector_decomp 패턴) — [0.5,1cm) hit_before/after/Δ, [1,1.5cm) hit_before/after/Δ, corrector_oracle_gain + §6 caveat 검증 결과 + §7 decision-note list + §8 plan-010 후보 ≥ 2 + §9 변경 이력 + §10 plan-009.1 carry-over instruction (plan-008.1 와 묶음)
- frontmatter: `lb_score: null` + `status: partial (carry-over to plan-009.1 for LB submission)` + `best_submission: <path>`

---

## §N+1. 작업량 총 회계 (v1.3)

| Phase | commit 수 | runtime 추정 |
|---|---|---|
| c1.3 (v1.3 docs) | 1 | < 1 min |
| G0 preflight + oracle_decomp + **cap_saturation_extended** (v1.3 확장) | 2 (c2, c2.1) | 2~3 min (cap_saturation 측정 +1min) |
| **G1 Phase 1 ★ Ranking loss (robust main)** | **3 (c3, c4, c5)** | **4~5 min (5-fold selector retrain)** |
| **G2 Phase 2 ★ Corrector 강화 (additive 4 sub-exp)** | **3 (c6, c7, c8)** | **12~15 min (4 sub-exp × ~3min boundary fit, v1.2 의 9~10min 보다 +3~5min)** |
| G3 Phase 3a top-K filter | 2 (c9, c10) | 1 min (inference only) |
| G4 Phase 3b Coarse-to-fine (조건부) | 2 (c11, c12) | 4~5 min |
| G5 Phase 3c Set Transformer (조건부) | 2 (c13, c14) | 5~7 min |
| G_final synthesis (LB 미제출) | 1 (c16) | 1 min |
| **총 (모든 Phase)** | **17 commits** | **~28 min** |
| **총 (Phase 3b/3c skip, 조건부 saturation)** | **13 commits** | **~19 min** |

→ v1.2 의 16 commits / 25min → v1.3 의 17 commits / 28min (+1 commit, +3min, additive 4 sub-exp 의 informativeness trade-off)

---

## §N+2. results.md 필수 항목 (§10.3 참조)

---

## §N+3. 통계 함정 & caveats

1. **NDCG@1 의 temperature 선택** — temperature=0.5 default. 너무 sharp (0.1) → gradient vanish, 너무 soft (1.0) → 효과 없음. G1 미달 시 fallback 옵션 c (0.5→0.3) 시도.
2. **ListMLE gradient 불안정** — 후보 25 개 의 permutation log-prob 합산. 후속 후보 (k=20~25) 의 log-norm 이 작아 gradient noise. weight 0.5 로 절반.
3. **Pairwise margin 0.1 hyperparam** — sorted pair 의 score 차이가 0.1 미만이면 loss 발생. 0.1 = logit 0.063 (margin p50, plan-008 측정) 의 2배 — 적절 추정.
4. **top-K filter K=5 default 의 근거** — plan-008 H001 oracle 천장 0.7562 는 *25 후보 전체*. K=5 로 축소 시 oracle 천장 감소 가능. G2 의 산출에서 top-5 hit ratio 사전 측정 가능.
5. **Coarse-to-fine 의 Stage 1 cheap filter 정확성** — cosine sim 기반 top-5 가 oracle best 를 놓치면 (Stage 1 recall < 0.95) Stage 2 무관 fail. §8.3 assert 필수.
6. **Set Transformer overfit risk** — data 10K, 후보 25, head 4 → overfit 우려 *high*. early stop + L2 reg 강화 필수.
7. **(v1.3 통일) boundary.py partial 수정의 backward_compat — cap 인자화 binding** — `compute_corrector_loss` 신설 시 *기존 default 동작 보존* 필수. `tests/backward_compat/` 의 plan-004/005 smoke test 통과 verify. **cap 은 직접 정수 교체 X — `train_net(corrector_cap: float = 0.006)` 인자화** (default 0.006 보존 + plan-009 wrapper 에서 `corrector_cap=0.012` override). v1.2 의 §5.1 직접 정수 교체 spec 과의 충돌 해소.
8. **Band weight 의 normalization** — band_specific_corrector_loss 의 weight (1, 2, 3, 0.5) 가 total loss magnitude 를 변경 → learning rate 영향 가능. (a) weighted mean (default 보존) (b) 또는 lr × 0.5 보정.
9. **(v1.1 유지 + v1.3 추가) LB 제출 0 회 + OOF→LB gap 추정 신뢰도** — 본 plan 내 LB 제출 *0 회* (할당량 소진 인계). 모든 Phase 의 submission.csv 는 *생성·박제만*. plan-008.1 carry-over + plan-009.1 carry-over 묶음 = 다음 날 사용자 수동 dacon-submit 호출. **(v1.3 추가) corrector arch + cap + loss 동시 변경은 학습 dist 변경 → OOF→LB gap 안정성 약화 가능**. plan-005/008 의 +0.022 gap 추정은 selector 변경 위주 — corrector main lever 변경 시 gap drift 가능 → carry-over 시점 actual 측정 후 plan-010 anchor 갱신.
10. **plan-008 의 family_effect +0.0037 의 함의** — 후보 풀 확장 ROI marginal. 본 plan ranking + corrector 강화 가 family 위에서 동작 — *후보 풀 변경 X* 결정의 직접 근거.
11. **Variant A regime_bias variance check** — selector report 의 `regime_bias_table` 의 variance > 1e-10 시 `variant_a_residue` severe. G1 (ranking) 과 G2 (corrector) 양쪽에서 verify.
12. **top1_ranking_acc 측정 정의** — `argmax(selector_score) == argmin(per_candidate_err)` 의 sample 비율. plan-008 c7 의 0.172 와 *동일 정의* 로 비교 필수.
13. **(v1.3 G0 격상) Corrector cap_saturation_extended evidence** — plan-005 의 0.0358 (cap=0.006 raw 27) → extended 25 cands 의 cap=0.006 saturation rate 재측정. **G0 acceptance criterion 격상** (v1.2 의 caveat 권장 → v1.3 의 필수 측정). 결과 분기:
    - 재현 (≤ 0.05): cap 이 binding 아님 → G2 sub-exp a (cap만) 의 expected gain 약. attribution 측정 후 band/arch 가 main lever 확정 가능.
    - 강화 (≥ 0.08): cap 도 binding lever → G2 sub-exp a 도 main 후보. 4 sub-exp attribution 측정 의 informativeness 강화.
14. **(v1.3 보수 조정) [1, 1.5cm) hit ≥ 0.30 의 회복률 가정** — plan-005 의 9.77% (cap=0.006) → 30% (cap=0.012 + band + arch) 는 *3x 회복*. v1.2 의 4x (40%) 보다 보수. 근거: (a) cap_saturation 3.58% 실측 → cap 확장의 직접 효과 약. (b) band-specific loss 의 [1,1.5cm) weight 3x → loss gradient 강화. (c) arch capacity 강화 → small/large shift 분리 학습. 30% 달성 시 OOF +0.03~0.05 (cap_saturation 약 시) 또는 +0.05~0.07 (cap_saturation 강 시).
15. **(v1.2 유지) Oracle 1.5cm ceiling 의 extended pool 보정** — plan-005 의 1.5cm = 0.8478 은 raw 27 cands. extended 25 cands 의 1.5cm 는 G0 의 `oracle_decomp.json` 에서 실측. plan-008 의 oracle 1cm +3.7pp 회복 (0.7188 → 0.7562) 패턴 답습 시 oracle 1.5cm 도 +2~3pp 추정 → ~0.87~0.88. 본 plan 의 LB target 0.74~0.80 는 이 ceiling 의 88~92% 도달 가정.
16. **(v1.3 신규) G1 → G2 ordering 의 risk 분리** — G1 (ranking) 은 selector.py partial only (arch 보존, boundary.py touch X) → G1 fail 시도 baseline 0.6503 retention 가능 (selector.py revert). G2 (corrector) 는 boundary.py + arch + cap + loss 4 곳 동시 변경 → fragile. G2 가 G1 위에서 측정되므로 G2 fail 시 G1 의 +0.02~0.04 retention 보장. v1.2 의 fragile-first ordering 의 risk 해소.
17. **caveat #13 (plan-008 §N+3) — ranking 한계 framework 본질** — 본 plan G1 = ranking 직접 측정. G1 OOF < 0.6703 시 caveat #13 직접 검증 (framework 자체 한계 vs loss 부족). Phase 3c (Set Transformer) 진입 시 동일 risk 적용.
18. **(v1.3 신규) G2 corrector attribution 의 informativeness 가치** — additive 4 sub-exp (cap만/band만/arch만/all) 의 ΔOOF 측정 자체가 *G2 OOF 결과와 독립적인 산출물*. G2 OOF 가 marginal (warn-only) 이어도 attribution 정보 = plan-010 의 corrector sub-lever 선정 anchor (예: band 가 +2pp, arch 가 +1pp, cap 이 +0.5pp 면 plan-010 = band 강화 main). 즉 G2 entry 의 informativeness ROI 는 OOF 달성과 무관하게 확보.
19. **(v1.3 삭제 명시) v1.2 caveat #16 (compound 효과 의문) 의 reasoning 오류** — v1.2 의 "G1 corrector 성공 시 selector 의 ranking 부담 *감소*" 주장은 cand 수 동일 + hit zone 후보 수 ↑ 시 *tie-breaking 부담 증가* 로 사실은 *유지 또는 증가*. v1.3 의 G1 ranking → G2 corrector 순서로 *해당 framing 자체 무관* (G2 가 G1 위 compound 측정) — caveat 삭제.

---

## §N+4. 변경 이력

- v1 (2026-05-12): 초안. plan-008 의 main_bottleneck="ranking" 결론 + carry-over 2 항목 (LB + corrector hook) 통합. Phase 1 (cheap, no arch) + Phase 2 (mid) + Phase 3 (big, 조건부) sequence 채택.
- v1.1 (2026-05-12): **LB 제출 정책 0 회로 변경** (할당량 소진 인계). G0 = preflight (LB 회수 → plan-008.1 그대로 carry-over) + G_final = best submission *경로* 박제 (LB 미제출, plan-009.1 carry-over). caveat #9 갱신, severe `lb_quota_exhausted` 제거. spec @ §0/§0.5/§2/§4/§10/§N+1/§N+3.
- v1.2 (2026-05-12): main lever 재정렬 — corrector 강화 가 main, ranking loss 가 secondary. 사용자 challenge 반영 (oracle 1.5cm = 84.78% 발견). [SUPERSEDED by v1.3]
- **v1.3 (2026-05-12)**: **Phase 순서 재배치 (B 안) — fragile/robust 위험도로 G1/G2 순서 swap**. v1.2 의 oracle 1.5cm = 0.8478 ceiling 발견은 *전략적 anchor* 유지 (target LB 0.74~0.80 보수 조정). cap_saturation 3.58% 실측이 v1.2 의 "cap 확장 main" framing 과 충돌 → fragile lever (corrector — boundary.py + arch + cap + loss 4 곳) 를 G2 로, robust lever (ranking — selector.py partial only) 를 G1 으로 swap. 변경:
  - **§0**: H1 = ranking (G1 main robust, plan-008 §10.2.1 직접 후속), H2 = corrector (G2 main 추가 push, additive ablation). target LB 0.75~0.82 → 0.74~0.80 보수 조정 (fragile lever variance 반영).
  - **§0.5 G-gates**: G1 = ranking (v1.2 의 G2 격상), G2 = corrector (v1.2 의 G1 격하). 조건부 threshold (0.78 / 0.75) 동일.
  - **§0.5 severe**: `ranking_loss_failure` 신설 (G1 severe, main 1순위 격상), `corrector_strengthen_marginal` 신설 (G2 warn-only if G1 OOF ≥ 0.70, severe if < 0.70). v1.2 의 `corrector_strengthen_failure` → `corrector_strengthen_marginal` 로 격하. v1.2 의 `ranking_loss_marginal` → `ranking_loss_failure` 로 격상. `oracle_decomp_artifact_missing` 에 cap_saturation_extended 포함.
  - **§0.5 commit chain**: c3~c5 = ranking (격상), c6~c8 = corrector (격하, **4 additive sub-exp**). 16 commits → 17 commits.
  - **§0.5 paths blacklist**: boundary.py CORRECTOR_CAP 직접 정수 교체 추가 (caveat #7 통일).
  - **§1.4 가설**: H1 = ranking (G1 main), H2 = corrector (G2 main 추가 push).
  - **§2.1 In-scope**: ranking 격상 / corrector additive ablation 명시 / cap 인자화 (default 0.006 보존 + wrapper override).
  - **§2.2 Out-of-scope**: boundary.py CORRECTOR_CAP 직접 정수 교체 추가 (caveat #7 통일).
  - **§3.3 평가 점수**: G1 main metric (top1_acc, gap_ranking), G2 main metric (per-band hit, corrector_oracle_gain), G0 신규 (cap_saturation_extended).
  - **§4 G0**: cap_saturation_extended.py 신설 (extended 25 cands binding rate 재측정) — corrector main framing evidence anchor.
  - **§5 (격상)**: Ranking loss (v1.2 §6 내용 이동, G1 main robust). selector.py partial only, boundary.py touch X.
  - **§6 (격하 + additive)**: Corrector 강화 (v1.2 §5 내용 이동 + cumulative 3 sub-exp → **additive 4 sub-exp** a/b/c/d). attribution 측정 + best 채택.
  - **§N+1**: 16 commits / 25min → 17 commits / 28min (+1 commit, +3min, additive sub-exp trade-off).
  - **§N+3 caveats**: #7 통일 (cap 인자화 binding), #13 G0 격상 (cap_saturation_extended), #14 보수 조정 (40% → 30%), #16 신규 (G1/G2 ordering risk 분리), #18 신규 (G2 attribution informativeness), #19 신규 (v1.2 caveat #16 reasoning 오류 삭제 명시).

---

## §N+5. 참조

- **`analysis/plan-005/corrector_decomp.{md,json}` — ★ G2 추가 push 의 직접 근거 (best-raw-cand error 8-bin 분포 + corrector 회복률 + cap_saturation overall_rate 0.0358 실측)**
- plan-005 STAGE 6 (Variant A LB 0.6796) — 본 plan baseline 결정 anchor
- plan-007 framework 대체 시도 실패 — 본 plan G5 의 risk anchor
- plan-008 `next_plan_candidates.md` §10.2.1 (ranking 6 카테고리 ROI 표) — **본 plan §5 G1 main 의 직접 spec source**
- plan-008 §7 (corrector band-specific) — 본 plan §6 G2 의 carry-over (additive sub-exp 의 b/d 에 spec 그대로 사용)
- `WORKFLOW.md` §0.5, §11, §12 convention
