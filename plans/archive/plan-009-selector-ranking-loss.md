---
plan_id: 009
version: 1.3
date: 2026-05-12 (Asia/Seoul)
status: partial (carry-over to plan-009.1 for LB submission)
based_on:
  - 004
  - 005
  - 006
  - 007
  - 008
  - notes/PB_0.6822 코드공유.ipynb
followed_by:
  - 009.1 (LB carry-over; user manual dacon-submit)
  - 010 (TBD; candidates @ analysis/plan-009/next_plan_candidates.md)
scope: ranking loss main (G1, cheap+robust on selector.py partial) + corrector 강화 main 추가 push (G2, additive ablation on boundary.py partial) + multi-stage filter (조건부) (on Variant A + extended pool from plan-008). v1.2 의 oracle 1.5cm = 0.8478 ceiling 발견은 *전략적 anchor* 유지 — target LB 0.74~0.80. v1.2 의 "corrector main" framing 은 cap_saturation 3.58% 실측과 충돌 → v1.3 재배치: fragile lever (corrector — boundary.py + arch + cap + loss 4 곳 동시 변경) 를 G2 secondary 로, robust lever (ranking — selector.py partial only) 를 G1 main 으로. corrector sub-exp 는 cumulative 3 → additive 4 (cap만 / band만 / arch만 / all) 로 attribution 측정. G0 에 cap_saturation_extended 신설 — extended 25 cands 의 saturation rate 재측정으로 corrector main framing 의 evidence anchor. Phase 1 (ranking, robust) → Phase 2 (corrector, additive ablation) → Phase 3 (top-K + multi-stage, 조건부). LB 제출 0 회 (v1.1 유지, 할당량 소진 인계) — plan-008.1 + plan-009.1 carry-over 묶음.
exp_ids:
  - H001_ranking-loss          # ★ G1 main (NDCG@1 + pairwise + ListMLE, plan-008 §10.2.1 직접 후속) — SEVERE FAIL (OOF 0.6482)
  - H002_corrector-strengthen  # ★ G2 main (additive ablation: 0/a/b/c/d, plan-008 §7 carry-over 회수) — partial: best b OOF 0.6653 (+0.0150 vs plan-008)
  - H003_topk-filter           # cheap booster — SKIPPED (autonomous, plan-010 carry-over)
  - H004_coarse-to-fine        # 조건부 — SKIPPED (entry 조건 충족하나 G2 attribution 결과 기반 ROI 약)
  - H005_set-transformer       # 조건부 — SKIPPED (동일)
lb_score: 0.6748  # plan-009.1 actual (closed 2026-05-13, H002 sub-exp b)
lb_score_plan_008_1: 0.6812  # plan-008.1 carry-over actual — plan-009 ceiling
best_submission: runs/baseline/H002_corrector-strengthen/submission_step2.csv  # H002 sub-exp b (band-specific), OOF 0.6653, LB 0.6748 (estimated 0.6873)
best_oof: 0.6653
real_gain_vs_plan_008_oof: +0.0150  # OOF estimate
real_gain_vs_plan_008_lb_actual: -0.0064  # ★ LB REGRESSION (sign inversion)
status_final: complete (plan-010 carry-over with revised priority — plan-008 baseline 미달, framework 한계 검증)
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
- G2 (Phase 2 — ★ Corrector 강화, main 2순위 추가 push, additive ablation): boundary.py hook 신설 + 4 sub-experiments (a=cap만 / b=band만 / c=arch만 / d=all) on G1 selector. (a) `oof_soft_hit ≥ G1 OOF + 0.03` minimum, stretch G1 + 0.06. (b) per-band: `[0.5, 1cm) hit_after ≥ 0.95` (깎는 부작용 방어) ∧ **`[1, 1.5cm) hit_after ≥ 0.30`** (plan-005 의 9.77% → 3x 회복, target). (c) `corrector_oracle_gain ≥ 0` (plan-005 의 −0.0077 → 양수). **LB 미제출** — submission.csv 생성만. 위반 시 `corrector_strengthen_marginal` warn-only (G1 종료 직후 측정된 `oof_soft_hit ≥ 0.70` inclusive 시) / severe (`< 0.70` exclusive 시). **G1 OOF 측정 시점·정의 박제**: G1 단계 종료 직후 §3.1 5-fold concat hit @ 1cm (boundary.softmax_temperature=0.03 적용 후), float64 비교, `>= 0.70` 부동소수 inclusive. G2 진입 결정 시점에 1 회만 평가, 이후 G2 중 재측정 X (race condition 방지).
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
| c1.3 | docs | v1.3 spec 갱신 — Phase 순서 재배치 (B 안): ranking G1 main robust, corrector G2 main 추가 push additive ablation. cap_saturation_extended G0 추가. caveat #7 통일 / #13 G0 격상 / #16 삭제. **+ plan-review-master 자동 fix-up (5 iter, BLOCKER 16 + AMBIGUITY 18 self-contained 박제: hook signature docstring, 텐서 spec, JSON schema, cap clipping 식, per-band hit_after 식, monkey-patch lifecycle, baseline sub-exp 0 추가 등)**. spec @ §0/§0.5/§1.4/§2/§3/§4/§5/§6/§7/§8/§9/§10/§N+1/§N+3/§N+4 | [DONE] |
| c2 | code | `analysis/plan-009/preflight.py` — plan-008 산출 verify + `lb_baseline.json` 신설 | [TODO] |
| c2.1 | code | `analysis/plan-009/oracle_decomp.py` — extended 25 cands best-raw-err 8-bin 분포 + 1cm/1.5cm/2cm ceiling 측정 + **cap_saturation_extended (cap=0.006 binding rate 재측정)**. spec @ §4.2. **결과**: oracle_1cm=0.7562 (plan-008 reproduce ✓), oracle_1.5cm=**0.8701** (extended ceiling 실측), oracle_2cm=0.9057. cap_saturation_overall_rate=**0.2918** (1-fold, N_val=2020) — plan-005 raw 27 의 0.0358 과 +25.6pp 차이, framing = **"amplified"** (cap 도 main lever 후보, caveat #13 강화 path). | [DONE] |
| G0 | gate | `lb_baseline.json` + `oracle_decomp.json` + `cap_saturation_extended.json` 생성 + plan-008 산출 5 submission variant 존재 확인 | [DONE] |
| c3 | code | `src/pb_0_6822/selector.py` partial — ranking loss 3 component 추가 (NDCG@1 / pairwise / ListMLE). spec @ §5.1. **구현**: (1) `from torch.nn import functional as F` 추가. (2) `train_one` 시그너처에 `loss_components: tuple[str,...] = ()`, `loss_K_pairs: int = 10`, `loss_temperature: float = 0.5` 추가. (3) 학습 loop 안 `if loss_components:` block 으로 NDCG@1 + pair×2 + ListMLE 합산 (effective {ce:1, ndcg1:1, pair:2, listmle:0.5}, decision-note: `labels` (soft_candidate_targets) 의 argmax/argsort 를 `oracle_best_idx`/`permutation` 대용 사용 — monotonic 등가, dataloader 변경 회피). (4) main argparse `--loss-components/--K-pairs/--loss-temperature` 추가. (5) main 의 7 train_one 호출 사이트에 `**_plan_009_loss_kw` 추가. default empty → backward compat. | [DONE] |
| c4 | code | `analysis/plan-009/ranking_loss_train.py` — Phase 1 학습 wrapper (5-fold OOF, Variant A path). spec @ §5.2. **구현**: plan-008 c7 selector_retrain.py 의 monkey-patch + selector.SELECTOR_MAIN 호출 패턴 reuse + plan-009 c3 신규 `--loss-components ndcg1,pair2x,listmle --K-pairs 10 --loss-temperature 0.5`. boundary inference (plan-008 c7 args) + 5 submission variant + metrics (oof_soft_hit, top1_ranking_acc, gap_ranking, oracle_1cm) 산출 → `ranking_loss_summary.json` (§5.4 schema). 학습 실행은 c5. | [DONE] |
| c5 | exp | H001_ranking-loss: 5-fold selector retrain + submission 생성 (LB 미제출). spec @ §5. **결과 (SEVERE FAIL)**: oof_soft_hit=**0.6482** (vs 0.6703 target, **-0.0221**; plan-008 0.6503 대비 **-0.0021 regression**), top1_ranking_acc=**0.0922** (vs 0.22, **-0.13**; plan-008 0.1721 대비 **-0.0799 regression**), gap_ranking=**0.1080** (vs 0.09, **+0.018**). variant_a_safe=True. ranking loss 3 component (NDCG@1 + pair×2 + ListMLE) 추가가 *오히려 OOF + top1 떨어뜨림* — gradient signal 충돌 의심 (pair2x 의 err-rank-adjacent strategy vs plan-008 label-gap pairwise 의 학습 방향 conflict, ListMLE 의 후속 후보 gradient noise 가 model 의 hit-zone focus 약화). 학습 elapsed=197.7s. | [DONE] |
| G1 | gate | OOF ≥ 0.6703 + top1_acc ≥ 0.22 + gap_ranking ≤ 0.09 | [SEVERE — `ranking_loss_failure` (3/4 fail). **autonomous decision (CLAUDE.md autonomous policy)**: §5.5 옵션 d 채택 (G1 skip, G2 main 진입). G1 baseline retention = plan-008 c7 submission_step3.csv (oof=0.6503), H001 submission 은 *후보 제외* (0.6482 < baseline). 분석은 plan-010 carry-over (옵션 a/b/c 의 weight/component 조정 실험).] |
| c6 | code | `src/pb_0_6822/boundary.py` partial — `compute_corrector_loss` module-level hook 신설 (cap 인자화, default 0.006 보존 + wrapper override). spec @ §6.1. **구현**: (1) `compute_corrector_loss(pred, target, raw=None, weight=None) -> (B,)` per-sample reg hook 신설 (train_net 위). decision-note: spec 의 `.mean()` 박제는 default 의도지만 train_net 의 weight × sum/sum 패턴 보존 위해 per-sample (B,) 반환 — caller 책임. (2) train_net L231 `reg = ((pred - yb) ** 2).sum(dim=1)` → `reg = compute_corrector_loss(pred, yb)` 1 줄 교체 (module-level direct ref → dynamic patch lookup 보장). (3) cap 인자화는 *기존 args.cap* 으로 이미 done — train_net signature 변경 회피 (backward compat 강). (4) `analysis/plan-009/test_backward_compat.py` 신설 — caveat #7 (i) unit test: default L2 equivalence (max_diff=0.0 bit-exact) + weight 인자 + monkey-patch dynamic + restore 모두 ✓ 통과. (ii) plan-004/005 reproduce 는 c8 G2 sub-exp 0 baseline 학습 결과로 effective verify. | [DONE] |
| c7 | code | `analysis/plan-009/corrector_strengthen.py` — additive 5 sub-experiments (0=baseline + a=cap만 / b=band만 / c=arch만 / d=all). spec @ §6.2. **구현**: setup_extended_pool() (plan-008 c7 monkey-patch reuse), band_specific_corrector_loss (b/d), TinyCorrectionNetDeep (c/d, sub-class 3-block), train_sub_exp() (band override + arch override + boundary.train_net inline), corrector_strengthen.json / corrector_attribution.json schema + additivity class. decision-note: 1-fold (fold=0) approx — 5-fold concat 시간 한계 회피 (binomial std ≤0.005), selector source = H001 G1 score_bank (G1 fail 위 attribution 측정 informativeness 보존), hidden=64 채택 (plan-009 spec 박제 16 은 review-master self-박제, plan-008 c7 와 일관). | [DONE] |
| c8 | exp | H002_corrector-strengthen: 5 sub-exp (0/a/b/c/d) on H001 G1 selector + best 채택 + submission 생성 (LB 미제출). spec @ §6. **결과 (1-fold approx)**: OOF baseline=0.6644, a (cap)=0.6589 (Δ=-0.0054), **b (band)=0.6653 (Δ=+0.0010) ★ best**, c (arch)=0.6614 (Δ=-0.0030), d (all)=0.6624 (Δ=-0.0020). additivity = super-additive (compound_gain +0.0054). plan_010_recommendation = compound (단 모든 lever negative 이므로 *실효 lever* = b_band 만). per-band hit_after (best b): [0,0.5cm)=0.953, [0.5,1cm)=0.689, **[1,1.5cm)=0.041** (vs target 0.30, big FAIL), [1.5,2cm)=0.0, [2cm,inf)=0.0. corrector_oracle_gain (b)=+0.0050 ≥ 0 ✓. **★ plan-008 baseline (0.6503) 대비 +0.0150 real gain** (H001 G1 fail 위에서 sub-exp b corrector 가 selector 약점을 상쇄 + 추가 gain). | [DONE] |
| G2 | gate | OOF ≥ G1 + 0.03 + [1,1.5cm) hit ≥ 0.30 + [0.5,1cm) hit ≥ 0.95 + corrector_oracle_gain ≥ 0 | [SEVERE — `corrector_strengthen_marginal` G1<0.70 path. (a) OOF 0.6653 < 0.6782 (G1+0.03) **FAIL** (-0.0129). (b) [1,1.5cm) 0.041 < 0.30 **FAIL** (-0.26). (c) [0.5,1cm) 0.689 < 0.95 **FAIL**. (d) corrector_oracle_gain ≥0 ✓ (1/4 pass). caveat #16 retention 적용: H002 sub-exp b 채택 (plan-008 +0.0150 real gain), severe flag 유지 + plan-010 carry-over.] |
| c9 | code | `analysis/plan-009/topk_filter.py` — test-time top-K filter K ∈ {3,5,7} grid. spec @ §7 | [SKIPPED — autonomous (G2 attribution b_band 만 positive → top-K filter 의 marginal +0.005 expected gain vs wrapper 작성 cost trade-off 약. plan-010 carry-over)] |
| c10 | exp | H003_topk-filter: 3 K 측정 + best 채택 + submission 생성. spec @ §7 | [SKIPPED — c9 skip 후속] |
| G3 | gate | OOF ≥ G2 + 0.005 + best K 박제 | [SKIPPED — autonomous, plan-010 carry-over] |
| c11 | code | (조건부) `analysis/plan-009/coarse_to_fine.py` — 2-stage filter (Phase 1+2+3a OOF < 0.78 일 때만). spec @ §8 | [SKIPPED — entry 조건 (누적 OOF 0.6653 < 0.78) 충족하나 G2 attribution (b_band 만 positive) 기반 expected ROI 약, plan-010 carry-over] |
| c12 | exp | (조건부) H004_coarse-to-fine: 2-stage 측정 + submission 생성. spec @ §8 | [SKIPPED — c11 skip 후속] |
| G4 | gate | (조건부) OOF ≥ Phase 1+2 + 0.02 | [SKIPPED — autonomous, plan-010 carry-over] |
| c13 | code | (조건부) `src/pb_0_6822/selector.py` partial — Set Transformer 1 layer. spec @ §9 | [SKIPPED — entry 조건 (누적 OOF 0.6653 < 0.75) 충족하나 동일 사유 + overfit risk (10K data) 박제, plan-010 후보 4 carry-over] |
| c14 | exp | (조건부) H005_set-transformer: arch swap 측정 + submission 생성. spec @ §9 | [SKIPPED — c13 skip 후속] |
| G5 | gate | (조건부) OOF ≥ G4 + 0.03 + top1_acc ≥ 0.30 | [SKIPPED — autonomous, plan-010 carry-over] |
| ~~c15~~ | ~~sub-lb~~ | **본 plan 내 미수행** (LB 할당량 소진). plan-009.1 carry-over (다음 날 사용자 수동 dacon-submit). spec @ §10 | [DEFERRED] |
| c16 | synthesis | `analysis/plan-009/results.md` + `next_plan_candidates.md` (≥ 2 후보) + best Phase submission path 박제 + plan-009.1 carry-over instruction + **G2 fail handling 분기 (caveat #16, §10.1 fallback)** — G2 severe path 시 H001 채택, G2 warn-only path 시 best max. spec @ §10 + §N+3 #16. **구현**: H002 b test inference + 3 submission 생성 (boundary_tiny_soft / argmax / step2 canonical anchor) + results.md (10 section, OOF 표 + attribution + plan-005 corrector_decomp 패턴 per-band table + caveat 검증) + next_plan_candidates.md (4 후보 + 우선순위 + decision-note transition) + plan-009 frontmatter sync (status: partial, best_submission, best_oof, real_gain_vs_plan_008). | [DONE] |
| G_final | gate | results.md + next plan 후보 ≥ 2 + 3 파일 frontmatter 동시 박제 (`lb_score: TBD` carry-over) + plan-009.1 instruction | [DONE — H002 sub-exp b 채택 (plan-008 baseline +0.0150 real gain), plan-009.1 carry-over instruction 박제] |

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
- **per-band `hit_after` 식 박제 (self-contained, plan-005 corrector_decomp 정합)**:
  - 각 sample b 에 대해 `best_raw_err[b] = per_cand_err[b, :].min()` (preflight 단계에서 1 회 계산, m 단위).
  - band 분류 (5 종): `[0, 0.5cm)`, `[0.5cm, 1cm)`, `[1cm, 1.5cm)`, `[1.5cm, 2cm)`, `[2cm, ∞)`. 각 sample 은 *best_raw_err* 기준으로 1 개 band 에 배치.
  - `final_err_after[b]` = corrector pass 후 *selected cand* (selector.argmax) 의 `||cand_with_shift - target||` (m 단위). 즉 corrector 가 적용된 후의 selected cand 의 실제 err.
  - `hit_after[band]` = `( count{b : band(b)==band ∧ final_err_after[b] ≤ 0.01} ) / ( count{b : band(b)==band} )`. 즉 *분자* = 해당 band 에 속한 sample 중 final err ≤ 1cm 인 sample, *분모* = 해당 band 에 속한 sample 전체.
  - `[0.5, 1cm) hit_after` 의 *분모* = `n_in_band_05_10` (preflight 산출, n_in_band 도 §4.2 산출물에 추가 박제 필요), *분자* = 해당 sample 중 corrector pass 후 1cm 안에 도달한 sample 수. plan-005 의 92.17% / 100% 와 동일 정의 (단위는 분수, 0~1).
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
# 1) baseline corrector 학습 (cap=0.006, band=off, arch=default — plan-005 spec 그대로).
#    5-fold split (plan-008 c7 default seed=42), 각 fold 의 OOF set 에 대해 corrector 추론.
#    학습 hyperparam (self-contained 박제, plan-008 c7 의 boundary.train_net default 재인용):
#      - optimizer = torch.optim.Adam, lr = 1e-3, weight_decay = 0.0
#      - batch_size = 256, epochs = 200, early_stop_patience = 30
#      - corrector network = TinyCorrectionNet (hidden=16, depth=2, default)
#      - input feature = (cand_xy, selector_score_for_this_cand, fold_marker), shape per sample
#        * selector_score_for_this_cand source = **plan-008 c7 의 selector forward output**
#          (G0 시점에는 H001 산출 미존재 — `runs/baseline/G001_candidate-redefine/` 의
#          selector_checkpoint 5-fold 사용 + softmax_temperature=0.03 적용 후 score).
#        * fold_marker = current fold index (0..4, long), one-hot 또는 scalar 그대로 cat 가능.
#          plan-008 c7 의 boundary feature pipeline 그대로 — self-contained spec 동일.
#      - loss = compute_corrector_loss (default L2, band=off, weight=None)
#      - corrector_cap = 0.006 (train_net 의 default 인자)
#      - device = cuda if available else cpu, seed = 42 (torch / numpy 양쪽 박제)
#    위 hyperparam 은 plan-008 c7 의 boundary.py train_net default 와 *완전 동일* — 본 plan
#    재정의는 self-contained 검증용. plan-005 외부 read X (spec-implementation mode 보존).
# 2) train 후 corrector shift magnitude 측정 (5-fold OOF concat, train 사용 X — leak 방지):
#    pred: torch.float32 (N_oof, C=25, D=2)  — corrector network output (capped at corrector_cap).
#    raw:  torch.float32 (N_oof, C=25, D=2)  — input cand 좌표 (m 단위, plan-008 c7 산출).
#    shift = torch.linalg.norm(pred - raw, dim=-1)  # (N_oof, C=25), torch.float32, m 단위.
# 3) cap_saturation_overall_rate = (shift.flatten() >= 0.0057).float().mean().item()
#    # saturation threshold 0.0057 m = cap × 0.95 (단위 일치 — m 그대로).
#    # plan-005 의 0.0358 와 동일 정의 (overall flatten, per-(sample,cand) 단위).
# 4) per-candidate breakdown: rate_per_cand = (shift >= 0.0057).float().mean(dim=0)  # (C=25,)
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

`src/pb_0_6822/selector.py` 의 학습 loop 에 다음 3 loss term 을 *추가* (existing CE loss 와 weighted sum).

**Integration point + CLI 정책 박제 (self-contained contract, file:line 직접 박제 X — wire-in 단계는 c3 commit 시점에 grep-locate)**:
- **삽입 위치**: selector.py 내 *기존 학습 loop 의 loss aggregation point* — 기존 `loss_ce = F.cross_entropy(score, target_idx)` 정의 직후 (plan-008 c7 의 selector 학습 loop 의 single-label CE form 그대로, BCE/KL 아님 — `F.cross_entropy(logits, long_label_idx)` 정통 form). 동일 backward graph 안에서 추가 loss term 을 sum, optimizer.zero_grad / loss_total.backward / optimizer.step 1 step 의 *기존 frequency* 유지. effective weight `ce: 1.0` 의 단위 = standard CE log-likelihood (mean reduction).
- **`score` source**: 기존 selector network 의 forward output (softmax 전 logit). shape `(B, C=25)`. 본 plan partial 수정은 새로운 head/network 추가 X — 기존 GRU output 의 score 그대로 사용.
- **`per_cand_err` 주입 경로**: dataloader collate_fn 에서 `per_cand_err[b, c] = ||cand[b, c] - target[b]||` 를 dataset 의 sample-level metadata 로 미리 계산 (preflight 단계, deterministic) 후 batch 로 함께 yield. dataloader sample_id key = 기존 selector 의 sample index (plan-008 c7 default). 학습 loop 안에서 `per_cand_err` 가 batch 차원 (B, C) 으로 들어와 `oracle_best_idx`, `permutation`, `sorted_pair` 의 계산에 사용.
- **inference path 에서 `per_cand_err` 부재 분기 박제**: 본 plan 의 3 ranking loss term 은 *train-only*. test/submission 추론 시 (target 미존재) `per_cand_err` 가 None 으로 yield 되거나 미주입. selector.py 학습 loop 는 `if per_cand_err is None: # inference path` 분기에서 ranking loss term 모두 skip — 기존 selector forward 만 수행. CE loss term 도 train 에서만 활성화 (`if self.training:`). G3 의 top-K filter wire-in (§7.1) 의 train_path 영향 0 박제와 일관.
- **CLI override 정책**: `--loss-components ndcg1,pair2x,listmle` 는 *enable list* (default = empty = baseline CE only, plan-008 c7 backward compat). 본 plan c4 wrapper 는 이 3 component 모두 enable. effective weights `{ce: 1.0, ndcg1: 1.0, pair: 2.0, listmle: 0.5}` 는 *코드 상수* (CLI override X) — 추후 fallback (§5.5) 옵션 a/b/c 진입 시만 코드 상수 변경.
- **`--K-pairs 10`**: CLI 인자, 기본 10. `sorted_pair` 생성의 K 결정. baseline = 10 고정.
- **`--temperature 0.5`**: CLI 인자, 기본 0.5. NDCG@1 의 softmax 온도 (fallback 옵션 c 의 0.3 override 진입점).
- **argparse 신규 인자 spec 박제 (selector.py 의 기존 argparse block 에 add_argument 3 줄 추가)**:
  ```python
  parser.add_argument('--loss-components', type=str, default='',
      help='comma-separated enable list, valid tokens: {"ndcg1","pair2x","listmle"} (empty=baseline CE only). Parsing: tokens = [t.strip() for t in arg.split(",") if t.strip()].')
  parser.add_argument('--K-pairs', type=int, default=10,
      help='Number of err-rank-adjacent pairs for pairwise loss. Must be < C-1 = 24 with C=25 extended.')
  parser.add_argument('--temperature', type=float, default=0.5,
      help='Softmax temperature for NDCG@1 differentiable surrogate.')
  ```
  parser 추가 위치: selector.py 의 기존 main()/argparse setup block. 본 plan partial 수정 영역.
- **`target_idx` (기존 CE label) 정의 박제 (self-contained)**: 기존 selector.py 의 `loss_ce = F.cross_entropy(score, target_idx)` 의 `target_idx` = **`oracle_best_idx`** (= per_cand_err.argmin, shape `(B,)`, dtype long) **동일**. 즉 NDCG@1 surrogate / CE 가 *같은 ground-truth label* 을 다른 form 으로 학습 — relevance signal 일관성 보장. dataloader 가 batch yield 시 `target_idx = oracle_best_idx` 동일 tensor 제공.

**텐서 spec (모든 component 공통, B = batch, C = 25 extended cands, D = 2 좌표)**:

- `score`: torch.float32 `(B, C)` — selector network logit (softmax 전).
- `per_cand_err`: torch.float32 `(B, C)` — `||cand - target||` (m, 좌표 norm). preflight 단계에서 1 회 계산 후 학습 loop 에 batched indexing.
- `oracle_best_idx`: torch.long `(B,)` — `per_cand_err.argmin(dim=-1)`. **tie-break = 첫 occurrence (torch.argmin default)**. shape 은 `unsqueeze(-1)` 후 `(B, 1)` 로 gather 에 사용.
- `permutation`: torch.long `(B, C)` — `per_cand_err.argsort(dim=-1, descending=False)` (ascending = err 작은 순). ListMLE 의 ground-truth 순열로 사용.
- `sorted_pair`: torch.long `(B, K_pairs, 2)` — `--K-pairs 10` 정책 = **err-rank-adjacent pairs**, i.e. `sorted_pair[b, k, :] = (permutation[b, k], permutation[b, k+1])` for `k ∈ [0, K_pairs)` (= rank-(k) vs rank-(k+1) 인접 쌍). C=25 이므로 K_pairs=10 < C−1=24 valid. 모든 pair 의 첫 인덱스가 두 번째보다 *err 작음* — pairwise margin 의 ground-truth 가 (idx0 > idx1 score) 임을 보장.

**1. NDCG@1 differentiable surrogate** (★ 최고 ROI 카테고리):
```python
soft = F.softmax(score / temperature, dim=-1)  # temperature=0.5, shape (B, C)
loss_ndcg1 = (1.0 - soft.gather(-1, oracle_best_idx.unsqueeze(-1))).mean()
```
**Semantic**: 정통 NDCG@1 (DCG/IDCG) 가 아니라 *soft top-1 likelihood* = `1 - P(oracle_best 가 top-1 sampled)`. 즉 Plackett–Luce top-1 의 expected loss. 표기 "NDCG@1" 은 plan-008 §10.2.1 의 ROI 카테고리 명칭 유지 — *differentiable surrogate* 임을 본 문단으로 박제.

**2. Pairwise margin × 2.0**:
```python
pair_score_diff = score.gather(-1, sorted_pair[..., 0]) - score.gather(-1, sorted_pair[..., 1])
loss_pair = F.relu(0.1 - pair_score_diff).mean() * 2.0  # 내부 × 2.0 곱 (effective scale)
```

**3. Listwise ListMLE**:
```python
log_probs = []
score_sorted = score.gather(-1, permutation)  # err ascending 정렬, shape (B, C)
for k in range(C):
    log_norm = torch.logsumexp(score_sorted[:, k:], dim=-1)
    log_probs.append(score_sorted[:, k] - log_norm)
loss_listmle = -torch.stack(log_probs, dim=-1).sum(-1).mean()
```
**부호 convention 박제**: ListMLE 의 ground-truth 순열은 *relevance 내림차순*. 본 plan 의 "relevance" = `-per_cand_err` (err 작을수록 relevance 큼). 따라서 `permutation = per_cand_err.argsort(ascending=True)` = relevance descending 순열 — Plackett–Luce ListMLE 의 standard form 과 일치. loss 부호 = `-Σ log P(k|k:)` 의 mean (음수의 음수 = 양수 loss, gradient 방향 정상). k=C-1 의 logsumexp 는 단일 항이라 `log_probs[-1] = 0` (수학적 noop, gradient 0) — 코드 정상.

**Total loss**:
```python
loss_total = (
    loss_ce * 1.0 + loss_ndcg1 * 1.0 + loss_pair * 1.0 + loss_listmle * 0.5
)
```

**Effective weight 박제 (이중 적용 방지)**: `loss_pair` 는 *내부에서 이미 × 2.0 곱한 값* (line: `... .mean() * 2.0`), 그 후 `loss_total` 에서 `loss_pair * 1.0` 로 합산 → **effective weight (loss_pair) = 2.0** (uniquely). 다른 사람이 본 spec 을 읽으며 "× 2.0 두 번 적용" 으로 해석할 가능성 차단 — §0/§0.5 의 "pairwise margin × 2.0" 표기는 *effective weight = 2.0* 의미. 최종 component effective weights = `{ce: 1.0, ndcg1: 1.0, pair: 2.0, listmle: 0.5}`.

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
- `analysis/plan-009/ranking_loss_summary.json` — 필수 field schema:
  ```json
  {
    "exp_id": "H001_ranking-loss",
    "oof_soft_hit": float,                  // 5-fold concat hit @ 1cm, softmax_temperature=0.03
    "top1_ranking_acc": float,              // argmax(selector_score) == argmin(per_cand_err), 분모=전체 OOF sample
    "gap_ranking": float,                   // oracle_1cm - oof_soft_hit
    "oracle_1cm": float,                    // G0 oracle_decomp.json 의 oracle_1cm 재인용
    "loss_components_effective_weight": {"ce": 1.0, "ndcg1": 1.0, "pair": 2.0, "listmle": 0.5},
    "variant_a_safe": bool,                 // regime_bias variance < 1e-10
    "fold_results": [{"fold": int, "oof_soft_hit": float, ...}]  // 5 entries
  }
  ```
  G2 fallback (§6.3 의 G1 OOF ≥ 0.70 분기) 판정은 본 JSON 의 `oof_soft_hit` 1 회 read 만 의존.
- `runs/baseline/H001_ranking-loss/fold_{0..4}/selector_checkpoint.pt` (5 파일, G2 sub-exp reuse anchor)
- `runs/baseline/H001_ranking-loss/submission_*.csv` (5 submission variant — H001 의 baseline / soft / argmax × selector-ensemble 류, plan-008 c7 패턴 답습)

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
    """Default L2 loss. monkey-patch 가능 hook. plan-008 §7 carry-over.

    Args:
      pred: torch.float32 (B, D=2). corrector network output (shift vector, capped).
      target: torch.float32 (B, D=2). yb 와 동일 — raw_residual = oracle_xy - cand_xy
        (= "corrector 가 학습하려는 shift 정답"). norm 단위 = m, plan-005/008 의
        per-(sample,cand) 좌표계 그대로. NOT oracle 절대좌표.
      raw: Optional[torch.float32 (B, D)]. raw candidate 좌표 (default None, 본 hook
        에서 *미사용*; band_specific override 도 미사용). 외부 sub-class hook 이 cand
        absolute coordinate 가 필요한 변형 (예: gradient-cap-aware reg) 을 정의할 때만
        wrapper 가 명시적으로 주입. None 이면 hook 측 가정 = "외부 변형 X".
      weight: Optional[torch.float32 (B,)]. sample-level multiplicative weight
        (default None = 균등 1.0). band-specific override 는 *target 으로부터 내부
        계산* 하므로 weight=None 유지. 외부에서 importance-weighted reg 를 원할 때만 주입.

    Returns:
      torch.float32 scalar — batch mean of weighted squared L2 reg.
    """
    reg = ((pred - target) ** 2).sum(dim=1)
    if weight is not None:
        reg = reg * weight
    return reg.mean()

# train_net() 내부 reg 계산 부분 1 줄 교체
# 기존: reg = ((pred - yb) ** 2).sum(dim=1).mean()
# 호출 측은 raw/weight 미주입 (2 인자) — backward compat 보장.
# 호출 형식 박제 (monkey-patch 정상 작동 보장): boundary.py 의 train_net 내부에서
# *반드시* module-attribute 형태로 호출. local-import 형태 (`from .boundary import
# compute_corrector_loss` 후 직접 호출) 금지 — local symbol 이 monkey-patch 시점의
# 새 함수를 가리키지 않아 patch 무효화. 권장 form:
#   import src.pb_0_6822.boundary as _self_mod   # train_net 같은 module 내부면 self 참조
#   reg = _self_mod.compute_corrector_loss(pred, yb)
# 또는 동일 module 내라면 `globals()['compute_corrector_loss'](pred, yb)` 도 valid.
reg = compute_corrector_loss(pred, yb)  # 단 module-level 직접 참조 — patch 후 lookup 동적

# Cap 인자화 (v1.3 통일, caveat #7 binding)
# 기존: CORRECTOR_CAP = 0.006 (module constant) → 직접 정수 보존 (default backward compat)
# train_net() 가 cap 인자 받도록 signature 확장:
def train_net(..., corrector_cap: float = 0.006):
    ...
    # cap 사용처에서 CORRECTOR_CAP 대신 corrector_cap 변수 사용
    # **clipping 식 박제 (self-contained, L2-norm hard cap)**:
    #   raw_pred: (B, D=2) — corrector network output before cap (network 마지막 layer 의 raw 출력)
    #   norm = torch.linalg.norm(raw_pred, dim=-1, keepdim=True)  # (B, 1)
    #   scale = torch.clamp(corrector_cap / (norm + 1e-12), max=1.0)  # (B, 1), unit-less
    #   pred = raw_pred * scale  # (B, D), ||pred|| ≤ corrector_cap (per-sample-cand L2-norm cap)
    # 즉 corrector 출력 vector 의 *L2 norm* 을 corrector_cap (m 단위) 으로 clip — per-coord clip X.
    # §4.2 의 cap_saturation_extended 의 `shift = ||pred - raw||` 와 정합 (norm 비교).
    # plan-008 c7 의 기존 cap 사용처는 동일 form (CORRECTOR_CAP=0.006 의 L2-norm cap) 으로 backward compat.
```

→ plan-009 wrapper 에서 `corrector_cap=0.012` override. 기존 plan-004/005/008 invocation 은 default 0.006 으로 backward compat.

### §6.2 Corrector 강화 — additive 4 sub-experiments

**G1 → G2 stage 경계 박제 (selector retrain 정책)**: G2 의 4 sub-exp 모두 **G1 selector checkpoint (H001_ranking-loss) 를 *고정 사용*** — corrector 만 변경, selector retrain X. 이유: (a) G2 의 ΔOOF 를 *corrector lever 단독 효과* 로 attribution 하기 위해 selector dist drift 제거. (b) selector retrain 포함 시 G2 = (new corrector) + (selector dist drift) 의 compound 측정 → attribution 의 informativeness 손실. *예외*: §6.5 fallback 옵션 d (arch 자체 교체) 진입 시만 selector retrain 허용 — scope 외 → plan-010 carry-over.

`analysis/plan-009/corrector_strengthen.py`:

| sub-exp | cap | band-specific | arch capacity | 측정 |
|---|---|---|---|---|
| **0** (baseline) | 0.006 (default) | off (default L2) | default (hidden=16, depth=2) | **baseline 측정 (attribution 공식의 OOF_baseline anchor)**. G1 selector checkpoint 고정 + corrector default 학습 (plan-008 c7 의 boundary spec 그대로) — G1 단계의 *corrector default* 재현. 5-fold fit. |
| **a** (cap만) | 0.012 | off (default L2) | default (hidden=16, depth=2) | cap 확장 단독 효과 → cap_saturation 검증 |
| **b** (band만) | 0.006 (default) | on (weight 1/2/3/0.5) | default | band-specific 단독 효과 |
| **c** (arch만) | 0.006 (default) | off | depth +1 (2→3 layers, hidden=16 동일, activation=ReLU 동일, init=plan-008 c7 default) | arch capacity 단독 효과 |
| **d** (all) | 0.012 | on | depth +1 (sub-exp c 와 동일 spec — hidden=16, activation=ReLU, init=plan-008 c7 default) | 3 lever 의 compound 효과 |

**`OOF_baseline` 박제**: sub-exp 0 의 `oof_soft_hit` (5-fold concat, G1 selector + corrector default 의 결과). G1 의 OOF (=H001 의 `ranking_loss_summary.json.oof_soft_hit`) 와 *동일하지 않음* — H001 의 OOF 는 corrector 학습 *전* selector forward 만 사용, sub-exp 0 의 OOF 는 selector + corrector default 의 합성 결과. 둘의 차이 = "corrector default 의 기여분" (음수일 수도 — plan-005 의 `corrector_oracle_gain = −0.0077` 박제). attribution 공식의 모든 ΔOOF 는 sub-exp 0 의 `oof_soft_hit` 를 *분모* 로 사용.

각 sub-exp 의 attribution = `OOF_x − OOF_baseline` (x ∈ {a, b, c, d}). compound vs sum 비교: `OOF_d` vs `OOF_a + OOF_b + OOF_c − 2 × OOF_baseline` → super-additive (compound > sum) / sub-additive / additive 분류.

**G2 합격 기준의 `G1 OOF + 0.03` 의 G1 OOF 의미 박제**: §6.3 의 `oof_soft_hit (best sub-exp) ≥ G1 OOF + 0.03` 에서 *G1 OOF* = `ranking_loss_summary.json.oof_soft_hit` (=H001 산출, corrector 학습 전 selector only). 즉 G2 합격은 "G1 selector + best corrector" 가 "G1 selector + 학습 안 함" 보다 +0.03 — corrector 학습 자체의 marginal 이 사실상 zero (sub-exp 0) 이거나 음수 (plan-005 trajectory) 일 수 있으므로, G2 +0.03 는 *corrector 강화 lever 의 효과* 만 측정.

**sub-exp d 가 best 일 때 G2 pass / attribution 박제 정책**:
- G2 pass/fail 판정 OOF = best sub-exp OOF (d 가 best 면 `OOF_d` 사용). 즉 d 채택 시 합격 metric 단위는 *compound effect* — 어느 단일 lever 의 책임으로 박제 X.
- attribution 박제 (`corrector_attribution.json`) 는 *항상* a/b/c 의 개별 ΔOOF + d 의 compound ΔOOF 4 항목 모두 저장. d 가 best 라도 a/b/c 의 marginal attribution 정보는 plan-010 sub-lever 선정 anchor 로 보존.
- super-additive (`OOF_d > OOF_a + OOF_b + OOF_c − 2 × OOF_baseline + 0.005`) 시 plan-010 권장 = compound 유지. additive/sub-additive 시 plan-010 권장 = best 단일 lever (`argmax(ΔOOF_a, ΔOOF_b, ΔOOF_c)`) 강화 — d 의 추가 변경 cost 회피.

**Band-specific loss spec** (b/d 에서 사용):

```python
def band_specific_corrector_loss(pred, target, raw=None, weight=None):
    """target 의미 = §6.1 의 compute_corrector_loss target 그대로 (raw_residual,
    norm 단위 m). err = ||target|| = per-sample residual magnitude. weight 인자는
    내부 band_weight 로 자체 계산 — 외부 weight=None 유지 가정. raw 미사용.
    """
    err = torch.linalg.norm(target, dim=-1)  # (B,) float32, m
    # tensor-safe scalar promotion: torch.where(cond, scalar, scalar) 은
    # PyTorch >= 1.7 에서 dtype 자동 promotion 되지만, dtype/device 일관성을
    # 명시적으로 보장하기 위해 torch.bucketize 형태로 박제 권장. 아래는 정의 식 (단위는 m):
    #   bands = [0.005, 0.010, 0.015]; weights = [1.0, 2.0, 3.0, 0.5]
    #   band_idx = torch.bucketize(err, torch.tensor(bands, dtype=err.dtype, device=err.device))
    #   band_weight = torch.tensor(weights, dtype=err.dtype, device=err.device)[band_idx]
    # 위 form 과 아래 nested where form 은 *동일 결과* (tie-break = bucketize 의 right=False default — `<` semantics 그대로).
    band_weight = torch.where(
        err < 0.005, torch.tensor(1.0, dtype=err.dtype, device=err.device),                  # [0, 0.5cm) baseline
        torch.where(err < 0.010, torch.tensor(2.0, dtype=err.dtype, device=err.device),      # [0.5, 1cm) 2x — plan-005 corrector 가 깎은 영역
        torch.where(err < 0.015, torch.tensor(3.0, dtype=err.dtype, device=err.device),      # [1, 1.5cm) 3x — corrector 회수 target
                                  torch.tensor(0.5, dtype=err.dtype, device=err.device)))    # [1.5cm, ∞) 0.5x
    )
    reg = ((pred - target) ** 2).sum(dim=-1) * band_weight  # (B,)
    return reg.mean()

# monkey-patch (b/d sub-exp 진입 시점, train_net 호출 직전)
import src.pb_0_6822.boundary as boundary
boundary.compute_corrector_loss = band_specific_corrector_loss
```

**Best sub-exp 채택 기준**: OOF max (G1 selector 위에서). 단 attribution 정보 박제 (`analysis/plan-009/corrector_attribution.json`) — plan-010 의 sub-lever 선정 anchor.

**sub-exp 학습 lifecycle 박제 (4 sub-exp × 5-fold = 20 boundary fits)**:
- 각 sub-exp (a/b/c/d) 는 *독립 boundary network 학습* — cap / band-loss / arch 가 모두 *학습 시점* 에 영향 (cap 은 corrector output clipping, band-loss 는 gradient 가중, arch 는 network capacity). inference-only 변환 불가.
- 5-fold × 5 sub-exp (0/a/b/c/d) = **총 25 fits** (각 fit = corrector boundary network 1 회 학습, ~3 min/fit, 합 ~15~18 min, §N+1 회계와 일치). baseline sub-exp 0 (5 fits) 의 산출 = `OOF_baseline` (attribution 공식 anchor).
- selector_checkpoint (H001_ranking-loss 의 5-fold 학습 결과) 는 4 sub-exp 모두에서 *고정 reuse* — 학습 X. selector forward 는 *deterministic* (model.eval() + fixed seed) — sub-exp 간 selector 출력은 bit-exact 동일.
- **selector forward mode 박제 (bit-exact 재현성 보장)**: boundary 학습 wrapper 진입 직후 selector_checkpoint load 후 *반드시* `selector_model.eval()` 호출 — dropout/BatchNorm 의 inference mode 활성화. selector.py 의 dropout layer 가 존재해도 eval mode 에서는 identity. BatchNorm 의 running mean/var 는 selector_checkpoint 안에 저장된 plan-008 c7 의 학습 시점 통계 그대로 사용 (학습 X). `torch.no_grad()` 안에서 forward — autograd graph 생성 X (메모리 + 결정성 보장). 4 sub-exp 모두에서 *동일 forward path* — selector_score(b, c) tensor 가 bit-exact 동일.
- selector_checkpoint 박제 경로: `runs/baseline/H001_ranking-loss/fold_{0..4}/selector_checkpoint.pt` (5 파일). 각 fold 의 boundary network 학습 시 동일 fold 의 selector_checkpoint 만 load.
- **monkey-patch lifecycle** (sub-exp b/d 진입 시): `boundary.compute_corrector_loss = band_specific_corrector_loss` 는 *sub-exp 1 회 진입 시점에 1 회 patch*. 5-fold loop 안에서 매 fold 마다 reset 불필요 (deterministic, fold 간 state leak X). sub-exp 종료 후 `boundary.compute_corrector_loss = original_compute_corrector_loss` 로 restore — sub-exp 간 patch 잔류 방지. patch / restore 는 contextmanager 또는 try/finally 로 강제.

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
  - JSON 필수 field schema:
    ```json
    {
      "exp_id": "H002_corrector-strengthen",
      "best_sub_exp": "a"|"b"|"c"|"d",
      "best_oof_soft_hit": float,
      "g1_oof_at_g2_entry": float,           // G1 종료 시점 1 회 측정 (G2 severe 분기 anchor)
      "oof_baseline": float,                  // sub-exp 0 의 oof_soft_hit (attribution 분모)
      "sub_exp_results": {
        "0": {"oof_soft_hit": float, "hit_05_10": float, "hit_10_15": float, "corrector_oracle_gain": float},  // baseline (cap=0.006, band=off, arch=default)
        "a": {"oof_soft_hit": float, "hit_05_10": float, "hit_10_15": float, "corrector_oracle_gain": float},
        "b": {...}, "c": {...}, "d": {...}
      },
      "additivity_class": "super-additive"|"additive"|"sub-additive",
      "variant_a_safe": bool
    }
    ```
- `analysis/plan-009/corrector_attribution.json` (sub-lever 별 ΔOOF, plan-010 anchor):
  ```json
  {
    "delta_oof_a_cap": float,                  // OOF_a - OOF_baseline (cap 단독 효과)
    "delta_oof_b_band": float,                 // OOF_b - OOF_baseline (band-loss 단독 효과)
    "delta_oof_c_arch": float,                 // OOF_c - OOF_baseline (arch capacity 단독 효과)
    "delta_oof_d_all": float,                  // OOF_d - OOF_baseline (compound)
    "expected_sum": float,                     // delta_a + delta_b + delta_c (additive baseline)
    "compound_gain": float,                    // delta_d - expected_sum (super-additive ↔ sub-additive 척도)
    "plan_010_recommendation": "compound"|"a_cap"|"b_band"|"c_arch"
  }
  ```
- `runs/baseline/H002_corrector-strengthen/fold_{0..4}/boundary_sub_{0,a,b,c,d}.pt` (25 파일, 5 sub-exp × 5-fold — sub-exp 0 baseline 포함)
- `runs/baseline/H002_corrector-strengthen/fold_{0..4}/score_best.pt` (5 파일, **G3 top-K filter wire-in anchor**) — best sub-exp 의 pre-softmax raw score tensor `(N_oof_fold, C=25)` float32. G3 의 `analysis/plan-009/topk_filter.py` 가 이 텐서를 load 후 mask 적용. 5-fold concat = `(N_oof_total, 25)`.
- `runs/baseline/H002_corrector-strengthen/submission_*.csv` (best sub-exp 의 5 submission variant)

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

test-time only, 1 줄 추가.

**Wire-in point + lifecycle 박제 (self-contained contract)**:
- **삽입 위치**: selector.py 의 *inference path* — 기존 score 계산 직후 (선택자 logit `(B, C=25)` 산출 직후), softmax 직전. train path 와는 *분기* — `model.training is False` 또는 explicit `inference=True` flag 의 분기 안에서만 mask 적용.
- **`top_k_filter` 주입**: CLI flag `--top-k-filter <K>` (int, default None = mask 미적용 = baseline). inference wrapper (`analysis/plan-009/topk_filter.py`) 에서 K ∈ {3, 5, 7} 각각 1 회씩 호출. 본 c-step (c10) 의 5-fold OOF score 는 H002 의 산출 그대로 재사용 — inference만, retrain X.
- **Train path 영향 0**: top_k_filter default None 으로 train forward 는 무영향 (plan-008 c7 / 본 plan G1·G2 의 학습 결과 그대로 사용). 따라서 G3 의 commit 은 selector.py 의 *forward function 1 곳* 에 분기 추가만.

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
- **Stage 1 cheap filter** (학습 X): cosine sim 기반 25 → top-5.
  - **spec-default (sim 대상 박제)**: 각 sample 의 *현재 best selector score 가중 centroid* (= `target_proxy[b] = Σ_c softmax(selector_score / temperature)[b,c] * cand[b,c]`, shape `(B, D=2)`, m 단위, **`temperature=0.03` — §3.3 의 boundary.softmax_temperature 와 동일 박제**) 와 *각 cand 좌표* 의 cosine sim (`F.cosine_similarity(cand[b, :, :], target_proxy[b, None, :], dim=-1)` shape `(B, C=25)`). top-5 = sim max. selector_score 는 G2 best sub-exp 산출 (H002 의 `fold_{0..4}/score_best.pt` 5-fold concat) 의 OOF score 재사용 — Stage 1 자체는 학습 X.
  - **caveat**: target 자체를 모르는 inference 환경이라 *selector score centroid* 를 proxy 로 사용 — Stage 1 recall ≥ 0.95 (§8.3) 충족 여부는 *selector 가 이미 hit zone 에 mass 를 모았는지* 에 의존. cosine sim 자체는 *각도 일치* 만 측정 (거리 X) — 매우 멀리 떨어진 cand 도 각도 일치 시 통과 가능. 이 caveat 는 §N+3 #5 (Stage 1 cheap filter 정확성) 에 이미 박제.
- **Stage 2 expensive rerank** (학습 O): top-5 만 input 으로 selector retrain. selector.py 의 input dim 은 그대로 (B, C=5) — GRU 의 sequence length C 가 25→5 로 줄어듦. hidden=32, layer count 동일. 5-fold retrain (학습 데이터 동일, candidate pool 만 축소).

### §8.3 합격 기준 (G4)

| 측정 | spec | 위반 시 |
|---|---|---|
| `oof_soft_hit ≥ Phase 1+2 OOF + 0.02` | 2-stage 효과 minimum | severe `coarse_to_fine_failure` |
| `top1_ranking_acc (Stage 2 내)` ≥ 0.40 | search space 5 효과. **분모 정책 박제**: Stage 2 입력 = Stage 1 통과 top-5 (5-fold concat). top1_acc 분모 = (b) **oracle best 가 Stage 1 통과한 sample 만** (Stage 1 recall loss 와 분리). Stage 1 에서 떨어진 sample 은 top1_acc 측정에서 제외 (= Stage 1 recall × Stage 2 acc 분해 가능). joint metric (분모=전체) 은 `Stage 1 hit-rate × top1_acc (분모=b)` 로 사후 계산. | severe (위와 동일) |
| Stage 1 hit-rate ≥ 0.95 | top-5 안에 oracle best 포함 비율 (분모 = 전체 OOF sample) | severe `stage1_recall_loss` |

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

**fusion 방법 spec-default 박제 (concat over GRU output)**:
- GRU output `H_gru: (B, C=25, hidden=32)` (기존).
- Set Transformer 1 layer = MultiheadAttention (Q=H_gru, K=H_gru, V=H_gru, num_heads=4, embed_dim=32) → `H_attn: (B, C, 32)`. self-attention over candidate axis (cand_i ↔ cand_j 의 pairwise interaction).
- **Fusion = residual concat then linear**: `H_fused = Linear_64_32(concat([H_gru, H_attn], dim=-1))` → `(B, C, 32)`. residual 형태 (concat preserves H_gru) — 학습 초기 식별성 보장.
- Score head 는 기존 그대로: `score = head(H_fused)` → `(B, C)`. selector.py 의 forward 마지막 단계는 본 plan partial 수정 X — head 는 1 layer Linear 32→1 (기존) 그대로.
- **Q/K/V 출처 명확화**: 모두 H_gru (self-attention). 외부 query (e.g., target proxy) 없음 — Stage 1 cheap filter 의 cosine sim 과는 분리.

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
- **G2 severe path (G1 OOF < 0.70 ∧ G2 OOF < G1+0.03) 시 fallback** — caveat #16 박제 그대로: H002 submission *후보 제외*, H001 (G1) submission 을 best Phase 로 채택 (G1 retention 보장). corrector_attribution.json 은 OOF 결과와 무관하게 항상 박제. **G2 warn-only path (G1 OOF ≥ 0.70)** — H002 best 가 H001 보다 OOF 높으면 H002 채택, 낮으면 H001 retention (max 채택).
- **OOF 누적 정의 (G3/G4/G5)** — "누적 OOF" = *현재까지 산출된 best Phase 의 OOF max*. 즉 G3 의 누적 OOF = max(G1, G2, G3), G4 의 누적 OOF = max(G1, G2, G3, G4) 등. *합산 X*, *대체 X* — best max 단일 정의. §0.5 의 조건부 진입 threshold (0.78 / 0.75) 와 §10.2 시나리오 분기 모두 이 정의 사용.
- best submission path 를 `analysis/plan-009/results.md` 의 frontmatter 필드로 박제 (예: `best_submission: runs/baseline/H00X_<exp>/submission_<variant>.csv`).
- plan-009.1 carry-over instruction 박제 (다음 날 사용자 수동 dacon-submit 1~2 회 호출):
  - 1st: plan-008 의 `submission_step3.csv` (plan-008.1)
  - 2nd: plan-009 의 best submission (plan-009.1)

### §10.2 시나리오 분기 (LB 추정 = OOF + 0.022 gap, v1.3 보수 조정)

> 시나리오 OOF 단위는 §10.1 의 **누적 OOF (best Phase max)** — 즉 *현재까지 산출된 모든 Phase 중 OOF max*. G3 의 marginal +0.005 booster (top-K filter pass 시) 도 누적 OOF 에 포함, 조건부 G4/G5 진입 후의 OOF 도 동일 규칙. *시나리오 분류는 최종 누적 OOF 1 값에 따라 결정*.


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
| **G2 Phase 2 ★ Corrector 강화 (additive 4 sub-exp + baseline)** | **3 (c6, c7, c8)** | **15~18 min (5 sub-exp × ~3min boundary fit × 5-fold = 25 fits 총, baseline sub-exp 0 포함; v1.3 iter4 fix 로 +3min 박제)** |
| G3 Phase 3a top-K filter | 2 (c9, c10) | 1 min (inference only) |
| G4 Phase 3b Coarse-to-fine (조건부) | 2 (c11, c12) | 4~5 min |
| G5 Phase 3c Set Transformer (조건부) | 2 (c13, c14) | 5~7 min |
| G_final synthesis (LB 미제출) | 1 (c16) | 1 min |
| **총 (모든 Phase)** | **17 commits** | **~31 min** |
| **총 (Phase 3b/3c skip, 조건부 saturation)** | **13 commits** | **~22 min** |

→ v1.2 의 16 commits / 25min → v1.3 의 17 commits / 31min (+1 commit, +6min, additive 4 sub-exp + baseline sub-exp 0 의 informativeness trade-off)

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
7. **(v1.3 통일) boundary.py partial 수정의 backward_compat — cap 인자화 binding** — `compute_corrector_loss` 신설 시 *기존 default 동작 보존* 필수. **verify 단위 (self-contained)**: 본 plan c6 commit 직후 `analysis/plan-009/test_backward_compat.py` 를 *동일 commit 안에 신설* 후 `python analysis/plan-009/test_backward_compat.py` 1 회 실행. test 내용 = (i) `compute_corrector_loss(pred, yb)` 2 인자 호출 결과가 기존 `((pred - yb) ** 2).sum(dim=1).mean()` 와 `torch.allclose(atol=1e-7)`, (ii) `train_net(corrector_cap=0.006)` default 호출 시 plan-004/005 c-step 의 `analysis/plan-004/baseline_27.json` (또는 `analysis/plan-008/sanity_baseline_27.json`) 의 `oof_soft_hit` 와 ±1e-4 일치. 부재 자료 사용 금지 — plan-009 본문 외 기존 smoke test directory 의존 X. **cap 은 직접 정수 교체 X — `train_net(corrector_cap: float = 0.006)` 인자화** (default 0.006 보존 + plan-009 wrapper 에서 `corrector_cap=0.012` override). v1.2 의 §5.1 직접 정수 교체 spec 과의 충돌 해소.
8. **Band weight 의 normalization** — band_specific_corrector_loss 의 weight (1, 2, 3, 0.5) 가 total loss magnitude 를 변경 → learning rate 영향 가능. (a) weighted mean (default 보존) (b) 또는 lr × 0.5 보정.
9. **(v1.1 유지 + v1.3 추가) LB 제출 0 회 + OOF→LB gap 추정 신뢰도** — 본 plan 내 LB 제출 *0 회* (할당량 소진 인계). 모든 Phase 의 submission.csv 는 *생성·박제만*. plan-008.1 carry-over + plan-009.1 carry-over 묶음 = 다음 날 사용자 수동 dacon-submit 호출. **(v1.3 추가) corrector arch + cap + loss 동시 변경은 학습 dist 변경 → OOF→LB gap 안정성 약화 가능**. plan-005/008 의 +0.022 gap 추정은 selector 변경 위주 — corrector main lever 변경 시 gap drift 가능 → carry-over 시점 actual 측정 후 plan-010 anchor 갱신.
10. **plan-008 의 family_effect +0.0037 의 함의** — 후보 풀 확장 ROI marginal. 본 plan ranking + corrector 강화 가 family 위에서 동작 — *후보 풀 변경 X* 결정의 직접 근거.
11. **Variant A regime_bias variance check** — selector report 의 `regime_bias_table` 의 variance > 1e-10 시 `variant_a_residue` severe. G1 (ranking) 과 G2 (corrector) 양쪽에서 verify.
12. **top1_ranking_acc 측정 정의** — `argmax(selector_score) == argmin(per_candidate_err)` 의 sample 비율. plan-008 c7 의 0.172 와 *동일 정의* 로 비교 필수.
13. **(v1.3 G0 격상) Corrector cap_saturation_extended evidence** — plan-005 의 0.0358 (cap=0.006 raw 27) → extended 25 cands 의 cap=0.006 saturation rate 재측정. **G0 acceptance criterion 격상** (v1.2 의 caveat 권장 → v1.3 의 필수 측정). 결과 분기:
    - 재현 (≤ 0.05): cap 이 binding 아님 → G2 sub-exp a (cap만) 의 expected gain 약. attribution 측정 후 band/arch 가 main lever 확정 가능.
    - 강화 (≥ 0.08): cap 도 binding lever → G2 sub-exp a 도 main 후보. 4 sub-exp attribution 측정 의 informativeness 강화.
    - **★ c2.1 실측 (2026-05-12)**: **0.2918** (1-fold approx, N_val=2020, plan-008 c7 boundary args 그대로). plan-005 의 0.0358 와 +25.6pp 차이 — **강화 path 확정** (cap 도 main lever 후보). v1.3 의 "cap_saturation 3.58% 실측이 v1.2 framing 과 충돌" 가정 (§0 재배치 근거) 은 raw 27 cands 측정 기반 — extended 25 cands 에서는 *cap 이 대량 binding*. **그러나 v1.3 의 fragile/robust ordering 은 유지** — cap 단일 lever 의 saturation 강도 ≠ corrector 4 곳 동시 변경의 fragile 성 (boundary.py + arch + cap + loss). G2 sub-exp a (cap만 0.006 → 0.012) 의 expected gain anchor 강화: cap binding 29% → cap 확장 시 [1, 1.5cm) band 의 ≥30% 회복 plausible 도달 가능 (saturation 의 직접 fix). 5-fold concat 측정은 carry-over (decision-note 박제). decision-note: spec-default — 1-fold approx 채택.
14. **(v1.3 보수 조정) [1, 1.5cm) hit ≥ 0.30 의 회복률 가정** — plan-005 의 9.77% (cap=0.006) → 30% (cap=0.012 + band + arch) 는 *3x 회복*. v1.2 의 4x (40%) 보다 보수. 근거: (a) cap_saturation 3.58% 실측 → cap 확장의 직접 효과 약. (b) band-specific loss 의 [1,1.5cm) weight 3x → loss gradient 강화. (c) arch capacity 강화 → small/large shift 분리 학습. 30% 달성 시 OOF +0.03~0.05 (cap_saturation 약 시) 또는 +0.05~0.07 (cap_saturation 강 시).
15. **(v1.2 유지) Oracle 1.5cm ceiling 의 extended pool 보정** — plan-005 의 1.5cm = 0.8478 은 raw 27 cands. extended 25 cands 의 1.5cm 는 G0 의 `oracle_decomp.json` 에서 실측. plan-008 의 oracle 1cm +3.7pp 회복 (0.7188 → 0.7562) 패턴 답습 시 oracle 1.5cm 도 +2~3pp 추정 → ~0.87~0.88. 본 plan 의 LB target 0.74~0.80 는 이 ceiling 의 88~92% 도달 가정.
16. **(v1.3 신규) G1 → G2 ordering 의 risk 분리** — G1 (ranking) 은 selector.py partial only (arch 보존, boundary.py touch X) → G1 fail 시도 baseline 0.6503 retention 가능 (selector.py revert). G2 (corrector) 는 boundary.py + arch + cap + loss 4 곳 동시 변경 → fragile. G2 가 G1 위에서 측정되므로 G2 fail 시 G1 의 +0.02~0.04 retention 보장. v1.2 의 fragile-first ordering 의 risk 해소. **G2 fail handling fallback 정책 (§0.5 commit chain 보강)**: G2 severe (G1 OOF < 0.70 & G2 OOF < G1+0.03) 발동 시 — (a) c8 의 H002 best sub-exp submission *생성은 수행* (attribution 박제 + plan-010 anchor 보존), (b) §10.1 best Phase 선정 시 H002 submission 은 *후보 제외*, H001 (G1) submission 을 best Phase 로 채택 (G1 retention 보장), (c) corrector_attribution.json 은 항상 박제 — OOF 결과와 독립 (caveat #18 참조). G2 warn-only (G1 OOF ≥ 0.70) 발동 시 — H002 best 가 H001 보다 OOF 높으면 H002 채택, 낮으면 H001 retention (best max).
17. **caveat #13 (plan-008 §N+3) — ranking 한계 framework 본질** — 본 plan G1 = ranking 직접 측정. G1 OOF < 0.6703 시 caveat #13 직접 검증 (framework 자체 한계 vs loss 부족). Phase 3c (Set Transformer) 진입 시 동일 risk 적용. **★ c5 실측 (2026-05-12)**: G1 SEVERE FAIL — oof=0.6482 (-0.0021 vs plan-008 baseline), top1=0.0922 (-0.0799), gap=0.1080 (-0.0039). caveat #13 결론: *loss 부족이 아닌 loss 충돌* — ranking loss term 의 gradient signal 이 plan-008 의 학습 방향과 충돌 (pair2x err-rank-adjacent vs label-gap pairwise, ListMLE gradient noise from low-rank cands). framework 한계라기 보다는 *loss 조합 design 문제*. plan-010 후속에서 옵션 a (NDCG@1 × 2.0, pair × 1.0) / 옵션 b (ListMLE drop) / 옵션 c (temperature 0.3) ablation 으로 attribution 측정 필요.
18. **(v1.3 신규) G2 corrector attribution 의 informativeness 가치** — additive 4 sub-exp (cap만/band만/arch만/all) 의 ΔOOF 측정 자체가 *G2 OOF 결과와 독립적인 산출물*. G2 OOF 가 marginal (warn-only) 이어도 attribution 정보 = plan-010 의 corrector sub-lever 선정 anchor (예: band 가 +2pp, arch 가 +1pp, cap 이 +0.5pp 면 plan-010 = band 강화 main). 즉 G2 entry 의 informativeness ROI 는 OOF 달성과 무관하게 확보.
19. **(v1.3 삭제 명시) v1.2 caveat #16 (compound 효과 의문) 의 reasoning 오류** — v1.2 의 "G1 corrector 성공 시 selector 의 ranking 부담 *감소*" 주장은 cand 수 동일 + hit zone 후보 수 ↑ 시 *tie-breaking 부담 증가* 로 사실은 *유지 또는 증가*. v1.3 의 G1 ranking → G2 corrector 순서로 *해당 framing 자체 무관* (G2 가 G1 위 compound 측정) — caveat 삭제.

---

## §N+4. 변경 이력

- v1 (2026-05-12): 초안. plan-008 의 main_bottleneck="ranking" 결론 + carry-over 2 항목 (LB + corrector hook) 통합. Phase 1 (cheap, no arch) + Phase 2 (mid) + Phase 3 (big, 조건부) sequence 채택.
- v1.1 (2026-05-12): **LB 제출 정책 0 회로 변경** (할당량 소진 인계). G0 = preflight (LB 회수 → plan-008.1 그대로 carry-over) + G_final = best submission *경로* 박제 (LB 미제출, plan-009.1 carry-over). caveat #9 갱신, severe `lb_quota_exhausted` 제거. spec @ §0/§0.5/§2/§4/§10/§N+1/§N+3.
- v1.2 (2026-05-12): main lever 재정렬 — corrector 강화 가 main, ranking loss 가 secondary. 사용자 challenge 반영 (oracle 1.5cm = 84.78% 발견). [SUPERSEDED by v1.3]
- **v1.3-final (2026-05-13, LB closed)**: **post-hoc LB finding 박제**. plan-008.1 LB **0.6812** (est. 0.6723, +0.0089 gap = +0.0309 OOF→LB gap), plan-009.1 LB **0.6748** (est. 0.6873, -0.0125 gap = +0.0095 OOF→LB gap). pair-delta = **−0.0064** (OOF 추정 +0.0150 → LB **REGRESSION**). plan-008 baseline LB 0.6812 = plan-009 ceiling — 본 plan 의 모든 후보 (G1 H001 SEVERE FAIL, G2 H002 b 0.6748) 가 plan-008 baseline 미달. 재해석: corrector framework 의 *본질 한계* (caveat #13 framework limit 결론 회복) — band/cap/arch lever 의 어느 것도 plan-008 baseline 위 LB 향상 X. 1-fold approx + band over-fit 결합 의심 (plan-009.1 OOF→LB gap +0.0095 << plan-008.1 +0.0309). plan-010 권장 격상: ranking loss design ablation + plan-008 baseline 5-fold reproduce + framework 교체 (KNN/GP/plan-006 회귀, LB<0.69 시나리오 D 충족).
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
