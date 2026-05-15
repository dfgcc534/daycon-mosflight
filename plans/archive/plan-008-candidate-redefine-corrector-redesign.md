---
plan_id: 008
version: 2.7
date: 2026-05-12 (Asia/Seoul)
status: partial (carry-over)
based_on:
  - 004
  - 005
  - 006
  - 007
  - notes/PB_0.6822 코드공유.ipynb
scope: candidate pool greedy set-cover expansion (Strategy D, oracle 0.85+ target, stretch 0.90) + pruning + (secondary) corrector band-specific on Variant A baseline (regime infra discarded). Family 4 per_regime drop (regime backdoor), snap drop (4th derivative noise), fs_3d_binormal reformulation.
exp_ids:
  - G001_candidate-redefine
  - G002_corrector-band   # DEFERRED to plan-009
lb_score: null  # TBD — quota_exhausted_2026-05-12, plan-008.1 carry-over
severe_flags:
  - redefinition_severely_insufficient (G1)
  - selector_no_improvement (G2)
warn_flags: [diagnostic_inconclusive, sanity_baseline_drift, family_effect_marginal]
---

# plan-008 v2.6 — Greedy Set-Cover Expansion + Containment-Based Pruning + Corrector Band-Specific (on Variant A)

## §0. 한 줄 목적

> **plan-008 의 main lever + 2 secondary lever (v2.3 reviewer 피드백 반영):**
>
> 1. **★ Candidate pool greedy set-cover expansion (Strategy D)** — oracle **0.7188 → 0.85+ (aspirational target, stretch 0.90)**. *예상 spec-simulate 결과 0.78~0.82* (§N+3 caveat #20, diminishing-returns 가정); 본 plan 의 *intended outcome band* 는 `0.78 ≤ oracle < 0.85` warn-only 범위, 0.85 도달은 set-cover marginal 회수율이 caveat 추정보다 높을 때 보너스. severe (`redefinition_severely_insufficient`) 트리거는 `oracle < 0.78`. 후보 template_pool 에서 *oracle 기여도 가장 큰* 것을 greedy 하게 add (set cover 인식). 모든 family pre-defined + 동시 추가 X — *data-driven 순차 선택*.
>
> 2. **(secondary) Pruning — structural redundancy 기반 (v2.4 reviewer 피드백 반영)** — 27 후보의 *pairwise containment / coordinate similarity* 분석으로 redundant 후보 식별 (selector pick rate *무관*). 두 후보 i, j 가 (a) `containment_strict` (hit_i ⊆ hit_j) 또는 (b) `containment_soft ≥ 0.95 + coord_dist < 0.005` + `hit_rate[j] > hit_rate[i]` 만족 시 i 제거. **Step 1 측정 후 main/secondary 위치 결정** (drift 와 ranking gap 분해 결과 의존). 효과 측정 (Δsoft_hit) 은 post-pruning sanity check 만.
>
> 3. **(secondary) Corrector band-specific 재설계** — plan-005 `corrector_oracle_gain = −0.0077` finding 의 *알려진 fix*. main lever 위에 +0.02~0.04 LB booster. cap 0.006 fallback 옵션 (→ 0.008) 포함.
>
> **Reviewer 피드백 (v2.3) 반영**:
> - Family 4 (per_regime_specialized) **drop** — regime backdoor self-contradiction (Variant A baseline 폐기 의도 위배)
> - Family 5 snap (4th derivative) **drop** — 11-pt trajectory 의 noise 4중 증폭, 노트북 L1 "보수적 baseline" 정신 위배
> - Family 3 fs_3d_banking → **fs_3d_binormal** (Frenet local frame binormal 사용, world z 직접 의존 X)
> - Selector G2 fallback 강화 — pairwise_loss / fine_distill / epoch_plus 추가 (단순 hidden 변경 외)
> - *Selector ranking 개선 자체* 는 본 plan scope X (plan-007 framework 대체 시도 실패) → plan-009 task 박제
>
> **Baseline 확정 = Variant A (GRU + physics, NO regime)** — plan-005 STAGE 6 의 LB 0.6796 (regime LB marginal +0.001 noise 입증).
>
> **LB 제출 = 본 plan 내 0 회** (할당량 소진). submission.csv 생성만 + plan-008.1 carry-over.
>
> **Target LB**: **0.78~0.85**. LB 회수는 *plan-008.1*.

---

## §0.5 Quick Reference (autonomous loop 매 turn 읽는 section)

### 합격 기준 (G-gate sequence)

- Step 1 진단 (v2.5): **oracle miss sample (~2800)** residual decomposition + 가지치기 후보 list + softmax diffusion 측정 + per-regime oracle gap (sanity only) + **`argmax_hit` 측정 (selector 의 top-1 픽 좌표 hit rate) + ranking-vs-drift 분해**. 위반 시 `diagnostic_inconclusive` warn-only. **v2.4 → v2.5**: mask `worst_regime ∈ {10,16,17}` → `oracle_miss = err.min > 0.01` (Variant A regime 폐기 정합, main lever 의 직접 target).
- Step 2a 가지치기: oracle 변화 < 0.001 인 redundant 후보만 제거. 제거 후 `oracle_after ≥ 0.7170` (= plan-005 oracle 0.7188 − 0.0018 허용). 위반 시 `oracle_drop` severe.
- Step 2b **Greedy set cover (Strategy D)**: pruned pool 에서 시작 → `template_pool` (모든 family 후보) 에서 *oracle 증가 최대* 인 후보를 greedy 하게 1 개씩 add. 종료 조건: `delta < 0.001` OR `len(pool) >= 50` OR `oracle ≥ 0.90`. 최종 `oracle_final ≥ 0.85` minimum, stretch 0.90. **0.78 ≤ oracle < 0.85** 시 warn-only. **oracle < 0.78** 시 `redefinition_severely_insufficient` severe.
- Step 3 selector 재학습 (Variant A path, `regime_prior_strength=0`): (a) OOF hit ≥ Variant A baseline 0.6570 + 0.043 ≈ `0.70` minimum (v2.4 완화, 이전 0.71). (b) **(v2.6 신규)** Sanity baseline 측정: 동일 hyperparam 으로 27 후보 OOF 가 Variant A 0.6570 ± 0.005 재현. (c) **(v2.6 신규)** `family_effect = oof_extended − sanity_baseline_27 ≥ +0.03` (family 의 *순* 회수 효과). **LB 미제출 (할당량 소진)** — submission.csv 생성만. (a) 위반 시 `selector_no_improvement` severe. (b)(c) 위반 시 warn-only (`sanity_baseline_drift` / `family_effect_marginal`).
- Step 4 corrector 재설계 (secondary): per-band hit — `[0.5, 1cm] hit_after ≥ 0.95` ∧ `[1, 1.5cm] hit_after ≥ 0.30`. 전체 OOF ≥ Step 3 OOF + 0.02 (secondary booster minimum, 이전 +0.04 → +0.02 로 완화). **LB 미제출** — submission.csv 생성만. 위반 시 `corrector_band_failure` severe.
- **LB 제출: 본 plan 내 0 회** (오늘 할당량 소진). submission.csv 는 Step 3/4 끝에 생성만 + 박제. LB 회수는 *plan-008.1 carry-over* (다음 날 사용자 수동 dacon-submit 호출).
- `lb_score` frontmatter 3 파일 (`plans/plan-008-*.md` top + `.results.md` + `analysis/plan-008/results.md`) 동시 박제. plan-004/006/007 패턴 답습.

### G-gates

- G0: STAGE 1 진단 (v2.5: **oracle miss mask** residual decomposition + 가지치기 list + softmax diffusion + per-regime oracle gap (sanity) + ranking-vs-drift 분해) [DONE ebd4979] — n_oracle_miss=2812, prune=24 (strict), main_bottleneck=ranking, warn=diagnostic_inconclusive
- G1: STAGE 2 후보 풀 재정의 완료 — Step 2a (가지치기, oracle ≥ 0.7170) + Step 2b (**Greedy set cover Strategy D**, oracle ≥ 0.85) [DONE b22f86c, SEVERE redefinition_severely_insufficient — oracle_final=0.7543 < 0.78]
- G2: STAGE 3 selector 재학습 (Variant A path) — OOF ≥ 0.70 + submission.csv 생성 (LB 미제출) [DONE 1a8c05c, SEVERE selector_no_improvement — oof_soft=0.6503, family_effect +0.0037]
- G3: STAGE 4 corrector 재설계 (secondary) — band hit 검증 + 전체 OOF ≥ Step 3 + 0.02 + submission.csv 생성 (LB 미제출) [DEFERRED 4277a21 — boundary.py LOSS_ATTR 부재, plan-009 hook 신설 필요]
- G4: (선택) STAGE 5 test-internal validation [SKIPPED — 선택 stage, G2 severe 위 ROI 낮음]
- G_final: STAGE 6 synthesis + plan-009 후보 + 3 파일 frontmatter 동시 박제 (`lb_score: TBD` — carry-over) [DONE 5f47c0a — results.md + next_plan_candidates.md (4 시나리오 + 6 카테고리 ROI 표 + Phase 1~4) + plan-008.results.md sync]

### Commit chain (next-up)

| # | type | spec section | status |
|---|---|---|---|
| c1 | docs | `plans/plan-008-candidate-redefine-corrector-redesign.md` v1 작성 | [DONE] |
| c1.1 | docs | v2 spec 갱신 — radical expansion + Variant A baseline 확정. spec @ §0~§7, §N+3 | [DONE in v2/v2.1] |
| c1.2 | docs | v2.2 spec 갱신 — Option A (CandidateSpec schema 확장). spec @ §2.1/§5.2.0/§6.1.5/§N+3 #11 | [DONE in v2.2] |
| c1.3 | docs | v2.3 spec 갱신 — reviewer 피드백 (Family 4/5 drop, fs_3d_binormal, cap fallback, selector fallback 강화) + Strategy D (greedy set cover) + Step 1 ranking-vs-drift 분해. spec @ §0~§N+4 | [DONE in v2.3] |
| c1.4 | docs | v2.4 spec 갱신 — pruning 기준 변경 (selector pick rate → structural containment). spec @ §0/§0.5/§1.4/§2.1/§4.1/§5.1/§N+4 | [DONE a52b984] |
| c1.5 | docs | v2.5 spec 갱신 — STAGE 1 mask `worst_regime ∈ {10,16,17}` → `oracle_miss` (Variant A 정합, main lever 직접 target). G2 OOF 0.71 → 0.70 완화. spec @ §0/§0.5/§4 + caveat 박제 | [DONE a52b984] |
| c1.6 | docs | v2.6 spec 갱신 — Plan agent 검토 반영. (1) §7.1 LB 잔재 fix, (2) §6.0 sanity_baseline_27 신설 (family 효과 분리), (3) caveat #20 (oracle 0.85 낙관), (4) §6.2 assert 강화 (regime_bias_table 분산). spec @ §0/§0.5/§6.0/§6.2/§6.4/§6.6/§7.1/§7.7/§N+3/§N+4 | [DONE a52b984] |
| c1.7 | docs | v2.7 spec 갱신 — plan-review-master 5-iter sweep (16 BLOCKER + 14 AMB fix) + §4.1 pruning auto-relaxation. spec @ §0/§0.5/§3/§4/§5/§6/§7/§N+3 #17/§N+4 | [DONE ce7366c] |
| c2 | code | `analysis/plan-008/diagnostic.py` — STAGE 1 진단 (**oracle miss residual** + structural pruning containment + softmax diffusion + per-regime oracle sanity + ranking-vs-drift 분해). spec @ §4 | [DONE ebd4979] |
| c2.5 | code | `src/pb_0_6822/selector.py` partial 수정 — CandidateSpec schema 확장 (Option A, 3 곳). spec @ §5.2.0 | [DONE 89f3b3f] — smoke 6 pass, dim (27,16) (27→32) ✓ |
| G0 | gate | diagnostic.{json,md} 박제 + dominant cause(s) + prune list + margin 분포 + per-regime oracle gap | [DONE ebd4979] — warn diagnostic_inconclusive (dominant_causes=[]), main_bottleneck="ranking" |
| c3 | code | `src/pb_0_6822/candidates_extended.py` — 5 family 후보 정의 모듈 (Family 4 drop, snap drop, fs_3d_binormal). spec @ §5.2 | [DONE 9e8b61f] |
| c4 | code | `analysis/plan-008/prune_and_redefine.py` — Step 2a (prune, v2.7 incremental safety) + Step 2b (**Greedy set cover Strategy D**). spec @ §5 | [DONE b22f86c] |
| c5 | exp | G001-step2: oracle 측정 (pruned 0.7173 + greedy 6 templates added → oracle_final 0.7543). spec @ §5 | [DONE b22f86c] |
| G1 | gate | Step 2a 0.7173 ≥ 0.7170 ✓ ∧ Step 2b 0.7543 < 0.85 ✗ (**SEVERE redefinition_severely_insufficient**, plan-009 carry-over per §0.5) | [DONE b22f86c, SEVERE] |
| c5.5 | code | `analysis/plan-008/sanity_baseline_27.py` — 27 후보 + 새 hyperparam Variant A 5-fold OOF baseline 측정 (v2.6 신규, family 효과 분리용). spec @ §6.0 | [DONE 637f7e2, 1a8c05c run] — soft_hit=0.6466 |
| c6 | code | `analysis/plan-008/selector_retrain.py` — Variant A path 강제 wrapper (regime_prior_strength=0). spec @ §6 | [DONE 637f7e2 + 48adfed + 62b6344] |
| c7 | exp | G001-step3: 5-fold selector + 기존 corrector full-fit + submission 생성 (LB 미제출). spec @ §6 | [DONE 1a8c05c] — soft=0.6503 oracle=0.7562 |
| ~~c8~~ | ~~sub-lb~~ | 본 plan 내 미수행 (LB 할당량 소진). plan-008.1 carry-over (다음 날). spec @ §8 | [DEFERRED] |
| G2 | gate | OOF ≥ 0.70 ✗ (0.6503) + family_effect ≥ +0.03 ✗ (+0.0037) → **SEVERE selector_no_improvement + 2 warn**. §6.5 fallback skip (autonomous, family_effect marginal). | [DONE 1a8c05c, SEVERE+2WARN] |
| c9 | code | `analysis/plan-008/corrector_band.py` — band-specific corrector loss + 학습 wrapper. spec @ §7 | [DONE 4277a21] — LOSS_ATTR=None detect, defer |
| c10 | exp | G002-step4: corrector 재학습 + per-band hit 측정. spec @ §7 | [DEFERRED 4277a21] — plan-009 (boundary.py hook 부재) |
| ~~c11~~ | ~~sub-lb~~ | 본 plan 내 미수행 (LB 할당량 소진). plan-008.1 carry-over. spec @ §8 | [DEFERRED] |
| G3 | gate | per-band hit + 전체 OOF + submission — DEFERRED (plan-009) | [DEFERRED 4277a21] |
| c12 | code | (선택) `analysis/plan-008/test_internal.py` — STAGE 5. spec @ §9 | [SKIPPED] — 선택 stage, G2 severe 위 ROI 낮음 |
| c13 | exp | (선택) G002-step5: test-internal grid search. spec @ §9 | [SKIPPED] |
| G4 | gate | (선택) gap 50%+ 회수 | [SKIPPED] |
| c14 | synthesis | `analysis/plan-008/results.md` + `next_plan_candidates.md` (≥ 2 시나리오 후보 + **§10.2.1 의 ranking 6 카테고리 ROI 표 필수 박제**, v2.6). spec @ §10 | [DONE 5f47c0a] |
| G_final | gate | results.md + next plan 후보 ≥ 2 + 3 파일 frontmatter 동시 박제 | [DONE 5f47c0a] |

### Plan-specific severe (WORKFLOW.md §12.3 default 위 추가분)

- `diagnostic_inconclusive`: Step 1 의 dominant cause 가 어느 것도 임계 (`corr > 0.3` 또는 `z_pct > 0.20`) 도달 못함. warn-only — Step 2b 의 family 선택을 selection_rate fallback 으로.
- `oracle_drop`: Step 2a 가지치기 후 `oracle_pruned < 0.7170`. severe — 가지치기 일부 rollback.
- `redefinition_severely_insufficient`: Step 2b 의 최종 `oracle_final < 0.78`. severe — 5 family 모두 효과 부족, 진단 재검토 + 새 family 시도 (또는 plan-009 carry-over).
- `redefinition_partial`: Step 2b 의 `0.78 ≤ oracle_final < 0.85`. **warn-only** (severe X) — minimum 미달 but 부분 효과. plan-009 family 후속 list 박제.
- `selector_no_improvement`: Step 3 OOF < 0.70. severe — 새 후보 풀이 selector 에게 *학습 가능* 한지 검증 (selector hidden 48→64 hyperparam fallback 가능).
- **`sanity_baseline_drift`** (v2.6): Step 3 의 sanity_baseline_27_oof 가 Variant A baseline 0.6570 ± 0.005 밖. **warn-only** — hyperparam 변경 효과가 family 효과와 혼재 가능, 결과 해석 주의 (별도 측정 단계 추가 권장).
- **`family_effect_marginal`** (v2.6): `family_effect = oof_extended_pool − sanity_baseline_27_oof < +0.02`. **warn-only** — oracle 0.85 *회수율 낮음* 신호. plan-009 의 selector arch 교체 / ranking-specific loss 후보 강화 trigger.
- `corrector_band_failure`: Step 4 의 `[0.5, 1cm] hit_after < 0.95` OR `[1, 1.5cm] hit_after < 0.30`. severe — λ tuning 또는 arch 재검토.
- `regime_residue`: Step 3 selector 학습 시 `regime_prior_strength != 0` 또는 `fit_regime_bins/assign_regimes/candidate_regime_bias` 호출 발견. severe — Variant A path 위반.
- `schema_v22_residue`: Step 3 selector 학습 진입 시 §6.1.5 의 assert 위반 (CandidateSpec 의 4 신규 fields 부재 또는 backward-compat 깨짐 또는 cand_dim 예상값 다름). severe — Option A 수정 누락.
- **`regime_backdoor_residue`** (v2.3): `candidates_extended.py` 또는 template_pool 에 `reg16`/`reg17`/`reg10` 같은 regime ID 식별자를 후보 이름/계수에 hard-code 한 후보 발견. severe — Family 4 drop 결정 위반, Variant A baseline 의 regime 폐기 의도 실질 위배.
- **`world_axis_dependence`** (v2.3): `candidates_extended.py` 에 world z 축 직접 의존 후보 (`z_scale` field 의 *명시적* world coordinate 사용) 발견. severe — fs_3d_binormal 재정의 위반, Frenet local frame invariance 의도 위배. (단 motion_terms 의 d1.z, acc.z 등 *원천 trajectory 좌표* 사용은 허용 — 후보 정의의 *명시적 world bias* 가 trigger.)
- `submission_shape_mismatch`: plan-004/006/007 동일.
- ~~`lb_unsubmitted`~~: **본 plan scope X** (LB 미제출 — 할당량 소진). plan-008.1 carry-over 의 trigger 가 됨.
- ~~`dacon_submit_skill_missing`~~: **본 plan 미사용** (LB 미제출).
- ~~`lb_anomaly`~~: 본 plan 내 LB 회수 X — plan-008.1 carry-over 시 trigger (`|lb_score − 0.6796| ≥ 0.05`).

### Plan-specific paths (WORKFLOW.md §12.5/§12.6 추가/제외)

- whitelist 추가:
  - `analysis/plan-008/**` (특히 `diagnostic.py`, `prune_and_redefine.py`, `selector_retrain.py`, `corrector_band.py`, `test_internal.py`, `*.json`, `*.md`)
  - `runs/baseline/G001_candidate-redefine/**`, `runs/baseline/G002_corrector-band/**` (submission.csv + ckpt)
  - `src/pb_0_6822/candidates_extended.py` (신규 모듈 — 5 family 정의)
  - **`src/pb_0_6822/selector.py` 의 partial 수정** (Option A, v2.2 결정): `CandidateSpec` dataclass, `candidate_spec_features` 함수, `make_candidate_features` 의 interactions term 만. Attn-GRU model / 학습 로직 미수정.
- blacklist 추가:
  - `src/pb_0_6822/selector.py` 의 **arch / 학습 로직 영역** (model class definitions, `train_one`, `run_fold`, `SELECTOR_MAIN`, `run_full_fit`) — schema 확장 외 수정 X
  - `src/pb_0_6822/boundary.py` (plan-004 lock-in, import only)
  - `runs/baseline/P001_*/**`, `runs/baseline/E001_*/**`, `runs/baseline/E002_*/**`, `runs/baseline/E003_*/**`, `runs/baseline/F001_*/**`, `runs/baseline/F002_*/**`
  - `analysis/plan-{004,005,006,007}/**` (이전 plan 산출, read-only)
  - `plans/plan-{001..007}*` (앞선 plan 본문 수정 X)
  - `notes/PB_0.6822 코드공유.ipynb` (원본 보존)

### Decision-note 사용 예 (자율 결정 시 commit msg 박제)

- `decision-note: spec-default — Baseline = Variant A (regime 폐기 확정, plan-005 STAGE 6 의 LB 0.6796 + regime LB marginal +0.001 noise 근거). regime_prior_strength=0 강제.`
- `decision-note: spec-default — Step 1 진단 (v2.5) = plan-005 worst-100 + corrector_decomp + margin_hist 재활용 + **oracle_miss mask residual decomposition** (≈ 2800 sample, main lever 의 직접 target) + per-regime oracle gap (sanity only, decision 무관). v2.4 의 worst_regime mask 는 self-contradiction (Variant A regime 폐기 + diagnostic regime 사용).`
- `decision-note: spec-default — Step 2a 가지치기 = **structural containment** (v2.4): pairwise containment_strict (hit_i ⊆ hit_j) 또는 containment_soft ≥ 0.95 + coord_dist < 0.005 + hit_rate[j] > hit_rate[i] → i 제거. Safety check: 제거 후 oracle delta < 0.001. selector pick rate 사용 X (selector 거동 무관, robust).`
- `decision-note: spec-default — Step 2b 재정의 = 5 family 모두 정의 (trig/arc/Frenet-Serret/per-regime/higher-order/cross-term) → oracle filter (family marginal < 0.01 시 drop, hybrid C path).`
- `decision-note: spec-default — Step 3 selector arch = plan-004 Attn-GRU 그대로 (hidden=48, layers=2). 후보 수만 27 → ~40 (pruned ~22 + new ~18). hidden 변경은 fallback 만 (G2 미달 시 64).`
- `decision-note: spec-default — Step 4 corrector loss = band-specific hinge ([0,0.5]: 0, [0.5,1]: protect λ=1, [1,1.5]: recover λ=2, [1.5,∞]: 0). λ grid search fallback (max 5 회).`
- `decision-note: spec-default — Step 5 test-internal validation = 본 plan scope 선택 + carry-over to plan-009.`
- `decision-note: spec-default — Oracle target = 0.85 minimum, 0.90 stretch. 0.78 ≤ x < 0.85 warn-only, x < 0.78 severe.`
- `decision-note: spec-default — Corrector framing = secondary (Step 3 결과 0.78+ 도달 시 corrector 생략 plan-009 이관 가능).`
- `decision-note: spec-default — DATA_ROOT = repo/data/, DEVICE = cuda:1 (plan-004/006/007 일관성).`
- `decision-note: spec-default — CandidateSpec schema 확장 (Option A, v2.2): 4 신규 fields (omega_scale, arc_curvature, z_scale → binormal_scale, family_id) + default → 기존 27 backward-compat. selector.py 의 schema/feature 함수 partial 수정 허용, arch/학습 로직 미수정.`
- `decision-note: spec-default — cand_dim 16 → 32 (per-candidate): par/perp/dist 3 + spec 16 (scalar 9 + family one-hot 7) + ctx 9 + interactions 4.`
- `decision-note: spec-default (v2.3) — Strategy D (Greedy set cover) 채택: template_pool 에서 매 iteration oracle 증가 최대인 1 개 add. 종료: delta < 0.001 OR pool ≥ 50 OR oracle ≥ 0.90. 사전 family 동시 추가 X, hybrid C path 폐기. 이유: data-driven set cover 인식, redundant 후보의 중복 add 회피.`
- `decision-note: spec-default (v2.3) — Family 4 (per_regime_specialized) drop: reviewer #1 catch — regime ID 가 후보 이름/계수에 hard-code 되면 Variant A baseline (regime 폐기) self-contradiction. 같은 dynamics 는 Family 1 trig + Family 2 arc 가 cover.`
- `decision-note: spec-default (v2.3) — Family 5 snap (4th derivative) drop: reviewer #5 — 11-pt trajectory 의 4th derivative noise 4중 증폭, 노트북 L1 보수적 baseline 위배. multi_step_rk2 만 유지.`
- `decision-note: spec-default (v2.3) — Family 3 fs_3d_banking → fs_3d_binormal 재정의: reviewer #6 — world z 직접 의존 → Frenet local frame binormal (T × N) 진폭 사용. world orientation invariant.`
- `decision-note: spec-default (v2.3) — Corrector cap fallback: 0.006 → 0.008 (1.33x) 옵션. §7.4 의 grid search 에 cap 도 포함 (G3 [1, 1.5cm) 회수 < 0.30 일 때 활성). 노트북 L1 의 "tiny correction" 정신 여전히 1cm 미만으로 유지.`
- `decision-note: spec-default (v2.3) — Selector G2 fallback 강화: hidden 48→64 (1) + pairwise_loss_weight × 1.5/2.0 (2) + fine_distill_weight × 1.5/2.0 (3) + epoch_plus 5→8/10 (4). 최대 4 fallback 시도. reviewer #4 catch — 후보 +50% 대응 장치 필요.`
- `decision-note: spec-default (v2.3) — Pruning framing 격하: main lever → "Step 1 측정 결과 의존" (ranking-vs-drift 분해 후 결정). 단 후보 풀 cleanup 효과 (selector 학습 부담 감소) 는 unconditional.`
- `decision-note: spec-default (v2.3) — Selector ranking 자체 개선 (arch 교체, ranking-specific loss) 은 본 plan scope X (plan-007 framework 대체 시도 실패). plan-009 후보로 박제.`

---

## §1. 배경

### §1.1 plan-007 인계 (key findings)

| 측정 | 값 | 출처 |
|---|---|---|
| 단일 공식 framework raw ceiling (CMA-ES + MLP coeff) | **0.6482** | plan-007 Step 4 |
| LB (plan-007 Step 3 best basis) | 0.6598 | plan-007 lb_log.md |
| **Oracle (best of 27, raw)** | **0.7188** | plan-005 |
| Oracle gap from raw ceiling | **−0.0706 (= 7.06pp)** | 계산 |

핵심 인계 (plan-007 진단으로 확정):
- CMA-ES local min 의심 *기각*. 0.6482 → 0.7188 의 7pp gap = **선형 family 의 구조적 한계**.
- → plan-008 main lever = **새 family 추가 (재정의), 같은 family 의 확장 X**.

### §1.2 plan-005 인계 (corrector + selector finding)

| finding | 값 | 출처 |
|---|---|---|
| Corrector_oracle_gain | **−0.0077** | plan-005 oracle_summary.json — corrector 가 oracle 깎음 |
| Corrector 의 [0.5, 1cm] band 부작용 | **−7.83pp** (hit 100% → 92.17%) | plan-005 corrector_decomp.md |
| Corrector 의 [1, 1.5cm] band 회수 | +9.77% (0% → 9.77%) | plan-005 corrector_decomp.md |
| Selector top-1 ranking 정확도 (= 1등 픽 == 27 중 *정답에 가장 가까운* 후보 = oracle best) | **12.6%** | plan-005 selector_decomp |
| Selector argmax hit (= 1등 픽이 1cm 안 들어오는 비율, *절대 best 아니어도 OK*) | ~0.65 (추정, Step 1 측정) | 미박제 |
| **Ranking gap** (= oracle 0.7188 − selector argmax hit, "selector 가 *hit 후보를 놓치는* 비율") | **~7pp** (추정, Step 1 정확 측정) | Step 1 진단 |
| Worst regime (16/17/10) hit | 0.22 / 0.26 / 0.41 | plan-006 §2.3 |

**`top-1 ranking 12.6%` 의 *정확한* 의미 (중요)**:
- "selector 가 *27 후보 중 정답에 *가장 가까운* 후보* 를 1등으로 픽한 비율 = 12.6%"
- **NOT** "selector 1등 픽이 1cm 안에 들어가는 비율"
- 1cm 안 (hit zone) 에 후보가 *여럿* 있을 때 selector 가 *2등이나 3등* 픽해도 hit OK
- → 실제 selector argmax hit 은 ~65% (= soft hit 0.66 와 비슷, ~0.5pp 차이)
- **진짜 병목 = ranking gap 7pp** (= oracle 71.88% − argmax ~65%): "후보 풀에 hit 가 *존재* 하는데도 selector 가 *non-hit 후보* 를 1등 픽한 sample"

### §1.3 plan-005 STAGE 6 인계 (**Variant A baseline 정당성**)

| Variant | OOF (soft) | LB | OOF→LB gap | 비고 |
|---|---|---|---|---|
| full (GRU + physics + regime) | 0.6599 | 0.6806 | +0.0207 | plan-004 측정 |
| **A (GRU + physics, NO regime)** | 0.6570 | **0.6796** | **+0.0226** | E002 (max gap) |
| B (physics + regime, no GRU) | 0.6547 | 0.6704 | +0.0157 | E003 |
| E (physics only, no GRU/regime) | 0.6524 | 0.6692 | +0.0168 | plan-006 |

**핵심**:
- **regime LB marginal = +0.0010pp ≈ noise** (full − A). → regime infra 폐기 확정.
- GRU LB marginal = +0.0102pp (real lift). → GRU 유지.
- **Variant A 의 OOF→LB gap = +0.0226 (max)** → 본 plan 의 OOF 개선이 LB 에서 *더 큰 amplification*.
- **Variant A 가 본 plan baseline**: LB 시작점 0.6796.

### §1.4 본 plan 의 핵심 가설

| 가설 | 검증 방법 | 합격 산출 |
|---|---|---|
| H1 (v2.5): **oracle miss sample (~2800)** 의 residual 이 *특정 dynamics* (rotation/curvature/z) 와 강하게 상관 | Step 1 oracle_miss mask residual decomposition | dominant cause 1~3 개 → template_pool 구성 가이드 |
| **H1.5 (v2.3 신규)**: 6pp selector→oracle gap 분해 시 `ranking_recoverable_gap` vs `drift_component` 비율 | Step 1 `argmax_hit` 측정 + 분해 | 측정 결과 (ranking dominant / drift dominant / mixed) |
| H2: structural containment 기반 가지치기로 redundant 후보 ≥ 5 개 제거 + oracle 손실 < 0.001 | Step 2a containment matrix + safety check | pruned pool, pairwise containment 박제 |
| **H3 (v2.3 갱신)**: Greedy set cover 로 template_pool 에서 순차 add → oracle ≥ 0.85 (stretch 0.90) | Step 2b greedy iteration | extended pool + iteration log |
| H4: Variant A path 로 새 후보 풀 selector 학습 → OOF ≥ 0.70 | Step 3 5-fold OOF | OOF |
| H5: corrector band-specific 재설계로 per-band hit 회복 + 전체 OOF +0.02 | Step 4 OOF + per-band | OOF |

**H5 정량 derivation (v2.6)**: plan-005 `corrector_decomp.md` 측정값 — corrector 가 `[0.5, 1cm]` band 의 hit 을 100% → 92.17% (−7.83pp) 깎고 `[1, 1.5cm]` band 의 hit 을 0% → 9.77% (+9.77pp) 회수. Population-weighted 전체 효과 ≈ `−0.078 × p_band[0.5,1] + 0.098 × p_band[1,1.5]` ≈ −0.0077 (= `corrector_oracle_gain` 일치). Band-specific re-design 의 목표 = 손실 측 (band [0.5,1]) `−7.83pp → ≥ −2pp` (회수 5.83pp × p_band[0.5,1] ≈ 0.20) → 전체 +0.012. 동시에 회수 측 (band [1,1.5]) `+9.77pp → +18~20pp` (cap 0.006→0.008 fallback) → 전체 +0.016. 합 ≈ **+0.025~0.030**, 안전 buffer 적용 → minimum +0.02 박제.
| H6: (선택) test-internal hyperparam re-tune | Step 5 grid search | gap 회수율 |

→ H1~H5 *데이터로 검증*. **main lever = H3 (greedy set cover)**, secondary = H2 (pruning, H1.5 결과에 따라) + H5 (corrector).

---

## §2. Scope (명시적)

### §2.1 In-scope

| 항목 | 값 |
|---|---|
| Baseline | **Variant A** (GRU + physics, regime_prior_strength=0) — plan-005 STAGE 6 path 답습 |
| 후보 풀 출발점 | plan-004 의 27 candidates (`src/pb_0_6822/selector.CANDIDATES`) |
| 가지치기 | **structural containment** (v2.4): pairwise hit-set containment + coord similarity + hit_rate dominance. Safety check: oracle delta < 0.001. selector pick rate *사용 X* (selector 거동 무관). |
| 새 family (5+, 20+ 후보) | trig (rotation, 3~5) + circular_arc (2~3) + Frenet-Serret 3D (2~3) + per-regime specialized (3) + higher-order jerk/snap/multi-step (2~3) + cross-term (speed_slope·d1 등, 2~3) |
| Family-level filter | Step 2b 의 oracle marginal contribution < 0.01 시 drop (hybrid C path) |
| Selector 재학습 | plan-004 의 Attn-GRU + `regime_prior_strength=0` |
| Corrector 재설계 (secondary) | loss 함수만 교체 (band-specific hinge), arch 유지 |
| **CandidateSpec schema 확장** (선택 Option A 채택) | `src/pb_0_6822/selector.py` 의 `CandidateSpec` dataclass + `candidate_spec_features` 함수 + `make_candidate_features` 의 interactions term 확장. 새 fields: `omega_scale`, `arc_curvature`, `z_scale`, `family_id`. 기존 27 후보는 default=0 → backward-compat 유지. |
| LB 제출 | **0 회 (본 plan scope X — 할당량 소진)**. submission.csv 생성만, dacon-submit skill 호출 X. 다음 날 plan-008.1 carry-over 로 *사용자 수동* 제출. |
| 산출 위치 | `analysis/plan-008/**`, `runs/baseline/G001_*/**`, `runs/baseline/G002_*/**`, `src/pb_0_6822/candidates_extended.py`, `src/pb_0_6822/selector.py` (schema 확장만, partial 수정) |

### §2.2 Out-of-scope (절대 안 함)

| 항목 | 이유 |
|---|---|
| **LB 자율 제출 (dacon-submit skill 호출)** | **오늘 일일 할당량 소진** — 본 plan 내 LB 호출 X. submission.csv 생성만 + 다음 날 carry-over 수동 제출. |
| 다중 LB 제출 | 본 plan 0 회. |
| **Regime infra 전체** (`fit_regime_bins`, `assign_regimes`, `candidate_regime_bias`) | plan-005 STAGE 6 의 LB +0.001 noise 측정 → 폐기 확정. v2.5: 학습/모델 입력 X. Step 1 진단의 **informational sanity only** (per_regime_oracle_sanity 표 + oracle_miss_regime_dist_sanity 박제) — decision 무관. main residual decomposition 은 `oracle_miss` mask 위에서 옴. |
| 18-bin regime bias 표 | 폐기. 새 family 후보의 EB cell degeneracy 위험 회피. |
| Selector arch 교체 (TCN/Transformer/MLP coeff) | plan-007 의 framework 대체 시도가 0.6482 ceiling 확정. 본 plan 은 Variant A path 유지 (arch 동일). |
| `src/pb_0_6822/selector.py` 의 **arch / 학습 로직** 수정 | lock-in. **단 CandidateSpec schema 확장 + cand_feat 함수의 spec/interactions 부분 확장은 허용** (Option A, v2.2 결정). Attn-GRU model class / `train_one` / `run_fold` / `SELECTOR_MAIN` 등 학습 로직은 미수정. |
| `src/pb_0_6822/boundary.py` 수정 | lock-in, import only. corrector loss 재학습은 wrapper 에서 monkey-patch. |
| ~~다중 LB 제출 (3 회 이상)~~ | **v2.1 잔재 삭제** — 본 plan LB = 0 회 (할당량 소진, carry-over). 위 row 와 동일 의미. |
| End-to-end 학습 | plan-004 의 2-stage sequential 유지. |
| 27 후보의 *family 정의 변경* | 가지치기는 *제거* 만, 기존 후보 수정 X. |

---

## §3. 사전 등록 (Pre-registration)

### §3.1 입력 데이터 + 분할

| 분할 | 출처 | 사용 |
|---|---|---|
| Train (10K) | `data/train/` + `train_labels.csv` | Step 1~4 fit |
| Test (10K) | `data/test/` | Step 3/4 inference + submission |
| Fold 정의 | `selector.stable_fold_id(sample_id, 5)` | OOF 정합성 |
| Test internal (50K, 선택) | `data/test/` 의 sub-trajectory | Step 5 (선택) hyperparam re-tune |

### §3.2 합격 기준 (정량)

- **G0**: `analysis/plan-008/diagnostic.json` 의 `dominant_causes` ≥ 1 entry + `prune_candidates` list + `margin_top1_top2` 분포 + `per_regime_oracle_gap` 박제.
- **G1**:
  - Step 2a: `oracle_after_prune ≥ 0.7170` (= 0.7188 − 0.0018)
  - Step 2b:
    - 5 family 각각 marginal contribution 측정 → `< 0.01` 인 family drop (hybrid filter)
    - 최종 `oracle_final ≥ 0.85` (minimum)
    - `0.78 ≤ oracle_final < 0.85` → warn-only (plan-009 후속)
    - `oracle_final < 0.78` → severe
  - Per-regime worst (16/17/10) 의 `oracle_after_final ≥ 0.55` — **warn-only sanity** (regime infra 폐기 확정 §2.2, regime 사용 = diagnostic-time *informational* only, decision/severe 무관)
- **G2**: 5-fold OOF hit (soft) **≥ 0.70** (Variant A baseline 0.6570 + 0.043 minimum, v2.4 완화 — 이전 0.71) + `submission.csv` schema OK. **LB 회수 X (carry-over)**.
- **G3**:
  - `[0, 0.5cm) hit_after ≥ 0.99`
  - `[0.5, 1cm) hit_after ≥ 0.95`
  - `[1, 1.5cm) hit_after ≥ 0.30`
  - 전체 OOF ≥ Step 3 OOF + 0.02 (secondary minimum, 이전 +0.04 → +0.02 완화 — main lever 가 candidate-level 이므로 corrector 는 booster)
  - **LB 회수 X (carry-over)**
- **G4** (선택): `oof_to_lb_gap_recovered ≥ 0.01` (Step 5 수행 시 — 본 plan 미제출 + test-internal 추정만).
- **G_final**: `results.md` + `next_plan_candidates.md` (후보 ≥ 2) + 3 파일 frontmatter `lb_score: TBD` 동시 박제 (carry-over 명시). 다음 날 plan-008.1 commit 으로 LB 회수 시 `<float>` 갱신.

### §3.3 평가

- **OOF hit**: 5-fold OOF concatenated hit@1cm
- **Oracle**: `(err.min(axis=1) <= 0.01).mean()` (best-of-N)
- **Per-band hit**: `[0, 0.5), [0.5, 1.0), [1.0, 1.5), [1.5, 2.0), [2.0, ∞)` 5 bin
- **Per-regime hit**: 18 regime (filter 용도, 모델 입력 X)
- **Family marginal contribution**: 각 family 추가 시 oracle 변화 (cumulative ablation)
- ~~**LB**: 2 회 (Step 3 + Step 4)~~ → **v2.1+ : LB 본 plan 내 0 회** (할당량 소진, submission.csv 2 종 생성만, plan-008.1 carry-over).

---

## §4. STAGE 1 — 진단 (c2)

### §4.1 측정 식

```python
# analysis/plan-008/diagnostic.py
import json
import numpy as np
from pathlib import Path
from src.pb_0_6822 import selector

REPO = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO / "data"
PLAN005_DIR = REPO / "analysis/plan-005"
PLAN006_DIR = REPO / "analysis/plan-006"
PLAN007_DIR = REPO / "analysis/plan-007"
ANALYSIS_DIR = REPO / "analysis/plan-008"

R_HIT = 0.01

def stage1_diagnostic() -> dict:
    """v2.5: Oracle miss sample 의 residual decomposition + 가지치기 후보 + softmax diffusion + per-regime oracle gap (sanity only).

    v2.4 → v2.5 핵심 변경:
      - mask: `worst_regime ∈ {10, 16, 17}` → `oracle_miss = err.min(axis=1) > R_HIT`
      - 이유: regime 폐기 (Variant A) 와 정합 + main lever (oracle 천장 회수) 의
              *직접* target = oracle miss sample (27 후보 모두 1cm 밖) 분포 분석.
      - regime 사용 = informational sanity only (per_regime_oracle 표는 유지하되 decision 무관).
    """

    # ── 1. 입력 로드 ──
    ids, train_y = selector.read_labels(DATA_ROOT / "train_labels.csv")
    train_x = selector.load_stack(DATA_ROOT / "train", ids)
    end_idx = train_x.shape[1] - 1
    # v2.6 boundary check: 11-pt trajectory → end_idx=10. higher_order_coord_func 의 진짜 jerk 는
    # end_idx ≥ 4 필요 (`x[end_idx-3]` 까지 indexing). §5.2.1 의 jerk_vec 정합 보장.
    assert end_idx >= 4, f"trajectory 길이 부족 — end_idx={end_idx}, jerk/curvature 식 indexing 불가"

    cands = selector.make_candidates(train_x, end_idx, horizon=2)
    bins = selector.fit_regime_bins(train_x, end_idx)         # sanity only (per_regime_oracle 표)
    regimes = selector.assign_regimes(train_x, end_idx, bins) # sanity only

    z_oof = np.load(PLAN005_DIR / "corrected_oof.npz")
    corrected_cands = z_oof["corrected"]

    # plan-004 (full) selector scores (가지치기 분석용)
    z_scores = np.load(REPO / "runs/baseline/P001_pb-0-6822-fullrun" / "oof_selector_scores.npz")
    oof_scores = z_scores["ens_scores"]

    # ── 2. Residual decomposition (v2.5: oracle miss sample 전체) ──
    # v2.6 fix: oracle 정의 정합 — §1.1 "Oracle (best of 27, raw) = 0.7188" 은 *raw* cands 위 측정.
    #          oracle_miss_mask 는 raw err matrix 위에서 계산해야 expected miss_rate ≈ 0.2812 정합.
    #          corrected err 는 ranking gap decomposition (§4 step 4) 에서 별도 사용.
    err_raw = np.linalg.norm(cands - train_y[:, None, :], axis=2)           # raw oracle
    err = np.linalg.norm(corrected_cands - train_y[:, None, :], axis=2)     # corrected (downstream)
    best_idx = err_raw.argmin(axis=1)                                        # raw best (oracle 정의)
    best_pred = cands[np.arange(len(train_y)), best_idx]
    err_vec = best_pred - train_y

    # ⭐ v2.5 main mask: oracle miss = 27 *raw* 후보 모두 1cm 밖
    # → plan-008 main lever (oracle 천장 0.7188 → 0.85+ 회수) 의 *직접* target population
    oracle_miss_mask = err_raw.min(axis=1) > R_HIT   # shape (N,), ~2800 True (= 1 − 0.7188)
    n_oracle_miss = int(oracle_miss_mask.sum())

    # selector.motion_terms 규약 (plan-004 lock-in):
    #   p0  = train_x[:, end_idx, :]                                   # 현재 위치
    #   d1  = train_x[:, end_idx, :] - train_x[:, end_idx - 1, :]       # 1차 차분 = 속도 estimate
    #   acc = (train_x[:, end_idx, :]
    #          - 2*train_x[:, end_idx - 1, :]
    #          + train_x[:, end_idx - 2, :])                            # 2차 차분 = 가속도 estimate
    p0, d1, acc = selector.motion_terms(train_x, end_idx)
    tangent = d1 / (np.linalg.norm(d1, axis=1, keepdims=True) + 1e-8)
    err_par = (err_vec * tangent).sum(axis=1)
    err_perp_vec = err_vec - err_par[:, None] * tangent
    err_perp_xy = np.linalg.norm(err_perp_vec[:, :2], axis=1)
    err_z = err_vec[:, 2]

    d2 = train_x[:, end_idx - 1] - train_x[:, end_idx - 2]
    omega_z = np.arctan2(
        d2[:, 0] * d1[:, 1] - d2[:, 1] * d1[:, 0],
        d2[:, 0] * d1[:, 0] + d2[:, 1] * d1[:, 1]
    )
    # v2.6 fix: kinematic curvature (motion 자체) — residual 무관.
    #   K = ||d2_perp|| / ||d1||²  where d2_perp = d2 − (d2·t̂)·t̂ ;  t̂ = d1/||d1||
    eps = 1e-8
    d1_norm = np.linalg.norm(d1, axis=1) + eps
    d2_par_scalar = (d2 * tangent).sum(axis=1)                 # scalar projection on tangent
    d2_perp_vec = d2 - d2_par_scalar[:, None] * tangent
    d2_perp_norm = np.linalg.norm(d2_perp_vec, axis=1)
    curvature = d2_perp_norm / (d1_norm ** 2 + eps)             # kinematic curvature
    # 잔차 측면 perp magnitude (residual statistics 용도, dominant_causes "z_axis"/"perp" 분기) 는 따로 유지
    residual_perp_xy = err_perp_xy
    prev_acc = d2 - (train_x[:, end_idx - 2] - train_x[:, end_idx - 3])
    jerk_norm = np.linalg.norm(acc - prev_acc, axis=1)

    # v2.5: oracle miss sample 위에서 dominant cause 도출
    w = oracle_miss_mask
    err_norm_w = np.linalg.norm(err_vec[w], axis=1)
    corr_rotation = float(np.corrcoef(np.abs(omega_z[w]), err_norm_w)[0, 1])
    corr_curvature = float(np.corrcoef(curvature[w], err_norm_w)[0, 1])
    corr_jerk = float(np.corrcoef(jerk_norm[w], err_norm_w)[0, 1])

    err_par_var = float((err_par[w] ** 2).sum())
    err_perp_var = float((err_perp_xy[w] ** 2).sum())
    err_z_var = float((err_z[w] ** 2).sum())
    total_var = err_par_var + err_perp_var + err_z_var
    par_pct = err_par_var / total_var
    perp_pct = err_perp_var / total_var
    z_pct = err_z_var / total_var

    # v2.5 sanity: regime sub-breakdown (informational only, decision 무관)
    # oracle miss sample 의 regime 분포 — main 분석 결과와 정합 확인용
    oracle_miss_regime_dist = {}
    for r in range(18):
        n_r = int(((regimes == r) & oracle_miss_mask).sum())
        if n_r > 0:
            oracle_miss_regime_dist[int(r)] = {
                "n_in_miss": n_r,
                "miss_rate": float(((regimes == r) & oracle_miss_mask).sum() / max((regimes == r).sum(), 1)),
            }

    dominant_causes = []
    if corr_rotation > 0.3 or perp_pct > 0.4:
        dominant_causes.append({"cause": "rotation", "evidence": {
            "corr_rotation": corr_rotation, "perp_pct": perp_pct
        }, "recommended_family": "trig"})
    if corr_curvature > 0.3:
        dominant_causes.append({"cause": "curvature", "evidence": {
            "corr_curvature": corr_curvature
        }, "recommended_family": "circular_arc"})
    if z_pct > 0.20:
        dominant_causes.append({"cause": "z_axis", "evidence": {"z_pct": z_pct},
                                 "recommended_family": "frenet_serret_3d"})
    if corr_jerk > 0.3:
        dominant_causes.append({"cause": "jerk", "evidence": {"corr_jerk": corr_jerk},
                                 "recommended_family": "higher_order_jerk"})

    # ── 3. 가지치기 후보 도출 (v2.4: Structural containment 기반, selector pick rate 무관) ──
    # Reviewer feedback 반영: 효과 측정 (selector pick rate) 대신 *구조적 redundancy*
    # 후보 i 가 j 에 의해 dominated → i 제거 (selector 거동 무관, robust)
    K_orig = 27
    hit_matrix = (err <= R_HIT)                 # (N, 27)
    hit_rate = hit_matrix.mean(axis=0)           # (27,)
    coord_dist_matrix = np.zeros((K_orig, K_orig))
    containment_soft = np.zeros((K_orig, K_orig))  # ratio of hit_i ∩ hit_j / |hit_i|
    containment_strict = np.zeros((K_orig, K_orig), dtype=bool)

    for i in range(K_orig):
        for j in range(K_orig):
            if i == j: continue
            coord_dist_matrix[i, j] = float(np.linalg.norm(
                corrected_cands[:, i] - corrected_cands[:, j], axis=1
            ).mean())
            n_i = hit_matrix[:, i].sum()
            if n_i == 0:
                containment_soft[i, j] = 1.0
                containment_strict[i, j] = True
            else:
                both = (hit_matrix[:, i] & hit_matrix[:, j]).sum()
                containment_soft[i, j] = float(both / n_i)
                if both == n_i and hit_matrix[:, j].sum() >= n_i:
                    containment_strict[i, j] = True

    # v2.7 fix (caveat #17): LiDAR jittering + 모기 5~10mm scale 고려 시 5mm 좌표 정합은 가혹.
    #   → 보수적 default (0.95, 5mm) 로 1차 식별, 결과 < 3 개면 *자동* 완화 (0.90, 10mm) 재탐색.
    #   tier 박제 (summary.prune_threshold_tier ∈ {"strict_v2.4", "relaxed_v2.7"}).
    def _identify_prune(soft_thr: float, dist_thr: float) -> list:
        out = []
        for i in range(K_orig):
            for j in range(K_orig):
                if i == j: continue
                # 판정 1: strict containment (hit_i ⊆ hit_j) + j 가 더 많이 hit
                strict_ok = bool(containment_strict[i, j] and hit_rate[j] > hit_rate[i])
                # 판정 2: soft containment ≥ soft_thr + 좌표 거의 동일 (dist_thr 이내) + j 가 더 많이 hit
                soft_ok = bool(
                    containment_soft[i, j] >= soft_thr
                    and coord_dist_matrix[i, j] < dist_thr
                    and hit_rate[j] > hit_rate[i]
                )
                if strict_ok or soft_ok:
                    # Safety check: i 제거 후 oracle 손실 측정
                    # v2.6 정합: §1.1 oracle (best of 27, *raw*) = 0.7188 정의와 동일하게 err_raw 위에서 측정.
                    # containment / hit_matrix 는 corrected err 위에서 도출 (selector 가 본 hit 인지가 redundancy
                    # 의도와 더 부합) 이지만, oracle safety 는 *raw* baseline 위에서 정량 보장.
                    kept_mask = np.ones(K_orig, dtype=bool)
                    kept_mask[i] = False
                    oracle_after_raw = float((err_raw[:, kept_mask].min(axis=1) <= R_HIT).mean())
                    oracle_before_raw = float((err_raw.min(axis=1) <= R_HIT).mean())
                    delta = oracle_before_raw - oracle_after_raw
                    if delta < 0.001:
                        out.append({
                            "idx": i,
                            "name": selector.CANDIDATES[i].name,
                            "dominator_idx": j,
                            "dominator_name": selector.CANDIDATES[j].name,
                            "rule": "strict" if strict_ok else "soft",
                            "containment_soft": float(containment_soft[i, j]),
                            "coord_dist": float(coord_dist_matrix[i, j]),
                            "hit_rate_i": float(hit_rate[i]),
                            "hit_rate_j": float(hit_rate[j]),
                            "oracle_delta_if_removed": delta,
                        })
                    # v2.6 break 의미 명시: dominator j1 가 safety 실패해도 j2 로 retry 안 함 — *intentional single-dominator*.
                    # 이유: pair-wise containment 는 transitive 하지 않고, 한 dominator 가 cover 못 하면
                    #       structural redundancy 가 다르다는 의미이므로 다른 j 로 fallback 의도 X. (§5.1 의
                    #       aggregate safety check 가 jointly removal 의 최종 보증 — caveat #16 참고.)
                    break
        return out

    # 1차: strict default (0.95, 5mm)
    prune_candidates = _identify_prune(soft_thr=0.95, dist_thr=0.005)
    prune_threshold_tier = "strict_v2.4"
    prune_threshold_used = {"soft": 0.95, "dist": 0.005}
    # 2차: 결과 < 3 → 자동 완화 (0.90, 10mm) 재탐색 — LiDAR jittering 보정 (caveat #17)
    if len(prune_candidates) < 3:
        prune_candidates = _identify_prune(soft_thr=0.90, dist_thr=0.010)
        prune_threshold_tier = "relaxed_v2.7"
        prune_threshold_used = {"soft": 0.90, "dist": 0.010}

    # ── 4. Selector hit gap decomposition (review Point 3 — 진짜 병목 식별) ──
    # 정정: "top-1 ranking 12.6%" = "27 중 *진짜 best* 정확 픽 비율" (oracle best 와 일치).
    #       NOT "1등 픽이 1cm 안 들어가는 비율".
    # → 진짜 metric = selector argmax hit (1등 픽이 1cm 안) + ranking gap (oracle − argmax_hit).
    argmax_idx = oof_scores.argmax(axis=1)
    argmax_pred = corrected_cands[np.arange(len(train_y)), argmax_idx]   # shape (N, 3)
    argmax_err = np.linalg.norm(argmax_pred - train_y, axis=1)
    selector_argmax_hit = float((argmax_err <= R_HIT).mean())

    # Soft prediction (existing temp=0.03)
    from src.pb_0_6822 import boundary as _bnd
    soft_pred = _bnd.soft_select(corrected_cands, oof_scores, temperature=0.03)
    soft_err = np.linalg.norm(soft_pred - train_y, axis=1)
    selector_soft_hit = float((soft_err <= R_HIT).mean())

    oracle_hit = float((err.min(axis=1) <= R_HIT).mean())
    top1_ranking_acc = float((argmax_idx == best_idx).mean())

    # Gap decomposition
    gap_ranking = oracle_hit - selector_argmax_hit    # selector 가 *hit 후보* 를 놓치는 비율
    gap_drift   = selector_argmax_hit - selector_soft_hit   # soft 가 argmax 보다 손해 (양수) 또는 이득 (음수)

    selector_gap_decomposition = {
        "oracle_hit": oracle_hit,
        "selector_argmax_hit": selector_argmax_hit,
        "selector_soft_hit": selector_soft_hit,
        "top1_ranking_accuracy": top1_ranking_acc,   # informational (12.6% 가 그대로 나오는지)
        "gap_ranking": gap_ranking,                   # 진짜 main 병목
        "gap_drift": gap_drift,                       # soft drift 효과 (작을 가능성 큼)
        "main_bottleneck": "ranking" if gap_ranking > abs(gap_drift) else "drift",
    }

    # ── 5. Softmax diffusion 측정 (drift 가설의 supporting data) ──
    sorted_scores = np.sort(oof_scores, axis=1)[:, ::-1]
    margin = sorted_scores[:, 0] - sorted_scores[:, 1]
    margin_hist = {
        "p10": float(np.percentile(margin, 10)),
        "p25": float(np.percentile(margin, 25)),
        "p50": float(np.percentile(margin, 50)),
        "p75": float(np.percentile(margin, 75)),
        "p90": float(np.percentile(margin, 90)),
    }
    # 작은 margin = top1/top2 가 비슷 → soft averaging 의 centroid drift 가능성
    # 단 §4 의 selector_gap_decomposition 이 *직접 측정* — drift 가 진짜 문제인지 binary 판정
    # v2.6 unit/threshold 정당성:
    #   - oof_scores 는 selector.SELECTOR_MAIN ens_scores → softmax 입력 *logit* (plan-004 lock-in).
    #   - margin = sorted_score[:, 0] - sorted_score[:, 1] = logit 단위.
    #   - threshold 0.1: temperature=0.03 일 때 softmax(0.1/0.03) ≈ softmax(3.33) → top1/top2 weight ratio ≈ e^3.33 ≈ 28x.
    #     margin < 0.1 (= logit 단위) 면 top2 의 weight 가 top1 의 1/28 이상 = 비중 ≥ 3.5% → soft drift signal.
    #   - 단 informational only — main_bottleneck binary 결정은 selector_gap_decomposition (위 §4 step 4) 의 정량 비교.
    softmax_diffusion_signal = margin_hist["p50"] < 0.1

    # ── 6. Per-regime oracle gap ──
    per_regime_oracle = {}
    for r in range(18):
        mask = regimes == r
        if mask.sum() == 0: continue
        per_regime_oracle[int(r)] = {
            "n": int(mask.sum()),
            "current_oracle": float((err[mask].min(axis=1) <= R_HIT).mean()),
            "gap_to_target": float(0.85 - (err[mask].min(axis=1) <= R_HIT).mean()),
        }

    # ── 7. 박제 (v2.5: oracle miss 기반 main + regime sanity 별도) ──
    summary = {
        "mask_strategy": "oracle_miss_v2.5",   # ← v2.4 worst_regime 에서 변경
        "n_oracle_miss": n_oracle_miss,
        "oracle_miss_rate": float(oracle_miss_mask.mean()),   # ≈ 0.2812 (= 1 − 0.7188)
        "residual_breakdown_oracle_miss": {                    # ← v2.4 _worst 에서 변경
            "par_pct": par_pct, "perp_pct": perp_pct, "z_pct": z_pct,
            "corr_rotation": corr_rotation, "corr_curvature": corr_curvature,
            "corr_jerk": corr_jerk,
        },
        "dominant_causes": dominant_causes,
        "prune_candidates": prune_candidates,
        "prune_count": len(prune_candidates),
        "prune_threshold_tier": prune_threshold_tier,    # v2.7: "strict_v2.4" or "relaxed_v2.7"
        "prune_threshold_used": prune_threshold_used,    # v2.7: {soft, dist} 실제 사용값
        "selector_gap_decomposition": selector_gap_decomposition,  # main_bottleneck 판정
        "margin_top1_top2": margin_hist,
        "softmax_diffusion_signal": softmax_diffusion_signal,
        # 아래 두 항목 = informational sanity only (regime 사용, decision 무관)
        "per_regime_oracle_sanity": per_regime_oracle,
        "oracle_miss_regime_dist_sanity": oracle_miss_regime_dist,
    }
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    (ANALYSIS_DIR / "diagnostic.json").write_text(json.dumps(summary, indent=2))
    return summary
```

### §4.2 산출

- `analysis/plan-008/diagnostic.json` — 위 dict
- `analysis/plan-008/diagnostic.md`:
  - 1 줄 결론 (dominant cause + 가지치기 후보 수 + **main_bottleneck (ranking vs drift)**)
  - **Selector gap decomposition 표** (oracle_hit, argmax_hit, soft_hit, ranking_acc, gap_ranking, gap_drift) — top-1 ranking 12.6% 의 *정확한 의미* + 진짜 metric 박제
  - **Oracle miss residual breakdown (v2.5)**: par/perp/z 분산 비율 + corr_rotation/curvature/jerk — *oracle miss sample (~2800)* 위에서 측정 (regime 무관, plan main lever 의 직접 target)
  - **가지치기 후보 표 (v2.4 structural containment)**: 각 row = (i, j, rule, containment_soft, coord_dist, hit_rate_i, hit_rate_j, oracle_delta)
  - margin 분포 (drift 가설 supporting)
  - **per-regime oracle gap 표 (sanity only, v2.5 격하)**: 18 regime 별 oracle hit rate — decision 영향 X, 익숙한 grouping 으로 sanity check 만
  - **oracle miss regime 분포 (sanity only)**: oracle miss sample 의 regime 별 집중도 — main residual breakdown 결과와 정합 확인용

### §4.3 G0 합격 기준

- `mask_strategy == "oracle_miss_v2.5"` 박제 (regime mask 사용 X — Variant A 정합)
- `n_oracle_miss` 박제 (예상 ~2800, oracle 0.7188 → miss rate 0.2812)
- `dominant_causes` ≥ 1 entry — **oracle miss residual 위에서 도출** (없으면 `diagnostic_inconclusive` warn)
- **`prune_candidates` 리스트 박제 (v2.4 structural containment + v2.7 auto-relaxation)**: 각 entry 가 (i, j, rule, containment_soft, coord_dist, hit_rate_i, hit_rate_j, oracle_delta) 모두 포함. **v2.7**: 1차 (0.95, 5mm) 결과 < 3 시 자동 (0.90, 10mm) 재탐색 — `prune_threshold_tier` 박제 (caveat #17). 비어 있어도 통과 (relaxed 도 0 시 후보 풀 inherent diversity 신호).
- **Containment matrices 박제** (informational, sanity): `containment_soft` (K×K), `coord_dist_matrix` (K×K), `hit_rate` (K,) — pairwise 구조 audit 가능
- **`selector_gap_decomposition` 박제** — `main_bottleneck` ∈ {"ranking", "drift"} 결정. 결과에 따라 plan-008 의 main lever 우선순위 *데이터 기반* 재조정:
  - `main_bottleneck == "ranking"` (gap_ranking ≫ gap_drift): selector 강화 (Point 4 의 pairwise/distill fallback) 가 main, pruning 은 secondary
  - `main_bottleneck == "drift"` (gap_drift ≫ gap_ranking): pruning 이 main, selector 강화는 secondary (이전 v2.1 의 가설)
- `margin_top1_top2` 박제
- `per_regime_oracle_sanity` + `oracle_miss_regime_dist_sanity` 박제 (sanity only, decision 무관)

### §4.4 시간 예산

- ~30 초 (plan-005/007 데이터 재사용)

---

## §5. STAGE 2 — 가지치기 (Step 2a) + Greedy Set-Cover 재정의 (Step 2b, Strategy D) (c3, c4, c5)

### §5.1 Step 2a — 가지치기 (v2.4: Structural Containment 기반)

**v2.4 변경**: selector pick rate 기반 → **pairwise structural containment** 기반 (reviewer feedback). selector 거동 무관, 후보 풀의 *내재적 redundancy* 만 봄.

#### Identification criteria (§4 의 diagnostic.py 에서 도출)

후보 i 가 *redundant* 로 판정되는 조건 (둘 중 하나):

1. **Strict containment**: `hit_i ⊆ hit_j` (i 가 hit 인 모든 sample 에서 j 도 hit) + `hit_rate[j] > hit_rate[i]`
2. **Soft containment**: `containment_soft(i, j) ≥ 0.95` (i 의 hit 중 95%+가 j 의 hit) + `coord_dist(i, j) < 0.005m` (좌표 거의 동일) + `hit_rate[j] > hit_rate[i]`

#### Safety verification (oracle 손실 방지)

식별된 redundant 후보를 *일괄 제거* 후 oracle 손실 측정:

```python
def softmax_np(x: np.ndarray, temp: float = 0.03) -> np.ndarray:
    """Row-wise softmax over last axis (numpy). temp 작을수록 sharp."""
    z = x / temp
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def step2a_prune(
    prune_candidates: list,
    cands_27: np.ndarray,
    train_y: np.ndarray,
    oof_scores: np.ndarray,           # shape (N, 27) — §4.1 의 P001 ens_scores
) -> dict:
    """v2.4: structural containment 로 식별된 redundant 후보 일괄 제거 + oracle safety."""
    prune_idx = [p["idx"] for p in prune_candidates]   # diagnostic 의 structural list
    kept_mask = np.ones(27, dtype=bool)
    kept_mask[prune_idx] = False

    cands_pruned = cands_27[:, kept_mask, :]
    err_pruned = np.linalg.norm(cands_pruned - train_y[:, None, :], axis=2)
    oracle_pruned = float((err_pruned.min(axis=1) <= R_HIT).mean())
    oracle_orig = float((np.linalg.norm(cands_27 - train_y[:, None, :], axis=2).min(axis=1) <= R_HIT).mean())

    # Post-pruning sanity (Method 1, post-hoc effect — informational only)
    # oof_scores 의 pruned subset 으로 soft_hit 재측정
    scores_pruned = oof_scores[:, kept_mask]
    weights_pruned = softmax_np(scores_pruned, temp=0.03)
    soft_pred_pruned = (cands_pruned * weights_pruned[:, :, None]).sum(axis=1)
    soft_hit_pruned = float((np.linalg.norm(soft_pred_pruned - train_y, axis=1) <= R_HIT).mean())

    return {
        "pruning_method": "structural_containment_v2.4",
        "pruned_count": len(prune_idx),
        "remaining_count": int(kept_mask.sum()),
        "pruned_pairs": prune_candidates,   # (i, j) 쌍 + rule (strict/soft) + containment 박제
        "oracle_orig": oracle_orig,
        "oracle_pruned": oracle_pruned,
        "oracle_delta": oracle_pruned - oracle_orig,
        # v2.6 threshold 일치 — §3.2 G1 의 `oracle_after_prune ≥ 0.7170` (= 0.7188 − 0.0018) 와 정합.
        # 이전 v2.4: per-pair safety 는 0.001 사용 (§4.1) — pair-wise filter 라 더 엄격.
        # 본 aggregate safety 는 0.0018 (G1 spec) — 일괄 제거 후 누적 손실 허용 한계.
        "oracle_safe": (oracle_orig - oracle_pruned) < 0.0018,
        # Post-hoc sanity (informational, decision 무관)
        "soft_hit_pruned_posthoc": soft_hit_pruned,
        "kept_indices": [int(i) for i in np.where(kept_mask)[0]],
        "kept_names": [selector.CANDIDATES[i].name for i in np.where(kept_mask)[0]],
    }
```

→ 예상 결과: 27 → ~17~22 candidates (5~10 redundant 제거, containment 기반).

#### v2.4 의 의도

- **Identification**: 후보 풀의 *내재적* 구조 (selector 무관)
- **Verification**: oracle preservation (필수, severe `oracle_drop` trigger)
- **Sanity**: post-hoc effect (informational, decision 영향 X)

이 3 단계 분리가 reviewer 의 "selector pick rate 의존 → robustness 부족" 비판 해소.

### §5.2 Step 2b — `CandidateSpec` schema 확장 (Option A) + **Greedy Set-Cover (Strategy D)** Template Pool

**v2.3 변경**: 이전 hybrid C path (5 family pre-defined + family-level marginal filter) → **Strategy D (greedy set cover)** 로 교체. 모든 template 을 `template_pool` 에 두고 *iteration 마다 oracle 증가 최대* 인 1 개를 greedy add. set-cover 인식 + 후보 중복 add 회피.

#### §5.2.0 `src/pb_0_6822/selector.py` schema 확장 (Option A)

**3 곳 partial 수정** (arch / 학습 로직 *영역 외*):

**1. CandidateSpec dataclass 확장** ([selector.py:76-84](src/pb_0_6822/selector.py#L76-L84)):

```python
@dataclass(frozen=True)
class CandidateSpec:
    name: str
    d1: float = 0.0
    par: float = 0.0
    perp: float = 0.0
    d2: float = 0.0
    jerk: float = 0.0
    time_scale: float = 1.0
    # === v2.2 신규 fields (plan-008 family 확장) ===
    omega_scale: float = 0.0       # trig family — 각속도 배율
    arc_curvature: float = 0.0     # arc family — 호 곡률 배율
    z_scale: float = 1.0           # Frenet-Serret 3D — z 축 배율
    family_id: int = 0             # 0=base/frenet/acc/jerk/latency, 1=trig, 2=arc,
                                   # 3=frenet_serret_3d, 4=per_regime, 5=higher_order, 6=cross_term
```

기존 27 후보는 모두 `family_id=0` + 새 fields default → **backward-compat 완전 유지**.

**2. `candidate_spec_features` 함수 확장** ([selector.py:220-223](src/pb_0_6822/selector.py#L220-L223)):

```python
N_FAMILY = 7   # base + 6 new families

def candidate_spec_features(count: int, candidates_list=None) -> np.ndarray:
    """v2.2: family one-hot 추가. 기존 6 dim + family one-hot 7 dim = 13 dim per candidate.
       candidates_list: 외부에서 전달 가능 (extended candidates 사용 시).
                        None 이면 module-level CANDIDATES 사용 (기존 호환)."""
    cand_list = candidates_list if candidates_list is not None else CANDIDATES
    rows = []
    for spec in cand_list:
        scalar_feats = [
            spec.d1, spec.par, spec.perp, spec.d2, spec.jerk, spec.time_scale,
            spec.omega_scale, spec.arc_curvature, spec.z_scale,   # 신규 scalar
        ]
        family_onehot = [1.0 if i == spec.family_id else 0.0 for i in range(N_FAMILY)]
        rows.append(scalar_feats + family_onehot)
    spec = np.asarray(rows, dtype=np.float32)[None, :, :]
    return np.repeat(spec, count, axis=0)
```

→ Per-candidate feature dim: 6 → **9 (scalar) + 7 (one-hot) = 16**.

**3. `make_candidate_features` 의 interactions term 확장** ([selector.py:440](src/pb_0_6822/selector.py#L440)):

```python
def make_candidate_features(x, end_idx, candidates, horizon=2, direction=1.0, candidates_list=None):
    """v2.2: interactions term 이 omega_scale + arc_curvature 도 포함."""
    p0, d1, acc = motion_terms(x, end_idx)
    # ... (기존 par/perp/dist 계산 동일)
    spec = candidate_spec_features(len(x), candidates_list=candidates_list)
    # spec shape: (N, n_cand, 16) — 9 scalar + 7 one-hot

    ctx_base = ...   # 기존 9 dim
    ctx = ctx_base[:, None, :].repeat(candidates.shape[1], axis=1)

    # 기존 interactions: spec.par × ctx.acc_par + spec.perp × ctx.perp_norm
    # 신규 interactions: + spec.omega_scale × ctx.turn_cos
    #                  + spec.arc_curvature × ctx.curvature
    interactions = np.concatenate([
        spec[:, :, 1:3] * ctx[:, :, [3, 4]],      # 기존 (par × acc_par, perp × perp_norm)
        spec[:, :, 6:7] * ctx[:, :, 6:7],          # 신규 omega_scale × turn_cos
        spec[:, :, 7:8] * ctx[:, :, 7:8],          # 신규 arc_curvature × curvature
    ], axis=2)

    return np.concatenate([par/scale, perp/scale, dist/scale, spec, ctx, interactions], axis=2).astype(np.float32)
```

→ Per-candidate feature dim 최종: par/perp/dist (3) + spec (16) + ctx (9) + interactions (4) = **32 dim** (기존 ~25 → 32).

#### §5.2.1 `src/pb_0_6822/candidates_extended.py` — 6 Family 후보 정의

`CandidateSpec` 의 새 fields 활용:

**Family 1: Trig (rotation) — 4 후보**

```python
from src.pb_0_6822.selector import CandidateSpec

TRIG_CANDIDATES = [
    CandidateSpec("rot_low_080",  d1=2.0, omega_scale=0.8, family_id=1),
    CandidateSpec("rot_mid_100",  d1=2.0, omega_scale=1.0, family_id=1),
    CandidateSpec("rot_mid_120",  d1=2.0, omega_scale=1.2, family_id=1),
    CandidateSpec("rot_high_150", d1=2.0, omega_scale=1.5, family_id=1),
]

def make_rot_candidates(x, end_idx, horizon=2):
    """Batch coord func — 모든 TRIG_CANDIDATES 좌표 한 번에 산출. shape (N, 4, 3)."""
    p0, d1, _ = selector.motion_terms(x, end_idx)
    d2 = x[:, end_idx-1] - x[:, end_idx-2]
    omega_z = np.arctan2(
        d2[:,0]*d1[:,1] - d2[:,1]*d1[:,0],
        d2[:,0]*d1[:,0] + d2[:,1]*d1[:,1]
    )
    preds = []
    for spec in TRIG_CANDIDATES:
        preds.append(rot_coord_func(x, end_idx, horizon=horizon, spec=spec,
                                     _p0=p0, _d1=d1, _omega_z=omega_z))
    return np.concatenate(preds, axis=1).astype(np.float32)


def rot_coord_func(x, end_idx, *, horizon=2, spec, _p0=None, _d1=None, _omega_z=None):
    """Per-spec coord func — greedy set-cover 가 호출하는 unit. shape (N, 1, 3).

    §5.3 의 step2b_greedy_set_cover 가 template_pool 의 (name, spec, coord_func) tuple
    에서 coord_func(train_x, end_idx, horizon=2, spec=spec) 호출 → (N, 1, 3) 기대.
    """
    if _p0 is None:
        _p0, _d1, _ = selector.motion_terms(x, end_idx)
    if _omega_z is None:
        d2 = x[:, end_idx-1] - x[:, end_idx-2]
        _omega_z = np.arctan2(
            d2[:,0]*_d1[:,1] - d2[:,1]*_d1[:,0],
            d2[:,0]*_d1[:,0] + d2[:,1]*_d1[:,1]
        )
    theta = _omega_z * spec.omega_scale * horizon
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    d1_rot = np.stack([
        cos_t*_d1[:,0] - sin_t*_d1[:,1],
        sin_t*_d1[:,0] + cos_t*_d1[:,1],
        spec.z_scale * _d1[:,2]
    ], axis=1)
    pred = _p0 + d1_rot * horizon                    # shape (N, 3)
    return pred[:, None, :].astype(np.float32)        # shape (N, 1, 3)
```

**Family 2~6**: 동일 패턴 — 각 family 의 `CandidateSpec` 에 fields 채우고, batch `make_*_candidates(x, end_idx, horizon=2)` + per-spec `*_coord_func(x, end_idx, *, horizon, spec) → (N, 1, 3)` 한 쌍 정의. `*_coord_func` 시그너처는 §5.3 greedy set-cover 가 호출하는 unit.

| Family | family_id | 주 fields |
|---|---|---|
| trig | 1 | `omega_scale`, `z_scale` |
| arc | 2 | `arc_curvature`, `time_scale` (호 길이) |
| frenet_serret_3d | 3 | `z_scale` (= binormal_scale, §5.2.3 의미 재정의) |
| ~~per_regime_specialized~~ | ~~4~~ | **drop (v2.3 reviewer #1)** — regime backdoor 회피 |
| higher_order | 5 | `jerk`, `par`, `d2` (jerk + multi-step RK2) |
| cross_term | 6 | `par`, `omega_scale` (speed_slope × d1 등) |

**Family 2 — Arc** (`make_arc_candidates`, `arc_coord_func`):

```python
ARC_CANDIDATES = [
    CandidateSpec("arc_continue", arc_curvature=1.0, family_id=2),
    CandidateSpec("arc_decel",    arc_curvature=0.9, family_id=2),
    CandidateSpec("arc_accel",    arc_curvature=1.1, family_id=2),
]

def arc_coord_func(x, end_idx, *, horizon=2, spec):
    """3-point circular arc fit + arclength extrapolation. shape (N, 1, 3).
       arc_curvature 가 호 진행 속도 배율 (1.0=등각속도, 0.9=감속, 1.1=가속)."""
    p0, d1, _ = selector.motion_terms(x, end_idx)   # p0 (N,3), d1 (N,3)
    d2 = x[:, end_idx-1] - x[:, end_idx-2]           # (N, 3)
    # 평면 (xy) 호 진행각 estimate
    omega_z = np.arctan2(
        d2[:,0]*d1[:,1] - d2[:,1]*d1[:,0],
        d2[:,0]*d1[:,0] + d2[:,1]*d1[:,1]
    )                                                # (N,) — 호 진행각
    theta = omega_z * spec.arc_curvature * horizon   # arc 진행각 × 배율
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    arc_step = np.stack([
        cos_t * d1[:,0] - sin_t * d1[:,1],
        sin_t * d1[:,0] + cos_t * d1[:,1],
        d1[:,2],                                      # world z 그대로
    ], axis=1) * horizon
    pred = p0 + arc_step                              # (N, 3)
    return pred[:, None, :].astype(np.float32)


def make_arc_candidates(x, end_idx, horizon=2):
    return np.concatenate(
        [arc_coord_func(x, end_idx, horizon=horizon, spec=s) for s in ARC_CANDIDATES],
        axis=1,
    ).astype(np.float32)
```

**Family 3 — Frenet-Serret 3D** (`make_fs3d_candidates`, `fs3d_coord_func`, v2.3 binormal frame):

```python
FS3D_CANDIDATES = [
    CandidateSpec("fs_3d_planar",       z_scale=0.0, family_id=3),  # planar (tau=0)
    CandidateSpec("fs_3d_low_torsion",  z_scale=0.5, family_id=3),
    CandidateSpec("fs_3d_binormal",     z_scale=1.0, family_id=3),  # binormal (B=T×N)
]

def fs3d_coord_func(x, end_idx, *, horizon=2, spec):
    """Frenet-Serret 3D — T/N/B local frame 의 binormal 진폭. shape (N, 1, 3).

    T = unit(d1)
    N = unit(d2 − (d2·T)T)           # d2 의 T-수직 성분
    B = T × N                          # binormal
    kappa_estimate = ||d2 − (d2·T)T|| / max(||d1||², ε)
    pred = p0 + d1·horizon + spec.z_scale (= binormal_scale) · ||d1|| · horizon · kappa · B
    """
    p0, d1, _ = selector.motion_terms(x, end_idx)
    d2 = x[:, end_idx-1] - x[:, end_idx-2]
    eps = 1e-9
    d1_norm = np.linalg.norm(d1, axis=1, keepdims=True) + eps    # (N, 1)
    T = d1 / d1_norm
    d2_par = (np.einsum("ij,ij->i", d2, T))[:, None] * T          # (N, 3)
    d2_perp = d2 - d2_par                                          # (N, 3)
    N_norm = np.linalg.norm(d2_perp, axis=1, keepdims=True) + eps
    N = d2_perp / N_norm
    B = np.cross(T, N)                                              # binormal (N, 3)
    kappa = (N_norm[:, 0] / (d1_norm[:, 0] ** 2 + eps))[:, None]    # (N, 1)
    binormal_term = spec.z_scale * d1_norm * horizon * kappa * B    # (N, 3)
    pred = p0 + d1 * horizon + binormal_term
    return pred[:, None, :].astype(np.float32)


def make_fs3d_candidates(x, end_idx, horizon=2):
    return np.concatenate(
        [fs3d_coord_func(x, end_idx, horizon=horizon, spec=s) for s in FS3D_CANDIDATES],
        axis=1,
    ).astype(np.float32)
```

**Family 5 — Higher-order** (`make_higher_order_candidates`, `higher_order_coord_func`, snap drop):

```python
HIGHER_ORDER_CANDIDATES = [
    CandidateSpec("jerk_acc_par_120", par=1.2, jerk=0.1, family_id=5),
    CandidateSpec("multi_step_rk2",   d1=2.0,  d2=1.0,   family_id=5),
]

def higher_order_coord_func(x, end_idx, *, horizon=2, spec):
    """Per-spec dispatch:
      - spec.jerk > 0  → jerk-augmented: pred = p0 + par·d1·h + 0.5·d2·h² + (jerk/6)·jerk_vec·h³
      - 그 외 (d1>0 ∧ d2>0) → 11-step RK2: dt = horizon/11, x_{k+1} = x_k + d1·dt + 0.5·d2·dt²

    v2.6 fix: 이전 변수명 `d3` 는 2차 차분 (acceleration) 식이었음 — 진짜 jerk = 3차 차분.
              `jerk_vec` = x[end-1] − 3·x[end-2] + 3·x[end-3] − x[end-4]  (3rd-order finite difference).
    """
    p0, d1, acc = selector.motion_terms(x, end_idx)
    if spec.jerk > 0:
        d2 = x[:, end_idx-1] - x[:, end_idx-2]
        # 진짜 jerk = 3차 차분 (boundary: end_idx ≥ 4 필요)
        jerk_vec = (x[:, end_idx-1] - 3*x[:, end_idx-2]
                    + 3*x[:, end_idx-3] - x[:, end_idx-4])
        pred = p0 + spec.par * d1 * horizon + 0.5 * d2 * horizon**2 \
               + (spec.jerk / 6.0) * jerk_vec * horizon**3
    else:
        d2 = x[:, end_idx-1] - x[:, end_idx-2]
        n_steps = 11
        dt = horizon / n_steps
        pred = p0.copy()
        d1_step = spec.d1 * d1 / max(horizon, 1e-9)   # d1 per dt unit
        d2_step = spec.d2 * d2 / max(horizon**2, 1e-9)
        for _ in range(n_steps):
            pred = pred + d1_step * dt + 0.5 * d2_step * dt**2
    return pred[:, None, :].astype(np.float32)


def make_higher_order_candidates(x, end_idx, horizon=2):
    return np.concatenate(
        [higher_order_coord_func(x, end_idx, horizon=horizon, spec=s) for s in HIGHER_ORDER_CANDIDATES],
        axis=1,
    ).astype(np.float32)
```

**Family 6 — Cross-term** (`make_cross_term_candidates`, `cross_term_coord_func`):

```python
CROSS_TERM_CANDIDATES = [
    CandidateSpec("speed_slope_d1_120",  par=1.2, omega_scale=0.5, family_id=6),
    CandidateSpec("speed_norm_acc_par",  par=1.0, jerk=0.3,        family_id=6),
    CandidateSpec("omega_speed",         par=1.0, omega_scale=1.0, family_id=6),
]

def cross_term_coord_func(x, end_idx, *, horizon=2, spec):
    """Feature × motion cross-term — per-sample 적응.
       speed_slope ≈ (||d1|| − ||d1_prev||) / ||d1_prev|| (스칼라 per sample)
       cross_term = spec.par · speed_slope · d1 + spec.omega_scale · turn_cos · d1
       pred = p0 + d1·horizon + cross_term · horizon
    """
    p0, d1, _ = selector.motion_terms(x, end_idx)
    eps = 1e-9
    d1_prev = x[:, end_idx-1] - x[:, end_idx-2]    # (N, 3)
    d1_norm = np.linalg.norm(d1, axis=1) + eps
    d1_prev_norm = np.linalg.norm(d1_prev, axis=1) + eps
    speed_slope = (d1_norm - d1_prev_norm) / d1_prev_norm    # (N,)
    # turn_cos = unit(d1) · unit(d1_prev)
    cos_turn = np.einsum("ij,ij->i", d1, d1_prev) / (d1_norm * d1_prev_norm)
    cross_term = spec.par * speed_slope[:, None] * d1 \
                 + spec.omega_scale * cos_turn[:, None] * d1
    pred = p0 + d1 * horizon + cross_term * horizon
    return pred[:, None, :].astype(np.float32)


def make_cross_term_candidates(x, end_idx, horizon=2):
    return np.concatenate(
        [cross_term_coord_func(x, end_idx, horizon=horizon, spec=s) for s in CROSS_TERM_CANDIDATES],
        axis=1,
    ).astype(np.float32)
```

#### §5.2.2 통합 — Extended candidates list

```python
# candidates_extended.py
from src.pb_0_6822 import selector as _sel

# kept_families 의미 contract (v2.6 명시):
#   - Strategy D (Greedy) 가 *individual template* 단위 add 이지만, 본 통합 함수는
#     family-level on/off filter 인터페이스를 사용 (구현 단순성 + base whitelist 정합).
#   - greedy 결과 selected_templates 의 family set 으로 derive:
#       kept_families = {spec.family_id_name for (_, spec, _) in iteration_log if added}
#     (family_id_name = "trig"/"arc"/"frenet_serret_3d"/"higher_order"/"cross_term")
#   - 한 family 의 *일부 template* 만 greedy 가 select 했어도 family 통째 include —
#     selector 학습 입장에서 family one-hot 신호 분리 위한 *과대 포함* OK.
#     단 *전체 family 가 한 번도 select 안 됨* 시 그 family drop.
#   - 진짜 template-level granularity 필요 시 plan-009 후보 (`get_extended_candidates_list_by_template`).

def get_extended_candidates_list(kept_indices, kept_families):
    """기존 pruned 27 + 5 family kept ones → CandidateSpec list."""
    base_kept = [_sel.CANDIDATES[i] for i in kept_indices]  # base family_id=0

    new_specs = []
    if "trig" in kept_families: new_specs.extend(TRIG_CANDIDATES)
    if "arc" in kept_families: new_specs.extend(ARC_CANDIDATES)
    if "frenet_serret_3d" in kept_families: new_specs.extend(FS3D_CANDIDATES)
    # v2.3: Family 4 (per_regime_specialized) drop — regime backdoor 회피. 분기 제거.
    if "higher_order" in kept_families: new_specs.extend(HIGHER_ORDER_CANDIDATES)
    if "cross_term" in kept_families: new_specs.extend(CROSS_TERM_CANDIDATES)

    return base_kept + new_specs   # list[CandidateSpec]

def make_candidates_extended(x, end_idx, horizon=2, kept_indices=None, kept_families=None):
    """기존 27 (pruned) + 새 family 후보 좌표 + 통합 CandidateSpec list."""
    cands_base_27 = _sel.make_candidates(x, end_idx, horizon)
    cands_base_kept = cands_base_27[:, kept_indices, :]

    new_cands_list = [cands_base_kept]
    if "trig" in kept_families: new_cands_list.append(make_rot_candidates(x, end_idx, horizon))
    if "arc" in kept_families: new_cands_list.append(make_arc_candidates(x, end_idx, horizon))
    # ... 4 more families

    return np.concatenate(new_cands_list, axis=1).astype(np.float32)
```

**총 후보 수**: 27 (pruned base) + 4 (trig) + 3 (arc) + 3 (fs3d) + 2 (higher_order, snap drop) + 3 (cross_term) = **27 + 15 = 42 후보** (모든 family kept 시; Family 4 per_regime drop v2.3 반영). family filter 후 ~32~37.

#### §5.2.3 Template Pool 구성 — v2.3 (Family 4 drop, snap drop, fs_3d_binormal)

**Reviewer 피드백 반영 후 template_pool** (모든 family 통합, 13 templates):

**Trig family (4 templates, family_id=1)**:
- `rot_low_080`: `CandidateSpec(d1=2.0, omega_scale=0.8, family_id=1)`
- `rot_mid_100`: `(d1=2.0, omega_scale=1.0, family_id=1)`
- `rot_mid_120`: `(d1=2.0, omega_scale=1.2, family_id=1)`
- `rot_high_150`: `(d1=2.0, omega_scale=1.5, family_id=1)`
- 좌표: `R(omega_z · omega_scale · horizon) · d1 + p0`

**Arc family (3 templates, family_id=2)**:
- `arc_continue`: `(arc_curvature=1.0, family_id=2)` (등각속도 외삽)
- `arc_decel`: `(arc_curvature=0.9, family_id=2)` (감속 90%)
- `arc_accel`: `(arc_curvature=1.1, family_id=2)` (가속 110%)
- 좌표: 3-point arc fit + arclength extrapolation (`||d1|| × horizon`)

**Frenet-Serret 3D family (3 templates, family_id=3, v2.3 변경)**:
- `fs_3d_planar`: `(binormal_scale=0.0, family_id=3)` (평면 운동 가정, tau=0)
- `fs_3d_low_torsion`: `(binormal_scale=0.5, family_id=3)`
- `fs_3d_binormal`: `(binormal_scale=1.0, family_id=3)` (**reviewer #6: world z 의존 X**)
- 좌표: 3D Frenet local frame (T = tangent, N = normal, B = T × N) → B 방향 진폭 `binormal_scale × ||d1|| × horizon × kappa_estimate`
- **CandidateSpec 의 `z_scale` field → `binormal_scale` 로 *의미 재정의*** (이름 v2.2 그대로 호환, 사용 의미만 binormal-frame 으로 변경). docstring 갱신:
  ```python
  z_scale: float = 1.0   # v2.3: Frenet binormal frame magnitude (NOT world z).
                         # Frenet-Serret family 만 사용.
  ```

**Higher-order family (2 templates, family_id=5, v2.3 변경 — snap drop)**:
- `jerk_acc_par_120`: `(par=1.2, jerk=0.1, family_id=5)`
- `multi_step_rk2`: `(d1=2.0, d2=1.0, family_id=5)` (11 step RK2 integration)
- ~~`snap_term`~~: **drop** (4th derivative noise 4중 증폭, reviewer #5)

**Cross-term family (3 templates, family_id=6)**:
- `speed_slope_d1_120`: `(par=1.2, omega_scale=0.5, family_id=6)` (plan-007 Step 3 의 핵심 cross-term)
- `speed_norm_acc_par`: `(par=1.0, jerk=0.3, family_id=6)`
- `omega_speed`: `(omega_scale=1.0, par=1.0, family_id=6)`
- 좌표: feature × motion cross-term (per-sample 적응 도입)

~~**Per-regime specialized family (family_id=4)**~~: **v2.3 drop**. reviewer #1 — regime ID hard-code = Variant A baseline self-contradiction. 같은 dynamics 는 trig + arc family 가 cover (regime 16 sharp turn = `rot_high_150`, regime 17 slow turn = `rot_low_080`, regime 10 decel arc = `arc_decel`).

**총 template_pool 크기: 4 + 3 + 3 + 2 + 3 = 15 templates** (이전 v2.2 의 19 → 15, Family 4 의 -3 + snap -1).

#### §5.2.4 `build_template_pool()` — greedy set-cover 진입점

`step2b_greedy_set_cover` 가 받는 `list[(name, spec, coord_func)]` tuple list 를 산출하는 build 함수 (v2.6 신설 — 인터페이스 갭 해소).

```python
# candidates_extended.py
def build_template_pool() -> list:
    """§5.2.3 의 15 templates 를 (name, spec, coord_func) tuple list 로 정렬.
       §5.3 step2b_greedy_set_cover 의 template_pool 인자로 전달.

       각 family 의 *per-spec* coord_func (return shape (N, 1, 3)) 를 binding —
       batch make_*_candidates (return (N, K, 3)) 가 아닌 per-spec wrapper.
    """
    return (
        [(s.name, s, rot_coord_func)         for s in TRIG_CANDIDATES]          # 4
        + [(s.name, s, arc_coord_func)        for s in ARC_CANDIDATES]           # 3
        + [(s.name, s, fs3d_coord_func)       for s in FS3D_CANDIDATES]          # 3
        + [(s.name, s, higher_order_coord_func) for s in HIGHER_ORDER_CANDIDATES]  # 2
        + [(s.name, s, cross_term_coord_func) for s in CROSS_TERM_CANDIDATES]    # 3
    )
```

**사용** (`analysis/plan-008/prune_and_redefine.py`):

```python
from src.pb_0_6822 import candidates_extended as cx
template_pool = cx.build_template_pool()         # list[(name, spec, coord_func)], len = 15
result = step2b_greedy_set_cover(
    cands_pruned, train_y, train_x, end_idx,
    template_pool=template_pool,
    kept_pruned_specs=KEPT_PRUNED_SPECS,
    regimes=regimes,
)
```

### §5.3 Greedy Set-Cover Algorithm (Strategy D, v2.3 core)

**Algorithm**: pruned pool 에서 시작 → template_pool 의 모든 후보 중 *oracle 증가가 가장 큰* 1 개를 add → 반복. 종료: delta < threshold OR pool size limit OR oracle target.

```python
def step2b_greedy_set_cover(
    cands_pruned: np.ndarray,           # shape (N, M_pruned, 3) — Step 2a 결과
    train_y: np.ndarray,                 # shape (N, 3)
    train_x: np.ndarray,                 # shape (N, T, 3) — template 좌표 계산용
    end_idx: int,
    template_pool: list,                 # list[(name, spec, coord_func)] — §5.2.3 의 13~15 templates
    kept_pruned_specs: list,             # CandidateSpec list — Step 2a 결과의 kept base specs (M_pruned 개)
    regimes: np.ndarray,                 # shape (N,) — informational sanity only (per-regime breakdown 용)
    *,
    target_oracle: float = 0.90,
    max_pool_size: int = 50,
    min_delta: float = 0.001,
) -> dict:
    """Greedy set-cover: 매 iteration oracle 최대 증가 후보 add.

    Oracle 측정 set (v2.6 명시): **full train (in-sample, all N samples)**.
      - §1.1 의 "Oracle (best of 27, raw) = 0.7188" 정의 (plan-005) 와 동일 set.
      - OOF/CV split 사용 X — pool selection 자체는 train-set oracle 최대화가 목표.
      - 일반화 verification 은 Step 3 의 5-fold OOF (G2) 에서 별도 수행.
      - 만약 OOF-based pool selection 이 필요하면 plan-009 후보로 박제 (v2.6 caveat #21).

    종료 조건 (any):
      1. compute_oracle(pool) >= target_oracle  → 0.90 stretch goal 달성
      2. len(pool) >= max_pool_size              → pool 폭주 방지
      3. best_delta < min_delta                  → 의미 있는 회수 X
    """
    pool_cands = cands_pruned              # 좌표 array, shape (N, M, 3)
    pool_specs = list(kept_pruned_specs)   # CandidateSpec list
    remaining_templates = list(template_pool)

    err = np.linalg.norm(pool_cands - train_y[:, None, :], axis=2)
    oracle_current = float((err.min(axis=1) <= R_HIT).mean())

    iteration_log = [{
        "iter": 0, "added": None, "oracle": oracle_current,
        "delta": 0.0, "pool_size": pool_cands.shape[1],
    }]

    while oracle_current < target_oracle and pool_cands.shape[1] < max_pool_size:
        best_template, best_delta, best_new_cands = None, 0.0, None

        for tmpl in remaining_templates:
            name, spec, coord_func = tmpl
            new_cands = coord_func(train_x, end_idx, horizon=2, spec=spec)   # shape (N, 1, 3) — 1 후보
            cands_test = np.concatenate([pool_cands, new_cands], axis=1)
            err_test = np.linalg.norm(cands_test - train_y[:, None, :], axis=2)
            oracle_test = float((err_test.min(axis=1) <= R_HIT).mean())
            delta = oracle_test - oracle_current

            if delta > best_delta:
                best_template, best_delta, best_new_cands = tmpl, delta, new_cands

        if best_delta < min_delta:
            break    # 더 추가해도 의미 없음

        # Commit best template
        pool_cands = np.concatenate([pool_cands, best_new_cands], axis=1)
        pool_specs.append(best_template[1])
        oracle_current = oracle_current + best_delta
        remaining_templates.remove(best_template)

        iteration_log.append({
            "iter": len(iteration_log),
            "added_template": best_template[0],   # name
            "family_id": best_template[1].family_id,
            "oracle": oracle_current,
            "delta": best_delta,
            "pool_size": pool_cands.shape[1],
        })

    # Per-regime breakdown (final pool)
    per_regime_final = {}
    for r in [10, 16, 17]:
        mask = regimes == r
        if mask.sum() == 0: continue
        err_final = np.linalg.norm(pool_cands - train_y[:, None, :], axis=2)
        per_regime_final[int(r)] = {
            "n": int(mask.sum()),
            "oracle_after_greedy": float((err_final[mask].min(axis=1) <= R_HIT).mean()),
        }

    return {
        "iteration_log": iteration_log,
        "oracle_final": oracle_current,
        "total_candidates_final": pool_cands.shape[1],
        "pool_specs_final": [
            {"name": s.name, "family_id": s.family_id} for s in pool_specs
        ],
        "templates_added_count": len(iteration_log) - 1,
        "templates_remaining_count": len(remaining_templates),
        "stop_reason": (
            "target_oracle_reached" if oracle_current >= target_oracle
            else "max_pool_size_reached" if pool_cands.shape[1] >= max_pool_size
            else "delta_below_threshold"
        ),
        "per_regime_worst": per_regime_final,
    }
```

**예상 iteration 패턴** (가상):

```
iter 0: pool = pruned_22, oracle = 0.7170 (baseline)
iter 1: + rot_high_150 (regime 16/17 sharp turn cover) → oracle 0.745, delta +0.028
iter 2: + arc_decel (regime 10 decel arc) → oracle 0.770, delta +0.025
iter 3: + speed_slope_d1_120 (per-sample 적응) → oracle 0.792, delta +0.022
iter 4: + rot_low_080 → oracle 0.811, delta +0.019
iter 5: + fs_3d_binormal → oracle 0.829, delta +0.018
iter 6: + arc_continue → oracle 0.842, delta +0.013
iter 7: + multi_step_rk2 → oracle 0.851, delta +0.009 ✓ target 도달, stop
```

→ pool size = 22 + 7 = **29 candidates**, oracle final ≈ 0.85.

**장점 (vs hybrid C path)**:
- *Set cover 인식*: 비슷한 후보 (예: rot_mid_100 + rot_mid_120) 가 동시 add 되면 marginal 이 zero 라 자동 skip
- *Data-driven*: family 전체 add 대신 *individual template* 단위 add → 더 정밀
- *Stop condition*: oracle 목표 도달 시 즉시 종료, pool 폭주 방지

### §5.4 산출

- `analysis/plan-008/prune_summary.json` (Step 2a)
- `analysis/plan-008/greedy_set_cover.json` (Step 2b — iteration log + final pool + stop reason)
- `analysis/plan-008/redefine.md` — iteration trace table + final pool 구성 + per-regime worst 회복
- `src/pb_0_6822/candidates_extended.py` — template_pool 정의 모듈 (5 family, 15 templates)

### §5.5 G1 합격 기준 (자동 판정)

- Step 2a: `oracle_pruned ≥ 0.7170`
- Step 2b (greedy set cover):
  - `iteration_log` 박제 (각 iter 의 added template + oracle + delta)
  - `oracle_final ≥ 0.85` (minimum) — 초과 시 통과
  - `0.78 ≤ oracle_final < 0.85` → warn-only flag (`redefinition_partial`)
  - `oracle_final < 0.78` → severe (`redefinition_severely_insufficient`)
  - `stop_reason` ∈ {"target_oracle_reached", "max_pool_size_reached", "delta_below_threshold"}
- Per-regime worst (16/17/10) 의 `oracle_after_greedy ≥ 0.55` — **warn-only sanity** (regime infra 폐기 §2.2, decision/severe 무관)

### §5.6 시간 예산

- 5 family template 구현: ~2 시간 (Family 4 drop + snap drop 으로 v2.2 대비 -1 시간)
- Greedy set cover iteration: ~30 분 (15 templates × ~10 iter × oracle 측정)

---

## §6. STAGE 3 — Selector 재학습 (Variant A path) (c6, c7, c8)

### §6.0 Sanity baseline — Family 효과 분리 (v2.6 신규)

**의도**: Step 3 의 G2 OOF 임계 (0.70) 가 *family 추가 효과* 인지 *hyperparam 변경 효과* 인지 분리. 메인 검증 [CRITICAL] 3 의 cheap fix.

**측정**:

```python
# Step 3 진입 전, extended pool 학습 *전에* 동일 hyperparam 으로 27 후보 baseline 측정
# analysis/plan-008/sanity_baseline_27.py
import copy
# v2.6 fix: monkey-patch 직전에 module-level CANDIDATES / make_candidates 의 백업을 저장.
#          본 sanity baseline 자체는 baseline 그대로 학습이라 patch 가 강제는 아니지만,
#          ORIGINAL_* 변수는 §6.1 의 EXTENDED patch 직전 backup 으로도 재사용되는 의도.
ORIGINAL_27_CANDIDATES = copy.deepcopy(selector.CANDIDATES)
ORIGINAL_make_candidates = selector.make_candidates
selector.CANDIDATES = ORIGINAL_27_CANDIDATES   # plan-004 의 원본 27 (sanity baseline 진입 시 무변)
selector.make_candidates = ORIGINAL_make_candidates

selector.SELECTOR_MAIN([
    '--root', str(DATA_ROOT),
    '--out-dir', str(SANITY_RUN_DIR),
    '--models', 'attn_gru',
    '--folds', '5', '--fold-limit', '5',
    '--regime-prior-strength', '0',     # Variant A path 동일
    '--pre-epochs', '10', '--fine-epochs', '8', '--freeze-fine-epochs', '3',
    '--epoch-plus', '5', '--patience', '4',
    '--hidden', '48', '--batch', '4096',   # 동일 hyperparam
    '--device', 'cuda:1',
])
# → sanity_baseline_27_oof 측정
```

**기대값**: Variant A baseline 0.6570 (plan-005 STAGE 6) ± 0.005 재현. 재현 시 hyperparam 효과 marginal 확정.

**Step 3 G2 OOF 의 family 효과**:
- `family_effect = oof_extended_pool − sanity_baseline_27`
- 의도: family 효과 ≥ +0.03 (extended pool 의 *진짜* 회수율)
- family 효과 < +0.02 이면 → oracle 0.85 의 *회수율 낮음* 신호, plan-009 selector arch 교체 후보 강화

**산출**:
- `analysis/plan-008/sanity_baseline_27.json`: `{sanity_baseline_27_oof, hyperparam_set, n_folds}`
- `runs/baseline/G001_sanity-27/oof_selector_scores.npz` (재사용 안하지만 박제)

**시간 예산**: ~30 분 (5-fold OOF, plan-004 의 fold timing 답습).

### §6.1 학습 방법

```python
# analysis/plan-008/selector_retrain.py
import json
import numpy as np
from pathlib import Path
from src.pb_0_6822 import selector
from src.pb_0_6822 import candidates_extended

ANALYSIS_DIR = Path(__file__).resolve().parents[1] / "analysis/plan-008"

# v2.6 binding source 명시 — §5 산출 JSON 의 정확한 key:
#   - prune_summary.json   : key `"kept_indices"` (§5.1 step2a_prune 의 return dict)
#   - greedy_set_cover.json: key `"pool_specs_final"` → entry list → 각 `spec.family_id` 추출
KEPT_INDICES = json.loads((ANALYSIS_DIR / "prune_summary.json").read_text())["kept_indices"]
_FAMILY_ID_NAME = {1: "trig", 2: "arc", 3: "frenet_serret_3d", 5: "higher_order", 6: "cross_term"}
_greedy = json.loads((ANALYSIS_DIR / "greedy_set_cover.json").read_text())
KEPT_FAMILIES = sorted({
    _FAMILY_ID_NAME[s["family_id"]]
    for s in _greedy["pool_specs_final"]
    if s["family_id"] in _FAMILY_ID_NAME
})

# v2.2: schema 확장된 CandidateSpec list 사전 구성
EXTENDED_CANDIDATES = candidates_extended.get_extended_candidates_list(KEPT_INDICES, KEPT_FAMILIES)
# EXTENDED_CANDIDATES: list[CandidateSpec], length = ~35~40

# Monkey-patch CANDIDATES module-level + make_candidates 좌표 생성
selector.CANDIDATES = EXTENDED_CANDIDATES   # spec 일치 (candidate_spec_features 가 자동 활용)
selector.make_candidates = lambda x, end_idx, horizon=2: candidates_extended.make_candidates_extended(
    x, end_idx, horizon, kept_indices=KEPT_INDICES, kept_families=KEPT_FAMILIES
)
# → candidate_spec_features(count) 는 새 CANDIDATES 의 16-dim spec 자동 생성

# plan-004 run_full.py 호출, Variant A path 강제
from src.pb_0_6822 import run_full
# regime_prior_strength=0 강제: SELECTOR_MAIN args 에 명시
selector.SELECTOR_MAIN([
    '--root', str(DATA_ROOT),
    '--out-dir', str(RUN_DIR),
    '--models', 'attn_gru',
    '--folds', '5', '--fold-limit', '5',
    '--regime-prior-strength', '0',     # ⭐ Variant A 핵심
    '--pre-epochs', '10', '--fine-epochs', '8', '--freeze-fine-epochs', '3',
    '--epoch-plus', '5', '--patience', '4',
    '--hidden', '48', '--batch', '4096',
    '--device', 'cuda:1',
])

# corrector 는 기존 (plan-004) 그대로 적용 (Step 3 단계, Step 4 에서 재설계)
run_full.run_boundary()
```

### §6.1.5 CandidateSpec schema 확장 검증 (v2.2 Option A)

학습 진입 전 assert:

```python
import inspect
from src.pb_0_6822 import selector

# 1. 새 fields 존재 검증
sample_spec = selector.CandidateSpec("dummy")
assert hasattr(sample_spec, "omega_scale"), "schema v2.2 미적용"
assert hasattr(sample_spec, "arc_curvature"), "schema v2.2 미적용"
assert hasattr(sample_spec, "z_scale"), "schema v2.2 미적용"
assert hasattr(sample_spec, "family_id"), "schema v2.2 미적용"

# 2. 기존 27 후보의 backward-compat 검증 (default 적용)
# v2.6 fix: §6.1 monkey-patch 이후 selector.CANDIDATES 는 base_kept (family_id=0) + new specs (family_id ∈ 1..6)
#          가 섞임. 따라서 *family_id == 0* filter 로 base family 만 검증 (slice [:27] 사용 X).
base_specs = [c for c in selector.CANDIDATES if c.family_id == 0]
assert len(base_specs) >= 1, "base family (family_id=0) 후보가 monkey-patch 이후에도 한 개 이상 남아 있어야 함"
for c in base_specs:
    assert c.omega_scale == 0.0, f"base candidate {c.name} 의 omega_scale 변경됨"
    assert c.arc_curvature == 0.0, f"base candidate {c.name} 의 arc_curvature 변경됨"

# 3. cand_feat 차원 확인
test_cands = selector.make_candidates(train_x, 10, horizon=2)
test_feat = selector.make_candidate_features(train_x, 10, test_cands, horizon=2)
print(f"cand_feat shape: {test_feat.shape}")   # (N, ~35~40, ~32)
expected_dim = 3 + 16 + 9 + 4   # par/perp/dist + spec_v22 + ctx + interactions_v22 = 32
assert test_feat.shape[2] == expected_dim, f"cand_dim mismatch: {test_feat.shape[2]} != {expected_dim}"
```

위반 시 `schema_v22_residue` severe (selector.py 의 Option A 수정 누락).

### §6.2 Variant A path 검증 (severe `regime_residue` 방지) — **v2.6 강화**

학습 전 + 후 assert (메인 검증 [SUGGEST] 5 의 cheap fix):

```python
# 1. 학습 전: regime 코드 호출 가능성 검증
import inspect
src = inspect.getsource(selector.SELECTOR_MAIN)
# regime_prior_strength=0 이 학습 안에서 enforce 됨 (selector.py 가 regime bias 를 0 으로 곱함)
# v2.6 binding: cli_args 는 §6.1 의 `selector.SELECTOR_MAIN([...])` 호출 list 와 동일 객체.
# 본 assert 진입 *전에* 변수에 저장 후 호출하는 패턴 사용:
#     cli_args = ['--root', str(DATA_ROOT), ..., '--regime-prior-strength', '0', ...]
#     selector.SELECTOR_MAIN(cli_args)
assert '--regime-prior-strength' in cli_args and cli_args[cli_args.index('--regime-prior-strength') + 1] == '0'

# 2. 학습 후 (v2.6 신규): 산출 npz 의 regime_bias_table 검증
import numpy as np
z = np.load(RUN_DIR / "oof_selector_scores.npz")
if "regime_bias_table" in z.files:
    rbt = z["regime_bias_table"]    # shape (18, n_cand) or (n_cand,)
    rbt_var = float(np.var(rbt))
    # Variant A path 강제: regime_bias_table 모두 0 (또는 분산 0 = 모든 cell 동일 상수)
    assert rbt_var < 1e-10, f"regime_residue: regime_bias_table 분산 {rbt_var} > 1e-10 (Variant A 위배)"
else:
    # 또는 키 자체 부재 (regime bias 가 학습에 등장 안 함)
    pass

# 3. (v2.6 신규) ens_scores 의 per-regime stratification 검증
# 만약 regime 영향 *zero* 면 같은 candidate 의 ens_scores 가 regime 별로 동일 분포여야 함
oof_scores = z["ens_scores"]    # (N, n_cand)
regimes_oof = z.get("regimes")   # (N,) — Variant A path 에서도 informational
if regimes_oof is not None:
    # informational only: 만약 regime 별 mean score 차이가 *큰* candidate 있으면 의심
    per_regime_mean_diff = []
    for c in range(oof_scores.shape[1]):
        per_regime_scores = [oof_scores[regimes_oof == r, c].mean() for r in range(18) if (regimes_oof == r).any()]
        per_regime_mean_diff.append(max(per_regime_scores) - min(per_regime_scores))
    median_diff = float(np.median(per_regime_mean_diff))
    # informational: median diff > 1.0 이면 regime 영향 의심 (단 candidate features 자체가 regime-correlated 일 수 있음 — warn only)
    if median_diff > 1.0:
        print(f"WARN: per-regime ens_scores diff median = {median_diff:.3f} > 1.0 — Variant A path 잔재 의심")
```

위반 (1 또는 2 의 assert 실패) 시 즉시 escalate (`regime_residue` severe). 3 은 informational warn-only.

### §6.3 산출

- `runs/baseline/G001_candidate-redefine/oof_selector_scores.npz` — shape (N_train, N_final)
- `runs/baseline/G001_candidate-redefine/test_selector_scores.npz` — shape (N_test, N_final)
- `runs/baseline/G001_candidate-redefine/submission_step3.csv` — selector + 기존 corrector + soft averaging
- `analysis/plan-008/selector_retrain.json` — OOF metrics (overall + per-regime + top-K)

### §6.4 G2 합격 기준 (자동 판정)

- OOF hit (soft) **≥ 0.70** (v2.4 완화, 이전 0.71)
- **Sanity baseline check (v2.6 신규)**:
  - `sanity_baseline_27_oof ∈ [0.652, 0.662]` (Variant A baseline 0.6570 ± 0.005 재현 — hyperparam 효과 marginal 확정)
  - 재현 실패 시 (out of band) → `sanity_baseline_drift` warn (hyperparam 변경 효과가 family 효과와 *혼재* 가능, 결과 해석 주의)
  - **`family_effect = oof_extended_pool − sanity_baseline_27_oof ≥ +0.03`** — family 의 *순* 회수 효과 최소 임계
  - `family_effect < +0.02` → `family_effect_marginal` warn-only (oracle 0.85 *회수율 낮음* 신호, plan-009 selector arch 교체 후보 강화 trigger)
- **Ranking gap** (= oracle − selector argmax hit) **≤ 7pp** (plan-005 추정값 유지 minimum, *추가 악화 방지*)
  - top-1 ranking 정확도 (oracle best 정확 픽 비율) 는 *informational* only (12.6% → 13%+ 이면 보너스, 단 main metric 아님)
  - 진짜 metric = argmax hit (= 1등 픽 *1cm 안* 비율). oracle - argmax 가 작아야 selector 가 풀 안의 hit 후보를 잘 찾는다는 신호.
- `submission_step3.csv` schema OK
- `regime_residue` severe 미발동
- **LB 회수 X (carry-over)** — submission.csv 박제만
- **변수 단일성 (v2.6 명시)**: Step 3 의 G2 OOF 는 *기존 corrector (plan-004 lock-in)* 위에서 측정. Step 4 (§7) 는 *새 corrector (band-specific)* 위에서 측정 → Step 3 G2 vs Step 4 G3 비교 시 (selector 동일, corrector 만 변경) 한 변수 분리. Step 3 안에서 corrector 도 변경 X (§6.1 의 `run_full.run_boundary()` 는 plan-004 기존 corrector full-fit).

### §6.5 Fallback (G2 미달 시) — **v2.3 강화 (reviewer #4)**

후보 +50% (27→~30+) 에 대한 *학습 보조 장치* 4 종 순차 시도:

1. **hidden 48 → 64**: selector capacity 증가
   ```python
   selector.SELECTOR_MAIN([..., '--hidden', '64', ...])
   ```
2. **pairwise_loss_weight 강화**: 1.0 → 1.5 → 2.0 (default + 50%/100%). pairwise margin loss 가 비슷한 후보 사이 ranking 신호 강화.
   ```python
   selector.SELECTOR_MAIN([..., '--pairwise-loss-weight', '1.5', ...])
   ```
3. **fine_distill_weight 강화**: 0.3 (default) → 0.5 → 0.7. fine-stage teacher distillation 으로 pretrain knowledge 보존 + 안정화.
   ```python
   selector.SELECTOR_MAIN([..., '--fine-distill-weight', '0.5', ...])
   ```
4. **epoch_plus 강화**: 5 → 8 → 10. full-fit 시 더 긴 학습.
   ```python
   selector.SELECTOR_MAIN([..., '--epoch-plus', '8', ...])
   ```

**Fallback 순서**: 1 → 2 → 3 → 4 (순차 또는 조합). 각 시도마다 5-fold OOF 측정.

**최종 미달 시**: 4 회 fallback 모두 OOF < 0.70 이면 `selector_no_improvement` severe. plan-009 의 selector arch 교체 후보 trigger.

→ 시간 예산: fallback 1 회당 ~30 분, 최대 4 회 = +2 시간.

### §6.6 시간 예산

- sanity baseline (27 후보 + 새 hyperparam Variant A) 5-fold OOF: ~30 분 (v2.6 신규)
- selector 재학습 (extended pool): ~30 분
- corrector full-fit (기존, plan-004 그대로): ~10 분
- submission.csv 생성 (LB 미제출): ~수 분

---

## §7. STAGE 4 — Corrector Band-Specific 재설계 (secondary) (c9, c10, c11)

### §7.1 Framing

**Secondary lever**: 본 plan 의 main 은 Step 2/3 의 candidate-level 개선. Corrector 재설계는 *알려진 plan-005 fix* 의 추가 booster.

**Conditional path (v2.6 — LB 잔재 fix)**:
- 본 plan LB 제출 = **0 회** (v2.1 carry-over 결정). Step 3 LB 측정 *없음* — *Step 3 OOF* 도달치 기반 가치 평가:
  - Step 3 OOF ≥ 0.75: corrector 재설계 가치 *상대적으로 낮음* → 진행 but expected gain +0.02 정도
  - Step 3 OOF 0.70~0.75: corrector 재설계 *standard ROI* → +0.03~0.05
  - Step 3 OOF < 0.70 (severe trigger): Step 3 fallback 우선, corrector 후순위

→ Step 3 OOF 결과와 무관하게 Step 4 실행 (submission.csv 2 종 박제 — Step 3 / Step 4). **LB 제출 0 회 (carry-over)** — plan-008.1 에서 *최종 산출 1 종* (Step 4) 만 수동 dacon-submit 권장 (§8.6 option B).

### §7.2 Band-Specific Loss

```python
# analysis/plan-008/corrector_band.py
import torch
import torch.nn.functional as F

def band_specific_corrector_loss(
    corrected_pred, raw_pred, target,
    lambda_protect=1.0, lambda_recover=2.0,
    cap: float = 0.006,                   # corrector head output clamp (||corrected - raw|| <= cap), v2.3 fallback grid
    lambda_keep: float = 0.5,             # [0, 0.5cm) band 보존 weight (G3 0.99 hard 정합)
    target_recover: float = 0.009,        # band_recover hinge target — 0.9cm 안 들어오기
):
    """Band hinge loss + cap penalty.

    Input shape contract (v2.6 명시):
      - corrected_pred : torch.Tensor, shape (B, 3) — corrector head output (selected single pred per sample, *post* soft-select)
      - raw_pred       : torch.Tensor, shape (B, 3) — soft-selected raw (pre-correction) pred
      - target         : torch.Tensor, shape (B, 3) — ground-truth label
      - 모든 텐서: float32, same device, gradient flow 는 corrected_pred 에서만.
      - boundary.py 의 기존 corrector head 가 per-sample 단일 좌표 (B, 3) 산출이라 가정 — plan-004 lock-in 답습.
        (만약 boundary 가 per-candidate (B, K, 3) 산출이면 caller 가 soft-select 후 (B, 3) 전달)

    Band hinge loss + cap penalty:
       [0, 0.5cm)  : 보존 hinge — corrected 가 raw 보다 멀어지면 lambda_keep 처벌 (G3 [0,0.5) ≥ 0.99 정합)
       [0.5, 1cm)  : 보호 hinge — 멀어지면 lambda_protect 처벌
       [1, 1.5cm)  : 회수 hinge — target_recover 안 들어오면 lambda_recover 처벌
       [1.5+ cm)   : 무시 (cap 한계)
       Cap penalty: ||corrected - raw|| > cap 시 over-cap 만큼 추가 처벌 (모든 band 공통).

       cap parameter 의미: corrector head 의 *output magnitude* clamp.
         - v2.3 default 0.006 (= 6mm, 노트북 L1 의 "tiny correction" 정신)
         - fallback 0.008 (§7.4 grid search): G3 [1, 1.5cm) 회수 < 0.30 시 1.33x 활성
       Cap 이 *loss 안 hinge* 로 enforce 되는 이유: corrector arch (boundary.py lock-in) 미수정 정책.
    """
    err_raw = torch.norm(raw_pred - target, dim=1)
    err_new = torch.norm(corrected_pred - target, dim=1)
    delta = torch.norm(corrected_pred - raw_pred, dim=1)        # corrector 변경량

    band_keep    = (err_raw < 0.005)
    band_protect = (err_raw >= 0.005) & (err_raw < 0.010)
    band_recover = (err_raw >= 0.010) & (err_raw < 0.015)

    loss = torch.tensor(0.0, device=err_raw.device)
    if band_keep.any():
        loss = loss + lambda_keep * F.relu(err_new[band_keep] - err_raw[band_keep]).mean()
    if band_protect.any():
        loss = loss + lambda_protect * F.relu(err_new[band_protect] - err_raw[band_protect]).mean()
    if band_recover.any():
        loss = loss + lambda_recover * F.relu(err_new[band_recover] - target_recover).mean()
    # Cap penalty (전 band 공통) — corrector head output 의 *magnitude clamp* enforce
    loss = loss + F.relu(delta - cap).mean()
    return loss
```

### §7.3 학습

```python
import inspect
from src.pb_0_6822 import boundary

# Monkey-patch target 식별 (boundary.py 자체는 lock-in, *import 후 attribute 만* 교체).
# 후보 attribute names: 'compute_corrector_loss' (예상 default), 'corrector_loss', 'loss_fn'.
# boundary 모듈에서 *L2 또는 hinge 류 corrector loss 를 *반환* 하는 callable 을 찾는다.
CANDIDATE_ATTRS = ["compute_corrector_loss", "corrector_loss", "loss_fn"]
LOSS_ATTR = next(
    (a for a in CANDIDATE_ATTRS if callable(getattr(boundary, a, None))),
    None,
)
assert LOSS_ATTR is not None, (
    "boundary 모듈에서 corrector loss attribute 식별 실패. "
    f"후보: {CANDIDATE_ATTRS}. 실제 boundary 모듈 dir: {[a for a in dir(boundary) if 'loss' in a.lower() or 'corr' in a.lower()]}"
)

ORIG_LOSS = getattr(boundary, LOSS_ATTR)
setattr(boundary, LOSS_ATTR, lambda c, r, t, *a, **k: band_specific_corrector_loss(
    c, r, t, lambda_protect=1.0, lambda_recover=2.0
))

from src.pb_0_6822 import run_full
run_full.run_boundary()

# 학습 후 복원 (다른 plan 산출 영향 차단)
setattr(boundary, LOSS_ATTR, ORIG_LOSS)
```

**Lock-in 정합**: `boundary.py` 본문 *미수정* (whitelist 외) — 본 patch 는 *Python attribute 교체* 만으로 file 변경 X. `LOSS_ATTR` 의 정확한 attribute name 박제는 c9 (corrector_band.py 작성) 진입 시 `dir(boundary)` 출력의 `*loss*` / `*corr*` candidate 1 개를 식별 + commit msg `decision-note:` 1 줄 박제 (`decision-note: spec-default — boundary.LOSS_ATTR = '<name>' (dir() 식별 결과)`).

### §7.4 λ + cap Grid Search (fallback) — **v2.3 cap 추가 (reviewer #2)**

G3 미달 시 순차 grid search:

**λ tuning**:
- `[0.5, 1cm)` 보호 < 0.95 → `lambda_protect`: 1.0 → 1.5 → 2.0 → 2.5 → 3.0
- `[1, 1.5cm)` 회수 < 0.30 → `lambda_recover`: 2.0 → 2.5 → 3.0 → 3.5 → 4.0

**Cap tuning (v2.3 추가)**:
- 만약 λ tuning 후에도 `[1, 1.5cm)` 회수 < 0.30 면 → `cap`: 0.006 → 0.008 (1.33x) 시도
- 노트북 L1 의 "tiny correction" 정신 여전히 1cm 미만으로 유지
- **cap 0.008 의 수학**: sample 1.5cm + perfect direction 0.8cm 이동 → new err = 0.7cm ✓
  - 단 direction error 도 함께 영향 — λ_recover 강화 + cap 0.008 의 결합으로 회수율 ↑ 기대

**최대 시도**: λ grid 5 회 + cap 2 옵션 = 최대 7 회. 시간 절약 위해 *step-wise* (λ 먼저 5 회 → 그래도 미달 시 cap 0.008 + λ 재시도 2 회).

→ 시간 예산: fallback 최대 +1 시간.

### §7.5 산출

- `runs/baseline/G002_corrector-band/submission_boundary_tiny_soft.csv` + `submission.csv` 사본
- `runs/baseline/G002_corrector-band/boundary_tiny_correction_report.json` — per-band + per-regime
- `analysis/plan-008/corrector_band_summary.{json,md}`

### §7.6 G3 합격 기준

- `[0, 0.5cm) hit_after ≥ 0.99`
- `[0.5, 1cm) hit_after ≥ 0.95`
- `[1, 1.5cm) hit_after ≥ 0.30`
- 전체 OOF ≥ Step 3 OOF + 0.02 (secondary minimum)
- **LB 회수 X (carry-over)** — submission.csv 박제만

### §7.7 시간 예산

- corrector 재학습: ~30 분
- λ grid search (최대): +30 분
- submission.csv 생성 (LB 미제출): ~수 분

---

## §8. LB 제출 정책 — **본 plan 0 회 (할당량 소진), plan-008.1 carry-over**

### §8.1 본 plan 내 LB 제출 미수행

**이유**: 2026-05-12 일자 dacon 일일 제출 할당량 소진. 본 plan 종료 후 *다음 날* (2026-05-13 KST) plan-008.1 carry-over 로 LB 회수.

본 plan 내 변경:
- Step 3 / Step 4 끝에 **submission.csv 생성만** (dacon-submit skill 호출 X)
- `lb_log.md` 의 `lb_score` 컬럼 = `TBD` (carry-over 사유 박제)
- frontmatter `lb_score: TBD` + `status: partial` (carry-over)
- ~~c8, c11~~ (sub-lb commit) → **DEFERRED 상태** (plan-008.1 로 이관)

### §8.2 submission.csv 박제 위치

```
runs/baseline/G001_candidate-redefine/submission.csv     ← Step 3 산출
runs/baseline/G002_corrector-band/submission.csv         ← Step 4 산출 (본 plan 최종)
```

→ 다음 날 사용자 또는 plan-008.1 가 *수동* dacon-submit (할당량 reset 후).

### §8.3 `analysis/plan-008/lb_log.md` 포맷 (carry-over 박제)

```markdown
| timestamp_kst             | exp_id                       | step | isSubmitted | lb_score | detail |
|---------------------------|------------------------------|------|-------------|----------|--------|
| 2026-05-12T??:00+09:00    | G001-step3-selector          | 3    | false       | TBD      | quota_exhausted_2026-05-12, deferred to plan-008.1 |
| 2026-05-12T??:00+09:00    | G002-step4-corrector         | 4    | false       | TBD      | quota_exhausted_2026-05-12, deferred to plan-008.1 |
```

### §8.4 plan-008.1 carry-over spec (다음 날 commit)

다음 날 (2026-05-13 또는 그 이후):

```python
# plan-008.1 의 commit
# c1 (또는 plan-008 의 후속 commit): LB 회수
Skill(skill="dacon-submit",
      args="runs/baseline/G002_corrector-band/submission.csv G002-step4-corrector")
# 응답: {isSubmitted: True, lb_score: <float>, ...}
```

회수 후:
- `analysis/plan-008/lb_log.md` row 갱신 (`isSubmitted: true`, `lb_score: <float>`, `detail: OK (carry-over from 2026-05-12)`)
- 3 파일 frontmatter `lb_score: TBD → <float>` + `status: partial → all_complete`
- single commit c14.1 또는 plan-008.1 의 c1

### §8.5 `lb_score` frontmatter 동시 갱신 (3 파일, plan-008.1 시점)

- `plans/plan-008-candidate-redefine-corrector-redesign.md` top-level
- `plans/plan-008-candidate-redefine-corrector-redesign.results.md`
- `analysis/plan-008/results.md`

→ 본 plan 마감 시 `TBD` 박제, plan-008.1 회수 시 *동시* 갱신.

### §8.6 Step 3 LB 옵션 (carry-over 시점)

dacon 일일 할당량 = 2 회 가정 시:
- 다음 날 (2026-05-13) 가용 = 2 회
- 옵션 A: Step 3 + Step 4 둘 다 제출 (할당량 다 씀)
- 옵션 B: Step 4 만 (최종 산출, 1 회) — 나머지 1 회 다음 작업용 보존

→ **옵션 B 권장**. Step 3 LB 는 informational (selector 단독 효과 측정), Step 4 LB 가 본 plan 의 main 산출. plan-008.1 spec 에서 결정.

---

## §9. STAGE 5 — (선택) Test-Internal Validation (c12, c13)

### §9.1 측정 식

```python
def evaluate_pipeline(test_x, subsamples, temp: float, lambda_recover: float) -> float:
    """Step 3/4 pipeline 의 *test-internal* hit@1cm 측정.
       각 (end_idx, target) sub-sample 에 대해 candidate 좌표 + corrector + soft-select 적용.
       Returns: hit rate aggregated over all subsamples."""
    hits, total = 0, 0
    from src.pb_0_6822 import boundary as _bnd
    for end_idx, target in subsamples:
        cands = selector.make_candidates(test_x, end_idx, horizon=2)         # uses monkey-patched extended pool
        scores = selector.predict_scores(test_x, end_idx, cands)              # selector inference (Variant A)
        # corrector 적용 (Step 4 의 lambda_recover 가 hyperparam — 학습된 head 의 추론 시점 weight 으로 사용)
        corrected = _bnd.apply_corrector(cands, lambda_recover=lambda_recover)
        pred = _bnd.soft_select(corrected, scores, temperature=temp)          # (N, 3)
        err = np.linalg.norm(pred - target, axis=1)
        hits += int((err <= R_HIT).sum())
        total += int(err.shape[0])
    return float(hits / max(total, 1))


def stage5_test_internal_tune():
    """Test 11점 trajectory 의 내부 sub-sample 50K = test 분포 free label.
    Step 3/4 의 hyperparam (corrector λ, selector temp) 만 grid search.

    `step4_oof_hit` 는 outer scope (Step 4 산출 §7.5 의 oof_hit value) 에서 주입 — 본 함수 호출 직전에
    `step4_oof_hit = json.load(...analysis/plan-008/corrector_band.json)['oof_hit']` 등으로 binding."""

    test_ids = selector.read_submission_ids(DATA_ROOT / "sample_submission.csv")
    test_x = selector.load_stack(DATA_ROOT / "test", test_ids)

    subsamples = []
    for end_idx in range(5, 10):
        target = test_x[:, end_idx + 1]
        subsamples.append((end_idx, target))

    best_estimate = -np.inf
    best_params = None
    for temp in [0.02, 0.03, 0.05]:
        for lambda_recover in [1.5, 2.0, 2.5]:
            hit = evaluate_pipeline(test_x, subsamples, temp, lambda_recover)
            if hit > best_estimate:
                best_estimate = hit
                best_params = {"temp": temp, "lambda_recover": lambda_recover}
    return {
        "best_params": best_params,
        "test_internal_hit": best_estimate,
        "estimated_lb_gain": best_estimate - step4_oof_hit,
    }
```

### §9.2 G4 합격 기준 (선택)

- `estimated_lb_gain ≥ 0.01`
- 미달 시 carry-over plan-009

### §9.3 시간 예산

- ~1 일

---

## §10. STAGE 6 — Synthesis + plan-009 후보 (c14)

### §10.1 `analysis/plan-008/results.md`

frontmatter:
```yaml
---
plan_id: 008
based_on:
  - 004
  - 005
  - 006
  - 007
finished_at: <ISO8601 KST>
status: partial (carry-over to plan-008.1 for LB submission)
exp_ids_completed:
  - G001_candidate-redefine
  - G002_corrector-band
lb_exp_id: G002-step4-corrector
lb_score: TBD (quota_exhausted_2026-05-12, deferred to plan-008.1)
lb_submitted_at: null
---
```

본문:
- Step 1 진단 (dominant cause + 가지치기 list + margin signal + per-regime oracle gap)
- Step 2 oracle trajectory (27 → pruned → 5-family extended → filtered final)
- Family marginal table (각 family contribution + kept/dropped)
- Step 3 selector OOF + top-1 ranking + LB
- Step 4 corrector per-band hit + 전체 OOF + LB
- (선택) Step 5 test-internal gain
- plan-004 (0.6806) → 007 (0.6598) → 008 LB trajectory
- decision-note 박제 list

### §10.2 `analysis/plan-008/next_plan_candidates.md`

**최소 후보 2 개**. 시나리오 분기:

**시나리오 A — Step 4 LB ≥ 0.80**:
1. Test-internal validation 본격 적용 + plan-007 MLP coeff 재시도 — 0.83+ target
2. 새 family 더 확장 (Step 1 진단의 secondary cause 회수)

**시나리오 B — Step 4 LB 0.75~0.80**:
1. Selector arch 교체 (TCN/Transformer) — **ranking gap 7pp 회수 시도** (oracle 안 후보를 잘 픽하는 능력 강화). top-1 ranking 정확도 (12.6% → 25%+) 는 *상관 metric*, main metric 은 argmax hit ↑ via gap_ranking ↓.
2. Corrector arch deeper / attention-based

**시나리오 C — Step 4 LB 0.70~0.75**:
1. Family 재검토 — 효과 없었던 family 제거 + 다른 family 시도
2. Pruning 강화 / 완화 ablation

**시나리오 D — Step 4 LB < 0.70**:
1. plan-006 framework 으로 회귀 (corrector 기존 loss + regime 재도입 검토)
2. plan-008 전체 결정 재검토 (Variant A baseline 의 적합성)

각 후보 4 항목: 근거 metric / 예상 ROI / 작업 범위 / 선행 조건.

#### §10.2.1 Cross-scenario: Ranking 개선 6 카테고리 (v2.6 신규 — 필수 박제)

**근거**: plan-008 의 ranking 능력 동결 (caveat #4, #13). top-1 ranking 12.6% 의 *직접 원인* = 현 loss 가 *binary hit/miss* 만 학습 (cross-entropy soft target = "1cm 안 후보들 균등") — *진짜 best 픽* 학습 X. 모든 시나리오 (A~D) 에서 plan-009 main task 후보 가능.

`next_plan_candidates.md` 의 별도 section "Ranking 개선" 에 **6 카테고리 + ROI 표** 필수 박제. 각 카테고리:

**카테고리 1: Loss 변경 (★ 최고 ROI, arch 보존)**

| 후보 | mechanism | 예상 LB gain | 비용 |
|---|---|---|---|
| **1.3 NDCG@1 differentiable** | `loss = 1 − softmax(score)[oracle_best]` — top-1 ranking 의 differentiable proxy | **+0.03** | ★ (loss 함수만 교체) |
| 1.1 Pairwise margin | sorted pair 에 hinge — score 순서가 err 순서 일치 강제 | +0.02 | ★ |
| 1.2 Listwise (ListMLE) | `−log P(top-1 = oracle_best)` — top-1 log-likelihood 직접 | +0.02~0.03 | ★ (gradient 불안정 risk) |
| 1.4 Focal ranking | hard sample 가중 (1−score_best)^γ — 88% miss case 집중 | +0.01~0.02 | ★ (다른 loss 와 조합) |

**카테고리 2: Selector arch 교체 (★★★ big change)**

| 후보 | mechanism | 예상 LB gain | risk |
|---|---|---|---|
| 2.1 Set Transformer | candidate set 의 self-attention (`cand_i ↔ cand_j` 직접 비교) — 현 framework 의 "trajectory hidden 만 attend" 한계 해소 | +0.04~0.05 | mid (overfit) |
| 2.2 Twin pairwise | (i, j) binary classifier + round-robin — ranking 직접 학습 | +0.03~0.05 | mid (inference 666×) |
| 2.3 Transformer (full) | trajectory + 후보 통합 token sequence + bi-directional | +0.05 | **high (overfit, data 10K)** |
| 2.4 TCN | 1D causal conv — GRU 의 sequential bottleneck 제거 | +0.01~0.02 | low (marginal, seq 짧음) |

**카테고리 3: Multi-stage selector (★★ 분해)**

| 후보 | mechanism | 예상 LB gain | 비용 |
|---|---|---|---|
| 3.1 Coarse-to-fine 2-stage | Stage 1 cheap filter 37→top-5, Stage 2 expensive rerank — search space 5 로 축소 → ranking 정확도 ↑ | +0.03~0.05 | ★★★ (2 model train) |
| **3.2 Hard top-K filter** | test-time only: softmax 전 top-3 외 후보 −inf — centroid drift 직접 fix | **+0.02** | ★ (학습 X, 1 줄 추가) |
| 3.3 Per-trajectory family routing | family_pred → 해당 family 후보만 ranking — 시나리오 의존 | +0.02~0.03 | ★★ (hard routing gradient) |

**카테고리 4: Score combination 재설계 (★★ 작은 변경)**

| 후보 | mechanism | 예상 LB gain | risk |
|---|---|---|---|
| 4.1 Confidence-weighted | `final = c × gru + (1−c) × bias`, c=σ(MLP(hidden)) — GRU uncertainty 반영 | +0.01~0.02 | low |
| 4.2 Outlier penalty | `−λ × ∥cand − centroid∥` — soft 평균 안정성 | +0.01 | low |
| 4.3 Bias × GRU multiplicative | `bias × σ(gru)` 곱셈 — train hit_rate 낮은 후보 강제 down-weight | +0.01~0.02 | mid (physics_bias 틀린 sample 회수 불가) |

**카테고리 5: Non-parametric class (caveat #20 박제)**

| 후보 | mechanism | 예상 LB gain | 비용 |
|---|---|---|---|
| 5.1 KNN nearest-neighbor | K=5 유사 trajectory 의 t+1 displacement 평균 | +0.03~0.05 | ★★ |
| 5.2 GP residual | train residual GP fit → test posterior | +0.02~0.04 | ★★★ |
| 5.3 Per-sample MLP regression | direct (x,y,z) 회귀 (candidate 우회) | +0.02~0.05 | ★★★★ (overfit risk) |
| 5.4 Stacked residual | XGBoost on per-candidate errors | +0.02~0.04 | ★★ |

**카테고리 6: Other (carry-over caveat 박제)**

| 후보 | 출처 caveat | 비고 |
|---|---|---|
| 6.1 Regime-agnostic per-sample formula 회귀 | #12 | Family 4 drop 의 long-tail 회수 |
| 6.2 Greedy brute-force (template_pool 2^15 조합 전체 search) | #14 | local optimum 회피 |
| 6.3 Field 분리 (binormal_scale_fs / world_z_keep_trig) | #15 | semantic clean-up |

#### §10.2.2 plan-009 권장 sequence (v2.6 신규)

**Phase 1 (cheap, no arch)**: 1.3 NDCG@1 + 1.1 Pairwise margin + 3.2 Hard top-K filter → 누적 **+0.05~0.07** (LB 0.73~0.75 가능)

**Phase 2 (mid)**: 3.1 Coarse-to-fine 2-stage → +0.03~0.05 추가

**Phase 3 (big, risky)**: 2.1 Set Transformer (Phase 1 결과 미흡 시) → +0.05

**Phase 4 (carry-over 시나리오 D)**: 5.x non-parametric (KNN / per-sample MLP) → +0.03~0.05

→ Phase 1 만으로도 plan-009 의 minimal viable. plan-008.1 carry-over LB 측정 후 결정.

### §10.3 G_final 합격 기준

- `results.md` + `next_plan_candidates.md` 작성
- 후보 ≥ 2 + 4 항목 박제 (시나리오 A~D)
- **(v2.6 신규) §10.2.1 의 ranking 개선 6 카테고리 ROI 표 박제 필수** — `next_plan_candidates.md` 의 별도 section "Ranking 개선" 에 6 카테고리 모두 + Phase 1~4 sequence 포함
- **(v2.6 신규) §10.2.2 의 plan-009 권장 sequence (Phase 1~4) 박제**
- 3 파일 frontmatter `lb_score` 동시 갱신 + `status` 통일
- 모든 G-gate [DONE]

---

## §N+1. 작업량 총 회계

- 코드: 5 file (`diagnostic.py`, `prune_and_redefine.py`, `selector_retrain.py`, `corrector_band.py`, `test_internal.py`) + `candidates_extended.py` (6 family 정의, ~400 lines)
- 학습:
  - Selector 재학습 (Step 3): ~30 분
  - Corrector 재학습 (Step 4, λ tuning 포함): ~1 시간
  - (선택) Step 5: ~1 일
- 분석: ~2 시간 (진단 + 6 family oracle 측정 + filter)
- **LB 제출: 0 회 (본 plan, 할당량 소진)** + plan-008.1 carry-over 1~2 회 (다음 날)
- Synthesis: ~30 분
- **총 wall-time 예산: ~4~6 시간** (Step 5 제외) — LB 대기 시간 0 (carry-over)

---

## §N+2. results.md 필수 항목

- exp_id (G001/G002), plan_id (008), based_on (004 + 005 + 006 + 007)
- lb_exp_id (G002-step4-corrector), lb_score (Step 4 최종), lb_submitted_at
- Step 1 diagnostic
- Step 2 oracle trajectory + 6 family marginal + filter 결과
- Step 3 selector OOF + top-1 ranking + LB (Variant A path 검증)
- Step 4 corrector per-band hit + 전체 OOF + LB
- (선택) Step 5 test-internal gain
- plan-004 → 007 → 008 LB trajectory + Variant A 단순화 가치 박제
- plan-009 후보 ≥ 2 + 4 항목
- decision-note 박제 list

---

## §N+3. 통계 함정 & caveats

1. **가지치기의 sample-specific 손실**: aggregate oracle delta < 0.001 기준이 *전체* 만 봄. Worst regime (16/17/10) 에서 unique best 후보 손실 가능. G1 합격 기준에 `per_regime worst oracle_after ≥ oracle_orig - 0.005` 추가 검증.
2. **새 family 의 cand_feat 차원 호환성** (v2.2 Option A 로 해소): `CandidateSpec` schema 에 `omega_scale`, `arc_curvature`, `z_scale`, `family_id` 4 fields 추가 + `candidate_spec_features` 가 9 scalar + 7 family one-hot 출력 + `make_candidate_features` 의 interactions term 에 새 cross-term 2 개 추가. 기존 27 후보는 모두 default → backward-compat. 미적용 시 `schema_v22_residue` severe — §6.1.5 의 assert 가 검증.
3. **Variant A path 강제**: `regime_prior_strength=0` 이 selector.py 의 모든 학습 + inference 경로에 enforce 되는지 검증. `regime_residue` severe 발동 시 즉시 escalate.
4. **6 family 동시 학습의 selector 부담**: ~40 후보로 학습 시 sample 별 cross-entropy 가 더 분산. top-1 ranking 정확도 (현 12.6% — *27 중 정답에 가장 가까운 후보 정확 픽 비율*, 1cm 안 hit 비율 아님) 가 더 떨어질 가능성. 단 *진짜 metric* 은 argmax hit (1등 픽이 1cm 안 들어오는 비율) 과 그것의 oracle gap (ranking gap). top-1 ranking 은 *상관 metric*, hit 가 여러 개 있는 sample 에선 best 가 아닌 후보 픽해도 OK. → soft averaging 의 cover 효과 + pruning 의 cluster 분리 모두 *secondary* 효과.
5. **새 family 의 train hit_rate 낮음**: physics_bias 는 train hit_rate 기반 — 새 family 가 train 에서 hit_rate 낮으면 physics_bias 작음 → selector 안 픽. → bias 초기값 검토 또는 family-marker one-hot 으로 selector 가 명시적 학습 가능.
6. **Oracle 0.90 stretch 의 원리적 한계**: 모기 비행의 측정 noise + 환경 변수 (다른 모기, 바람) 가 fundamental floor. floor 0.05~0.10pp 면 ceiling 0.90~0.95. floor 0.15+ 면 ceiling 0.80~0.85. regime 1 (현 hit 0.93) 의 *fundamental noise floor* 가 ceiling 의 indicator.
7. **Corrector band 경계의 sharp transition**: hinge loss 가 0.010 에서 sharp transition. sample 의 err_raw 가 noise 로 0.010 근처면 학습 불안정. → 향후 smooth (sigmoid 근사) 검토.
8. **Test-internal subsample 의 non-stationarity**: plan-007 Step 1 finding — sliding window 의 KS 통계상 train end_idx=10 분포와 다름. test-internal 도 같은 risk. Step 5 grid search 의 *test-internal* overfit 위험. → Step 5 informational only.
9. **LB 비동기 carry-over**: plan-002/003/004/006/007 패턴 — `lb_score: TBD` carry-over, follow-up commit 으로 갱신.
10. **regime 폐기의 *late-stage* 위험**: 만약 새 family 가 특정 regime 에 강하게 localized 효과 (예: trig 후보가 regime 16/17 에서만 효과) 이면 regime bias 가 그 효과를 강화해 줄 수 있음. 본 plan 은 plan-005 STAGE 6 의 +0.001 noise 측정으로 그 lift 포기. Step 3 LB 미달 시 plan-009 의 "새 family 전용 regime bias 표 재추가" 검토.
11. **`CandidateSpec` schema 확장의 backward-compat 위험** (Option A v2.2): selector.py 의 `CandidateSpec` 에 4 fields 추가 시 기존 27 의 dataclass 인스턴스화는 default 로 호환되지만, *외부 import 가 `CandidateSpec.__init__` 의 positional args 순서에 의존* 하면 깨질 수 있음. 본 plan 에서 `selector.CANDIDATES` 의 27 entry 는 모두 keyword args 사용 (확인 필요), 외부 import (분석/테스트) 는 본 plan whitelist 내 `analysis/plan-008/**` 만. 다른 plan 산출 (plan-004~007) 은 read-only — `CandidateSpec` 인스턴스화 가능성 검토 (다행히 모두 module-level import만, 인스턴스 직접 생성 X). 위반 시 c2 (diagnostic) 단계에서 ImportError 발생 → 즉시 catch.
12. **Family 4 drop 의 regime-specialized sample 손실 위험** (v2.3): reviewer #1 의 self-contradiction 회피 위해 Family 4 (per_regime_specialized) drop. 단 regime 16/17/10 의 sharp-turn / decel-arc dynamics 가 trig (rot_high_150) + arc (arc_decel) family 로 *완전* 회수된다는 보장 없음. 부분 회수 시 G1 의 per_regime worst oracle ≥ 0.55 미달 가능. plan-009 후보 = "regime-agnostic per-sample formula 회귀" (per-sample MLP 가 coefficient 회귀, regime ID 사용 X).
13. **Plan-008 의 ranking 능력 한계** (v2.3): top-1 ranking 12.6% (plan-005 측정) 은 본 plan 에서 *그대로 유지* (selector arch 미수정). 본 plan 의 LB 회수는 (a) 후보 풀 확장 + (b) corrector + (c) pruning 의 부수 효과 만. selector ranking 자체의 개선은 plan-009 의 main task. Step 4 LB < 0.74 면 ranking 능력 동결의 hard ceiling 가능성 검토.
14. **Strategy D (greedy set cover) 의 local-optimum 위험** (v2.3): Greedy algorithm 은 *전역 optimal pool* 보장 X. 매 iteration *국소 best* 만 선택 → 후보 A + B 가 *함께* add 시 oracle 더 높은데 A 또는 B 단독 add 의 marginal 이 낮으면 둘 다 skip 가능. 단 본 plan 의 template_pool (15 개) 은 작아 *조합* 의 brute-force 도 가능 (`2^15 = 32768`, 각 oracle 측정 ~0.1초 → ~1 시간). G1 fallback 으로 "조합 brute-force" 활성 가능 (plan-009 후보).
15. **`binormal_scale` field 의미 의존** (v2.3): `CandidateSpec.z_scale` field 의 이름은 v2.2 그대로 유지하되 *Frenet-Serret family 안에서만* binormal frame 의미. 다른 family (trig 등) 는 여전히 world z 방향 (간접) 사용 가능 — make_rot_candidates 가 `d1[:, 2] * spec.z_scale` 식으로 world z 직접 곱하면 위배. v2.3 결정 = trig family 의 `z_scale=1.0` default 유지 (= world z 그대로 사용), Frenet-Serret family 만 *binormal frame 의미*. 향후 plan-009 에서 field 이름 *명시적 분리* (binormal_scale_fs / world_z_keep_trig) 검토.
16. **Structural containment 의 *retrained selector* 한계** (v2.4): containment 식별은 selector 거동 *무관* (강점). 단 *retrained selector* 가 redundant 라고 식별된 후보 i 를 *어떻게 활용할지* 모름 — 만약 retrained 가 i 의 unique pattern 을 발견했다면 제거 손해. Safety check (oracle delta < 0.001) 으로 *aggregate* 손해는 막지만 *retrained selector 의 distribution shift* 는 못 잡음. Step 3 (selector 재학습) 의 OOF 측정이 진짜 verification — G2 미달 시 pruning rollback 검토.
17. **Containment 의 *threshold 민감도*** (v2.4 → v2.7 자동 완화): `containment_soft ≥ 0.95` + `coord_dist < 0.005m` 는 magic number. LiDAR 40ms 측정 jittering + 모기 5~10mm scale 고려 시 5mm 좌표 정합은 가혹 → 너무 엄격하면 pruning 거의 없음. **v2.7 자동 완화**: §4.1 의 `_identify_prune` 가 default (0.95, 5mm) 로 1차 식별 후 `len(prune_candidates) < 3` 이면 *자동* 으로 (0.90, 10mm) 재탐색. `summary.prune_threshold_tier ∈ {"strict_v2.4", "relaxed_v2.7"}` 박제로 어느 tier 가 채택됐는지 audit. 사용자 수동 결정 분기 제거. (이전 v2.4: 사용자 수동 검토 — `prune_count` 박제 → 결정 → 재실행.)
18. **Oracle miss mask 의 sample 분포 균질성 위험** (v2.5): `err.min(axis=1) > R_HIT` 가 ~2800 sample (28%). 이 안에 *여러 다른 dynamics* (sharp turn + decel arc + z-drift) 가 섞여 있으면 single `corr_rotation` 같은 aggregate corr 이 *희석* 가능. 즉 dominant cause 가 *진짜 mixed* 일 때 진단이 부정확. mitigation: oracle_miss_regime_dist_sanity 박제로 regime grouping 확인 + caveat #1 (per-regime worst oracle 손해 검증) 으로 보완. 추가 분석 (clustering) 은 plan-009 후보.
19. **Mask 변경의 plan-005 worst-100 재활용 호환성** (v2.5): plan-005 의 worst-100 worker (heuristic 분석) 는 regime 기반. v2.5 의 `oracle_miss` mask 는 regime 무관 — 두 분석의 sample 집합이 *부분 disjoint* 가능. plan-005 인계 데이터 사용은 *informational sanity* 만 (per_regime_oracle_sanity 표), main residual decomposition 은 v2.5 oracle_miss 위에서 *독립* 측정.
20. **Oracle 0.85 minimum 의 낙관 위험** (v2.6, Plan agent 검토 [CRITICAL] 2): template_pool 15 (실효 13~14, snap drop + Family 4 drop 후) 의 평균 marginal 회수 가정 +0.019 × 7 templates = +0.13 → oracle 0.85 도달 *낙관*. 실제는 set-cover *diminishing returns* 따를 가능성 큼 — first 1~2 add +0.03~0.04, 3 번째 +0.015, 이후 < 0.01 → 7 templates × 평균 0.014 → +0.10 → **oracle 0.81 도달 추정**. 정확히 `redefinition_partial` warn-only band (0.78~0.85). mitigation: (a) §0.5 의 warn-only band 가 graceful degradation 제공, (b) plan-009 후보 list 에 *진짜 new class* (KNN nearest-neighbor / GP residual / per-sample MLP 회귀) 박제 권장. 본 plan 의 v2.6 framing 변경 X — minimum 0.85 유지하되 *예상치 0.78~0.82* 로 inner expectation 조정 + plan-009 carry-over path 강화.

---

## §N+4. 변경 이력

- v2.7 (2026-05-12): **§4.1 pruning auto-relaxation (사용자 추가 요청 반영)**.
  - **이유**: 사용자 지적 — "LiDAR 40ms jittering + 모기 5~10mm scale 고려 시 5mm 좌표 정합은 가혹. Pruning 쌍이 3개 미만 시 auto-relaxation 추가하라". v2.4 의 manual review 분기 → v2.7 자동.
  - **§4.1 `_identify_prune` 헬퍼 신설**: (soft_thr, dist_thr) 파라미터화. 1차 (0.95, 5mm) → `len < 3` 이면 자동 (0.90, 10mm) 재탐색.
  - **summary 박제**: `prune_threshold_tier ∈ {"strict_v2.4", "relaxed_v2.7"}` + `prune_threshold_used`.
  - **§4.3 G0 합격 기준**: "비어 있어도 통과" 조건에 v2.7 relaxed 도 0 시 inherent diversity 신호 명시.
  - **§N+3 caveat #17 갱신**: 자동 완화 mechanism 박제, 사용자 수동 결정 분기 제거.
  - **모든 spec 외 v2.6 의 의도 유지**.
- v2.6 (2026-05-12): **Plan agent 비판적 검토 반영 — 4 항목 박제 (격리 권고는 기각, cheap fix 만 채택)**.
  - **이유**: 사용자 요청 — "전체 계획 flow 검토. 사용자의 의도에 맞춰서. 각 step 이 타당한가 기준으로 검토. 서브에이전트 호출해서 피드백받고 메인 에이전트가 검증한다". Plan agent 가 6 항목 비판 (3 CRITICAL + 2 SUGGEST + 1 정합 OK). 메인 검증 결과: Step 4 격리 권고 기각 (사용자 의도 §0 의 secondary 에 명시), template_pool 부족 정당 (단 warn-only band 가 graceful mechanism 제공), Step 3 sanity baseline + assert 강화는 cheap fix 채택.
  - **§7.1 LB 잔재 fix**: "LB 제출 2 회 보장" → "submission.csv 2 종 박제, LB 제출 0 회 (carry-over)". v2.1 의 LB 0 회 결정과 정합. Step 3 *OOF* 기반 conditional path (이전 *LB* 기반).
  - **§6.0 신설 (sanity baseline)**: Step 3 진입 전 27 후보 + 새 hyperparam OOF baseline 측정. `family_effect = oof_extended − sanity_baseline_27` 분리. G2 합격 기준에 family_effect ≥ +0.03 추가.
  - **§6.2 assert 강화**: 학습 후 regime_bias_table 분산 < 1e-10 검증 + per-regime ens_scores diff median 측정 (informational warn). v2.5 의 "regime_prior_strength=0 enforce 확인" 만 → 학습 *후* 산출 검증까지 확장.
  - **§6.4 G2 합격 기준**: sanity_baseline_27_oof 재현 + family_effect ≥ +0.03 추가.
  - **§6.6 / §7.7 시간 예산**: "LB 제출 ~수 분" → "submission.csv 생성 (LB 미제출) ~수 분" + §6.6 에 sanity baseline ~30 분 추가.
  - **§0.5 severe 추가**: `sanity_baseline_drift` (warn-only), `family_effect_marginal` (warn-only).
  - **§0.5 commit chain**: c1.6 (v2.6 spec) + c5.5 (sanity_baseline_27.py) 신설.
  - **§N+3 caveat #20 신설**: oracle 0.85 minimum 의 낙관 위험 정량 박제 (실제 예상 0.78~0.82, plan-009 carry-over path 강화).
  - **§10.2.1 / §10.2.2 신설**: Cross-scenario ranking 개선 6 카테고리 ROI 표 (Loss / Arch / Multi-stage / Score / Non-parametric / Other) + plan-009 권장 sequence (Phase 1~4). c14 G_final 합격 기준에 필수 박제 명시.
  - **모든 spec 외 v2.5 의 oracle_miss mask / structural containment / Strategy D / Family 4-5 drop / Variant A baseline / LB carry-over 의도는 유지**.
- v2.5 (2026-05-12): **STAGE 1 mask 교체 — `worst_regime ∈ {10,16,17}` → `oracle_miss = err.min > 0.01`** (Variant A 정합, main lever 의 직접 target).
  - **이유**: 사용자 지적 — "G0 에서 왜 regime 안에서 오차를 탐색해? 그냥 oracle 밖의 분포를 바로 볼 수는 없나?". v2.4 의 worst_regime mask 는 self-contradiction (Variant A regime 폐기 + diagnostic regime 사용). plan-008 main lever (oracle 천장 0.7188 → 0.85+ 회수) 의 *직접* target = oracle miss sample (~2800, 28%).
  - **§4.1 diagnostic 의 mask 교체**: `worst_mask = np.isin(regimes, [10, 16, 17])` → `oracle_miss_mask = err.min(axis=1) > R_HIT`.
  - **§4 summary key 변경**: `residual_breakdown_worst` → `residual_breakdown_oracle_miss`, `per_regime_oracle` → `per_regime_oracle_sanity` (격하). 신규 키 `mask_strategy`, `n_oracle_miss`, `oracle_miss_rate`, `oracle_miss_regime_dist_sanity`.
  - **§4.2 산출 markdown**: Oracle miss residual breakdown 박제 + per-regime oracle gap 표 "sanity only" 명시.
  - **§4.3 G0 합격 기준**: `mask_strategy == "oracle_miss_v2.5"` 박제 강제 + dominant cause 도출이 oracle miss 위에서 옴 명시.
  - **§0.5 / §1.4 H1 / commit chain c1.5 / decision-note**: oracle miss framing 으로 갱신.
  - **§N+3 caveats**: #18 (oracle miss sample 의 dynamics mixed 위험), #19 (plan-005 worst-100 sample 부분 disjoint) 신규.
  - **G2 OOF 0.71 → 0.70**: v2.4 의 별도 완화 (0.6570 + 0.05 → 0.043) 도 v2.5 박제에 포함.
  - **모든 spec 외 v2.4 의 structural containment / Strategy D / Family 4-5 drop / Variant A baseline / LB carry-over 의도는 유지**.
- v1 (2026-05-12): 초안 — plan-007 진단 + plan-005 인계 + plan-006 worst regime 데이터 기반. 5 step + synthesis. LB 2 회. corrector 재설계 main lever 2 framing.
- v2 (2026-05-12): **Framing 재정의 — radical expansion + pruning for soft + Variant A baseline 확정**.
  - **§0 / §0.5 / §1 / §2 spec 갱신**: main lever 2 (재정의 + pruning), secondary (corrector). regime infra 완전 폐기. Target LB 0.78~0.85.
  - **§4 STAGE 1 진단**: softmax diffusion + per-regime oracle gap 추가.
  - **§5 STAGE 2**: 5+ family (6 family) 동시 정의 + family marginal filter (hybrid C path). 20+ 새 후보. oracle target 0.85 minimum / 0.90 stretch.
  - **§6 STAGE 3**: Variant A path 강제 (`regime_prior_strength=0`). OOF target 0.70 (v2.4 완화, 이전 0.71). severe `regime_residue` 추가.
  - **§7 STAGE 4**: secondary framing. OOF target Step 3 + 0.02 (이전 +0.04 완화).
  - **§N+3 caveats**: 10 항목 (이전 7 + 새 3 — Variant A path + 6 family fitting + Oracle 0.90 limit).
  - **모든 spec 외 STAGE / G-gate / commit chain / severe / 두 main lever 의도는 v1 의 구조 유지**.
- v2.1 (2026-05-12): **LB 제출 정책 변경 — 본 plan 0 회, plan-008.1 carry-over**.
  - **이유**: 2026-05-12 dacon 일일 할당량 소진 — 본 plan 내 LB 호출 X.
  - **§0 / §0.5 / §2.2 / §3.2 / §6.4 / §7.6 spec 갱신**: LB 회수 요구사항 모두 carry-over 처리.
  - **commit chain**: c8 (sub-lb Step 3), c11 (sub-lb Step 4) → DEFERRED 상태 (plan-008.1 으로 이관).
  - **§8 LB 정책 전면 재작성**: submission.csv 박제만 + carry-over spec + plan-008.1 의 옵션 (Step 3+4 둘 다 vs Step 4 만).
  - **severe**: `lb_unsubmitted`, `dacon_submit_skill_missing`, `lb_anomaly` → 본 plan scope X 명시.
  - **frontmatter (§10.1)**: `status: partial (carry-over)`, `lb_score: TBD`, `lb_submitted_at: null` default.
  - **모든 spec 외 v2 의 main lever / oracle target / Variant A baseline 의도는 v2 유지**.
- v2.4 (2026-05-12): **Pruning criterion 변경 — selector pick rate → structural containment**.
  - **이유**: 사용자 질문 — "효과 측정이 아니라 후보 두개의 상관관계를 보고 포함관계가 되면 삭제해야하는거아닌가". selector pick rate 기반 가지치기는 *현재 selector 거동* 에 의존 → robustness 부족.
  - **§4.1 diagnostic 의 prune_candidates 식별 로직 교체**: pairwise containment (strict / soft ≥ 0.95) + coord_dist < 0.005m + hit_rate dominance. selector_pick_idx 사용 X.
  - **§5.1 Step 2a**: identification + verification + sanity 3 단계 분리:
    - Identification = structural containment (selector 무관)
    - Verification = oracle preservation < 0.001
    - Sanity = post-hoc soft_hit_pruned (informational)
  - **§0 / §0.5 / §2.1 / §1.4 H2**: containment 기반 framing 으로 갱신.
  - **§0.5 decision-note**: "Step 2a 가지치기 = structural containment" 박제 (selector pick rate 사용 X 명시).
  - **모든 spec 외 v2.3 의 Strategy D / family 정의 / corrector / Variant A baseline 의도는 유지**.
- v2.3 (2026-05-12): **Reviewer 피드백 8 항목 반영 + Strategy D (Greedy Set Cover) 채택**.
  - **§0 한 줄**: main lever = greedy set cover (이전 radical expansion + pruning 동급). pruning 격하 — "Step 1 측정 후 결정".
  - **§0.5 G-gates**: Step 1 의 `argmax_hit` + ranking-vs-drift 분해 추가. Step 2 algorithm 변경 (hybrid C → Strategy D).
  - **§0.5 severe 추가**: `regime_backdoor_residue` (Family 4 식 잔재), `world_axis_dependence` (world z 직접 후보).
  - **§0.5 decision-notes**: Strategy D 채택, Family 4/5 drop, fs_3d_binormal, cap fallback, selector fallback 강화, ranking improvement → plan-009.
  - **§1.4 H1.5 신규**: ranking-vs-drift dominance 측정 가설.
  - **§5.2.3**: 6 family → 5 family + 13~15 templates. Family 4 (per_regime) drop. snap drop. fs_3d_banking → fs_3d_binormal (binormal frame).
  - **§5.3 신설**: Greedy Set-Cover algorithm code + iteration log + stop conditions + 예상 trace.
  - **§5.4/§5.5/§5.6**: 산출 + G1 + 시간 갱신 (greedy 기준).
  - **§6.5 fallback 강화**: 4 단계 (hidden / pairwise / distill / epoch_plus) 순차.
  - **§7.4 fallback cap 추가**: λ + cap 0.006 → 0.008 (1.33x).
  - **§N+3 caveats**: 12 (Family 4 drop 위험), 13 (ranking 능력 한계), 14 (greedy local-optimum 위험), 15 (binormal_scale 의미) 신규.
  - **모든 spec 외 v2/v2.1/v2.2 의 main lever / oracle target / Variant A baseline / LB 0 회 carry-over / Option A schema 확장 의도는 유지**.
- v2.2 (2026-05-12): **CandidateSpec schema 확장 (Option A) — 새 family 의 selector 학습 신호 회수**.
  - **이유**: 사용자 질문 — "후보를 이런식으로 추가해도 gru-attention 이 학습할 수 있어?" 가 plan-004 의 `cand_feat` 구조 (par/perp/dist + spec + ctx + interactions) 의 *spec/interactions* term 이 `CandidateSpec` schema 에 종속되어 새 family 가 spec=0 default 시 selector 학습 신호 부분 손실하는 critical gap 식별.
  - **Option A 채택**: `src/pb_0_6822/selector.py` 의 schema/feature 함수 partial 수정 (arch/학습 로직 *영역 외*).
  - **§2.1 In-scope**: CandidateSpec schema 확장 (4 신규 fields) 추가.
  - **§2.2 Out-of-scope**: selector.py 의 arch/학습 로직 영역만 lock-in 유지 (schema 확장은 허용).
  - **§0.5 Plan-specific paths**: selector.py 의 partial 수정 영역 whitelist 명시.
  - **§5.2.0 신설**: `CandidateSpec` dataclass + `candidate_spec_features` + `make_candidate_features` 의 interactions term 3 부분 partial 수정 spec.
  - **§5.2.1 / §5.2.2 / §5.2.3**: 6 family 의 `CandidateSpec` 정의 + `candidates_extended.py` 통합 함수.
  - **§6.1 selector_retrain.py**: `selector.CANDIDATES = EXTENDED_CANDIDATES` monkey-patch + `make_candidates` 좌표 생성 wrapper.
  - **§6.1.5 신설**: schema_v22_residue severe + 학습 진입 전 assert 검증.
  - **§0.5 severe**: `schema_v22_residue` 추가.
  - **§N+3 caveats**: #2 갱신 (Option A 로 해소 명시), #11 신설 (backward-compat 위험).
  - **모든 spec 외 v2/v2.1 의 main lever / oracle target / Variant A baseline / LB 0 회 carry-over 의도는 유지**.

---

## §N+5. 참조

- `plans/plan-004-pb-0-6822-fullrun.md` (PB framework + LB 0.6806)
- `plans/plan-005-pb-0-6822-diagnostic.md` (corrector_oracle_gain + per-band + Variant A LB carry-over)
- `plans/plan-006-minimal-variant-e-lb.md` (단일 공식 0.6491 + worst regime per-regime)
- `plans/plan-007-formula-tuning.md` (단일 공식 CMA-ES ceiling 0.6482 + family 구조 한계)
- `analysis/plan-005/corrector_decomp.{json,md}` (per-band hit)
- `analysis/plan-005/oracle_summary.{json,md}` (raw oracle 0.7188, corrector_oracle_gain)
- `analysis/plan-005/selector_decomp.json` (top-1 ranking 12.6%, margin_hist, family_selection_rate)
- `analysis/plan-005/results.md` (**STAGE 6 — Variant A LB 0.6796, regime LB marginal +0.001 noise** ← 본 plan baseline 정당성)
- `analysis/plan-005/variants_ab_lb.py` (Variant A path 코드 reference)
- `analysis/plan-005/lb_log.md` (Variant A/B LB 박제)
- `analysis/plan-007/local_optimum_diagnostic.md` (CMA-ES local min 기각)
- `analysis/plan-006/variant_e_oof.json` (single-formula 0.6491)
- `notes/PB_0.6822 코드공유.ipynb` (원본 framework)
- `WORKFLOW.md` §0.5, §11, §12 convention
- `CLAUDE.md` (autonomous execution policy)
- `src/pb_0_6822/{selector,boundary}.py` (plan-004 lock-in, import only — make_candidates / SELECTOR_MAIN / BOUNDARY_MAIN)
- `src/submit.py` (dacon-submit infra)
