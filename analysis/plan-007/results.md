---
plan_id: 007
based_on:
  - 004
  - 005
  - 006
finished_at: 2026-05-12T16:38:00+09:00
status: complete
exp_ids_completed:
  - F001_formula-ga
  - F002_formula-mlp
lb_exp_id: F001_formula-ga-step3
lb_score: 0.6598
lb_submitted_at: 2026-05-12T16:34:30+09:00
lb_recovered_at: 2026-05-12T17:00:00+09:00
---

# plan-007 결과 — Single-Formula CMA-ES + Basis Ablation + Per-Sample MLP

## 요약 (한 문단)

plan-006 의 단일 공식 baseline (0.6491 argmax-corrected OOF) 위에 4 단계 progression (sliding window validity → CMA-ES baseline → basis ablation → MLP coefficient regression) 을 적용해 단일 공식 framework 의 *데이터 driven* ceiling 을 측정. 최종 OOF hit = **0.6482** (Step 4 MLP per-sample), plan-006 baseline 과 거의 동급 (-0.09pp). H4 가설 ("per-sample MLP 가 단일 공식 ceiling 돌파") 는 +0.0095pp 향상으로 *technically* 통과했으나 시나리오 A 임계 (+0.010) 미달 — **시나리오 B** (단일 공식 framework 한계 인정, plan-008 에서 multi-formula 또는 corrector 재설계 필요) 결론.

## §1. Step 별 결과

### Step 1 (G0): sliding window validity

| 측정 | 값 |
|---|---|
| n_original (end_idx=10, horizon=2) | 10,000 |
| n_sliding (end_idx ∈ [5,8], horizon=2, target=train_x[:,end+2]) | 40,000 |
| KS p-value | 0.000000 (statistic=0.045150) |
| Quantile-by-quantile RMSE | 0.001252 m |
| Threshold 통과 | quantile RMSE < 0.0015 → ✓ (OR 조건 만족) |
| **aug_usable** | **TRUE** → Step 2~4 가 50K pool (sliding 40K ∪ original 10K) 사용 |

**해석**: KS strict fail 은 large-N 민감도 (N=10K vs 40K) 의 인공적 효과. quantile 기준 분포는 1.25mm 이내 정합 — sliding window aug 가 *aggregate* 분포 보존. caveat §N+3 #1 의 "조건부 분포 미검증" 한계 적용.

### Step 2 (G1): CMA-ES baseline (6 motion terms)

| 측정 | 값 |
|---|---|
| train pool | 50K (sliding 40K + original 10K) |
| single_fit_best_hit (50K in-sample) | 0.6342 |
| **oof_hit_5fold (G1 metric)** | **0.6403** (∈ [0.62, 0.78] ✓) |
| convergence_last_50_range | 0.000160 (< 0.005 ✓) |
| best_params (a, b, c, d, e, f) | [1.502, 1.949, 0.172, 0.423, 0.077, 0.027] |
| baseline x0 (plan-006 frenet_par120_perp_neg020) | [1.98, 1.20, -0.20, 0, 0, 0], fitness=0.6320 |
| **LB** | **0.657** (c5.1 closed, 사용자 회수) |
| Elapsed | ~1 분 (plan 예산 30 분 대비 30x 빠름) |

**해석**: CMA-ES 가 6 자유도 fit 으로 plan-006 raw baseline 0.6320 → 0.6403 (+0.83pp). best_params 가 plan-006 x0 와 *큰 차이* (특히 d=0.42 양수의 d2 항) — 6 자유도 fit 이 plan-006 의 3 자유도 (a, b, c) 가 놓친 acceleration history (d2, jerk) contribution 흡수.

### Step 3 (G2): basis ablation (4 new variables × cumulative)

| step | added | best_hit | marginal_gain | kept? |
|---|---|---|---|---|
| 1 | `speed_slope_d1` | 0.6356 | +0.0014 | ✓ kept |
| 2 | `rotation_term` | 0.6387 | +0.0031 | ✓ kept |
| 3 | `speed_norm_acc_par` | 0.6391 | +0.0004 | ✗ dropped |
| 4 | `v_mean3_minus_d1` | 0.6387 | +0.0000 | ✗ dropped |

| 측정 | 값 |
|---|---|
| **best_basis_vars** | `[d1, acc_par, acc_perp, d2, jerk, ts_term, speed_slope_d1, rotation_term]` (8) |
| **best_basis_hit** | **0.6387** (vs stage2_best_hit 0.6342, +0.0045 누적 향상) |
| **LB** | **0.6598** (c8.1 closed, 사용자 회수, 본 plan 최종) |
| Elapsed | ~3 분 |

**해석 + H3 검증**:
- H3 가설 = "새 변수의 marginal contribution 양수" → 2/4 변수만 유의미 (marginal_gain ≥ 0.001).
- `rotation_term` (+0.31pp) 이 최대 contributor — plan-006 regime 16/17 의 trig 비선형 hypothesis 부분 확인. 모기 비행의 회전 곡선이 단일 공식 raw 가 놓치는 nontrivial 신호.
- `speed_slope_d1` (+0.14pp) 은 marginal (just over 0.001 threshold). speed 변동 cross-term 효과 약함.
- `speed_norm_acc_par`, `v_mean3_minus_d1` 은 (+0.04pp, +0.00pp) drop — *redundant* (이전 변수와 정보 겹침) 또는 LB-irrelevant 신호.
- 단일 공식 ceiling: stage2 0.6342 → stage3 0.6387 (+0.45pp). plan-006 corrected baseline 0.6491 대비 여전히 -1.04pp.

### Step 4 (G3): per-sample MLP coefficient regression

| 측정 | 값 |
|---|---|
| arch | Linear(13→32) + SiLU + Linear(32→8), ~300 params, bias init = stage3_best_params |
| feat_dim | 13 (pos_mean·std·range 9 + speed_mean·std·max·last 4) |
| n_coeffs | 8 (= len(best_basis_vars)) |
| fold loop | model + optimizer fresh reinit per fold (OOF leakage X) |
| train_loader | 다른 4 folds 의 *모든* view (sliding + original); val = remaining fold 의 *original end_idx=10 only* |
| **oof_hit (G3 metric)** | **0.6482** (≥ 0.6437 threshold ✓) |
| oof_gain_vs_stage3 | +0.0095 (G3 PASS, but < 0.010 scenario A threshold → **시나리오 B**) |
| fold best_val_hit | [0.6619, 0.6453, 0.6481, 0.6500, 0.6355] (range 2.6pp) |
| Elapsed | 36 sec (cuda 2.8.0+cu128) |
| LB | **미제출** (§7.5, plan-008 후보로 carry-over) |

**해석 + H4 검증**:
- H4 가설 = "per-sample MLP 가 global 단일 공식 ceiling 돌파" → +0.0095pp 향상 — G3 통과 (+0.005 minimum), 그러나 *marginal* 수준 (시나리오 A 의 +0.010 미달).
- fold 간 편차 큼 (range 2.6pp) — train data fold 의 hetero-quality 영향.
- **단일 공식 framework 의 측정 ceiling = 0.6482** — plan-006 의 argmax(corrected) 0.6491 와 거의 동급 (-0.09pp 차이).
- 결론: 단일 공식 + per-sample MLP 의 ceiling = plan-006 baseline 과 *유사*. 새로운 ceiling 돌파는 *없음* (per-sample 적응이 보여준 +0.0095 향상은 noise 위지만 *주력 무기* 수준 아님).

## §2. 단일 공식 cumulative trajectory

| stage | metric | value | Δ from previous |
|---|---|---|---|
| plan-006 baseline (argmax + corrector) | OOF hit | 0.6491 | — |
| Step 2 CMA-ES (6 vars, 5-fold OOF) | oof_hit_5fold | 0.6403 | -0.0088 vs plan-006 |
| Step 3 best basis (8 vars, single fit on 50K) | best_basis_hit | 0.6387 | -0.0016 vs Step 2 single fit |
| Step 4 per-sample MLP (5-fold OOF) | oof_hit | 0.6482 | +0.0095 vs Step 3 single |

**중요 caveat**: Step 2 G1 metric 은 *5-fold OOF*, Step 3 은 *50K single fit in-sample hit* (plan §6.2 의 stage3_ablation 함수가 반환하는 best_basis_hit). 두 단계의 measurement plan 이 *다름* — 직접 비교 시 노이즈 floor 차이. Step 4 의 OOF (+0.0095 vs stage3 single) 가 *공정한* 비교는 아니나 G3 metric 의 정의 (plan §3.2) 가 그렇게 박제되어 있어 spec 그대로 적용. 실제 plan-006 0.6491 대비 ceiling 측정값 = Step 4 oof 0.6482 가 가장 신뢰성 있는 endpoint.

## §3. LB 제출

| 시점 | exp_id | step | isSubmitted | lb_score | detail |
|---|---|---|---|---|---|
| 2026-05-12T16:31:35+09:00 | F001_formula-ga-step2 | 2 | true | **0.657** | 사용자 회수 (c5.1 closed) |
| 2026-05-12T16:34:30+09:00 | F001_formula-ga-step3 | 3 | true | **0.6598** | 사용자 회수 (c8.1 closed) — 본 plan 최종 LB |
| (미제출) | F002_formula-mlp | 4 | — | — | plan §7.5 — plan-008 또는 carry-over |

3 파일 frontmatter `lb_score = 0.6598` (Step 3 최종 LB).

### §3.1 OOF ↔ LB 정합

| stage | OOF | LB | gap |
|---|---|---|---|
| plan-006 (argmax+corrector) | 0.6491 | 0.6822 | +0.0331 |
| Step 2 (CMA-ES 6 vars) | 0.6403 | 0.6570 | +0.0167 |
| Step 3 (best basis 8 vars) | 0.6387 | **0.6598** | +0.0211 |

- Step 3 > Step 2 LB 차이 +0.28pp — basis ablation 의 +0.31pp marginal gain (rotation_term) 이 LB 상으로도 sign 유지.
- 두 LB 모두 plan-006 0.6822 보다 -2.24~2.52pp 하락 — 단일 공식 raw framework 의 LB 손해 확정. plan-006 의 corrector+argmax-ensemble effect 가 단일 공식보다 LB 에서 강함.
- OOF-LB gap = 단일 공식 (0.017~0.021) < plan-006 (0.033) → 단일 공식 LB 는 OOF 와 상대적으로 더 align (corrector leak amplification 없음).

## §4. plan-008 후보

scenario B 분기 (Step 4 OOF +0.0095 < +0.010) 적용. `next_plan_candidates.md` 별도 파일.

## §5. decision-note 박제 list

(git log --grep "decision-note" --oneline 로 사후 audit 가능)

1. **c2/G0**: KS p<0.075 strict fail 이지만 quantile RMSE<0.0015 통과로 OR 조건 만족 (§4.3 spec). large-N 민감도 caveat §N+3 #1 박제.
2. **c3**: cma-4.4.4 자동 install (autonomous policy §12.3 dep-install). GMS = original 10K 의 d1 norm 평균 = 0.025574 m.
3. **c4**: single fit tolfun=1e-5 로 100 gen 조기 종료 (plan 예산 200 gen 절반). convergence_last_50_range=0.000160 << 0.005 → 수렴 충분, cma_es_no_convergence 부재.
4. **c5/c8**: DACON 응답 lb_score 미회수 (plan-006 패턴 답습). carry-over c5.1/c8.1 예약.
5. **c6**: rotation_term의 cross(d2,d1)[2] inlined 2D rotation matrix (cos·xy - sin·xy + z 보존). speed_slope_d1 의 normalize mean_speed 분모 eps=1e-9 clamp.
6. **c7**: marginal_gain ≥ 0.001 inclusive boundary (§3.2 / §6.4). speed_norm_acc_par 의 +0.0004 < 0.001 strict drop. cumulative ordering (②①④③) = plan §0.5 L105 decision-note 그대로 적용.
7. **c9/c10**: plan §7.2 의 train_loader composition ("4 folds 의 모든 view, leakage X via parent fold inheritance") 구현. val = original end_idx=10 only. fold 별 model+optimizer fresh reinit.
8. **c10**: Step 4 LB 미제출 (plan §7.5). 결과 시나리오 B branch (+0.0095 < +0.010).
