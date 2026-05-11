---
plan_id: 004
exp_id: P001_pb-0-6822-fullrun
lb_exp_id: P001_pb-0-6822-fullrun
lb_score: 0.6806
lb_submitted_at: 2026-05-11T20:10:12+09:00
status: all_complete
date: 2026-05-11 (Asia/Seoul)
---

# plan-004 results — PB_0.6822 Notebook Full-Fit + 18×27 Regime Distribution Audit

LB carry-over closed: dacon-submit 응답 `{isSubmitted: True, detail: 'Success'}` → 사용자 회수 LB = **0.6806** (commit c11.1, 2026-05-11).
노트북 작성자 주장 0.6822 와 gap = 0.0016 (≈ 재현 가능 범위). 기존 best LB (B001 0.60) 대비 +0.0806 lift.

## §1. Exp summary

| field | value |
|---|---|
| exp_id | `P001_pb-0-6822-fullrun` |
| plan_id | 004 |
| based_on | notes/PB_0.6822 코드공유.ipynb (cell 4 selector + cell 6 boundary) |
| compute | server `cuda:1` (L40S) |
| wall_time | selector 286.4s + boundary 259.5s ≈ 9 min |
| seed | selector=20260506 / boundary=20260606 |

## §2. CV / OOF metrics

### §2.1 Selector (5-fold OOF, 10000 rows)

| metric | value |
|---|---|
| `oof_tcn_gru_ensemble_soft.metrics.hit` | 0.6511 |
| `oof_tcn_gru_ensemble_argmax_soft_gate.metrics.hit` | 0.6511 |
| `oof_tcn_gru_ensemble_argmax.hit` | 0.6488 |
| `oof_tcn_gru_ensemble_argmax.mean` | 0.011996 m |
| `covered_rows` | 10000 |
| `full_pre_epochs / full_fine_epochs` | 14 / 8 |

(source: `runs/baseline/P001_pb-0-6822-fullrun/tcn_gru_selector_report.json`)

### §2.2 Boundary corrector (fold 0 val partition, 2020 rows)

| metric | value |
|---|---|
| `soft.metrics.hit` | **0.6718** |
| `gate.metrics.hit` | **0.6748** |
| `argmax.hit` | 0.6723 |
| `candidate_oracle.hit` | 0.7277 (upper bound) |
| `soft.metrics.mean` | 0.011707 m |

**Selector baseline (same fold 0 val partition)**:
- selector OOF soft hit = 0.6624 (from `BASELINE` log line, temperature=0.07)
- Corrector lift: +0.0094 absolute (= 0.6718 − 0.6624)

`corrector_no_convergence` severe trigger 미발생 (0.6718 > 0.6624).

(source: `runs/baseline/P001_pb-0-6822-fullrun/boundary_tiny_correction_report.json`)

## §3. 18×27 Regime distribution audit (G3.5)

| metric | value |
|---|---|
| `n_total` | 10000 |
| `regime_count` | 18 |
| `candidate_count` | 27 |
| min regime sample count | **274** (regime 5) |
| max regime sample count | 916 (regime 13) |
| `degenerate_count` (sample < 50) | **0** |
| `hyper_specialized_count` | **19** cells |

Regime histogram (in regime-id order):
`[661, 629, 663, 458, 615, 274, 544, 701, 592, 562, 546, 355, 549, 916, 476, 749, 354, 356]`

전 18 regime 이 empirical Bayes shrinkage 신뢰성 영역 (sample≥50) 안에 있음 — `regime_degenerate` warn 미발생. 19개 hyper-specialized cell 의 분포는 후속 plan 의 regime 재설계 anchor 정보.

(source: `analysis/plan-004/regime_distribution.{json,md}`)

## §4. LB

| field | value |
|---|---|
| submission file | `runs/baseline/P001_pb-0-6822-fullrun/submission.csv` (= soft csv 사본) |
| submitted_at_kst | 2026-05-11T20:10:12+09:00 |
| isSubmitted | true |
| lb_score | **0.6806** (carry-over closed; gap vs 노트북 0.6822 = 0.0016) |
| detail | Success |

(log: `analysis/plan-004/lb_log.md`)

**핵심 결과**: P001 LB 0.6806 → 기존 best (B001 0.60) 대비 **+0.0806 lift**, plan-003 R006 (0.5688) 대비 **+0.1118 lift**. 노트북 framework 의 우리 데이터 적용이 *측정 가능한 점수 신호* 가 나옴을 확정 (§0 narrative 갈래 1 검증 완료).

## §5. 다음 plan 후보 (post-G_final 분석 기반)

1. **regime 재설계** (G3.5 anchor 기반): 19개 hyper-specialized cell 의 분포를 보면 regime 14~17 (high speed × high curve) 영역에 over-specialized cell 이 집중. (speed × curvature × speed_slope) 3축 외 추가 축 (예: jerk, latency) 도입으로 regime 18 → 24~36 확장 ablation.
2. **selector epoch budget 확장**: 본 plan 은 notebook default 의 70% (pre=10/fine=8) 로 단축. early stopping wait 가 patience=4 까지 도달 못 한 fold 다수 → 더 긴 epoch 으로 OOF hit ceiling 탐색.
3. **boundary corrector multi-fold**: 본 plan 은 fold 0 val partition 만 평가. 5-fold concatenated OOF + per-fold lift 측정 시 spec §7 의 `selector_oof_hit_soft_baseline` 비교가 더 정밀해짐.
4. **R001 / B001 ensemble** 가능성: 본 plan LB 가 R001/B001 (0.5688 / 0.60) 보다 높으면 단일 winner, 낮거나 비슷하면 winning-pattern 통합 (plan-003 R006 패턴 답습) 후보.

## §6. decision-note 박제 list

본 plan 전 commit chain 의 자율 결정 (CLAUDE.md autonomous policy):

- spec-default — extraction approach (notebook cell → .py module, papermill X) — src/* convention 정합성
- spec-default — exp_id=P001_pb-0-6822-fullrun (P prefix = Public-baseline reimplementation)
- spec-default — submission=soft csv 사본 (continuous probability-weighted blend, leaderboard 친화)
- spec-default — 학습 device = cuda:1 (server agent 1번 GPU 강제, plan-003 의 cuda:0 패턴 답습)
- spec-default — c5 v2 `--smoke` interface 유지 (§0.5 "c2/c3/c5 는 [DONE] 상태로 본문 보존" 적용)
- **runtime-fix — selector.py L1215 for-epoch 루프 진입 시 `model.train()` 추가** (notebook 추출 시 cudnn RNN backward eval-mode 버그 노출 — smoke 1 epoch 통과, full 10 epoch 에서 epoch 2 backward 크래시. 1라인 fix 로 학습 가능. 노트북 환경에서 cudnn 비활성 또는 다른 PyTorch 버전 으로 우회되었을 가능성)
- spec-default — regime_degenerate threshold = 50 samples
- spec-default — hyper_specialized threshold = ratio ≥ 1.5 OR (ratio ≤ 0.5 AND hit ≥ 0.01)
- spec-default — c11 dacon-submit 자율 호출 = 사용자 승인 X (CLAUDE.md autonomous policy)
- spec-default — partial branch 처리 (isSubmitted=True, lb_score=None) — plan-003 R006 carry-over 패턴 답습

## §N+3. caveats

본 plan §N+3 항목 그대로 유효 (`plans/plan-004-pb-0-6822-fullrun.md` 참조). 추가:

- selector.py L1215 의 `model.train()` runtime-fix 는 추출 모듈에 박제됨 — 후속 plan 이 동일 cell 4 를 재추출하면 같은 버그 재발 가능. 후속 plan 의 추출 정책 명문화 권장.
- boundary corrector OOF 가 fold 0 val partition 만 — 5-fold concatenated hit 미산출 (v3 spec §7 의 `selector_oof_hit_soft_baseline`/`overall_oof_hit_soft` 키 미생성). 본 plan baseline 박제는 충분하나, v3 spec 의 corrector_no_convergence 자동 판정 logic 은 다음 plan 에서 boundary 모듈을 5-fold 모드로 확장 시 활성화.
