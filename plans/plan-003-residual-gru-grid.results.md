---
plan_id: 003
finished_at: 2026-05-11T00:15+09:00
status: partial   # carry-over: lb_score 회수 대기
exp_ids_completed:
  - R001_baseline-residual-gru
  - R002_physics-features
  - R003_ema-extrapolate
  - R004_wingbeat-oscillation
  - R005_loss-mse
  - R006_combined-winners
exp_ids_skipped: []
best_exp_id_cv_ablation: R001_baseline-residual-gru   # cv argmin among R001..R005 (R001 = 0.013383)
combined_winning_components: []                       # winning = {R002:F, R003:F, R004:F, R005:F}
combined_fallback: false                              # R006.cv == R001.cv ≤ R001.cv + 0.001
lb_exp_id: R006_combined-winners
lb_score: null                                        # carry-over pending — 사용자가 dacon.io 회수 후 박제
lb_submission_path: runs/baseline/R006_combined-winners/submission.csv
lb_metric: hit_rate_at_1cm
lb_submitted_at: 2026-05-11T00:08+09:00
train_device: cuda:0
total_train_time_sec: 268.7   # R001~R005 합산 (R006 = R001 cp, 추가 학습 0)
---

# plan-003 results — Residual GRU Lean Baseline + Component Ablation Grid + Winning-Components Combined

본 문서는 WORKFLOW.md §6 의 plan ↔ results 1:1 응답서. 본문은 `analysis/plan-003/results.md` 와 `analysis/plan-003/winning_trace.md` + `analysis/plan-003/lb_log.md` 를 참조한다.

## §1. exp 별 산출

각 exp 의 상세 (status, started_at, duration, 핵심 metric, best path, baseline diff vs B001 + vs R001, 특이사항) 는 `analysis/plan-003/results.md` §2 참조. 요약:

| exp_id | status | duration (s) | cv_mean_eucl ± std | hit@0.10 | best run dir |
|---|---|---:|---:|---:|---|
| R001_baseline-residual-gru | complete | 37.4 | 0.013383 ± 0.000718 | 0.9935 | runs/baseline/R001_baseline-residual-gru |
| R002_physics-features | complete | 60.3 | 0.015157 ± 0.000499 | 0.9929 | runs/baseline/R002_physics-features |
| R003_ema-extrapolate | complete | 83.9 | 0.014038 ± 0.000976 | 0.9936 | runs/baseline/R003_ema-extrapolate |
| R004_wingbeat-oscillation | complete | 37.7 | 0.013476 ± 0.000684 | 0.9936 | runs/baseline/R004_wingbeat-oscillation |
| R005_loss-mse | complete | 31.0 | 0.013388 ± 0.000580 | 0.9934 | runs/baseline/R005_loss-mse |
| R006_combined-winners | complete | 0 (cp) | 0.013383 ± 0.000718 | 0.9935 | runs/baseline/R006_combined-winners |

**lb_score** 는 lb_exp_id 한 행 (R006_combined-winners) 만 박제 대상. 현재 carry-over pending (dacon.io 페이지 수동 회수 필요).

## §2. baseline diff (config diff vs B001 + vs R001)

각 R00x 의 단일 변경 변수 — 모두 plan-003 §2.1 spec 준수, 1 변수 원칙 위반 0건. 상세 표는 `analysis/plan-003/results.md` §3.1, §3.2.

## §3. 외부 시스템 결과 (LB)

- 1회 자율 제출 (`dacon-submit` skill, CLAUDE.md autonomous policy + plan §0.5 박제). API 응답 `{isSubmitted: True, detail: Success}`.
- 상세: `analysis/plan-003/lb_log.md`.

## §4. 특이사항

- NaN/Inf, training divergence, OOM 0건.
- R001~R005 모든 학습 device = cuda:0 (decision-note v6).
- R006 학습 skip (winning=0 → R001 cp 분기), 별도 학습 시간 0.
- `combined_no_improvement` warn: 본 plan 미발동 (R006.cv == R001.cv).
- caveat #10 (LB 1점 으로 CV-LB ranking divergence 직접 측정 불가). 다음 plan 후보 5 LB 제출 plan 으로 회수.

## §5. 다음 단계 후보 (enumeration only — local 권한)

`analysis/plan-003/results.md` §9 참조. 12 후보 enumerated.

핵심 anchor (caveat 박제):
1. winning 기준 보수성 (Δ < 0 strict) → strict mode + R006 재학습 plan 의 의미.
2. closed-form B001 paired-floor (5/5 fold) → ensemble (R00x + B001) plan 의 의미.
3. physics jerk SNR / EMA α / Huber δ 의 *각각의 default 가 본 데이터에 mismatch* — sweep plan 들의 의미.
