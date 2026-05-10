---
plan_id: 002
finished_at: 2026-05-10T14:11+09:00
status: all_complete
exp_ids_completed:
  - S001_cspline-natural-full
  - S002_cspline-notaknot-full
  - S003_cspline-window-grid
  - S004_smoothing-spline-tuned
exp_ids_skipped: []
best_exp_id_cv: S003_cspline-window-grid
best_exp_id_lb: S001_cspline-natural-full
submission_paths:
  - runs/baseline/S001_cspline-natural-full/submission.csv
  - runs/baseline/S002_cspline-notaknot-full/submission.csv
  - runs/baseline/S003_cspline-window-grid/submission.csv
  - runs/baseline/S004_smoothing-spline-tuned/submission.csv
lb_scores:
  S001: 0.4932
  S002: 0.1204
  S003: 0.4926
  S004: 0.2178
lb_metric: hit_rate (반경 비공개; plan-001 LB 0.60 = B001, 본 plan 4 변형 모두 그 아래)
lb_submitted_at_first: 2026-05-10T05:09+09:00
lb_submitted_at_last:  2026-05-10T05:12+09:00
lb_collected_at: 2026-05-10T14:11+09:00
cv_lb_spearman_rho: 0.90  # n=5 (B001 + S001~S004), 강한 비례 (S001/S003 만 미세 swap, ΔLB=0.0006)
---

# plan-002 results — Cubic spline interpolation baseline

본 frontmatter 는 WORKFLOW.md §6 의 plan results 의무 요소. 본문 분석은
`analysis/plan-002/results.md` 에 박제.

## per-experiment 요약 (status / cv / 핵심 metric)

| exp_id | status | started_at | duration_sec | cv_mean_eucl ± std | per-axis MAE | hit@0.10 | best run dir | baseline diff vs B001 (mean Δ) | sign 일관성 | lb_score (hit_rate) | 특이사항 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| S001_cspline-natural-full   | complete | 2026-05-10T04:42 | 5.4   | 0.01742 ± 0.00071 | [0.0096, 0.0096, 0.0066] | 0.9842 | runs/baseline/S001_cspline-natural-full   | +0.00448 | 1.00 | 0.4932 | full-window natural BC; H1 confirmed; **본 plan LB 1위** |
| S002_cspline-notaknot-full  | complete | 2026-05-10T04:42 | 5.3   | 0.05370 ± 0.00282 | [0.0277, 0.0288, 0.0235] | 0.8815 | runs/baseline/S002_cspline-notaknot-full  | +0.04076 | 1.00 | 0.1204 | aggressive 외삽; CV·LB 모두 worst |
| S003_cspline-window-grid    | complete | 2026-05-10T04:43 | 226.8 | 0.01740 ± 0.00071 | [0.0096, 0.0096, 0.0066] | 0.9842 | runs/baseline/S003_cspline-window-grid    | +0.00446 | 1.00 | 0.4926 | chosen=[(5,nat),(5,nat),(4,nat)]; CV-best, LB 는 S001 과 0.0006 tie |
| S004_smoothing-spline-tuned | complete | 2026-05-10T04:48 | 17.1  | 0.03322 ± 0.00270 | [0.0191, 0.0176, 0.0115] | 0.9506 | runs/baseline/S004_smoothing-spline-tuned | +0.02027 | 1.00 | 0.2178 | s=[1e-4,1e-4,1e-4]; H3 CV·LB 모두 refuted |

CV-best = **S003** (cv_mean_eucl 0.01740; B001 floor 0.01294 미달, +0.00446 worse, sign 일관성 100%).
LB-best (S001~S004 중) = **S001** (hit_rate 0.4932; S003 0.4926 와 ΔLB=0.0006 = noise tie). B001 LB 0.60 미달.

## H1/H2/H3 verdict (CV + LB 모두 회수)

- H1 (full-window 보간 ≥ B001): **CV·LB 모두 confirmed**. CV: natural 0.0174 / not-a-knot 0.0537. LB: 0.4932 / 0.1204 — 둘 다 B001 LB 0.60 미달.
- H2 (windowed grid ≈ B001): **CV·LB 모두 partially refuted**. CV 0.0174 영역 (S001 동급), LB 0.4926 (S001 과 tie). small-window natural 이 dominate 하지만 B001 LB·CV 모두 미달.
- H3 (smoothing 이 B001 위협): **CV·LB 모두 refuted**. CV 0.0332 / LB 0.2178 — fit 영역 axis MAE 의 이득이 외삽 mean_eucl 에도, LB hit_rate tail 에도 transfer 안 됨 (가장 강한 refute).

## best 선택 사유

- best_exp_id_cv = S003: cv_mean_eucl 0.01740 (S001=0.01742 보다 1e-5 작음, tie-break 작은-window).
- best_exp_id_lb = S001: LB 0.4932 (S003=0.4926 보다 0.0006 큼; CV-LB rank 미세 swap, 단 noise 영역 tie 로 해석).

## CV-LB 상관 분석 (점수 회수 완료)

n=5 (B001 + S001~S004) Spearman ρ(CV ↑ ↔ LB ↓) = **+0.90** (강한 비례). 자세히는 `analysis/plan-002/results.md §7` 박제.

| exp | CV mean_eucl | LB hit_rate | CV rank | LB rank |
|---|---|---|---|---|
| B001 | 0.01294 | 0.60   | 1 | 1 |
| S003 | 0.01740 | 0.4926 | 2 | 3 |
| S001 | 0.01742 | 0.4932 | 3 | 2 |
| S004 | 0.03322 | 0.2178 | 4 | 4 |
| S002 | 0.05370 | 0.1204 | 5 | 5 |

**핵심**: CV 가 LB 의 신뢰성 있는 proxy. 향후 plan 은 CV 만으로 우선순위 결정 가능 (LB 슬롯 절약).

## submission 결과

- 4 csv 모두 sample_submission 스키마 100 % 일치 (rows=10000, dtype float64, NaN/Inf 0).
- LB API 응답 4건 모두 `{isSubmitted: True, detail: Success}`.
- 4 LB 점수 회수 완료 (2026-05-10T14:11 KST 사용자 dacon.io 확인 후 전달).
- Budget 4/5 사용. 1 contingency 슬롯 (미사용).

## 다음 plan 후보 (enumeration only — 우선순위 X)

1. Kalman / Savitzky-Golay 입력 평활 + polyfit
2. velocity model — t=0 instantaneous derivative 등속 외삽 (B001 일반화)
3. ensemble (B001, S003, S004)
4. neural seq2seq (LSTM / Transformer)
5. per-axis combo of {polyfit, cspline, smoothing}
6. hit-radius probing (별도 plan)
