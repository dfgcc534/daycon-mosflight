# plan-002 LB submission log

대회 ID: 236716 (DACON muflight). team: 나행. budget: 5/일. 본 plan 사용: 4 슬롯 (1 contingency 예비).

## 제출 기록

| order | exp_id | submitted_at (KST) | submission_filename | api_response | lb_score |
|---|---|---|---|---|---|
| 1 | S004_smoothing-spline-tuned   | 2026-05-10T05:09 | runs/baseline/S004_smoothing-spline-tuned/submission.csv   | `{isSubmitted: True, detail: Success}` | 0.2178 |
| 2 | S003_cspline-window-grid      | 2026-05-10T05:11 | runs/baseline/S003_cspline-window-grid/submission.csv      | `{isSubmitted: True, detail: Success}` | 0.4926 |
| 3 | S001_cspline-natural-full     | 2026-05-10T05:11 | runs/baseline/S001_cspline-natural-full/submission.csv     | `{isSubmitted: True, detail: Success}` | 0.4932 |
| 4 | S002_cspline-notaknot-full    | 2026-05-10T05:12 | runs/baseline/S002_cspline-notaknot-full/submission.csv    | `{isSubmitted: True, detail: Success}` | 0.1204 |

dacon.io 대회 페이지 (236716) submission 목록 캡처 출처 (2026-05-10 14:11 KST 갱신 시점):

| dacon submission_id | exp_id | dacon_submitted_at | lb_score |
|---|---|---|---|
| 1446261 | S002_cspline-notaknot-full   | 2026-05-10 14:11:19 | 0.1204 |
| 1446260 | S001_cspline-natural-full    | 2026-05-10 14:11:11 | 0.4932 |
| 1446259 | S003_cspline-window-grid     | 2026-05-10 14:11:03 | 0.4926 |
| 1446258 | S004_smoothing-spline-tuned  | 2026-05-10 14:10:47 | 0.2178 |

(dacon 표시 시각은 server 처리 시각, dacon_submit_api 의 KST 제출 시각과 다름. lb_score 만 신뢰 source.)

## 제출 순서 근거 (plan §8.2 유망도 순)

1. **S004** (smoothing tuned): plan-001 에 부재한 noise-absorbing bias. CV ↔ LB 격차 회수 가치 최대.
2. **S003** (per-axis grid): per-axis 동역학 차이 LB 보상 검증.
3. **S001** (natural full): flat extrapolation prior 의 LB 가치.
4. **S002** (not-a-knot full): aggressive extrapolation prior. S001/S002 LB 격차 자체가 신호.

## Budget 운영

- 본 plan 사용: 4 슬롯 (S004, S003, S001, S002).
- contingency 예비: 1 슬롯 (재제출 / 다른 plan 긴급 ablation).
- 모든 제출은 같은 일자 (2026-05-10 KST). carry-over 없음.

## 점수 회수 (carry-over closed)

2026-05-10T14:11 KST 사용자가 dacon.io 대회 페이지 (236716) 에서 4 점수 확인 후 server agent 에 전달.
4 lb_score 일괄 갱신 (본 lb_log.md + frontmatter `lb_scores` + registry `notes` + analysis/plan-002/results.md §7).
plan-002 status: `partial` → `all_complete`.

decision-note: data-complete — 4 LB 점수 회수 후 carry-over 종결. §8.4 `lb_unsubmitted` severe trigger 미발동 (정상 marathon).
