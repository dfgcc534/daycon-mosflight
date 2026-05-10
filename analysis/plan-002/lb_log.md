# plan-002 LB submission log

대회 ID: 236716 (DACON muflight). team: 나행. budget: 5/일. 본 plan 사용: 4 슬롯 (1 contingency 예비).

## 제출 기록

| order | exp_id | submitted_at (KST) | submission_filename | api_response | lb_score |
|---|---|---|---|---|---|
| 1 | S004_smoothing-spline-tuned   | 2026-05-10T05:09 | runs/baseline/S004_smoothing-spline-tuned/submission.csv   | `{isSubmitted: True, detail: Success}` | TBD |
| 2 | S003_cspline-window-grid      | 2026-05-10T05:11 | runs/baseline/S003_cspline-window-grid/submission.csv      | `{isSubmitted: True, detail: Success}` | TBD |
| 3 | S001_cspline-natural-full     | 2026-05-10T05:11 | runs/baseline/S001_cspline-natural-full/submission.csv     | `{isSubmitted: True, detail: Success}` | TBD |
| 4 | S002_cspline-notaknot-full    | 2026-05-10T05:12 | runs/baseline/S002_cspline-notaknot-full/submission.csv    | `{isSubmitted: True, detail: Success}` | TBD |

## 제출 순서 근거 (plan §8.2 유망도 순)

1. **S004** (smoothing tuned): plan-001 에 부재한 noise-absorbing bias. CV ↔ LB 격차 회수 가치 최대.
2. **S003** (per-axis grid): per-axis 동역학 차이 LB 보상 검증.
3. **S001** (natural full): flat extrapolation prior 의 LB 가치.
4. **S002** (not-a-knot full): aggressive extrapolation prior. S001/S002 LB 격차 자체가 신호.

## Budget 운영

- 본 plan 사용: 4 슬롯 (S004, S003, S001, S002).
- contingency 예비: 1 슬롯 (재제출 / 다른 plan 긴급 ablation).
- 모든 제출은 같은 일자 (2026-05-10 KST). carry-over 없음.

## 점수 회수 carry-over

`dacon_submit_api` 가 post-only API (점수 조회 미지원). LB 점수는 dacon.io 대회 페이지에서 사용자가 직접 확인 후 server agent 에 전달 필요. 4 점수 회수까지 plan-002 status = `partial`. 점수 도착 시 본 lb_log.md `lb_score` 칸 + registry `notes` + `plans/plan-002-cubic-spline.results.md` frontmatter `lb_scores` 동시 갱신 → status `all_complete`.

decision-note: data-partial — 4 submission isSubmitted=True 확인됨. LB 점수 회수만 carry-over. 본 carry-over 박제로 §8.4 의 `lb_unsubmitted` severe trigger 회피 (carry-over 사유 박제됨).
