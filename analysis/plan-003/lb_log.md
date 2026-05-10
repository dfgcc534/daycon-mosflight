# plan-003 LB submission log

대회 ID: 236716 (DACON muflight). team: 나행. budget: 5/일. 본 plan 사용: 1 슬롯 (의무 자율 제출).

## 제출 기록

| order | exp_id | submitted_at (KST) | submission_filename | api_response | combined_fallback | winning_trace_summary | lb_score |
|---|---|---|---|---|---|---|---|
| 1 | R006_combined-winners | 2026-05-11T00:08 | runs/baseline/R006_combined-winners/submission.csv | `{isSubmitted: True, detail: Success}` | False | winning=0 → R006 = R001 (0.013383) 비트 동일 사본 | (carry-over pending) |

## 자율 제출 사유 (decision-note)

- CLAUDE.md autonomous policy + plan §0.5: "best 1 LB 제출 (필수, skip 불가, 자율 실행)" + decision-note `c14 LB 제출 = autonomous loop 가 사용자 승인 없이 dacon-submit skill 1회 호출`.
- skill SKILL.md 의 사전 confirm 단계는 본 plan 의 autonomous 박제로 override (plan-003 §8.3, c14 spec 일치).

## lb_exp_id 결정 trace

```
R001 cv_mean_eucl = 0.013383
R002 physics      = 0.015157  Δ=+0.001775  non-winning
R003 ema          = 0.014038  Δ=+0.000656  non-winning
R004 wingbeat     = 0.013476  Δ=+0.000093  non-winning
R005 mse          = 0.013388  Δ=+0.000005  non-winning
→ winning = {R002:F, R003:F, R004:F, R005:F}, count = 0
→ R006 = R001 직접 복제 (config·ckpt·summary 모두 비트 동일, exp_id 만 R006_combined-winners)
→ R006.cv = 0.013383 = R001.cv ≤ R001.cv + 0.001 → fallback = False
→ lb_exp_id = R006_combined-winners
```

## 점수 회수 (carry-over)

DACON API 는 `post_submission_file` 만 제공 → LB 점수 자동 회수 불가.
본 plan-002 §8.2 carry-over 패턴 동일 — 사용자가 dacon.io 대회 페이지 (236716) `mysubmission` 에서 score 확인 후 server agent 에 전달, 별도 commit 으로 갱신:
- 본 lb_log.md 의 lb_score 컬럼
- registry.csv 의 R006_combined-winners 행 notes 컬럼 (`+lb=0.XX`)
- plans/plan-003-residual-gru-grid.results.md frontmatter (`lb_score`)
- analysis/plan-003/results.md 의 종합 표

회수 전: `status: partial`. 회수 후: `status: all_complete`.

## Budget 운영

- 본 plan 사용: 1 슬롯.
- 잔여: 4 슬롯 (다른 plan 또는 계약 제출용).
