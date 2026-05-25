---
plan_id: {lane}-{NNN}     # lane 형식 (예: a-001). legacy plan 은 NNN 단독. WORKFLOW.md §4 참조.
version: 1
date: YYYY-MM-DD (Asia/Seoul)
status: draft
based_on:
  - (선행 plan 또는 exp_id 목록)
scope: (selection-only / full-stack / 등)
# ── project-specific keys (옵션, 필요시 활성화) ──
# subset:
#   {project schema 정의, 예: coins / timeframes / folds / cells_per_config}
# impl_path: {project value, 예: self_reimpl / vendor_call}
---

# plan-{lane}-{NNN} — (제목)

## §0. 한 줄 목적

> **(plan 의도를 한 줄로 박제. autonomous loop 가 의도 위배 판단의 anchor 로 사용.)**

---

## §0.5 Quick Reference (autonomous loop 매 turn 읽는 section)

> *WORKFLOW.md §12 의 autonomous execution protocol 이 매 turn 읽는 section. plan 본문 다른 section 은 *필요 시* 만 부분 read. 본 §0.5 만이 *self-updating log* 역할 — agent 가 매 commit 후 [TODO]→[DONE] 만 update.*

### G-gates (commit 단위 milestone)

- G0: STAGE 0 thesis + pre-reg                       [TODO]
- G1: STAGE 1 final commit (= STAGE1_winners.md)     [TODO]
- ...
- G_final: results.md 완료                            [TODO]

### commit chain (next-up)

- c1: STAGE 0 thesis + pre-reg              spec @ §1, §3        [TODO]
- c2: (다음 commit)                          spec @ §X.Y          [TODO]
- c3: ...                                    spec @ §X.Y          [TODO]
- ...
- c_final: results.md 작성                  spec @ §N+2           [TODO]

### plan-specific severe (WORKFLOW.md §12.3 default 위 추가분)

- (plan 별 추가 severe trigger 가 있으면 명시. 없으면 "(없음, default 만)")

### plan-specific paths (WORKFLOW.md §12.5/§12.6 default 위 추가/제외)

- whitelist 추가: (plan 별 데이터 fetch 경로 등; 없으면 "(없음)")
- blacklist 추가: (plan 별 보호 영역; 없으면 "(없음)")

### Decision-note 사용 예 (자율 결정 시 commit msg 박제)

- `decision-note: spec-default — library default 채택 (예: sklearn LogisticRegression C=1.0)`
- `decision-note: data-partial — 1 day 누락, N days 중 N-1 사용 (정합 검증 PASS)`

---

## §1. (배경 / 이전 plan 인계)

...

---

## §2. Scope (명시적)

### §2.1 In-scope

| 항목 | 값 |
|---|---|
| ... | ... |

### §2.2 Out-of-scope (절대 안 함)

| 항목 | 이유 |
|---|---|
| ... | ... |

---

## §3. 사전 등록 (Pre-registration)

### §3.1 Fold split (data 분할 패턴이 있는 project 만)

(train / select / holdout)

### §3.2 합격 기준

(조건 A~D 정량 정의)

### §3.3 평가 점수 / median 집계

...

---

## §4~§N. STAGE 0 ~ N

각 STAGE 의 변경 부품 / 후보 / mini-grid / 작업량 회계 (cell / task / unit) / 산출물 / 종료 조건.

---

## §N+1. 작업량 총 회계 (cell / task / unit 등)

...

---

## §N+2. results.md 필수 항목

...

---

## §N+3. 통계 함정 & caveats

...

---

## §N+4. 변경 이력

- v1 (YYYY-MM-DD): 초안

---

## §N+5. 참조

- (선행 plan / WORKFLOW.md / 외부 데이터 source 등)
