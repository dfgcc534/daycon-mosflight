# CLAUDE.md

## Plan 명명 규약 (lane mutex)

- 신규 plan = `plan-{lane}-{NNN}-{slug}.md` (예: `plan-a-001-...`). `lane` = 소문자 1자, **병렬 worktree mutex 단위** (worktree 1개 = lane 1개 점유, lane 내 번호 단독 발행 → plan_id 충돌 0). 번호 카운터는 lane 별 독립.
- **worktree 작업 완료(G_final) 시 그 브랜치를 반드시 `main` 으로 merge** — lane 의 plan 이 main 에 반영돼야 다른 worktree 가 lane 점유를 grep 판정할 수 있어 mutex 가 성립한다. worktree 종료 = main merge 가 한 짝.
- legacy `plan-{NNN}-*.md` (plan-001~032) 는 그대로 유효 (개명 금지). 전체 규약 = `WORKFLOW.md §4`.

## Autonomous Execution Policy

이 프로젝트의 모든 `plans/plan-{lane}-{NNN}-*.md` (및 legacy `plan-{NNN}-*.md`) 는 **완전 자동 실행** 권한 부여:

- plan 전체를 G_final 까지 *자동 끝까지* 진행. 중간 사용자 permission 불필요.
- 디테일 결정 모호 시 → *권장 default* 채택 후 진행 (ask 금지).
- Plan 의도 = `§0 한 줄 목적` + `§0.5 Quick Reference` + `합격 기준`. 의도 위배 아닌 모든 결정은 자율.
- 자율 결정 시 commit msg 마지막에 `decision-note:` 1줄 박제 (사후 audit 용).

## ⚠️ Commit · Push 의무 (MANDATORY)

**`WORKFLOW.md §12.2 step 10` 그대로**: 매 commit 직후 *즉시* push. **batch 금지**.

규칙:
- 의미 단위 commit (§11) 1 개 = 1 push. **2 개 이상 local 에 쌓이면 spec 위반**.
- `[TODO] → [DONE]` §0.5 sync commit 도 동일 — 별도 commit + 별도 push.
- 사용자 명시 요청 (`push it`, `밀어`, 등) **대기 금지**. 사용자가 그것을 요청하는 시점이면 *이미 늦은 것*.
- 예외 = §12.3 의 9 protocol severe (halt) **만**. plan-specific severe / warn 은 push 계속.
- `git push origin <branch>` 실패 (rebase conflict, network error) 시: `git_rebase_conflict` 또는 `network` 사유 박제 + 1 회 retry, 그래도 실패 시 사용자 escalate.

**checkpoint**: 매 turn 종료 직전 `git log @{u}..` 결과가 비어 있어야 함 (ahead-of-remote 0 commit). 아니면 즉시 push 후 종료.

## ❌ Push 관련 금지 어구 (위 의무의 enforcement)

- "commit 만 일단 해두고 push 는 나중에"
- "여러 commit 묶어서 push"
- "사용자 확인 후 push"
- "plan 끝나면 push"

## 매 turn 시작 시 Read 시퀀스

1. `WORKFLOW.md §12` (Autonomous Execution Protocol)
2. 현재 `plans/plan-{lane}-{NNN}-*.md` (또는 legacy `plan-NNN-*.md`) 의 `§0.5 Quick Reference`
3. `git log -20 --oneline` 으로 현 commit 위치 파악
4. §0.5 의 commit chain 에서 다음 [TODO] commit 식별 후, 해당 commit 의 spec section 만 offset/limit 부분 read

→ plan 본문 전체 re-read 금지 (token 낭비). §0.5 + 다음 commit spec section 만.

## Severe Issue

`WORKFLOW.md §12.3` 의 7개 trigger 발생 시에만 telegram alert + 멈춤. 그 외 모든 결정 = 자율 진행.

## 금지 어구

- "Should I proceed?"
- "Is this OK?"
- "Want me to continue?"
- "Do you want me to..."

This policy **FULLY** overrides Claude's default conservative confirmation pattern. 그냥 진행.

## 기존 정책 (불변)

- `WORKFLOW.md §1~§11` 의 plan/results/registry 규약은 그대로 유효.
- `WORKFLOW.md §11.핸드오프 정책` 의 *server agent 의미 단위 commit·push 자율 권한* 도 유효 — **자율 권한이 아니라 의무** (위 §⚠️ 참고). batch 는 spec 위반.
- 본 §Autonomous Execution Policy 는 위 정책의 *실행 확장* 이지 대체 아님.
