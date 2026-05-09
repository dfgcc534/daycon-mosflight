# CLAUDE.md

## Autonomous Execution Policy

이 프로젝트의 모든 `plans/plan-NNN-*.md` 는 **완전 자동 실행** 권한 부여:

- plan 전체를 G_final 까지 *자동 끝까지* 진행. 중간 사용자 permission 불필요.
- 디테일 결정 모호 시 → *권장 default* 채택 후 진행 (ask 금지).
- Plan 의도 = `§0 한 줄 목적` + `§0.5 Quick Reference` + `합격 기준`. 의도 위배 아닌 모든 결정은 자율.
- 자율 결정 시 commit msg 마지막에 `decision-note:` 1줄 박제 (사후 audit 용).

## 매 turn 시작 시 Read 시퀀스

1. `WORKFLOW.md §12` (Autonomous Execution Protocol)
2. 현재 `plans/plan-NNN-*.md` 의 `§0.5 Quick Reference`
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
- `WORKFLOW.md §11.핸드오프 정책` 의 *server agent 의미 단위 commit·push 자율 권한* 도 유효.
- 본 §Autonomous Execution Policy 는 위 정책의 *실행 확장* 이지 대체 아님.
