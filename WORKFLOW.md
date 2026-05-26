# WORKFLOW

local과 server를 분리하고 plan → run → plan 반복으로 실험을 진행하기 위한 프로토콜.

---

## §1. 목적 / 적용 범위

**푸는 문제**
- 의도(가설/변수 선택)와 실행(코드/학습/빌드)을 환경 별로 분리한다.
- 모든 실험에 monotonic 추적성 부여 — 어떤 결과가 어떤 의도로 나왔는지 사후 재구성 가능.
- local ↔ server 핸드오프 단위를 *코드*가 아닌 *문서*로 둬 경계를 명확히 한다.

**적합** — 변수를 바꿔가며 비교 실험 반복 / 실행 환경이 무거움(GPU·대용량·긴 학습) / 중단·재개·인계 빈번.
**부적합** — 일회성 스크립트·탐색적 프로토타이핑 / 단발 작업 / plan 작성 비용 > 실행 비용.

---

## §2. 역할 분리

| 행위 | local | server |
|---|:-:|:-:|
| 의도 결정 (가설, 변수 선택) | O | X |
| 구현 (코드 작성, config 생성) | X | O |
| 실행 (학습, 평가, 빌드) | X | O |
| 산출물 기록 (registry, results) | X | O |
| 결과 분석, 다음 의사결정 | O | X |

- local은 *무엇을 왜* 정한다 — *어떻게*는 정하지 않는다.
- server는 plan 명시 범위만 수행 — 의도를 추측해 확장하지 않는다.
- 핸드오프 매개체는 VCS-synced repository 내 텍스트 산출물 한 종류뿐.

---

## §3. 핵심 객체 (artifacts)

| 객체 | 작성자 | 용도 | 수명 |
|---|---|---|---|
| Plan file | local | 한 묶음의 실험 요청서 | 영구 (이력) |
| Results file | server | plan에 대한 응답서 | 영구 (이력) |
| Experiment registry | server (도구로 갱신) | 누적 진실의 원천, 단일 파일 | 영구 (append-only) |
| Run directory | server | 실험별 산출물 컨테이너 | 영구 (텍스트), 가변 (binary) |
| Config snapshot | server (실행 시 동결) | 실행 시점 설정의 frozen 사본 | run dir과 동일 |

이 외 모든 산출물은 위 5개 중 하나에 귀속되어야 한다. 귀속처가 없으면 만들지 않는다.

---

## §4. 명명 규약

**Plan 파일**
```
plan-{lane}-{NNN}-{slug}.md            ← 요청 (local)
plan-{lane}-{NNN}-{slug}.results.md    ← 응답 (server)
```
- `lane`: 소문자 알파벳 1자 (`a`~`z`). **병렬 worktree mutex 단위** — 각 worktree는 미사용 lane 1개를 점유하고 그 lane 안의 번호 발행을 단독 소유. 다른 lane 끼리는 번호 겹쳐도 충돌 아님 (`plan-a-005` ≠ `plan-b-005`).
- `NNN`: lane **내부**에서 zero-pad 3자리, monotonic 증가. **카운터는 lane 별 독립** (전역 단일 X). gap 금지 — lane 내부에서 취소된 plan도 빈 results.md (`status: canceled`, reason 필수)로 자리 채움.
- `slug`: kebab-case 1~3 단어, 그 plan이 다루는 *질문* 또는 *변경 변수*. 모델/도구/백본 이름으로만 짓지 않는다.
- 요청과 응답은 같은 `{lane}-{NNN}-{slug}` 페어 — 1:1.

**Lane mutex 규약 (병렬 worktree 충돌 방지)**
- **사유**: 여러 worktree 동시 plan 발행 시 전역 단일 `NNN` 카운터는 race (둘 다 같은 번호 집어 충돌). lane 을 worktree 별 분리 → 번호 발행 lane-local → **lock 없이 mutex**.
- **lane 점유** = worktree 진입 시 미사용 알파벳 1개 claim. 별도 lock 파일 불요 — `ls plans/plan-{lane}-*` grep 으로 점유·다음 번호 판정.
- **G_final 자율 merge 의무**: plan `G_final` 도달 시 agent 가 *사용자 승인 없이* 그 worktree 브랜치를 `main` 으로 merge (절차 §12.10). merge 전까지 lane plan 이 `main` 에 부재 → 타 worktree 가 점유 grep 불가 → **mutex 붕괴**. 즉 *G_final = main merge* 가 한 짝. merge 후에도 번호 monotonic (같은 lane 재진입 시 다음 번호부터).
- **legacy backward-compat**: lane 없는 `plan-{NNN}-{slug}` (plan-001~032) 그대로 유효 (개명/이전 금지), lane 없는 단일 직렬 track 취급. 신규 plan 부터 lane 형식.

**Experiment ID**
```
{prefix}{NNN}_{slug}
```
- `prefix`: 프로젝트가 정의 (단일 namespace면 하나, 종류별 분리면 여러 개).
- `NNN`: zero-pad 3자리, **재사용 금지** (실패한 실험도 번호 소진). plan_id와 별개 카운터 — 한 plan에서 여러 exp_id 발행 가능.
- `slug`: 그 실험의 변경 변수 (단일 권장) 또는 데이터/loss 키워드.

**Config / Run 위치**
```
configs/{type}/{exp_id}.{ext}
runs/{type}/{exp_id}/
```
- `type`은 prefix와 무관할 수 있으나 보통 일치시킨다. 디렉토리명과 파일 stem에 같은 `{exp_id}` 토큰이 그대로 등장한다.

**4-way 토큰 일치 (불변)** — 같은 `exp_id` 토큰이 ① plan 본문 헤더 ② registry `id` 필드 ③ config 파일명 ④ run 디렉토리명 네 군데에 동일하게 등장한다. grep 한 번으로 cross-reference 성립.

---

## §5. Plan 파일 의무 요소

**Frontmatter (YAML)**
- `plan_id`: `{lane}-{NNN}` (lane 형식, 예: `a-001`). legacy plan 은 `NNN` 단독 (§4 backward-compat).
- `date`: 작성일 (timezone 명시)
- `inspired_by`: 선행 plan_id/exp_id 목록 — ★ **약한 관계** (동기·lesson·evidence 출처만, 코드 인계 의미 X). 기존 plan의 `based_on`은 backward-compat로 동등 유지, 신규는 `inspired_by`.
- `code_reuse`: 명시적 코드 carry 목록 — default `[]` (**from-scratch 권장**). carry 항목마다 `{module, symbols, reason}` 박제. 명시 없는 import = 자동 carry 금지.
- `exp_ids`: 이 plan에서 발행될 exp_id 목록

**본문 섹션 (모두 필수)**

| 섹션 | 내용 |
|---|---|
| 배경 | 어떤 결과/관찰로 이 plan을 짰는지 — 인과. 선행 plan의 *부정 evidence / lessons* 만 인계 ("plan-NNN의 X 실패 → 본 plan은 Y"). 코드 carry는 frontmatter `code_reuse`로만 (default from-scratch). |
| 가설 | 무엇을 검증/반박하려 하는지 — 명제 |
| 실험 목록 | 각 exp_id마다: type, baseline, 변경 변수(단일 권장), config 경로, 기대 runtime, 성공 기준, 실패 시 분기 |
| 서버 작업 순서 | enumerated 단계. server는 이 외 작업을 하지 않는다 |
| Out of scope | 이 plan에서 *명시적으로 안 할 것* |
| 참조 | 선행 results, 영구 결정 문서 링크 |

하나라도 빠진 문서는 plan으로 인정하지 않는다 — server는 즉시 실패 응답을 작성한다.

---

## §6. Results 파일 의무 요소

**Frontmatter (YAML)**
- `plan_id`: 짝 plan의 NNN
- `finished_at`: 완료 timestamp (timezone 명시)
- `status`: `all_complete | partial | failed | canceled`
- `exp_ids_completed`: 끝까지 수행된 exp_id 목록
- `exp_ids_skipped`: 스킵/실패한 exp_id 목록 + 사유

**본문** — 각 exp_id마다: 상태(`complete|failed|canceled`) · 실행시간(started_at, duration) · 핵심 metric · best artifact 경로(run dir 상대) · baseline 대비 config diff(key 단위 또는 unified diff) · 외부 시스템 결과(있을 때만) · 특이사항(수렴, OOM, 중단 등).

**다음 단계 후보** — server는 데이터 근거로 다음 plan 후보를 *나열*만 한다. 결정·우선순위는 local의 권한.

---

## §7. Run 디렉토리 의무 파일

```
runs/{type}/{exp_id}/
├── config.snapshot.{ext}     ← 모든 default가 resolved 된 동결본
├── summary.{json|yaml}        ← registry와 동기되는 키
├── history.{json|csv}         ← 시계열 (per-step 또는 per-epoch)
├── {stage}.log                ← stdout/stderr (단계별)
└── artifacts/                 ← 무거운 산출물 (체크포인트, 이미지) — VCS 추적 X
```
- `summary` 키는 registry 엔트리 스키마와 1:1 — 한쪽이 갱신되면 다른 쪽도 동시 갱신.
- `artifacts/` 외 모든 텍스트 파일은 VCS 추적.
- 위 구조를 만족하지 않으면 그 실험은 *완료되지 않은 것*으로 본다.

---

## §8. Lifecycle / 상태 전이

```
   (local)          (server)         (server)          (local)         (local)
plan written ─▶ in_progress ─▶ results written ─▶ analyzed ─▶ next plan written
   │
   └─ 또는 ─▶ canceled  (results: status=canceled, reason 필수)
```

| 상태 | 진입 | 종료 |
|---|---|---|
| written | local이 plan commit | server가 수신 |
| in_progress | server가 작업 시작 | results 작성 시작 |
| results written | results + run dir + registry 갱신 commit | local이 pull |
| analyzed | local이 결과 검토 | 다음 plan 작성 |
| canceled | plan이 더 이상 유효치 않음 | 빈 results.md 작성 |

전이는 단방향. results written 이후엔 그 plan을 수정하지 않는다 — 필요하면 새 plan을 발행한다.

---

## §9. 불변 규약 (invariants)

1. **ID 단조성** — plan_id, exp_id 모두 monotonic, 재사용 금지. 단 plan_id 의 monotonic 은 *lane 내부* 에서 성립 (lane 간 번호 독립 — `plan-a-005`·`plan-b-005` 공존 정상; §4 lane mutex).
2. **한 변수 원칙** — 한 exp의 config는 baseline 대비 최소 키만 변경. 변경 키 수는 results diff 섹션에 정확히 기록.
3. **Plan 자기-완결** — plan은 외부 컨텍스트(채팅·메모리)에 의존하지 않고 단독 재구성 가능.
4. **Registry append-only** — 도구로만 갱신, 직접 편집 금지. 정정도 새 행 추가 (`type: correction`, `corrects: <id>`).
5. **Plan ↔ Results 1:1** — 한 plan에 정확히 하나의 results. 응답 분할 금지.
6. **4-way 토큰 일치** — §4의 네 군데에 같은 exp_id 토큰 그대로 등장.
7. **Frozen snapshot** — 실행 직후 config snapshot 변경 금지. 원본을 수정해도 snapshot은 그대로.

---

## §10. 안티패턴 (금지)

- **역할 침범** — local이 코드 작성·실행 / server가 가설 수정·새 변수 도입 (§2 위반).
- **추적성 파괴** — plan 없이 server 자체 실험 시작 / registry 갱신 없이 다음 실험 / results 없이 plan 종료.
- **자기-완결 위반** — plan이 외부 채팅·구두 합의를 전제 / results가 plan에 없는 새 가설을 검증하고 그 결과만 기록 (§9-3).
- **단일 변수 위반** — 한 exp에서 여러 변수 동시 변경(원인 분리 불가) / 리팩터링과 변수 변경을 한 commit에 혼재 (§9-2).
- **핸드오프 우회** — 사용자 모르게 비동기 push로 상태 변경 / 한 commit에서 여러 plan·results 동시 처리.

---

## §11. 핸드오프 정책

**Sync 대상 (텍스트 metadata)** — plan·results 파일 / registry + 자동 렌더 인덱스 / config + snapshot / run dir의 summary·history·log / 코드·영구 결정 문서.

**Sync 비대상 (무거운 binary, 재현 가능)** — 체크포인트·가중치 / 학습 로그 raw 디렉토리 / 도커 이미지·번들 / 데이터셋 원본·전처리 캐시 / 대용량 manifest (path만 기록).

**트리거 / 자동화 경계**

| 방향 | 시점 | 행위 |
|---|---|---|
| local → server | plan 작성 후 | local이 명시적 push |
| server → local | results + registry 갱신 후 | server agent가 commit·push (의미 단위), local이 명시적 pull |

- server agent의 의미 단위 commit·push 는 **의무** (자율 권한 아님 — `CLAUDE.md §⚠️`). commit 단위 = plan 1 / results 1 / code change 1 분리. binary 산출물 혼재 금지.

---

## §12. Autonomous Execution Protocol

server 측 agent가 plan 1개를 G0 → G_final 까지 사용자 개입 없이 자동 실행할 때의 protocol. `CLAUDE.md` 의 Autonomous Execution Policy 와 짝.

### §12.1 Invoke
interactive (tmux/screen): `> plan-NNN 진행` / 비대화형: `claude -p "plan-NNN 진행" --permission-mode acceptEdits`

### §12.2 매 turn step 시퀀스
1. Read `CLAUDE.md` (auto-load)
2. Read `WORKFLOW.md §12` (본 절)
3. Read `plans/plan-NNN-*.md` 의 `§0.5 Quick Reference`
4. `git log -20 --oneline` 으로 현 commit 위치 파악
5. §0.5 commit chain 의 다음 [TODO] commit 식별
6. 그 commit 의 spec section 만 offset/limit 부분 read
7. `git pull --rebase` (conflict → severe)
8. 코드/테스트/문서 작성 (§12.5 whitelist 준수)
9. self-check: `pytest tests/` + backward_compat smoke + invariant smoke
10. pass → commit (decision-note 포함) + push. fail → severe alert + 멈춤
11. §0.5 의 [TODO] → [DONE] (commit hash) 1줄 update (§12.6 blacklist 의 *유일한 예외*)
11.5. **G-gate check** (현 commit이 STAGE 마지막 commit일 때만): 해당 STAGE 모든 c{i}가 §0.5에서 [DONE]인지 확인 — 누락 시 severe `stage_incomplete`.
11.6. **§0.5 ↔ git log sync** (매 commit): §0.5 [DONE] hash가 `git log --format=%H` 에 모두 존재하는지 grep — 불일치 시 severe `qr_log_mismatch`.
12. 다음 commit으로 진행 (step 5 return). G_final 도달 시 §12.10.

### §12.3 Severe Issue (오직 이 9개만 멈춤)
1. `pytest_fail` — exit_code ≠ 0
2. `backward_compat` — `tests/backward_compat/*` 중 1개 이상 fail
3. `task_failure_rate ≥ 0.30` — `summary.json` 의 `n_failed / n_total ≥ 0.30`
4. `same_commit_msg_3x` — 같은 commit msg 3 turn 연속 시도 (stuck 감지)
5. `turn_count > 30` — safety 상한
6. `git_rebase_conflict` — auto-resolve 불가
7. `path_whitelist_violation` — §12.5 외 파일 수정 시도
8. `stage_incomplete` — STAGE 마지막 commit 시점에 §0.5 [TODO] 잔여 (step 11.5)
9. `qr_log_mismatch` — §0.5 [DONE] hash가 git log에 부재 (step 11.6)

*severe 아님* (자율 진행): spec 모호 → 권장 default + `decision-note` / borderline 합격·탈락 → 기준 기계 적용 / performance suboptimal → 의도 위배 아니면 진행 / 데이터 fetch 부분 실패 → retry 후 partial drop / dependency 누락 → 자율 install / minor format·lint → 자동 수정 / 코드 style → 권장 default(PEP8 등).

### §12.4 Telegram Alert
`~/.config/telegram.env` 의 `TELEGRAM_BOT_TOKEN` + `TELEGRAM_CHAT_ID` 사용.

| 시점 | 빈도 |
|---|---|
| session 시작 | 1 |
| STAGE (G-gate) 완료 | 6~8 / plan |
| severe issue | 사건당 1 |
| G_final 도달 | 1 |

→ plan 1개 정상 실행 = ~10~15 alerts.

### §12.5 Path Whitelist (수정 가능)
`src/**/*.py` · `tests/**/*.py` · `configs/**/*.yaml` · `analysis/plan-{NNN}/**/*` (현 plan NNN) · `runs/{type}/{exp_id}/**` (§4 4-way token 패턴).

### §12.6 Path Blacklist (절대 수정 금지)
`plans/*.md` (pre-registered — 단 step 11 의 `§0.5 [TODO]→[DONE]` update만 허용) · `CLAUDE.md` · `.claude/settings.json`·`settings.local.json` · `WORKFLOW.md` · `vendor/**` · 그 외 모든 파일.
→ blacklist 변경 시도 = §12.3 #7 severe.

### §12.7 Destructive Ops 절대 금지
`git push --force`/`-f` · `git push --no-verify` · `git reset --hard` · `git checkout {ref} -- .` · `git clean -f` · `git branch -D` · 임의 path `rm -rf`.
→ 예외: `runs/{type}/{exp_id}/` 의 *직접 산출* 폐기는 OK.

### §12.8 Session Pre-flight (session 시작 1회)
1. `~/.config/telegram.env` 존재 + parse 확인
2. dummy alert 발신 (`/loop {plan-id} 시작, commit X 부터`)
3. dummy alert 실패 → abort
4. `git status` clean 확인 (uncommitted 잔여 → abort, 사용자 수동 정리 요청)
5. `git pull --rebase origin {branch}` (conflict → abort)

### §12.9 Decision Note 규약
자율 결정 commit msg 마지막에: `decision-note: <category> — <한줄 사유>`
category: `spec-default` · `lint-fix` · `dep-install` · `data-partial` · `retry-3x`.
사후 audit: `git log --grep "decision-note" --oneline`.

### §12.10 종료 정책
| 시나리오 | 조치 |
|---|---|
| G_final 도달 (lane plan) | **worktree → main 자율 merge** (사용자 승인 없이 — §4 lane lifecycle) → 자연 종료, telegram ("plan-{lane}-NNN 완료 + main merge, hash=...") |
| G_final 도달 (legacy lane 없는 plan) | 자연 종료, telegram ("plan-NNN 완료, hash=...") — auto-merge 비대상 |
| severe issue | 멈춤, telegram alert, session 유지 (사용자 결정 후 재개) |
| max_turns (>30) | severe 와 동일 |

**G_final auto-merge 절차 (sync-then-ff)** (lane plan, 사용자 confirm 불필요 — **worktree 안에서 직접 수행, `checkout main`·primary 작업본 미접근**): ① 최종 commit+push 확인 (`git log @{u}..` 비어 있음) → ② `git fetch origin && git merge origin/main` (**worktree 브랜치에서** 타 lane 변경 통합; cross-lane conflict 는 여기서 surface) → ③ `git push origin HEAD:main` (**fast-forward** — 브랜치가 이제 origin/main 을 포함하므로 ff 성립) → ④ telegram (merge hash). 박제: main 미진행 시 lane commit 이 선형 반영, main 진행 시 step② 의 `Merge origin/main into worktree-<branch>` commit 이 통합 marker (lane commit 은 `plan-{lane}-NNN cN` 메시지로 자기식별). **conflict 시** (step②): `git_rebase_conflict` 박제 + 수동 resolve→commit 후 step③ 재시도, 불가 시 멈춤+escalate — 자동 강제 merge / `-X` 전략 / squash 금지. ※ **구 `checkout main → merge --no-ff` 방식 폐기** (실측 교훈): worktree 는 primary 가 점유한 `main` 을 checkout 못 하고, 임시 detached-worktree 의 `--no-ff` merge commit 은 lane 브랜치에 안 남아 *다음 commit 에서 divergence* 를 유발. sync-then-ff 는 (a) primary 작업본 무접근, (b) conflict 를 브랜치에서 처리, (c) divergence 원천 차단.
