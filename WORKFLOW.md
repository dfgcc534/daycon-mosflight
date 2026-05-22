# WORKFLOW

local과 server를 분리하고 plan → run → plan 반복으로 실험을 진행하기 위한 프로토콜.

---

## §1. 목적 / 적용 범위

### 푸는 문제
- 의도(가설/변수 선택)와 실행(코드/학습/빌드)을 환경 별로 분리한다.
- 모든 실험에 monotonic 추적성을 부여한다 — 사후에 어떤 결과가 어떤 의도로 나왔는지 재구성 가능하게 한다.
- local과 server 사이 핸드오프 단위를 *코드*가 아닌 *문서*로 둠으로써 경계를 명확히 한다.

### 도입할 가치 있는 경우
- 변수를 바꿔가며 비교 실험을 반복한다.
- 실행 환경이 무겁다 (GPU, 대용량 데이터, 긴 학습 시간).
- 중단·재개·인계가 빈번하다.

### 도입하지 말아야 하는 경우
- 일회성 스크립트, 탐색적 프로토타이핑.
- 실험 비교가 의미 없는 단발 작업.
- plan 작성 비용이 실행 비용보다 큰 경우.

---

## §2. 역할 분리

| 행위 | local | server |
|---|:-:|:-:|
| 의도 결정 (가설, 변수 선택) | O | X |
| 구현 (코드 작성, config 생성) | X | O |
| 실행 (학습, 평가, 빌드) | X | O |
| 산출물 기록 (registry, results) | X | O |
| 결과 분석, 다음 의사결정 | O | X |

원칙:
- local은 *무엇을 왜* 할지 정한다. *어떻게*는 정하지 않는다.
- server는 plan에 명시된 범위만 수행한다. 의도를 추측해서 확장하지 않는다.
- 핸드오프 매개체는 VCS-synced repository 내의 텍스트 산출물 한 종류뿐이다.

---

## §3. 핵심 객체 (artifacts)

| 객체 | 작성자 | 용도 | 수명 |
|---|---|---|---|
| Plan file | local | 한 묶음의 실험 요청서 | 영구 (이력) |
| Results file | server | plan에 대한 응답서 | 영구 (이력) |
| Experiment registry | server (도구로 갱신) | 누적 진실의 원천, 단일 파일 | 영구 (append-only) |
| Run directory | server | 실험별 산출물 컨테이너 | 영구 (텍스트), 가변 (binary) |
| Config snapshot | server (실행 시 동결) | 실행 시점 설정의 frozen 사본 | run dir과 동일 |

이 외의 모든 산출물은 위 5개 중 하나에 귀속되어야 한다. 귀속처가 없으면 만들지 않는다.

---

## §4. 명명 규약

### Plan 파일

```
plan-{NNN}-{slug}.md            ← 요청 (local)
plan-{NNN}-{slug}.results.md    ← 응답 (server)
```

- `NNN`: zero-pad 3자리, monotonic 증가.
- gap 금지 — 취소된 plan도 빈 results.md (`status: canceled`, reason 필수)로 자리를 채운다.
- `slug`: kebab-case, 1~3 단어. 그 plan이 다루는 *질문* 또는 *변경 변수*. 모델/도구/백본 이름으로만 짓지 않는다.
- 요청과 응답은 같은 NNN-slug 페어 — 1:1.

### Experiment ID

```
{prefix}{NNN}_{slug}
```

- `prefix`: 프로젝트가 정의 (단일 namespace면 하나, 종류별 분리면 여러 개).
- `NNN`: zero-pad 3자리, **재사용 금지**. 실패한 실험도 번호를 소진한다.
- `NNN`은 plan_id와 별개 카운터. 한 plan에서 여러 exp_id를 발행할 수 있다.
- `slug`: 그 실험의 변경 변수 (단일 권장) 또는 데이터/loss 키워드.

### Config / Run 위치

```
configs/{type}/{exp_id}.{ext}
runs/{type}/{exp_id}/
```

- `type`은 prefix와 무관할 수 있으나 보통 일치시킨다.
- 디렉토리 이름과 파일 stem에 같은 `{exp_id}` 토큰이 그대로 등장한다.

### 4-way 토큰 일치 (불변)

같은 `exp_id` 토큰이 다음 4 군데에 동일하게 등장해야 한다:

1. plan 본문에서 그 실험을 가리키는 헤더
2. registry 엔트리의 `id` 필드
3. config 파일명
4. run 디렉토리명

이 일치는 grep 한 번으로 cross-reference가 성립함을 의미한다.

---

## §5. Plan 파일 의무 요소

### Frontmatter (YAML)

- `plan_id`: NNN
- `date`: 작성일 (timezone 명시)
- `inspired_by`: 선행 plan_id 또는 exp_id 목록 (★ **약한 관계** — 동기 / lesson / evidence 출처만. 코드 인계 의미 X).
  - *기존 plan-NNN 의 `based_on` field 는 backward-compat 으로 동등 의미로 유지. 신규 plan 부터 `inspired_by` 사용.*
- `code_reuse`: 명시적 코드 carry 목록 — `[]` (default = 빈 list, **from-scratch implementation 권장**). carry 항목마다 `{module: <path>, symbols: [<fn1>, ...], reason: <한 줄>}` 형태로 박제. 명시 없는 코드 import = 자동 carry 금지.
- `exp_ids`: 이 plan에서 발행될 exp_id 목록

### 본문 섹션 (모두 필수)

| 섹션 | 내용 |
|---|---|
| 배경 | 어떤 결과/관찰을 보고 이 plan을 짰는지 — 인과. **선행 plan 의 *부정 evidence / lessons learned* 만 인계** (예: "plan-NNN 의 X 가 실패 → 본 plan 은 Y 시도"). *선행 plan 의 코드 / arch / framework 자동 carry 의무 없음* — 명시적 reuse 가 필요하면 frontmatter `code_reuse` 에 항목별 박제. default = from-scratch implementation 으로 *paradigm 재발명 자유* 확보. |
| 가설 | 무엇을 검증/반박하려 하는지 — 명제 |
| 실험 목록 | 각 exp_id마다: type, baseline, 변경 변수 (단일 권장), config 경로, 기대 runtime, 성공 기준, 실패 시 분기 |
| 서버 작업 순서 | enumerated 단계. server는 이 외 작업을 수행하지 않는다 |
| Out of scope | 이 plan에서 *명시적으로 안 할 것* |
| 참조 | 선행 results, 영구 결정 문서 등의 링크 |

위 중 하나라도 빠진 문서는 plan으로 인정하지 않는다 — server는 즉시 실패 응답을 작성한다.

---

## §6. Results 파일 의무 요소

### Frontmatter (YAML)

- `plan_id`: 짝이 되는 plan의 NNN
- `finished_at`: 완료 timestamp (timezone 명시)
- `status`: `all_complete | partial | failed | canceled`
- `exp_ids_completed`: 실제로 끝까지 수행된 exp_id 목록
- `exp_ids_skipped`: 스킵/실패한 exp_id 목록 + 사유

### 본문 섹션

각 exp_id마다 다음 필드를 포함한다:

- 상태 (`complete | failed | canceled`)
- 실행 시간 (started_at, duration)
- 핵심 metric 값
- best artifact의 경로 (run dir 기준 상대 경로)
- baseline 대비 config diff (key 단위 또는 unified diff)
- 외부 시스템 결과 (제출/평가 등이 있을 때, 없으면 생략)
- 특이사항 (수렴 양상, OOM, 중단 등)

### 다음 단계 후보

server가 본 데이터에 근거해 가능한 다음 plan 후보를 *나열*만 한다. 결정은 local의 권한 — server가 우선순위를 정하지 않는다.

---

## §7. Run 디렉토리 의무 파일

```
runs/{type}/{exp_id}/
├── config.snapshot.{ext}     ← 모든 default가 resolved 된 동결본
├── summary.{json|yaml}        ← registry와 동기되는 키
├── history.{json|csv}         ← 시계열 (per-step 또는 per-epoch)
├── {stage}.log                ← stdout/stderr (학습/평가/빌드 등 단계별)
└── artifacts/                 ← 무거운 산출물 (체크포인트, 이미지) — VCS 추적 X
```

- `summary` 파일의 키는 registry 엔트리 스키마와 1:1 대응. 둘 중 하나가 갱신되면 다른 쪽도 동시에 갱신된다.
- `artifacts/` 외의 모든 텍스트 파일은 VCS로 추적된다.
- 디렉토리가 위 구조를 만족하지 않으면 그 실험은 *완료되지 않은 것*으로 간주한다.

---

## §8. Lifecycle / 상태 전이

```
       (local)             (server)             (server)            (local)             (local)
plan written  ──▶  in_progress  ──▶  results written  ──▶  analyzed  ──▶  next plan written
   │                  │                   │                                    │
   │              [server picks up]   [VCS sync]                          [VCS sync]
   │
   └─ 또는 ─▶ canceled  (results: status=canceled, reason 필수)
```

진입/종료 조건:

| 상태 | 진입 조건 | 종료 조건 |
|---|---|---|
| written | local이 plan 파일 commit | server가 해당 plan을 수신 |
| in_progress | server가 작업 시작 | results 파일 작성 시작 |
| results written | results 파일 + run dir + registry 갱신 commit | local이 pull |
| analyzed | local이 결과 검토 | 다음 plan 작성 |
| canceled | plan이 더 이상 유효하지 않다고 판단 | 빈 results.md 작성 |

전이는 단방향. 한 번 results written 이후엔 그 plan을 수정하지 않는다 — 수정이 필요하면 새 plan을 발행한다.

---

## §9. 불변 규약 (invariants)

1. **ID 단조성** — plan_id, exp_id 모두 monotonic, 재사용 금지.
2. **한 변수 원칙** — 한 exp의 config는 baseline 대비 최소한의 키만 변경한다. 변경 키 수는 results의 diff 섹션에 정확히 기록된다.
3. **Plan 자기-완결** — plan은 외부 컨텍스트(채팅 로그, 메모리 등)에 의존하지 않고 단독으로 재구성 가능해야 한다.
4. **Registry append-only** — 도구로만 갱신, 직접 편집 금지. 정정도 새 행 추가로 표현한다 (`type: correction`, `corrects: <id>`).
5. **Plan ↔ Results 1:1** — 한 plan에는 정확히 하나의 results가 대응한다. 응답을 분할하지 않는다.
6. **4-way 토큰 일치** — §4의 4 군데에 같은 exp_id 토큰이 그대로 등장한다.
7. **Frozen snapshot** — 실행 직후 config snapshot은 변경 금지. config 원본을 수정해도 snapshot은 그대로.

---

## §10. 안티패턴 (금지)

### 역할 침범
- local이 코드 작성·실행 → server 역할 침범.
- server가 가설 수정, 새 변수 도입 → local 역할 침범.

### 추적성 파괴
- plan 없이 server가 자체 실험 시작.
- registry 갱신 없이 다음 실험 시작.
- results 없이 plan 종료.

### 자기-완결 위반
- plan에서 외부 채팅 로그/구두 합의를 전제로 한 표현.
- results에서 plan에 없는 새 가설을 검증하고 그 결과만 기록.

### 단일 변수 위반
- 한 exp에서 여러 변수를 동시에 변경해 원인 분리 불가능.
- 코드 리팩터링과 변수 변경을 한 commit에 섞기.

### 핸드오프 우회
- agent의 비동기 push로 사용자 모르게 상태 변경.
- 한 commit에서 여러 plan/results를 동시에 처리.

---

## §11. 핸드오프 정책

### Sync 대상 (텍스트 metadata)

- plan 파일, results 파일
- registry, registry로부터 자동 렌더된 인덱스
- config 파일과 snapshot
- run dir의 summary / history / log
- 코드, 영구 결정 문서

### Sync 비대상 (무거운 binary, 재현 가능 산출물)

- 체크포인트, 모델 가중치
- 학습 로그 도구의 raw 디렉토리
- 도커 이미지/번들
- 데이터셋 원본, 전처리 캐시
- 대용량 텍스트 manifest (필요 시 path만 기록)

### 트리거 시점

| 방향 | 시점 | 행위 |
|---|---|---|
| local → server | plan 파일 작성 후 | local이 명시적으로 push |
| server → local | results + registry 갱신 후 | server agent가 commit·push (의미 단위 분리), local이 명시적으로 pull |

### 자동화 경계

- server agent는 의미 단위로 commit·push를 자율 수행한다 (results/run/registry 갱신 단위). local 측 plan 핸드오프는 사용자 명시 push로만 트리거된다.
- commit 단위는 의미 단위 — plan 1개, results 1개, code change 1개를 분리한다.
- 한 commit에 binary 산출물을 섞지 않는다.

---

## §12. Autonomous Execution Protocol

server 측 agent 가 plan 1개를 G0 → G_final 까지 사용자 개입 없이 자동 실행할 때 따르는 protocol. `CLAUDE.md` 의 Autonomous Execution Policy 와 짝.

### §12.1 Invoke

interactive Claude Code session (tmux/screen 안) 에서:
```
> plan-NNN 진행
```

또는 비대화형:
```
claude -p "plan-NNN 진행" --permission-mode acceptEdits
```

### §12.2 매 turn step 시퀀스

1. Read `CLAUDE.md` (auto-load)
2. Read `WORKFLOW.md §12` (= 본 절)
3. Read `plans/plan-NNN-*.md` 의 `§0.5 Quick Reference`
4. `git log -20 --oneline` 으로 현 commit 위치 파악
5. §0.5 의 commit chain 에서 다음 [TODO] commit 식별
6. 그 commit 의 spec section 만 offset/limit 으로 부분 read
7. `git pull --rebase` (rebase conflict → severe)
8. 코드/테스트/문서 작성 (§12.5 path whitelist 준수)
9. self-check: `pytest tests/` + backward_compat smoke + invariant smoke
10. self-check pass → commit (decision-note 포함) + push. fail → severe alert + 멈춤
11. §0.5 의 [TODO] → [DONE] (commit hash) 1줄 update (이게 §12.6 blacklist 의 *유일한 예외*)
11.5. **G-gate check** (현 commit 이 STAGE 의 마지막 commit 인 경우만):
      - 해당 STAGE 의 모든 c{i} 가 §0.5 에서 [DONE] 마킹됐는지 확인
      - 누락 발견 시 severe (`stage_incomplete`) trigger
11.6. **§0.5 ↔ git log sync check** (매 commit):
      - §0.5 의 [DONE] hash 가 `git log --format=%H` 결과에 모두 존재하는지 grep
      - 불일치 발견 시 severe (`qr_log_mismatch`) trigger
12. 다음 commit 으로 진행 (step 5 로 return), G_final 도달 시 §12.10 따라 종료

### §12.3 Severe Issue (오직 이 9개만 멈춤)

1. `pytest_fail` — exit_code ≠ 0
2. `backward_compat` — `tests/backward_compat/*` 중 1개 이상 fail
3. `task_failure_rate ≥ 0.30` — `summary.json` 의 `n_failed / n_total ≥ 0.30`
4. `same_commit_msg_3x` — 같은 commit msg 가 3 turn 연속 시도됨 (stuck 감지)
5. `turn_count > 30` — safety 상한
6. `git_rebase_conflict` — auto-resolve 불가
7. `path_whitelist_violation` — §12.5 외 파일 수정 시도
8. `stage_incomplete` — STAGE 마지막 commit 시점에 §0.5 의 [TODO] 잔여 (§12.2 step 11.5)
9. `qr_log_mismatch` — §0.5 의 [DONE] hash 가 git log 에 부재 (§12.2 step 11.6)

다음은 *severe 아님* — 자율 진행:
- spec 모호 → *권장 default* 채택 + commit msg `decision-note` 박제
- borderline 합격/탈락 → 합격 기준 기계적 적용
- performance suboptimal → plan 의도 위배 아니면 진행
- 데이터 fetch 부분 실패 → retry 후 partial task drop
- dependency 누락 → `uv sync` / `pip install` 자율 시행
- minor format / lint → 자동 수정
- 코드 style 선택 → 권장 default (PEP8 등) 채택

### §12.4 Telegram Alert

`~/.config/telegram.env` 의 `TELEGRAM_BOT_TOKEN` + `TELEGRAM_CHAT_ID` 사용.

발신 시점:
| 시점 | 빈도 |
|---|---|
| session 시작 (1회) | 1 |
| STAGE (G-gate) 완료 | 6~8 / plan |
| severe issue 발생 | 사건당 1 |
| G_final 도달 | 1 |

→ plan 1개 정상 실행 = ~10~15 alerts.

### §12.5 Path Whitelist (수정 가능 영역)

- `src/**/*.py`
- `tests/**/*.py`
- `configs/**/*.yaml`
- `analysis/plan-{NNN}/**/*` ← 현재 plan 의 NNN
- `runs/{type}/{exp_id}/**` ← 실험 산출 (WORKFLOW.md §4 의 4-way token 패턴)

### §12.6 Path Blacklist (절대 수정 금지)

- `plans/*.md` — pre-registered. 단 §12.2 step 11 의 `§0.5 [TODO]→[DONE]` update 만 허용
- `CLAUDE.md`
- `.claude/settings.json` / `.claude/settings.local.json`
- `WORKFLOW.md`
- `vendor/**`
- 위 외 모든 파일

→ blacklist path 변경 시도 = §12.3 #7 severe trigger.

### §12.7 Destructive Ops 절대 금지

- `git push --force` / `git push -f`
- `git push --no-verify`
- `git reset --hard` / `git checkout {ref} -- .` / `git clean -f`
- `git branch -D`
- `rm -rf` 임의 path

→ 예외: `runs/{type}/{exp_id}/` 의 *직접 산출* 폐기는 OK.

### §12.8 Session Pre-flight (session 시작 1회)

session 첫 turn 에서:
1. `~/.config/telegram.env` 존재 + parse 가능 확인
2. dummy alert 발신 (`/loop {plan-id} 시작, commit X 부터`)
3. dummy alert 실패 → session abort
4. `git status` clean 확인 (uncommitted 잔여 → abort, 사용자 수동 정리 요청)
5. `git pull --rebase origin {branch}` (rebase conflict → abort)

### §12.9 Decision Note 규약

자율 결정한 commit 의 msg 마지막에:
```
decision-note: <category> — <한줄 사유>
```

category 예시:
- `spec-default` — plan 본문에 명시 안 된 디테일에 권장 default 채택
- `lint-fix` — minor format/lint 자동 수정
- `dep-install` — dependency 누락 으로 자율 install
- `data-partial` — 데이터 fetch 부분 실패 후 partial task drop
- `retry-3x` — 3회 retry 후 진행

사후 audit:
```
git log --grep "decision-note" --oneline
```

### §12.10 종료 정책

| 시나리오 | 조치 |
|---|---|
| G_final 도달 | 자연 종료, telegram 알림 ("plan-NNN 완료, hash=...") |
| severe issue | 멈춤, telegram alert, session 유지 (같은 session 에서 사용자 결정 후 재개) |
| max_turns (>30) | severe 와 동일 |
