---
name: plan-review-master
description: plan-review 스킬을 sub-agent 로 반복 호출하며 BLOCKER (Must-fix) / AMBIGUITY (Should-fix) 결함이 0건이 될 때까지 plan 본문을 자동 수정한다. 사용자의 의도와 전체 섹션 구조는 보존. audit 3축 pass-through 지원.
---

# plan-review-master

`plan-review` 스킬을 컨텍스트 독립적인 sub-agent 로 반복 호출해 plan 본문의 BLOCKER / AMBIGUITY 결함을 자동 수정한다. **사용자 의도와 전체 섹션 구조는 보존**하고, "정의 추가·명확화" 만 수행한다.

## 운영 규칙 (사용자 합의 사항)

| 항목 | 결정 |
|---|---|
| 자동 수정 대상 | **BLOCKER (Must-fix) + AMBIGUITY (Should-fix)** 만. MINOR 는 보고만, 수정 안 함. |
| audit 3축 pass-through | 지원. 두 번째 인자 `audit` → plan-review 의 §5 audit 트리거가 sub-agent 로 전달. audit 의 `REAL_BUG_*` / `REFERENCE_PARITY` 는 BLOCKER 동급 fix, `METRIC_SEMANTIC` / `REALISM_GAP` 은 AMBIGUITY 동급 fix, `DOC_GAP` 은 보고만. |
| iteration 안전장치 | **max 5 iter**. 동일 결함 재발 감지 시 escalate (해당 issue 자동 수정 포기, 사용자에게 manual fix 요청). |
| 컨텍스트 격리 이유 | plan-review 는 반드시 sub-agent 에서 실행. main session 의 conversation history (사용자 의도·기존 fix 시도) 를 모르는 상태로 점검해야 false positive·자가편향이 줄어든다. |
| **개발자 의도의 위치** | **plan 본문이 개발자 의도의 self-contained 표현**. plan 외부 (소스 코드, 외부 모듈, 다른 framework 등) 에서 의도를 찾지 않는다. plan 내부 표현이 애매하면 frontmatter 의 `based_on` / `supersedes` 가 가리키는 *직전 plan* 만 참고 가능. 그 외 외부 source read 금지. |
| **점검의 본질** | (1) plan 내부 일관성 — frontmatter / §1 목적 / 본문 spec / §6 deferred 가 같은 의도를 가리키는가. (2) plan 명세대로 코드를 simulate 했을 때 §1 이 표현한 *이번 plan 의 책임* 과 비슷한 outcome 을 얻을 수 있는가 (예: "데이터 적립" 이 목적이면 출력 schema 가 그 적립을 지원하는가). |
| **plan 별 의도 mode 추론** | iter loop 시작 전 frontmatter + §1 만 한 번 read 해 mode 결정 (§2.0 참고). mode 별로 점검 강도 다름 — 외부 reference 정합을 추구하는 plan 과 알고리즘 명세-코드 변환을 추구하는 plan 은 BLOCKER 후보 자체가 다름. |

## 절대 금지

- main session 에서 plan-review 직접 실행 (`Skill` 호출 금지). 컨텍스트 격리 깨진다.
- plan frontmatter 의 *모든 metadata key* 변경 금지 (예: `plan_id`, `version`, `exp_ids`, `data_window`, `*_pin` (외부 reference pin), `status`, `supersedes`, `based_on`, `followed_by` 등). frontmatter 는 plan 의 의도 mode 추론 시그널이기도 하므로 변형 시 §1.5 mode 결정이 흔들릴 수 있음.
- `^## ` / `^### ` 섹션 추가·삭제·재배치.
- 실험 가설·실행 디렉션·stage 구분의 의미적 변경.
- 사용자 confirm 요청 (자동 수정이 본 스킬의 핵심 가치).
- plan 외 다른 파일 수정. (audit 모드의 engine/strategy 코드는 read-only, plan 본문에 경고 블록만 추가)
- `Write` 로 plan 전체 덮어쓰기. 항상 `Edit` 으로 단일 변수 패치.
- **plan 외부 source code read 금지**. main session 의 *재검증* 단계에서 외부 모듈 / pinned 외부 디렉토리 / 다른 framework 코드의 source 를 직접 읽지 않는다. plan 본문이 self-contained 표현이며, plan 이 외부 source 를 인용해도 그 인용 자체 (line 인용 / SHA / path) 가 plan 의 spec 박제로 충분 — 외부 source 의 자가 해석 시도는 plan 의미 왜곡 위험. *예외*: audit 모드의 audit 전용 sub-agent 가 인용한 file:line 만 read (그것도 main session 이 아닌 sub-agent 책임).
- **외부 모듈 signature / dataclass / function path 의 inline 박제 시도 금지**. sub-agent 가 "외부 의존, signature 미박제" 를 BLOCKER 로 flag 해도, 해당 외부 source 의 pin (SHA / commit / path / 모듈 식별자 중 어느 하나) 이 plan frontmatter 또는 본문에 박제되어 있으면 fix 시도 안 함 (FP). plan 의 의도가 "외부 reference 정합" 이라면 그 외부의 코드 디테일을 plan 안에 박제하는 것은 plan-review-master 의 책임 밖이며, plan 자체의 *외부 정합 의도* 와도 충돌.
- **외부 reference 와의 mechanical (output-by-output / trade-by-trade) parity 점검 금지**. 그런 정합은 plan §1 이 *명시적 목표* 로 박은 경우에만 plan 내부 일관성 차원에서 점검 — 외부 source 와의 실제 비교는 plan-review-master 책임 밖.

---

## 인자 해석

`plan-review` 의 인자 해석을 그대로 재사용 ([.claude/commands/plan-review.md:12-18](../../commands/plan-review.md#L12-L18)).

- 인자 없음 → `plans/` 안 가장 최신 `plan-NNN-*.md` 중 같은 이름 `.results.md` pair **없는** 것 자동 선택.
- 첫 번째 인자가 숫자 (예: `003`) → `plans/plan-{NNN-zero-pad-3}-*.md` resolve. 매칭 0개 또는 2개 이상 → 사유 출력 후 종료.
- 첫 번째 인자가 그 외 문자열 → 임의 파일 경로. 부재 시 사유 출력 후 종료.
- **두 번째 인자 `audit`** (case-insensitive) → sub-agent 호출 시 plan-review 에 `audit` 인자 전달. 반환되는 audit 섹션도 같은 fix 루프에 합류.
- 두 번째 인자가 다른 문자열 → 사유 한 줄 출력 후 plan-review 만 (audit 미실행).

해석 결과: `plan_abs_path`, `audit_arg` (`"audit"` 또는 `""`).

---

## Steps

### 1. 인자 해석 + 초기화

```
iter_max = 5
iter_count = 0
prev_signatures = set()                       # 직전 iter 의 verified issue signatures
total_fixed = {BLOCKER:0, AMBIGUITY:0, audit_high:0, audit_med:0}
escalated = []                                # 동일 결함 재발 → manual fix 요청
minor_log = []                                # 보고만 하는 MINOR / audit_low
iter_log = []                                 # 사용자 노출용 진행 로그
last_report_md = None                         # 마지막 iter 의 plan-review 보고서
```

plan 경로 결정 실패 → 사유 한 줄 출력 후 종료. 이후 단계 진입하지 않음.

### 1.5 plan 의도 mode 추론 (iter loop 진입 전 1회만 수행)

iter loop 안에서 매번 mode 를 다시 판정하면 plan 본문이 fix 로 변형됨에 따라 mode 가 흔들릴 위험이 있다. 따라서 **iter loop 시작 전 plan 의 frontmatter + §1 본문만 한 번 read** 하여 mode 를 결정하고 main session state 에 박제, 이후 모든 iter 에서 재사용한다.

#### 1.5.1 mode 시그널

| 시그널 (frontmatter / §1 본문) | mode 가중치 |
|---|---|
| frontmatter 에 외부 source pin (예: `*_pin: <SHA-or-id>`, `based_on: <외부 source 인용>`) 존재 | **reference-aligned** ↑↑ |
| §1 본문에 "정합", "박제 그대로", "최소 수정", "외부 reference 와 동일", "변경 없이 재사용" 류 표현 | **reference-aligned** ↑ |
| §1 본문에 "본 plan 은 ... 의도적 분리", "*보존*" (외부와의 의도적 deviation 명시) | **reference-aligned** 보강 (deviation 자체가 reference-aligned 의 변종) |
| frontmatter 에 외부 pin 없음 + §1 본문이 알고리즘 / 절차 명세 + "충실 구현", "스펙 → 코드 변환", "본 plan 의 spec 을 그대로 코드화" 류 표현 | **spec-implementation** ↑↑ |
| frontmatter 의 `based_on` 이 *직전 plan* 만 인용 (외부 source 가 아닌) + §1 이 결과 분석·robustness·집계 류 | **spec-implementation** (또는 'analysis' 변종) ↑ |
| 시그널 충돌 / 둘 다 약함 | **spec-implementation** 으로 fallback (보수적 — plan 본문 자족성 검증을 default 로) |

#### 1.5.2 mode 별 점검 강도

**`reference-aligned` mode** (plan 이 *외부 reference 와의 정합* 을 추구):
- BLOCKER 후보 좁힘: plan 본문 안의 *외부 reference 의 의도 박제 누락* 만 BLOCKER. 외부 reference 의 *코드 디테일 박제 누락* 은 FP (plan 의 의도 자체가 외부 코드를 그대로 따르는 것).
- 외부 reference 와의 *deviation 명시* 가 §6 deferred / V#-style exception 으로 박제되어 있는지 점검 (deviation 이 implicit 이면 AMBIGUITY).
- "외부 source 의 signature inline 박제 요구" → 항상 FP (§2.3 휴리스틱 표 참조).

**`spec-implementation` mode** (plan 이 *본문 spec 의 코드화* 를 추구):
- BLOCKER 후보 확장: plan 본문 spec 자체의 자족성 (식 / 시그너처 / dtype / 경계) 강하게 점검.
- **plan 명세대로 코드를 simulate 했을 때 §1 이 표현한 outcome 과 일치하는가** 를 main session 의 재검증 단계에서 한 번 더 점검 (단순히 spec 정의 부재 여부가 아니라 *spec 이 §1 의 outcome 을 만들 수 있는지* 의 인과 chain).
- 외부 source 자체 부재 가정 — 외부 의존 BLOCKER 는 항상 BLOCKER 로 간주 (FP 아님).

#### 1.5.3 mode 박제

추론 결과를 `plan_intent_mode ∈ {"reference-aligned", "spec-implementation"}` 변수에 저장. 이후 §2.3 재검증 / §2.5 fix 패턴이 mode 에 따라 분기.

mode 결정이 명확히 되지 않으면 사용자에게 한 줄 안내 후 `spec-implementation` 으로 fallback (예: "plan 의도 mode 추론 실패 — `spec-implementation` 으로 진행. 외부 정합을 추구하는 plan 이라면 frontmatter 의 pin metadata 를 추가 후 재실행 권장.").

### 2. Iteration loop (while iter_count < iter_max)

각 iter 는 다음 5 단계로 구성된다.

#### 2.1 plan-review sub-agent spawn

`Agent(subagent_type="general-purpose")` 한 개 호출 (foreground, 결과 대기). prompt 템플릿:

```
당신의 임무: plan-review 스킬을 실행하여 단일 plan 파일에 대한 점검 보고서를 출력한다.

수행 절차:
1. Read 로 plan-review 스킬 명세를 학습:
   /Users/dryas/Desktop/backtest/.claude/commands/plan-review.md
2. 그 스킬의 절차를 그대로 수행:
   - 첫 번째 인자 (plan path): {PLAN_ABS_PATH}
   - 두 번째 인자 (audit toggle): {AUDIT_ARG_OR_EMPTY}
3. 스킬 명세에 정의된 마크다운 보고서를 그대로 stdout 으로 출력. 추가 commentary 금지.

추가 점검 규약 (plan-review-master 측 박제):
- **plan 의도 mode**: `{PLAN_INTENT_MODE}` (`reference-aligned` 또는 `spec-implementation`).
  - `reference-aligned`: 본 plan 은 외부 reference 와의 정합 추구 + 백테스트 문제만 최소 수정. 외부 reference 의 *signature / dataclass / function path 의 inline 박제 부재* 를 BLOCKER 로 flag 하지 말 것 (plan frontmatter 또는 본문에 외부 source pin 이 있으면 자족 충족). 외부 reference 와의 *deviation* 이 §6 deferred / V#-style exception 으로 명시되어 있으면 OK.
  - `spec-implementation`: 본 plan 은 알고리즘 / 절차 명세 → 코드 변환 추구. plan 본문 spec 의 자족성 (식 / 시그너처 / 경계 / 단위) 을 강하게 점검. 외부 source 자체 부재 가정.
- **plan 외부 source read 금지**: plan 본문이 개발자 의도의 self-contained 표현. 외부 모듈 / 다른 framework 코드 / pinned 외부 디렉토리 등의 source 를 직접 읽지 말 것 (audit 모드의 audit-전용 sub-agent 는 예외).
- **추가 점검 축 — spec-simulate vs §1 outcome 정합**: plan §1 이 표현한 *이번 plan 의 outcome* (예: "데이터 적립", "175-cell matrix 산출", "alpha 검증") 을 plan 본문 spec 만으로 *시뮬레이션* 했을 때 해당 outcome 과 비슷한 결과를 얻을 수 있는가? 단순히 spec 정의가 박제되어 있는지가 아니라, *spec 이 §1 의 outcome 을 만들 수 있는 인과 chain* 을 갖는지 점검. 인과 끊김 시 BLOCKER (코드 작성 가능성 축에서 기록).

규약:
- 스킬 명세에 따라 내부적으로 5 (audit 시 +3) sub-agent 병렬 spawn 가능. plan-review-master 측 박제 (위 3 항목) 를 spawn 한 sub-agent prompt 에도 그대로 forward.
- 어떤 파일도 *수정하지 않는다*. 보고서 출력만.
- 메인 컨텍스트가 결과를 파싱하므로 마크다운 형식 (BLOCKER 섹션, 권장 수정 순서, audit 섹션) 엄수.
- 본 결과는 main session 에서 자동 fix 루프의 입력으로 쓰인다는 점만 인지.
- 본 sub-agent 의 컨텍스트는 main session 으로부터 격리되어 있어야 한다 — 이전 fix 시도나 사용자 conversation 을 신뢰하지 말 것.
```

`{PLAN_ABS_PATH}` 와 `{AUDIT_ARG_OR_EMPTY}` 는 §1 의 결과로 치환, `{PLAN_INTENT_MODE}` 는 §1.5 의 결과로 치환. sub-agent 는 plan-review.md 의 §3 절차에 따라 자체적으로 5 (또는 audit 시 +3) sub-agent 를 재귀 spawn 하며, 위 3 placeholder 의 값을 spawn 하는 sub-agent prompt 에도 forward 한다.

`last_report_md` 에 응답 마크다운 저장.

#### 2.2 보고서 파싱 (`parse_report`)

plan-review 출력 형식 ([.claude/commands/plan-review.md:96-118](../../commands/plan-review.md#L96-L118), audit [.claude/commands/plan-review.md:261-300](../../commands/plan-review.md#L261-L300)) 을 정규식 + 라인 인용 추출로 파싱.

```
issues = {
    BLOCKER: [],          # (1) 코드 작성 가능성 — 버킷별 BLOCKER 섹션
    AMBIGUITY: [],        # (2) 의도 분리/연속성 verdict "흐림" + 권장 수정 순서 비-BLOCKER
    MINOR: [],            # 본문 인용된 MINOR (있으면)
    audit_high: [],       # 🔴 REAL_BUG_INFLATE/DEFLATE/NEUTRAL + REFERENCE_PARITY
    audit_med: [],        # 🟢 REALISM_GAP / METRIC_SEMANTIC
    audit_low: [],        # DOC_GAP
}
```

각 issue 의 데이터 구조:

```python
{
    "severity": "BLOCKER",                # BLOCKER / AMBIGUITY / MINOR / audit_high / audit_med
    "title": "신호 식 0-width 처리 정의 부재",
    "line": 142,                          # 인용된 plan 내 대표 라인
    "line_range": (132, 152),             # ±10 컨텍스트
    "rationale": "본문 사유 한 줄",
    "raw": "원본 마크다운 bullet",        # signature·debug 용
}
```

추출 규칙:
1. `## 결론 (한 문장)` → `BLOCKER {N}건, AMBIGUITY {N}건+, MINOR {N}건` sanity-check.
2. `## (1) 코드 작성 가능성 — 버킷별 BLOCKER` 의 `### 버킷 N` 안 bullet `- **{title}** [plan:LINE](path#LLINE) — 사유` → `BLOCKER`.
3. `## (2) 의도 분리/연속성` 에서 verdict "흐림" 인 (A)/(B) → 본문 인용 라인을 `AMBIGUITY` 후보. 주변 ±10 라인을 line_range 로.
4. `## 권장 수정 순서` 의 항목 중 BLOCKER 미커버 → `AMBIGUITY` 추가 (중복 dedupe).
5. audit 모드:
   - `#### 🔴 REAL_BUG_INFLATE`, `#### 🟡 REAL_BUG_NEUTRAL / REAL_BUG_DEFLATE / REFERENCE_PARITY` → `audit_high`.
   - `#### 🟢 REALISM_GAP / METRIC_SEMANTIC` → `audit_med`.
   - `DOC_GAP` → `audit_low`.

파싱 실패 (형식 mismatch) → "파싱 실패" 로 iter 종료, fix 루프 break, final report 에 raw markdown 첨부.

#### 2.3 main session 재검증 (`is_real_problem`)

main session 이 각 issue 에 대해 직접 plan 본문을 `Read` (line_range 주변 ±10) 후 실재 결함인지 판단. **자신 없으면 real problem 으로 간주** (보수적).

판정 휴리스틱:

| 결함 유형 | False positive 조건 |
|---|---|
| "식·계산 정의 부재" | line_range ±10 안에 수식 또는 의사코드 블록 존재 |
| "시그너처·dtype 미명시" | 같은 섹션 내 ` ```python ... ``` ` 코드 블록에 함수 시그너처 정의 |
| "데이터 경계 모호" | 명시적 inclusive/exclusive 어구 (`< T`, `≤ T`, `[a, b)` 등) 존재 |
| "외부 의존" | 근처에 외부 reference pin (frontmatter 또는 본문의 `*_pin:` 류 metadata), commit hash, 또는 명시적 path 존재 |
| **"외부 모듈 signature / dataclass inline 박제 부재"** | plan frontmatter 또는 본문에 외부 source pin (SHA / commit / path / 모듈 식별자 중 어느 하나) 이 명시되어 있으면 **항상 FP** — 외부 signature 의 inline 박제는 plan-review-master 의 비목표. plan 은 외부 reference 의 *의도 (룰, 진입점, filter, 논리 근거)* 만 박제하고 코드 디테일 (dict 구조, 변수명, 함수 호출 chain) 은 본인 framework 자유. plan 의도 mode 가 'reference-aligned' 인 경우 특히 강한 FP. |
| **"외부 코드 흐름 / dispatch 구조 박제 부재"** | 동일 — 외부 source pin 이 plan 에 있으면 FP. |
| **"외부 reference 와 mechanical (output-by-output) 정합 미보장"** | **항상 FP** — 외부 source 와의 실제 동작 정합은 plan §1 이 그것을 *명시적 목표* 로 박은 경우에만 plan 내부 일관성 차원에서 점검 (외부 source read 안 함). 그 외엔 plan-review-master 의 책임 밖. |
| AMBIGUITY (의도 분리 흐림) | plan 다른 섹션에 관련 정의·cross-ref 존재 |
| audit_high / audit_med | 인용된 외부 file:line 을 read → 메커니즘이 코드와 일치하지 않으면 false positive |

판정 결과:
- `fixable`: real problem AND signature ∉ prev_signatures
- `recurring`: real problem AND signature ∈ prev_signatures
- `skipped_fp`: false positive (보고만, 수정 안 함)

`signature(severity, issue) = (severity, issue.line // 20, normalize_keyword(issue.title))`
- `// 20`: fix 후 line shift 로 정확 매칭 깨짐 → 20-라인 버킷 비교
- `normalize_keyword`: 소문자 + 공백 정규화 + 핵심 keyword (예: "신호 식 0-width 처리 정의 부재" → "0-width 처리")

#### 2.4 종료 조건 검사

```
if not fixable and not recurring:
    iter_log.append("iter N/5: 0 verified Must/Should-fix → 종료")
    break  # 성공 종료
```

`fixable + recurring == 0` 이 양쪽 (sub-agent 보고 0건이거나, 보고는 있어도 main session 검증 0건) 모두 만족하는 시점이 종료점.

#### 2.5 fix 적용 + escalate

**`fixable` 처리** — `apply_fix(issue)`:
- `severity` 별 패턴 (모두 plan 본문만 수정, 외부 source 는 read 도 fix 도 안 함):
  - `BLOCKER` 식 부재 → 해당 위치에 수식 한 줄 + 변수 정의 추가.
  - `BLOCKER` 시그너처 미명시 → ` ```python def fn(...) -> ReturnType: ...``` ` 한 블록 추가 (단 이는 *본인 framework* 의 함수 시그너처 — 외부 source 의 signature 인라인 박제 시도 금지).
  - `BLOCKER` 경계 모호 → 부등호 / 구간 표기 명시.
  - `BLOCKER` 외부 의존 → 외부 source pin (commit hash / path / 모듈 식별자) 이 plan 어딘가에 박제되어 있는지 확인. 박제되어 있으면 **FP 처리** (skip). 박제 부재 시 **사용자 escalate** (어떤 외부 source 를 의도하는지 main session 이 자가 판단 금지).
  - `BLOCKER` 단일 변수 위반 → 한 실험에서 변경 변수 1개로 좁히는 문구 추가 (변수 자체는 안 바꿈, 명시만).
  - `AMBIGUITY` 정성 → 정량 치환 (예: "충분히 큼" → "p_value < 0.005"). 누락 cross-ref 추가. skip 룰 일관성 한 줄 추가.
  - `audit_high` → plan 본문에 **결함 경고 + spec 보완** 한 블록 추가 (코드 미수정).
  - `audit_med` → metric 정의 / 실거래 갭 가정 섹션에 한두 줄 추가.

- **mode 별 추가 룰**:
  - `plan_intent_mode == "reference-aligned"` 이고 BLOCKER/AMB 가 *외부 source 의 코드 디테일 inline 박제* 를 요구 → **FP 처리, fix 시도 안 함** (§2.3 휴리스틱 표의 외부 signature/dispatch FP 룰).
  - `plan_intent_mode == "reference-aligned"` 이고 BLOCKER/AMB 가 *외부 source 와의 deviation 명시* 를 요구 → §6 deferred 표 또는 §2 fix table 의 "*보존*" / "*의도적 분리*" 류 한 줄 추가로 fix.
  - `plan_intent_mode == "spec-implementation"` 이고 BLOCKER/AMB 가 *plan 본문 spec 의 식·시그너처·경계 박제 부재* → 통상 fix 패턴 적용 (식 / 시그너처 / 경계 추가).
  - 어느 mode 든 BLOCKER/AMB 가 *외부 source 의 자가 해석 결과* 를 plan 에 박는 fix 를 요구하는 것처럼 보이면 → **escalate** (main session 자가편향 위험, 사용자 판단 필요).
- 수정 절차:
  1. plan 의 issue.line ±20 `Read` → 정확한 컨텍스트 캡처.
  2. 최소 변경 `old_string` / `new_string` 작성 (line number prefix 제외).
  3. `Edit` 호출. **단일 변수 원칙**: 한 Edit = 한 issue.
  4. `total_fixed[severity] += 1`.

**`recurring` 처리**:
- 자동 fix **시도 안 함**.
- `escalated` 에 추가 (중복 sig 는 한 번만 등록).

**signature 갱신**:
```
prev_signatures = {signature(s, i) for s, i in fixable + recurring}
```
다음 iter 에서 같은 sig 가 또 나오면 그것도 recurring. main session fix 가 효과 없는 이슈에서 무한 루프 차단.

#### 2.6 iter log

```
iter_log.append(
    f"iter {iter_count}/{iter_max}: "
    f"BLOCKER {len(issues['BLOCKER'])} / AMB {len(issues['AMBIGUITY'])}"
    f"{f' / audit_high {len(issues[\"audit_high\"])}' if audit_arg else ''} 보고 → "
    f"verified {len(fixable)+len(recurring)} (FP {len(skipped_fp)}) → "
    f"fixed {len(fixable)}, escalated {len(recurring)}"
)
```

minor / audit_low 는 `minor_log` 에 누적 (final report 에서 보고만).

### 3. max iter 도달

`while` 의 `else:` 가지 (loop 가 break 없이 끝나면):
```
iter_log.append(f"iter {iter_max}/{iter_max}: max iter 도달 — 잔여 결함 있을 수 있음")
```

---

## 4. 최종 출력 (단일 마크다운, 파일 저장 없음)

```markdown
## plan-review-master 실행 결과

| 항목 | 값 |
|---|---|
| plan | {plan-basename} |
| audit 모드 | {on / off} |
| iterations | {iter_count} / {iter_max} |
| 종료 사유 | {0건 도달 / max iter / parse failure} |
| 총 수정 (BLOCKER) | {count} |
| 총 수정 (AMBIGUITY) | {count} |
| 총 수정 (audit_high) | {count, audit 모드만} |
| 총 수정 (audit_med) | {count, audit 모드만} |
| escalated (manual fix 필요) | {count} |
| MINOR / audit_low (보고만) | {count} |

### Iteration 진행
- iter 1/5: BLOCKER 5 / AMB 2 보고 → verified 6 (FP 1) → fixed 6, escalated 0
- iter 2/5: BLOCKER 1 / AMB 0 보고 → verified 1 (NEW) → fixed 1
- iter 3/5: 0 verified Must/Should-fix → 종료

### Escalated (사용자 manual fix 필요)
- **{title}** [{plan-basename}:{LINE}]({relative/path}#L{LINE}) — 사유: 자동 fix 후 sub-agent 가 동일 결함 재보고. 사용자 판단 필요.

### MINOR / 참고 (자동 수정 안 함)
- **{title}** [{plan-basename}:{LINE}]({relative/path}#L{LINE}) — 한 줄 사유

### 마지막 plan-review 보고서 (참고)
{last_report_md, 1500단어 이내로 압축}
```

출력 형식 규약:
- 모든 line 인용은 `[basename:LINE](relative/path#LLINE)` 마크다운 링크 (cwd 기준 상대경로, 절대경로 금지).
- 본 보고는 stdout 만. plan 본문 수정은 iteration 중 `Edit` 으로 이미 반영됨.
- 추가 파일 생성 금지.

---

## 비기능 규약

- main session 은 plan-review 보고서 외 다른 source 를 신뢰하지 않는다 (사용자 추가 입력 없는 한).
- fix 적용 직전 사용자 confirm 요청하지 않는다 — 자동 적용이 명시적 요구사항.
- iteration 중 plan 외 파일 수정 금지. (audit 시 audit agent 가 인용한 engine/strategy 코드 read 는 허용)
- 매 iter 의 plan-review sub-agent 는 새로운 컨텍스트 (main session conversation 미공유) 에서 수행되어야 한다.
- 본 스킬 종료 후 사용자가 결과 리뷰 → 필요 시 escalated 항목 manual fix → 다시 실행 (idempotent: 이미 fix 된 plan 은 iter 1 에서 0건 도달).
