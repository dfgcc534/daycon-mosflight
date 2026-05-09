Plan 파일 한 개를 두 축으로 점검하고 터미널에 마크다운 보고서만 출력한다 (파일 저장 없음).

두 축:
1. **코드 작성 가능성** — 이 plan 한 파일만으로 구현자가 추가 추론·검색 없이 식·시그너처·dtype·예외·단일 변수 경계까지 결정해 코드를 작성·실행 단계 직전까지 진입할 수 있는가.
2. **의도 분리/연속성** — 개발자의 의도가 plan 전체 흐름에 잘 드러나고 stage·phase 별로 분리되어 있는가.

점검은 **plan 파일 한 개의 본문만** 본다. 외부 규약 문서·프로젝트 컨벤션·다른 파일은 일절 읽지 않는다.

옵션 3축 (2번째 인자 `audit` 일 때만 추가 실행):
3. **백테스트 변환 충실도** — 외부 reference 박제 코드를 backtest engine 으로 변환하는 과정에서 *논리 결함* 과 *실거래 시뮬레이션 현실성 갭* 을 감사. plan 본문 외에 외부 reference source / backtest engine source 도 함께 읽는다.

## 인자 해석

- 인자 없음 → 현재 작업 디렉토리의 `plans/` 안에서 가장 최신 `plan-NNN-*.md` 중 같은 이름의 `.results.md` pair 가 **없는** 것을 자동 선택.
- 첫 번째 인자가 숫자 (예: `002`) → `plans/plan-{NNN-zero-pad-3}-*.md` 로 resolve. 매칭이 0개거나 2개 이상이면 사유 출력 후 종료.
- 첫 번째 인자가 그 외 문자열 → 임의 파일 경로로 간주. 파일 부재 시 사유 출력 후 종료.
- **두 번째 인자 `audit`** (옵션, case-insensitive) → 기본 plan-review 출력 후 §5–§6 의 백테스트 변환 충실도 audit 추가 실행.
- 두 번째 인자가 다른 문자열 → 사유 출력 후 plan-review 만 진행 (audit 미실행).

## Steps

### 1. plan 로드
- 인자 해석으로 plan 절대경로 결정. 결정 실패 시 즉시 종료.
- `Read` 로 plan 본문 전체를 읽는다. 다른 파일은 §5 트리거가 아니면 읽지 말 것.
- 동일 디렉토리에 `<plan-stem>.results.md` 존재 시 §5 audit 의 "이미 알려진 결함" 섹션을 읽기 위해 메모해 둔다 (§1~§4 단계에서는 미사용).

### 2. 섹션 버킷팅
- plan 본문에서 `^## ` 로 시작하는 헤딩을 모두 추출 → 각 섹션의 시작·끝 라인 산출.
- 섹션을 라인 합계 기준으로 **최대 4개 버킷에 균등 분배** (greedy: 가장 누적이 적은 버킷에 다음 섹션 할당). 큰 섹션 하나가 1 버킷을 단독 점유해도 무방. 헤딩이 4개 미만이면 그 수만큼 버킷 사용.
- 각 버킷의 섹션 라벨과 line range 를 메모해 둔다 (예: 버킷 1 = "§1~§3, L29-L140").

### 3. 5 sub-agent 병렬 호출
한 메시지 안에서 다음을 모두 spawn (foreground, 결과를 기다린다).

#### 3a. 코드 작성 가능성 축 — 버킷 수만큼 (최대 4개)
각 버킷마다 `Agent(subagent_type="general-purpose")` 호출. prompt 템플릿:

```
당신의 임무: 다음 plan 파일의 지정된 섹션만 보고 "구현자가 추가 추론·검색 없이
바로 코드를 작성·실행 직전까지 진입할 수 있는가" 점검.

plan 절대경로: {ABS_PATH}
담당 섹션: {SECTION_LABELS}, line range {LINE_RANGE}

규약:
- 위 plan 파일만 Read 한다. 다른 어떤 파일도 읽지 말 것.
- 담당 섹션 외의 라인은 cross-reference 용으로만 가볍게 참조 가능. 점검 대상은 담당 섹션.

위반 후보 (모두 BLOCKER/AMBIGUITY/MINOR 중 하나로 분류):
- 식·계산 정의 부재 (평균식, 경계조건 i=0, ZeroDivision/0-width 처리 등)
- 시그너처·반환 타입·dtype·예외 클래스 미명시
- timezone, inclusive/exclusive, list vs Enum 등 데이터 경계 모호
- 외부 저장소·외부 노트·이전 채팅 의존 (commit hash 미지정 포함)
- 한 실험·단계의 변경 변수가 단일하지 않거나 단일성이 implicit
- public API 와 내부 export 이름의 충돌, 누락된 re-export
- 자동 통과/실패 분기가 정성 표현으로 결정 불가능

분류 등급:
- BLOCKER : 코드 작성 자체가 불가능한 결함
- AMBIGUITY : 구현자가 해석 분기를 만나는 모호함
- MINOR : 개선 권고

보고 형식:
- 등급별로 묶어 한국어 bullet list. 각 finding 에 plan 내 line number 인용 필수.
- 600~700 단어 이내. 추측 금지 — 본문에 적힌 내용만으로 판정.
```

#### 3b. 의도 분리 축 — 1 agent
동일 메시지에서 병렬 spawn. `Agent(subagent_type="general-purpose")`. prompt:

```
당신의 임무: 다음 plan 파일 전체를 읽고 두 verdict 를 산출.

plan 절대경로: {ABS_PATH}

규약: 위 파일만 Read. 다른 파일 금지.

(A) 전체 narrative 연결성
  - 배경 → 가설 → 실험 → 실행의 인과 chain 이 끊김 없이 흐르는가?
  - "왜 이 가설들인가"·"왜 이 실험이 그 가설을 검증하는가" 매핑이 본문에 명시적인가?

(B) stage / 묶음 분리
  - 한 실험이 정확히 한 가설만 담는가?
  - 묶음(stage·phase) 간 의존성·skip 룰이 일관되게 적용되었는가?
  - 한 실험에서 여러 변수를 동시 변경해 의도 경계가 흐려지지 않는가?

각 verdict: "잘 드러남 / 부분적으로 흐림 / 흐림" 3단계 중 하나.
근거 line 인용 3개 이상 필수. 흐림 지점은 구체 라인 지목.

700 단어 이내, 한국어.
```

### 4. 합산 보고 (plan-review 본체)
5 agent 결과 수신 후, 메인 컨텍스트가 직접 다음 구조로 터미널에 마크다운 출력.

```markdown
## 결론 (한 문장)
{plan 식별자}: BLOCKER {N}건, AMBIGUITY {N}건+, MINOR {N}건. (A) {verdict} / (B) {verdict}.

## (1) 코드 작성 가능성 — 버킷별 BLOCKER
### 버킷 1 ({라벨})
- **{항목명}** [{plan-basename}:{LINE}]({relative/path}#L{LINE}) — 한 줄 사유
### 버킷 2 ({라벨}) …
### 버킷 3 ({라벨}) …
### 버킷 4 ({라벨}) …

(BLOCKER 0건인 버킷은 "BLOCKER 없음" 한 줄. AMBIGUITY/MINOR 는 별도 섹션
없이 권장 수정 순서에서 선별 인용.)

## (2) 의도 분리/연속성
### (A) narrative — {verdict}
- 근거 line 인용 (3개 이상)
### (B) stage 분리 — {verdict}
- 흐림 지점 구체 지목 (line 인용)

## 권장 수정 순서
1. 가장 critical 한 BLOCKER 부터 N개 (각 항목 한 줄)
```

### 출력 형식 규약
- 모든 line 인용은 `[plan-basename:LINE](relative/path#LLINE)` 마크다운 링크 (VSCode 확장 환경에서 클릭 가능).
- relative path 는 cwd 기준. 절대경로 금지.
- "결론" 한 문장이 가장 위 — 사용자가 한눈에 critical 여부 판단.
- 본 보고서 외에 어떤 파일도 쓰지 않는다.

---

### 5. 백테스트 변환 충실도 audit (2nd arg = `audit` 일 때만 진행)

§4 보고 출력 직후 진행. 트리거 조건이 충족되지 않으면 §6 종료로 직진.

본 audit 는 **모든 backtest 프로젝트에 적용 가능한 추상 감사 프로토콜**이다. 특정 strategy/reference/엔진 컨벤션에 의존하지 않으며, 경로·심볼·용어를 prompt-time 에 자동 발견한다.

#### 5.1. 사전 메타 수집 (메인 컨텍스트가 직접 수행)

§1 에서 읽은 plan 본문 + 필요 시 cwd 의 directory listing (Bash `ls`/`find` 또는 Glob) 만 사용해 다음을 결정. 추가 Read 호출은 결정에 필수 불가결한 경우 (예: `pyproject.toml`, `package.json`, top-level `README`) 1~2 회만 허용.

- **REFERENCE_ROOT** (외부 reference / 박제 / 외부참조 코드 베이스, 선택사항):
  - plan 본문에서 `vendor_commit:`, `vendor pin:`, `reference_pin:`, `pinned_at:`, `upstream:`, `vendor/...`, `third_party/...`, `external/...`, `reference/...` 패턴 우선 탐색 (frontmatter 호환성을 위해 기존 컨벤션 키워드 유지 — 라벨일 뿐 의미는 모두 generic *외부 reference*).
  - 박제 commit hash 가 명시되면 `<top-level-reference-dir>/<sub-name>/<hash>/` 형태로 결정 (예: `external/foo/abc123/`, `reference/xxx/sha-deadbeef/`).
  - 명시가 없으면 cwd 하위 `vendor/`, `third_party/`, `external/`, `reference/` 디렉토리 중 존재하는 것을 root 로 (디렉토리 이름은 프로젝트 컨벤션 hit 용도). 모두 부재 시 **REFERENCE_ROOT = none** 으로 marking (audit 는 진행, reference-parity 카테고리만 N/A 처리).
- **ENGINE_ROOT** (백테스트 엔진/실행 루프/브로커 시뮬레이션 코드):
  - plan 본문에서 `engine`, `broker`, `simulator`, `backtest`, `runner` 키워드와 함께 등장하는 path 토큰 우선.
  - 미명시 시 cwd 하위 다음 패턴을 순차 탐색: `**/backtest/`, `**/engine/`, `**/simulator/`, `**/broker/`, `lib/backtest/`, `core/backtest/`, `pkg/*/backtest/`. 첫 매치 사용.
  - 그래도 결정 안 되면 사용자에게 *prompt 안에서* "ENGINE_ROOT 미식별, 추정 경로 X 로 진행" 한 줄 안내 후 진행. 완전 부재 시 §6 종료.
- **STRATEGY_ROOT** (신호 생성 / 알파 로직):
  - plan 본문에서 `strategy`, `signal`, `alpha`, `indicator`, `feature` 키워드와 path 토큰 우선.
  - 미명시 시 cwd 하위 `**/strategies/`, `**/strategy/`, `**/signals/`, `**/alpha/` 순차. 첫 매치 사용. 부재 시 ENGINE_ROOT 와 동일 디렉토리에서 신호 모듈 추정.
- **METRICS_ROOT** (성과·리스크 지표 산출 코드, 선택사항):
  - plan 본문에서 `metrics`, `report`, `summary`, `stats`, `analytics` 우선. 미명시 시 cwd 하위 `**/metrics/`, `**/reports/`, `**/stats/` 순차. 부재 시 N/A.
- **CONFIG_FIELDS** (현실성 ghost-config 후보 자동 추출):
  - plan 본문에서 yaml/dict-like config block 안에 등장하는 키 중 *실거래 마찰* 관련 후보를 휴리스틱 추출: `slippage*`, `*_bps`, `funding*`, `borrow*`, `latency*`, `spread*`, `tick_size`, `lot_size`, `commission*`, `maker_fee*`, `taker_fee*`, `partial_fill*`, `queue*`, `fill_probability*`. plan 에 등장하면 audit agent C 가 *해당 키 별 코드 wire-in 검증* 을 수행해야 함.
- **KNOWN_FLAW_IDS** (이미 알려진 결함, 재보고 차단):
  - `<plan-stem>.results.md` 존재 시 본문에서 `## Known Limitations`, `## 결함 목록`, `## 알려진 결함`, `## Known Issues` 섹션을 찾아 헤딩 아래 bullet 의 ID 토큰 (예: `H#3`, `B#1`, `C#2`, `bug-001`, 임의 prefix 허용) 을 추출. 부재 시 빈 리스트.

위 메타를 한 블록의 마크다운 표로 보고에 포함할 것 (사용자가 어떤 경로로 audit 가 돌았는지 1초 안에 확인 가능하도록).

ENGINE_ROOT 결정 실패 → audit 미실행, "audit skipped: engine 경로 미식별" 한 줄 출력 후 §6 종료.

#### 5.2. 3 sub-agent 병렬 호출

한 메시지 안에서 3개 모두 spawn (foreground). 각 agent prompt 의 공통 헤더 (placeholder 는 §5.1 결과로 치환):

```
당신은 backtest 시뮬레이션 충실도 감사관이다.

목표 (사용자 원문):
"백테스트 논리 에러 있는지 탐색해줘. 사용자 의도를 백테스트로 변환하는 과정
집중적으로 보고, 백테스트 시뮬레이션으로 거래가 실거래 프로세스와 비슷하게
일어날 수 있는지 확인해줘. 전체 프로세스를 병렬 에이전트 호출해서 진행."

위 의도를 다음 두 축으로 구조화:
1) 사용자 의도 (plan 본문 + 외부 reference 박제본 — 존재 시) 를 backtest 코드로 옮기는 과정의 *논리 결함*. plan 의 spec/식/경계 정의를 ground truth 로 삼고, 코드가 그 의도를 충실히 구현했는지 검증. reference 박제본이 있으면 부가적 ground truth 로 활용.
2) 시뮬레이션이 *실거래 프로세스* 와 합치하지 않는 갭.

plan 절대경로:        {PLAN_ABS_PATH}
REFERENCE_ROOT:       {REF_ROOT_ABS_OR_NONE}
ENGINE_ROOT:          {ENGINE_ROOT_ABS}
STRATEGY_ROOT:        {STRATEGY_ROOT_ABS_OR_NONE}
METRICS_ROOT:         {METRICS_ROOT_ABS_OR_NONE}
현실성 의심 config keys: {CONFIG_FIELDS_LIST_OR_EMPTY}
이미 알려진 결함 ID (재보고 금지): {KNOWN_FLAW_IDS}

규약:
- 위에 명시된 plan + 디렉토리만 read. 그 외 파일 금지.
- REFERENCE_ROOT == none 이면 reference-parity 항목은 N/A. 내부 일관성 + 실거래 갭만 점검.
- 새로 발견한 결함만 보고. 알려진 결함의 *연관 측면* 발견 시 알려진 ID 와 차이를 명시.
- 추측 금지. 코드/plan 본문 인용으로만 판정.
- 각 finding 에 file:line 인용 + 메커니즘 + 추정 영향 + 제안 fix.
- 보고 분량 1500단어 이내, 한국어. 발견 못 하면 "no findings" 명시.

severity 분류 (필수):
- REAL_BUG_INFLATE  — 실재 버그, 결과를 알파-인플레 방향으로 왜곡
- REAL_BUG_DEFLATE  — 실재 버그, 결과를 deflate (보수적) 방향
- REAL_BUG_NEUTRAL  — 실재 버그, 방향성 mixed/측정-only
- REFERENCE_PARITY     — reference 의 의도와 backtest 의 의도가 의미상 분기 (REFERENCE_ROOT 존재 시에만)
- REALISM_GAP       — spec 상 합법, 실거래 mechanics 와 분기
- METRIC_SEMANTIC   — 산출 metric 의 정의/단위/분모 격차
- DOC_GAP           — 결함 아닌 문서 보강 필요

각 finding 형식:
**[ID] {short title}** — {severity}
- Refs: {file:line} (engine/strategy/metrics) / {file:line} (reference, 있으면)
- Mechanism: {plan spec / 사용자 의도 (또는 reference 의도, 있으면) vs 실제 backtest 구현 차이}
- Impact: {qual + (가능하면) quant 추정 — 영향 trade 수, 영향 PnL/sharpe 추정 폭}
- Fix: {단일 변수 fix proposal — 한 PR 분량}

ID 는 자유 prefix + 일련번호 (예: A#1, B#2). 알려진 ID 와 충돌 회피.
```

각 agent 의 *고유* 임무. 카테고리는 일반 backtest 도메인에 보편적이며, 프로젝트 특화 기호 (예: 특정 함수명, 상태 라벨, 거래소명) 를 가정하지 않는다. agent 는 각 카테고리에 해당하는 *프로젝트 코드의 대응 위치* 를 직접 발견해야 함.

##### 5.2.a Signal / Entry conversion fidelity (agent A)
다음 카테고리만 점검:
- **Signal-time look-ahead bias** — 신호 생성/평가 시점에 그 봉 *이후* 데이터를 직접 또는 사전계산 feature 를 통해 간접 참조하는지. 인디케이터/feature 의 rolling window 가 미래 봉을 포함하는지.
- **Reference 신호 정의 vs backtest 신호 추출** (REFERENCE_ROOT 존재 시) — reference 가 signal/event 를 정의하는 규칙과 backtest 의 signal 추출 결과가 동치인지. 필터·랭킹·dedup 규칙의 분기.
- **Entry timing semantics** — signal bar 자체에 진입 vs 다음 봉 open vs 다른 정책. gap 봉 처리 (signal close 기준 target 이 다음 봉 open 에서 이미 deep ITM/OTM 일 때 거동).
- **Pre-validation chain** — entry 전 적용되는 필터 (거리/볼륨/시간대 등) 의 *적용 시점* (signal-time vs entry-time) 일관성.
- **Pre-computed forward-looking field leakage** — reference 또는 backtest 가 signal record 에 *미래 outcome* (target_reached, MFE, MAE, future_label 등) 을 사전 계산해 박는 경우, signal-time decision 이 이 필드를 직접/간접 사용하면 hidden look-ahead.
- **Indicator / feature warmup** — ATR/EMA/볼린저 등의 warmup 봉 수가 충족된 후부터 신호가 발화하는지. 첫 봉부터 신호가 나오면 NaN-driven 또는 partial-window 신호 의심.
- **Null-model / random benchmark 의 universe parity** — random sampler 가 actual 의 filter chain 과 *동치인 후보 집합* 에서 표본 추출하는지. universe size mismatch 는 분포 비교를 인공적으로 압축/팽창.
- **Liquidity / volume / session blackout 누락** — 저유동성 시간대, 거래소 점검 윈도우, funding settle 직전 등에서 신호 차단 정책 부재.
- **Multi-event 동시 발화 처리** — 같은 봉에 여러 신호가 발화할 때의 dedup/선택 규칙. 임의 순서 의존성.
- **Same-bar exit + 신규 entry over-trading edge** — 한 봉 안에서 청산 → 같은 봉의 새 신호로 ENTRY_PENDING 가능 여부. 실거래 latency 모순.
- **State machine 결정성** — signal/pending/in-position 전이 그래프가 deterministic 한지, race-condition / missing-state edges 존재 여부.

##### 5.2.b Exit / PnL conversion fidelity (agent B)
다음 카테고리만 점검:
- **Hit / breach 정의 정합성** — 가격 band 와 candle [low, high] 비교 규칙, direction-aware 여부, 동등성 기호 (`>=` vs `>`) 처리. 이미 알려진 결함의 *추가 측면* 만 보고, 동일 측면은 skip.
- **Fill price vs hit price 정합성** — hit 는 어디서 인정하고 PnL 은 어디 가격으로 박는지 (band edge vs band center vs candle.high/low vs target.price). 두 가격이 일치하지 않으면 인플레/deflate.
- **Horizon / time-stop exit price** — reference 가 horizon expire 시 사용하는 가격 (close vs MAE worst-point vs signal close 등) 과 backtest 의 선택이 일치하는지.
- **Hard-stop / forced-stop fill** — gap-through 봉 (stop level 을 지나쳐 open 되는 봉) 에서 stop_price 박힘 여부. 실제로는 더 나쁜 가격 fill 이지만 backtest 는 stop_price 그대로 박는 경우.
- **Boundary 강제 청산 (fold/window/EOL)** — 라벨링 정합 (fold_boundary vs data_end vs eol), 가격 정책 (close vs market), n_trades 합산 invariant.
- **Holding period off-by-one** — exit_index − entry_index 가 reference horizon 정의와 일치하는지. `>=` vs `>` 한 봉 차이.
- **MAE / MFE per-trade 필드 populate** — Trade dataclass 에 선언만 되고 engine 이 채우지 않는지. 채우지 않으면 percentile 보고가 silent zero.
- **Fee 모델 granularity** — round-trip 단일값 vs (entry/exit 분리, maker/taker 분리). entry leg 의 실제 주문 타입 (market vs limit) 과 적용 fee 일관성.
- **Equity update 공식** — fixed-notional (PnL_usd / initial_equity) vs compounding (`equity *= 1+net`). 채택된 모델이 reference 와 일치하는지, sharpe/calmar 의미가 어느 쪽인지.
- **Risk-adjusted metric annualization** — sharpe/sortino/calmar 의 `periods_per_year` 인자가 *return series 의 빈도* (per-trade / per-bar / per-day) 와 일치하는지. 빈도 mismatch 는 metric 을 √(ratio) 배로 인플레/deflate.
- **Drawdown granularity** — equity series 가 trade-close-only 인지 bar-by-bar mark-to-market 인지. trade-only 면 intra-trade DD 가 보고에서 제거되어 Calmar/MDD 인플레.
- **Exit-reason invariant** — `sum(exit_reason_counts) == n_trades` 가드 존재 여부. 미래 reason 추가 시 silent drop 위험.
- **Data window timestamps 정합성** — `data_window.since/until` vs `equity_curve.ts[0]/ts[-1]` 가 같은 의미 (bar open vs close, warmup 포함 여부) 인지.

##### 5.2.c Realistic execution simulation gaps (agent C)
다음 카테고리만 점검:
- **Marginal-friction config 의 wire-in 검증** — §5.1 의 `CONFIG_FIELDS` 각 키마다 `grep -rn "{key}" {ENGINE_ROOT} {STRATEGY_ROOT}` 로 *선언 외의 read 사용처* 가 존재하는지 검증. 선언만 있고 어떤 fill/PnL 산출 경로에서도 안 읽히면 **REAL_BUG_INFLATE — config-only ghost** 로 보고. 대표 후보: slippage, funding, latency, partial_fill, fill_probability, queue_priority, spread.
- **Limit fill probability** — limit order 가 단순 *price touch* (1-tick wick) 만으로 100% 체결 인정되는지. 큐 우선순위 / 큐 앞 잔량 / partial fill 모델의 부재.
- **Maker vs taker fee 분리** — limit fill 은 maker (low/rebate), market entry 는 taker (high). 단일 round-trip fee 는 둘 중 어느 쪽에 보수적인지 명시 부재.
- **Tick / lot size rounding** — target/stop/entry 가격이 거래소 tick 으로 snap 되는지. 자산별 tick 메타데이터 보유 여부.
- **Bid/ask spread 모델** — close-only 가격으로 매수/매도 차이를 흡수하지 않으면 spread 가 0 으로 박힘.
- **Signal → order latency** — bar close → 다음 봉 open 사이의 latency 가 0 으로 가정되는지. 실제 50~500ms (또는 분 단위) latency 의 영향 평가.
- **Compounding vs fixed-notional risk-budget 의미** — fixed-notional 시 equity 대비 노출 비율이 시간에 따라 변함 → MDD 가 인공 압축. compounding 시 의미는 다름.
- **Single-position 제약 opportunity cost** — already-in-position 으로 인해 폐기되는 신호 비율 (counter 가 있으면 정확 보고). t1_hit_rate 등 분모가 *executed subset* 임을 명시 안 하면 generalization 오해.
- **Multi-asset / cross-symbol survivorship** — exp 가 현재 listed 자산만 사용하면 delisted/blowup 자산 제외 → 결과 generalization 오버스테이트.
- **Data 무결성** — timezone (UTC vs local), open_time vs close_time 의 의미 박제, 결측봉 처리 (forward-fill vs drop) 의 명시 여부.
- **거래소 maintenance / halt / suspended trading** — 모델링 부재 시 그 구간이 "정상 trading" 으로 백테스트 됨.

#### 5.3. Audit 합산 보고

3 agent 결과 수신 후, plan-review 본체 보고 *아래에 추가* 출력. 형식:

```markdown
---

## 백테스트 변환 충실도 audit

### Audit 메타
| 항목 | 경로/값 |
|---|---|
| plan | {plan-basename} |
| REFERENCE_ROOT | {abs path or "none"} |
| ENGINE_ROOT | {abs path} |
| STRATEGY_ROOT | {abs path or "n/a"} |
| METRICS_ROOT | {abs path or "n/a"} |
| 검사한 friction config keys | {list or "none"} |
| 알려진 결함 ID 차단 수 | {count} |

### 결론 (한 문장)
신규 결함 {N}건 (REAL_BUG_INFLATE {n} / REAL_BUG_DEFLATE {n} / REAL_BUG_NEUTRAL {n} / REFERENCE_PARITY {n} / REALISM_GAP {n} / METRIC_SEMANTIC {n} / DOC_GAP {n}). 누적 결과 인플레 추정: {대/중/소/없음}.

### 신규 결함 (severity 별 정렬)

#### 🔴 REAL_BUG_INFLATE
- **[ID] {title}** — refs / mechanism / impact / fix (5 lines max each)

#### 🟡 REAL_BUG_NEUTRAL / REAL_BUG_DEFLATE / REFERENCE_PARITY
- ...

#### 🟢 REALISM_GAP / METRIC_SEMANTIC / DOC_GAP
- ...

### Cross-agent confirmation
2개 이상 agent 가 동일 결함 또는 인접 결함 보고 시 별도 명시 (신뢰도↑).

### 권장 fix 우선순위
1. {가장 큰 영향 결함} — 재실행 비용 / 단일변수 분리 가능성 명시
2. ...

### Already-known 결함 cross-check
agent 보고가 알려진 ID 와 *연관 측면* 을 짚는 경우, 차이점만 한 줄.
```

```

### 출력 형식 규약 (audit 추가)
- 모든 file 인용은 `[basename:LINE](relative/path#LLINE)` 마크다운 링크 (cwd 기준 상대경로).
- reference 박제본 인용도 동일.
- audit 결과로 results.md / plan / 다른 파일을 *수정하지 않는다*. 보고만 출력.
- 사용자가 보고에서 결함을 results.md (또는 동등 위치) 의 Known Limitations 에 추가하길 원하면 별도 요청 후 수정.

### 6. 종료
보고 출력 후 종료. plan 파일도 results 파일도 수정하지 않는다.
