---
plan_id: 002
version: 1
date: 2026-05-10 (Asia/Seoul)
status: draft
based_on:
  - 001
  - B001_linear-2pt
scope: full-stack (closed-form spline baseline + 조건부 submission)
exp_ids:
  - S001_cspline-natural-full
  - S002_cspline-notaknot-full
  - S003_cspline-window-grid
  - S004_smoothing-spline-tuned
---

# plan-002 v1 — Cubic Spline Interpolation Baseline (closed-form, no learning)

## §0. 한 줄 목적

> **학습 없이 3차 스플라인 보간법(Cubic Spline Interpolation)으로 11개 좌표를 부드러운 곡선에 피팅한 뒤 +80 ms 지점을 외삽하는 *문헌 검증된* baseline 4 변형 (S001~S004) 을 plan-001 floor (B001 cv_mean_eucl=0.01294, public LB 0.60) 에 대해 정량 비교하고, *4 변형 모두 dacon public LB 에 제출해 4 점수를 회수* 함으로써 보간/평활/윈도우 변형의 CV-LB 우열을 박제한다. LB 점수 4개 회수는 본 plan 의 *의무 산출* 이며, 미회수 시 G_final 종료 불가.**

---

## §0.5 Quick Reference (autonomous loop 매 turn 읽는 section)

### 합격 기준 (G-gate sequence)

- 모든 4 exp 의 **5-fold CV mean_eucl + per-axis MAE + hit_rate@{0.05,0.10,0.20,0.50} m** 가 registry + run dir 에 기록.
- 4 exp 모두 NaN/Inf 0건 (산술 안정성). 위반 시 `spline_numerical` severe.
- best of {S001..S004} mean_eucl 가 **B001 floor 0.01294 ± 1 fold-σ (= 0.01294 ± 0.00058)** 와 비교된 결과가 results.md 에 명시.
- **4 무조건 LB 제출 (필수, skip 불가)**: S001~S004 4 exp 모두 test 10,000개 예측 → `runs/baseline/{exp_id}/submission.csv` 4개 생성 + dacon public leaderboard 에 *반드시* 4회 제출 + 4 LB 점수 회수. 1일 5/일 한도 내에서 4 슬롯 사용 (1 슬롯은 contingency 예비). 제출 우선순위 (유망도 순) = S004 → S003 → S001 → S002. CV mean_eucl 와 무관하게 실행 (CV ≠ LB metric, hit_rate@1cm 는 별도 정보).
- **LB 점수 4개 results frontmatter 박제 의무**: 4 점수 모두 `plans/plan-002-cubic-spline.results.md` 의 `lb_scores` dict 에 기록되어야 G_final 종료 가능. 미회수 (예: budget 부족) 시 `status: partial` 로 마감 후 다음 일자 carry-over → 4 점수 채워질 때 비로소 `all_complete` 갱신.

### G-gates

- G0: STAGE 0 인프라 commit chain 완료 (cubic spline predictor + runner method dispatch + tests green) [TODO]
- G1: STAGE 1 S001 (natural BC, full 11pt), S002 (not-a-knot BC, full 11pt) 결과 기록 [TODO]
- G2: STAGE 2 S003 per-axis (window × BC) grid 결과 기록 [TODO]
- G3: STAGE 3 S004 smoothing spline (per-axis s grid) 결과 기록 [TODO]
- G_final: 4 submission.csv 생성 + 4 LB 제출 + LB 점수 회수 + results.md 작성 [TODO]

### Commit chain (next-up)

| # | type | spec section | status |
|---|---|---|---|
| c1 | code | `src/baselines/cubic_spline.py` — `predict_cspline(X, window, bc_type, t_target)`, `predict_cspline_per_axis(X, configs)`, `tune_per_axis_cspline(X, y, grid, k=5)`, `predict_smoothing_spline(X, s_per_axis, t_target)`, `tune_per_axis_smoothing(X, y, s_grid, k=5)`. spec @ §4 | [TODO] |
| c2 | code | `src/run.py` 확장 — cfg `method` 키 도입 (default `polyfit` ← 후방호환), `cspline` / `smoothing_spline` 분기. spec @ §4.4 | [TODO] |
| c3 | test | `tests/test_cubic_spline.py` — synthetic linear/quadratic 정확 외삽, NaN/Inf 부재, B001-동등 input 에서 finite output. spec @ §4.5 | [TODO] |
| G0 | gate | `pytest -q tests/` exit 0; `python -c "from src.baselines.cubic_spline import predict_cspline; print('ok')"` 성공; 기존 B001~B004 backward-compat smoke (configs 그대로 재실행 1회 → registry 동일 cv_mean_eucl) | [TODO] |
| c4 | exp S001 | `configs/baseline/S001_cspline-natural-full.yaml` + run + `runs/baseline/S001_cspline-natural-full/{summary,history,run.log,config.snapshot}` + registry append. spec @ §5 | [TODO] |
| c5 | exp S002 | `configs/baseline/S002_cspline-notaknot-full.yaml` + run + registry. spec @ §5 | [TODO] |
| G1 | gate | S001, S002 summary.json + registry 행 존재; cv_mean_eucl 유한 (NaN/Inf 0); 두 exp 모두 mean_eucl 기록 | [TODO] |
| c6 | exp S003 | `configs/baseline/S003_cspline-window-grid.yaml` + run + registry. spec @ §6 | [TODO] |
| G2 | gate | S003 summary 에 `final_chosen_per_axis` (axis × (window, bc_type)) 기록, registry 행 존재 | [TODO] |
| c7 | exp S004 | `configs/baseline/S004_smoothing-spline-tuned.yaml` + run + registry. spec @ §7 | [TODO] |
| G3 | gate | S004 summary 에 `final_chosen_s_per_axis` 기록, registry 행 존재 | [TODO] |
| c8 | sub-gen | `src/submit.py` 확장 (cspline / smoothing_spline method 분기 추가, polyfit 후방호환 보존) + S001~S004 4 exp 의 test 10k 예측 → `runs/baseline/{S00x}/submission.csv` 4개 (sample_submission 스키마 동일). c8 commit 에 4 csv 모두 포함. **스키마 검증 fail 시 `submission_schema_fail` severe**. spec @ §8.1 | [TODO] |
| c8b | sub-lb | **`dacon-submit` skill 사용해 dacon public LB 에 4 회 제출 (의무)** — 1일 5/일 budget 내 S004 → S003 → S001 → S002 순으로 `Skill(skill="dacon-submit", args="<runs/baseline/{S00x}/submission.csv> <exp_id>")` 4회 호출. 각 호출 응답으로 LB 점수 회수해 `analysis/plan-002/lb_log.md` 의 4 행에 (exp_id, submitted_at KST, lb_score, filename) 기록. skill 부재 시 `dacon_submit_skill_missing` severe → 사용자에게 skill 설치 escalate. **점수 4개 모두 회수될 때까지 G_final 진입 불가**. spec @ §8.2 | [TODO] |
| c9 | docs | `analysis/plan-002/results.md` + `plans/plan-002-cubic-spline.results.md` (frontmatter 에 4 LB 점수 dict 포함). spec @ §N+2 | [TODO] |
| G_final | gate | 위 모두 완료 + §0.5 [TODO]→[DONE] sync (§12.6 blacklist 의 유일한 예외) | [TODO] |

### Plan-specific severe (WORKFLOW.md §12.3 default 위 추가분)

- `spline_numerical`: 어떤 exp 의 `pred` 에 NaN/Inf 1건 이상 → scipy 사용/BC 설정/window 경계 버그.
- `cspline_too_good`: best of {S001..S004} cv_mean_eucl < 0.005 → B001 floor 대비 비현실적 개선, 데이터/leakage/평가 함수 버그 의심 (sanity check).
- `backward_compat_drift`: G0 의 backward-compat smoke 에서 B001~B004 의 cv_mean_eucl 가 registry 기존 값과 4 자리 이상 어긋남 → 새 method 도입이 기존 polyfit 분기 회귀시킨 것.
- `lb_unsubmitted`: G_final 진입 시점에 `lb_scores` dict 의 4 점수 중 1개 이상 미회수 + carry-over 사유 미박제 → 본 plan 의 *의무 산출* 위반. 사용자에게 escalate.
- `submission_schema_fail`: 4 submission.csv 중 1개라도 sample_submission 스키마 검증 fail → c8 미완료, 재생성 또는 코드 버그 수정.
- `dacon_submit_skill_missing`: c8b 진입 시 `dacon-submit` skill 이 가용 skill 목록에 부재 → 사용자에게 skill 설치 escalate (telegram alert: "plan-002 c8b: dacon-submit skill 미설치. `.claude/skills/dacon-submit/` 또는 `~/.claude/skills/dacon-submit/` 에 skill 정의 후 plan resume").

### Plan-specific paths (WORKFLOW.md §12.5/§12.6 추가/제외)

- whitelist 추가: `analysis/plan-002/**` (특히 `analysis/plan-002/lb_log.md` — 4 LB 점수 박제), `runs/baseline/S00*/**`, `configs/baseline/S00*.yaml`, `src/baselines/cubic_spline.py`, `tests/test_cubic_spline.py`
- whitelist 확장 (기존 파일 수정 허용): `src/run.py` (method dispatch 만 — `predict_for_config` 분기 추가), `src/submit.py` (cspline / smoothing_spline method dispatch 추가, polyfit 후방호환 보존), `src/baselines/__init__.py` (re-export)
- blacklist 추가: `runs/baseline/B00*/**` (plan-001 산출 — 절대 수정 금지), `configs/baseline/B00*.yaml` (plan-001 pre-reg)

### Decision-note 사용 예 (자율 결정 시 commit msg 박제)

- `decision-note: spec-default — scipy.interpolate.CubicSpline default extrapolate=True 채택 (외삽 명시 활성화)`
- `decision-note: spec-default — UnivariateSpline 의 s_grid 권장 default {0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1} 채택 (s=0 = 보간, s>0 = 평활)`
- `decision-note: spec-default — clamped BC 의 end-derivative 추정 = (last - prev)/40ms (B001 의 외삽 가설과 동일)`
- `decision-note: spec-default — LB 제출 순서 S004→S003→S001→S002 (유망도 순, §8.2 박제). budget 소진 시 위쪽부터 회수`
- `decision-note: data-partial — dacon 5/일 budget 잔여 < 4 → 잔여 슬롯만 제출, 미제출 exp 는 다음 일자 carry-over 후 다시 시도`

---

## §1. 배경

### 1.1 plan-001 결과 인계 (registry + plan-001-polyfit-baseline.results.md)

| exp_id | window | degree | cv_mean_eucl | per_axis_mae [x, y, z] | hit@0.10 |
|---|---|---|---|---|---|
| **B001_linear-2pt** | 2 | 1 | **0.01294** ± 0.00058 | [0.0070, 0.0071, 0.0050] | 0.992 |
| B002_linear-3pt | 3 | 1 | 0.01576 | [0.0087, 0.0087, 0.0059] | 0.989 |
| B003_quad-3pt | 3 | 2 | 0.02017 | [0.0105, 0.0110, 0.0085] | 0.976 |
| B004_per-axis-grid | (2,2,2) | (1,1,1) | 0.01294 | [0.0070, 0.0071, 0.0050] | 0.992 |

- public LB hit_rate (B001 제출): **0.60** (plan-001 results frontmatter `lb_score_public`).
- plan-001 §1.3 의 핵심 통찰: **11점 전체 polyfit 은 2점 등속 외삽보다 모두 나쁨.** 오래된 시점의 노이즈가 fit 을 dominate. 최적은 *최근 소수 시점 + 낮은 degree* 영역.

### 1.2 본 plan 의 가설 출발점 — Cubic Spline Interpolation

문헌 (사용자 제시 — 이집트숲모기 Aedes aegypti 비행 행동 모니터링 2023~2024 논문):
- Mask R-CNN (탐지) + **3차 스플라인 보간법 (궤적 추정)** 결합 → 약 96 % 정확도 (논문 보고치).
- 3차 스플라인은 11점 polyfit 과 다른 inductive bias: **piecewise C² cubic** — 각 segment 가 local 이라 *원리상* 오래된 시점의 노이즈가 fit 끝 (extrapolation 영역) 에 직접 영향을 덜 줄 수 있다.

비판적 prior:
- plan-001 결과는 "전역 fit 이 노이즈를 모은다" 를 강하게 시사. cubic spline 도 *전역 BC* (not-a-knot, natural) 를 선택하면 동일 함정 가능성.
- *Smoothing* spline (s>0) 은 입력 노이즈 자체를 흡수해 외삽 안정성이 다를 수 있음 — plan-001 에 *부재* 한 inductive bias.

### 1.3 본 plan 의 결정적 근거

세 가지 sub-가설을 한 plan 안에서 *동시* 검증한다:

- **H1 (full-window 보간)**: 11pt natural / not-a-knot 보간 + extrapolate → polyfit 11pt 와 비슷한 mean_eucl 영역 (0.03~0.05) 일 것. 즉 B001 대비 patently 열등.
- **H2 (windowed 보간)**: window ∈ {4, 5, 7, 11} × BC ∈ {natural, not-a-knot, clamped} 그리드에서 작은 window + clamped BC 가 B001 에 *근접* 가능 (= window=2 등속 외삽의 일반화). but ≤ B001 보장 X.
- **H3 (smoothing spline)**: 입력 노이즈 흡수 효과로 적절한 s 선택 시 B001 floor 를 *위협* 가능. 본 plan 의 가장 유망 후보 — plan-001 에 부재한 inductive bias.

→ 셋 다 *기록 자체* 가 다음 plan 의 의사결정 anchor (예: smoothing 이득 0 → neural model plan 이 smoothing 우회 가능; smoothing 이득 ↑ → 전처리 plan 우선).

---

## §2. Scope

### §2.1 In-scope

| 항목 | 값 |
|---|---|
| 모델 | per-axis cubic spline (interpolating + smoothing) closed-form |
| 학습 | 없음 (CV 는 hyperparameter selection only — plan-001 §2.1 과 동일 원칙) |
| 보간 hyperparameter | window ∈ {4, 5, 7, 11}, bc_type ∈ {natural, not-a-knot, clamped} (window > 3 보장; 3 점 cubic spline 은 degenerate) |
| 평활 hyperparameter | UnivariateSpline `k=3`, `s` ∈ {0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1} per-axis CV-tuned |
| target time | +80 ms (스펙 고정) |
| primary dev metric | mean 3D Euclidean distance (m) |
| 보조 metric | per-axis MAE, hit_rate @ {0.05, 0.10, 0.20, 0.50} m (plan-001 §3.3 와 동일) |
| CV | 5-fold, seed=42, `src/io.py:kfold_split` 그대로 재사용 (plan-001 §3.1 deterministic split) |

### §2.2 Out-of-scope (절대 안 함)

| 항목 | 이유 |
|---|---|
| neural model (LSTM/Transformer/MLP) | 별도 plan. 본 plan 은 closed-form spline 검증만 |
| polyfit 분기 회귀/재실행 | plan-001 의 결과는 영구. backward-compat smoke 만 통과시키면 됨 |
| Kalman filter / Savitzky-Golay 입력 평활 | 별도 plan (smoothing spline 과 inductive bias 다름) |
| target horizon ≠ +80 ms | 스펙 위반 |
| ensemble (S001..S004 + B001 평균 등) | 별도 plan — baseline 단일 모델만 |
| hit-radius 추정 (leaderboard probing) | 5/일 한도 절약, 별도 plan |
| test 데이터를 fit 에 사용 | 대회 룰 위반 |
| `s` 값을 leaderboard 점수로 selection | overfitting LB. CV mean_eucl argmin 만 |

---

## §3. 사전 등록 (Pre-registration)

### §3.1 Fold split

- plan-001 §3.1 과 **완전 동일** 함수 (`src/io.py:kfold_split(ids, k=5, seed=42)`) 그대로 재사용. 재구현 금지.
- 같은 fold = 같은 val ids → S00x 와 B00x 의 fold-level 비교가 가능 (paired comparison).

### §3.2 합격 기준

| 조건 | 정의 |
|---|---|
| A. 인프라 정상 | `pytest tests/` green; `from src.baselines.cubic_spline import predict_cspline` import OK; G0 의 backward-compat smoke (B001~B004 cv_mean_eucl 4자리 일치) |
| B. 산술 안정성 | 4 exp 모두 fold-level 및 OOF prediction 에 NaN/Inf 0건 (위반 시 `spline_numerical` severe) |
| C. 결과 무회귀 (sanity) | best of {S001..S004} cv_mean_eucl ≥ 0.005 (위반 시 `cspline_too_good` severe) |
| D. 비교 박제 | results.md 에 4 exp × {cv_mean_eucl ± std, per-axis MAE, hit_rate@0.10, vs B001 fold-paired Δ} 표가 존재 |
| E. 4 LB 제출 (의무, skip 불가) | 4 exp (S001~S004) 모두 submission.csv 생성 (sample_submission 스키마 100 % 일치) + **dacon public leaderboard 에 4 회 실제 제출** + 4 점수 results frontmatter `lb_scores` dict 에 기록. 5/일 budget 부족 시 유망도 순 (S004→S003→S001→S002) 으로 회수 가능한 만큼 제출, 잔여는 다음 일자 carry-over (status `partial` + 사유 박제, 익일 재진입 시 `all_complete` 갱신). **4 점수 미회수 + carry-over 미박제 시 `lb_unsubmitted` severe — G_final 종료 불가** |

### §3.3 평가 점수 / 집계 (plan-001 §3.3 와 동일)

- per fold metric: `mean_eucl`, per-axis MAE, hit_rate(r) for r ∈ {0.05, 0.10, 0.20, 0.50}
- per exp metric: 5 fold mean ± std (median 아님 — small CV)
- exp 비교: cv_mean_eucl argmin (1차), 동률 시 작은 window / 작은 s 우선 (2차 tie-breaker)
- B001 vs S00x 비교: **same-fold paired Δ** 도 추가 보고 (5 fold 각각의 mean_eucl 차이 평균 + 부호 일관성).
  - **산출 위치**: c9 (results.md 작성 commit) 안에서 계산. 전용 코드 파일 추가 안 함 — `analysis/plan-002/results.md` 본문의 paired comparison 섹션에 표 형태 (5 fold × {B001, S00x, Δ}) + summary 줄 (mean Δ ± SE, 부호 일관성 비율) 박제.
  - **데이터 소스**: plan-001 의 fold-level metric 은 `runs/baseline/B00*/history.json` 또는 `summary.json` 의 `fold_metrics` 키 (§4.2 schema 와 동일) 에서 read. 실제 키 schema 가 §4.2 와 다른 경우 fallback 규칙: (a) 우선 `summary.json` 의 `fold_metrics[i].mean_eucl` 시도, (b) 부재 시 `history.json` 안에서 `fold` int + `mean_eucl` float 키를 자동 탐색, (c) 둘 다 부재 시 `backward_compat_drift` severe → 사용자 escalate (plan-001 schema 변동 시점이며 임의 추정 금지).
  - **Δ 정의**: `Δ_fold[i] = S00x.fold_metrics[i].mean_eucl - B001.fold_metrics[i].mean_eucl` (음수 = S00x 가 B001 보다 좋음). 전체 평균 `mean Δ`, 부호 일관성 = `sum(sign(Δ_fold) == sign(mean Δ)) / 5`.

---

## §4. STAGE 0 — 인프라 (G0)

### §4.1 산출 파일

- `src/baselines/cubic_spline.py` (신규):
  - `predict_cspline(X: np.ndarray, window: int, bc_type: str, t_target: int = 80, timesteps=TIMESTEPS_MS) -> np.ndarray`:
    - X: (n, 11, 3). return shape (n, 3). 마지막 `window` 점만 사용. 변수명 통일: 본 §4 본문의 모든 `t[-window:]` 류 표기는 `timesteps[-window:]` 의 shorthand.
    - bc_type ∈ {"natural", "not-a-knot", "clamped"}.
    - **scipy 호출 contract** (axis 별 sample-loop **default 강제**; vectorized 변종 사용 안 함):
      ```python
      # default sample-loop (안전, scipy 버전 비의존). 10k × 3 axes ≈ 30k 호출, < 30s.
      assert window >= 4, "cubic spline 은 window >= 4 필요 (3점은 degenerate)"
      n = X.shape[0]
      out = np.zeros((n, 3), dtype=np.float64)
      for i in range(n):
          for ax in range(3):
              y = X[i, -window:, ax]                    # shape (window,)
              bc = bc_type if bc_type != "clamped" else ((1, (y[1]-y[0])/40), (1, (y[-1]-y[-2])/40))
              cs = scipy.interpolate.CubicSpline(timesteps[-window:], y, bc_type=bc, extrapolate=True)
              out[i, ax] = cs(t_target)
      return out
      ```
      - **vectorized 변종은 본 plan 에서 사용 금지** — clamped batched derivative (shape `(n,)`) 의 scipy 지원이 버전 의존이라 reproducibility 위협. 본 plan 의 모든 cspline 산출은 위 sample-loop 으로 통일 (성능이 부족하면 별도 plan 에서 다룸). decision-note: spec-default — cspline 호출 sample-loop only.
    - clamped 일 때 양 끝 derivative = 차분 추정: 좌측 `(y[1] - y[0]) / 40`, 우측 `(y[-1] - y[-2]) / 40`. `40` = TIMESTEPS_MS 의 인접 step 차이 (40 ms; 일반화 시 `(timesteps[-1]-timesteps[-2])` 사용).
  - `predict_cspline_per_axis(X, configs_per_axis: list[(window, bc_type)], t_target=80)`:
    - axis 별 다른 (window, bc_type) 적용. plan-001 의 `predict_per_axis` 와 동일 구조.
    - **window 가드 forwarding**: 각 cell `(w, bc)` 에 대해 `predict_cspline` 호출 시 동일 `assert w >= 4` 가 즉시 발동 (per-axis wrapper 가 별도 가드 안 함 — 호출 path 가 결국 `predict_cspline` 으로 수렴하므로).
  - `tune_per_axis_cspline(X_tr, y_tr, grid: list[(window, bc_type)], t_target=80, k: int = 5, seed: int = 42) -> (chosen: list[tuple[int, str]] of len 3, errors: dict[tuple[int, str], np.ndarray of shape (3,)])`:
    - **호출 contract**: 함수는 *내부에서* `kfold_split(ids_of_X_tr, k=k, seed=seed)` 로 추가 k-fold 를 돌려 grid 평가. 각 cell (w, bc) 에 대해 inner k-fold 의 val MAE 평균을 axis 별로 산출 → `errors[(w, bc)]` (shape (3,)). `chosen[i] = argmin_{(w,bc)} errors[(w,bc)][i]`. plan-001 의 `tune_per_axis` 동일 패턴.
    - 호출자 (runner): outer 5-fold 의 *각 outer fold 의 train 부분만* (X_tr, y_tr) 로 본 함수 호출 → outer val 에 chosen 적용해 fold OOF prediction 산출. inner k 와 outer k 는 같은 5 (총 5×5=25 inner fit per outer fold).
  - `predict_smoothing_spline(X, s_per_axis: list[float], t_target=80, k=3, timesteps=TIMESTEPS_MS, s_grid: list[float] | None = None) -> np.ndarray`:
    - axis 별 sample-loop: `spl = scipy.interpolate.UnivariateSpline(timesteps, X[i, :, axis], k=k, s=s_per_axis[axis], w=None)` 후 `spl(t_target)`. 11pt 전체 사용. (UnivariateSpline 은 (n, 11) batch 미지원 — sample 별 새 객체 필수. 10k × 3 axes = 30k spline fit, 추정 < 60s.)
    - **fit 실패 fallback chain** (NaN/Inf 0건 invariant 충족용; 순서대로 시도):
      1. `UnivariateSpline(...)` 가 `dfitpack.error` / `RuntimeError` 발생 시 → 같은 sample/axis 에 대해 `s_grid` (호출자 제공) 안의 *현 s 보다 큰* 값들을 오름차순으로 재시도. `s_grid=None` 또는 next 후보 부재 시 step 2 로 진행.
      2. `CubicSpline(timesteps, y, bc_type="not-a-knot", extrapolate=True)(t_target)` 으로 보간 결과 사용. 이 호출이 fail 또는 비-finite 산출 시 step 3.
      3. **last-resort finite 보장**: `out[i, ax] = float(X[i, -1, ax])` (마지막 입력값을 그대로 사용 — t_target 위치의 등속-zero 외삽 — 항상 finite).
    - fallback 발동 시 매 sample/axis 단위 plain log 1줄 + summary.json `oof_summary.smoothing_fallback_count` 카운터 증가 (step 별로 1/2/3 카운트 분리). 최종 결과는 *항상 finite* (last-resort 보장).
  - `tune_per_axis_smoothing(X_tr, y_tr, s_grid: list[float], t_target=80, k: int = 3, n_folds: int = 5, seed: int = 42)`:
    - 호출 contract: 함수 내부에서 `kfold_split(ids_of_X_tr, k=n_folds, seed=seed)` 로 inner k-fold. 각 s ∈ s_grid 에 대해 inner val 의 axis MAE 평균 → axis 별 argmin → `chosen_s_per_axis: list[float] of len 3`. (위 함수의 `k` 는 spline degree 고정 3, `n_folds` 가 CV fold 수 — 변수명 분리.)

- `src/baselines/__init__.py` (수정 — re-export):
  - 기존 `window_polyfit` re-export 유지. `cubic_spline` 의 4 함수도 추가.

- `src/run.py` (수정 — method dispatch):
  - cfg 에 새 키 `method` ∈ {"polyfit", "cspline", "smoothing_spline"} (default = "polyfit" → 기존 B001~B004 후방호환).
  - `predict_for_config(X, cfg)` 가 `cfg["method"]` 로 분기:
    - "polyfit" → 기존 로직 (`predict` / `predict_per_axis`).
    - "cspline" → `predict_cspline` 또는 `predict_cspline_per_axis`. cfg `per_axis` 가 list 면 axis 별 (window, bc_type), `"tune"` 이면 fold 내 grid tuning, scalar 면 global (window, bc_type).
    - "smoothing_spline" → `predict_smoothing_spline`. cfg `s_per_axis` 가 list 면 axis 별, `"tune"` 이면 fold 내 s_grid tuning.
  - tune 분기는 plan-001 의 polyfit per_axis tune 과 동일 패턴: per-fold 내부 train 으로 grid → val 로 평가 → fold OOF prediction.
  - summary.json 에 method 별 적절한 chosen 정보 기록 (`final_chosen_per_axis` 또는 `final_chosen_s_per_axis`).

- `tests/test_cubic_spline.py` (신규):
  - synthetic 1: t=[-400..0]/40 ms 의 11pt linear (slope 1) → cubic spline (any BC) 의 t=80 평가가 절대오차 < 1e-9.
  - synthetic 2: 11pt quadratic → cubic spline (**not-a-knot, clamped 만**) 의 t=80 평가가 절대오차 < 1e-9 (이 두 BC 는 quadratic 정확 표현). natural BC 는 끝점 f''=0 가정과 quadratic (f''=2a≠0) 가 충돌하므로 본 케이스에서 *제외* — natural 의 거동은 합성 #1 (linear, natural BC 도 만족) 에서 이미 검증.
  - smoothing spline s=0 의 보간 영역 일치: UnivariateSpline(k=3, s=0) 의 입력 11점 (t∈[-400, 0]) 재평가가 원본과 절대오차 < 1e-9 (s=0 = 보간 정의). 외삽점 t=+80 에서의 CubicSpline 과의 등치는 BC 차이 (UnivariateSpline 의 FITPACK BC vs CubicSpline 의 not-a-knot/natural/clamped) 때문에 *보장되지 않음* → 외삽 등치 검증은 안 함.
  - finite output: random (n=8, 11, 3) input 에 대해 NaN/Inf 0건.
  - clamped BC end-derivative 가 정의대로 적용되는지 (synthetic linear 에서 slope 정확 회복).

### §4.2 산출물 schema (run dir summary.json — registry 와 1:1, plan-001 §4.2 확장)

```json
{
  "exp_id": "S00X_...",
  "method": "cspline" | "smoothing_spline",
  "n_train": 10000,
  "n_val_per_fold": 2000,
  "k": 5,
  "fold_metrics": [{"fold": 0, "mean_eucl": ..., "per_axis_mae": [..,..,..], "hit_rate": {...}, "chosen_per_axis": ... | null}, ...],
  "cv_mean_eucl": ...,
  "cv_std_eucl": ...,
  "cv_per_axis_mae": [...],
  "cv_hit_rate": {"0.05": ..., "0.10": ..., "0.20": ..., "0.50": ...},
  "oof_summary": {"smoothing_fallback_count": {"step1_s_retry": 0, "step2_cubicspline": 0, "step3_last_input": 0}, "n_oof_samples": 10000},
  "final_chosen_per_axis": [[4, "clamped"], [5, "natural"], [4, "clamped"]] | null,
  "final_chosen_s_per_axis": [1e-4, ...] | null,
  "config": {...}
}
```

### §4.3 종료 조건 (G0)

- `pytest -q tests/` exit 0 (기존 테스트 + 신규 `test_cubic_spline.py`).
- `python -c "from src.baselines.cubic_spline import predict_cspline; from src.run import run_baseline; print('ok')"` 성공.
- backward-compat smoke: B001~B004 의 4 config 를 `src.run.main` 로 재실행 → registry 마지막 4 행의 cv_mean_eucl 가 plan-001 기록과 **`abs(new - old) < 1e-4`** 충족 (= "4자리 일치" 의 정량 정의). 위반 시 `backward_compat_drift` severe.
  - 재실행 산출은 새 행으로 append 됨 (registry append-only). 이는 invariant 준수.

### §4.4 src/run.py 변경 범위 (백워드 호환성)

- 변경 키: `predict_for_config` 1개 함수만 method-dispatch.
- 변경 전 `B00*.yaml` 들에 `method` 키가 없으므로 default "polyfit" 이 발동 → 기존 동작 보존.
- `tune_per_axis` 분기 내부도 `cfg["method"]` 로 polyfit/cspline 선택. cspline tune 의 grid 는 cfg `grid` (list of `[window, bc_type]`).
- summary.json 의 `final_chosen_per_axis` 키 schema 통일: cspline 의 경우 **항상 `[window: int, bc_type: str]` 순서** (예: `[4, "clamped"]`). polyfit 의 경우 `[window: int, degree: int]` 순서. 둘 다 첫 elem = window, 두 번째 elem = method-specific 두번째 변수. results.md 에 method 명시.
- **config 키 `k` 의 의미 박제**: cfg 의 `k` = CV fold 수 (= 5; plan-001 §3.1 와 동일). smoothing spline 함수 내부의 `k` (= spline degree, 3 으로 hardcoded) 와 *별개의 변수*. config 에 spline degree 를 노출하지 않음 — 본 plan 의 모든 smoothing spline 은 `k=3` (cubic) 고정.

### §4.5 tests/test_cubic_spline.py 합격 조건

- 5개 테스트 케이스 (위 §4.1 의 5개) 모두 pass.
- 기존 `test_window_polyfit.py`, `test_io.py`, `test_eval.py` 회귀 0건.

---

## §5. STAGE 1 — Full-window cubic spline (G1)

### §5.1 S001_cspline-natural-full

| 항목 | 값 |
|---|---|
| type | baseline |
| baseline_id | B001_linear-2pt |
| 단일 변경 변수 | method polyfit(w=2, d=1) → cspline(window=11, bc_type=natural) |
| method | per-axis natural cubic spline through 11 points → evaluate at t=+80 ms |
| config | `configs/baseline/S001_cspline-natural-full.yaml`: `{method: cspline, window: 11, bc_type: natural, t_target: 80, k: 5, seed: 42}` |
| 기대 runtime | < 60 s (10k × 3 axes × 5 fold = 150k spline 평가) |
| 성공 기준 | summary 기록 완료, cv_mean_eucl 유한 |
| 가설 | H1 — natural BC 의 "끝 곡률 0" 가정이 외삽을 *flat* 하게 만들어 0.02~0.05 영역 |

### §5.2 S002_cspline-notaknot-full

| 항목 | 값 |
|---|---|
| baseline_id | S001 |
| 단일 변경 변수 | bc_type natural → not-a-knot |
| method | per-axis not-a-knot cubic spline through 11 points → t=+80 ms |
| config | `configs/baseline/S002_cspline-notaknot-full.yaml`: `{method: cspline, window: 11, bc_type: not-a-knot, t_target: 80, k: 5, seed: 42}` |
| 기대 runtime | < 60 s |
| 성공 기준 | summary 기록 완료, cv_mean_eucl 유한 |
| 가설 | H1 — not-a-knot 은 끝점 근방 cubic 을 그대로 연장 → S001 보다 외삽이 "튀는" 경향. mean_eucl 가 더 클 것 |

### §5.3 G1 종료 조건

- `runs/baseline/S001_cspline-natural-full/summary.json`, `runs/baseline/S002_cspline-notaknot-full/summary.json` 모두 존재.
- registry 에 두 행 append. cv_mean_eucl finite (NaN/Inf 0).

---

## §6. STAGE 2 — Windowed cubic spline grid (G2)

### §6.1 S003_cspline-window-grid

| 항목 | 값 |
|---|---|
| baseline_id | best of {S001, S002} (cv_mean_eucl argmin; **동률 시 S001 (natural BC) 우선** — alphabetical / 작은 exp_id) |
| 단일 변경 변수 | full-window → per-axis (window, bc_type) grid |
| grid | (window, bc_type) ∈ {(4,natural), (4,not-a-knot), (4,clamped), (5,natural), (5,not-a-knot), (5,clamped), (7,natural), (7,not-a-knot), (7,clamped), (11,natural), (11,not-a-knot), (11,clamped)} (12 cells) |
| selection | per-axis: 5-fold CV val per-axis MAE argmin (plan-001 §6.4 와 동일 패턴) |
| config | `configs/baseline/S003_cspline-window-grid.yaml`: `{method: cspline, per_axis: "tune", grid: [[4,"natural"], [4,"not-a-knot"], [4,"clamped"], ...], t_target: 80, k: 5, seed: 42}` |
| 기대 runtime | < 5 min (12 cells × 3 axes × 5 fold ≈ 180 group fit, axis-별 sample-loop 포함) |
| 성공 기준 | summary 의 `final_chosen_per_axis` 가 axis 별 (window, bc_type) 1쌍 — 결과 기록 자체가 목적 |
| 가설 | H2 — small window + clamped 가 우세할 것. clamped 의 right-derivative = (last-prev)/40 ms 는 B001 의 등속 외삽과 *수학적으로 유사* — 따라서 B001 에 근접 가능 |

### §6.2 G2 종료 조건

- `runs/baseline/S003_cspline-window-grid/summary.json` 존재.
- summary 에 `final_chosen_per_axis` (axis 0/1/2 × (window, bc_type)) 기록.
- registry 1행 append.

---

## §7. STAGE 3 — Smoothing spline tuned (G3)

### §7.1 S004_smoothing-spline-tuned

| 항목 | 값 |
|---|---|
| baseline_id | best of {S001, S002, S003} (cv_mean_eucl argmin; **동률 시 S001 → S002 → S003 순 alphabetical 우선**) |
| 단일 변경 변수 | interpolating cubic spline → smoothing cubic spline (k=3, s>0 가능) |
| method | per-axis `scipy.interpolate.UnivariateSpline(t, x, k=3, s=s_axis)` for s_axis ∈ s_grid; t=+80 ms 평가 |
| s_grid | {0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1} (6 cells) |
| selection | per-axis: 5-fold CV val per-axis MAE argmin |
| config | `configs/baseline/S004_smoothing-spline-tuned.yaml`: `{method: smoothing_spline, s_per_axis: "tune", s_grid: [0, 1.0e-5, 1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1], t_target: 80, k: 5, seed: 42}` |
| 기대 runtime | < 10 min (6 s × 3 axes × 5 fold × 10k sample × spline fit ≈ 9M fit. UnivariateSpline 은 fit 빠름 — 1 fit ≈ 10~30 µs → 90~270 s 영역) |
| 성공 기준 | summary 의 `final_chosen_s_per_axis` 가 axis 별 s 1개 — 결과 기록이 목적 |
| 가설 | H3 — 입력 노이즈 (per-axis std ≈ 0.025~0.035 m on label-last_input) 흡수 효과로 적절한 s 가 B001 에 근접 또는 위협. **본 plan 의 가장 유망 후보** |

### §7.2 G3 종료 조건

- `runs/baseline/S004_smoothing-spline-tuned/summary.json` 존재 + `final_chosen_s_per_axis` 기록.
- registry 1행 append.

---

## §8. STAGE 4 — 4 Submission + LB 회수 + Results (G_final)

본 stage 의 설계 원리: CV mean_eucl 와 LB hit_rate@1cm 는 *상관 있지만 동일하지 않은* metric. 따라서 4 spline 변형 (S001~S004) 각각이 *서로 다른 inductive bias* 를 LB 에 노출시켜 다음 plan 의 의사결정 anchor 를 4 점 박제한다 (1 점 만으론 LB-CV 상관 추정 불가).

### §8.1 4 Submission 생성 (c8)

1. **`src/submit.py` 확장** — 현재 polyfit 만 분기. `predict_with_exp(exp_id)` 가 `cfg["method"]` 로 dispatch:
   - `"polyfit"` → 기존 로직 (B00x 후방호환 — 변경 없이 동작 보장).
   - `"cspline"` → `predict_cspline` / `predict_cspline_per_axis` (cfg `per_axis` 가 list 면 axis 별, `"tune"` 이면 summary.json 의 `final_chosen_per_axis` 사용).
   - `"smoothing_spline"` → `predict_smoothing_spline` (`s_per_axis` 가 `"tune"` 이면 summary.json 의 `final_chosen_s_per_axis` 사용).
2. S001~S004 4 exp 각각에 대해 test 전체 10,000 sample 예측 → `runs/baseline/{S00x}/submission.csv`. closed-form 이라 train 재fit 불필요. **S003/S004 의 hyperparameter 확정 default = full-train 재tune** (decision-note: spec-default — fold 별 chosen 이 axis 마다 다를 수 있어 fold 합산 의결이 ambiguous; full-train (10k 전체) 위에 *내부 5-fold CV 한 번* (= `tune_per_axis_cspline` / `tune_per_axis_smoothing` 의 default n_folds=5, seed=42) 으로 grid 재평가 후 axis 별 argmin 채택. summary.json 의 `final_chosen_per_axis` / `final_chosen_s_per_axis` 는 이 full-train 재tune 결과를 박제. 5-fold split seed 는 plan-001 §3.1 와 동일 = 42 → outer CV 와 다른 split 이지만 paired comparison 은 outer fold 결과로 별도 산출. plan-001 B004 와 동일 패턴).
3. **스키마 assert** (plan-001 §7.1 동일, 모든 csv 마다 통과 필수):
   - `rows == 10000`
   - `columns == ["id", "x", "y", "z"]`
   - `NaN.sum() == 0`, `np.isinf(...).sum() == 0`
   - `dtype["x|y|z"] == float64`
   - `set(submission.id) == set(sample_submission.id)`
   - 위반 시 `submission_schema_fail` severe.
4. c8 commit 에 4 submission.csv (`runs/baseline/S00*/submission.csv`) 모두 포함. csv 는 가벼움 (10k 행 × 4 column ≈ 200 KB / 파일) → text VCS 추적 OK (artifacts/ 미사용).

### §8.2 4 LB 제출 + 회수 (c8b) — *의무, skip 불가*

> **본 단계는 plan-002 의 핵심 산출. 4 LB 점수 미회수 시 G_final 종료 절대 불가.** csv 만 만들고 끝나는 plan 이 아님.

**제출 우선순위 (유망도 순, 박제)**: S004 → S003 → S001 → S002.
- 1순위 **S004** (smoothing tuned): plan-001 에 부재한 noise-absorbing bias. CV ↔ LB 차이가 가장 클 후보 → 1순위로 회수해 다음 plan 의사결정 가치 최대화.
- 2순위 **S003** (per-axis grid): per-axis 동역학 차이가 LB 에서 보상되는지 (B004 가 CV 동률이라 plan-001 에서 미제출 → 본 plan 이 처음 회수).
- 3순위 **S001** (natural full): 전역 BC + 11pt 보간이 polyfit 11pt (LB 미회수) 와 어떻게 갈리는지 — *flat extrapolation* prior 의 LB 가치.
- 4순위 **S002** (not-a-knot full): scipy default BC. S001 대비 *aggressive extrapolation* prior — 두 BC 의 LB 격차 자체가 다음 plan (Kalman/평활 전처리) 의사결정 anchor.

**제출 메커니즘** — *primary path 는 `dacon-submit` skill*:

- **`dacon-submit` skill 호출** (의무 사용):
  - c8b 진입 시 autonomous loop 가 `Skill(skill="dacon-submit", args="...")` 4회 호출. 각 호출 = 1 exp 의 csv 업로드 + LB 점수 회수.
  - skill 인자 형식 (skill 명세에 따름; 미정 시 `decision-note: spec-default — args="<csv_path> <exp_id> [memo]"` 추정 사용):
    1. `Skill(skill="dacon-submit", args="runs/baseline/S004_smoothing-spline-tuned/submission.csv S004_smoothing-spline-tuned 'plan-002 S004 smoothing tuned'")`
    2. `Skill(skill="dacon-submit", args="runs/baseline/S003_cspline-window-grid/submission.csv S003_cspline-window-grid 'plan-002 S003 per-axis grid'")`
    3. `Skill(skill="dacon-submit", args="runs/baseline/S001_cspline-natural-full/submission.csv S001_cspline-natural-full 'plan-002 S001 natural full'")`
    4. `Skill(skill="dacon-submit", args="runs/baseline/S002_cspline-notaknot-full/submission.csv S002_cspline-notaknot-full 'plan-002 S002 not-a-knot full'")`
  - skill 응답으로 LB 점수 회수 → `analysis/plan-002/lb_log.md` 행 append + registry `notes` 갱신 + `plans/plan-002-cubic-spline.results.md` frontmatter `lb_scores` dict 갱신.
  - 각 제출 사이 30~60 s 텀 (skill 자체가 rate limit 핸들러를 가지면 그대로 의지).
  - **skill 부재 또는 호출 fail 시**: `dacon_submit_skill_missing` severe trigger → telegram alert ("plan-002 c8b: dacon-submit skill 미사용. skill 정의 또는 install 필요. 우선순위 S004→S003→S001→S002. 점수 회수 후 lb_log.md 채우고 resume") → autonomous loop 일시정지. 사용자가 skill 설치 + 4 csv 업로드 + lb_log.md 4 행 기록 + commit 후 다음 turn 에서 resume.
  - skill 의 응답이 LB 점수 비공개/지연 (예: dacon 의 채점 큐 지연) 인 경우 → skill 명세에 따라 retry 또는 deferred return. autonomous loop 는 skill 정책에 따름.

- **fallback (skill 호출 자체가 막힌 경우)**: 사용자가 dacon.io 대회 페이지 (236716) 제출 메뉴에서 4 csv 순차 업로드 + 점수 회수 + lb_log.md 4 행 기록 + commit. autonomous loop 는 다음 turn 시작 시 `git log -1 --grep "plan-002 c8b"` hit 으로 resume 감지.

**Budget 운영**:
- dacon public LB **5/일**. 5 슬롯 중 4 사용, **1 슬롯은 contingency 예비** (예: 스키마 fail 후 재제출, plan-003 긴급 ablation, 또는 LB 점수 회의적 시 재제출 검증).
- 잔여 budget < 4 인 경우 (예: 같은 날 다른 plan 의 제출이 이미 있음) 유망도 순으로 회수 가능한 만큼만 제출, 잔여 exp 는 다음 일자 carry-over → 별도 commit (c8b-carry-1, c8b-carry-2 …) 으로 추가 LB 점수 append. status 는 그동안 `partial`, 4개 모두 채워지면 `all_complete` 로 results.md frontmatter 갱신.
- 모든 제출 후 LB hit_rate (반경 비공개, 분모 = test sample 수) 회수.

**기록 위치 (3 곳 모두 박제 의무)**:
- `analysis/plan-002/lb_log.md`: 4행 표 (`exp_id | submitted_at (KST) | lb_score | submission_filename`) + 분기 (A/B) + budget 운영 사유.
- registry 의 해당 exp 행 `notes` 컬럼에 `lb_score=0.XX` 추가 (`registry append-only` invariant 위반 X — `notes` 는 rationale 필드. 또는 새 행 `type=correction, corrects=<exp_id>` 으로 append 가능).
- `plans/plan-002-cubic-spline.results.md` frontmatter `lb_scores: {S001: 0.XX, S002: 0.XX, S003: 0.XX, S004: 0.XX}` dict — **4 키 모두 채워져야 status `all_complete`**.

**G_final 진입 차단 조건**:
- `lb_scores` dict 4 점수 중 1개라도 미회수 + carry-over 사유 박제 부재 → `lb_unsubmitted` severe trigger → autonomous loop 정지, 사용자 escalate.

### §8.3 Results.md 산출 (c9)

- `analysis/plan-002/results.md` 본문:
  - 종합 표: 5 행 (B001 + S001~S004) × (method, key hp, cv_mean_eucl ± std, per-axis MAE, hit@{0.05,0.10,0.20,0.50}, **lb_score**, runtime).
  - per-experiment 분석 (plan-001-results 의 형식 그대로).
  - **B001 vs S001~S004** paired comparison: same-fold Δ (5 fold 각각의 mean_eucl 차이 + 부호 일관성).
  - S003 의 axis 별 chosen (window, bc_type) 분해.
  - S004 의 axis 별 chosen s 분해 + s_grid 의 axis MAE 곡선.
  - H1, H2, H3 검증/기각 명시.
  - **CV ↔ LB 상관 분석**: 5 점 (B001 + S001~S004) 의 (cv_mean_eucl, lb_score) 산점 + Spearman ρ. CV 가 LB 의 proxy 로 신뢰 가능한지 박제 — 다음 plan 들의 selection 전략 anchor.
  - 다음 plan 후보 *enumeration only* (가중 spline, Kalman 전처리, neural model, ensemble 등; 우선순위 X — local 권한).

- `analysis/plan-002/lb_log.md`: 4 행 표 (`exp_id | submitted_at (KST) | lb_score | submission_filename`) + budget 운영 사유 (잔여 슬롯, carry-over 발생 시 일자).

- `plans/plan-002-cubic-spline.results.md`: WORKFLOW.md §6 frontmatter (`plan_id, finished_at, status, exp_ids_completed, exp_ids_skipped, best_exp_id_cv, best_exp_id_lb, submission_paths: list[4], lb_scores: {S001, S002, S003, S004}, lb_metric, lb_submitted_at_first, lb_submitted_at_last`) + 각 exp 의 (status, started_at, duration, 핵심 metric, best path, baseline diff, lb_score, 특이사항).

### §8.4 G_final 종료 조건 (의무 list — 누락 시 G_final 진입 불가)

- 4 exp summary.json + registry 4 행 + 4 submission.csv (스키마 검증 통과) + **4 LB 점수 (lb_log.md + results.md frontmatter `lb_scores` dict)** + analysis/plan-002/results.md + plans/plan-002-cubic-spline.results.md 모두 commit.
- §0.5 의 모든 [TODO] → [DONE] 마킹 (commit hash 포함, §12.6 blacklist 예외).
- 4 LB 점수 모두 회수 완료. **carry-over 발생 시**: status `partial` 로 마감 + 다음 일자 추가 commit (c8b-carry-N) 으로 잔여 LB 점수 보강 → 4 점수 모두 채워진 시점에 results.md frontmatter `status: all_complete` 로 갱신하는 추가 commit.
- *`lb_scores` dict 의 4 키 중 1개라도 비어 있고 carry-over 사유 미박제 시 `lb_unsubmitted` severe trigger → G_final 종료 절대 불가.*

---

## §N+1. 작업량 회계

| 단위 | 수 |
|---|---|
| code commit (c1, c2) | 2 |
| test commit (c3) | 1 |
| exp commit (c4, c5, c6, c7) | 4 |
| sub-gen commit (c8) | 1 (4 submission.csv 동시 포함) |
| sub-lb commit (c8b) | 1 (4 LB 점수 + lb_log.md. budget carry-over 발생 시 동일 chain 의 추가 commit 으로 분할 가능) |
| docs commit (c9) | 1 |
| **총 commit** | **10** (carry-over 시 +1~+3) |
| G-gate | 5 (G0, G1, G2, G3, G_final) |
| 총 fit 호출 (예상) | S001+S002 = 2 × 5 fold × 3 axes = 30 spline×10k samples. S003 = 12 cells × 5 fold × 3 axes × 10k. S004 = 6 s × 5 fold × 3 axes × 10k × UnivariateSpline. 총 wall time 추정 < 20 min |

---

## §N+2. results.md 필수 항목

| 항목 | 내용 |
|---|---|
| frontmatter | `plan_id=002, finished_at (KST), status, exp_ids_completed, exp_ids_skipped, best_exp_id_cv, best_exp_id_lb, submission_paths (list[4]), lb_scores ({S001:..., S002:..., S003:..., S004:...}), lb_metric, lb_submitted_at_first, lb_submitted_at_last` |
| 본문 per exp | 상태, started_at, duration, cv_mean_eucl±std, per-axis MAE, hit_rate@4 radii, best run dir path, baseline diff vs B001, **lb_score**, 특이사항 |
| 종합 표 | 5 행 (B001 + S001~S004) × (method, hp, cv_mean_eucl, per_axis_mae, hit@0.10, **lb_score**, runtime) |
| paired comparison | B001 vs S001~S004 의 same-fold Δ (5 fold), 부호 일관성, fold-σ 와 비교 |
| H1/H2/H3 verdict | 각 가설별 *CV* + *LB* 양축 검증/기각/부분기각 |
| best 선택 사유 (CV) | argmin cv_mean_eucl + tie-break (작은 window, 작은 s) |
| best 선택 사유 (LB) | argmax lb_score |
| **CV-LB 상관** | 5 점 산점 + Spearman ρ + (CV winner ≠ LB winner) 발생 시 그 격차의 의미 분석 |
| submission 결과 | 4 exp 모두 제출, LB 4 점수, budget 운영 (carry-over 발생 여부 + 사유) |
| 다음 plan 후보 | enumeration only (가중 spline, Kalman 전처리, neural seq2seq, ensemble 등). 우선순위 X — local 권한 |

---

## §N+3. 통계 함정 & caveats

1. **paired Δ 가 fold-σ 보다 작으면 noise** — same-fold 비교라 paired t-test 기준 신뢰도가 더 좋지만, 5 fold 만으로는 ±0.0006 영역의 차이는 noise. results.md 에 fold-σ 와 함께 명시.
2. **smoothing spline 의 s 에 대한 selection bias** — 6 s × 3 axes = 18 candidates, 5-fold mean 으로 선정. plan-001 §N+3 #4 와 동일 caveat — hold-out 검증 X (LB 가 사실상 대신).
3. **clamped BC 의 derivative 추정 자체가 외삽 결과 좌우** — 본 plan 의 default = 차분 추정. 다른 추정 (가중 평균 등) 은 별도 plan.
4. **scipy 버전 dependence** — `CubicSpline` 의 `extrapolate=True` 동작은 scipy ≥ 1.0 안정. `python -c "import scipy; print(scipy.__version__)"` 결과를 G0 commit msg `decision-note: dep-install` 에 박제.
5. **UnivariateSpline 의 s=0 edge case** — 11pt 보간 spline 이 되며 noise 흡수 안 됨. CubicSpline (not-a-knot) 와 t=+80 평가 거의 동일해야 함 → §4.5 의 sanity test 가 이를 검증.
6. **단위 m 가정** — plan-001 §N+3 #5 와 동일 (competition-overview §3.3 근거).
7. **B001 산점 비교의 paired 가정** — `kfold_split` 가 deterministic + seed 동일이라 fold 멤버십 100 % 일치. 따라서 paired 비교 valid.
8. **LB metric 자체의 분산** — public LB 는 test 10k 의 일부 subset (반경/분모 비공개) 에 대한 hit_rate. 두 제출의 LB 차이가 0.005 영역 (n~1000 가정 시 SE ≈ √(p(1-p)/n) ≈ 0.015) 보다 작으면 noise. results.md 의 CV-LB 상관 분석에서 명시.
9. **5/일 budget 운영의 변동성** — 본 plan 외 제출 (탐색 / 다른 plan) 이 같은 날 발생 시 4 슬롯 미확보 가능. plan 시작 전 dacon 제출 페이지에서 잔여 budget 확인 후 진행 (autonomous loop 도 G_final 직전에 체크 — 미달 시 carry-over 명시 후 다음 일자 재진입).
10. **CV winner ≠ LB winner 가능성** — hit_rate@1cm 는 cv_mean_eucl 가 잡지 못하는 *tail behavior* (큰 오차 sample 의 비율) 을 측정. CV best 가 LB best 와 다를 수 있음 — 이는 plan 결함이 아니라 *정보 자체* (다음 plan 의 metric proxy 결정 anchor).

---

## §N+4. 변경 이력

- v1 (2026-05-10): 초안. plan-001 결과 (B001 floor 0.01294, public LB 0.60) 인계 + 문헌 (Aedes aegypti monitoring) 의 cubic spline trajectory estimation 근거를 바탕으로 4 exp closed-form spline baseline 설계. 초안은 *조건부 1 회 LB 제출* 모드.
- v1.1 (2026-05-10): 제출 정책 변경 — *조건부 1 회* → **4 무조건 LB 제출** (S004→S003→S001→S002 유망도 순). CV ≠ LB metric 인 점을 활용해 4 inductive bias 의 LB 신호를 동시 회수해 다음 plan 의 selection 전략 anchor 강화. §0.5 합격기준 / §3.2 E / §8 / §N+1~§N+3 동기화.
- v1.2 (2026-05-10): 제출을 *의무 산출* 로 강제 — §0 한 줄 목적 / §0.5 합격기준 / §3.2 E / §8.2 에 "skip 불가" 명시, severe trigger `lb_unsubmitted` + `submission_schema_fail` 신설, §8.2 에 auto-submit / user-escalation 분기 박제, `src/submit.py` 확장을 c8 에 명시 추가, whitelist 에 `src/submit.py` / `analysis/plan-002/lb_log.md` 추가.
- v1.3 (2026-05-10): 제출 mechanism 을 `dacon-submit` skill 호출로 박제 — §0.5 c8b + §8.2 의 primary path 가 `Skill(skill="dacon-submit", args=...)` 4 회. skill 부재 시 `dacon_submit_skill_missing` severe trigger 신설. skill args 형식 추정 + fallback (수동 dacon.io 업로드) 분기 명시.

---

## §N+5. 참조

- `plans/plan-001-polyfit-baseline.md` (선행 plan, 4 baseline + EDA 근거)
- `plans/plan-001-polyfit-baseline.results.md` (B001~B004 결과 + LB 0.60)
- `notes/competition-overview.md` (대회 사양, +80 ms target, hit-rate metric)
- `WORKFLOW.md` §4 (4-way token), §5 (plan obligations), §6 (results), §7 (run dir), §11 (handoff), §12 (autonomous protocol)
- `CLAUDE.md` (autonomous execution policy)
- 데이터: `data/{train,test}/*.csv` (각 11 row), `data/train_labels.csv`, `data/sample_submission.csv`
- 코드 인계: `src/io.py` (kfold_split deterministic, TIMESTEPS_MS, T_TARGET_MS), `src/eval.py` (summarize), `src/run.py` (method dispatch 확장 대상), `src/baselines/window_polyfit.py` (per_axis tune 구조 참조)
- 외부 라이브러리: `scipy.interpolate.CubicSpline` (bc_type ∈ {natural, not-a-knot, clamped}, extrapolate=True), `scipy.interpolate.UnivariateSpline` (k=3, smoothing factor s)
- 문헌 근거 (사용자 제시): Aedes aegypti 비행 행동 모니터링 (2023~2024) — Mask R-CNN + cubic spline trajectory estimation (∼96 % 보고치). 본 plan 은 *trajectory estimation* (보간) 부분을 +80 ms *외삽* 영역에 적용해 문헌 baseline 의 과제 적합성을 정량 측정.
