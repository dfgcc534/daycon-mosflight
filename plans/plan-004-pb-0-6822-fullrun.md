---
plan_id: 004
version: 3
date: 2026-05-11 (Asia/Seoul)
status: draft
based_on:
  - 001
  - 002
  - 003
  - notes/PB_0.6822 코드공유.ipynb
scope: full-stack (notebook code extraction + project data full-fit + 18×27 regime distribution audit + autonomous LB submission)
exp_ids:
  - P001_pb-0-6822-fullrun
lb_score: 0.6806
---

# plan-004 v3 — PB_0.6822 Notebook Full-Fit + 18×27 Regime Distribution Audit (server `cuda:1` 강제)

## §0. 한 줄 목적

> **사용자 narrative — 본 plan 의 검증 두 갈래** (`@notes/PB_0.6822 코드공유.ipynb` 인계):
>
> 1. **이 방법의 LB 확인**: 노트북 framework (27 물리 후보 + Attn-GRU selector + 18×27 regime bias + Tiny boundary corrector 2-stage) 의 cell 4/6 을 `src/pb_0_6822/{selector,boundary}.py` 로 추출 → project `data/` 에 적용 → full 5-fold 학습 + corrector full-fit → `dacon-submit` skill 자율 호출 (CLAUDE.md autonomous policy) → **이 방법이 우리 데이터에서 어떤 LB 점수를 내는가** 박제.
> 2. **regime × state 표의 sample 분포 확인**: notebook 의 `candidate_regime_bias()` 가 in-memory 로만 계산하는 18-regime × 27-candidate (= regime × candidate-state) empirical Bayes bias 표의 **train sample 분포** (regime 별 histogram + degenerate regime (sample<50) flag + (regime, candidate) hyper-specialized cell flag) 를 `analysis/plan-004/regime_distribution.{json,md}` 로 박제 (notebook 에 *없는* 새 검증).
>
>    **명료화**: §8 가 산출하는 표는 selector internal bias *value* 의 재추출이 아니라, **bias 표와 동일한 18×27 grid 위에서 (a) regime 별 train sample count, (b) (regime, candidate) hit-rate marginalize** 를 계산한 *audit-side reconstruction* (§8.1 spec L451-456 의 `hit_table[r,c] = (err[regimes==r] <= R_HIT)[c].mean()` 공식). bias value 자체는 노트북 in-memory 객체로만 존재하고 본 plan 은 그것을 호출/추출하지 않는다 (§4.0 L224 `candidate_regime_bias` 직접 호출 X 박제와 일치).
>
> **두 갈래 모두 의무 산출** — 하나라도 미달 시 G_final 종료 불가. **학습 device = `cuda:1` 강제** (server agent 의 1번 GPU 사용, Mac mps 학습 v1 시도 후 폐기).

---

## §0.5 Quick Reference (autonomous loop 매 turn 읽는 section)

### 합격 기준 (G-gate sequence)

- 노트북 cell 4 (~94KB SELECTOR_MAIN) + cell 6 (~22KB BOUNDARY_MAIN) 의 추출이 *완전* (CANDIDATES list 27개, fit_regime_bins/assign_regimes/candidate_regime_bias 등 핵심 함수 보존). 위반 시 `extraction_drift` severe.
- 1-fold smoke (`--fold-limit 1`, epoch=1) 의 cv hit-rate 가 finite + 추출 버그 catch (NaN/Inf/ImportError 없음). 위반 시 `selector_no_convergence` severe.
- Full 5-fold selector + boundary corrector full-fit 완료, 모든 metric finite (selector = 5-fold 전체 fold-level metric, corrector = `run_full.py --phase boundary` 호출 단위 metric. corrector 의 `--fold 0` 인자는 noteb cell 6 의 "primary fold" pointer 일 뿐 5-fold loop 자체는 모듈 default 동작에 위임 — §7 G3 csv schema 검증으로 자동 판정). 위반 시 `nn_numerical` severe.
- `submission_boundary_tiny_soft.csv` schema 가 `data/sample_submission.csv` 와 100% 일치 (row count, column names, id 순서). 위반 시 `submission_shape_mismatch` severe.
- **18-regime × 27-candidate 표 + sample 분포 박제 의무**: `analysis/plan-004/regime_distribution.{json,md}` 에 (a) regime별 sample count histogram, (b) 18×27 hit-rate table, (c) degenerate regime list (sample < 50), (d) hyper-specialized cell list 모두 기록. 미박제 시 `regime_unaudited` severe.
- **best 1 LB 제출 (필수, skip 불가, 자율 실행)**: `runs/baseline/P001_pb-0-6822-fullrun/submission.csv` → autonomous loop 가 `dacon-submit` skill 1회 호출 (사용자 승인 X) + 1 LB 점수 회수.
- **lb_score frontmatter 박제 의무**: 회수된 LB 점수가 §10 L568-571 의 3개 파일 (`plans/plan-004-pb-0-6822-fullrun.md` top-level, `plans/plan-004-pb-0-6822-fullrun.results.md`, `analysis/plan-004/results.md`) frontmatter 의 `lb_score` 필드에 **동시** 박제되어야 G_final 종료 가능 (AND 조건; 한 파일이라도 누락이면 G_final 미통과). 박제 commit 단위 = **단일 commit c11** (3 파일을 한 commit 에 함께 staged; plan-002/003 LB carry-over 패턴 답습). partial 분기 (`lb_score: TBD`) 도 동일 — 점수 도착 시 follow-up commit `c11.1` 에서 3 파일 *동시* 갱신. 미회수 시 `lb_unsubmitted` severe.

### G-gates

- G0: STAGE 0 인프라 (모듈 추출 + smoke import + 기존 tests backward-compat) [DONE]
- G1: STAGE 1 1-fold smoke 통과 (cv hit-rate finite, no extraction drift) [DONE]
- G2: STAGE 2 full 5-fold selector 학습 완료 (`oof_selector_scores.npz` + `test_selector_scores.npz` finite) [DONE]
- G3: STAGE 3 full boundary corrector + `submission_boundary_tiny_{soft,argmax}.csv` 생성 [DONE]
- G3.5: STAGE 3.5 18×27 regime distribution 분석 박제 (`analysis/plan-004/regime_distribution.{json,md}`) [DONE]
- G_final: `submission.csv` schema 검증 + dacon-submit 자율 호출 + lb_score 박제 + results.md [DONE — lb=0.6806]

### Commit chain (next-up)

| # | type | spec section | status |
|---|---|---|---|
| c1 | docs | `plans/plan-004-pb-0-6822-fullrun.md` 작성 (본 파일) | [DONE e74486b] |
| c2 | code | `src/pb_0_6822/__init__.py` + `selector.py` extract (notebook cell 4 → standalone module). argparse signature 보존. spec @ §4.1 | [DONE 7bc9cd7] |
| c3 | code | `src/pb_0_6822/boundary.py` extract (cell 6, **유일 수정**: `import train_tcn_gru_candidate_selector as base` → `from src.pb_0_6822 import selector as base`). spec @ §4.2 | [DONE c01b7d1] |
| c4 | test | `tests/test_pb_0_6822_smoke.py` — 모듈 import + `CANDIDATES len==27` + `TinyCorrectionNet` 인스턴스화. spec @ §4.3 | [DONE 0f82129] |
| G0 | gate | `pytest tests/test_pb_0_6822_smoke.py` + 기존 tests green (backward-compat) — 63 tests pass | [DONE] |
| c5 | code | `src/pb_0_6822/run_full.py` orchestrator + `configs/baseline/P001_pb-0-6822-fullrun.yaml` + `.gitignore` 1줄. spec @ §4.4 | [DONE 4023272] |
| c5.1 | fix | `src/pb_0_6822/selector.py` L1215 — for-epoch 루프 시작 시 `model.train()` 추가 (cudnn RNN backward eval-mode 버그 fix; smoke 1 epoch 통과지만 full 10 epoch 에서 epoch 2 backward 크래시 → 1라인 fix). decision-note: runtime-fix. | [TODO this commit] |
| c6 | exp smoke | 1-fold smoke (`run_full.py --smoke`) → `runs/baseline/P001_pb-0-6822-fullrun/smoke/`. spec @ §5 | [DONE b35307c re-verified] |
| G1 | gate | smoke summary finite, no extraction drift — selector_soft_hit=0.6441 boundary_soft_hit=0.6609 cuda:1 | [DONE] |
| c7 | exp selector | Full 5-fold selector (`--fold-limit 5`, no `--skip-full`, pre=10 fine=8 freeze=3 patience=4 epoch_plus=5). spec @ §6 | [DONE this commit] |
| G2 | gate | `oof_selector_scores.npz` + `test_selector_scores.npz` finite + shape OK — (10000,27) both | [DONE] |
| c8 | exp corrector | Full boundary corrector (`--make-test`, `--test-score-bank`, epochs=12 fine=8 patience=4). spec @ §7 | [DONE this commit] |
| G3 | gate | 2 csv 생성, finite, shape == sample_submission.csv — boundary OOF soft=0.6718 > selector baseline 0.6624 (corrector_no_convergence 미발생) | [DONE] |
| c9 | analysis | `analysis/plan-004/regime_distribution.py` → `regime_distribution.{json,md}`. spec @ §8 | [DONE this commit] |
| G3.5 | gate | 18 regime histogram + 18×27 hit table + degenerate flag + hyper-specialized cell 모두 박제 — degenerate=0, hyper_specialized=19 | [DONE] |
| c10 | sub-gen | `runs/baseline/P001_pb-0-6822-fullrun/submission.csv` = soft csv 사본 + schema 100% 일치 검증. spec @ §9 | [DONE this commit] |
| c11 | sub-lb | **`dacon-submit` skill 자율 호출** + `analysis/plan-004/lb_log.md` 박제 + **3 파일 frontmatter `lb_score` 동시 갱신** (`plans/plan-004-pb-0-6822-fullrun.md` top-level + `plans/plan-004-pb-0-6822-fullrun.results.md` + `analysis/plan-004/results.md`). spec @ §10 + §0.5 L42 AND 조건. | [DONE 416bf0e partial → c11.1 closed lb=0.6806] |
| c11.1 | sub-lb close | LB 점수 carry-over close — 3 파일 frontmatter `lb_score: TBD` → `0.6806` + status `partial` → `all_complete` 동시 갱신, lb_log.md / registry.csv 갱신. plan-003 R006 패턴 답습. | [DONE this commit] |
| G_final | gate | LB 점수 회수 + 모든 G-gate [DONE] + §0.5 sync — **lb=0.6806** | [DONE] |

### Plan-specific severe (WORKFLOW.md §12.3 default 위 추가분)

- `extraction_drift`: c6 smoke 의 cv hit-rate 가 NaN/Inf 또는 추출된 모듈에서 NameError/ImportError 발생 → cell-local 변수 (IPython display 등) 누락 의심.
- `selector_no_convergence`: selector train loss NaN/Inf 또는 OOF top-1 accuracy < 1/27 (= 0.037, 무작위 추측).
- `corrector_no_convergence`: boundary train loss NaN/Inf 또는 학습 후 **OOF soft hit-rate** (= `submission_boundary_tiny_soft.csv` 기반 holdout-fold 평균 hit@1cm) 가 *추가 보정 전 selector OOF soft hit-rate* (= `oof_selector_scores.npz` 의 argmax-free probability-weighted blend hit@1cm) 보다 낮음. (비교 split = OOF, metric = soft hit-rate. argmax csv 와 test-set 점수는 비교 대상 X.)
- `nn_numerical`: 학습 중 NaN/Inf 또는 gradient NaN.
- `regime_degenerate`: 18 regime 중 sample <50 인 regime ≥ 1 (notebook 에 없는 새 검증, **warn only, severe X** — 정보 박제만).
- `regime_unaudited`: G3.5 진입 시점에 `analysis/plan-004/regime_distribution.{json,md}` 미존재.
- `submission_shape_mismatch`: submission.csv shape ≠ sample_submission.csv (row count, column names, id 순서).
- `lb_unsubmitted`: G_final 진입 시점에 `lb_score` 미회수 + carry-over 사유 미박제.
- `dacon_submit_skill_missing`: c11 진입 시 `dacon-submit` skill 부재 → 사용자 escalate.
- `backward_compat_drift`: G0 의 기존 51 tests 가 검증하는 baseline exp (= **B001_linear-2pt** `cv_mean_eucl ≈ 0.01294`, §1.1 표 첫 행) 의 등록값과 4자리 이상 어긋남. 본 plan c2/c3 추출이 기존 모듈을 회귀시키지 않았음을 lock-in.

### Plan-specific paths (WORKFLOW.md §12.5/§12.6 추가/제외)

- whitelist 추가:
  - `src/pb_0_6822/__init__.py`, `src/pb_0_6822/selector.py`, `src/pb_0_6822/boundary.py`, `src/pb_0_6822/run_full.py`
  - `tests/test_pb_0_6822_smoke.py`
  - `configs/baseline/P001_pb-0-6822-fullrun.yaml`
  - `runs/baseline/P001_pb-0-6822-fullrun/**` (ckpt 는 `.gitignore` 제외 패턴 적용)
  - `analysis/plan-004/**` (특히 `regime_distribution.{py,json,md}`, `lb_log.md`, `results.md`)
  - `.gitignore` 1회 1줄 수정 (`runs/baseline/P001_*/ckpt/` 패턴 추가)
- blacklist 추가:
  - `notes/PB_0.6822 코드공유.ipynb` (원본 보존, 절대 수정 금지)
  - plan-001/002/003 산출 (`runs/baseline/{B,S,R}00*/**`, `configs/baseline/{B,S,R}00*.yaml`, `analysis/plan-{001,002,003}/**`)

### Decision-note 사용 예 (자율 결정 시 commit msg 박제)

- `decision-note: spec-default — extraction approach (notebook cell → .py module, papermill X) — src/* convention 정합성 + out_dir 제어 깔끔`
- `decision-note: spec-default — exp_id=P001_pb-0-6822-fullrun (P prefix = Public-baseline reimplementation)`
- `decision-note: spec-default — submission=soft csv 사본 (continuous probability-weighted blend, leaderboard 친화)`
- **`decision-note: spec-default — 학습 device = cuda:1 강제 (server agent 의 1번 GPU). plan-003 의 cuda:0 강제 패턴 답습하되 GPU 번호만 1번으로 변경. CUDA_VISIBLE_DEVICES 의존 X — run_full.py 의 --device 인자에 "cuda:1" 명시. 다중 GPU 환경에서도 1번만 사용해 결과 reproducibility 보장.`**
- `decision-note: spec-default — epoch budget = notebook default 의 ~70% (이전 v1 의 Mac mps 60min 가정에서 GPU 환경으로 갱신, 실제 wall-time 은 GPU 가속으로 더 단축 예상)`
- `decision-note: spec-default — regime_degenerate threshold = 50 samples (empirical Bayes shrinkage 신뢰성 영역)`
- `decision-note: spec-default — hyper_specialized threshold = (regime, cand) hit-rate 가 해당 regime mean 의 ratio ≥ 1.5 (positive over-specialization) OR (ratio ≤ 0.5 AND hit_rate ≥ 0.01) (negative under-specialization, low-end floor 0.01 로 random-zero cell 노이즈 컷). 비대칭 조건 의도 — §8.1 L463 코드 박제와 일치.`
- `decision-note: spec-default — 1-fold smoke 별도 dir (runs/.../P001*/smoke/) for full output 보호`
- `decision-note: spec-default — c11 dacon-submit 자율 호출 = 사용자 승인 X (CLAUDE.md autonomous policy)`
- `decision-note: spec-default — DATA_ROOT = repo/data/ (open.zip 자동 해제 if needed); notebook 의 LOCAL_DATA_ROOT 경로 무시`

---

## §1. 배경

### §1.1 plan-001/002/003 결과 인계 (registry 기반)

| exp_id | plan | method | cv_mean_eucl | LB |
|---|---|---|---|---|
| **B001_linear-2pt** | 001 | polyfit (w=2, d=1) | **0.01294** ± 0.00058 | **0.60** |
| S001_cspline-natural-full | 002 | cspline natural 11pt | 0.01742 ± 0.00071 | 0.4932 |
| R001_baseline-residual-gru | 003 | linear + GRU 잔차 + Huber | 0.01338 ± 0.00072 | (R006 으로 통합) |
| R006_combined-winners | 003 | R001 복제 (winning=0) | = R001 | **0.5688** |

핵심 인계:
- 가장 높은 LB = **B001 0.60**. R006 이 학습 모델임에도 R001 fallback (winning=0) 으로 LB 0.5688 → 오히려 closed-form 보다 약함.
- **노트북 PB_0.6822 (= 0.6822) 와 우리 best LB (0.60) 의 gap = 0.08** — 본 plan 의 잠재 회수 영역.
- 본 plan 의 목적은 *gap 회수* 자체보다 **노트북 framework 의 우리 데이터 적용 + 18×27 표 박제** (= 다음 plan 의 anchor 정보 확보).

### §1.2 노트북 framework 핵심 (대화 기반 정리)

(이미 사용자와의 사전 대화에서 정리 완료 — 자세한 내용은 `notes/PB_0.6822 코드공유.ipynb` cell 0 의 "Algorithm Notes: Physics Ladder" 참조)

1. 신경망에 *절대 좌표 책임 위임 X*. 27개 *물리 공식 후보 좌표* 가 절대 좌표 책임.
2. **Selector** (Attn-GRU): 27 후보 점수 logit + physics_bias (0.65) + regime_bias (0.45) 가산.
3. **18 regime × 27 candidate empirical Bayes 표**: (speed × curvature × speed_slope) quantile binning 으로 sample → regime, 각 (regime, candidate) cell 에 hit-rate 통계 + shrinkage.
4. **Corrector** (Tiny MLP): Frenet local frame 에서 ±0.6cm cap, zero-init delta, boundary weighting (1cm 근처 sample 에 loss 가중치 집중) 으로 selector 가 놓친 1~2cm zone sample 회수.
5. **2-stage sequential training**: selector → OOF score bank 저장 → corrector full-fit. OOF 로 누수 차단.

본 plan 의 검증 포인트 — §0 narrative 두 갈래 ↔ stage 매핑:

| 갈래 | 검증 명제 | 검증 stage | 합격 산출 |
|---|---|---|---|
| 1. LB 확인 | notebook framework 을 우리 데이터에 1:1 적용 시 LB 점수가 어떻게 나오는가? (재현 X, *측정*) | §6 STAGE 2 (selector 5-fold) + §7 STAGE 3 (boundary corrector) → §9 STAGE 4 (submission) + §10 STAGE 5 (LB 회수) | frontmatter `lb_score: <float>` + `analysis/plan-004/results.md` |
| 2. 표 분포 확인 | 18 regime 각각의 train sample 수가 empirical Bayes shrinkage 신뢰성 영역 (50+) 안에 있는가? + (regime, candidate) hyper-specialized cell 이 존재하는가? | §8 STAGE 3.5 (regime distribution audit) | `analysis/plan-004/regime_distribution.{json,md}` |

두 갈래는 *독립적 검증* — 갈래 1 (LB) 점수가 낮아도 갈래 2 (표 분포) 박제는 후속 plan 의 regime 재설계 anchor 로 그 자체 가치. 학습 산출 (`oof_selector_scores.npz`) 은 두 갈래 모두에 입력 (갈래 1: corrector → submission, 갈래 2: regime hit-table 의 baseline).

---

## §2. Scope (명시적)

### §2.1 In-scope

| 항목 | 값 |
|---|---|
| 모듈 추출 | notebook cell 4/6 → `src/pb_0_6822/{selector,boundary}.py` (코드 logic 1:1 보존) |
| 데이터 | project `data/` (open.zip 자동 해제) |
| 학습 | full 5-fold selector + boundary corrector full-fit |
| 분석 | 18-regime × 27-candidate 표 + sample 분포 박제 |
| 제출 | autonomous loop 가 `dacon-submit` skill 1회 호출 (사용자 승인 X) |
| 점수 | LB hit-rate@1cm 1개 회수 + plan frontmatter 박제 |

### §2.2 Out-of-scope (절대 안 함)

| 항목 | 이유 |
|---|---|
| 노트북 0.6822 *재현* | 사용자 의도 = 우리 데이터 적용, 재현 X (data/seed 차이) |
| hyperparam 튜닝 | 1차 baseline 박제만 — tuning 은 후속 plan |
| 27 후보 수정 / regime 정의 변경 | extraction 단계 1:1 보존 — 변경은 후속 plan 의 ablation 변수 |
| End-to-end 학습 통합 | notebook 의 2-stage sequential 구조 그대로 보존 |
| GPU 번호 변경 | server `cuda:1` 강제 — `cuda:0` 사용 X (v2 갱신). 다중 GPU 환경에서도 1번 GPU 만 |
| Mac mps 학습 | v1 시도 후 폐기 (열 문제). 본 plan v2 는 server agent 가 학습 수행 — local handoff only |

---

## §3. 사전 등록 (Pre-registration)

### §3.1 Fold split

| 분할 | 값 |
|---|---|
| folds | 5 (notebook default, 본 plan 1차 baseline) |
| fold 할당 | `stable_fold_id(sample_id, 5)` (notebook cell 4 L147) — sample_id hash 기반 결정성 |
| seed | 20260606 (notebook cell 12 default), fold 별 변형 없음 |

### §3.2 합격 기준 (정량)

- **G0**: 모듈 import + `len(CANDIDATES) == 27` + `TinyCorrectionNet` 인스턴스화 + 기존 51 tests green
- **G1**: smoke cv hit-rate finite (no NaN/Inf, 0 ≤ hit ≤ 1)
- **G2**: `oof_selector_scores.npz["ens_scores"].shape == (N_train, 27)` + `test_selector_scores.npz["ens_scores"].shape == (N_test, 27)` + finite
- **G3**: `submission_boundary_tiny_soft.csv` shape == `data/sample_submission.csv` shape + 모든 좌표 finite. (`corrector_no_convergence` severe trigger 가 동시 발생해도 G3 자동 판정 자체는 csv schema 만 — severe 는 별도 alert/escalate, G3 통과 자체를 막지 않음. 단 severe 발생 시 §10 LB 자동 호출 단계로 진입 *전에* 사용자 escalate 필수.)
- **G3.5**: `regime_distribution.json` 에 18 regime histogram + 18×27 hit table + degenerate list 모두 존재
- **G_final**: `lb_score` 회수 (float, isSubmitted=True 응답) + `plans/plan-004-*.results.md` frontmatter 박제

### §3.3 평가 점수

- **CV**: 각 fold 별 selector soft/argmax/gate hit-rate + boundary 후 hit-rate (notebook cell 14 metric summary 형식)
- **LB**: dacon-submit 응답의 `lb_score` (carry-over 시 partial 처리)

---

## §4. STAGE 0 — 모듈 추출 인프라

### §4.0 추출 정책 + public symbol 노출 계약

**노트북 cell 추출 정책** (c2/c3 적용 — 이미 [DONE], v3 spec 명문화):

- 추출 대상: `notes/PB_0.6822 코드공유.ipynb` 의 **cell index 4** (SELECTOR_MAIN, ~94KB) + **cell index 6** (BOUNDARY_MAIN, ~22KB). 0-indexed. cell 0 (Algorithm Notes) + cell 12 (실행 진입) 은 reference 만 — 추출 X.
- 추출 방식: cell 본문 → `.py` 파일에 **1:1 복사**. 함수/클래스 정의·top-level 상수·`if __name__ == "__main__"` 블록 모두 보존.
- IPython magic / display / `%matplotlib` / `get_ipython()` 호출 발견 시: **제거 + 추출 노트 1줄 박제** (commit msg 또는 추출 모듈 top docstring). pseudocode/ASCII-art 주석은 그대로 보존.
- boundary.py 의 **유일한 의도적 수정**: `import train_tcn_gru_candidate_selector as base` → `from src.pb_0_6822 import selector as base` (그 외 어떤 라인도 수정 X; 추출 후 `grep "train_tcn_gru_candidate_selector" src/pb_0_6822/boundary.py` 가 0회 매칭이어야 함).
- 추출 검증: `python -c "from src.pb_0_6822 import selector, boundary"` 가 ImportError 없이 통과해야 G0 진입 자격.

**module top-level export 보장 symbol** (§4.3 smoke test + §8 분석 스크립트 의존성 lock-in):

| symbol | module | type | description |
|---|---|---|---|
| `CANDIDATES` | selector | `list` (len == 27) | 27 물리 공식 후보 (`Candidate` dataclass, `.name` attribute 보유) |
| `SELECTOR_MAIN` | selector | `Callable[[list[str] \| None], Any]` | argparse-based main entry (== `main`) |
| `make_candidates` | selector | `Callable[[np.ndarray, int, int], np.ndarray]` | 후보 좌표 생성. signature `(train_x, end_idx, horizon=2)` → shape `(N, 27, 2)`, dtype float64 |
| `fit_regime_bins` | selector | `Callable[[np.ndarray, int], dict[str, np.ndarray]]` | quantile bin edges. signature `(train_x, end_idx)` → dict (key = 축 이름, value = bin edges array) |
| `assign_regimes` | selector | `Callable[[np.ndarray, int, dict], np.ndarray]` | sample → regime id. signature `(train_x, end_idx, bins)` → shape `(N,)`, dtype int, values in `[0, 17]` |
| `candidate_regime_bias` | selector | `Callable` | empirical Bayes bias (signature 노트북 cell 4 기준 보존, 본 plan 에서는 직접 호출 X — 18×27 표는 §8 에서 별도 reconstruction) |
| `R_HIT` | selector | `float` | hit threshold (notebook 기본값 = `0.01` m; 추출 후 §4.3 smoke 에서 `assert selector.R_HIT == 0.01` 로 lock-in. 다른 값이면 §8.1 hit_table 임계 불일치 → `extraction_drift` severe) |
| `read_labels` | selector | `Callable[[Path], tuple]` | `(ids, train_y)` — `ids: list[str]`, `train_y: np.ndarray shape (N, 2)` |
| `load_stack` | selector | `Callable[[Path, list[str]], np.ndarray]` | shape `(N, T, 2)`, dtype float64 |
| `stable_fold_id` | selector | `Callable[[str, int], int]` | sample_id → fold id (notebook cell 4 기준, hash-based modulo) |
| `TinyCorrectionNet` | boundary | `nn.Module` | `__init__(dim, hidden)` — dim/hidden 의 기본값은 notebook cell 6 default |
| `BOUNDARY_MAIN` | boundary | `Callable` | argparse-based main entry (== `main`) |

위 11개 symbol 중 하나라도 `getattr(module, name)` AttributeError 시: G0 `extraction_drift` severe escalate.

### §4.1 `src/pb_0_6822/selector.py` (c2)

- 노트북 cell 4 (~94KB, ~2120 lines) 전체 추출 (§4.0 정책 적용)
- `if __name__ == "__main__": main()` 보존 (단, 추출 모듈은 `from src.pb_0_6822 import selector` import 경로로도 호출 가능해야 함)
- argparse signature: 노트북 cell 4 의 `parser.add_argument(...)` 호출 *전체 보존* (추가/제거 X). 본 plan 의 `run_full.py` 가 사용하는 인자 집합 (닫힌 enum): `--root, --out-dir, --models, --folds, --fold-limit, --pre-epochs, --fine-epochs, --freeze-fine-epochs, --epoch-plus, --min-epochs, --patience, --hidden, --batch, --skip-full, --device`. 이 집합 외 인자는 모듈 default 사용. 새 인자 추가 금지.
- `SELECTOR_MAIN = main` 노출 보존 (§4.0 export 표 참조)
- IPython-specific 호출 (display, `%matplotlib`, `get_ipython()`) 발견 시 §4.0 정책 적용 (제거 + 박제)

### §4.2 `src/pb_0_6822/boundary.py` (c3)

- 노트북 cell 6 (~22KB, ~517 lines) 전체 추출 (§4.0 정책 적용)
- **유일 수정**: `import train_tcn_gru_candidate_selector as base` → `from src.pb_0_6822 import selector as base` (정확히 이 1 라인만 변경; §4.0 grep 검증 적용)
- argparse signature: 노트북 cell 6 의 `parser.add_argument(...)` 호출 *전체 보존* (추가/제거 X). 본 plan 의 `run_full.py` 가 사용하는 인자 집합 (닫힌 enum): `--root, --out-dir, --folds, --fold, --hidden, --epochs, --fine-epochs, --min-epochs, --patience, --batch, --cap, --apply-scale, --score-bank, --test-score-bank, --make-test, --device`. 이 집합 외 인자는 모듈 default 사용. 새 인자 추가 금지.
- `BOUNDARY_MAIN = main` 노출 보존 (§4.0 export 표 참조)

### §4.3 `tests/test_pb_0_6822_smoke.py` (c4)

```python
def test_selector_import():
    from src.pb_0_6822 import selector
    assert len(selector.CANDIDATES) == 27
    assert all(hasattr(c, "name") for c in selector.CANDIDATES)

def test_boundary_import():
    from src.pb_0_6822 import boundary
    assert boundary.TinyCorrectionNet is not None
    # 인스턴스화 smoke only — dim/hidden 은 임의 placeholder (forward 미실행).
    # 실제 학습 시 dim/hidden 은 notebook cell 6 default 사용 (§4.0 export 계약).
    import torch
    model = boundary.TinyCorrectionNet(dim=20, hidden=64)
    assert model is not None

def test_regime_functions_signature():
    from src.pb_0_6822 import selector
    assert callable(selector.fit_regime_bins)
    assert callable(selector.assign_regimes)
    assert callable(selector.candidate_regime_bias)
    # R_HIT lock-in: §4.0 export 계약의 notebook 기본값 0.01 m 검증.
    assert selector.R_HIT == 0.01, f"R_HIT drift: {selector.R_HIT} != 0.01"

def test_export_contract_completeness():
    """§4.0 export 계약 11 symbol 전수 attribute 존재 + callability 검증."""
    from src.pb_0_6822 import selector, boundary
    for name in ("CANDIDATES", "SELECTOR_MAIN", "make_candidates",
                 "fit_regime_bins", "assign_regimes", "candidate_regime_bias",
                 "R_HIT", "read_labels", "load_stack", "stable_fold_id"):
        assert hasattr(selector, name), f"selector missing export: {name}"
    for name in ("TinyCorrectionNet", "BOUNDARY_MAIN"):
        assert hasattr(boundary, name), f"boundary missing export: {name}"
    assert callable(selector.make_candidates)
    assert callable(selector.SELECTOR_MAIN)
    assert callable(selector.read_labels)
    assert callable(selector.load_stack)
    assert callable(selector.stable_fold_id)
    assert callable(boundary.BOUNDARY_MAIN)
```

### §4.4 `src/pb_0_6822/run_full.py` orchestrator (c5)

```python
# pseudo — src/pb_0_6822/run_full.py
import argparse, zipfile
from pathlib import Path
from src.pb_0_6822 import selector, boundary

REPO = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO / "data"
OUT_DIR   = REPO / "runs/baseline/P001_pb-0-6822-fullrun"
DEVICE    = "cuda:1"  # server agent 1번 GPU 강제 (§2.2, decision-note 박제)

def ensure_data_extracted():
    """data/open.zip → data/{train,test,train_labels.csv,sample_submission.csv} 자동 해제.
       train_labels.csv 부재 + open.zip 부재 시 즉시 FileNotFoundError raise (fail-fast;
       사용자 escalate — DATA_ROOT 가 잘못되었거나 zip 미배포 가능성)."""
    if not (DATA_ROOT / "train_labels.csv").exists():
        zip_path = DATA_ROOT / "open.zip"
        if not zip_path.exists():
            raise FileNotFoundError(
                f"data 미배포: {zip_path} 와 train_labels.csv 둘 다 부재. "
                f"DATA_ROOT={DATA_ROOT} 확인 후 zip 배치 또는 직접 해제 필요."
            )
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(DATA_ROOT)

def run_smoke():
    """STAGE 1 (G1): 1-fold smoke (selector phase 1 epoch × 1 fold). boundary 미실행."""
    smoke_dir = OUT_DIR / "smoke"
    smoke_dir.mkdir(parents=True, exist_ok=True)
    selector.SELECTOR_MAIN([
        '--root', str(DATA_ROOT), '--out-dir', str(smoke_dir),
        '--models', 'attn_gru',
        '--folds', '5', '--fold-limit', '1',
        '--pre-epochs', '1', '--fine-epochs', '1', '--freeze-fine-epochs', '1',
        '--epoch-plus', '0', '--min-epochs', '1', '--patience', '1',
        '--hidden', '48', '--batch', '4096',
        '--skip-full',  # smoke 는 full-fit X
        '--device', DEVICE,
    ])
    # smoke_dir/summary.json 검증은 외부 G1 gate (§5)

def run_selector():
    """STAGE 2 (G2): full 5-fold selector + full-fit test 추론."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    selector.SELECTOR_MAIN([
        '--root', str(DATA_ROOT), '--out-dir', str(OUT_DIR),
        '--models', 'attn_gru',
        '--folds', '5', '--fold-limit', '5',
        '--pre-epochs', '10', '--fine-epochs', '8', '--freeze-fine-epochs', '3',
        '--epoch-plus', '5', '--min-epochs', '5', '--patience', '4',
        '--hidden', '48', '--batch', '4096',
        '--device', DEVICE,
        # no --skip-full → test_selector_scores.npz 자동 생성
    ])

def run_boundary():
    """STAGE 3 (G3): boundary corrector full-fit. selector npz 입력 필수."""
    score_bank      = OUT_DIR / 'oof_selector_scores.npz'
    test_score_bank = OUT_DIR / 'test_selector_scores.npz'
    assert score_bank.exists(),      f"selector OOF npz missing: {score_bank}"
    assert test_score_bank.exists(), f"selector test npz missing: {test_score_bank}"
    boundary.BOUNDARY_MAIN([
        '--root', str(DATA_ROOT), '--out-dir', str(OUT_DIR),
        '--folds', '5', '--fold', '0',
        '--score-bank', str(score_bank),
        '--test-score-bank', str(test_score_bank),
        '--epochs', '12', '--fine-epochs', '8', '--min-epochs', '5', '--patience', '4',
        '--hidden', '64', '--batch', '8192',
        '--cap', '0.006', '--apply-scale', '1.0',
        '--make-test',
        '--device', DEVICE,
    ])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--phase', choices=['smoke', 'selector', 'boundary', 'full'], default='full',
        help='smoke=STAGE 1 / selector=STAGE 2 / boundary=STAGE 3 (npz 선행) / full=selector→boundary',
    )
    args = parser.parse_args()
    ensure_data_extracted()
    if args.phase == 'smoke':    run_smoke()
    elif args.phase == 'selector': run_selector()
    elif args.phase == 'boundary': run_boundary()
    elif args.phase == 'full':
        run_selector()
        run_boundary()

if __name__ == "__main__":
    main()
```

---

## §5. STAGE 1 — 1-fold Smoke (c6)

- **실행**: `python -m src.pb_0_6822.run_full --phase smoke`
- **산출** (`runs/baseline/P001_pb-0-6822-fullrun/smoke/` — full 산출과 격리):
  - `summary.json` (notebook cell 14 형식 기반, 최소 키 spec — 추출 모듈이 추가 키를 더 박제해도 무관):
    ```json
    {
      "phase": "smoke",
      "fold_limit": 1,
      "epoch_actual": 1,
      "cv_hit_rate_soft":    <float>,
      "cv_hit_rate_argmax":  <float>,
      "cv_top1_accuracy":    <float>,
      "wall_time_sec":       <float>
    }
    ```
  - `oof_selector_scores.npz` (1 fold val partition only, shape `(N_fold_val, 27)`)
- **G1 합격 기준** (자동 판정):
  - `summary.json` 존재 + parse 가능
  - `cv_hit_rate_soft` finite (= `np.isfinite(x) and 0.0 <= x <= 1.0`)
  - `cv_top1_accuracy >= 1/27 ≈ 0.0370` (random baseline; 미달 시 `extraction_drift` severe)
  - stdout/stderr substring 검사: smoke 호출은 `python -m src.pb_0_6822.run_full --phase smoke 2>&1 | tee runs/baseline/P001_pb-0-6822-fullrun/smoke/run.log` 패턴으로 실행하고, 자동 판정 단계에서 `grep -E "Traceback|NaN|Inf" run.log` 가 0건이어야 G1 통과. (NaN/Inf 라는 단어가 정상 log 에 포함될 수 있다면 정규식을 `-E "Traceback|\\bNaN\\b|\\bInf\\b"` 로 좁힘.)
- **시간 예산**: ~5min (1 epoch × 1 fold, server `cuda:1`)

---

## §6. STAGE 2 — Full Selector 5-fold (c7)

- **실행**: `python -m src.pb_0_6822.run_full --phase selector`
- **산출** (`runs/baseline/P001_pb-0-6822-fullrun/`):
  - `oof_selector_scores.npz` — key `ens_scores`, shape `(N_train, 27)`, dtype float32 또는 float64. `N_train = len(read_labels(data/train_labels.csv)[0])`
  - `test_selector_scores.npz` — key `ens_scores`, shape `(N_test, 27)`. `N_test = len(pd.read_csv(data/sample_submission.csv))`
- **G2 합격 기준** (자동 판정):
  - 두 npz 모두 존재 + `np.load(...)["ens_scores"]` 로드 가능
  - 두 array 모두 `np.isfinite(x).all() == True`
  - shape 0번째 축이 위 N_train / N_test 와 일치, 1번째 축 == 27
- **시간 예산**: ~10~20min (server `cuda:1`, GPU 가속)

---

## §7. STAGE 3 — Full Boundary Corrector (c8)

- **실행**: `python -m src.pb_0_6822.run_full --phase boundary` (STAGE 2 의 npz 두 개가 `OUT_DIR/` 에 선행 존재 필수 — 없으면 `AssertionError`)
- **`--fold 0` semantic** (§4.4 L334 인자): notebook cell 6 의 BOUNDARY_MAIN 은 단일 인자 `--fold` 를 받지만 모듈 default 동작이 *5-fold 전체 학습 + `--fold` 값을 primary fold pointer (submission 생성 시 reference fold) 로 사용*. 따라서 `--fold 0` 은 "fold 0 만 학습" 이 아니라 "5-fold 전체 학습, primary = fold 0". G3 csv schema 가 5-fold 결과의 weighted blend 임을 보장하는 것이 G3 자동 판정의 책임 (corrector 내부 학습 동작은 reference-aligned mode 로 노트북 default 위임).
- **산출** (`runs/baseline/P001_pb-0-6822-fullrun/`):
  - `submission_boundary_tiny_soft.csv` (continuous probability-weighted blend; **본 plan 의 LB 제출 source**)
  - `submission_boundary_tiny_argmax.csv` (hard argmax; informational only, LB 제출 X)
  - `boundary_tiny_correction_report.json` schema (minimum keys; corrector convergence 판정 enabler):
    ```json
    {
      "per_fold": [{"fold": 0, "oof_hit_soft": <float>, "oof_hit_argmax": <float>, "delta_mean": <float>, "delta_std": <float>}, ...],
      "selector_oof_hit_soft_baseline": <float>,
      "selector_oof_hit_argmax_baseline": <float>,
      "overall_oof_hit_soft": <float>,
      "overall_oof_hit_argmax": <float>
    }
    ```
    semantic: `per_fold[k].oof_hit_soft` 는 fold k 의 *val partition* 한정 hit@1cm (fold-local). `overall_oof_hit_soft` 는 5 fold val partition 의 *concatenated* hit@1cm (전체 sample 1회 평균; 평균-of-평균 X). `corrector_no_convergence` severe 판정은 `overall_oof_hit_soft < selector_oof_hit_soft_baseline` 인지로 비교 (per-fold 가 아닌 overall). **`selector_oof_hit_soft_baseline` 산출 책임**: boundary 모듈이 `--score-bank` 로 받은 `oof_selector_scores.npz` 의 `ens_scores` 를 사용해 corrector 학습 *전* (zero-init delta 적용 전) 시점에 5-fold concatenated hit@1cm 을 *재산출* 하여 report 에 박제. 즉 selector 모듈은 baseline metric 박제 책임 X (npz raw score 만 제공), 산출은 boundary 모듈 안에서 closure. 추출 모듈이 추가 키를 더 박제해도 무관 (extraction_drift 룰과 동일).
- **G3 합격 기준** (자동 판정):
  - 3개 산출 파일 모두 존재
  - 두 csv 의 shape == `pd.read_csv(data/sample_submission.csv).shape`
  - 두 csv 의 `columns` == sample_submission columns (순서 포함)
  - 두 csv 의 첫 column (id) == sample_submission 첫 column (값 + 순서)
  - 두 csv 의 좌표 column 들 모두 `pd.to_numeric(errors='coerce').notna().all()` == True
- **시간 예산**: ~5~10min (server `cuda:1`)

---

## §8. STAGE 3.5 — 18×27 Regime Distribution Analysis (c9)

### §8.1 분석 스크립트

```python
# analysis/plan-004/regime_distribution.py
import json
import numpy as np
from pathlib import Path
from src.pb_0_6822 import selector

REPO = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO / "data"
OUT_DIR = REPO / "runs/baseline/P001_pb-0-6822-fullrun"
ANALYSIS_DIR = REPO / "analysis/plan-004"

def main():
    ids, train_y = selector.read_labels(DATA_ROOT / "train_labels.csv")
    train_x = selector.load_stack(DATA_ROOT / "train", ids)
    end_idx = train_x.shape[1] - 1

    bins = selector.fit_regime_bins(train_x, end_idx)
    regimes = selector.assign_regimes(train_x, end_idx, bins)
    hist = np.bincount(regimes, minlength=18)

    cands = selector.make_candidates(train_x, end_idx, horizon=2)
    err = np.linalg.norm(cands - train_y[:, None, :], axis=2)
    R_HIT = selector.R_HIT  # 0.01

    # 18×27 hit-rate table
    hit_table = np.zeros((18, 27), dtype=np.float64)
    mean_dist_table = np.zeros((18, 27), dtype=np.float64)
    for r in range(18):
        mask = regimes == r
        if mask.sum() > 0:
            hit_table[r] = (err[mask] <= R_HIT).mean(axis=0)
            mean_dist_table[r] = err[mask].mean(axis=0)

    # Degenerate flags
    degenerate_regimes = [int(r) for r in range(18) if hist[r] < 50]
    # Hyper-specialized cells: (regime, candidate) hit-rate ≥ regime_mean × 1.5 또는 ≤ × 0.5
    regime_means = hit_table.mean(axis=1)
    hyper_cells = []
    for r in range(18):
        for c in range(27):
            if regime_means[r] > 0:
                ratio = hit_table[r, c] / regime_means[r]
                if ratio >= 1.5 or (ratio <= 0.5 and hit_table[r, c] >= 0.01):
                    hyper_cells.append({
                        "regime": int(r),
                        "candidate": int(c),
                        "candidate_name": selector.CANDIDATES[c].name,
                        "hit_rate": float(hit_table[r, c]),
                        "regime_mean": float(regime_means[r]),
                        "ratio": float(ratio),
                    })

    summary = {
        "n_total": int(len(train_y)),
        "regime_histogram": [int(h) for h in hist],
        "regime_bin_edges": {k: list(v) for k, v in bins.items()},
        "candidate_names": [c.name for c in selector.CANDIDATES],
        "hit_table": hit_table.tolist(),
        "mean_dist_table": mean_dist_table.tolist(),
        "regime_means": regime_means.tolist(),
        "degenerate_regimes": degenerate_regimes,
        "degenerate_count": len(degenerate_regimes),
        "hyper_specialized_cells": hyper_cells,
        "hyper_specialized_count": len(hyper_cells),
    }

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    (ANALYSIS_DIR / "regime_distribution.json").write_text(json.dumps(summary, indent=2))
    write_markdown(summary)

def write_markdown(summary):
    """regime_distribution.md 작성. 최소 schema (이 순서 보존, 추가 섹션은 구현자 자유):
       § Header: `# Regime × Candidate Distribution (plan-004)` + `n_total`, `R_HIT`, `degenerate_count`, `hyper_specialized_count` summary 한 줄.
       § Regime Histogram: 18-row markdown table `| regime | count | bin_summary |`.
       § 18×27 Hit-Rate Table: header row = 27 candidate name (truncate 가능), data row 18개 (값 = `f"{x:.3f}"`).
       § Degenerate Regimes: `degenerate_regimes` list 박제 (regime id + count).
       § Hyper-Specialized Cells: `hyper_cells` list 박제 (regime, candidate_name, hit_rate, ratio)."""
    ...

if __name__ == "__main__":
    main()
```

### §8.2 산출

- `analysis/plan-004/regime_distribution.json` — 전체 분석 결과
- `analysis/plan-004/regime_distribution.md` — 사람 읽는 markdown 표 + 해석

### §8.3 검증

- regime histogram 의 18 entry 모두 존재
- 18×27 table 의 모든 cell 이 finite (`degenerate_regimes` 의 cell 은 0 가능)
- degenerate count + hyper-specialized count 모두 보고

---

## §9. STAGE 4 — Submission 준비 (c10)

- **사본 생성**: `shutil.copy2(OUT_DIR / "submission_boundary_tiny_soft.csv", OUT_DIR / "submission.csv")` (mtime 보존; byte-identity 보장)
- **검증 코드** (실패 시 `AssertionError` raise → `submission_shape_mismatch` severe):
  ```python
  import pandas as pd
  sub = pd.read_csv(OUT_DIR / "submission.csv")
  ref = pd.read_csv(DATA_ROOT / "sample_submission.csv")
  assert sub.shape == ref.shape, f"shape: sub={sub.shape} ref={ref.shape}"
  assert list(sub.columns) == list(ref.columns), "columns mismatch"
  assert (sub.iloc[:, 0].astype(str).values == ref.iloc[:, 0].astype(str).values).all(), "id order mismatch"
  coord_cols = sub.columns[1:]
  assert sub[coord_cols].apply(pd.to_numeric, errors='coerce').notna().all().all(), "non-finite coords"
  ```
- 본 검증을 통과해야 c11 (§10 LB 제출) 진입 자격.

---

## §10. STAGE 5 — LB 제출 + 박제 (c11)

- **자율 호출** (CLAUDE.md autonomous policy, 사용자 승인 X):
  `Skill(skill="dacon-submit", args="runs/baseline/P001_pb-0-6822-fullrun/submission.csv P001_pb-0-6822-fullrun")`
- **응답 dict**: `{isSubmitted: bool, lb_score: float | None, detail: str}`
- **응답 4분기 처리 정책**:

  | (isSubmitted, lb_score) | 처리 | frontmatter `lb_score` | results `status` | severe |
  |---|---|---|---|---|
  | (True, float) | full success — lb_log 박제 + frontmatter + results.md 모두 갱신 | `<float>` (소수 4자리) | `all_complete` | — |
  | (True, None) | partial — 점수 비동기 대기. lb_log + results 박제, 점수 도착 시 follow-up commit (c11.1) 로 갱신 | `TBD` | `partial` | — |
  | (False, *) | submission failed — `detail` 박제 후 **1회 retry** (60초 sleep 후 재호출; retry trigger = `detail.lower()` 안의 substring 매칭 `rate_limit` / `timeout` / `network` / `5xx` / `eof` 등 *일시적* 실패 시그널 1개 이상 (case-insensitive: 응답 detail 을 lowercase 정규화 후 비교). `auth` / `invalid_file` / `quota_exceeded` 등 *영구* 실패 시그널이면 retry 생략하고 즉시 escalate). 재실패 시 `lb_unsubmitted` severe escalate. | (미박제) | (미작성) | `lb_unsubmitted` |
  | Skill 미존재 / 호출 exception | 즉시 `dacon_submit_skill_missing` severe escalate | (미박제) | (미작성) | `dacon_submit_skill_missing` |

- **`analysis/plan-004/lb_log.md` 포맷** (markdown table, append-only; 첫 제출 시 헤더 + 1행, 후속 제출 시 데이터행 1줄 추가):

  ```markdown
  | timestamp_kst             | exp_id                    | isSubmitted | lb_score | detail |
  |---------------------------|---------------------------|-------------|----------|--------|
  | 2026-05-11T18:23:45+09:00 | P001_pb-0-6822-fullrun    | true        | 0.6234   | OK     |
  ```

  필드 spec:
  - `timestamp_kst`: ISO8601 with `+09:00` offset (Asia/Seoul)
  - `exp_id`: `P001_pb-0-6822-fullrun`
  - `isSubmitted`: `true` / `false` (lowercase)
  - `lb_score`: 소수 4자리 float (예: `0.6234`) 또는 `TBD` (partial) 또는 `null` (false)
  - `detail`: dacon-submit 응답 `detail` 필드, 80자 이상 시 truncate + `...` 추가

- **`lb_score` frontmatter 갱신** (3개 파일 동시):
  1. `plans/plan-004-pb-0-6822-fullrun.md` top-level frontmatter `lb_score: <float|TBD|null>` (현재 `null` → 갱신)
  2. `plans/plan-004-pb-0-6822-fullrun.results.md` frontmatter `lb_score: <float|TBD|null>`
  3. `analysis/plan-004/results.md` frontmatter `lb_score: <float|TBD|null>`

  `plans/plan-004-pb-0-6822-fullrun.md` 본문 수정은 §12.6 blacklist 의 §0.5 [TODO]→[DONE] 갱신과 동일 예외.

- **`analysis/plan-004/results.md` 작성**: `plan-003/results.md` 형식 참조, §N+2 필수 항목 포함.
- **`plans/plan-004-pb-0-6822-fullrun.results.md` 작성**: frontmatter (`lb_score`, `status`, `exp_id`) + §N+2 필수 항목.

---

## §11. 작업량 총 회계

- 코드 추출: cell 4 (~94KB) + cell 6 (~22KB) = ~117KB 1회성 translation
- 학습: selector ~10~20min + corrector ~5~10min ≈ ~15~30min (server `cuda:1`, GPU 가속)
- 분석: ~2min
- 제출: 1 API call
- **총 wall-time 예산: ~30~40min (server `cuda:1`)**

---

## §N+2. results.md 필수 항목

(plan-003 format 참조)

- exp_id, plan_id, lb_exp_id, lb_score, lb_submitted_at
- per-fold CV metrics + soft/argmax/gate hit-rate
- 18×27 표 요약 (degenerate count, hyper-specialized count)
- 다음 plan 후보 (post-G_final 분석 기반)
- decision-note 박제 list

---

## §N+3. 통계 함정 & caveats

1. **노트북 0.6822 ≠ 우리 LB**: notebook 작성자의 환경/seed/데이터 분할이 다를 수 있어 우리 점수가 0.6822 보다 낮을 수 있음. *어떤 점수든 회수하면 G_final 충족* (재현이 목표 아님).
2. **Server `cuda:1` 강제** (v2): 다중 GPU 환경에서도 1번 GPU 만 사용해 결과 reproducibility 보장. plan-003 의 cuda:0 강제 패턴 답습하되 GPU 번호만 1번으로. (v1 의 Mac mps 학습 폐기 — 열 문제 + 학습 fold 4 중단 사례 후 결정)
3. **18 regime 의 도메인 적합성**: 노트북은 모기 비행 가정으로 (speed × curvature × speed_slope) 축 선택. 우리 데이터 (동일 dacon) 가 같은 분포라면 호환, 다르면 degenerate regime 증가. **G3.5 결과가 후속 plan 의 regime 재설계 anchor**.
4. **2-stage sequential vs end-to-end**: 본 plan 은 노트북 그대로 sequential. end-to-end 통합은 ablation 가치 있으나 후속 plan.
5. **dacon-submit 의 lb_score 비동기**: plan-002/003 패턴 — `lb_score: TBD` + `status: partial` 마감 후 점수 도착 시 follow-up commit 으로 `all_complete` 갱신.

---

## §N+4. 변경 이력

- v1 (2026-05-11): 초안 — plan-004 신규 작성, c1~c11 commit chain 박제, G0~G_final 7개 gate 정의. compute=Mac mps 가정.
- v2 (2026-05-11): **compute 변경: Mac mps 학습 폐기 → server `cuda:1` 강제**. v1 c6 1-fold smoke 통과 후 c7 full 5-fold 학습 중 Mac 열 문제로 fold 4 중단. server agent 가 학습 인계받음 (local push → server pull → 학습 → results push). §0 한 줄, §2.2 out-of-scope, §0.5 decision-note, §6/§7 시간 예산, §11 작업량 회계, §N+3 caveats 갱신. `run_full.py --device cuda:1` 강제, config yaml `device: cuda:1` 명시. 모든 spec 외 sequence/G-gate/severe/commit chain 은 v1 유지.
- v3 (2026-05-11): **plan-review 결과 + 사용자 narrative 반영 spec 강화** (BLOCKER 12 / AMBIGUITY 23 해소).
  - **narrative 두 갈래 명문화** (§0 한 줄 목적 + §1.2 검증 포인트 매핑 표): (1) 이 방법의 LB 확인, (2) regime × state 표 sample 분포 확인.
  - §4.0 신규: 노트북 cell 추출 정책 + 11개 public symbol 노출 계약 표 (`CANDIDATES`, `make_candidates`, `fit_regime_bins`, `assign_regimes`, `R_HIT`, `read_labels`, `load_stack`, `stable_fold_id`, `candidate_regime_bias`, `TinyCorrectionNet`, `SELECTOR_MAIN`/`BOUNDARY_MAIN`).
  - §4.1/§4.2 argparse signature trailing `...` 제거 → 닫힌 enum 으로 lock-in (`run_full.py` 사용 인자 집합 명시).
  - §4.4 `run_full.py`: `--smoke` 단일 플래그 → `--phase {smoke,selector,boundary,full}` 분리 CLI. 각 phase 의 진입 조건 (selector npz 선행) `assert`.
  - §5/§6/§7 phase 단독 실행 명령 + 자동 판정 G-gate 임계 박제 (G1: `cv_top1_accuracy >= 1/27`, G2: `np.isfinite + shape`, G3: csv schema 4축 비교).
  - §9 submission 사본 (`shutil.copy2`) + 4-line `assert` 검증 코드 직박.
  - §10 dacon-submit 응답 4분기 처리 표 + `lb_log.md` markdown table 포맷 (ISO8601 KST, 소수 4자리, detail 80자 truncate) + `lb_score` frontmatter 3-file 갱신 명시.
  - **commit chain 영향**: c2/c3/c5 는 [DONE] 상태로 본문 보존; v3 spec 강화는 c4 (smoke test) 부터 적용. 기존 c2/c3 추출 결과가 §4.0 export 계약과 불일치 시 c2.1/c3.1 refactor commit 추가 (현재까지 audit 미실시).
  - 모든 spec 외 sequence (STAGE 0~5) / G-gate (G0~G_final) / severe trigger / commit chain ID (c1~c11) / 두 갈래 narrative 의도 는 v1/v2 유지.

---

## §N+5. 참조

- `notes/PB_0.6822 코드공유.ipynb` (소스, read-only)
- `WORKFLOW.md` §0.5, §11, §12 convention
- `plans/plan-003-residual-gru-grid.md` (format reference, c12/c13/c14 LB 자율 제출 패턴)
- `CLAUDE.md` (autonomous execution policy)
- `registry.csv` (exp_id append target)
- `src/submit.py` (dacon-submit infra)
