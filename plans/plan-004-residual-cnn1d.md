---
plan_id: 004
version: 2
date: 2026-05-11 (Asia/Seoul)
status: draft
based_on:
  - 003
  - R001_baseline-residual-gru
  - B001_linear-2pt
scope: single-variable architecture swap — GRU 메인 모델만 1D CNN 으로 교체 (다른 모든 hyperparameter / baseline 외삽 / feature / loss / training schedule 은 R001 비트 동일). 1 ablation exp + 자동 LB 제출 1점.
exp_ids:
  - C001_baseline-residual-cnn1d
---

# plan-004 v1 — Residual 1D CNN Architecture Swap (Single Variable vs R001)

## §0. 한 줄 목적

> **R001_baseline-residual-gru (plan-003, CV 0.013383, LB 0.5688) 의 *메인 모델 부분만* `ResidualGRU(hidden=64, layers=2, dropout=0.1)` 에서 `ResidualCNN1D(channels=64, layers=2, kernel=3, dropout=0.1, padding=same)` 로 교체한 C001 1개 exp 로 *architecture 축의 paired Δ vs R001* 박제. R001 의 다른 모든 변수 (baseline=linear-2pt, feature=relative, loss=huber, lr=1e-3, weight_decay=1e-4, batch=64, epochs=100, early-stop patience=10, seed=42, fold ensemble mean) 는 **비트 동일**. CV 측정 → submission.csv 생성 → `dacon-submit` skill 1회 자율 제출 (사용자 승인 X) → LB 점수 회수 후 R001 LB 0.5688 vs C001 LB 의 paired delta 박제. 본 plan 의 의무 산출은 (a) C001 cv_mean_eucl ± std, (b) R001 vs C001 paired Δ 표 (5-fold), (c) C001 LB 1점, (d) architecture 축 정보 — caveat 박제 (closed-form B001 LB 0.60 floor 미달 prior 의 architecture-invariance 여부).**

---

## §0.5 Quick Reference (autonomous loop 매 turn 읽는 section)

### 합격 기준 (G-gate sequence)

- C001 의 **5-fold CV mean_eucl + per-axis MAE + hit_rate@{0.05,0.10,0.20,0.50} m** 가 registry + run dir 에 기록.
- C001 학습 NaN/Inf 0건 + training divergence 0건. 위반 시 `nn_numerical` severe.
- C001 cv_mean_eucl 가 B001 floor 0.012941 와 paired Δ ≤ +0.005 — 부호 규약: **`Δ_fold[i] = C001.fold_metrics[i].mean_eucl - B001.fold_metrics[i].mean_eucl` (음수 = C001 우월)**. 5fold mean Δ 이 임계값. 위반 시 `residual_no_convergence` severe.
- C001 5-fold mean(cv_mean_eucl) < 0.030 (sanity; mean 기준, max-fold 아님). 위반 시 `nn_no_signal` severe.
- C001 `submission.csv` 생성 (스키마 100% 일치) + `dacon-submit` skill 1회 자율 호출 + API 응답 `isSubmitted=True` 회수 → `analysis/plan-004/lb_log.md` 1행 기록. **submission 자체는 G_final 통과 의무**; LB 점수 회수는 plan-003 §6 carry-over 패턴 (dacon.io 수동 회수) 동일 — partial 종료 허용 (`status: partial`, `lb_score: null`). 자율 호출 자체 실패 (`isSubmitted=False`) 시 `lb_unsubmitted` severe.
- **단일 변수 원칙 유지 검증**: R001 config 와 C001 config 의 *의미적* diff 는 (a) `model.type` 1줄 신규 추가 (R001 부재 → cnn1d-residual), (b) `model.kernel` 1줄 신규 추가, (c) `model.hidden` → `model.channels` rename 1줄 (= caveat #3 의 의미적 동치 키, factory 분기에서 1줄 카운트) 만 허용. 그 외 키 차이 시 `single_variable_violation` severe. *config-level* 1줄 동치 카운트 = 신규 추가 2 + rename 1 = **diff 3줄 fixed**.

### G-gates

- G0: STAGE 0 인프라 commit chain 완료 (1D CNN model module + run.py method dispatch 확장 + train_residual.py 의 model factory 확장 + tests green + backward-compat smoke) [TODO]
- G1: STAGE 1 C001 학습 + cv 평가 + R001 paired Δ 표 산출 [TODO]
- G_final: C001 submission.csv 생성 + dacon-submit skill 자율 호출 + LB 점수 회수 + results.md 작성 [TODO]

### Commit chain (next-up)

| # | type | spec section | status |
|---|---|---|---|
| c1 | code | `src/models/residual_cnn1d.py` 신규 — `ResidualCNN1D(input_dim, channels=64, layers=2, kernel=3, dropout=0.1, padding="same")`. spec @ §4.1 | [TODO] |
| c2 | code | `src/training/train_residual.py` 확장 — `_build_model(cfg)` factory 분기 (gru-residual ↔ cnn1d-residual). 후방호환 보존 (기존 R001~R006 재실행 시 cv 4자리 일치). spec @ §4.2 | [TODO] |
| c3 | code | `src/run.py` method dispatch 확장 — `method="cnn1d-residual"` 분기 추가 (또는 `cfg["model"]["type"]` 키로 분기). 기존 gru-residual 분기 비트 동일 보존. spec @ §4.3 | [TODO] |
| c4 | test | `tests/test_residual_cnn1d.py` 신규 (forward shape + parameter count + 1-batch overfit + 후방호환 smoke = R001 재실행 cv 절대오차 `< 1e-5`). spec @ §4.4 | [TODO] |
| G0 | gate | `pytest -q tests/` exit 0; R001 backward-compat smoke (`|cv_mean_eucl_new - 0.013383| < 1e-5`); torch import + device probe 성공 (cuda 부재 시 §0.5 의 `cuda_unavailable` decision-note 경로) | [TODO] |
| c5 | exp C001 | `configs/baseline/C001_baseline-residual-cnn1d.yaml` + 5-fold 학습 + ckpt/fold{0..4}.pt + registry append. spec @ §5 | [TODO] |
| G1 | gate | C001 summary.json + 5 fold ckpt + cv_mean_eucl finite + B001 paired Δ ≤ +0.005 + R001 paired Δ 표 산출 가능 | [TODO] |
| c6 | sub-gen | `src/submit.py` 확장 (cnn1d-residual method 분기; gru-residual 분기 비트 동일 보존) → C001 `submission.csv` 생성 + 스키마 검증. spec @ §5.3 | [TODO] |
| c7 | sub-lb | **`dacon-submit` skill 자율 호출 (사용자 승인 X)** — C001 submission.csv 1회 제출 → LB 점수 회수 → `analysis/plan-004/lb_log.md` 1행 기록 + registry notes 갱신. skill 부재 시 `dacon_submit_skill_missing` severe. spec @ §5.4 | [TODO] |
| c8 | docs | `analysis/plan-004/results.md` + `plans/plan-004-residual-cnn1d.results.md` (frontmatter `lb_exp_id`, `lb_score`, `paired_delta_vs_r001`). spec @ §7 | [TODO] |
| G_final | gate | 위 모두 완료 + §0.5 [TODO]→[DONE] sync + lb_score 박제 | [TODO] |

### Plan-specific severe (WORKFLOW.md §12.3 default 위 추가분)

- `nn_numerical`: 학습 중 loss NaN/Inf 또는 gradient NaN 발생 → torch dtype/lr/gradient clip 점검. 자동 복구 X.
- `residual_no_convergence`: C001 의 5fold paired mean Δ vs B001 (부호 규약: C001-B001) 가 +0.005 초과 → 학습 루프 / target 정의 / inverse transform 버그 의심.
- `nn_no_signal`: C001 5-fold mean(cv_mean_eucl) ≥ 0.030 → architecture/data pipeline/normalization 근본 버그 의심.
- `cuda_oom`: GPU OOM 발생 시 batch_size 64→32→16 자동 감소 후 재시도 (3 단계, lr 등 다른 hyperparameter 불변; gradient accumulation 미적용 — 단일 변수 위반 회피 위해). 모두 fail 시 severe.
- `cuda_unavailable`: torch.cuda.is_available()=False → severe X, decision-note `device-fallback` 박제 + device=cpu + epochs=50 (R001 의 plan-003 §0.5 동일 fallback 규약). 단 backward_compat_drift 임계값 (`1e-5`) 위반 가능성 — caveat #9 박제.
- `lb_unsubmitted`: c7 의 `dacon-submit` skill 자율 호출 API 응답이 `isSubmitted=False` 또는 호출 자체 fail → submission 자체가 외부 시스템에 도달하지 못함. LB 점수 회수 미완료 (carry-over pending) 는 severe X — partial 종료 가능.
- `submission_schema_fail`: C001 submission.csv 가 sample_submission 스키마 검증 fail (10000 rows, ID 열 일치, column order 일치).
- `dacon_submit_skill_missing`: c7 진입 시 `dacon-submit` skill 부재 → 사용자 escalate.
- `backward_compat_drift`: G0 의 R001 재실행 cv_mean_eucl 가 registry 기존 값 0.013383 과 절대오차 `|Δ| ≥ 1e-5` → model factory 도입이 GRU 분기에 영향. (= L48 G0 gate 와 동일 정량 임계값, 통일).
- `single_variable_violation`: C001 config 와 R001 config 의 의미적 diff 가 §0.5 합격 기준의 "diff 3줄 fixed" 규약 위반 (신규 키 ≠ {model.type, model.kernel}, rename ≠ {hidden→channels}). config diff tool: `yq` 또는 `python -c "import yaml; diff(...)"`, 의미적 동치 키는 `_canonical_key_map = {"hidden": "channels"}` 기반 normalize 후 비교.
- `cnn1d_import_error`: `from src.models.residual_cnn1d import ResidualCNN1D` 실패 (예: PyTorch < 1.10 의 `padding="same"` 미지원, syntax error 등) → 즉시 severe. caveat #5 분리 (NaN/Inf 와 다른 카테고리).

### Plan-specific paths (WORKFLOW.md §12.5/§12.6 추가/제외)

- whitelist 추가:
  - `src/models/residual_cnn1d.py` (신규)
  - `src/models/__init__.py` (re-export 1줄 추가)
  - `tests/test_residual_cnn1d.py` (신규)
  - `configs/baseline/C001_baseline-residual-cnn1d.yaml` (신규)
  - `runs/baseline/C001_baseline-residual-cnn1d/**` (ckpt 는 `.gitignore` 으로 제외)
  - `analysis/plan-004/**` (`analysis/plan-004/lb_log.md`, `analysis/plan-004/results.md`)
- whitelist 확장: `src/run.py` (method dispatch 만), `src/submit.py` (cnn1d-residual method dispatch 만), `src/training/train_residual.py` (model factory 분기만)
- blacklist 추가:
  - `runs/baseline/B00*/**`, `configs/baseline/B00*.yaml` (plan-001 산출)
  - `runs/baseline/S00*/**`, `configs/baseline/S00*.yaml` (plan-002 산출)
  - `runs/baseline/R00*/**`, `configs/baseline/R00*.yaml` (plan-003 산출 — 절대 수정 금지)
  - `src/models/residual_gru.py` (plan-003 산출 — model factory 확장 시에도 GRU 모듈 자체 수정 금지)
  - `analysis/plan-001/**`, `analysis/plan-002/**`, `analysis/plan-003/**`

### Decision-note 사용 예 (자율 결정 시 commit msg 박제)

- `decision-note: spec-default — ResidualCNN1D 구조 (layers=2 fixed): Conv1d(input_dim → channels, kernel=3, padding="same") + ReLU + Dropout(0.1) → Conv1d(channels → channels, kernel=3, padding="same") + ReLU + Dropout(0.1) → 마지막 timestep feature [:, :, -1] → Linear(channels → 3). 모든 Conv1d 가 동일 패턴 반복 (residual shortcut 없음, BatchNorm 없음, projection 없음 — "Residual" 명칭은 *학습 paradigm* (= baseline 외삽 + neural 잔차) 의미이지 ResNet skip-connection 아님). layers=2 (R001 GRU layers=2 와 depth-match), channels=64 (R001 hidden=64 와 width-match)`
- `decision-note: spec-default — Conv1d 입력 shape 변환: (B, T, C) → permute → (B, C, T) → conv → permute → (B, T, C). 마지막 timestep 추출 = out[:, -1, :]`
- `decision-note: spec-default — padding="same" (PyTorch ≥ 1.10 지원). kernel=3 + padding=same → output sequence length 보존 (T=11 in, T=11 out)`
- `decision-note: spec-default — weight init = PyTorch default (Kaiming uniform for Conv1d, Linear)`
- `decision-note: spec-default — R001 와 동일: lr=1e-3, weight_decay=1e-4, batch=64, epochs=100, early-stop patience=10, seed=42, Huber loss δ=1.0, AdamW, cudnn.deterministic=True, fold ensemble mean, device cuda:0`
- `decision-note: spec-default — C001 학습 device = cuda:0 (R001 와 동일). 다중 GPU 환경에서도 0번 만 사용 (CUDA_VISIBLE_DEVICES 환경변수 의존 X)`
- `decision-note: spec-default — c7 LB 제출 = autonomous loop 가 사용자 승인 없이 dacon-submit skill 1회 호출 (CLAUDE.md autonomous policy + plan-003 c14 동일 패턴)`
- `decision-note: spec-default — model.type 키 신규 도입 (gru-residual | cnn1d-residual). R001 config 후방호환: model.type 부재 시 gru-residual default 적용. registry 의 기존 R001~R006 행 영향 0`
- `decision-note: spec-default — param count 불일치 (GRU layers=2 hidden=64 ≈ 50k params, CNN1D layers=2 channels=64 kernel=3 ≈ 25k params) 는 caveat #1 박제. param-match 변형 (channels=96 등) 은 본 plan 의 scope X (별도 plan)`

---

## §1. 배경

### §1.1 plan-003 결과 인계 (registry + analysis/plan-003/results.md)

| exp_id | plan | method | cv_mean_eucl ± std | LB hit@1cm |
|---|---|---|---|---|
| **B001_linear-2pt** | 001 | polyfit (w=2, d=1) closed-form | 0.012941 ± 0.000584 | **0.60** |
| S001_cspline-natural-full | 002 | cspline natural 11pt | 0.017418 ± 0.000713 | 0.4932 |
| **R001_baseline-residual-gru** | 003 | linear-2pt + GRU(64,2,0.1) huber + relative | **0.013383 ± 0.000718** | (= R006) |
| R002 physics features | 003 | + physics features (input_dim=13) | 0.015157 ± 0.000499 | 미제출 |
| R003 ema-extrapolate | 003 | baseline_type=ema (α=0.5) | 0.014038 ± 0.000976 | 미제출 |
| R004 wingbeat-oscillation | 003 | + FFT (input_dim=12) | 0.013476 ± 0.000684 | 미제출 |
| R005 loss-mse | 003 | huber → mse | 0.013388 ± 0.000580 | 미제출 |
| R006_combined-winners | 003 | winning=0 → R001 비트 동일 사본 | 0.013383 ± 0.000718 | **0.5688** |

핵심 인계 사실 (plan-003 §4 + winning_trace):

- **R001 lean residual-GRU 가 B001 closed-form 보다 5/5 fold 모두 paired-loss** (mean Δ = +0.000442, sign agreement 5/5). LB 도 동방향 (B001 0.60 > R006 0.5688, Δ = -0.0312).
- ablation 4 component (physics/EMA/wingbeat/loss-MSE) 모두 R001 대비 non-winning (mean Δ ∈ [+5e-6, +1.8e-3]) → R006 = R001 trivial 분기.
- plan-003 §9 의 12 개 다음 plan 후보 중 **#4 Architecture 비교 (TCN/Transformer/MLP-residual)** 의 부분 instance 가 본 plan.
- CV-LB ρ ≈ +0.90 prior (plan-002 분석, plan-003 §6 외부 검증 통과) → C001 의 CV 가 R001 (0.013383) 보다 ↓ 면 LB 0.5688 보다 ↑ 가능성, ↑ 면 LB ↓ 가능성.

### §1.2 본 plan 의 가설 출발점

사용자 제시 (대화 anchor):

> "plan-003 의 메인 모델 부분 (GRU) 만 1D CNN 으로 교체."

원리:

- residual = `y_true - linear_extrap(X)` 의 분포는 시퀀스 11pt × 3축의 short-range temporal pattern. GRU 의 *순차적 hidden state* 보다 *고정 receptive field convolutional filter* 가 short-range residual pattern 에 더 적합할 가능성.
- 11pt 시퀀스에서 kernel=3 의 2-layer 1D CNN 의 effective receptive field = 5pt → 잔차의 local 변화 (직선 외삽 대비 +/- 2 step 떨림) 를 hard-coded inductive bias 로 학습.
- 학습 안정성: CNN 은 RNN 의 vanishing gradient 문제 없고, parallel 학습 가능 → 속도 ↑ (R001 의 fold 당 ~7s 수준에서 더 빠를 가능성).
- 같은 channel/layer (64, 2) 로 lean baseline 원칙 유지 — param count 는 CNN 이 더 적음 (caveat #1 박제).

비판적 prior (plan-003 §4 finding 의 architecture-invariance 가설):

- plan-003 의 결정적 finding = "잔차 ~mm 영역에서 neural baseline 의 학습 가치는 closed-form floor 대비 음수". 이게 *GRU 특이성* 이면 CNN 으로 회복 가능; *architecture-invariant 한 데이터 특성* (잔차의 mm-scale + 10k sample 부족) 이면 CNN 도 동일 floor 미달.
- 본 plan 의 1 LB 점수 회수가 이 prior 의 *직접 검증*: C001 LB > 0.60 면 architecture 의 가치, C001 LB ≤ 0.5688 정도면 architecture-invariant prior 강화.

### §1.3 본 plan 의 결정적 근거

H1 (CNN1D vs GRU): 같은 hidden/layers/loss/baseline/feature 조건에서 1D CNN 의 inductive bias (locality + translation invariance) 가 11pt × 3축 short-range residual 학습에 GRU 보다 적합 → cv_mean_eucl 가 R001 대비 ↓.

기각 시 (Δ ≥ 0):
- CNN1D 의 locality bias 가 본 데이터에 mismatch (= R001 의 GRU 도 사실상 short-range pattern 만 사용 중, architecture 차이 음영) → plan-005 후보로 *kernel size sweep* (kernel ∈ {3, 5, 7}) 또는 *receptive field 확장* (dilated conv, TCN).
- 또는 *neural baseline 자체가 closed-form floor 미달이 architecture-invariant* — plan-003 finding 강화 → plan-005 후보로 *Ensemble (B001 + neural) weighted mean*.

채택 시 (Δ < 0):
- architecture 축의 정보 확보 → plan-005 후보로 *multi-architecture grid* (CNN + TCN + Transformer-encoder, channel=64, layers=2 동일).

각 결과 (CV verdict + LB 1점) 가 다음 plan 들의 anchor.

---

## §2. Scope

### §2.1 In-scope

| 항목 | 값 |
|---|---|
| 모델 | ResidualCNN1D(input_dim=3, channels=64, layers=2, kernel=3, dropout=0.1, padding="same") — **1 exp 만** |
| 변경 변수 (vs R001) | **`model.type` (gru-residual → cnn1d-residual) + `model.kernel` (신규 키, 값=3) 만**. 그 외 모든 키 R001 비트 동일 |
| 학습 | PyTorch + AdamW (lr=1e-3, weight_decay=1e-4), Huber loss δ=1.0, batch=64, epochs=100, early-stop patience=10, seed=42 — **R001 와 비트 동일** |
| Baseline 외삽 | linear-2pt (= B001 식; R001 와 비트 동일) |
| Input feature | relative coords (input_dim=3; R001 와 비트 동일) |
| Inference | fold ensemble mean (5 ckpt 평균; R001 와 비트 동일) |
| target time | +80 ms (스펙 고정) |
| primary dev metric | mean 3D Euclidean distance (m) |
| 보조 metric | per-axis MAE, hit_rate @ {0.05, 0.10, 0.20, 0.50} m |
| CV | 5-fold, seed=42, `src/io.py:kfold_split` 그대로 재사용 (R001 와 동일 fold 분할 → fold-level paired 비교 가능) |
| Test 예측 (LB) | C001 fold ensemble mean → 1 submission.csv |
| LB 제출 | 1개만 dacon-submit skill 자율 호출 (사용자 승인 X) — lb_exp_id = C001 (fallback 분기 없음 — 본 plan 1 exp) |

### §2.2 Out-of-scope (절대 안 함)

| 항목 | 이유 |
|---|---|
| CNN channels/layers/kernel/dropout sweep | 단일 변수 위반. 별도 plan (CNN hyperparameter sweep) |
| param-match 변형 (channels=96 등) | 본 plan = "메인 모델만 교체" 의 lean baseline 원칙. param-match 는 별도 plan |
| TCN / Transformer / MLP-residual | 본 plan 의 scope 가 *1D CNN 단일 교체*. multi-architecture grid 는 plan-005 후보 |
| Loss / baseline / feature 변경 | plan-003 의 R002~R005 ablation 결과 인계 — 본 plan 은 *architecture 축만* 측정 |
| Ensemble (C001 + R001 + B001) | 별도 plan |
| TTA / data augmentation | 별도 plan |
| 2개 이상 LB 제출 | 사용자 결정 — 1점만. 본 plan 의 의무 산출은 C001 LB 1점 |
| 11pt → 다른 길이 시퀀스 | 데이터 스펙 고정 |
| fold 별 ensemble weight tuning | 단순 mean (R001 와 동일) |
| polyfit/cspline/GRU 분기 회귀 수정 | plan-001/002/003 산출 영구. backward-compat smoke 만 통과시킴 |
| ckpt 의 VCS 추적 | 무거운 binary, `.gitignore` 로 제외 (plan-003 동일 규칙) |

---

## §3. 사전 등록 (Pre-registration)

### §3.1 Fold split

- plan-001/002/003 §3.1 과 **완전 동일** 함수 (`src/io.py:kfold_split(ids, k=5, seed=42)`).
- 같은 fold = 같은 val ids → C001 와 R001/B001 의 fold-level paired 비교 가능.

### §3.2 합격 기준

| 조건 | 정의 |
|---|---|
| A. 인프라 정상 | `pytest tests/` green; `from src.models.residual_cnn1d import ResidualCNN1D` import OK; G0 backward-compat smoke (R001 재실행 cv_mean_eucl 절대오차 `|Δ| < 1e-5` vs registry 박제값 0.013383). G0 환경 규약: §4.5 의 seed/cudnn 정량 spec 동일 적용 |
| B. 산술 안정성 | C001 fold-level 학습 + OOF prediction 에 NaN/Inf 0건. training loss 가 1 epoch 내 NaN 발생 시 `nn_numerical` severe |
| C. 수렴 | C001 5fold paired mean Δ vs B001 ≤ +0.005. 부호 규약 = `Δ_fold[i] = C001.fold_metrics[i].mean_eucl - B001.fold_metrics[i].mean_eucl`. B001 cv_mean_eucl 박제값 = 0.012941 (registry 행 B001_linear-2pt). 위반 시 `residual_no_convergence` severe |
| D. sanity | C001 5-fold mean(cv_mean_eucl) < 0.030 (mean 기준, max-fold 아님). 위반 시 `nn_no_signal` severe |
| E. 단일 변수 검증 | §0.5 합격 기준의 "diff 3줄 fixed" 규약: 신규 키 추가 {`model.type=cnn1d-residual`, `model.kernel=3`} + rename {`model.hidden=64` → `model.channels=64`}. 검증 도구 = `src/diff_config.py` (신규 helper, c4 의무) 의 canonical key map normalize. 위반 시 `single_variable_violation` severe |
| F. 비교 박제 | `analysis/plan-004/results.md` 에 C001 × {cv_mean_eucl±std, per-axis MAE, hit_rate@{0.05,0.10,0.20,0.50}, vs B001 paired Δ (mean + 5 fold + sign agreement), vs R001 paired Δ (mean + 5 fold + sign agreement + fold-σ_R001 배수)} 표 |
| G. **lb_exp_id 1 LB 제출 (의무, autonomous loop 자율 실행)** | `lb_exp_id` 값 = **`C001_baseline-residual-cnn1d`** (registry id 와 동일 full string). C001 `submission.csv` 생성 (스키마 100% 일치) → **autonomous loop 가 사용자 승인 없이 `dacon-submit` skill 1회 호출** → API 응답 `isSubmitted=True` 회수 → results frontmatter `lb_exp_id: C001_baseline-residual-cnn1d` + `lb_submission_path` + `lb_submitted_at` 박제. **LB 점수 회수 (dacon.io 수동) 는 carry-over 패턴 — `lb_score: null` + `status: partial` 으로 G_final partial 종료 허용** (plan-003 §6 carry-over 동일). 자율 호출 자체 fail (`isSubmitted=False` 또는 skill error) 시 `lb_unsubmitted` severe |

### §3.3 평가 점수 / 집계 (plan-003 §3.3 와 동일)

- per fold metric: `mean_eucl`, per-axis MAE, hit_rate(r) for r ∈ {0.05, 0.10, 0.20, 0.50}
- per exp metric: 5 fold mean (1차) ± **표본 std (ddof=1)** (2차) — `numpy.std(..., ddof=1)` 또는 동치
- exp 비교: 단일 exp (C001) → R001 paired Δ + B001 paired Δ 만
- R001 vs C001 **same-fold paired Δ**: 5 fold 각각의 mean_eucl 차이 + 부호 일관성 (5 fold 중 음수 비율, *not* "동일 부호" — 음수 = C001 우월이 본 plan 의 채택 시그널)
  - 데이터 소스: R001 → `runs/baseline/R001_baseline-residual-gru/history.json` 의 `fold_metrics[i].mean_eucl` 필드 (i=0..4) (plan-003 산출, 불변); C001 → `runs/baseline/C001_baseline-residual-cnn1d/history.json` 의 동일 필드
  - Δ 정의 (부호 규약 통일): `Δ_fold[i] = C001.fold_metrics[i].mean_eucl - R001.fold_metrics[i].mean_eucl` (음수 = C001 우월)

### §3.4 Architecture winning 기준 (results.md anchor)

- **fold-σ_R001 정의**: R001 의 5 fold `mean_eucl` 표본 std (ddof=1). 데이터 소스 = `runs/baseline/R001_baseline-residual-gru/history.json` 의 `fold_metrics[i].mean_eucl` (i=0..4) 5개 값 → `numpy.std(values, ddof=1)`. plan-003 §2.1 박제값 = **0.000718** (이 값을 본 plan 검증 임계값으로 hardcode; 재계산 결과가 `|Δ| > 1e-6` 으로 다르면 `backward_compat_drift` severe — plan-003 산출 불변 보장).
- `winning_loose(C001) = paired mean Δ vs R001 < 0` (loose 기준)
- `winning_strict(C001) = (paired mean Δ vs R001 < 0) ∧ (|paired mean Δ| ≥ 0.000718) ∧ (sign agreement ≥ 4/5 fold 음수)` (strict 기준)
- 두 기준 모두 박제 (loose / strict). frontmatter 의 `architecture_winning_loose` / `architecture_winning_strict` bool 필드 (§7).

---

## §4. STAGE 0 — 인프라 (G0)

### §4.1 `src/models/residual_cnn1d.py` (신규)

```python
import torch
import torch.nn as nn
from torch import Tensor


class ResidualCNN1D(nn.Module):
    """1D CNN 잔차 예측 모듈 (R001 ResidualGRU 의 architecture-swap 대응).

    입력: (B, T=11, input_dim) — feature_fn 출력 그대로 (dtype: torch.float32)
    출력: (B, 3) — 잔차 (Δx, Δy, Δz)  (dtype: torch.float32)

    구조 (layers=2 fixed in this plan; layers ≥ 1 generalizable but lean baseline 원칙 으로 본 plan 은 2 만):
        Conv1d(input_dim → channels, kernel, padding="same") + ReLU + Dropout
        Conv1d(channels → channels, kernel, padding="same") + ReLU + Dropout
        마지막 timestep feature [:, :, -1] → Linear(channels → 3)

    *"Residual" 명칭 = 학습 paradigm (baseline 외삽 + neural 잔차) 의미. ResNet skip-connection (shortcut + add) 미적용. BatchNorm 미적용. projection 미적용. lean baseline 원칙 — caveat #1 박제.*
    """

    def __init__(self, input_dim: int, channels: int = 64, layers: int = 2,
                 kernel: int = 3, dropout: float = 0.1):
        super().__init__()
        if not isinstance(input_dim, int) or input_dim < 1:
            raise ValueError(f"input_dim must be positive int, got {input_dim!r}")
        if layers < 1:
            raise ValueError(f"layers >= 1, got {layers}")
        if kernel % 2 == 0:
            raise ValueError(f"kernel must be odd for padding='same', got {kernel}")

        convs = []
        in_ch = input_dim
        for _ in range(layers):
            convs.append(nn.Conv1d(in_ch, channels, kernel_size=kernel, padding="same"))
            convs.append(nn.ReLU())
            convs.append(nn.Dropout(dropout))
            in_ch = channels
        self.conv = nn.Sequential(*convs)
        self.fc = nn.Linear(channels, 3)

    def forward(self, X: Tensor) -> Tensor:
        # X: (B, T, C_in) → permute → (B, C_in, T) → conv → (B, C_out, T)
        h = self.conv(X.permute(0, 2, 1))
        # 마지막 timestep feature: (B, C_out, T) → (B, C_out)
        last = h[:, :, -1]
        return self.fc(last)
```

- `input_dim`: required positional arg (default 없음 — caller 가 `_build_model` 에서 cfg 값 전달).
- `channels=64` (R001 hidden=64 width-match), `layers=2` (R001 GRU layers=2 depth-match), `kernel=3` (effective receptive field at layers=2 = 5pt).
- `padding="same"` (PyTorch ≥ 1.10 필요; 미지원 시 `cnn1d_import_error` severe). output T=11 보존. 마지막 timestep 추출 = `h[:, :, -1]` (= R001 의 `out[:, -1, :]` 와 동일 의미).
- Dropout 위치: 모든 Conv1d 뒤 (마지막 Conv1d 뒤에도 ReLU+Dropout). Linear 직전에 Dropout 1회 → train 시 forward 마지막 단계 정규화. R001 (`nn.GRU(dropout=)` 은 layer 사이만) 와 미세한 비대칭 — caveat #10 박제.
- weight init: PyTorch default (Conv1d: Kaiming uniform fan_in; Linear: Kaiming uniform fan_in) — R001 default 와 통계적 동등치 아님 (caveat #11 박제, 정보성).

### §4.2 `src/training/train_residual.py` 확장 — model factory (public API)

기존 R001~R006 의 학습 함수 (`train_fold`) 는 model 객체를 외부 주입 받음 (plan-003 §4.4 spec). 본 plan 은 *모델 생성 부분만* public factory 로 분리:

```python
# src/training/train_residual.py
import torch.nn as nn


def build_model(cfg: dict) -> nn.Module:
    """Public model factory. cfg["model"]["type"] 키로 분기.

    후방호환: model.type 부재 시 'gru-residual' default 적용 (plan-003 R001~R006 영향 0).
    Import path (public): `from src.training.train_residual import build_model`.
    """
    model_cfg = cfg["model"]
    model_type = model_cfg.get("type", "gru-residual")

    if model_type == "gru-residual":
        from src.models.residual_gru import ResidualGRU
        return ResidualGRU(
            input_dim=model_cfg["input_dim"],
            hidden=model_cfg.get("hidden", 64),
            layers=model_cfg.get("layers", 2),
            dropout=model_cfg.get("dropout", 0.1),
        )
    elif model_type == "cnn1d-residual":
        from src.models.residual_cnn1d import ResidualCNN1D
        return ResidualCNN1D(
            input_dim=model_cfg["input_dim"],
            channels=model_cfg.get("channels", 64),
            layers=model_cfg.get("layers", 2),
            kernel=model_cfg.get("kernel", 3),
            dropout=model_cfg.get("dropout", 0.1),
        )
    else:
        raise ValueError(f"unknown model.type: {model_type!r} (expected one of: 'gru-residual', 'cnn1d-residual')")
```

- **Public API name** = `build_model` (private `_build_model` 아님 — caller 가 `src/run.py` / `src/submit.py` 에서 import 하므로 public). private 명칭 vs cross-module call 충돌 회피.
- 기존 `train_fold(model, ...)` 시그너처 변경 X — caller 측 (`src/run.py` 의 `_train_and_predict_residual_fold` 또는 helper) 에서 `build_model(cfg)` 한 번 호출 후 주입.
- **Key dispatch 규약**: GRU 분기는 `cfg["model"]["hidden"]` 만 read; CNN1D 분기는 `cfg["model"]["channels"]` 만 read. R001 config 에 `model.hidden=64` 만 존재 (`model.channels` 부재) → GRU 분기 진입 (default `model.type=gru-residual`) → cv 일치 보장. C001 config 에 `model.channels=64` 만 존재 (`model.hidden` 부재) → CNN1D 분기 진입. 두 키 동시 존재 시 `single_variable_violation` severe.
- R001 config 에 `model.type` 키 부재 → default `gru-residual` 적용 → 재실행 cv 절대오차 `< 1e-5` (backward-compat smoke 합격 기준 = `backward_compat_drift` severe 방지).
- `src/diff_config.py` (신규 helper, c4 의무): canonical key map `{"hidden": "channels"}` 으로 R001↔C001 의 *의미적* diff 카운트 (= 3줄 fixed 검증). single_variable_violation 자동 검증에 사용.

### §4.3 `src/run.py` method dispatch 확장

기존 `method="gru-residual"` 분기 보존. 본 plan 의 분기 옵션 2가지:

**옵션 A (권장, 채택)**: `cfg["model"]["type"]` 키만으로 model factory 분기 (§4.2 의 `build_model`). `method` 키는 그대로 `gru-residual` 유지 (= "residual learning paradigm" 의 의미; architecture 는 model.type 으로 분리).

**옵션 B (기각)**: `method="cnn1d-residual"` 신규 분기. `cfg["method"]` 와 `cfg["model"]["type"]` 의 의미 중복.

→ **채택 = 옵션 A**. `method` 는 residual learning 의 학습 패러다임 (= "linear_extrap + neural 잔차"), `model.type` 은 neural 부분의 architecture. 의미 분리 깔끔.

- **`method` 키 의미 (재정의)**: residual learning paradigm 식별자 (= "baseline 외삽 + neural 잔차 예측 + Huber/MSE loss + early-stop"). R001~R006 + C001 모두 `method: gru-residual` 동일 — 이게 single_variable_violation 위반 아닌 이유 (architecture 만 model.type 으로 분기).
- **Registry 표기**: `method` 컬럼은 두 exp 모두 `gru-residual` → registry 만으로는 architecture 식별 불가. 식별은 `exp_id` (R prefix vs C prefix) + `config.snapshot.yaml` 의 `model.type` 으로. 후속 paired 비교 자동 식별 시 `model.type` lookup 필수 (caveat #12 박제).
- 후방호환 보존: R001 config (model.type 부재) → default gru-residual → R001 재실행 cv 절대오차 `< 1e-5`.
- C001 config: `method: gru-residual` (= residual paradigm 동일), `model.type: cnn1d-residual` (= architecture 만 변경).

### §4.4 `tests/test_residual_cnn1d.py` (신규)

각 test 의 합격 기준은 정량 표현. seed=42 (numpy + torch + cuda) + cudnn.deterministic=True 사전 설정 fixture 사용.

| test | 검증 내용 (정량) |
|---|---|
| `test_forward_shape` | `ResidualCNN1D(input_dim=3).forward(torch.randn(8, 11, 3))` → shape `(8, 3)` exact (`assert out.shape == (8, 3)`); dtype `torch.float32` |
| `test_forward_shape_variable_input_dim` | input_dim ∈ {3, 12, 13, 22} 각각 → output shape `(8, 3)` exact. (input_dim 일반화 검증 — R002/R004/R006 차원 호환 보장; plan-005 multi-feature 활용 prior) |
| `test_parameter_count_in_range` | `n_params = sum(p.numel() for p in model.parameters())` for default (input_dim=3, channels=64, layers=2, kernel=3). `10_000 < n_params < 100_000` (sanity 범위; 실제 expected ≈ 25k — caveat #1) |
| `test_1batch_overfit` | seed=42 고정. random `X = torch.randn(16, 11, 3)`, random `target = torch.randn(16, 3)`. optimizer = `torch.optim.AdamW(model.parameters(), lr=1e-3)`, loss = `nn.HuberLoss(delta=1.0)`. epoch 0 forward pass 의 `loss_init = loss_fn(model(X), target).item()` 박제. 50 epoch full-batch 학습 (per epoch: zero_grad → forward → loss → backward → step) 후 `loss_final = loss_fn(model(X), target).item()`. **합격 = `loss_final <= 0.1 * loss_init`** (90% 이상 감소) |
| `test_backward_compat_r001` | R001 config dict (`{"model": {"input_dim": 3, "hidden": 64, "layers": 2, "dropout": 0.1}}` — `model.type` 부재) → `build_model(cfg)` 호출 → `isinstance(result, ResidualGRU)` True. C001 config dict (`{"model": {"type": "cnn1d-residual", "input_dim": 3, "channels": 64, "layers": 2, "kernel": 3, "dropout": 0.1}}`) → `isinstance(result, ResidualCNN1D)` True |
| `test_padding_same_preserves_length` | `model = ResidualCNN1D(input_dim=3, channels=64, layers=2, kernel=3, dropout=0.0)`. forward hook 으로 `model.conv` 의 각 Conv1d output shape capture → `(8, 64, 11)` (T=11 보존). hook 등록 = `module.register_forward_hook(lambda m, i, o: captured.append(o.shape))`, 검증 = 모든 capture 의 `shape[-1] == 11` |
| `test_diff_config_canonical` | `src/diff_config.py` 의 `count_semantic_diff(R001_cfg, C001_cfg, canonical={"hidden": "channels"})` → 3 (= §3.2 E 의 "diff 3줄 fixed") |

추가 test (G0 smoke):
- `tests/test_models.py` (신규 또는 기존 확장) — `build_model({"model": {"type": "gru-residual", "input_dim": 3}})` → ResidualGRU; `build_model({"model": {"type": "cnn1d-residual", "input_dim": 3}})` → ResidualCNN1D; `build_model({"model": {"type": "unknown", "input_dim": 3}})` → `ValueError` raise 검증.

### §4.5 G0 backward-compat smoke

- R001 (plan-003 산출) config 로 5-fold 재학습 → cv_mean_eucl 계산.
- **재현성 환경 (정량)**:
  - `torch.manual_seed(42)`, `torch.cuda.manual_seed_all(42)` (R001 fold seed = 42 + fold_idx, plan-003 §0.5 동일)
  - `numpy.random.seed(42)`
  - `torch.backends.cudnn.deterministic = True`
  - `torch.backends.cudnn.benchmark = False`
  - `torch.use_deterministic_algorithms(True)` (PyTorch ≥ 1.8; nondeterministic op fallback warn 무시)
  - device = `cuda:0` 강제 (cuda 부재 환경에선 `cuda_unavailable` decision-note → CPU + epochs=50; 단 이 경우 backward_compat 4자리 일치 미보장 — `cv_drift_cpu_mode` warn 박제, severe X)
- 합격 기준: `|cv_mean_eucl_new - 0.013383| < 1e-5` (절대오차). 위반 시 `backward_compat_drift` severe — model factory 도입이 GRU 분기에 영향 → 즉시 멈춤.
- 검증 비용: ~37s (R001 학습 시간, plan-003 §2.1 박제), G0 단계에서 1회만. R001 ckpt 재사용 X (학습 자체를 처음부터 재실행).

---

## §5. STAGE 1 — C001 학습 + LB 제출 (G1 + G_final)

### §5.1 C001 config

`configs/baseline/C001_baseline-residual-cnn1d.yaml`:

```yaml
exp_id: C001_baseline-residual-cnn1d
type: baseline
plan_id: "004"
baseline_id: R001_baseline-residual-gru   # 비교 anchor 명시 (registry baseline_id 필드)
method: gru-residual                       # residual paradigm (factory 분기 키는 model.type)
feature_components: [relative]             # R001 와 동일
baseline_type: linear                      # R001 와 동일
loss_type: huber                           # R001 와 동일
t_target: 80
k: 5
seed: 42
model:
  type: cnn1d-residual                     # ← 본 plan 의 단일 변경 변수 (신규 키 +1)
  channels: 64                              # ← R001 hidden=64 의 rename (canonical map: hidden ↔ channels)
  layers: 2                                 # R001 layers=2 와 depth-match
  kernel: 3                                 # ← 본 plan 의 신규 키 +1 (R001 부재)
  dropout: 0.1                              # R001 와 동일
  input_dim: 3                              # R001 와 동일
training:
  lr: 1.0e-3                                # R001 와 동일
  weight_decay: 1.0e-4                      # R001 와 동일
  batch: 64                                 # R001 와 동일
  epochs: 100                               # R001 와 동일
  early_stop_patience: 10                   # R001 와 동일
  seed: 42                                  # R001 와 동일
```

- **의미적 diff vs R001** (= §0.5 합격 기준의 "diff 3줄 fixed" 규약 검증 대상):
  1. `model.type` 신규 키 (R001 부재 → cnn1d-residual)
  2. `model.kernel` 신규 키 (R001 부재 → 3)
  3. `model.hidden=64` → `model.channels=64` rename (canonical key map `{"hidden": "channels"}`)
- **메타 키 diff** (의미적 diff 카운트에서 제외 — `single_variable_violation` 검증 시 ignore):
  - `exp_id`: R001 → C001 (재명명, identity)
  - `baseline_id`: B001 → R001 (비교 anchor)
  - `plan_id`: "003" → "004"
- 그 외 모든 키 (method, feature_components, baseline_type, loss_type, t_target, k, seed, model.layers/dropout/input_dim, training.\*) R001 와 비트 동일.

### §5.2 학습 + 검증

- 5-fold 학습 (seed=42, fold seed=42+fold_idx)
- device: cuda:0 (R001 와 동일)
- 예상 runtime: fold 당 ~5~10s (CNN 이 GRU 보다 빠를 예상), 전체 ~30~60s (R001 37.4s 와 비교)
- best ckpt: `runs/baseline/C001_baseline-residual-cnn1d/ckpt/fold{0..4}.pt` (.gitignore)
- summary.json: cv_mean_eucl, per-axis MAE, hit_rate, fold_best_val_mean_eucl, fold_best_epoch, total_train_time_sec

### §5.3 Submission 생성 (c6)

`src/submit.py` 확장:

- 기존 R001~R006 분기 (model.type 부재 시 gru-residual default) 비트 동일 보존.
- C001 분기 (model.type=cnn1d-residual): `build_model(cfg)` (§4.2 public API) + 5 ckpt load + fold ensemble mean predict → `runs/baseline/C001_baseline-residual-cnn1d/submission.csv`.
- **Key dispatch 규약 (submit.py)**: §4.2 와 동일 — `cfg["model"]["type"]` 부재 시 default `gru-residual` → R001~R006 분기 (`model.hidden` read); 명시 `cnn1d-residual` 시 → C001 분기 (`model.channels` read). 두 키 동시 read 시도 X (분기 후 한쪽만 read). single_variable_violation 검증은 §4.4 `test_diff_config_canonical` 가 자동 처리.
- ckpt path 규약: `runs/baseline/{exp_id}/ckpt/fold{i}.pt` (i=0..4) — plan-003 §0.5 와 동일 (R prefix → CNN prefix C 만 다름, 디렉토리 패턴 동일).
- 스키마 검증: 10000 rows, column order = sample_submission, ID 열 일치 (string equality). 위반 시 `submission_schema_fail` severe.
- output filename = `submission.csv` (run dir 안). plan-001/002/003 동일 규약.

### §5.4 LB 제출 (c7, autonomous loop 자율 실행)

- **`dacon-submit` skill 자율 호출 (사용자 승인 X)** — CLAUDE.md autonomous policy + plan-003 c14 동일 패턴.
- 입력: C001 `submission.csv` 절대경로 + 메타 (comment="plan-004 C001 residual-cnn1d single-var swap vs R001")
- **자율 호출 retry 정책**: 1차 호출 fail (network/auth/quota error) 시 30초 대기 후 1회 retry. 2차도 fail 시 `lb_unsubmitted` severe (즉시 멈춤). 자율 호출 자체 success (`isSubmitted=True`) 면 G_final 의무 통과 — LB 점수 회수는 별개.
- API 응답 박제: `analysis/plan-004/lb_log.md` 1행 — `submitted_at` (KST timestamp), `submission_filename` (run dir 기준 상대경로), `api_response` (JSON 전체), `lb_score: null (carry-over)`.
- **LB 점수 회수 패턴 (carry-over)**: DACON API 는 `post_submission_file` 만 제공 (score query 미지원). 사용자가 dacon.io 대회 페이지 (236716) `mysubmission` 에서 score 확인 → server agent 에 전달 → 별도 commit 으로 4 파일 동시 갱신:
  - `analysis/plan-004/lb_log.md` 의 lb_score 컬럼
  - `registry.csv` 의 C001 행 notes 컬럼 (`+lb=0.XX`)
  - `plans/plan-004-residual-cnn1d.results.md` frontmatter (`lb_score`)
  - `analysis/plan-004/results.md` 의 종합 표
- 회수 전: `status: partial`. 회수 후: `status: all_complete`. plan-003 §6 carry-over 동일.

### §5.5 budget

- 본 plan 사용: 1 슬롯
- 잔여: 4 슬롯/일 (다음 plan 또는 사용자 결정)

---

## §6. 작업량 총 회계

| 단위 | 수량 |
|---|---|
| 신규 코드 파일 | 3 (`src/models/residual_cnn1d.py`, `src/diff_config.py`, `tests/test_residual_cnn1d.py`) |
| 수정 코드 파일 | 3 (`src/training/train_residual.py` — public `build_model` factory, `src/run.py` — `cfg["model"]["type"]` 분기 (`gru-residual` ↔ `cnn1d-residual`), `src/submit.py` — 동일 dispatch) |
| 신규 config | 1 (`configs/baseline/C001_baseline-residual-cnn1d.yaml`) |
| 신규 run dir | 2 (`runs/baseline/C001_baseline-residual-cnn1d/`, G0 의 R001 재실행은 임시 run dir `runs/baseline/R001_baseline-residual-gru/.smoke/` — 산출물 폐기, registry 미박제, ckpt 별도 보관 X) |
| 신규 분석 dir | 1 (`analysis/plan-004/` — `results.md`, `lb_log.md`) |
| Exp 학습 | **C001 1 exp × 5 fold = 5 ckpt (의무 산출)** + G0 smoke R001 재실행 1 exp × 5 fold (임시 산출, smoke 검증 후 폐기 OK) |
| 자율 LB 제출 | 1 슬롯 (C001) |
| 신규 commit | 8 (c1~c8) |
| G-gate | 3 (G0, G1, G_final) |
| 예상 총 runtime | 5~10 분 = G0 R001 재실행 ~40s + G0 pytest ~10s + C001 학습 ~60s + submission ~5s + LB call ~5s + docs/commit 잔여. 분 단위 breakdown: G0 ~1분, G1 ~1분, G_final ~3~8분 (LB 응답 대기 가변) |

---

## §7. results.md 필수 항목

`plans/plan-004-residual-cnn1d.results.md` (frontmatter + 본문):

### Frontmatter
- `plan_id: 004`
- `finished_at: YYYY-MM-DDTHH:MM+09:00` (KST)
- `status: all_complete | partial | failed | canceled` (carry-over pending = `partial`)
- `exp_ids_completed: [C001_baseline-residual-cnn1d]`
- `exp_ids_skipped: []`
- `best_exp_id_cv: <C001_baseline-residual-cnn1d 또는 R001_baseline-residual-gru>` — 결정 규약: `paired_delta_vs_r001 < 0` 면 C001, 아니면 R001. 부호 규약 = §3.3 의 `Δ = C001 - R001` (음수 = C001 우월).
- `lb_exp_id: C001_baseline-residual-cnn1d` (full registry id string)
- `lb_score: <float 또는 null>` (carry-over pending 시 `null`; 회수 완료 시 dacon.io score 박제)
- `lb_submission_path: runs/baseline/C001_baseline-residual-cnn1d/submission.csv` (repo-relative)
- `lb_submitted_at: <KST timestamp>` (dacon-submit skill API 응답의 `isSubmitted=True` 회수 시점)
- `paired_delta_vs_r001: <float>` (5fold mean, 부호 규약 = §3.3)
- `paired_sign_agreement_vs_r001: <int>/5` (5 fold 중 Δ<0 (= C001 우월) 비율; "동일 부호" 아님)
- `paired_delta_vs_b001: <float>` (5fold mean, 부호 규약 = §3.2 C — `C001 - B001`)
- `paired_sign_agreement_vs_b001: <int>/5` (5 fold 중 Δ<0 비율)
- `architecture_winning_loose: <bool>` (= `paired_delta_vs_r001 < 0`)
- `architecture_winning_strict: <bool>` (= `(paired_delta_vs_r001 < 0) ∧ (|paired_delta_vs_r001| ≥ 0.000718) ∧ (paired_sign_agreement_vs_r001 ≥ 4)`)
- `fold_sigma_r001: 0.000718` (§3.4 박제값; 검증 시 history.json 에서 재계산해 `|Δ_sigma| < 1e-6` 일치 보장)
- `train_device: cuda:0` (cuda_unavailable fallback 시 `cpu` + `epochs=50` decision-note 박제)
- `total_train_time_sec: <float>` (C001 5 fold 학습 시간 합)

### 본문 섹션 (요약 표 + paired comparison + LB + caveat + 다음 plan 후보)

| 섹션 | 내용 |
|---|---|
| §1 종합 표 | C001 + R001 + B001 비교 행 (cv_mean_eucl ± std, per-axis MAE, hit@0.10, lb_score, train time) |
| §2 per-experiment 분석 | C001 fold_best_val_mean_eucl, fold_best_epoch, 학습 안정성, 특이사항 |
| §3 paired comparison | C001 vs R001 (5 fold Δ + mean Δ + fold-σ 배수 + sign agreement); C001 vs B001 (5 fold Δ + mean Δ) |
| §4 architecture winning verdict | loose / strict 두 기준 박제 + H1 (CNN1D vs GRU) 채택/기각 |
| §5 LB 회수 결과 | C001 LB score, R001 LB 0.5688 / B001 LB 0.60 와의 비교, CV-LB ρ=+0.90 prior 검증 |
| §6 caveats | param count 불일치, kernel=3 fixed, layers=2 fixed 등 |
| §7 다음 plan 후보 (enumeration) | local 의사결정 권한; server 우선순위 X |

---

## §8. 통계 함정 & caveats

- **caveat #1 (param count 불일치)**: GRU(input_dim=3 → hidden=64, layers=2) ≈ 50k params (3·64·4 input gates + 2·64·64·4·2 recurrent + bias ≈ 50k), CNN1D(input_dim=3 → channels=64, layers=2, kernel=3) ≈ 25k params (3·64·3 + 64·64·3 + bias ≈ 13.4k; 추정치, 정확값은 test_parameter_count_in_range 에서 측정). 같은 width/depth 라도 capacity 가 다름 → C001 의 결과를 "동일 architecture capacity 비교" 로 해석 금지. *lean baseline 원칙* (same hidden/layers 로 가장 자연스러운 swap) 채택. param-match 비교는 별도 plan.
- **caveat #2 (winning 기준)**: plan-003 §3.4 와 동일 loose 기준 (paired mean Δ < 0). strict 기준 (|Δ| ≥ fold-σ_R001 + sign ≥ 4/5) 도 박제. R001 fold-σ = 0.000718 (§3.4 박제값) → strict winning 임계값.
- **caveat #3 (config 키 이름 변경 hidden→channels)**: ResidualGRU 의 `hidden` vs ResidualCNN1D 의 `channels` 는 *의미적 동치 키* (width=64). canonical key map `{"hidden": "channels"}` 으로 single_variable_violation 검증 시 동치로 인정. `src/diff_config.py` 의 `count_semantic_diff` 함수가 자동 처리.
- **caveat #4 (kernel size fixed)**: kernel=3 fixed (effective receptive field 5pt at layers=2). 11pt 시퀀스 전체를 커버하려면 kernel ≥ 5 또는 dilated conv 필요. kernel sweep 은 별도 plan.
- **caveat #5 (padding=same)**: PyTorch ≥ 1.10 의 `padding="same"` 필요. version mismatch 또는 syntax error 시 `cnn1d_import_error` severe (import 단계 에러 — `nn_numerical` 카테고리와 분리).
- **caveat #6 (CV-LB ρ prior)**: plan-002 분석 ρ=+0.90, plan-003 §6 외부 검증 통과. C001 의 LB 회수 결과로 ρ 재검증. C001 CV ↓ + LB ↑ 동방향이면 ρ 유지, 비단조면 prior 재고.
- **caveat #7 (architecture-invariance 가설)**: plan-003 finding "neural baseline < closed-form floor" 의 *architecture 의존성* 여부. C001 LB > 0.60 면 GRU 특이성, ≤ 0.60 면 architecture-invariant prior 강화.
- **caveat #8 (fold ensemble interaction)**: 5 ckpt mean 의 *분산 감소 효과* 가 GRU vs CNN1D 에서 다를 수 있음. plan-003 R001 의 fold_best_val_mean_eucl = [0.01439, 0.01230, 0.01290, 0.01369, 0.01364] 의 std (0.000718) 와 C001 의 fold std 비교 박제. 박제 위치 = `analysis/plan-004/results.md` §2.
- **caveat #9 (CPU fallback drift)**: `cuda_unavailable` 시 device=cpu + epochs=50 fallback 진입하면 R001 backward-compat smoke 의 `|Δ| < 1e-5` 임계값 미보장 (cudnn deterministic 영역 외). `cv_drift_cpu_mode` warn 박제 후 진행 (severe X).
- **caveat #10 (Dropout 비대칭)**: ResidualCNN1D 는 모든 Conv1d 뒤에 Dropout 배치 (Linear 직전 마지막 Dropout 포함). ResidualGRU 는 `nn.GRU(dropout=)` 의 layer 사이 dropout 만 (마지막 layer 뒤 미적용). 미세한 정규화 강도 차이 — paired Δ 해석 시 architecture 본질 vs dropout 위치 confound 가능.
- **caveat #11 (weight init 비대등)**: PyTorch default Conv1d init = Kaiming uniform fan_in (gain=√5); GRU default init = uniform `[-1/√hidden, 1/√hidden]`. 통계적 동등치 아님 — C001 의 학습 trajectory 가 R001 과 다른 초기 조건에서 출발 (caveat #10 와 confound). lean baseline 원칙 으로 default 채택.
- **caveat #12 (registry method 컬럼 식별 불가)**: R001 + C001 모두 `method: gru-residual` (= residual paradigm 동일) → registry 의 method 컬럼만으로 architecture 식별 불가. paired comparison 자동 식별 시 `config.snapshot.yaml` 의 `model.type` 키 추가 lookup 필요.

---

## §9. 변경 이력

- v1 (2026-05-11): 초안 — plan-003 §9 #4 (Architecture 비교) 의 부분 instance. 1 exp (C001) + 1 LB 슬롯.
- v2 (2026-05-11): plan-review BLOCKER 12건 + 핵심 AMBIGUITY 8건 patch. 주요 변경:
  - §0.5 합격 기준 정량화: paired Δ 부호 규약 명시, "diff 3줄 fixed" 규약 (신규 키 2 + rename 1), backward_compat 임계값 통일 (`|Δ| < 1e-5`), `lb_unsubmitted` severe scope 축소 (isSubmitted=True 만 의무, score 회수는 carry-over).
  - §0.5 severe 추가: `cuda_unavailable` (CPU fallback decision-note), `cnn1d_import_error` (padding=same 호환성 분리).
  - §3.2 합격 기준 표 6항목 정량화 (G 의 lb_exp_id full string `C001_baseline-residual-cnn1d` 박제, partial 종료 허용).
  - §3.3 paired Δ 부호 규약 통일 (음수 = C001 우월), 표본 std ddof=1 명시.
  - §3.4 fold-σ_R001 = 0.000718 출처 박제 (R001 history.json fold_metrics[].mean_eucl 5개 표본 std ddof=1).
  - §4.1 ResidualCNN1D *Residual* 명칭 해석 명시 (paradigm-name, ResNet skip-connection 아님). input_dim default 부재 명시. dtype torch.float32 명시. ValueError 정량 예외.
  - §4.2 `_build_model` → `build_model` public API promote (private/cross-module 충돌 해소). key dispatch 규약 (hidden ↔ channels 분기 후 한쪽만 read).
  - §4.3 method 키 의미 재정의 ("residual paradigm" — architecture 는 model.type 으로 분리). registry method 컬럼 식별 불가 caveat #12 박제.
  - §4.4 test 7개 정량화 (test_1batch_overfit: optimizer/lr/loss/50epoch full-batch, loss_final ≤ 0.1 × loss_init).
  - §4.5 G0 smoke seed/cudnn/deterministic 환경 정량 spec.
  - §5.1 C001 config 의 의미적 diff 3줄 명시 (신규 키 model.type + model.kernel, rename hidden→channels), 메타 키 (exp_id/baseline_id/plan_id) 카운트 제외 명시.
  - §5.3 submit dispatch 규약 명시 (key dispatch + ckpt path + filename).
  - §5.4 LB 자율 호출 retry 정책 (1회 30s 대기 후 retry 1회) + carry-over 4 파일 동시 갱신 규약.
  - §6 R001 재실행 임시 run dir `.smoke/` 명시, ckpt 별도 보관 X.
  - §7 frontmatter 부호 규약 + paired_delta_vs_b001 추가 + fold_sigma_r001 hardcode + cuda_unavailable fallback 박제.
  - §8 caveats 5개 추가 (#9 CPU drift, #10 Dropout 비대칭, #11 weight init 비대등, #12 registry method 식별 불가).

---

## §10. 참조

- `plans/plan-003-residual-gru-grid.md` (선행 plan, R001 baseline)
- `plans/plan-003-residual-gru-grid.results.md` (R001 cv 0.013383, R006 LB 0.5688)
- `analysis/plan-003/results.md` (paired comparison + caveat #5~#16)
- `analysis/plan-003/lb_log.md` (LB carry-over 패턴)
- `WORKFLOW.md §12` (Autonomous Execution Protocol)
- `CLAUDE.md` (Autonomous Execution Policy)
- `notes/mosquito-trajectory-ideas.md` (사용자 아이디어 source)
- `configs/baseline/R001_baseline-residual-gru.yaml` (C001 의 single-variable swap baseline)
