---
plan_id: 007
version: 1
date: 2026-05-12 (Asia/Seoul)
status: draft
based_on:
  - 004
  - 005
  - 006
  - notes/PB_0.6822 코드공유.ipynb
scope: single-formula CMA-ES tuning + basis ablation + per-sample MLP coefficient regression (no corrector redesign)
exp_ids:
  - F001_formula-ga
  - F002_formula-mlp
lb_score: null
---

# plan-007 v1 — Single-Formula CMA-ES + Basis Ablation + Per-Sample MLP Coefficient Regression

## §0. 한 줄 목적

> **plan-006 의 single-formula 64.91% (단일 공식 `frenet_par120_perp_neg020`) + 84% in 1.5cm histogram 을 출발점으로, 4 단계에 걸쳐 단일 공식 ceiling 을 *데이터로* 끌어올린다.**
>
> 검증 명제:
>
> 1. **Step 1**: train trajectory 의 sliding window sub-sample 이 *original (end_idx=10) 과 같은 분포* 에서 추출되는가? (mosquito 비행의 stationarity 검증)
> 2. **Step 2**: 기존 27-family 의 변수 (d1, acc_par, acc_perp, d2, jerk, time_scale) 만으로 CMA-ES 최적화 시 단일 공식의 hit ceiling 은? (baseline ceiling 측정)
> 3. **Step 3**: 새 basis (speed_slope·d1, rotation_term, ‖d1‖·acc_par, v_mean3) 를 *하나씩* 추가하며 각 basis 의 *marginal* hit contribution 은? (basis ablation)
> 4. **Step 4**: Step 3 의 best basis 위에 *per-sample MLP coefficient regression* 을 얹으면 단일 공식 ceiling 을 얼마나 돌파하는가? (heterogeneity 적응)
>
> **본 plan 의 LB 제출 = Step 2 + Step 3 (2 회)**. Step 4 의 LB 제출은 후속 plan / carry-over (synthesis 단계에서 plan-008 후보로 박제).
>
> **Corrector 재설계는 plan-008 로 분리** (본 plan scope 미포함, 단일 공식 + selector 대체 까지만).

---

## §0.5 Quick Reference (autonomous loop 매 turn 읽는 section)

### 합격 기준 (G-gate sequence)

- Step 1 sliding window aug 의 distribution match: KS p > 0.075 ∨ quantile-by-quantile RMSE < 0.0015m (= plan-006 standard 의 *1.5배 완화*). 위반 시 aug 비사용 분기 (10K train) — `sliding_window_distribution_drift` warn-only (severe X).
- Step 2 CMA-ES OOF hit (single formula, 기존 변수) finite + `0.62 ≤ x ≤ 0.78` (inclusive). 위반 시 `cma_es_out_of_range` severe.
- Step 3 ablation 의 각 변수 marginal hit gain ≥ 0.001 인 변수만 best basis 에 포함. cutoff 이하는 drop. 18 ablation 회 (4 변수 × 진단 검증 + 변수 cumulative) 의 산출 모두 `analysis/plan-007/basis_ablation.{json,md}` 박제.
- Step 4 MLP 학습 OOF hit finite + Step 3 best baseline 보다 ≥ 0.005 향상 (= per-sample 적응의 minimum gain). 위반 시 `mlp_no_improvement` severe (MLP 가 단일 global coeff 보다 못함 = arch/loss/data 문제).
- LB 제출: Step 2 끝 + Step 3 끝 = 총 **2회**. Step 4 끝은 본 plan 미제출 (후속 plan-008 또는 carry-over).
- `lb_score` frontmatter 3 파일 (`plans/plan-007-*.md` top + `.results.md` + `analysis/plan-007/results.md`) 동시 박제. plan-004/006 패턴 답습.

### G-gates

- G0: STAGE 1 sliding window distribution validity check 통과 [TODO]
- G1: STAGE 2 기존 변수 CMA-ES + OOF + LB 제출 1회 [TODO]
- G2: STAGE 3 새 변수 ablation 완료 + best basis 결정 + LB 제출 1회 [TODO]
- G3: STAGE 4 MLP coefficient regression + OOF 향상 박제 (LB 미제출) [TODO]
- G_final: STAGE 5 synthesis + plan-008 후보 + 3 파일 frontmatter 동시 박제 [TODO]

### Commit chain (next-up)

| # | type | spec section | status |
|---|---|---|---|
| c1 | docs | `plans/plan-007-formula-tuning.md` 작성 (본 파일) | [TODO] |
| c2 | code | `analysis/plan-007/sliding_validity.py` — STAGE 1 sliding window distribution match check. spec @ §4 | [TODO] |
| G0 | gate | KS p > 0.075 ∨ quantile RMSE < 0.0015 (aug 사용 여부 결정) | [TODO] |
| c3 | code | `analysis/plan-007/cma_es_baseline.py` — STAGE 2 기존 변수 CMA-ES fit. spec @ §5 | [TODO] |
| c4 | exp | F001-step2: CMA-ES baseline fit + OOF + submission 생성. spec @ §5 | [TODO] |
| c5 | sub-lb | STAGE 2 dacon-submit + lb_log row + frontmatter 갱신. spec @ §8 | [TODO] |
| G1 | gate | Step 2 OOF finite ∈ [0.62, 0.78] + LB 1회 완료 | [TODO] |
| c6 | code | `analysis/plan-007/basis_ablation.py` — STAGE 3 새 변수 순차 ablation. spec @ §6 | [TODO] |
| c7 | exp | F001-step3: 4 변수 × ablation + best basis 결정. spec @ §6 | [TODO] |
| c8 | sub-lb | STAGE 3 dacon-submit (best basis with all kept terms) + lb_log + frontmatter. spec @ §8 | [TODO] |
| G2 | gate | basis_ablation.json 박제 + LB 2회차 완료 + best basis 명시 | [TODO] |
| c9 | code | `analysis/plan-007/mlp_coeff.py` — STAGE 4 MLP coefficient regression. spec @ §7 | [TODO] |
| c10 | exp | F002: MLP 학습 + OOF 측정 (LB 미제출). spec @ §7 | [TODO] |
| G3 | gate | MLP OOF ≥ Step 3 best + 0.005 | [TODO] |
| c11 | synthesis | `analysis/plan-007/results.md` + `next_plan_candidates.md` (≥ 2 후보). spec @ §9 | [TODO] |
| G_final | gate | results.md + next plan 후보 ≥ 2 + 3 파일 frontmatter 동시 박제 | [TODO] |

### Plan-specific severe (WORKFLOW.md §12.3 default 위 추가분)

- `sliding_window_distribution_drift`: Step 1 의 KS p ≤ 0.075 ∧ quantile RMSE ≥ 0.0015m. **warn-only (severe X)**, aug 비사용 분기 (Step 2~4 가 10K train 사용).
- `cma_es_out_of_range`: Step 2 OOF hit `x < 0.62` 또는 `x > 0.78` (양쪽 strict). 추정 구간 [0.65, 0.72] 밖이면 implementation 버그 또는 변수 정의 mismatch.
- `cma_es_no_convergence`: CMA-ES 200 generations 후 best fitness 의 *직전 50 generations 변동* > 0.005 (수렴 실패).
- `basis_overlap`: Step 3 의 어느 변수가 *추가 시 hit 감소* (marginal < 0). 다른 변수와 redundant 또는 overfit 의심.
- `mlp_no_improvement`: Step 4 MLP OOF < Step 3 best + 0.005. per-sample 적응이 효과 없음 — 데이터 heterogeneity 가 *parametric* 이 아닌 가능성 또는 MLP arch/loss 문제.
- `submission_shape_mismatch`: plan-004/006 와 동일.
- `lb_unsubmitted`: Step 2 + Step 3 의 LB 회수 실패.
- `dacon_submit_skill_missing`: c5 / c8 진입 시 skill 부재.
- `lb_anomaly`: `|lb_score − 0.6692| ≥ 0.05` (plan-006 LB 기준, equality 포함 trigger). 양/음 무관 — 큰 이상 시 집중 분석.

### Plan-specific paths (WORKFLOW.md §12.5/§12.6 추가/제외)

- whitelist 추가:
  - `analysis/plan-007/**` (특히 `sliding_validity.py`, `cma_es_baseline.py`, `basis_ablation.py`, `mlp_coeff.py`, `*.json`, `*.md`)
  - `runs/baseline/F001_formula-ga/**`, `runs/baseline/F002_formula-mlp/**` (submission.csv + ckpt)
- blacklist 추가:
  - `src/pb_0_6822/**` (plan-004 lock-in, import only)
  - `runs/baseline/P001_*/**`, `runs/baseline/E001_*/**` (plan-004/006 산출, read-only)
  - `analysis/plan-{004,005,006}/**` (이전 plan 산출, read-only)
  - `plans/plan-{001..006}*` (앞선 plan 본문 수정 X)
  - `notes/PB_0.6822 코드공유.ipynb` (원본 보존)

### Decision-note 사용 예 (자율 결정 시 commit msg 박제)

- `decision-note: spec-default — Step 1 distribution match threshold = plan-006 의 1.5배 완화 (KS p>0.075, quantile RMSE<0.0015). 비-stationary 일부 허용 + aug 사용 분기 확보.`
- `decision-note: spec-default — Step 2 변수 = (d1, acc_par, acc_perp, d2, jerk, time_scale) 6 개. 27-family 가 실제 사용하는 motion term 만.`
- `decision-note: spec-default — Step 3 변수 추가 순서 = ② speed_slope·d1 → ① rotation_term → ④ ‖d1‖·acc_par → ③ v_mean3 (이전 chat 우선순위).`
- `decision-note: spec-default — Step 4 MLP arch = 1 hidden layer × 32 units (~300 params), GA global coeff 를 bias 로 init, soft_hit_loss (sigmoid 근사 sharpness=200).`
- `decision-note: spec-default — corrector 재설계는 plan-008 분리 (본 plan scope 미포함, 단일 공식 + per-sample selector 대체 까지).`
- `decision-note: spec-default — Step 4 LB 미제출 (후속 plan-008 또는 carry-over). 본 plan LB = Step 2 + Step 3 의 2 회.`
- `decision-note: spec-default — CMA-ES popsize=30, maxiter=200, sigma0=0.3. cma library 사용.`
- `decision-note: spec-default — DATA_ROOT = repo/data/, DEVICE = cuda:1 (plan-004/006 일관성).`

---

## §1. 배경

### §1.1 plan-006 인계 (key findings)

| 측정 | 값 | 출처 |
|---|---|---|
| 단일 공식 `frenet_par120_perp_neg020` argmax (corrected) OOF hit | **0.6491** | plan-006 §5.5 |
| 단일 공식 soft (가중평균) OOF | 0.6524 | plan-006 §2.1 |
| LB (soft 제출) | **0.6692** | plan-006 |
| Oracle (best of 27, raw) | 0.7188 | plan-005 |
| 단일 공식 누적 hit @ 1.5cm | **~84%** (추정, best-of-27 84.78%) | plan-005 corrector_decomp |
| 단일 공식 누적 hit @ 2cm | ~88% | plan-005 |
| Per-regime worst | regime 16 (n=354, hit=0.22), regime 17 (n=356, hit=0.26), regime 10 (n=546, hit=0.41) | plan-006 |

핵심 인계:
- 단일 공식의 hit 분포가 *unimodal + tight* → mosquito 비행은 *qualitatively heterogeneous* 가 아니라 *parametrically varied*. 단일 공식 + 풍부한 basis 로 70%+ 가능 추정.
- regime 16/17 의 hit 0.22~0.26 은 *trig 비선형 (rotation)* 또는 *speed-dependent extrapolation* 의 systematic miss 가능성. 진단 필요 = Step 3 ablation 의 직접 검증.

### §1.2 본 plan 의 핵심 가설

| 가설 | 검증 방법 | 합격 산출 |
|---|---|---|
| H1: sliding window aug 가 distribution 보존 | Step 1 KS / quantile match | aug 사용 분기 결정 |
| H2: 단일 공식 + 기존 변수 의 CMA-ES ceiling > 65% (현 baseline) | Step 2 CMA-ES fit | OOF hit, ~68~72% 추정 |
| H3: 새 변수 (speed_slope·d1, rotation_term, ‖d1‖·acc_par, v_mean3) 의 marginal contribution 양수 | Step 3 ablation | 각 변수 marginal 박제, best basis 결정 |
| H4: per-sample MLP coefficient regression 이 global 단일 공식 ceiling 돌파 | Step 4 MLP 학습 | OOF ≥ Step 3 + 0.005 |

→ H1~H4 모두 *데이터로 검증*. plan-007 의 진짜 가치는 검증된 가설의 *수치적 박제* (다음 plan 의 anchor).

---

## §2. Scope (명시적)

### §2.1 In-scope

| 항목 | 값 |
|---|---|
| 단일 공식 출발점 | `frenet_par120_perp_neg020` (plan-006 1등) |
| 데이터 증폭 | sliding window (Step 1 검증 통과 시) — train 10K → 60K |
| 최적화 | CMA-ES (cma library, popsize=30, maxiter=200) |
| 새 변수 | speed_slope·d1, rotation_term, ‖d1‖·acc_par, v_mean3 (4 개) |
| 모델 | per-sample MLP coefficient regression (1 hidden × 32) |
| 학습 | Step 4 MLP, soft_hit_loss (sigmoid 근사) |
| LB 제출 | 2 회 (Step 2 + Step 3 끝) — Step 4 끝은 미제출 |
| 산출 위치 | `analysis/plan-007/**`, `runs/baseline/F001_formula-ga/**`, `runs/baseline/F002_formula-mlp/**` |

### §2.2 Out-of-scope (절대 안 함)

| 항목 | 이유 |
|---|---|
| Corrector 재설계 | plan-008 로 분리 (본 plan scope 미포함) |
| 27 후보 풀 확장 (27 → 35) | 본 plan 은 *단일 공식 + per-sample 회귀* — selector 대체 path |
| Test-internal validation set | 별개 idea, plan-008 후보 |
| Selector arch 교체 (TCN/Transformer) | 본 plan 은 selector *대체*, 개선 X |
| 다중 LB 제출 (3 회 이상) | 본 plan LB = 2 회 (Step 4 끝은 후속) |
| z 축 독립 보정 | Step 3 진단 결과에 따라 *조건부 추가* (basis ablation 내) — 무조건 추가 X |
| End-to-end 학습 통합 | Step 4 MLP 가 *standalone* (corrector 와 결합 X) |
| plan-004/005/006 모듈 수정 | lock-in, import only |

---

## §3. 사전 등록 (Pre-registration)

### §3.1 입력 데이터 + 분할

| 분할 | 출처 | 사용 |
|---|---|---|
| Train original (10K, end_idx=10) | `data/train/` + `train_labels.csv` | Step 1 baseline distribution, Step 2~4 main fit |
| Train sliding (60K, end_idx ∈ [5, 10]) | sliding window 추출 | Step 1 비교 분포, Step 2~4 (검증 통과 시) |
| Test (10K, end_idx=10) | `data/test/` | Step 2/3 inference + submission, Step 4 inference |
| Fold 정의 | `selector.stable_fold_id(sample_id, 5)` | OOF 정합성 (plan-004/006 와 동일) |

### §3.2 합격 기준 (정량)

- **G0**: Step 1 의 (KS test p > 0.075) ∨ (quantile-by-quantile RMSE < 0.0015m). 둘 중 하나 통과 시 aug 사용, 둘 다 실패 시 aug 비사용 (warn-only).
- **G1**: Step 2 OOF hit finite + `0.62 ≤ x ≤ 0.78`. submission.csv schema == `sample_submission.csv`. LB 회수 (float 또는 TBD/null).
- **G2**: Step 3 의 4 변수 각각의 marginal hit gain 박제 (양수든 음수든). best basis = (기존 6 + marginal > 0.001 인 새 변수 N 개). submission.csv + LB 회수.
- **G3**: Step 4 MLP OOF ≥ Step 3 best + 0.005.
- **G_final**: `analysis/plan-007/results.md` 작성 + `next_plan_candidates.md` 후보 ≥ 2 (각 후보 4 항목 박제) + 3 파일 frontmatter `lb_score` 동시 갱신.

### §3.3 평가

- **OOF hit**: 5-fold OOF concatenated hit@1cm (plan-006 §3.3 식과 동일, threshold R_HIT = 0.01m)
- **CMA-ES fitness**: `−(err ≤ R_HIT).mean()` (negative hit count, CMA-ES minimizes)
- **MLP loss**: `soft_hit_loss = sigmoid(sharpness × (err − threshold)).mean()`, sharpness=200, threshold=0.01
- **LB**: dacon submission 응답의 `lb_score`. 2 회 제출 (Step 2 + Step 3).

---

## §4. STAGE 1 — Sliding Window Validity Check (c2)

### §4.1 측정 식

```python
# analysis/plan-007/sliding_validity.py
import numpy as np
from scipy import stats
from src.pb_0_6822 import selector

def stage1_sliding_validity() -> dict:
    """train trajectory 의 sliding window sub-sample 과 original (end_idx=10) 의
    residual distribution 비교. 단일 최고 공식 적용 후 residual 분포 match check."""

    ids, train_y = selector.read_labels(DATA_ROOT / "train_labels.csv")
    train_x = selector.load_stack(DATA_ROOT / "train", ids)

    # ── 1. Original residuals (end_idx=10) ──
    cands_orig = selector.make_candidates(train_x, train_x.shape[1] - 1, horizon=2)
    best_idx = 17   # frenet_par120_perp_neg020 (plan-006 박제)
    pred_orig = cands_orig[:, best_idx, :]
    err_orig = np.linalg.norm(pred_orig - train_y, axis=1)   # [N=10K]

    # ── 2. Sliding window residuals (end_idx ∈ [5, 9], horizon=1) ──
    # 각 sub-trajectory: (x[0:end+1], target=x[end+1])
    err_slide_list = []
    for end_idx in range(5, 10):       # 5 단계 sliding (5, 6, 7, 8, 9)
        cands_sub = selector.make_candidates(train_x, end_idx, horizon=1)
        target_sub = train_x[:, end_idx + 1]
        pred_sub = cands_sub[:, best_idx, :]
        err_sub = np.linalg.norm(pred_sub - target_sub, axis=1)
        err_slide_list.append(err_sub)
    err_slide = np.concatenate(err_slide_list)   # [50K]

    # ── 3. KS test (two-sample) ──
    ks_stat, ks_pvalue = stats.ks_2samp(err_orig, err_slide)

    # ── 4. Quantile-by-quantile RMSE ──
    quantiles = np.linspace(0.05, 0.95, 19)
    q_orig = np.quantile(err_orig, quantiles)
    q_slide = np.quantile(err_slide, quantiles)
    quantile_rmse = float(np.sqrt(((q_orig - q_slide) ** 2).mean()))

    # ── 5. Histogram comparison (informational) ──
    bins = [0.0, 0.005, 0.010, 0.015, 0.020, 0.030, 0.050, 0.100, np.inf]
    hist_orig, _ = np.histogram(err_orig, bins=bins)
    hist_slide, _ = np.histogram(err_slide, bins=bins)

    # ── 6. Decision ──
    aug_usable = (ks_pvalue > 0.075) or (quantile_rmse < 0.0015)

    return {
        "n_orig": len(err_orig),
        "n_slide": len(err_slide),
        "ks_statistic": float(ks_stat),
        "ks_pvalue": float(ks_pvalue),
        "quantile_rmse": quantile_rmse,
        "threshold_ks_p": 0.075,
        "threshold_quantile_rmse": 0.0015,
        "aug_usable": aug_usable,
        "histogram_bins": [float(b) for b in bins[:-1]] + ["inf"],
        "histogram_orig_counts": [int(h) for h in hist_orig],
        "histogram_slide_counts": [int(h) for h in hist_slide],
        "histogram_orig_pct": [float(h / len(err_orig)) for h in hist_orig],
        "histogram_slide_pct": [float(h / len(err_slide)) for h in hist_slide],
    }
```

### §4.2 산출

- `analysis/plan-007/sliding_validity.json` — 위 dict
- `analysis/plan-007/sliding_validity.md`:
  - 1 줄 결론 (aug_usable + 근거)
  - histogram 비교 표 (2 column, original % vs sliding %)
  - KS / quantile RMSE 박제

### §4.3 G0 합격 기준 (자동 판정)

- `aug_usable = True` (KS p > 0.075 ∨ quantile RMSE < 0.0015m)
- 통과 → Step 2~4 가 sliding aug (60K) 사용
- 실패 → Step 2~4 가 original only (10K) 사용 + `sliding_window_distribution_drift` warn-only flag

### §4.4 시간 예산

- ~30 초 (numpy + scipy CPU)

---

## §5. STAGE 2 — 기존 변수 CMA-ES Baseline (c3, c4)

### §5.1 변수 정의 (6 자유도, sample 무관 상수)

```
pred = p0 + a·d1 + b·acc_par + c·acc_perp + d·d2 + e·jerk + f·time_scale_term

where:
  p0       = x[:, end_idx]                                    (현 위치)
  d1       = x[:, end_idx] − x[:, end_idx − 1]                (마지막 속도)
  d2       = x[:, end_idx − 1] − x[:, end_idx − 2]            (직전 속도)
  acc      = d1 − d2                                          (가속도)
  tangent  = d1 / ||d1||                                      (진행 방향)
  acc_par  = (acc · tangent) × tangent                        (접선 가속)
  acc_perp = acc − acc_par                                    (직교 가속)
  prev_acc = d2 − (x[:, end_idx − 2] − x[:, end_idx − 3])
  jerk     = acc − prev_acc                                   (저크)
  time_scale_term = time_scale_factor × d1                    (linear time warp)
```

### §5.2 CMA-ES 학습

```python
import cma

def fitness_step2(params, p0, d1, acc_par, acc_perp, d2, jerk, ts_term, target):
    a, b, c, d, e, f = params
    pred = p0 + a*d1 + b*acc_par + c*acc_perp + d*d2 + e*jerk + f*ts_term
    err = np.linalg.norm(pred - target, axis=1)
    return -(err <= 0.01).mean()    # CMA-ES minimizes

x0 = [1.98, 1.20, -0.20, 0.0, 0.0, 0.0]   # plan-006 best 에서 시작
sigma0 = 0.3
es = cma.CMAEvolutionStrategy(x0, sigma0, {
    'popsize': 30, 'maxiter': 200, 'tolfun': 1e-5, 'seed': 20260606,
})
while not es.stop():
    solutions = es.ask()
    es.tell(solutions, [fitness_step2(s, ...) for s in solutions])
best_params = es.result.xbest
best_hit = -es.result.fbest
```

### §5.3 산출

- `analysis/plan-007/cma_es_step2.json` — best_params, best_hit, convergence_history
- `runs/baseline/F001_formula-ga/submission_step2.csv` — test prediction (best_params 적용)
- `runs/baseline/F001_formula-ga/submission.csv` = `submission_step2.csv` 사본 (LB 제출용)

### §5.4 G1 합격 기준 (자동 판정)

- `cma_es_step2.json["best_hit"]` finite + `0.62 ≤ x ≤ 0.78`
- CMA-ES 마지막 50 generations 의 fitness 변동 < 0.005 (수렴)
- `submission.csv` schema 4-line assert (plan-006 §6 동일)
- LB 회수 (Step 2 = 2026-05-12 이후 자율 dacon-submit 호출)

### §5.5 시간 예산

- CMA-ES: ~30 분 (200 gen × 30 pop × ~100ms eval)
- Test inference + submission: ~10 초
- LB 제출: ~수 분

---

## §6. STAGE 3 — 새 변수 Ablation (c6, c7)

### §6.1 새 변수 정의

#### ② speed_slope · d1 (우선순위 1 — cross-term, per-sample 적응 도입)

```
speed_slope = (||x[end] − x[end−1]|| − ||x[end−4] − x[end−5]||) / mean_speed
mean_speed  = mean(||x[t+1] − x[t]||) for t ∈ [end−4, end−1]
new_term    = speed_slope · d1     (sample 마다 다른 effective d1 coefficient)
```

#### ① rotation_term (우선순위 2)

```
omega       = atan2(cross(d2, d1).z, dot(d2, d1))   (signed angular velocity, xy 평면)
R(theta)    = 2D rotation matrix
rot_term    = R(omega · horizon) · d1 − d1
horizon     = 2  (plan-004 default 와 일관)
```

**3D 데이터 주의**: `cross(d2, d1)` 의 z 컴포넌트만 사용 (xy 평면 회전 가정). z 축 회전 (banking) 은 caveat §N+3 #2 박제.

#### ④ ‖d1‖ · acc_par (우선순위 3 — 또 하나의 cross-term)

```
speed_norm  = ||d1|| / mean_speed
new_term    = speed_norm · acc_par
```

#### ③ v_mean3 − d1 (우선순위 4)

```
v_mean3     = (x[end] − x[end−3]) / 3
new_term    = v_mean3 − d1
```

### §6.2 Ablation 순서 + 측정 식

```python
def stage3_ablation():
    """4 변수 cumulative ablation. 각 단계마다 CMA-ES 재최적화."""

    base_vars = ['d1', 'acc_par', 'acc_perp', 'd2', 'jerk', 'ts_term']
    new_vars  = ['speed_slope_d1', 'rotation_term', 'speed_norm_acc_par', 'v_mean3_minus_d1']

    results = []
    current_vars = list(base_vars)
    prev_hit = stage2_best_hit   # from Step 2

    for new_var in new_vars:
        current_vars.append(new_var)
        best_params, best_hit = cma_es_fit(current_vars)
        marginal_gain = best_hit - prev_hit

        results.append({
            "added_var": new_var,
            "current_vars": list(current_vars),
            "best_params": [float(p) for p in best_params],
            "best_hit": float(best_hit),
            "marginal_gain": float(marginal_gain),
            "kept": marginal_gain >= 0.001,
        })

        if marginal_gain < 0.001:
            current_vars.pop()   # rollback drop
        else:
            prev_hit = best_hit

    return {
        "ablation_steps": results,
        "best_basis_vars": current_vars,
        "best_basis_params": [float(p) for p in cma_es_fit(current_vars)[0]],
        "best_basis_hit": prev_hit,
    }
```

### §6.3 산출

- `analysis/plan-007/basis_ablation.json` — 4 단계 ablation 결과 + best basis
- `analysis/plan-007/basis_ablation.md` — markdown 표 (변수, marginal_gain, kept/dropped, cumulative hit)
- `runs/baseline/F001_formula-ga/submission_step3.csv` + `submission.csv` 갱신

### §6.4 G2 합격 기준 (자동 판정)

- 4 변수 모두 ablation 수행 (kept/dropped 결정 박제)
- `best_basis_hit` finite + `> stage2_best_hit` (= 새 변수 추가가 *적어도 noise 위*)
- `submission.csv` schema 4-line assert
- LB 회수 (Step 3 dacon-submit)

### §6.5 시간 예산

- 4 × CMA-ES = ~2 시간 (변수 늘어날수록 1 evaluation 살짝 느려짐)
- LB 제출: ~수 분

---

## §7. STAGE 4 — Per-Sample MLP Coefficient Regression (c9, c10)

### §7.1 Arch (작음 — ~300 params)

```python
import torch.nn as nn

class CoefficientMLP(nn.Module):
    """sample 별 trajectory features → coefficient vector 출력.
    Step 3 best basis 의 N 자유도에 대해 N 개 coefficient 회귀."""

    def __init__(self, feat_dim: int, n_coeffs: int, global_init: np.ndarray):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, 32),
            nn.SiLU(),
            nn.Linear(32, n_coeffs),
        )
        # GA global best 를 bias 로 init → 학습 안 해도 Step 3 best 와 동일 동작
        with torch.no_grad():
            self.mlp[-1].bias.copy_(torch.tensor(global_init, dtype=torch.float32))
            self.mlp[-1].weight.zero_()        # delta 처음엔 0

    def forward(self, traj_features: torch.Tensor) -> torch.Tensor:
        return self.mlp(traj_features)   # shape (batch, n_coeffs)
```

### §7.2 학습

```python
# Trajectory features: 기존 selector 의 make_seq_features (6 step × seq_dim) 평탄화 또는 통계 요약
traj_features = compute_trajectory_features(train_x)   # shape (N, feat_dim)

def soft_hit_loss(pred, target, threshold=0.01, sharpness=200):
    err = torch.norm(pred - target, dim=1)
    return torch.sigmoid(sharpness * (err - threshold)).mean()

model = CoefficientMLP(feat_dim, n_coeffs=len(best_basis_vars), global_init=stage3_best_params)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

for epoch in range(50):
    for batch in dataloader:
        coeffs = model(batch.traj_features)
        pred = compute_pred(batch.basis_terms, coeffs)
        loss = soft_hit_loss(pred, batch.target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()
        optimizer.zero_grad()
    val_hit = evaluate_hit_rate(model, val_loader)
    # early stop by val_hit
```

### §7.3 산출

- `analysis/plan-007/mlp_coeff.json` — best epoch, OOF hit, MLP weights summary
- `runs/baseline/F002_formula-mlp/checkpoint.pt` — best MLP state_dict
- `runs/baseline/F002_formula-mlp/oof_predictions.npz` — OOF predictions (LB 미제출, 후속 활용)

### §7.4 G3 합격 기준 (자동 판정)

- `mlp_coeff.json["oof_hit"]` finite + `> stage3_best_hit + 0.005` (per-sample 적응의 minimum gain)
- 미달 시 `mlp_no_improvement` severe + 분석 박제 (왜 안 됐나)

### §7.5 LB 제출 = **본 plan 미수행**

- Step 4 의 LB 제출은 후속 plan-008 또는 carry-over 단계 (synthesis 에서 plan-008 후보로 박제)
- 본 plan 의 §10 LB 제출 사이클은 Step 2 + Step 3 의 2 회만

### §7.6 시간 예산

- MLP 학습: ~20~40 분 (50 epoch × 10K~60K sample, cuda:1)
- OOF inference + 박제: ~5 분

---

## §8. LB 제출 정책 (c5, c8)

### §8.1 자율 호출

```python
# Step 2 끝 (c5)
Skill(skill="dacon-submit",
      args="runs/baseline/F001_formula-ga/submission.csv F001_formula-ga-step2")

# Step 3 끝 (c8)
Skill(skill="dacon-submit",
      args="runs/baseline/F001_formula-ga/submission.csv F001_formula-ga-step3")
```

### §8.2 응답 4-분기 처리 (plan-006 §7.2 답습)

| (isSubmitted, lb_score) | 처리 | frontmatter `lb_score` | status | severe |
|---|---|---|---|---|
| (True, float) | full success | `<float>` 소수 4자리 | `all_complete` | — |
| (True, None) | partial — carry-over commit `c5.1`/`c8.1` | `TBD` | `partial` | — |
| (False, *) | retry 1회 (60초 sleep). 일시적/영구 분류 plan-006 답습. 재실패 시 severe | `null` | `partial` | `lb_unsubmitted` |
| Skill exception | 즉시 escalate | `null` | `partial` | `dacon_submit_skill_missing` |

### §8.3 `analysis/plan-007/lb_log.md` 포맷

```markdown
| timestamp_kst             | exp_id                       | step | isSubmitted | lb_score | detail |
|---------------------------|------------------------------|------|-------------|----------|--------|
| 2026-05-12T15:00:00+09:00 | F001_formula-ga-step2        | 2    | true        | 0.68xx   | OK     |
| 2026-05-12T17:00:00+09:00 | F001_formula-ga-step3        | 3    | true        | 0.70xx   | OK     |
```

### §8.4 `lb_score` frontmatter 동시 갱신 (3 파일)

- `plans/plan-007-formula-tuning.md` top-level `lb_score` — Step 3 최종값으로 갱신 (Step 2 는 lb_log 만)
- `plans/plan-007-formula-tuning.results.md` frontmatter
- `analysis/plan-007/results.md` frontmatter

→ Step 2 LB 회수 시 lb_log 만 박제, frontmatter `lb_score` 는 *Step 3 값* 으로 통일. Step 3 이 최종 LB.

---

## §9. STAGE 5 — Synthesis + plan-008 후보 (c11)

### §9.1 `analysis/plan-007/results.md`

frontmatter:
```yaml
---
plan_id: 007
based_on:
  - 004
  - 005
  - 006
finished_at: <ISO8601 KST>
status: all_complete | partial
exp_ids_completed:
  - F001_formula-ga
  - F002_formula-mlp
lb_exp_id: F001_formula-ga-step3
lb_score: <float|TBD|null>
lb_submitted_at: <ISO8601 KST>
---
```

본문:
- Step 1 sliding window validity 결론 (aug 사용 분기)
- Step 2 baseline CMA-ES OOF + LB
- Step 3 ablation 4 변수 marginal table + best basis
- Step 4 MLP OOF (LB 미제출 박제)
- plan-006 단일 공식 64.91% → 본 plan 의 cumulative 향상 trajectory
- decision-note 박제 list

### §9.2 `analysis/plan-007/next_plan_candidates.md`

**최소 후보 2 개 (G_final 조건)**. 결과 분기별:

**시나리오 A — MLP 가 단일 공식 ceiling 돌파 (Step 4 OOF > Step 3 + 0.01)**:
1. **Step 4 LB 제출 + corrector 재설계 결합** — F002 의 OOF predictions 위에 plan-008 의 band-specific corrector
2. **Test-internal validation set 구축** — Step 4 의 일반화 검증 + hyperparam re-tune

**시나리오 B — MLP 가 marginal gain 만 (Step 4 OOF ≤ Step 3 + 0.01)**:
1. **단일 공식 framework 한계 인정 → 27 후보 풀 확장 (35+)** — plan-005 worst-100 분석 기반
2. **selector arch 교체 + 본 plan basis 후보 풀에 추가** — discrete + continuous 하이브리드

각 후보의 4 항목 박제:
- 근거 metric (Step 4 OOF, marginal vs Step 3, oracle 0.7188 와의 gap)
- 예상 ROI
- 작업 범위
- 선행 조건

### §9.3 G_final 합격 기준

- `results.md` + `next_plan_candidates.md` 모두 작성
- 후보 ≥ 2 + 4 항목 박제
- 3 파일 frontmatter `lb_score` 동시 갱신 + `status: all_complete` (또는 `partial` + 사유)
- 모든 G-gate [DONE]

---

## §N+1. 작업량 총 회계

- 코드: 4 file (`sliding_validity.py`, `cma_es_baseline.py`, `basis_ablation.py`, `mlp_coeff.py`, 각 ~100~200 lines)
- 학습:
  - CMA-ES 5 회 (Step 2 + Step 3 의 4 ablation) ≈ ~2~3 시간
  - MLP Step 4 ≈ ~30 분
- 분석: ~30 분
- LB 제출: 2 회 (Step 2 + Step 3)
- Synthesis: ~30 분
- **총 wall-time 예산: ~4~5 시간**

---

## §N+2. results.md 필수 항목

(plan-003/004/005/006 format 답습)

- exp_id (F001/F002), plan_id (007), based_on (004 + 005 + 006)
- lb_exp_id, lb_score (Step 3 최종), lb_submitted_at
- Step 1 sliding validity (aug 사용 여부 + KS p + quantile RMSE)
- Step 2 CMA-ES baseline OOF + LB
- Step 3 ablation 4 변수 marginal + best basis
- Step 4 MLP OOF (LB 미제출 박제)
- 단일 공식 cumulative ceiling trajectory (plan-006 → Step 2 → Step 3 → Step 4)
- plan-008 후보 ≥ 2 + 4 항목
- decision-note 박제 list

---

## §N+3. 통계 함정 & caveats

1. **Sliding window stationarity 가정**: Step 1 의 KS / quantile 검사는 *aggregate* distribution 만 비교. *조건부* distribution (예: high-speed sample 에서만 차이) 은 감지 못함. aug 사용 결정 후 Step 2~4 의 train CV 가 *over-aug noise* 일 수 있음 — overfitting 대비 OOF 강조.
2. **rotation_term 의 3D 가정**: `cross(d2, d1).z` 만 사용해 xy 평면 회전만 모델링. 모기가 *banking* (3D 회전, z 축 포함) 하면 누락. Step 3 ablation 의 rotation_term marginal gain 이 *충분히 크지 않으면* 3D 회전 검토 (plan-008).
3. **CMA-ES 의 local optimum 위험**: 7~10 자유도 + step function fitness landscape 는 *multi-modal*. CMA-ES 가 local optimum 에 갇힐 위험 → multi-start (3 random init) 권장 (단 본 plan v1 은 single-start, 시간 절약).
4. **MLP의 over-aug 위험**: sliding window 60K 학습 시 *over-aug noise* 가 MLP 가 train 분포를 *과적합* 할 수 있음. validation = end_idx=10 의 10K 만 (original distribution). soft_hit_loss + early stop 으로 완화.
5. **soft_hit_loss 의 sharpness 선택**: sharpness=200 은 step function 의 강한 근사. 너무 sharp 면 gradient sparse, 너무 smooth 면 hit count 와 괴리. sharpness ablation 은 plan-008 변수.
6. **Step 4 LB 미제출의 위험**: per-sample MLP 의 generalization 을 LB 로 검증 안 한 채 plan 종료. 후속 plan-008 의 첫 task = Step 4 산출 LB 제출 (carry-over).
7. **Corrector 미적용**: 본 plan 은 단일 공식 + MLP 까지. plan-006 의 corrector_decomp 가 보여준 +0.89pp 의 *boundary correction* 효과는 본 plan 산출에 *미적용*. plan-008 의 corrector 재설계와 결합 시 LB 추가 회수 가능.
8. **dacon LB 의 비동기**: plan-002/003/004/006 패턴 — `lb_score: TBD` carry-over 가능, follow-up commit 으로 갱신.

---

## §N+4. 변경 이력

- v1 (2026-05-12): 초안 — plan-006 인계 (단일 공식 64.91%, 84% in 1.5cm) + 4 단계 progression (sliding validity → CMA-ES baseline → basis ablation → MLP coeff regression). LB 2 회 (Step 2 + Step 3), Step 4 LB 후속 plan-008. G0~G_final 5 gate, commit chain c1~c11. corrector 재설계 명시적 미포함.

---

## §N+5. 참조

- `plans/plan-004-pb-0-6822-fullrun.md` (PB framework + LB 0.6806 박제)
- `plans/plan-005-pb-0-6822-diagnostic.md` (component contribution + oracle 0.7188)
- `plans/plan-006-minimal-variant-e-lb.md` (단일 공식 64.91% + 84% in 1.5cm 인계)
- `analysis/plan-005/corrector_decomp.{json,md}` (error histogram, near-miss band)
- `analysis/plan-006/variant_e_oof.json` (single-formula argmax measurement)
- `notes/PB_0.6822 코드공유.ipynb` (원본 framework)
- `WORKFLOW.md` §0.5, §11, §12 convention
- `CLAUDE.md` (autonomous execution policy)
- `src/pb_0_6822/{selector,boundary}.py` (plan-004 lock-in, import only — make_candidates / motion_terms / read_labels / load_stack / stable_fold_id 의 export 계약 사용)
- `src/submit.py` (dacon-submit infra)
- `cma` library (CMA-ES Python 구현, pip install cma)
