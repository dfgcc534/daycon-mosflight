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
- Step 3 ablation 의 각 변수 marginal hit gain ≥ 0.001 인 변수만 best basis 에 포함. cutoff 이하는 drop. 총 5 CMA-ES fit (4 단계 cumulative addition + 1 최종 best basis 재확인 — §6.2 algorithm) 의 산출 모두 `analysis/plan-007/basis_ablation.{json,md}` 박제.
- Step 4 MLP 학습 OOF hit finite + Step 3 best baseline 보다 ≥ 0.005 향상 (= per-sample 적응의 minimum gain). 위반 시 `mlp_no_improvement` severe (MLP 가 단일 global coeff 보다 못함 = arch/loss/data 문제).
- LB 제출: Step 2 끝 + Step 3 끝 = 총 **2회**. Step 4 끝은 본 plan 미제출 (후속 plan-008 또는 carry-over).
- `lb_score` frontmatter 3 파일 (`plans/plan-007-*.md` top + `.results.md` + `analysis/plan-007/results.md`) 동시 박제. plan-004/006 패턴 답습.

### G-gates

- G0: STAGE 1 sliding window distribution validity check 통과 [DONE 117eeb4] — aug_usable=True (quantile RMSE 0.001252 < 0.0015)
- G1: STAGE 2 기존 변수 CMA-ES + OOF + LB 제출 1회 [TODO]
- G2: STAGE 3 새 변수 ablation 완료 + best basis 결정 + LB 제출 1회 [TODO]
- G3: STAGE 4 MLP coefficient regression + OOF 향상 박제 (LB 미제출) [TODO]
- G_final: STAGE 5 synthesis + plan-008 후보 + 3 파일 frontmatter 동시 박제 [TODO]

### Commit chain (next-up)

| # | type | spec section | status |
|---|---|---|---|
| c1 | docs | `plans/plan-007-formula-tuning.md` 작성 (본 파일) | [DONE 884b831] (+ v1.1 e189754) |
| c2 | code | `analysis/plan-007/sliding_validity.py` — STAGE 1 sliding window distribution match check. spec @ §4 | [DONE 117eeb4] |
| G0 | gate | KS p > 0.075 ∨ quantile RMSE < 0.0015 (aug 사용 여부 결정) | [DONE] aug_usable=True |
| c3 | code | `analysis/plan-007/cma_es_baseline.py` — STAGE 2 기존 변수 CMA-ES fit. spec @ §5 | [DONE 7be76fa] |
| c4 | exp | F001-step2: CMA-ES baseline fit + OOF + submission 생성. spec @ §5 | [DONE b7a2a4a] oof=0.6403 |
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
| H2: 단일 공식 + 기존 변수 의 CMA-ES ceiling 을 *측정* (가설 기대값 ~68~72%, baseline 0.6491 대비). G1 gate 하한 0.62 는 *측정 신뢰성* 의 sanity 최저선 (가설 미달도 측정값으로 박제) | Step 2 CMA-ES fit | OOF hit, ~68~72% 추정 |
| H3: 새 변수 (speed_slope·d1, rotation_term, ‖d1‖·acc_par, v_mean3) 의 marginal contribution 양수 | Step 3 ablation | 각 변수 marginal 박제, best basis 결정 |
| H4: per-sample MLP coefficient regression 이 global 단일 공식 ceiling 돌파 | Step 4 MLP 학습 | OOF ≥ Step 3 + 0.005 |

→ H1~H4 모두 *데이터로 검증*. plan-007 의 진짜 가치는 검증된 가설의 *수치적 박제* (다음 plan 의 anchor).

---

## §2. Scope (명시적)

### §2.1 In-scope

| 항목 | 값 |
|---|---|
| 단일 공식 출발점 | `frenet_par120_perp_neg020` (plan-006 1등) |
| 데이터 증폭 | sliding window (Step 1 검증 통과 시) — train 10K → 40K (end_idx ∈ [5,8], horizon=2; §6.1 speed_slope 의 end−5 lookback 와 boundary-safe) |
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
| Train sliding (40K, end_idx ∈ [5, 8], horizon=2, target=train_x[:, end_idx+2]) | sliding window 추출 (§4.1) | Step 1 비교 분포, Step 2~4 (검증 통과 시) |
| Test (10K, end_idx=10) | `data/test/` | Step 2/3 inference + submission, Step 4 inference |
| Fold 정의 | `selector.stable_fold_id(sample_id, 5)`. sliding 의 4 view (end_idx ∈ [5,8]) 는 부모 sample_id 의 fold 를 *상속* (leakage 방지 — 같은 sample_id 의 view 가 fold split 양쪽에 분산 X) | OOF 정합성 (plan-004/006 와 동일) |

### §3.2 합격 기준 (정량)

- **G0**: Step 1 의 (KS test p > 0.075) ∨ (quantile-by-quantile RMSE < 0.0015m). 둘 중 하나 통과 시 aug 사용, 둘 다 실패 시 aug 비사용 (warn-only).
- **G1**: Step 2 OOF hit finite + `0.62 ≤ x ≤ 0.78`. submission.csv schema == `sample_submission.csv`. LB 회수 (float 또는 TBD/null).
- **G2**: Step 3 의 4 변수 각각의 marginal hit gain 박제 (양수든 음수든). best basis = (기존 6 + marginal_gain ≥ 0.001 인 새 변수 N 개; inclusive — §6.2 의 `kept = marginal_gain >= 0.001` 와 동일). submission.csv + LB 회수.
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

    # ── 1. Original residuals (end_idx = train_x.shape[1] - 1 = 10, horizon=2) ──
    # selector.make_candidates 는 (N, 27, 3) 의 candidate 예측을 반환. 27-family ordering 은
    # plan-004 §4 (selector.CANDIDATE_LIST) 박제 — frenet_par120_perp_neg020 는 그 list 의
    # index 17 (plan-006 §5.5 박제). 본 plan 본문은 외부 source 직접 read 없이 index 만 인용
    # (decision-note: spec-default — selector lock-in 의 ordering 신뢰, 변경 X).
    cands_orig = selector.make_candidates(train_x, train_x.shape[1] - 1, horizon=2)
    best_idx = 17   # frenet_par120_perp_neg020 (= plan-004 CANDIDATE_LIST[17], plan-006 박제)
    pred_orig = cands_orig[:, best_idx, :]
    err_orig = np.linalg.norm(pred_orig - train_y, axis=1)   # [N=10K]

    # ── 2. Sliding window residuals (end_idx ∈ [5, 8], horizon=2 — original 과 동일 task) ──
    # decision-note: spec-default — sliding 도 horizon=2 사용. original 의 target 은 train_y (외부 label)
    # 이고 sliding 의 target 은 train_x[:, end_idx + 2] (within-trajectory). 둘 다 "end_idx 로부터 2 step
    # ahead 예측의 residual" 이라는 *같은 task* 의 분포 → KS / quantile 비교가 stationarity 측정 의도와 정합.
    # end_idx ∈ [5, 8] 의 lower bound = 5 는 §6.1 의 speed_slope (x[end−5] 사용) 의 최소 history 와
    #   호환 (Step 3 ablation 까지 모든 변수에 boundary-safe). upper bound = 8 은
    #   target = x[end+2] = x[10] 가 train_x 안에 존재 (shape[1]=11 가정).
    err_slide_list = []
    for end_idx in range(5, 9):        # 4 단계 sliding (5, 6, 7, 8) → 4 × 10K = 40K
        cands_sub = selector.make_candidates(train_x, end_idx, horizon=2)
        target_sub = train_x[:, end_idx + 2]
        pred_sub = cands_sub[:, best_idx, :]
        err_sub = np.linalg.norm(pred_sub - target_sub, axis=1)
        err_slide_list.append(err_sub)
    err_slide = np.concatenate(err_slide_list)   # [40K]

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
- 통과 → Step 2~4 가 sliding aug 사용 (sliding 40K ∪ original 10K = 총 **50K** train pool, §5.2 동일)
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
  time_scale_factor = ||d1|| / global_mean_speed              (per-sample 스칼라,
                                                              global_mean_speed = mean(||d1||) over
                                                              **original train 10K only** (end_idx=10
                                                              의 d1, aug 분기와 무관 — 단 한 번 사전
                                                              계산 후 고정. coefficient 정의의 aug
                                                              불변성 확보). fast sample > 1, slow < 1)
  time_scale_term   = time_scale_factor × d1                  (= ||d1|| · d1 / global_mean_speed,
                                                              speed-quadratic 항, a·d1 과 *redundant 아님*
                                                              — coefficient 가 sample-dependent)
```

**Numerical safety (모든 helper 공통)**: `||d1||`, `mean_speed`, `global_mean_speed` 가 0 인 sample (정지 또는 history 부재) 에서 ZeroDivision 방지 — 모든 normalize 분모에 `eps = 1e-9` 더해 clamp (`x / (denom + eps)`). 직접 reciprocal (e.g., `tangent = d1 / max(||d1||, eps)`) 도 동일 효과. CMA-ES fitness 의 NaN propagation 방지.

### §5.2 CMA-ES 학습

```python
import cma

def fitness_step2(params, p0, d1, acc_par, acc_perp, d2, jerk, ts_term, target):
    a, b, c, d, e, f = params
    pred = p0 + a*d1 + b*acc_par + c*acc_perp + d*d2 + e*jerk + f*ts_term
    err = np.linalg.norm(pred - target, axis=1)
    return -(err <= 0.01).mean()    # CMA-ES minimizes

x0 = [1.98, 1.20, -0.20, 0.0, 0.0, 0.0]   # 첫 3 entry = plan-006 best `frenet_par120_perp_neg020`
                                          #   의 effective coefficient seed (a≈1.98 on d1,
                                          #   b=1.20 on acc_par, c=-0.20 on acc_perp). 출처:
                                          #   plan-006 §5.5 의 argmax single-formula fit.
                                          # decision-note: spec-default — *init seed only*, CMA-ES
                                          #   가 더 나은 optimum 으로 자유 이동. 정확 mapping 식이
                                          #   불확실해도 sigma0=0.3 의 exploration 폭이 충분.
sigma0 = 0.3
es = cma.CMAEvolutionStrategy(x0, sigma0, {
    'popsize': 30, 'maxiter': 200, 'tolfun': 1e-5, 'seed': 20260606,
})
# fit data source = G0 분기 결과:
#   aug_usable == True  → sliding 40K (end_idx ∈ [5,8], horizon=2, target = train_x[:, end_idx+2])
#                        ∪ original 10K (end_idx=10, target = train_y) = 총 50K
#   aug_usable == False → original 10K (end_idx=10, horizon=2, target = train_y) 만
# 모든 fold (5-fold OOF) 외부 single fit. 같은 sample_id 의 sliding view 는 같은 fold 상속 (§3.1).
# OOF 측정은 별도 fold-wise 재호출 (동일 seed=20260606, popsize=30, maxiter=200 고정).

def _stack_train_terms(aug_usable: bool) -> tuple[np.ndarray, ...]:
    """G0 결과에 따라 train sample stack 생성 후 §5.1 의 6 basis term 을 한 번에 계산.

    반환 (모두 np.ndarray, float32, sample order = aug_usable 별 fixed):
      - p0      : (M, 3)   현 위치 = train_x[:, end_idx]
      - d1      : (M, 3)
      - acc_par : (M, 3)
      - acc_perp: (M, 3)
      - d2      : (M, 3)
      - jerk    : (M, 3)
      - ts_term : (M, 3)   time_scale_term = time_scale_factor(per-sample) × d1 (§5.1)
      - target  : (M, 3)   horizon=2 ahead 의 ground-truth (original 은 train_y, sliding 은 train_x[:, end_idx+2])
    M = 50K (aug_usable=True) 또는 10K (False).
    fold id 는 별도 (M,) int8 array 로 함께 반환 (zip 으로 동시 indexing). 같은 sample_id 의 sliding view 는
    parent fold 를 상속해 leakage 방지 (§3.1).
    """
    ...

p0, d1, acc_par, acc_perp, d2, jerk, ts_term, target = _stack_train_terms(aug_usable)
while not es.stop():
    solutions = es.ask()
    es.tell(solutions, [fitness_step2(s, p0, d1, acc_par, acc_perp, d2, jerk, ts_term, target)
                        for s in solutions])
best_params = es.result.xbest         # → test inference + submission 에 사용
single_fit_best_hit = -es.result.fbest  # in-sample fitness (전체 50K/10K 단일 fit)

# 별도 OOF 측정: 5 folds × CMA-ES 재호출. 각 fold k 마다 train = 4 folds 만, val = 1 fold 의
# original end_idx=10. 각 fold 의 best_params 로 val predict → 5 fold concat hit = oof_hit_5fold.
# (G1 의 판정 대상은 oof_hit_5fold; single_fit_best_hit 는 convergence 진단용으로만 박제.)
def run_5fold_oof_cma_es(aug_usable: bool) -> float:
    """5-fold OOF CMA-ES re-fit. 각 fold k 의 train = §3.1 fold 정의의 4 folds 의 모든 sample
    (aug_usable=True 면 sliding view + original; False 면 original 만). val = remaining fold 의
    *original end_idx=10 sample 만* (§7.2 와 동일 규약 — sliding view 는 val 에서 제외).
    각 fold 마다 새 CMA-ES instance (popsize=30, maxiter=200, sigma0=0.3, seed=20260606),
    fold 의 best_params 로 val predict. 5 fold concat → hit@1cm = oof_hit_5fold (float).
    """
    ...
oof_hit_5fold = run_5fold_oof_cma_es(aug_usable)
```

### §5.3 산출

- `analysis/plan-007/cma_es_step2.json` — `best_params` (single fit, test 적용), `single_fit_best_hit` (in-sample 진단용), `oof_hit_5fold` (G1 판정값), `convergence_history`
- `runs/baseline/F001_formula-ga/submission_step2.csv` — test prediction (best_params 적용)
- `runs/baseline/F001_formula-ga/submission.csv` = `submission_step2.csv` 사본 (LB 제출용)

### §5.4 G1 합격 기준 (자동 판정)

- `cma_es_step2.json["oof_hit_5fold"]` finite + `0.62 ≤ x ≤ 0.78` (= H2 검증 metric; OOF concatenated hit@1cm, plan-004/006 표준)
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
# d1, d2: shape (3,). np.cross(d2, d1): shape (3,), 그 중 z = [2] 인덱스.
omega       = atan2(np.cross(d2, d1)[2], np.dot(d2, d1))   (signed angular velocity, xy 평면)
R(theta)    = 2D rotation matrix applied to (d1[0], d1[1]); z component pass-through
rot_term    = R(omega · horizon)(d1) − d1                  (shape (3,) — xy 회전 + z 보존)
horizon     = 2  (plan-004 default 와 일관)
```

**3D 데이터 주의**: `np.cross(d2, d1)[2]` 만 사용 (xy 평면 회전 가정). z 축 회전 (banking) 은 caveat §N+3 #2 박제.

#### ④ ‖d1‖ · acc_par (우선순위 3 — 또 하나의 cross-term)

```
# mean_speed 정의는 ② 와 *동일* (decision-note: spec-default — t ∈ [end−4, end−1] 범위의
#   ||x[t+1] − x[t]|| 의 산술 평균. 본 plan 의 모든 'mean_speed' 참조는 단일 정의 사용).
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
def cma_es_fit(var_names: list[str]) -> tuple[np.ndarray, float]:
    """단일 helper. stage2 와 stage3 의 모든 ablation 에서 동일 호출.

    내부 규약 (변경 X — marginal_gain 측정의 noise floor 통일):
      - 입력 데이터 = §5.2 의 _stack_train_terms(aug_usable) 결과 (G0 분기 동일)
      - CMA-ES params: popsize=30, maxiter=200, tolfun=1e-5, sigma0=0.3, seed=20260606
      - x0 = (var_names 길이만큼 0.0, 단 d1/acc_par/acc_perp 의 첫 3 entry 는 plan-006 best 로 init)
      - fitness = -(err <= 0.01).mean()  (Step 2 식과 동일)
    반환: (best_params: shape (len(var_names),), best_hit: float = -es.result.fbest)
    """

base_vars = ['d1', 'acc_par', 'acc_perp', 'd2', 'jerk', 'ts_term']   # 'ts_term' = §5.1 의 'time_scale_term' 축약 (동치)
new_vars  = ['speed_slope_d1', 'rotation_term', 'speed_norm_acc_par', 'v_mean3_minus_d1']

def stage3_ablation():
    """4 변수 cumulative ablation. 각 단계마다 cma_es_fit 재호출 (동일 데이터 + 동일 seed)."""

    results = []
    current_vars = list(base_vars)
    prev_hit = stage2_best_hit   # from Step 2 — cma_es_fit(base_vars) 의 결과와 정확히 일치 (동일 helper)

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

### §6.3.1 Test-time inference (Step 2/3 공통)

```python
# Step 2 / Step 3 의 test prediction 생성. best_basis_vars 와 best_basis_params 만 다르고
# 식은 동일 — basis 항을 test_x (end_idx=10, horizon=2 사용) 위에서 계산 후 linear combination.

# var_name string ↔ basis term 식 매핑표 (compute_basis_terms 내부 dispatch).
# 각 term 의 식 출처는 §5.1 (base 6 변수) + §6.1 (new 4 변수).
BASIS_TERM_SPEC = {
    # base — §5.1
    'd1':                  lambda x, e, h: x[:, e]   - x[:, e-1],
    'd2':                  lambda x, e, h: x[:, e-1] - x[:, e-2],
    'acc_par':             lambda x, e, h: _acc_par(x, e),
    'acc_perp':            lambda x, e, h: _acc_perp(x, e),
    'jerk':                lambda x, e, h: _jerk(x, e),
    'ts_term':             lambda x, e, h: _time_scale_factor(x, e) * (x[:, e] - x[:, e-1]),
    # new — §6.1 (각 식은 §6.1 의 ②①④③ 정의 그대로)
    'speed_slope_d1':      lambda x, e, h: _speed_slope(x, e) * (x[:, e] - x[:, e-1]),
    'rotation_term':       lambda x, e, h: _rotation_term(x, e, horizon=h),
    'speed_norm_acc_par':  lambda x, e, h: (_d1_norm(x, e) / _mean_speed(x, e)) * _acc_par(x, e),
    'v_mean3_minus_d1':    lambda x, e, h: (x[:, e] - x[:, e-3]) / 3.0 - (x[:, e] - x[:, e-1]),
}

def compute_basis_terms(test_x: np.ndarray, end_idx: int, horizon: int,
                        var_names: list[str]) -> dict[str, np.ndarray]:
    """test_x: (N, T, 3). returns {var_name: (N, 3)} for var_name in var_names."""
    return {v: BASIS_TERM_SPEC[v](test_x, end_idx, horizon) for v in var_names}

def infer_test(test_x, best_basis_vars, best_basis_params):
    p0 = test_x[:, 10]                                # (N, 3)
    terms = compute_basis_terms(test_x, end_idx=10,
                                horizon=2,
                                var_names=best_basis_vars)   # dict {var_name: (N, 3)}
    pred = p0.copy()
    for coeff, var_name in zip(best_basis_params, best_basis_vars):
        pred = pred + coeff * terms[var_name]
    return pred   # (N, 3) — submission.csv 의 col2/3/4
# Step 2 의 best_basis_vars = base_vars (§5.1 의 6 변수 고정).
# Step 3 의 best_basis_vars = §6.2 의 marginal_gain ≥ 0.001 통과 변수 누적.
```

### §6.4 G2 합격 기준 (자동 판정)

- 4 변수 모두 ablation 수행 (kept/dropped 결정 박제)
- `best_basis_hit` finite + `≥ stage2_best_hit` (inclusive — §3.2 G2 / §6.2 의 `marginal_gain ≥ 0.001` 와 boundary 일관. 모든 새 변수가 drop 된 경우 base only 와 동치 = trivially pass 허용)
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
# Trajectory features (default: deterministic stats summary, sample 무관 상수 dim).
# decision-note: spec-default — stats summary 채택 (make_seq_features flatten 6×seq_dim 은 v2 후보).
#
# def compute_trajectory_features(x_window) -> np.ndarray, shape (N, feat_dim=13):
#     # x_window: (N, 6, 3) — last 6 steps before end_idx (포함 end_idx)
#     pos_mean  = x_window.mean(axis=1)               # (N, 3)
#     pos_std   = x_window.std(axis=1)                # (N, 3)
#     pos_range = x_window.max(axis=1) - x_window.min(axis=1)  # (N, 3)  — bounding-box 한 변
#     deltas    = np.diff(x_window, axis=1)           # (N, 5, 3)
#     speed_norms = np.linalg.norm(deltas, axis=2)    # (N, 5)
#     speed_mean = speed_norms.mean(axis=1, keepdims=True)  # (N, 1)
#     speed_std  = speed_norms.std(axis=1, keepdims=True)   # (N, 1)
#     speed_max  = speed_norms.max(axis=1, keepdims=True)   # (N, 1)
#     speed_last = speed_norms[:, -1:]                       # (N, 1)
#     return np.concatenate([pos_mean, pos_std, pos_range,
#                            speed_mean, speed_std, speed_max, speed_last], axis=1)   # (N, 3+3+3+1+1+1+1 = 13)
feat_dim = 13   # = pos_mean(3) + pos_std(3) + pos_range(3) + speed_mean(1) + speed_std(1) + speed_max(1) + speed_last(1)
traj_features = compute_trajectory_features(train_x[:, -6:])   # shape (N, 13)

def soft_hit_loss(pred, target, threshold=0.01, sharpness=200):
    err = torch.norm(pred - target, dim=1)
    return torch.sigmoid(sharpness * (err - threshold)).mean()

# OOF 5-fold loop. 각 fold k 마다:
#   - 매 fold 시작 시 model + optimizer **fresh reinit** (누적 학습 금지 — OOF leakage 방지).
#   - global_init=stage3_best_params 는 bias seed 로 모든 fold 에 공통 적용 (§7.1 arch).
for fold_k in range(5):
    model = CoefficientMLP(feat_dim, n_coeffs=len(best_basis_vars), global_init=stage3_best_params)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    # ... (이하 loop body 는 아래 epoch loop 와 batch contract 그대로 적용)
# (이 함수 골격은 의사코드 — 본문은 한 fold 만 단순화해 표시.)

# 한 fold 의 학습 루프 (위 fold_k loop 안에 들어감):
#   train_loader = 다른 4 folds 의 *모든 sample* (aug_usable=True 면 sliding view + original 둘 다 = 40K
#                  + 10K → 50K 의 4/5 = 40K; aug_usable=False 면 original 만 = 10K 의 4/5 = 8K). 같은
#                  sample_id 의 모든 view 는 같은 fold 에 묶이므로 4 folds 의 train 에는 *해당 sample_id 의
#                  모든 view 또는 전무* — leakage X.
#   val_loader   = remaining 1 fold 의 *original end_idx=10 sample 만* (distribution 일관성 —
#                  over-aug noise 평가 X). 즉 sliding view 는 val 에서 제외.
# 모든 fold concat → OOF hit (= 10K original 에 대한 5-fold OOF, plan-004/006 와 동일 정합).
# batch contract:
#   batch.traj_features : (B, feat_dim=13)        float32
#   batch.basis_terms   : (B, n_coeffs, 3)        float32 — §6.3.1 의 compute_basis_terms 결과를
#                                                  best_basis_vars 순서로 stack (var_name dim 사라짐)
#   batch.p0            : (B, 3)                  float32 — 현 위치 = train_x[:, end_idx]
#   batch.target        : (B, 3)                  float32
# coeffs (B, n_coeffs) 와 basis_terms (B, n_coeffs, 3) 의 broadcasting:
#   pred = p0 + (coeffs.unsqueeze(-1) * basis_terms).sum(dim=1)   # (B, 3)
def compute_pred(basis_terms: torch.Tensor, coeffs: torch.Tensor,
                 p0: torch.Tensor) -> torch.Tensor:
    return p0 + (coeffs.unsqueeze(-1) * basis_terms).sum(dim=1)

for epoch in range(50):
    for batch in dataloader:
        coeffs = model(batch.traj_features)                        # (B, n_coeffs)
        pred = compute_pred(batch.basis_terms, coeffs, batch.p0)    # (B, 3)
        loss = soft_hit_loss(pred, batch.target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()
        optimizer.zero_grad()
    val_hit = evaluate_hit_rate(model, val_loader)   # val_loader = 본 fold 의 original 10K subset
    # early stop spec: patience = 8 epoch, metric = val_hit (high-is-better),
    # min_delta = 1e-4. patience 초과 시 best_epoch state_dict restore 후 break.
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

→ **Edge case — Step 3 LB 회수 실패 (`null` 또는 `TBD`)**: 처리:
1. frontmatter `lb_score` = Step 3 회수 결과 (null/TBD) 그대로 박제 (Step 2 값으로 fallback X — *Step 3 가 본 plan 의 최종 LB* 라는 정책 유지).
2. frontmatter `lb_exp_id` 도 hard-coded `F001_formula-ga-step3` 유지 (실패도 박제 대상).
3. `status: partial` + 사유 `lb_step3_unrecovered`. carry-over commit (c8.1) 으로 후속 회수.
4. Step 2 LB 는 lb_log 에 그대로 남음 (정보 손실 X).

→ **Edge case — Step 2 LB > Step 3 LB**: 새 변수 추가가 LB 에서 *역효과* 라는 신호. 처리:
1. frontmatter `lb_score` 는 여전히 **Step 3 값** 으로 박제 (linear progression 의 *최종* LB 라는 의미 유지).
2. lb_log 에 추가 row + `detail: STEP2_BETTER` 표기.
3. `analysis/plan-007/results.md` 에 *regression* 절 추가 (OOF 와 LB 간 disagreement 분석).
4. `next_plan_candidates.md` 시나리오 분기에 본 신호를 *추가 anchor* 로 박제 (= 새 변수의 LB 일반화 실패는 단일 공식 framework 한계의 직접 증거 → 시나리오 B 의 후보 우선순위 상향).

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

> **두 임계의 분리** (G3 vs 시나리오 branch):
> - G3 (+0.005) = "MLP 가 global coeff *위* 어떤 식으로든 개선" 의 *최소* 통과선 (per-sample 적응이 noise 아닌지).
> - 시나리오 A/B branch (+0.010) = "MLP 가 다음 plan 의 *주력 무기* 로 쓸 만큼 *의미 있게* 개선" 의 임계. G3 통과 ≠ 시나리오 A (G3 만 통과하고 +0.005 ~ +0.010 사이면 시나리오 B 가 default).

**시나리오 A — MLP 가 단일 공식 ceiling *의미 있게* 돌파 (Step 4 OOF > Step 3 + 0.010)**:
1. **Step 4 LB 제출 + corrector 재설계 결합** — F002 의 OOF predictions 위에 plan-008 의 band-specific corrector
2. **Test-internal validation set 구축** — Step 4 의 일반화 검증 + hyperparam re-tune

**시나리오 B — MLP 가 marginal gain 만 (Step 4 OOF ≤ Step 3 + 0.010, G3 통과/미통과 무관)**:
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
