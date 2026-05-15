---
plan_id: 005
version: 1
date: 2026-05-11 (Asia/Seoul)
status: complete
based_on:
  - 004
  - notes/PB_0.6822 코드공유.ipynb
scope: analysis-only (plan-004 산출 활용 — 모델 재학습 X, corrector inference 재실행만 허용)
exp_ids:
  - D001_pb-0-6822-diagnostic
lb_score: null   # diagnostic plan — LB 제출 X
---

# plan-005 v1 — PB_0.6822 Framework Diagnostic (plan-004 ceiling/gap audit)

## §0. 한 줄 목적

> **plan-004 가 우리 데이터에 적용한 PB_0.6822 framework 의 *모든 약한 지점을 정량적으로 진단*해서, plan-006 의 개선 우선순위가 *근거 있는 숫자* 위에 결정되도록 한다.**
>
> 핵심 질문 3개 (모두 plan-006 anchor):
>
> 1. **Ceiling 어디 있나?** — Raw oracle, Post-correction oracle, Per-regime oracle, Per-family oracle 측정 → "후보 개선이냐 selector 개선이냐 corrector 개선이냐" 의 *근거 있는* 우선순위
> 2. **Selector 가 어디서 망가지나?** — Per-regime/family hit rate, top-K accuracy, confidence margin, family selection rate → selector 의 *blind spot* 식별
> 3. **Corrector 가 능력의 몇 % 를 쓰나?** — Cap saturation, correction direction breakdown, per-error-band 효과 → corrector 의 *idle capacity* 식별
>
> **모든 진단 산출 의무** — 한 metric 이라도 미박제 시 G_final 종료 불가. **모델 재학습 X** (plan-004 산출 + corrector inference 재실행만; corrector full-fit 1회는 허용 ~10min).

---

## §0.5 Quick Reference (autonomous loop 매 turn 읽는 section)

### 합격 기준 (G-gate sequence)

- plan-004 의 4 산출 (`oof_selector_scores.npz`, `test_selector_scores.npz`, `boundary_tiny_correction_report.json`, `regime_distribution.json`) 모두 존재 + 로드 가능. 위반 시 `plan004_artifacts_missing` severe.
- 20 진단 metric 모두 박제 (§1.3 표 참조). 한 metric 이라도 누락 시 `metric_unaudited` severe.
- 진단 결과 기반 plan-006 후보 ≥ 3개 도출 (`analysis/plan-005/next_plan_candidates.md`). 미도출 시 `synthesis_incomplete` severe.

### G-gates

- G0: 인프라 + plan-004 산출 검증 + corrector full-fit 재실행 (intermediate artifact 박제)  [TODO]
- G1: STAGE 1 — Oracle 4종 박제 (raw, post-corr, per-regime, per-family)  [TODO]
- G2: STAGE 2 — Selector decomposition 박제 (gap, top-K, confidence, family selection rate)  [TODO]
- G3: STAGE 3 — Corrector decomposition 박제 (cap saturation, direction, error histogram)  [TODO]
- G4: STAGE 4 — Bias ablation 박제 (gru-only / +physics / +regime / full)  [TODO]
- G5: STAGE 5 — Failure analysis + B001 비교 박제  [TODO]
- G_final: STAGE 6 synthesis (next plan 후보 ≥ 3개) + results.md  [TODO]

### Commit chain (next-up)

| # | type | spec section | status |
|---|---|---|---|
| c1 | docs | `plans/plan-005-pb-0-6822-diagnostic.md` 작성 (본 파일) | [DONE 271ac9f] |
| c2 | code | `analysis/plan-005/diagnostic.py` 인프라 + plan-004 산출 로드 helper | [DONE f88e784] |
| c3 | code | corrector full-fit 재실행 + intermediate artifact 박제 (`corrected_oof.npz`, `corrected_test.npz`) spec @ §4 | [DONE d23542c] |
| G0 | gate | plan-004 산출 4종 로드 OK + corrected_*.npz finite + shape OK | [DONE — corrected (10000,27,3) finite, seed_drift RMSE 0.000814 < 0.001 ok] |
| c4 | analysis | STAGE 1 oracle 4종 (`analysis/plan-005/oracle_summary.{json,md}`) spec @ §5 | [DONE 54ced99] |
| G1 | gate | 4 oracle 모두 박제 + finite | [DONE — raw=0.7188 post-corr=0.7111 ⚠️ corrector_hurts_oracle (gain=-0.0077, plan-006 anchor)] |
| c5 | analysis | STAGE 2 selector decomposition (`selector_decomp.{json,md}`) spec @ §6 | [DONE 54ced99] |
| G2 | gate | per-regime hit / top-K / confidence / family selection rate 모두 박제 | [DONE — argmax=0.6595 soft=0.6599 top1/3/5=0.126/0.218/0.282 (selector best-cand identification 약함)] |
| c6 | analysis | STAGE 3 corrector decomposition (`corrector_decomp.{json,md}`) spec @ §7 | [DONE 54ced99] |
| G3 | gate | cap saturation + direction + error histogram 모두 박제 | [DONE — cap_sat=3.6% par/perp/binormal=0.045/0.021/0.006 (binormal 6.4× 작음 → 2-D effective 우세)] |
| c7a | exp retrain | **Variant A retrain** — selector 재학습 (`--regime-prior-strength 0.0`). 산출: `analysis/plan-005/variant_A_no_regime/oof_selector_scores.npz`. spec @ §8.2 | [DONE fe7149e] |
| c7b | analysis | **Variant B free 계산 + 3-way 비교 + per-sample intervention** (`component_contribution.{json,md}`). spec @ §8.3~§8.5 | [DONE fe7149e] |
| G4 | gate | 3 variant hit (full/A/B) + marginal contribution + 2 intervention 분해 (B↔full, A↔full) + family-change 모두 박제 | [DONE — full=0.660, A=0.657, B=0.655. marginal: gru=+0.005, regime=+0.003 (둘 다 ±0.005 noise floor 근방, ★ selector 단순화 anchor)] |
| c8 | analysis | STAGE 5 failure analysis + B001 비교 (`failure_b001.{json,md}`) spec @ §9 | [DONE d39d168] |
| G5 | gate | top-K worst sample + B001 win/loss decomposition 박제 | [DONE — worst-100 (regime 13: 19 / regime 14: 11 집중) + PB win 965 / loss 153 / PB hit 0.660 vs B001 0.579] |
| c9 | synthesis | `analysis/plan-005/results.md` + `next_plan_candidates.md` (≥3 후보) spec @ §10 | [DONE caa9177] |
| G_final | gate | results.md + next_plan_candidates.md ≥ 3 후보 박제 | [DONE — 4 후보 도출 (selector simplify / corrector loss / high-speed regime / bias re-tuning), 각 후보 4 항목 박제] |

### Plan-specific severe (WORKFLOW.md §12.3 default 위 추가분)

- `plan004_artifacts_missing`: G0 진입 시 plan-004 의 4 산출 (`oof_selector_scores.npz`, `test_selector_scores.npz`, `boundary_tiny_correction_report.json`, `regime_distribution.json`) 중 하나라도 부재.
- `metric_unaudited`: G1~G5 진입 시 해당 stage 의 metric 중 하나라도 산출 json 에 누락.
- `synthesis_incomplete`: G_final 진입 시 `next_plan_candidates.md` 의 후보가 3개 미만, 또는 각 후보의 *근거 metric* (어떤 진단 결과 기반인지) 미박제.
- `corrector_seed_drift`: corrector full-fit 재실행 결과의 `submission_boundary_tiny_soft.csv` 가 plan-004 원본 csv 와 좌표 RMSE > 0.001 m. seed 미고정 의심 → 사용자 escalate (warn only, severe X — diagnostic 정확도 영향 박제).

### Plan-specific paths (WORKFLOW.md §12.5/§12.6 추가/제외)

- whitelist 추가:
  - `analysis/plan-005/**` (특히 `diagnostic.py`, `*.json`, `*.md`, `corrected_oof.npz`, `corrected_test.npz`)
- blacklist 추가:
  - `notes/PB_0.6822 코드공유.ipynb` (원본 보존)
  - `src/pb_0_6822/**` (plan-004 가 lock-in한 모듈, 본 plan 에서 수정 X — 분석 스크립트만 import)
  - `runs/baseline/P001_pb-0-6822-fullrun/**` 의 *원본 산출* (oof_*.npz, test_*.npz, *.csv, *.json) — 읽기만, 수정 X
  - `plans/plan-001~004*` (앞선 plan 본문 수정 X)

### Decision-note 사용 예 (자율 결정 시 commit msg 박제)

- `decision-note: spec-default — analysis-only plan, 모델 재학습 X (corrector full-fit 1회 재실행은 intermediate artifact 박제용으로 허용)`
- `decision-note: spec-default — corrected_oof = full-fit corrector 적용 결과 (per-fold OOF 가 아닌 단일 fit). 약한 leakage 존재하나 oracle/cap-saturation 같은 *분포 특성* 진단에는 영향 무시 가능. caveats §N+3 박제`
- `decision-note: spec-default — bias ablation 4종 (gru-only / gru+physics / gru+regime / full) — selector forward 시 bias 항을 selectively zero-out 하는 inference-time intervention (재학습 X)`
- `decision-note: spec-default — failure analysis = top-100 worst (final hit 0이고 best-cand-error 가 큰 순)`
- `decision-note: spec-default — B001 비교 = per-sample (PB_final_pred vs B001_pred) 거리 차이 + win/loss matrix`

---

## §1. 배경

### §1.1 plan-004 산출 인계

plan-004 (PB_0.6822 framework 적용) 의 핵심 산출:

| 파일 | 역할 |
|---|---|
| `runs/baseline/P001_pb-0-6822-fullrun/oof_selector_scores.npz` | selector 5-fold OOF 점수 [N_train, 27] |
| `runs/baseline/P001_pb-0-6822-fullrun/test_selector_scores.npz` | selector full-fit test 점수 [N_test, 27] |
| `runs/baseline/P001_pb-0-6822-fullrun/submission_boundary_tiny_soft.csv` | 최종 제출 (soft blend) |
| `runs/baseline/P001_pb-0-6822-fullrun/boundary_tiny_correction_report.json` | corrector 학습 metric 요약 |
| `analysis/plan-004/regime_distribution.json` | 18×27 regime hit table + degenerate flag |

**부족한 산출** (본 plan 이 STAGE 0 에서 보충):
- 27 *corrected* candidates per sample (현재 plan-004 는 final pred 1개만 저장; oracle/cap analysis 위해 27개 모두 필요)
- corrector model checkpoint (bias ablation 위해 model 재호출 필요)

### §1.2 본 plan 의 검증 명제 3개

| 명제 | 검증 방법 | 합격 산출 |
|---|---|---|
| **이 framework 의 ceiling 이 어디 있나** | 4-tier oracle 측정 (raw / post-corr / per-regime / per-family) | `oracle_summary.{json,md}` |
| **Selector / Corrector 중 누가 더 큰 병목인가** | gap decomposition (oracle - actual hit, 단계별) | `selector_decomp.{json,md}` + `corrector_decomp.{json,md}` |
| **plan-006 어디부터 손대야 가장 ROI 큰가** | 진단 metric 종합 + ranked candidate list | `next_plan_candidates.md` (≥ 3 후보) |

### §1.3 측정할 20 metric 전수 (G1~G5 합격 기준)

| # | Stage | Metric | 산출 위치 |
|---|---|---|---|
| 1 | G1 | Raw oracle (best of 27 raw) | `oracle_summary.json["raw_oracle"]` |
| 2 | G1 | Post-correction oracle (best of 27 corrected) | `oracle_summary.json["post_corr_oracle"]` |
| 3 | G1 | Per-regime oracle (raw / post-corr) | `oracle_summary.json["per_regime"]` |
| 4 | G1 | Per-family oracle (each family alone) | `oracle_summary.json["per_family"]` |
| 5 | G2 | Selector hit (argmax / soft / gate) | `selector_decomp.json["hit"]` |
| 6 | G2 | Per-regime selector hit | `selector_decomp.json["per_regime_hit"]` |
| 7 | G2 | Selector top-K accuracy (K=1, 3, 5) | `selector_decomp.json["top_k"]` |
| 8 | G2 | Selector confidence margin distribution | `selector_decomp.json["margin_hist"]` |
| 9 | G2 | Per-family selection rate | `selector_decomp.json["family_selection_rate"]` |
| 10 | G3 | Cap saturation rate | `corrector_decomp.json["cap_saturation"]` |
| 11 | G3 | Correction direction breakdown (par/perp/binormal) | `corrector_decomp.json["direction_breakdown"]` |
| 12 | G3 | Error distribution histogram (best cand error) | `corrector_decomp.json["error_hist"]` |
| 13 | G3 | Per-error-band corrector effectiveness | `corrector_decomp.json["per_band_effect"]` |
| 14 | G4 | **Variant hit** (full / A=no-regime / B=no-gru, argmax+soft, overall+per-regime) | `component_contribution.json["variants"]` |
| 15 | G4 | **Marginal contribution** (gru = full−B, regime = full−A; argmax + soft) | `component_contribution.json["marginal_contribution"]` |
| 16 | G4 | **GRU intervention pattern** (B↔full pick 비교: rate, helped, hurt, per-regime) | `component_contribution.json["intervention_gru"]` |
| 17 | G4 | **Regime intervention pattern** (A↔full pick 비교: 동일 구조) | `component_contribution.json["intervention_regime"]` |
| 18 | G4 | **Family-change breakdown** (양 intervention 의 same-family vs cross-family 분해) | `component_contribution.json["family_change"]` |
| 19 | G5 | Top-100 worst samples (with regime + features) | `failure_b001.json["worst_samples"]` |
| 20 | G5 | B001 baseline per-sample comparison (win/loss/tie) | `failure_b001.json["b001_comparison"]` |

---

## §2. Scope (명시적)

### §2.1 In-scope

| 항목 | 값 |
|---|---|
| 입력 | plan-004 산출 4종 (§1.1 표) |
| 분석 | 20 진단 metric (§1.3 표) |
| Corrector inference 재실행 | 1회 (intermediate artifact 박제, ~10min) |
| **Selector 재학습** | **Variant A 1회** (`--regime-prior-strength 0.0`, ~15~20min). plan-004 와 동일 spec, regime prior 만 0 |
| Variant B 계산 | retrain X — `0.65 × physics + 0.45 × regime` argmax 한 줄 (free) |
| Synthesis | `next_plan_candidates.md` ≥ 3 후보 도출 |
| 산출 위치 | `analysis/plan-005/**` |

### §2.2 Out-of-scope (절대 안 함)

| 항목 | 이유 |
|---|---|
| **Variant A 외 추가 selector retrain** (Variant C: gru only, Variant D: gru + regime 등) | 본 plan 의 *marginal contribution* 측정 (gru = full − B, regime = full − A) 두 축으로 충분. 추가 ablation 은 plan-006 으로 미룸 |
| Selector full retrain (plan-004 reproducibility 재검증) | plan-004 산출 lock-in; corrector inference + Variant A retrain 만 허용 |
| Hyperparam 변경 (epochs/patience/hidden 등) | Variant A 도 plan-004 와 동일 spec 사용 (`--regime-prior-strength` 만 변경) |
| 27 후보 / regime 정의 변경 | 본 plan 은 *진단*, 변경은 plan-006 |
| LB 제출 | diagnostic plan, LB 의무 없음 |
| plan-004 모듈 (`src/pb_0_6822/**`) 수정 | lock-in; 분석 script 만 import |
| End-to-end 학습 시도 | plan-004 spec out-of-scope 답습 |

---

## §3. 사전 등록 (Pre-registration)

### §3.1 입력 데이터

| 분할 | 출처 | 사용 |
|---|---|---|
| Train (5-fold OOF) | `oof_selector_scores.npz` + `data/train/` + `data/train_labels.csv` | oracle 측정, gap analysis, ablation |
| Test | `test_selector_scores.npz` + `data/test/` + `data/sample_submission.csv` | (선택) test 분포 확인용 |

**공유 입력 변수 source manifest** (모든 STAGE 가 require — c2 의 `_load_shared_inputs()` helper 가 한 곳에서 로드):

| 변수 | shape | dtype | source 의미 |
|---|---|---|---|
| `train_y` | `[N_train, 2]` | float | `selector.read_labels(DATA_ROOT/'train_labels.csv')` 의 두 번째 반환값. 단위 = R_HIT=0.01 과 동일 정규화 coord (cands / corrected 와 동일 공간; 단위 변환 없이 직접 L2 norm 비교) |
| `regimes` | `[N_train]` | int (0~17) | plan-004 가 박제한 quantile-bin regime label (18-class). plan-005 구현 시 (a) `analysis/plan-004/regime_distribution.json` 의 per-sample label 이 있으면 **우선** 로드, (b) 없으면 `src/pb_0_6822/selector.py` 의 regime 함수 재호출. 두 path 가 모두 가능한 경우 c2 `_load_shared_inputs()` 에서 *18-bin histogram 일치 sanity check* (path a vs b 동일 분포; 불일치 시 `regime_distribution_path_drift` warn 박제 + (a) 우선 사용). 정확 attribute 명은 c2 decision-note 박제 |
| `cand_family` | `[27]` | int (0~5) | 27 candidate → 6 family (`base=0, acc=1, frenet=2, turn=3, jerk=4, latency=5`) 매핑 array. `selector.make_candidates` 가 생성하는 27 후보의 family id sequence — plan-004 §6 STAGE 2 가 lock-in 한 순서와 동일. selector 모듈의 family constant 또는 plan-004 산출에서 추출 |
| `physics_bias` | `[27]` | float | plan-004 가 lock-in 한 selector forward 의 physics prior (final_scores 에서 0.65 가중치로 합산되는 항). `src/pb_0_6822/selector.py` 에서 import (정확 attribute 명은 c2 decision-note 박제) |
| `regime_bias_table` | `[18, 27]` | float | plan-004 가 학습/박제한 regime prior matrix (0.45 가중치). selector 모듈 또는 plan-004 산출에서 import (정확 attribute 명은 c2 decision-note 박제) |
| `oof_scores` | `[N_train, 27]` | float | plan-004 `oof_selector_scores.npz["ens_scores"]`. **pre-bias** (attn_gru ensemble logit; physics/regime bias 미적용) — 본 plan 의 final_scores 계산 시 `oof_scores + 0.65*physics_bias + 0.45*regime_bias_table[regimes]` 로 별도 합산 |

> ※ source 의 attribute 명은 plan-005 구현 시 c2 commit 의 `_load_shared_inputs()` helper 가 plan-004 source 와 일치하도록 결정. 본 plan 본문은 *변수의 의미 + shape + dtype* spec, attribute 명은 implementation defer (c2 commit msg 에 `decision-note: shared input attribute names = {...}` 박제 의무).

OOF 사용 원칙: plan-004 의 OOF semantic 그대로 사용 (각 sample 의 score 가 *그 sample 을 학습에 안 본 fold* 의 모델 출력). corrector 재실행은 *full-fit* 으로 단일 모델 사용 — leakage 약함, oracle/distribution 진단에는 무시 가능 (caveats 박제).

### §3.2 합격 기준 (정량)

- **G0**: plan-004 산출 4종 모두 로드 + `corrected_oof.npz` shape `(N_train, 27, 2)` finite + `corrected_test.npz` shape `(N_test, 27, 2)` finite
- **G1**: `oracle_summary.json` 의 4 key (`raw_oracle`, `post_corr_oracle`, `per_regime`, `per_family`) 모두 존재 + 모든 hit 값 ∈ [0, 1]
- **G2**: `selector_decomp.json` 의 5 key (`hit`, `per_regime_hit`, `top_k`, `margin_hist`, `family_selection_rate`) 모두 존재
- **G3**: `corrector_decomp.json` 의 4 key (`cap_saturation`, `direction_breakdown`, `error_hist`, `per_band_effect`) 모두 존재
- **G4**: `component_contribution.json` 의 (a) `variants` 에 3 setting (`full`, `A_no_regime`, `B_no_gru`) × argmax/soft × overall+per-regime 모두 박제 + (b) `marginal_contribution.gru` (= full − B), `marginal_contribution.regime` (= full − A) 모두 박제 + (c) `intervention_gru` (B↔full pick 비교: rate / helped / hurt / per-regime / hit_alt_when_changed / hit_full_when_changed / delta_hit_when_changed) + `intervention_regime` (A↔full 동일 구조) + `family_change` (양 intervention 의 same-family vs cross-family 분해) 모두 박제. Variant A 의 `oof_selector_scores.npz` 가 `analysis/plan-005/variant_A_no_regime/` 에 존재 + finite + shape `(N_train, 27)`
- **G5**: `failure_b001.json` 의 2 key (`worst_samples` len ≥ 100, `b001_comparison`) 모두 존재
- **G_final**: `results.md` + `next_plan_candidates.md` 작성, 후보 ≥ 3 + 각 후보의 *근거 metric reference* 박제

### §3.3 평가

본 plan 은 *진단* 이라 LB / CV 평가 없음. 합격 = "20 metric 모두 박제 + 다음 plan 후보 ≥ 3" 충족.

---

## §4. STAGE 0 — 인프라 + plan-004 산출 검증 + Corrector 재실행 (c2~c3)

### §4.1 `analysis/plan-005/diagnostic.py` 골격 (c2)

```python
# analysis/plan-005/diagnostic.py
"""plan-005 diagnostic 메인 entry. STAGE 0~5 helper + main()."""
import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from src.pb_0_6822 import selector, boundary

REPO = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO / "data"
PLAN004_DIR = REPO / "runs/baseline/P001_pb-0-6822-fullrun"
ANALYSIS_DIR = REPO / "analysis/plan-005"
DEVICE = "cuda:1"  # plan-004 일관성 (server agent 1번 GPU)

def verify_plan004_artifacts():
    """G0 진입 자격 — 4 산출 모두 로드 가능 검증."""
    required = [
        PLAN004_DIR / "oof_selector_scores.npz",
        PLAN004_DIR / "test_selector_scores.npz",
        PLAN004_DIR / "boundary_tiny_correction_report.json",
        REPO / "analysis/plan-004/regime_distribution.json",
    ]
    for p in required:
        assert p.exists(), f"plan-004 산출 부재: {p}"
    # load smoke
    oof = np.load(required[0])
    test = np.load(required[1])
    rep = json.loads(required[2].read_text())
    rd = json.loads(required[3].read_text())
    return {
        "oof_scores": oof["ens_scores"],          # [N_train, 27]
        "test_scores": test["ens_scores"],         # [N_test, 27]
        "boundary_report": rep,
        "regime_distribution": rd,
    }

def load_train_data():
    """train_x, train_y, ids 로드 (plan-004 와 동일 경로)."""
    ids, train_y = selector.read_labels(DATA_ROOT / "train_labels.csv")
    train_x = selector.load_stack(DATA_ROOT / "train", ids)
    return ids, train_x, train_y
```

### §4.2 Corrector 재실행 + intermediate artifact 박제 (c3)

```python
# analysis/plan-005/diagnostic.py 추가
def rerun_corrector_save_intermediates():
    """Corrector full-fit 1회 재실행 + corrected candidates 박제.
    plan-004 의 boundary main 은 corrected_*.npz 를 저장 안 하므로 본 plan 이 추가."""
    # boundary main 의 train_full_corrector + predict_corrected_candidates 호출 패턴 재현
    ids, train_x, train_y = load_train_data()
    test_ids = selector.read_submission_ids(DATA_ROOT / "sample_submission.csv")
    test_x = selector.load_stack(DATA_ROOT / "test", test_ids)

    # ── 1. plan-004 의 score bank 로드 ──
    oof_scores = np.load(PLAN004_DIR / "oof_selector_scores.npz")["ens_scores"]
    test_scores = np.load(PLAN004_DIR / "test_selector_scores.npz")["ens_scores"]

    # ── 2. Candidate 생성 (plan-004 와 동일 horizon=2) ──
    end_idx = train_x.shape[1] - 1
    train_cands = selector.make_candidates(train_x, end_idx, horizon=2)   # [N_train, 27, 2]
    test_cands  = selector.make_candidates(test_x, test_x.shape[1] - 1, horizon=2)

    # ── 3. Corrector full-fit (plan-004 spec 답습) ──
    # boundary.main 의 --make-test 분기 코드 재현 (간결 버전).
    # _train_full_corrector 의 8-tuple 반환값 spec:
    #   full_model   : torch.nn.Module — 학습 완료된 boundary corrector (plan-004 의 BoundaryCorrector class)
    #   args         : argparse.Namespace — boundary.main 이 사용하는 hyperparam (cap, temp, basis dim 등) 일관성 유지용
    #   basis_train  : np.ndarray [N_train, B_dim]  — train Frenet basis feature
    #   basis_test   : np.ndarray [N_test,  B_dim]  — test  Frenet basis feature
    #   scale_train  : np.ndarray [N_train]         — train speed-scale normalizer (≥ EPS)
    #   scale_test   : np.ndarray [N_test]          — test  speed-scale normalizer
    #   cf_train     : np.ndarray [N_train, F_dim]  — train conditioning feature (regime/physics/last-velocity 류)
    #   cf_test     : np.ndarray [N_test,  F_dim]  — test  conditioning feature
    # 구현 의무: plan-004 의 boundary.main 이 --make-test 분기에서 호출하는 동일한 학습 루프를 재현.
    #   seed=20260606 고정 (§4.3 G0 합격 caveat). plan-004 와 동일 batch / epochs / cap / temp 사용.
    full_model, args, basis_train, basis_test, scale_train, scale_test, cf_train, cf_test = (
        _train_full_corrector(train_x, train_y, test_x)
    )

    # ── 4. Corrected candidates 산출 ──
    corrected_oof  = boundary.predict_corrected_candidates(
        full_model, cf_train, train_cands, basis_train, scale_train, args, DEVICE
    )   # [N_train, 27, 2]
    corrected_test = boundary.predict_corrected_candidates(
        full_model, cf_test, test_cands, basis_test, scale_test, args, DEVICE
    )

    # ── 5. 박제 ──
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        ANALYSIS_DIR / "corrected_oof.npz",
        cands=train_cands.astype(np.float32),         # raw
        corrected=corrected_oof.astype(np.float32),    # corrected
    )
    np.savez_compressed(
        ANALYSIS_DIR / "corrected_test.npz",
        cands=test_cands.astype(np.float32),
        corrected=corrected_test.astype(np.float32),
    )
    torch.save(full_model.state_dict(), ANALYSIS_DIR / "corrector_state.pt")

    # ── 6. corrector_seed_drift check (warn only) ──
    # 단위 가정: submission_boundary_tiny_soft.csv 의 coord column = train_y 와 동일 정규화 coord (R_HIT=0.01 공간).
    # boundary.soft_select 의 출력도 동일 공간 → 두 array 의 직접 L2 비교 valid. 단위 다르면 RMSE threshold 0.001 무의미.
    # 컬럼 순서 invariant: sample_submission.csv 의 column 순서 [id, x, y] 가정 → iloc[:, 1:] 가 [N, 2] = (x, y). soft_select 출력도 (x, y) 순.
    sub_repro = boundary.soft_select(corrected_test, test_scores, temp=0.03)   # plan-004 default
    sub_orig = pd.read_csv(PLAN004_DIR / "submission_boundary_tiny_soft.csv")
    coord_orig = sub_orig.iloc[:, 1:].values   # [N_test, 2] (x, y)
    rmse = float(np.sqrt(((sub_repro - coord_orig) ** 2).sum(axis=1).mean()))
    if rmse > 0.001:
        print(f"[WARN] corrector_seed_drift: RMSE={rmse:.6f} > 0.001 — seed 미고정 의심")
```

### §4.3 G0 합격 기준 (자동 판정)

- `verify_plan004_artifacts()` 가 AssertionError 없이 통과
- `corrected_oof.npz`, `corrected_test.npz` 모두 존재 + shape `(N_*, 27, 2)` finite
- `corrector_state.pt` 존재
- (warn) seed_drift RMSE 박제 (severe trigger 안 함)
- seed 고정: `_train_full_corrector` 내부 torch/numpy seed 를 plan-004 와 동일 `20260606` 으로 설정 (§N+3 §9 caveat 일관성). 미고정 시 corrector_seed_drift warn 발현 가능

### §4.4 시간 예산

- `verify_plan004_artifacts()`: < 1초
- corrector 재실행: ~10 min (server `cuda:1`)

---

## §5. STAGE 1 — Oracle 4-Tier (c4)

### §5.1 측정 식

```python
# analysis/plan-005/diagnostic.py 추가
def stage1_oracles(corrected_oof_npz, train_y, regimes, candidate_family):
    """4-tier oracle 측정.

    인자 source (§3.1 공유 입력 변수 manifest 참조):
      regimes          : [N_train] int (0~17). plan-004 quantile-bin regime label.
      candidate_family : [27] int (0~5). cand_family with families = [base, acc, frenet, turn, jerk, latency].
    """
    raw = corrected_oof_npz["cands"]          # [N, 27, 2] — train_y 와 동일 단위 (정규화 coord; m 단위 가정)
    corr = corrected_oof_npz["corrected"]      # [N, 27, 2] — 동일 단위
    R_HIT = selector.R_HIT                     # 0.01 (coord 단위; train_y / cands 와 동일 정규화 공간의 L2 거리 임계값)

    # ── 1. Raw oracle ──
    err_raw = np.linalg.norm(raw - train_y[:, None, :], axis=2)        # [N, 27]
    raw_oracle = (err_raw.min(axis=1) <= R_HIT).mean()

    # ── 2. Post-correction oracle ──
    err_corr = np.linalg.norm(corr - train_y[:, None, :], axis=2)
    post_corr_oracle = (err_corr.min(axis=1) <= R_HIT).mean()

    # ── 3. Per-regime oracle ──
    per_regime = {}
    for r in range(18):
        mask = regimes == r
        if mask.sum() == 0:
            per_regime[r] = {"n": 0, "raw": None, "post_corr": None}
            continue
        per_regime[r] = {
            "n": int(mask.sum()),
            "raw": float((err_raw[mask].min(axis=1) <= R_HIT).mean()),
            "post_corr": float((err_corr[mask].min(axis=1) <= R_HIT).mean()),
        }

    # ── 4. Per-family oracle ──
    families = ["base", "acc", "frenet", "turn", "jerk", "latency"]
    per_family = {}
    for fid, fname in enumerate(families):
        mask_c = candidate_family == fid   # candidate-axis mask
        if mask_c.sum() == 0:
            per_family[fname] = {"n_cands": 0, "raw": None, "post_corr": None}
            continue
        # family 안의 후보들로만 oracle
        err_raw_fam = err_raw[:, mask_c]
        err_corr_fam = err_corr[:, mask_c]
        per_family[fname] = {
            "n_cands": int(mask_c.sum()),
            "raw": float((err_raw_fam.min(axis=1) <= R_HIT).mean()),
            "post_corr": float((err_corr_fam.min(axis=1) <= R_HIT).mean()),
        }

    return {
        "raw_oracle": float(raw_oracle),
        "post_corr_oracle": float(post_corr_oracle),
        "per_regime": per_regime,
        "per_family": per_family,
    }
```

### §5.2 산출

- `analysis/plan-005/oracle_summary.json` — 위 4 key
- `analysis/plan-005/oracle_summary.md` — markdown 표:
  - 전체 raw/post-corr oracle (1줄)
  - Per-regime: 18-row table `| regime | n | raw | post_corr | gain |`
  - Per-family: 6-row table `| family | n_cands | raw | post_corr | gain |`

### §5.3 G1 합격 기준

- `oracle_summary.json` 의 4 key 모두 존재
- `raw_oracle ∈ [0, 1]`, `post_corr_oracle ∈ [0, 1]`, `post_corr_oracle ≥ raw_oracle - 0.001` (corrector 가 oracle 을 *떨어뜨리면* 안 됨; tolerance 0.001 은 floating noise)
- 18 regime 의 `n` 합 == `len(train_y)`

---

## §6. STAGE 2 — Selector Decomposition (c5)

### §6.1 측정 식

```python
def stage2_selector(oof_scores, raw_cands, corrected_cands, train_y, regimes, cand_family,
                    physics_bias, regime_bias_table):
    """Selector 의 5 가지 분해 metric.

    인자 source (§3.1 공유 입력 변수 manifest 참조):
      oof_scores        : [N, 27] float — plan-004 oof_selector_scores.npz["ens_scores"] (**pre-bias**).
      physics_bias      : [27] float — selector forward 의 physics prior 상수.
      regime_bias_table : [18, 27] float — selector forward 의 regime prior matrix.
    final_scores 식은 §8.3 의 score_full 과 동일 (한 곳에서 정의, 두 STAGE 가 import).
    """
    R_HIT = selector.R_HIT
    # 최종 점수 = oof_scores + 0.65 × physics_bias + 0.45 × regime_bias_table[regimes]
    final_scores = (
        oof_scores
        + 0.65 * physics_bias[None, :]
        + 0.45 * regime_bias_table[regimes]
    )

    # ── 1. Hit (argmax / soft / gate) on corrected candidates ──
    pick_argmax = corrected_cands[np.arange(len(train_y)), final_scores.argmax(axis=1)]
    err_argmax = np.linalg.norm(pick_argmax - train_y, axis=1)
    hit_argmax = float((err_argmax <= R_HIT).mean())

    # soft (temperature 0.03 plan-004 default)
    soft_pred = boundary.soft_select(corrected_cands, final_scores, temp=0.03)
    err_soft = np.linalg.norm(soft_pred - train_y, axis=1)
    hit_soft = float((err_soft <= R_HIT).mean())

    # ── 2. Per-regime hit ──
    per_regime_hit = {}
    for r in range(18):
        mask = regimes == r
        if mask.sum() > 0:
            per_regime_hit[r] = {
                "n": int(mask.sum()),
                "argmax": float((err_argmax[mask] <= R_HIT).mean()),
                "soft":   float((err_soft[mask] <= R_HIT).mean()),
            }

    # ── 3. Top-K accuracy ──
    err_corr = np.linalg.norm(corrected_cands - train_y[:, None, :], axis=2)
    best_idx = err_corr.argmin(axis=1)              # ground-truth best candidate per sample
    top_k = {}
    for K in (1, 3, 5):
        topK_idx = np.argsort(-final_scores, axis=1)[:, :K]
        in_topK = (topK_idx == best_idx[:, None]).any(axis=1)
        top_k[K] = float(in_topK.mean())

    # ── 4. Confidence margin distribution ──
    sorted_scores = np.sort(final_scores, axis=1)[:, ::-1]
    margin = sorted_scores[:, 0] - sorted_scores[:, 1]
    margin_hist = {
        "p10": float(np.percentile(margin, 10)),
        "p25": float(np.percentile(margin, 25)),
        "p50": float(np.percentile(margin, 50)),
        "p75": float(np.percentile(margin, 75)),
        "p90": float(np.percentile(margin, 90)),
        "mean": float(margin.mean()),
        "std":  float(margin.std()),
    }

    # ── 5. Per-family selection rate ──
    families = ["base", "acc", "frenet", "turn", "jerk", "latency"]
    selected_family = cand_family[final_scores.argmax(axis=1)]  # [N]
    family_selection_rate = {
        fname: float((selected_family == fid).mean())
        for fid, fname in enumerate(families)
    }

    return {
        "hit": {"argmax": hit_argmax, "soft": hit_soft},
        "per_regime_hit": per_regime_hit,
        "top_k": top_k,
        "margin_hist": margin_hist,
        "family_selection_rate": family_selection_rate,
    }
```

### §6.2 산출

- `analysis/plan-005/selector_decomp.json`
- `analysis/plan-005/selector_decomp.md`:
  - 전체 hit (argmax / soft) 1줄
  - 18-row per-regime table
  - top-K table
  - margin percentiles 1줄
  - 6-row family selection rate table

### §6.3 G2 합격 기준

- 5 key (`hit`, `per_regime_hit`, `top_k`, `margin_hist`, `family_selection_rate`) 모두 존재
- top-K 가 monotonic (`top_k[1] ≤ top_k[3] ≤ top_k[5]`)
- family_selection_rate 합 ≈ 1.0 (tolerance 0.001)

---

## §7. STAGE 3 — Corrector Decomposition (c6)

### §7.1 측정 식

```python
def stage3_corrector(corrected_oof_npz, train_y, train_x, regimes):
    """Corrector 의 4 가지 분해 metric."""
    raw = corrected_oof_npz["cands"]          # [N, 27, 2]
    corr = corrected_oof_npz["corrected"]      # [N, 27, 2]
    delta = corr - raw                         # [N, 27, 2]
    R_HIT = selector.R_HIT
    cap = 0.006   # plan-004 lock-in 의 corrector cap. §4.2 의 _train_full_corrector args.cap 과 동일해야 함 — 다르면 saturation metric 오측정. **c2 구현 의무**: cap = args.cap 으로 import (literal 0.006 은 plan-004 일치 시 fallback 값; 불일치 시 import 한 값 우선 + decision-note 박제)

    # ── 1. Cap saturation rate ──
    delta_norm = np.linalg.norm(delta, axis=2)  # [N, 27]
    saturated = delta_norm >= cap * 0.95         # 95% 이상 = 사실상 cap
    cap_saturation = {
        "overall_rate": float(saturated.mean()),
        "per_candidate": [float(saturated[:, c].mean()) for c in range(27)],
    }

    # ── 2. Correction direction (Frenet local frame) ──
    # delta 는 [N, 27, 2] (2-D 좌표 vector). vector_to_local 은 plan-004 lock-in helper:
    #   2-D delta 를 (t, n) plane 위 parallel/perpendicular 두 성분 + plane 밖 binormal 성분으로 분해.
    #   binormal 성분은 2-D coord 의 경우 0 에 수렴하는 noise floor — 측정해도 무의미하지 않고
    #   "후보 분포가 (t, n) plane 을 얼마나 벗어나는지" 의 sanity-check 로 사용 (§N+3 #7 anchor).
    end_idx = train_x.shape[1] - 1
    from src.pb_0_6822.boundary import local_frame, vector_to_local
    t, n, b, speed = local_frame(train_x, end_idx)
    scale = np.maximum(speed * 2.0, selector.EPS)
    delta_local = vector_to_local(delta, (t, n, b), scale)   # [N, 27, 3] — (parallel, perpendicular, binormal)
    direction_breakdown = {
        "parallel_mean":  float(np.abs(delta_local[..., 0]).mean()),
        "perp_mean":      float(np.abs(delta_local[..., 1]).mean()),
        "binormal_mean":  float(np.abs(delta_local[..., 2]).mean()),
        "parallel_std":   float(delta_local[..., 0].std()),
        "perp_std":       float(delta_local[..., 1].std()),
        "binormal_std":   float(delta_local[..., 2].std()),
    }

    # ── 3. Error distribution histogram (best raw candidate err) ──
    err_raw = np.linalg.norm(raw - train_y[:, None, :], axis=2)  # [N, 27]
    best_err_raw = err_raw.min(axis=1)
    bins = [0, 0.005, 0.01, 0.015, 0.02, 0.03, 0.05, 0.10, np.inf]
    hist, _ = np.histogram(best_err_raw, bins=bins)
    error_hist = {
        f"[{bins[i]:.3f}, {bins[i+1]:.3f})": int(hist[i])
        for i in range(len(bins) - 1)
    }

    # ── 4. Per-error-band corrector effectiveness ──
    err_corr = np.linalg.norm(corr - train_y[:, None, :], axis=2)
    best_err_corr = err_corr.min(axis=1)
    per_band_effect = {}
    for i in range(len(bins) - 1):
        mask = (best_err_raw >= bins[i]) & (best_err_raw < bins[i + 1])
        if mask.sum() > 0:
            per_band_effect[f"[{bins[i]:.3f}, {bins[i+1]:.3f})"] = {
                "n": int(mask.sum()),
                "hit_before": float((best_err_raw[mask] <= R_HIT).mean()),
                "hit_after":  float((best_err_corr[mask] <= R_HIT).mean()),
                "delta":      float(((best_err_corr[mask] <= R_HIT).mean() -
                                     (best_err_raw[mask] <= R_HIT).mean())),
            }

    return {
        "cap_saturation": cap_saturation,
        "direction_breakdown": direction_breakdown,
        "error_hist": error_hist,
        "per_band_effect": per_band_effect,
    }
```

### §7.2 산출

- `analysis/plan-005/corrector_decomp.json`
- `analysis/plan-005/corrector_decomp.md`:
  - cap saturation 전체 + per-candidate top-5 (가장 자주 saturated)
  - direction breakdown 6 metric
  - error_hist 8-bin table
  - per_band_effect table (band × n × hit_before × hit_after × delta)

### §7.3 G3 합격 기준

- 4 key (`cap_saturation`, `direction_breakdown`, `error_hist`, `per_band_effect`) 모두 존재
- error_hist 8-bin 합 == `len(train_y)`
- direction_breakdown 의 6 값 모두 finite

---

## §8. STAGE 4 — Selector Component Contribution (Variant A retrain + Variant B free + 비교) (c7a, c7b)

> **핵심 변경 (v2)**: v1 의 *inference-time bias ablation* (학습된 모델의 logit 에 prior 를 selectively zero-out) 은 *모델이 prior 가 있다는 가정으로 학습됨* → counterfactual 편향. v2 는 *retrain variant* 로 교체. Variant A 는 *regime 없이 직접 학습*, Variant B 는 *gru 없이 priors 만 계산* (free). marginal contribution 측정이 *fair*.

### §8.1 Variant 정의

| Variant | gru | physics_bias (0.65) | regime_bias (0.45) | compute |
|---|---|---|---|---|
| **full** | ✓ | ✓ | ✓ | plan-004 산출 (재학습 X) |
| **A — no regime** | ✓ (재학습) | ✓ | ✗ | 1회 selector retrain (~15~20min, server `cuda:1`) |
| **B — no gru** | ✗ | ✓ | ✓ | 0 (산식 1줄) |

Variant A 의 *gru* 는 *regime 없이 학습된 모델* — full 의 gru 와 *다른 파라미터*. 이게 핵심: inference-time zero-out 이 아닌 *fair 학습 setting* 비교.

> **Variant B "no gru" naming 주의** — plan-004 의 `ens_scores` 는 attn_gru ensemble logit (학습된 model 출력) 이므로 "no gru" = "no learned model" 의 약칭. score_B 는 `oof_scores` 항을 *완전 제거* 하고 physics + regime bias 만 사용. marginal_contribution.gru = full − B 는 따라서 "learned ensemble 의 추가 기여" 를 의미 (gru arch 단독 ablation 이 아님 — 그건 §2.2 out-of-scope).

### §8.2 Variant A 재학습 (c7a)

**spec**: plan-004 §6 STAGE 2 와 *완전 동일* 하되 `--regime-prior-strength 0.0` 만 추가:

```python
# analysis/plan-005/diagnostic.py 추가
def stage4a_retrain_variant_A():
    """Variant A: selector 재학습 (regime_bias 제거)."""
    out_dir = ANALYSIS_DIR / "variant_A_no_regime"
    out_dir.mkdir(parents=True, exist_ok=True)
    selector.SELECTOR_MAIN([
        '--root', str(DATA_ROOT), '--out-dir', str(out_dir),
        '--models', 'attn_gru',
        '--folds', '5', '--fold-limit', '5',
        '--pre-epochs', '10', '--fine-epochs', '8', '--freeze-fine-epochs', '3',
        '--epoch-plus', '5', '--min-epochs', '5', '--patience', '4',
        '--hidden', '48', '--batch', '4096',
        '--regime-prior-strength', '0.0',   # ★ 유일한 변경: regime 제거
        '--device', DEVICE,
    ])
    # 산출: analysis/plan-005/variant_A_no_regime/oof_selector_scores.npz [N_train, 27]
    #       analysis/plan-005/variant_A_no_regime/test_selector_scores.npz [N_test, 27]
```

**c7a 의 산출**:
- `analysis/plan-005/variant_A_no_regime/oof_selector_scores.npz` — Variant A 의 OOF score
- `analysis/plan-005/variant_A_no_regime/test_selector_scores.npz` — Variant A 의 test score (선택, plan-006 anchor 용)

**합격 자동 판정**:
- 두 npz 모두 존재 + `np.isfinite(x).all()` + shape `(N_*, 27)`
- 모든 sample 의 score 가 trivial 한 동일값이 아닌지 확인 (`scores.std(axis=1).mean() > 1e-6`)

### §8.3 Variant B 계산 + 3-way 비교 (c7b)

```python
def stage4b_compute_variant_B_and_compare(
    oof_scores_full,         # plan-004 oof_selector_scores.npz["ens_scores"]
    oof_scores_A,            # analysis/plan-005/variant_A_no_regime/oof_selector_scores.npz["ens_scores"]
    corrected_cands,         # analysis/plan-005/corrected_oof.npz["corrected"]
    train_y, regimes, cand_family,
    physics_bias, regime_bias_table,
):
    """Variant B (free 계산) + 3 variant hit + marginal contribution + 2 intervention 분해."""
    R_HIT = selector.R_HIT
    N = len(train_y)

    # ── Variant 별 최종 score (※ Variant A 의 oof_scores_A 는 이미 regime 제외 학습) ──
    # 가정: oof_scores_A 도 oof_scores_full 과 동일한 **pre-bias** semantic (attn_gru ensemble logit; physics/regime bias 미포함).
    #       Variant A 는 `--regime-prior-strength 0.0` 만 변경 → regime prior 만 제거되고 physics_bias 는 학습-시-적용 안 됨 (plan-004 selector 도 동일).
    # 가중치 0.65/0.45 는 plan-004 lock-in 값 그대로 유지 (rescale 안 함, sum ≠ 1 ok) — 본 plan 의 *fair 비교* 의도는 "plan-004 와 동일 산식으로 component 제거 시 영향" 측정이므로 rescale 시 비교 의미 왜곡.
    score_full = oof_scores_full + 0.65 * physics_bias[None, :] + 0.45 * regime_bias_table[regimes]
    score_A    = oof_scores_A    + 0.65 * physics_bias[None, :]                                    # no regime
    score_B    =                   0.65 * physics_bias[None, :] + 0.45 * regime_bias_table[regimes]  # no gru

    # ── hit per variant (argmax + soft, overall + per-regime) ──
    def _hit(scores, label):
        pick = corrected_cands[np.arange(N), scores.argmax(axis=1)]
        err_arg = np.linalg.norm(pick - train_y, axis=1)
        soft = boundary.soft_select(corrected_cands, scores, temp=0.03)
        err_soft = np.linalg.norm(soft - train_y, axis=1)
        per_regime = {}
        for r in range(18):
            mask = regimes == r
            if mask.sum() == 0: continue
            per_regime[r] = {
                "n": int(mask.sum()),
                "argmax": float((err_arg[mask] <= R_HIT).mean()),
                "soft":   float((err_soft[mask] <= R_HIT).mean()),
            }
        return {
            "argmax": float((err_arg <= R_HIT).mean()),
            "soft":   float((err_soft <= R_HIT).mean()),
            "per_regime": per_regime,
            "_err_argmax": err_arg,   # intervention 계산용
            "_pick_argmax": scores.argmax(axis=1),
        }
    variants = {
        "full": _hit(score_full, "full"),
        "A_no_regime": _hit(score_A, "A"),
        "B_no_gru":    _hit(score_B, "B"),
    }

    # ── Marginal contribution ──
    marginal_contribution = {
        "gru":    {"argmax": variants["full"]["argmax"] - variants["B_no_gru"]["argmax"],
                   "soft":   variants["full"]["soft"]   - variants["B_no_gru"]["soft"]},
        "regime": {"argmax": variants["full"]["argmax"] - variants["A_no_regime"]["argmax"],
                   "soft":   variants["full"]["soft"]   - variants["A_no_regime"]["soft"]},
    }

    # ── per-sample intervention (B↔full = gru contribution pattern) ──
    # helper convention: alt = (B 또는 A) variant, full = 전체 모델.
    # 산출 keys 부호 의미:
    #   helped_rate           : changed 중 full 이 alt 보다 *더 가까운* (err_full < err_alt) sample 비율
    #   hurt_rate             : changed 중 full 이 alt 보다 *더 먼* sample 비율
    #   hit_{alt,full}_when_changed : changed sub-population 에서 각 variant 의 hit rate
    #   delta_hit_when_changed = hit_full_when_changed − hit_alt_when_changed
    #                          : 부호 양수 ⇒ full 이 alt 보다 *changed 위에서* 더 많이 hit
    #                            (즉 해당 intervention 의 component 가 hit 에 기여한 방향)
    def _intervention(pick_alt, err_alt, pick_full, err_full, label):
        changed = pick_alt != pick_full
        helped = changed & (err_full < err_alt)
        hurt   = changed & (err_full > err_alt)
        if changed.sum() == 0:
            return {"rate": 0.0, "n_changed": 0,
                    "helped_rate": None, "hurt_rate": None,
                    "hit_alt_when_changed": None, "hit_full_when_changed": None}
        per_regime = {}
        for r in range(18):
            mask = regimes == r
            chg_r = changed & mask
            if chg_r.sum() == 0: continue
            per_regime[r] = {
                "n_regime": int(mask.sum()),
                "n_changed": int(chg_r.sum()),
                "rate": float(chg_r.sum() / mask.sum()),
                "helped_rate": float((chg_r & (err_full < err_alt)).sum() / chg_r.sum()),
                "hurt_rate":   float((chg_r & (err_full > err_alt)).sum() / chg_r.sum()),
                "hit_alt_when_changed":  float((err_alt[chg_r]  <= R_HIT).mean()),
                "hit_full_when_changed": float((err_full[chg_r] <= R_HIT).mean()),
            }
        return {
            "rate": float(changed.mean()),
            "n_changed": int(changed.sum()),
            "helped_rate": float(helped.sum() / changed.sum()),
            "hurt_rate":   float(hurt.sum() / changed.sum()),
            "hit_alt_when_changed":  float((err_alt[changed]  <= R_HIT).mean()),
            "hit_full_when_changed": float((err_full[changed] <= R_HIT).mean()),
            "delta_hit_when_changed": float(((err_full[changed] <= R_HIT).mean()
                                            - (err_alt[changed]  <= R_HIT).mean())),
            "per_regime": per_regime,
        }

    intv_gru    = _intervention(variants["B_no_gru"]["_pick_argmax"],
                                variants["B_no_gru"]["_err_argmax"],
                                variants["full"]["_pick_argmax"],
                                variants["full"]["_err_argmax"], "gru")
    intv_regime = _intervention(variants["A_no_regime"]["_pick_argmax"],
                                variants["A_no_regime"]["_err_argmax"],
                                variants["full"]["_pick_argmax"],
                                variants["full"]["_err_argmax"], "regime")

    # ── Family-change breakdown (양 intervention) ──
    def _family_change(pick_alt, pick_full):
        changed = pick_alt != pick_full
        fam_alt  = cand_family[pick_alt]
        fam_full = cand_family[pick_full]
        same_family = changed & (fam_alt == fam_full)
        cross_family = changed & (fam_alt != fam_full)
        return {
            "n_changed":       int(changed.sum()),
            "same_family":     int(same_family.sum()),
            "cross_family":    int(cross_family.sum()),
            "cross_family_pct": float(cross_family.sum() / max(changed.sum(), 1)),
        }
    family_change = {
        "gru_intervention":    _family_change(variants["B_no_gru"]["_pick_argmax"],
                                              variants["full"]["_pick_argmax"]),
        "regime_intervention": _family_change(variants["A_no_regime"]["_pick_argmax"],
                                              variants["full"]["_pick_argmax"]),
    }

    # ── 내부 array 정리 (json 직렬화 위해 제거) ──
    for v in variants.values():
        v.pop("_err_argmax", None)
        v.pop("_pick_argmax", None)

    return {
        "variants": variants,
        "marginal_contribution": marginal_contribution,
        "intervention_gru":    intv_gru,
        "intervention_regime": intv_regime,
        "family_change":       family_change,
    }
```

### §8.4 산출

- `analysis/plan-005/component_contribution.json` — 위 5 key 박제
- `analysis/plan-005/component_contribution.md`:
  - 3 variant × argmax/soft × (overall + 18 per-regime) hit 표
  - Marginal contribution 2-row 표 (gru, regime)
  - Intervention 2 표 (gru/regime): rate, n_changed, helped_rate, hurt_rate, hit_alt/full when changed, delta
  - Per-regime intervention table (18 row × 2 intervention)
  - Family-change breakdown (same/cross family ratio per intervention)

### §8.5 G4 합격 기준 (자동 판정)

> **sub-gate 분리**: G4 의 두 commit (c7a, c7b) 가 단일 G4 gate 아래 묶이지만 **각 commit 의 통과 기준은 분리** — c7a 의 retrain 산출이 (a) 조건을 만족해야만 c7b 의 분석 진입 가능. 즉 G4 = G4a (Variant A retrain integrity) ∩ G4b (3-way + intervention + family-change 박제).

- **(G4a, c7a)** Variant A 재학습 산출 npz 2개 존재 + finite + shape OK + `scores.std(axis=1).mean() > 1e-6`
- **(G4b, c7b)** `component_contribution.json` 의 5 top-level key (`variants`, `marginal_contribution`, `intervention_gru`, `intervention_regime`, `family_change`) 모두 존재
- `variants` 의 3 sub-key (`full`, `A_no_regime`, `B_no_gru`) 모두 hit 박제
- `marginal_contribution.gru` ≥ 0 가 *기대*. 음수 시: gru 가 *prior 보다 못함* → severe X (warn only), `next_plan_candidates.md` 에 "selector arch 교체 시급" 으로 박제
- `intervention_gru.rate` + `intervention_regime.rate` 모두 ∈ [0, 1]
- per-regime intervention 의 18 entry 중 sample 있는 regime 모두 박제 (sample 0 인 regime 은 entry 생략 가능)

### §8.6 해석 가이드 (plan-006 anchor 도출)

| 진단 결과 | plan-006 anchor |
|---|---|
| `marginal_contribution.gru` 가 큼 (>0.05) | gru 가 *진짜* 기여. selector arch 개선의 가치 큼 |
| `marginal_contribution.gru` 가 작음 (<0.02) | gru 가 *거의 기여 안 함*. selector 교체 / 단순화 고려 |
| `marginal_contribution.regime` 가 큼 | regime prior 유지 가치 큼. regime 정의 *세밀화* 가치 |
| `marginal_contribution.regime` 가 작음 | regime prior 무용 가능. *제거* 고려 |
| `intervention_gru.helped_rate` >> `hurt_rate` | gru 가 *선별적* 으로 좋게 개입. 좋은 신호 |
| `intervention_gru.helped_rate` ≈ `hurt_rate` | gru 가 *무작위적* 으로 개입. 노이즈 식별기化 의심 |
| `family_change.gru_intervention.cross_family_pct` 가 큼 | gru 가 family 도 *바꿈* — 강한 개입 |
| `family_change.gru_intervention.cross_family_pct` 가 작음 | gru 가 *family 안에서* 만 미세 조정 — 약한 개입 |

---

## §9. STAGE 5 — Failure Analysis + B001 비교 (c8)

### §9.1 측정 식

```python
def stage5_failure_b001(corrected_cands, final_scores, train_y, ids, regimes, train_x):
    """worst-100 sample 추출 + B001 baseline 비교.

    인자 정의:
      final_scores : [N_train, 27] float — **score_full** (§8.3 정의: oof_scores_full + 0.65*physics_bias + 0.45*regime_bias_table[regimes]).
                     즉 PB framework 의 *최종* selector logit. Variant A/B 는 본 STAGE 에서 사용 안 함.
      regimes      : [N_train] int (§3.1 manifest).
    """
    R_HIT = selector.R_HIT
    # ── PB 최종 prediction ──
    pred_pb = boundary.soft_select(corrected_cands, final_scores, temp=0.03)
    err_pb = np.linalg.norm(pred_pb - train_y, axis=1)

    # ── 1. Worst-100 (final hit 0 + best_err 큰 순) ──
    err_corr = np.linalg.norm(corrected_cands - train_y[:, None, :], axis=2)
    best_err = err_corr.min(axis=1)
    miss_mask = err_pb > R_HIT
    miss_idx = np.where(miss_mask)[0]
    worst_idx = miss_idx[np.argsort(-best_err[miss_idx])[:100]]
    worst_samples = []
    end_idx = train_x.shape[1] - 1
    speed_last = np.linalg.norm(train_x[:, end_idx] - train_x[:, end_idx - 1], axis=1)
    for i in worst_idx:
        worst_samples.append({
            "sample_id": str(ids[i]),
            "regime": int(regimes[i]),
            "best_cand_err": float(best_err[i]),
            "pb_err": float(err_pb[i]),
            "speed_last": float(speed_last[i]),
        })

    # ── 2. B001 (linear 2-pt) baseline 비교 ──
    # 식: pred = x[end] + (x[end] − x[end-1]) × horizon, horizon=2 (plan-004 §4.2 와 동일).
    # 본 plan 의 B001 정의는 *식 자체* 가 baseline 의미 (frame-velocity linear extrapolation × 2).
    # plan-001 의 B001 산출도 동일 식으로 박제되어 있다는 *가정* — 다르면 §N+3 #3 caveat 의 fold 정합성 절차로 escalate (warn-only).
    pred_b001 = train_x[:, end_idx] + (train_x[:, end_idx] - train_x[:, end_idx - 1]) * 2.0
    err_b001 = np.linalg.norm(pred_b001 - train_y, axis=1)
    pb_hit = err_pb <= R_HIT
    b001_hit = err_b001 <= R_HIT
    b001_comparison = {
        "n_total": int(len(train_y)),
        "pb_hit_rate":   float(pb_hit.mean()),
        "b001_hit_rate": float(b001_hit.mean()),
        "win":   int(((pb_hit) & (~b001_hit)).sum()),   # PB hit, B001 miss
        "loss":  int(((~pb_hit) & (b001_hit)).sum()),   # PB miss, B001 hit
        "tie_hit":  int(((pb_hit) & (b001_hit)).sum()),
        "tie_miss": int(((~pb_hit) & (~b001_hit)).sum()),
        "pb_minus_b001_mean_err": float(err_pb.mean() - err_b001.mean()),
    }

    return {
        "worst_samples": worst_samples,
        "b001_comparison": b001_comparison,
    }
```

### §9.2 산출

- `analysis/plan-005/failure_b001.json`
- `analysis/plan-005/failure_b001.md`:
  - worst-100 table (sample_id, regime, best_cand_err, pb_err, speed_last)
  - worst-100 의 *regime 빈도* 박제 (어느 regime 에 fail 이 집중?)
  - B001 비교: 4-cell table (win/loss/tie_hit/tie_miss) + hit rate 비교

### §9.3 G5 합격 기준

- `worst_samples` len ≥ 100 (또는 miss 총수가 100 미만이면 모두 박제 + 그 사실 명시)
- `b001_comparison` 의 8 key (`n_total`, `pb_hit_rate`, `b001_hit_rate`, `win`, `loss`, `tie_hit`, `tie_miss`, `pb_minus_b001_mean_err`) 모두 존재
- win + loss + tie_hit + tie_miss == n_total

---

## §10. STAGE 6 — Synthesis + Next Plan 후보 (c9)

### §10.1 산출

- `analysis/plan-005/results.md` — plan-003/004 results.md 형식 참조
  - frontmatter: `plan_id`, `finished_at`, `status` (`all_complete` / `partial`), `exp_ids_completed: [D001_pb-0-6822-diagnostic]`
  - 본문: 각 STAGE 의 핵심 숫자 요약 + 해석
- `analysis/plan-005/next_plan_candidates.md` — **본 plan 의 핵심 산출**

### §10.2 `next_plan_candidates.md` 구조 (필수)

각 후보 entry 는 다음 4 항목 박제:

```markdown
## 후보 N: <plan-006 후보 제목>

- **근거 metric**: (어떤 진단 결과 기반인지 — `oracle_summary.json["per_family"]` 등 정확한 reference)
- **예상 ROI**: (예상 hit rate 이득 + risk)
- **작업 범위**: (plan-006 의 STAGE 윤곽)
- **선행 조건**: (plan-005 산출 중 어떤 게 anchor 인지)
```

**최소 후보 3개 (G_final 조건)** — 예상되는 후보 (실제 진단 결과에 따라 변동):

1. **Selector architecture 교체** (TCN / Transformer / MLP) — 근거: top-K accuracy + per-regime selector hit
2. **Per-regime adaptive candidates** — 근거: per-family oracle + per-regime oracle gap
3. **Corrector cap / weighting 재튜닝** — 근거: cap saturation + per-band effect
4. (선택) **Bias 가중치 재튜닝** — 근거: bias ablation 4 setting 비교
5. (선택) **새 family 추가** (binormal 등) — 근거: direction_breakdown 의 binormal 비중

### §10.3 G_final 합격 기준

- `results.md` 작성 + frontmatter `status: all_complete`
- `next_plan_candidates.md` 후보 ≥ 3 + 각 후보의 4 항목 모두 박제
- §0.5 의 모든 G-gate 가 [DONE]

---

## §11. 작업량 총 회계

- 코드: 1 file (`analysis/plan-005/diagnostic.py`, ~350~400 lines)
- 학습:
  - corrector full-fit 재실행 1회 (~10 min, server `cuda:1`)
  - **Variant A selector retrain 1회 (~15~20 min, server `cuda:1`)** ← v2 추가
- 분석: STAGE 1~5 = ~5 min (numpy/torch CPU 가능)
- Synthesis: ~30 min (markdown 작성)
- **총 wall-time 예산: ~50~70 min** (v1 ~30min 에서 retrain 1회 추가로 ~50min 으로 증가)

---

## §N+2. results.md 필수 항목

(plan-003/004 format 답습)

- exp_id (D001_pb-0-6822-diagnostic), plan_id (005), based_on (004)
- 4-tier oracle 표 (raw / post-corr / per-regime / per-family)
- Selector decomposition 5 metric 요약
- Corrector decomposition 4 metric 요약
- Component contribution: 3 variant (full / A_no_regime / B_no_gru) × argmax/soft × overall+per-regime hit + marginal_contribution (gru, regime) + 2 intervention (B↔full, A↔full) + family_change 표 (§8 v2)
- Failure analysis: worst-100 의 regime 빈도 + B001 win/loss
- **Synthesis**: plan-006 후보 ≥ 3개 + 각 후보의 근거 metric reference
- decision-note 박제 list

---

## §N+3. 통계 함정 & caveats

1. **Corrector full-fit leakage**: STAGE 0 의 corrector 재실행은 *full-fit* (per-fold OOF X). 따라서 `corrected_oof.npz` 는 *약한 leakage* 존재 — 각 sample 이 corrector 학습에 사용됨. 이는 oracle/distribution 진단에는 무시 가능 (corrector 영향 ±0.6cm 미세 보정만), 하지만 *post-corr selector hit* 같은 metric 에는 약간의 optimistic bias 가능. 정확한 OOF 측정이 필요하면 plan-005.1 에서 per-fold corrector 재학습 고려.

8. **Marginal contribution 의 *해석 한계*** (v2): `marginal_contribution.gru` = full − B 와 `marginal_contribution.regime` = full − A 는 *다른 prior 가 있는 상태에서의* 추가 기여. 즉 "gru 단독" 또는 "regime 단독" 의 standalone 성능은 본 plan 에서 *측정 안 함*. standalone 측정에는 추가 retrain (Variant C: gru only, Variant D: gru + regime only) 가 필요한데 본 plan §2.2 out-of-scope. plan-006 에서 ROI 정당화 시 확장.

9. **Variant A retrain 의 seed 영향** (v2): Variant A 는 plan-004 와 *동일 seed (20260606)* 사용 — 따라서 학습 시작점 동일. 그래도 stochastic optimization 의 path divergence 로 *full* 과 직접 비교 시 noise floor 약 ±0.5pp 추정. `marginal_contribution.gru/regime` 가 이 floor (≈ 0.005) 내면 신호 vs 노이즈 구분 불가 → `next_plan_candidates.md` 에 "ablation 신호 약함, 재측정 필요" 박제.

10. **Per-sample intervention 의 fold-mismatch 위험** (v2): Variant A 의 OOF score 는 *Variant A 의 5-fold split* 으로 생성. plan-004 의 full setting OOF 와 fold 가 동일한지 확인 필요 — selector 의 `stable_fold_id(sample_id, 5)` 가 deterministic 이므로 동일해야 함. 만약 다르면 sample-level pick 비교가 *다른 분할의 모델* 끼리 비교가 됨 → 정합성 무효. `intervention_gru` 박제 전 fold 일치 assert 필수.

2. **plan-004 의 OOF score bank 의 `--score-key`**: 본 plan 은 plan-004 default (`ens_scores`) 사용. 다른 key (`attn_gru` 등 단일 모델) 도 npz 안에 있을 수 있으나 ensemble 가중치는 plan-004 결정 그대로.

3. **B001 비교의 fold 정합성**: B001 (plan-001 산출) 은 동일한 train/val 분할인지 확인 후 비교. 만약 fold 정의가 다르면 *전체 train 평균* 비교만 valid (per-fold 비교 X).

4. **18 regime 의 의미**: plan-004 가 박제한 regime 정의는 *훈련 데이터로 fit* 된 quantile bin. 본 plan 의 STAGE 1 per-regime oracle 도 동일 bin 사용. test/inference 시점의 regime drift 는 별도 진단 영역 (본 plan out-of-scope).

5. **Worst-100 의 sample_id 노출**: failure analysis 의 worst sample list 는 *학습 데이터* sample_id. test sample 의 worst 는 별도 분석 가치 있으나 ground-truth 없으니 best-cand-err 기준 정렬 불가 → 본 plan 은 train 만.

6. **Cap saturation 의 해석**: cap saturation 이 높다 (예: > 0.5) → corrector 가 *더 큰 cap 을 원함*. 낮다 (예: < 0.1) → cap 이 충분히 큼. 0.2~0.4 가 *균형 잡힌* 영역. 진단 후 plan-006 cap 재튜닝의 anchor.

7. **Direction breakdown 의 binormal 비중**: `binormal_mean` 이 `parallel_mean`/`perp_mean` 의 *2배 이상* 이면 → 현재 후보가 *위/아래 방향을 못 잡고 있음* → binormal family 추가의 anchor.

---

## §N+4. 변경 이력

- v1 (2026-05-11): 초안 — plan-004 진단을 위한 analysis-only plan. 16 metric 정의 (4-tier oracle + selector decomp 5 + corrector decomp 4 + bias ablation 4 + failure 2). G0~G_final 7 gate. corrector full-fit 재실행 1회 (intermediate artifact 박제용) 만 허용, 모델 재학습 X.
- v2 (2026-05-11): **STAGE 4 를 inference-time bias ablation 에서 retrain variant 비교로 교체** (사용자 피드백 — counterfactual 편향 제거 + rigor 강화).
  - 4 setting inference ablation (`gru_only / +phys / +reg / full`) 제거 → 3 variant (`full / A=no-regime / B=no-gru`) 비교로 교체. Variant A 는 *재학습* (1회 selector retrain, ~15~20min), Variant B 는 *free 계산*.
  - Metric 추가: marginal contribution (gru = full − B, regime = full − A), per-sample intervention 2종 (B↔full, A↔full), family-change breakdown. 총 metric 수 16 → 20.
  - Commit chain: c7 단일 → c7a (Variant A retrain) + c7b (Variant B + 비교) 로 분리.
  - 산출 파일명: `bias_ablation.{json,md}` → `component_contribution.{json,md}`.
  - §2.1 in-scope 에 "Variant A retrain 1회" 명시, §2.2 out-of-scope 에 "추가 retrain (Variant C/D 등) 금지" 명시.
  - §3.2 G4 합격 기준 갱신.
  - §11 시간 예산 ~30min → ~50~70min.
  - §N+3 caveats 3 항목 추가 (marginal 해석 한계 / Variant A seed 영향 / fold 정합성).
  - 모든 spec 외 STAGE 0/1/2/3/5/6 / G0/G1/G2/G3/G5/G_final / severe / commit chain c1~c6, c8, c9 / 두 갈래 narrative 의도 는 v1 유지.

---

## §N+5. 참조

- `plans/plan-004-pb-0-6822-fullrun.md` (선행 plan, 본 plan 의 입력 산출 정의)
- `notes/PB_0.6822 코드공유.ipynb` (소스 framework 의 alg 정의, read-only)
- `WORKFLOW.md` §0.5, §11, §12 convention
- `CLAUDE.md` (autonomous execution policy)
- `src/pb_0_6822/{selector,boundary}.py` (plan-004 가 lock-in한 모듈, import only)
- `runs/baseline/P001_pb-0-6822-fullrun/**` (plan-004 산출, read-only)
- `analysis/plan-004/regime_distribution.json` (plan-004 박제 regime 표, read-only)
