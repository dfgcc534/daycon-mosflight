---
plan_id: 006
version: 2
date: 2026-05-12 (Asia/Seoul)
status: partial
based_on:
  - 004
  - 005
  - notes/PB_0.6822 코드공유.ipynb
  - notes/코드공유-upgrade.md
scope: minimal-variant LB validation (no retrain — physics_bias + soft averaging only, 1 LB submission)
exp_ids:
  - E001_minimal-variant-e
lb_score: TBD
---

# plan-006 v1 — Minimal Variant E LB Validation (physics_bias + soft averaging, no GRU, no regime)

## §0. 한 줄 목적

> **plan-005 의 핵심 통찰 — "PB framework 의 95% 가 장식이고, 진짜 엔진은 *27 후보 + physics_bias + soft averaging* 3 ingredients 뿐" — 을 *1 LB 제출*로 직접 입증한다.**
>
> 구체 명제 (LB 로 검증):
>
> 1. **Variant E** (physics_bias × 0.65 + soft averaging, GRU 제거, regime 제거) 의 LB 점수가 **`lb_score ≥ 0.6606`** (= plan-004 full LB 0.6806 − 0.02, **inclusive**) 이면 plan-005 통찰 입증.
> 2. **`lb_score < 0.6606`** (strict less-than) 이면 GRU/regime 의 *out-of-sample* 기여가 OOF 측정 (noise floor) 보다 큰 것 → plan-005 의 marginal contribution 측정 신뢰도 재검토 필요. (정확히 `lb_score == 0.6606` 인 경계 sample 은 *명제 1 (입증)* 분기.)
> 3. *어떤 결과든* plan-007 의 anchor: (a) 입증 시 단순화 path 정당 + 후보 다양화에 집중, (b) 미입증 시 GRU/regime 의 OOF↔LB gap 분석.
>
> **재학습 0, 추가 학습 0, compute < 5 min, LB 제출 1회**. 가장 cheap 한 검증.

---

## §0.5 Quick Reference (autonomous loop 매 turn 읽는 section)

### 합격 기준 (G-gate sequence)

- plan-005 의 corrected_*.npz / plan-004 의 selector npz 가 local 에 존재하거나 *재생성 OK*. 위반 시 `intermediate_unrecoverable` severe.
- Variant E OOF hit (soft) finite + 구간 `0.62 ≤ x ≤ 0.68` (**inclusive 양단**). 위반 (`x < 0.62` 또는 `x > 0.68`, 양쪽 strict) 시 `oof_out_of_range` severe (계산 버그 의심).
- `runs/baseline/E001_minimal-variant-e/submission.csv` schema 가 `data/sample_submission.csv` 와 100% 일치. 위반 시 `submission_shape_mismatch` severe.
- **LB 자율 제출 1회** (`dacon-submit` skill) + `lb_score` 회수.
- `lb_score` frontmatter 3 파일 (`plans/plan-006-*.md` top-level + `.results.md` + `analysis/plan-006/results.md`) 동시 박제. plan-002/003/004 carry-over 패턴 답습.

### G-gates

- G0: STAGE 0 인프라 + plan-004/005 산출 검증 (재생성 fallback 포함)  [DONE 4cca05d]
- G1: STAGE 1 — Variant E OOF hit (raw + corrected 둘 다) + per-regime 박제  [DONE 4cca05d]
- G2: STAGE 2 — `submission.csv` 생성 + schema 검증  [DONE 20612f8]
- G3: STAGE 3 — LB 자율 제출 + `lb_score` 회수  [DONE 54119b5] (partial — lb_score TBD carry-over)
- G_final: STAGE 4 — synthesis + plan-007 후보 + 3 파일 frontmatter 동시 박제  [DONE 0511e94] (partial — lb_score TBD)

### Commit chain (next-up)

| # | type | spec section | status |
|---|---|---|---|
| c1 | docs | `plans/plan-006-minimal-variant-e-lb.md` 작성 (본 파일) | [DONE 42faa2e] |
| c2 | code | `analysis/plan-006/variant_e.py` — Variant E score 계산 + submission CSV 생성 + LB 호출 helper. spec @ §4~§7 | [DONE a2e4f7c] |
| G0 | gate | plan-004 `test_selector_scores.npz` (or 재생성 path) + plan-005 corrected_*.npz (or 재생성 path) 모두 로드 가능 | [DONE 4cca05d] |
| c3 | analysis | STAGE 1 — `analysis/plan-006/variant_e_oof.{json,md}` (raw + corrected 둘 다, overall + per-regime). spec @ §5 | [DONE 4cca05d] |
| G1 | gate | OOF hit finite + ∈ [0.62, 0.68] + per-regime 18 entry 박제 | [DONE 4cca05d] |
| c4 | sub-gen | STAGE 2 — `runs/baseline/E001_minimal-variant-e/submission.csv` 생성 + 4-line schema assert. spec @ §6 | [DONE 20612f8] |
| G2 | gate | csv 존재, shape == sample_submission.csv, columns 일치, 좌표 finite | [DONE 20612f8] |
| c5 | sub-lb | STAGE 3 — `dacon-submit` skill 자율 호출 + `analysis/plan-006/lb_log.md` 박제 + 3 파일 frontmatter `lb_score` 동시 갱신. spec @ §7 | [DONE 54119b5] (partial) |
| G3 | gate | LB 점수 회수 (float, isSubmitted=True) + lb_log.md 박제 | [DONE 54119b5] (partial) |
| c6 | synthesis | `analysis/plan-006/results.md` + `next_plan_candidates.md` (≥ 2 후보). spec @ §8 | [DONE 0511e94] |
| G_final | gate | results.md + next plan 후보 ≥ 2 + 3 파일 frontmatter 동시 박제 | [DONE 0511e94] (partial) |
| c5.1 | carry-over | (대기) lb_score 회수 후 3 파일 frontmatter TBD → `<float>` + status partial → all_complete + lb_log.md row append | [PENDING — user-driven] |

### Plan-specific severe (WORKFLOW.md §12.3 default 위 추가분)

- `intermediate_unrecoverable`: G0 진입 시점에 plan-004 `test_selector_scores.npz` 와 plan-005 `corrected_test.npz` 둘 다 부재 + 재생성 path (plan-004 `run_full.py --phase selector` + plan-005 `diagnostic.py rerun_corrector`) 도 실행 불가능. 사용자 escalate.
- `oof_out_of_range`: Variant E OOF hit `x < 0.62` 또는 `x > 0.68` (양쪽 strict, equality 는 통과). 추정 구간 (~0.6517 ± 0.03) 밖이면 계산 버그 또는 score 정의 mismatch 의심.
- `submission_shape_mismatch`: submission.csv shape ≠ sample_submission.csv (row count, column names, id 순서).
- `lb_unsubmitted`: G3 진입 시점에 LB 미회수 + carry-over 사유 미박제.
- `dacon_submit_skill_missing`: c5 진입 시 `dacon-submit` skill 부재 → 사용자 escalate.
- `lb_anomaly`: **`|lb_score − 0.6806| ≥ 0.05`** (양/음 방향 무관, **equality 포함 trigger**). 즉 `lb_score ≤ 0.6306` 또는 `lb_score ≥ 0.7306`. plan-005 통찰의 *심각한* 위반 또는 측정 오류 — synthesis 단계에서 *집중 분석 의무*.

### Plan-specific paths (WORKFLOW.md §12.5/§12.6 추가/제외)

- whitelist 추가:
  - `analysis/plan-006/**` (특히 `variant_e.py`, `variant_e_oof.{json,md}`, `lb_log.md`, `results.md`, `next_plan_candidates.md`)
  - `runs/baseline/E001_minimal-variant-e/**` (특히 `submission.csv`)
- blacklist 추가:
  - `src/pb_0_6822/**` (plan-004 lock-in, import only)
  - `runs/baseline/P001_pb-0-6822-fullrun/**` (plan-004 산출, read-only)
  - `analysis/plan-005/**` (plan-005 산출, read-only)
  - `plans/plan-001~005*` (앞선 plan 본문 수정 X)
  - `notes/PB_0.6822 코드공유.ipynb` (원본 보존)

### Decision-note 사용 예 (자율 결정 시 commit msg 박제)

- `decision-note: spec-default — Variant E_corrected 채택 (plan-004 와 apples-to-apples). E_raw 는 OOF informational only, LB 제출 X`
- `decision-note: spec-default — physics_bias 가중치 = 0.65 (plan-004 default 답습, plan-005 와 일관성). regime 항 = 0 (Variant E 정의)`
- `decision-note: spec-default — soft averaging temperature = 0.03 (plan-004 default)`
- `decision-note: spec-default — corrected_test.npz 가 local 에 있으면 재사용, 없으면 plan-005 diagnostic.py 재실행 (~10min)`
- `decision-note: spec-default — LB 제출 1회 (Variant E_corrected 만). Variant F (uniform) sanity check 는 budget 절약 위해 LB 제출 X, OOF 만 박제`
- `decision-note: spec-default — exp_id=E001_minimal-variant-e (E prefix = Experimental simplification)`

---

## §1. 배경

### §1.1 plan-005 인계 — Variant E 추정 근거

plan-005 STAGE 4 의 측정값:

| Variant | OOF hit (soft) | 측정/추정 |
|---|---|---|
| full (GRU + physics + regime) | **0.6599** | 측정 |
| Variant A (GRU + physics, no regime) | 0.6570 | 측정 |
| Variant B (physics + regime, no GRU) | **0.6547** | 측정 |
| **Variant E (physics 만, no GRU, no regime)** | **~0.6517** | **추정** (= Variant B 0.6547 − regime 기여 0.003 = **0.6517**) |

LB 추정 (`OOF→LB gap = LB_full − OOF_full = 0.6806 − 0.6599 = +0.0207`, plan-004 측정):
- full LB **0.6806** (측정, plan-004)
- Variant E LB **~0.6724** (추정 = `0.6517 + 0.0207`, gap 동일 가정)

### §1.2 본 plan 의 검증 명제

| 명제 | 검증 방법 | 합격 산출 |
|---|---|---|
| **Variant E 가 full 대비 −2pp 이내** | LB 1회 제출 → lb_score 비교 | frontmatter `lb_score`, 0.66+ 이면 입증 |
| Per-regime 에서 *극단적 손실 regime* 식별 | OOF per-regime hit 분석 | `variant_e_oof.json["per_regime"]` |
| plan-005 OOF↔LB gap 의 *variant 간 일관성* 확인 | full vs Variant E 의 gap 비교 | `results.md` 비교 표 |

### §1.3 *왜* 이게 cheap 한가

| 기존 plan | 본 plan |
|---|---|
| plan-004: selector 5-fold 학습 ~15min + corrector ~10min | **재학습 0** |
| plan-005: corrector 재학습 ~10min + Variant A 재학습 ~20min | **재학습 0** |
| 본 plan: physics_bias 계산 + soft averaging 한 줄 | **compute < 1min** |

→ LB 제출 1회 빼면 wall-time *5 min 이내*. plan-005 의 corrected_test.npz 가 local 에 남아있으면 더 짧음.

---

## §2. Scope (명시적)

### §2.1 In-scope

| 항목 | 값 |
|---|---|
| 입력 | plan-004 `test_selector_scores.npz` (broad recompute 시), plan-005 `corrected_test.npz`, `data/train_labels.csv`, `data/test/` |
| 분석 | Variant E OOF + per-regime + Variant F (uniform) OOF informational |
| Submission CSV | 1개 (Variant E_corrected) |
| LB 제출 | 1회 (`dacon-submit` skill 자율 호출) |
| Synthesis | results.md + plan-007 후보 ≥ 2 |
| 산출 위치 | `analysis/plan-006/**`, `runs/baseline/E001_minimal-variant-e/**` |

### §2.2 Out-of-scope (절대 안 함)

| 항목 | 이유 |
|---|---|
| Selector / Corrector 재학습 | analysis-only plan, plan-005 산출 재사용 |
| Hyperparam 변경 (physics_bias 가중치, soft temperature 등) | plan-004 default 답습 — 변경은 plan-007 |
| 27 후보 / regime 정의 변경 | 본 plan 은 *단순화 검증*, 후보 개선은 plan-007 |
| Variant A (GRU + physics) LB 제출 | plan-005 의 Variant A 산출 충분, 본 plan 의 core hypothesis 는 Variant E |
| Variant F (uniform) LB 제출 | OOF informational only — LB budget 절약 |
| 다중 LB 제출 (2회 이상) | 본 plan 의 명제는 Variant E 단일 — 추가 비교는 plan-007 |
| plan-004/005 모듈 수정 | lock-in, import only |
| End-to-end 학습 | 본 plan analysis-only |

---

## §3. 사전 등록 (Pre-registration)

### §3.1 입력 데이터 + 분할

| 분할 | 출처 | 사용 |
|---|---|---|
| Train (5-fold OOF) | `data/train/` + `data/train_labels.csv` + plan-005 `corrected_oof.npz` | OOF hit 측정 |
| Test | `data/test/` + plan-005 `corrected_test.npz` | Submission CSV 생성 + LB 제출 |
| Fold 정의 | `selector.stable_fold_id(sample_id, 5)` (plan-004 와 동일) | OOF 정합성 |

### §3.2 합격 기준 (정량)

- **G0**: plan-005 `corrected_test.npz` + `corrected_oof.npz` **두 파일 모두 *물리적으로 존재 + `np.load` 통과* (AND)**. 부재 시 §4.2 재생성 fallback 실행 → fallback 완료 후 동일 AND 조건 재평가. 재평가 실패 시 `intermediate_unrecoverable` severe
- **G1**:
  - `variant_e_oof.json["E_corrected"]["soft"]` finite + **`0.62 ≤ x ≤ 0.68`** (inclusive 양단)
  - `variant_e_oof.json["E_raw"]["soft"]` finite + **`0.55 ≤ x ≤ 0.72`** (inclusive 양단, raw 는 분포 더 넓음)
  - `variant_e_oof.json["per_regime"]` 18 entry 중 sample 있는 regime 모두 박제. **sample == 0 인 regime 은 key 자체 생략 허용 (null/NaN 박제 금지)**. entry 들의 `n` 합 == `len(train_y)` 검증
- **G2**: `submission.csv` shape == `sample_submission.csv`, columns 일치, id 순서 일치, 좌표 모두 finite (= `pd.to_numeric(errors='coerce').notna().all()` ∧ `np.isfinite(...).all()`)
- **G3**: dacon-submit 응답 `isSubmitted=True` + `lb_score: float`. carry-over (`isSubmitted=True, lb_score=None`) 도 G3 부분 통과 (`status=partial`, frontmatter `lb_score: TBD`)
- **G_final**:
  - `analysis/plan-006/results.md` 작성
  - `analysis/plan-006/next_plan_candidates.md` 후보 ≥ 2 + 각 후보의 *근거 metric* 박제
  - 3 파일 (`plans/plan-006-minimal-variant-e-lb.md` top-level + `plans/plan-006-minimal-variant-e-lb.results.md` + `analysis/plan-006/results.md`) frontmatter `lb_score` 동시 갱신

### §3.3 평가

- **OOF hit (soft)** — 식 정의 (각 sample i ∈ [0, N_train), 후보 c ∈ {0..26}):
  - `score_E[i, c] = 0.65 × physics_bias[c]` (sample 무관 상수, broadcast to `(N_train, 27)`, dtype=float32)
  - `weight[i, c] = softmax(score_E[i, :] / temp)[c]`, `temp = 0.03` (softmax temperature; scores/temp 가 분자, 작을수록 winner-take-all)
  - `pred[i] = Σ_{c=0..26} weight[i, c] × corrected_cands[i, c, :]` (3-D 좌표의 가중평균, shape `(3,)`)
  - `hit[i] = (||pred[i] − train_y[i]||_2 ≤ R_HIT)`, `R_HIT = selector.R_HIT = 0.01` (m)
  - 최종 metric = `float(hit.mean())` ∈ [0, 1]
- **LB**: dacon submission 응답의 `lb_score` (hit_rate_at_1cm, soft submission 기준)
- **temp 값 (0.03) 의 근거**: plan-004 의 `search_temperature` 가 *full setting* 에서 OOF 최적화한 값. 본 plan 은 plan-004 와의 apples-to-apples 비교 위해 동일 temp 고정 — Variant E 에서의 temp 최적화는 plan-007 변수 (§N+3 caveat 4)

---

## §4. STAGE 0 — 인프라 + 산출 검증 (c2 일부)

### §4.1 `analysis/plan-006/variant_e.py` 골격

```python
# analysis/plan-006/variant_e.py
"""plan-006 Variant E (physics_bias + soft averaging) 검증 entry.
재학습 X, plan-004/005 산출 재사용. 부재 시 fallback 재생성.

Module-level import 정책:
  - selector  : STAGE 0~3 전반 (read_labels, load_stack, make_candidates,
                candidate_physics_bias, fit_regime_bins, assign_regimes,
                read_submission_ids, CANDIDATES, R_HIT, EPS)
  - boundary  : STAGE 1+ 의 soft_select 호출 (STAGE 0 verify_inputs 는 사용 X
                — module-level 두는 이유는 STAGE 1 진입 시 import 비용 0)
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from src.pb_0_6822 import selector, boundary

REPO = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO / "data"
PLAN004_DIR = REPO / "runs/baseline/P001_pb-0-6822-fullrun"
PLAN005_DIR = REPO / "analysis/plan-005"
ANALYSIS_DIR = REPO / "analysis/plan-006"
RUN_DIR     = REPO / "runs/baseline/E001_minimal-variant-e"
DEVICE = "cuda:1"  # plan-004/005 일관성

def verify_inputs():
    """G0 — 필수 산출 검증. 부재 시 재생성 path 안내."""
    paths = {
        "corrected_oof":  PLAN005_DIR / "corrected_oof.npz",   # heavy, gitignored
        "corrected_test": PLAN005_DIR / "corrected_test.npz",  # heavy, gitignored
    }
    missing = [k for k, p in paths.items() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"plan-005 corrected_*.npz 부재: {missing}. "
            f"재생성: `python -m analysis.plan-005.diagnostic --rerun-corrector` "
            f"(또는 plan-005 diagnostic.py 의 rerun_corrector_save_intermediates() 호출)."
        )
    return paths
```

### §4.2 재생성 fallback path

corrected_*.npz 부재 시:

```bash
# plan-005 diagnostic.py 재실행 (corrector full-fit + 산출 박제)
# ★ 주의: analysis/plan-005/ 는 hyphen 디렉토리 → Python package import 불가.
#   따라서 sys.path 직접 조작 후 module-as-script import.
cd /Users/dryas/Desktop/dacon-mosflight
python -c "
import sys
from pathlib import Path
plan005_dir = Path('analysis/plan-005').resolve()
sys.path.insert(0, str(plan005_dir))
import diagnostic
diagnostic.rerun_corrector_save_intermediates()
"
# ~10min, cuda:1 사용. 산출: analysis/plan-005/corrected_{oof,test}.npz
```

또는 cwd 가 `analysis/plan-005/` 인 상태에서 직접 실행:

```bash
cd /Users/dryas/Desktop/dacon-mosflight/analysis/plan-005
python -c "import diagnostic; diagnostic.rerun_corrector_save_intermediates()"
```

→ STAGE 0 자체는 *재학습 0* 이지만 fallback path 가 *plan-005 의 corrector 재학습 1회* 를 허용 (decision-note 박제).

### §4.3 G0 자동 판정

- `verify_inputs()` AssertionError/FileNotFoundError 없이 통과
- `corrected_test.npz["corrected"].shape == (N_test, 27, 3)` finite
- `corrected_oof.npz["corrected"].shape == (N_train, 27, 3)` finite

### §4.4 시간 예산

- 재생성 불필요: < 10초
- 재생성 필요: ~10min (corrector full-fit)

---

## §5. STAGE 1 — Variant E OOF Compute (c3)

### §5.1 측정 식

```python
def stage1_variant_e_oof() -> dict:
    """Variant E (physics_bias + soft averaging) 의 OOF hit 측정.
    E_raw (27 raw cands) + E_corrected (27 corrected cands) 둘 다.

    Returns dict with keys: E_raw, E_corrected, per_regime, F_corrected_soft,
        physics_bias_argmax_idx, physics_bias_argmax_name, physics_bias_top5_names.

    Plan-004 모듈 계약 (lock-in, 본 plan import only):
      - selector.candidate_physics_bias(candidates, target) → np.ndarray[27] (float32, centered)
          식: bias[c] = log(hit_rate[c] + 1e-4) - 18 * mean_err[c]; bias -= bias.mean()
          centered 보장: |bias.mean()| < 1e-5
          unique 보장 안 됨 (tie 시 numpy argmax = first-index 규약)
      - boundary.soft_select(cands [N,27,3], scores [N,27], temp float) → np.ndarray[N,3] (float32)
          식: weight[i,c] = softmax(scores[i,:] / temp)[c]
              pred[i, :] = Σ_c weight[i,c] × cands[i,c,:]
          temp: temperature (분자가 scores/temp), 작을수록 winner-take-all
      - selector.CANDIDATES: list[CandidateSpec] (len == 27), 각 .name: str (27개 unique 보장)
      - selector.R_HIT: float == 0.01 (m), hit threshold
    """
    # ── 1. 입력 로드 ──
    ids, train_y = selector.read_labels(DATA_ROOT / "train_labels.csv")
    train_x = selector.load_stack(DATA_ROOT / "train", ids)
    z = np.load(PLAN005_DIR / "corrected_oof.npz")
    raw_cands       = z["cands"].astype(np.float32)        # [N_train, 27, 3]
    corrected_cands = z["corrected"].astype(np.float32)    # [N_train, 27, 3]

    # ── 2. physics_bias 계산 (plan-004 와 동일 식) ──
    end_idx = train_x.shape[1] - 1
    train_cands_check = selector.make_candidates(train_x, end_idx, horizon=2).astype(np.float32)
    assert np.allclose(train_cands_check, raw_cands, atol=1e-5), \
        "raw_cands mismatch — plan-005 산출이 stale 일 수 있음"
    physics_bias = selector.candidate_physics_bias(raw_cands, train_y)  # [27] float32, centered
    assert physics_bias.shape == (27,), f"shape: {physics_bias.shape}"
    assert physics_bias.dtype == np.float32, f"dtype: {physics_bias.dtype}"
    assert abs(physics_bias.mean()) < 1e-5, f"not centered: mean={physics_bias.mean()}"

    # ── 3. Variant E score (sample 무관, 상수 27-vec; broadcast → [N_train, 27]) ──
    score = (0.65 * physics_bias[None, :]).astype(np.float32)
    score_E = np.broadcast_to(score, (len(train_y), 27))

    R_HIT = selector.R_HIT   # == 0.01 (m), assert in §4.3

    # ── 4. E_raw / E_corrected OOF hit ──
    def _measure(cands: np.ndarray) -> dict:
        """cands: [N, 27, 3] float32 → {argmax, soft, _err_argmax, _err_soft}.
        score 가 sample 무관 상수라 argmax 결과도 sample 무관 — numpy argmax
        first-index 규약 적용 (tie 발생 가능, physics_bias unique 보장 X)."""
        pick = cands[np.arange(len(train_y)), score_E.argmax(axis=1)]
        err_arg = np.linalg.norm(pick - train_y, axis=1)   # [N] float32
        soft = boundary.soft_select(cands, score_E, temp=0.03)   # [N, 3]
        err_soft = np.linalg.norm(soft - train_y, axis=1)
        return {
            "argmax": float((err_arg <= R_HIT).mean()),
            "soft":   float((err_soft <= R_HIT).mean()),
            "_err_argmax": err_arg, "_err_soft": err_soft,
        }
    E_raw       = _measure(raw_cands)
    E_corrected = _measure(corrected_cands)

    # ── 5. Per-regime hit ──
    # regime key 는 JSON 직렬화 시 string 으로 자동 변환 (json.dumps regime 0 → "0")
    # sample == 0 인 regime 은 entry 자체 생략 (§3.2 G1 — null/NaN 박제 금지)
    bins = selector.fit_regime_bins(train_x, end_idx)
    regimes = selector.assign_regimes(train_x, end_idx, bins)   # [N] int64, ∈ [0, 17]
    per_regime: dict[int, dict] = {}
    for r in range(18):
        mask = regimes == r
        if mask.sum() == 0: continue
        per_regime[int(r)] = {
            "n": int(mask.sum()),
            "E_corrected_soft": float((E_corrected["_err_soft"][mask] <= R_HIT).mean()),
            "E_raw_soft":       float((E_raw["_err_soft"][mask] <= R_HIT).mean()),
        }
    # G1 검증: sum of n == len(train_y)
    assert sum(v["n"] for v in per_regime.values()) == len(train_y), \
        f"per_regime n sum mismatch: {sum(v['n'] for v in per_regime.values())} != {len(train_y)}"

    # ── 6. Variant F (uniform) sanity check (informational) ──
    score_F = np.zeros((len(train_y), 27), dtype=np.float32)  # 모든 후보 동일
    soft_F = boundary.soft_select(corrected_cands, score_F, temp=0.03)
    err_F = np.linalg.norm(soft_F - train_y, axis=1)
    F_corrected_soft = float((err_F <= R_HIT).mean())

    # ── 7. 박제 ──
    for d in (E_raw, E_corrected):
        d.pop("_err_argmax", None); d.pop("_err_soft", None)
    # selector.CANDIDATES[i].name 은 str (27 unique 보장) — JSON 직렬화 시 list[str] 그대로
    summary: dict = {
        "E_raw":       E_raw,
        "E_corrected": E_corrected,
        "per_regime":  per_regime,
        "F_corrected_soft": F_corrected_soft,
        "physics_bias_argmax_idx":  int(physics_bias.argmax()),
        "physics_bias_argmax_name": str(selector.CANDIDATES[int(physics_bias.argmax())].name),
        "physics_bias_top5_names": [
            str(selector.CANDIDATES[i].name)
            for i in np.argsort(-physics_bias)[:5].tolist()
        ],
    }
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    (ANALYSIS_DIR / "variant_e_oof.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )
    return summary
```

### §5.2 산출

- `analysis/plan-006/variant_e_oof.json` — 위 8 key
- `analysis/plan-006/variant_e_oof.md`:
  - 전체 hit (E_raw argmax/soft, E_corrected argmax/soft, F_corrected_soft) 1줄
  - Per-regime 18 row table `| regime | n | E_corrected_soft | E_raw_soft |`
  - physics_bias 의 top-5 후보 이름 (해석용)
  - plan-005 의 Variant B (0.6547) 와 비교

### §5.3 G1 합격 기준 (자동 판정)

- 7 top-level key 모두 존재 (`E_raw`, `E_corrected`, `per_regime`, `F_corrected_soft`, `physics_bias_argmax_idx`, `physics_bias_argmax_name`, `physics_bias_top5_names`)
- `E_corrected.soft` finite + `0.62 ≤ x ≤ 0.68` (inclusive). 위반 (`x < 0.62` 또는 `x > 0.68`) 시 `oof_out_of_range` severe
- `E_raw.soft` finite + `0.55 ≤ x ≤ 0.72` (inclusive)
- `F_corrected_soft < E_corrected.soft` (**strict less-than**; equality 시 warn-only flag `physics_bias_anomaly` 발동). uniform 이 physics_bias 보다 *못함* 검증 — physics_bias 가 nonsense 가 아님 sanity check
- `per_regime` 의 모든 entry `n` 의 합 == `len(train_y)` (assert 통과)

### §5.4 시간 예산

- ~30 초 (numpy/torch CPU)

---

## §6. STAGE 2 — Submission CSV 생성 (c4)

### §6.1 측정 식

```python
def stage2_submission_csv():
    """Variant E test inference + submission.csv 생성."""
    # ── 1. Test 입력 ──
    test_ids = selector.read_submission_ids(DATA_ROOT / "sample_submission.csv")
    test_x   = selector.load_stack(DATA_ROOT / "test", test_ids)
    z = np.load(PLAN005_DIR / "corrected_test.npz")
    raw_test       = z["cands"]        # [N_test, 27, 3]
    corrected_test = z["corrected"]    # [N_test, 27, 3]

    # ── 2. physics_bias (train 으로부터, plan-005 STAGE 1 과 동일) ──
    ids, train_y = selector.read_labels(DATA_ROOT / "train_labels.csv")
    train_x = selector.load_stack(DATA_ROOT / "train", ids)
    end_idx = train_x.shape[1] - 1
    train_cands = selector.make_candidates(train_x, end_idx, horizon=2)
    physics_bias = selector.candidate_physics_bias(train_cands, train_y)

    # ── 3. Variant E test score ──
    score = np.broadcast_to(0.65 * physics_bias[None, :],
                            (len(test_ids), 27))

    # ── 4. Soft averaging on corrected_test ──
    pred = boundary.soft_select(corrected_test, score, temp=0.03)   # [N_test, 3]

    # ── 5. CSV 박제 ──
    # ref column 순서 계약 (sample_submission.csv): (id_col, coord_0, coord_1, coord_2) 4 컬럼.
    # 인덱스 0 = id, 1~3 = 좌표 — 순서 가정 위반 시 좌표가 잘못 매핑됨 → assert 로 검증.
    ref = pd.read_csv(DATA_ROOT / "sample_submission.csv")
    assert ref.shape[1] == 4, f"sample_submission columns != 4: {ref.shape[1]}"
    id_col_name = ref.columns[0]
    coord_col_names = list(ref.columns[1:])   # 길이 3
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    sub_df = pd.DataFrame({
        id_col_name:        test_ids,
        coord_col_names[0]: pred[:, 0],
        coord_col_names[1]: pred[:, 1],
        coord_col_names[2]: pred[:, 2],
    })
    sub_df.to_csv(RUN_DIR / "submission.csv", index=False)

    # ── 6. 5-line assert (plan-004 §9 답습 + non-finite 강화) ──
    sub = pd.read_csv(RUN_DIR / "submission.csv")
    assert sub.shape == ref.shape, f"shape: sub={sub.shape} ref={ref.shape}"
    assert list(sub.columns) == list(ref.columns), "columns mismatch"
    assert (sub.iloc[:, 0].astype(str).values == ref.iloc[:, 0].astype(str).values).all(), "id order"
    coord_cols = sub.columns[1:]
    # pd.to_numeric 으로 NaN 검출 + np.isfinite 으로 ±inf 까지 차단
    coord_numeric = sub[coord_cols].apply(pd.to_numeric, errors='coerce')
    assert coord_numeric.notna().all().all(), "non-numeric coord"
    assert np.isfinite(coord_numeric.to_numpy()).all(), "non-finite coord (NaN/±inf)"
```

### §6.2 G2 합격 기준 (자동 판정)

위 4-line assert 가 모두 통과. 실패 시 `submission_shape_mismatch` severe.

### §6.3 시간 예산

- ~10 초

---

## §7. STAGE 3 — LB 제출 + 박제 (c5)

### §7.1 자율 호출 (CLAUDE.md autonomous policy)

```
Skill(skill="dacon-submit",
      args="runs/baseline/E001_minimal-variant-e/submission.csv E001_minimal-variant-e")
```

### §7.2 응답 4-분기 처리 (plan-004 §10 패턴 답습)

| (isSubmitted, lb_score) | 처리 | frontmatter `lb_score` | status | severe |
|---|---|---|---|---|
| (True, float) | full success — 3 파일 frontmatter 동시 갱신 | `<float>` 소수 4자리 | `all_complete` | — |
| (True, None) | partial — 점수 비동기. lb_log + results 박제, 점수 도착 시 c5.1 carry-over | `TBD` (string) | `partial` | — |
| (False, *) | submission failed — `detail` 박제 후 1회 retry (60초 sleep). 분류 규약 (substring 매칭, case-insensitive): **일시적** = `detail.lower()` 에 `rate_limit` ∨ `timeout` ∨ `network` ∨ `5xx` ∨ `eof` ∨ `connection reset` ∨ `temporarily` 중 ≥1 매칭, **영구** = `auth` ∨ `unauthorized` ∨ `forbidden` ∨ `invalid_file` ∨ `bad_request` ∨ `quota_exceeded` ∨ `403` ∨ `400` 중 ≥1 매칭. **양쪽 모두 매칭 시 영구 우선** (즉시 escalate, retry 생략). **양쪽 모두 비매칭 시 일시적 default** (1회 retry). 재실패 시 escalate. | `null` (YAML null) | `partial` | `lb_unsubmitted` |
| Skill 미존재 / 호출 exception | 즉시 escalate | `null` | `partial` | `dacon_submit_skill_missing` |

**Frontmatter `lb_score` YAML 직렬화 규약**:
- `<float>`: number (예: `0.6712`)
- `TBD`: string (예: `lb_score: TBD`)
- `null`: YAML null (예: `lb_score: null`)
- downstream parser (registry / grep) 는 이 3 type 을 모두 처리해야 함 (string match 후 isnumeric check 권장)

### §7.3 `analysis/plan-006/lb_log.md` 포맷 (markdown table)

```markdown
| timestamp_kst             | exp_id                    | isSubmitted | lb_score | detail |
|---------------------------|---------------------------|-------------|----------|--------|
| 2026-05-12T14:23:45+09:00 | E001_minimal-variant-e    | true        | 0.6712   | OK     |
```

- `timestamp_kst`: ISO8601 with `+09:00`
- `lb_score`: 소수 4자리 float, `TBD` (partial), `null` (false)
- `detail`: 응답 detail, 80자 초과 시 truncate + `...`

### §7.4 `lb_score` frontmatter 동시 갱신 (3 파일)

1. `plans/plan-006-minimal-variant-e-lb.md` top-level `lb_score: <float|TBD|null>` (현재 `null` → 갱신)
2. `plans/plan-006-minimal-variant-e-lb.results.md` frontmatter `lb_score:`
3. `analysis/plan-006/results.md` frontmatter `lb_score:`

→ 단일 commit (c5) 에 3 파일 함께 staged. carry-over 시 follow-up commit `c5.1` 에서 3 파일 *동시* 갱신.

**Carry-over trigger 정책** (partial → all_complete):
- **polling 자동화 X** — autonomous loop 가 background polling 안 함 (resource 낭비 방지)
- **Trigger 종류** (둘 중 하나):
  1. *다음 사용자 turn invoke* — 사용자가 plan-006 status 재확인을 직접 요청
  2. *명시적 follow-up plan-006.1 실행* — plan-002/003/004 pattern 답습
- **점수 확인 명령**:
  ```bash
  python -c "
  from src.submit import poll_lb_score   # poll_lb_score(exp_id) → float | None
  print(poll_lb_score('E001_minimal-variant-e'))
  "
  ```
  또는 dacon submission history 페이지 수동 확인 후 `lb_score` 직접 입력.
- **점수 회수 후 c5.1 commit**:
  - 3 파일 frontmatter `lb_score: TBD` → `<float>` 동시 갱신
  - `lb_log.md` 에 **row append** (update 가 아닌 *chronological row 추가*, 동일 exp_id 의 다중 entry 허용)
  - frontmatter `status: partial` → `all_complete`

### §7.5 시간 예산

- LB 제출 + 응답 대기: ~30 초 ~ 수 분 (서버 응답 지연 가능)
- 최초 호출 timeout: **5 분** (dacon-submit skill default. 5 분 내 응답 없으면 일시적 실패로 분류 → 60초 sleep + retry)
- Retry 도 5 분 timeout, 재실패 시 `lb_unsubmitted` severe escalate

---

## §8. STAGE 4 — Synthesis + plan-007 후보 (c6)

### §8.1 `analysis/plan-006/results.md`

frontmatter:
```yaml
---
plan_id: 006
based_on:
  - 004
  - 005
finished_at: <ISO8601 KST>
status: all_complete | partial
exp_ids_completed:
  - E001_minimal-variant-e
lb_exp_id: E001_minimal-variant-e
lb_score: <float|TBD|null>
lb_submitted_at: <ISO8601 KST>
---
```

본문:
- 핵심 수치: Variant E LB vs plan-004 full LB, OOF E_corrected/E_raw, F_corrected
- plan-005 통찰의 *LB 입증/반증*
- per-regime Variant E hit + plan-004 의 per-regime hit 비교 (가능 시)
- decision-note 박제 list

### §8.2 `analysis/plan-006/next_plan_candidates.md`

**최소 후보 2개 (G_final 조건)**. 결과에 따라 분기:

**시나리오 A — LB 입증 (Variant E LB ≥ 0.66)**:
1. **후보 다양화 (high-speed regime 타겟)** — plan-005 의 worst-100 regime 분포 기반. 27 → 35 cand
2. **Variant E 위에 boundary corrector 재설계** — corrector loss 의 oracle 손실 회복

**시나리오 B — LB 미입증 (Variant E LB < 0.66)**:
1. **GRU/regime 의 OOF↔LB gap 분석** — plan-005 의 marginal contribution 신뢰도 재검토
2. **Variant E + minimal GRU/regime 의 hybrid LB 측정** — 어디서부터 정보가 살아나는지 binary search

각 후보의 4 항목 박제:
- 근거 metric (Variant E LB, OOF, per-regime 등 정확한 reference)
- 예상 ROI
- 작업 범위
- 선행 조건

### §8.3 G_final 합격 기준

- `results.md` + `next_plan_candidates.md` 모두 작성
- 후보 ≥ 2 + 4 항목 박제
- 3 파일 frontmatter `lb_score` 동시 갱신 + `status: all_complete` (또는 `partial` + carry-over 사유)
- 모든 G-gate [DONE]

---

## §N+1. 작업량 총 회계

- 코드: 1 file (`analysis/plan-006/variant_e.py`, ~150 lines)
- 학습: **0회** (corrector 재생성 fallback 발동 시 ~10min)
- 분석: STAGE 1+2 = ~1 min
- LB 제출: ~30 초 ~ 수 분
- Synthesis: ~20 min (markdown 작성)
- **총 wall-time 예산: ~5 min ~ 25 min** (corrector 재생성 여부에 따라)

---

## §N+2. results.md 필수 항목

(plan-003/004/005 format 답습)

- exp_id (E001_minimal-variant-e), plan_id (006), based_on (004 + 005)
- lb_exp_id, lb_score, lb_submitted_at
- Variant E OOF (E_raw + E_corrected) + per-regime
- Variant F (uniform) OOF informational
- plan-004 full 대비 비교 표 (OOF, LB, gap)
- plan-005 통찰의 *LB 입증/반증* 결론
- plan-007 후보 ≥ 2 + 4 항목
- decision-note 박제 list

---

## §N+3. 통계 함정 & caveats

1. **corrected_test.npz 의 seed 정합성**: plan-005 의 corrector 재학습 산출 — corrector 의 random init / dropout 등이 seed 에 의존. plan-005 산출이 *stale* 이거나 다른 seed 로 재생성되면 결과가 다를 수 있음. plan-005 의 `corrector_seed_drift` warn (RMSE 0.000814 m 박제됨) 으로 검증된 상태.
2. **OOF↔LB gap 의 variant 간 비상수**: plan-004 의 OOF→LB gap = +0.0207. Variant E 의 gap 이 동일하다는 보장 없음 — full-fit 효과가 component 별로 다를 수 있음. caveats §1 의 추정치 [0.66, 0.68] 도 이 불확실성 반영.
3. **physics_bias 의 train 의존성**: physics_bias 는 *train 데이터의 hit_rate × mean_err* 통계 — train 분포가 test 와 다르면 *systematic bias*. plan-004 spec 그대로 답습이므로 plan-004 와 동일 risk.
4. **Soft averaging 의 temperature 민감도**: temp=0.03 은 plan-004 default. plan-005 search_temperature 가 *full setting* 에서 최적화한 값이라 Variant E 에서 부적합 가능. 본 plan 은 plan-004 와의 비교 위해 동일 temp 유지 — temperature tuning 은 plan-007 변수.
5. **Variant E 의 argmax 는 trivial**: score 가 sample 무관 상수라 argmax 가 *모든 sample 동일 후보* 픽. plan-005 사용자 finding ("always pick most popular candidate") 의 정확한 구현. 예상 hit 매우 낮음 (~0.45). **plan-006 의 main metric 은 soft 만**, argmax 는 informational.
6. **Variant F (uniform) 의 의미**: physics_bias 까지 제거. 27 후보 *균등 가중 centroid*. 명시적 통계 prior 가 없으므로 27 후보 중 *나쁜 후보* (p0_2d1 등) 가 centroid 를 끌어내림 → 예상 hit ~0.55. plan-005 의 "Level 3" 단순화 시뮬레이션. *LB 제출 X* (informational).
7. **LB 제출 1회 의 reproducibility**: dacon submission 은 1 회당 *서로 다른 응답 가능* (서버 random seed 등). 본 plan 은 1회 제출 — 재현 위험 작지만 0 아님. 점수가 nominal 범위 `0.62 ≤ x ≤ 0.72` (inclusive 양단) 밖이면 *caveat*. severe 인 `lb_anomaly` 와는 별개 — `lb_anomaly` 는 `|x − 0.6806| ≥ 0.05` (= `x ≤ 0.6306` 또는 `x ≥ 0.7306`, §0.5 L74 와 일관). nominal caveat 은 *경고만*, severe 는 *escalate*. 두 구간 관계: `lb_anomaly` 구간 ⊃ nominal caveat 구간 (lb_anomaly 발동 시 nominal caveat 도 자동 발동).

---

## §N+4. 변경 이력

- v1 (2026-05-12): 초안 — plan-005 의 Variant E 추정 (~0.6517 OOF, ~0.67 LB) 을 *직접 LB 제출* 로 검증하는 cheap experiment. STAGE 0~4, G0~G_final 5 gate, commit chain c1~c6. 재학습 0 (corrector 재생성 fallback 만 허용). 1 LB 제출.
- v2 (2026-05-12): **plan-review 결과 반영 spec 강화** (BLOCKER 17 / AMBIGUITY 29+ / MINOR 22 해소).
  - **§0 명제 1+2 cutoff 명확화** (L25-L26): "−2pp 이내 (~0.66+)" 모순 → `lb_score ≥ 0.6606` (inclusive). 명제 2 = `lb_score < 0.6606` (strict). 경계 `== 0.6606` 은 명제 1 분기.
  - **§0.5 severe 경계 inclusive 통일** (L38, L74): `oof_out_of_range` 는 `0.62 ≤ x ≤ 0.68` 양단 inclusive. `lb_anomaly` 는 `|x − 0.6806| ≥ 0.05` (equality 포함 trigger).
  - **§1.1 Variant E 산수 정정** (L110): "Variant B − regime 기여 +0.003" → "= 0.6547 − 0.003 = 0.6517" 부호/수식 정합. **OOF→LB gap 산출식 박제** (L114): `0.6806 − 0.6599 = +0.0207`, Variant E LB ~0.6724.
  - **§3.2 G0 AND 명시** + **§3.2 G1 inclusive 명시** + **per-regime entry 생략 규약** (sample == 0 인 regime 은 key 자체 생략, null/NaN 박제 금지). **§3.3 OOF hit 식 박제** (softmax + 가중평균 5 줄 풀어 적음, dtype/temp 단위 명시).
  - **§4.1 module-level import 정책 docstring 추가** (boundary 는 STAGE 1+ 용, STAGE 0 verify_inputs 는 사용 X). **§4.2 plan-005 hyphen 디렉토리 import 해결** — `sys.path.insert` 또는 `cd analysis/plan-005` 두 path 박제.
  - **§5.1 plan-004 모듈 계약 docstring 추가** — `candidate_physics_bias` (식 + dtype + centered), `soft_select` (signature + 식 + temp 단위), `CANDIDATES.name` (str, unique 보장), `R_HIT` 값. dtype assert 박제.
  - **§5.1 _measure 함수 type hint + tie 규약 docstring**. **per_regime n 합 == len(train_y) assert 추가**. JSON 직렬화 시 regime key int → string 자동 변환 명시 + `ensure_ascii=False`.
  - **§5.3 G1 합격 기준** — top-level key 7 개로 정확히 enumeration. `F < E` strict less-than 명시.
  - **§6.1 sample_submission column 순서 가정 명시** — 4 컬럼 (id, coord_0, coord_1, coord_2) assert + 동적 column 매핑. **§6.1 5-line assert** — non-finite 검사를 `pd.to_numeric` + `np.isfinite` 둘 다 적용 (±inf 차단).
  - **§7.2 detail enum 정규식** — 일시적/영구 substring set 명시 (case-insensitive), 양쪽 매칭 시 영구 우선, 양쪽 비매칭 시 일시적 default. (False, \*) 의 frontmatter `null` 박제 (이전 "(미박제)" 와 모순 해소). **§7.4 carry-over trigger 정책 명시** — polling 자동화 X, 트리거 = 사용자 turn 또는 follow-up plan. `poll_lb_score` 함수 호출 명시. **§7.5 timeout 5 분 명시**.
  - **§N+3 nominal range vs lb_anomaly 관계 명시** — nominal `[0.62, 0.72]` 는 caveat 만, lb_anomaly `|x − 0.6806| ≥ 0.05` 는 severe.
  - 모든 spec 외 STAGE 0/1/2/3/4 / G-gate / commit chain c1~c6 / severe trigger 종류 / 두 검증 명제 의도 는 v1 유지.

---

## §N+5. 참조

- `plans/plan-004-pb-0-6822-fullrun.md` (full framework 정의 + LB 0.6806)
- `plans/plan-005-pb-0-6822-diagnostic.md` (Variant B/A 측정 + Variant E 추정 근거)
- `notes/PB_0.6822 코드공유.ipynb` (원본 framework)
- `notes/코드공유-upgrade.md` (plan-005 통찰 + upgrade idea)
- `analysis/plan-005/component_contribution.{json,md}` (Variant B 측정값)
- `analysis/plan-005/results.md` (plan-005 종합 결과)
- `WORKFLOW.md` §0.5, §11, §12 convention
- `CLAUDE.md` (autonomous execution policy)
- `src/pb_0_6822/{selector,boundary}.py` (plan-004 lock-in, import only)
- `src/submit.py` (dacon-submit infra)
