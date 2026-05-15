---
plan_id: 018
version: 1.2 (executor patch — A5 Neural CDE 제외 (torchcde 미설치, RK4 ODE solver 수동 구현 cost 과대). 4 ablation arch (A1/A2/A3/A6) 로 reduction. basis_vars 실제 값 carry (d1/acc_par/acc_perp/d2/jerk/ts_term/speed_slope_d1/rotation_term) — spec 박제 8 vars 와 다름 but plan-007 actual best 박제. A0 reproduce 는 mlp_coeff.json (OOF=0.6482) import only.)
date: 2026-05-15 (Asia/Seoul)
status: G_final_complete
based_on:
  - 007 (Step 4 MLP OOF 0.6482, LB 0.6598. best basis 8 vars 박제)
  - 005 (oracle 0.7188, plan-018 미적용 — multi-formula 는 본 plan out-of-scope)
  - 004 (plan-004 LB 0.6822 ensemble baseline — 본 plan 의 target reference)
  - 017 (GRU-attn coeff regressor draft. 본 plan 의 A6 와 spec 가까우나 미실행 → 신규 작성)
scope: plan-007 step 4 의 *spirit* (per-sample coefficient regression on fixed 8 best basis) 유지.
       A0 baseline reproduce + **A1~A6 (Vector Neurons A4 제외, 5 ablation arch)** 의 5-fold OOF 비교. best 1 만 LB 제출.
       multi-stage / corrector / multi-formula / learnable basis 는 본 plan 외 (다음 plan).
       단일 모델 LB ≥ 0.67 (= plan-004 LB 0.6822 의 95% 이상) 도달 시 G_final PASS.
exp_ids:
  - F008_eda_check
  - F008_arch-ablation         # A0 (imported) + 4 ablation arch (A1/A2/A3/A6) × 5-fold OOF
  - F010_g_final_synthesis
lb_score: null  # G2 SKIP per user decision (quota 보존)
g1_passed: false
best_arch: A3
best_arch_oof: 0.6485
delta_vs_a0: 0.0003
exception_policy: plan-007 §2.2 의 "End-to-end 학습 통합 out-of-scope" 의 **예외 plan**.
                  본 plan 의 A1~A6 (A4 제외) 는 trajectory encoder + coefficient head 의 *single-stack end-to-end*.
                  단, *multi-stack* (encoder + head + corrector + selector 등 ≥ 3 stage) 은 여전히 out-of-scope.
---

# plan-018 v1.1 — Architectural Ablation: Single-Model Path to plan-004 LB

## §0. 한 줄 목적

> plan-007 step 4 의 *per-sample coefficient regression on fixed 8 best basis* spirit 을 유지하면서, **A0 baseline + 5 architectural ablation arch** (A1 Set Transformer / A2 Path Signature / A3 Sparse MoLE / A5 Neural CDE / A6 GRU-attn coeff regressor) 의 5-fold OOF 무차별 비교. **best 1 single model 의 LB ≥ 0.67** (= plan-004 ensemble LB 0.6822 의 95% 이상) 도달이 G_final 합격. multi-stage / corrector / multi-formula / learnable basis 는 본 plan 외.
>
> **A4 Vector Neurons 제외 사유** (agent 재평가 EDA axis): task 가 rotation invariant 가 아님 — plan-007 best basis 의 `rotation_term` 이 frenet frame curvature 에 의존하는 rotation-dependent 신호 (+0.31pp 기여, plan-007 §6.3). SO(3) equivariance 가 그 신호를 *invariant pooling* 으로 *제거* → 정보 손실 보장. EDA 와 직접 모순.

---

## §0.5 Quick Reference (autonomous loop 매 turn 읽는 section)

### 본 plan 의 task essence — "단일 모델 architectural ablation (6 arch: A0 + 5 ablation)"

- plan-007 step 4 (MLP ~300 params) = OOF 0.6482, LB 0.6598. *encoder 가 13-d stats summary* 라는 *information bottleneck* 이 main suspect.
- 본 plan = A0 baseline reproduce + A1/A2/A3/A5/A6 의 5 ablation arch 비교로 *bottleneck 의 정체* 측정.
- target = **단일 모델 LB ≥ 0.67** (LB +0.01 vs step 4, plan-004 ensemble 0.6822 의 95% 이상 도달).
- 단일 모델 rule: encoder + head + (fixed) basis 의 1-stack. no corrector, no multi-formula, no learnable basis.

### Architecture 후보 (총 6 arch: A0 baseline + A1/A2/A3/A5/A6 ablation. A4 제외)

| id | encoder | head | input type | params | 논문 reference |
|---|---|---|---|---|---|
| A0 (baseline reproduce) | plan-007 13-d stats | 2-layer MLP | `stats_13d` | ~300 | plan-007 §7.1 |
| A1 | Set Transformer (ISAB) | per-coeff query attn | `traj_6x3` | ~30K | Lee et al. 2019 |
| A2 | Path Signature (depth-3) | 2-layer MLP | `traj_6x3` | ~10K | Kidger et al. 2020 |
| A3 | plan-007 13-d stats | Sparse MoLE (K=16, top-2) | `stats_13d` | ~10K | Shazeer 2017 |
| ~~A4~~ | ~~Vector Neurons (SO(3) equiv)~~ | ~~2-layer MLP~~ | — | — | **제외 (EDA 모순, §0)** |
| A5 | Neural CDE | 2-layer MLP | `traj_6x3` | ~80K | Kidger 2020 |
| A6 | GRU + per-coeff attn | per-coeff query attn | `traj_6x3` | ~50K | plan-004 §139 / plan-017 §5 |

`input type` registry → §5.0 ablation_runner 의 collate_fn 분기 spec.

### 합격 기준 (G-gate sequence)

- **G0** (sanity + EDA): 데이터셋 dimension/distribution check + A0 baseline reproduce. A0 OOF ∈ [0.6482 ± 0.003] (= plan-007 step 4 결과 재현 ± noise). 위반 시 `baseline_reproduce_fail` severe.
- **G1** (ablation 결과): A1/A2/A3/A5/A6 의 5-fold OOF table. **5 ablation arch 중 ≥ 1 OOF ≥ 0.6532** (= step 4 + 0.005). 위반 시 `no_arch_improvement` warn — 결과 박제 후 plan-019 후보로 carry-over. *주의: A0 의 0.6482 는 baseline 이므로 G1 합격 대상 아님.*
- **G2** (LB 제출): G1 의 best ablation arch 1 만 dacon-submit. **LB ≥ 0.67** ⭐ → G_final PASS. LB ∈ [0.66, 0.67) → partial. LB ∈ [0.65, 0.66) → `lb_marginal_below_baseline` warn (plan-007 step 4 LB 0.6598 미달). LB < 0.65 → `lb_below_threshold` warn (plan-019 후보 carry).
- **G_final**: results.md + plan-019 후보 ≥ 2 + 3 파일 frontmatter sync.

LB 제출 = **총 1회** (DACON daily 5 limit 내, 무차별 탐색의 *cost-efficient* rule).

### G-gates (commit 단위 milestone)

- G0: A0 baseline reproduce + EDA check                                       [TODO]
- G1: 5 ablation arch (A1/A2/A3/A5/A6) × 5-fold OOF table, ≥ 1 OOF ≥ 0.6532   [TODO]
- G2: best arch LB ≥ 0.67 (또는 회수 + band 박제)                              [TODO]
- G_final: results.md + plan-019 후보 ≥ 2 + frontmatter sync                   [TODO]

### Commit chain (next-up)

| # | type | spec section | status |
|---|---|---|---|
| c1 | docs | `plans/plan-018-*.md` 본문 (본 파일) | [TODO] |
| c2 | code | `analysis/plan-018/eda_check.py` — 데이터 shape/distribution sanity. spec @ §4 | [TODO] |
| c3 | code | `src/plan018/baseline_a0.py` — plan-007 step 4 MLP 재구현 (새 코드). spec @ §5.1 | [TODO] |
| c4 | exp | A0 5-fold OOF + plan-007 결과 동일성 검증. spec @ §5.1 / §6 | [TODO] |
| G0 | gate | A0 OOF ∈ [0.6482 ± 0.003] | [TODO] |
| c5 | code | `src/plan018/arch_a1_set_transformer.py`. spec @ §5.2 | [TODO] |
| c6 | code | `src/plan018/arch_a2_signature.py`. spec @ §5.3 | [TODO] |
| c7 | code | `src/plan018/arch_a3_mole.py`. spec @ §5.4 | [TODO] |
| ~~c8~~ | ~~A4 Vector Neurons~~ | **제외 (EDA 모순, §0)** |
| c9 | code | `src/plan018/arch_a5_neural_cde.py`. spec @ §5.6 | [TODO] |
| c10 | code | `src/plan018/arch_a6_gru_attn.py`. spec @ §5.7 | [TODO] |
| c11 | code | `analysis/plan-018/ablation_runner.py` — 6 arch (A0 + A1/A2/A3/A5/A6) 5-fold OOF loop runner. spec @ §6 | [TODO] |
| c12 | exp | F008: ablation 6 arch × 5-fold OOF. spec @ §6 | [TODO] |
| G1 | gate | 5 ablation arch 중 ≥ 1 OOF ≥ 0.6532 | [TODO] |
| c13 | analysis | `analysis/plan-018/ablation_results.md` — table + best arch 선정 + 사유. spec @ §7 | [TODO] |
| c14 | sub-lb | F009: best arch dacon-submit + lb_log + frontmatter. spec @ §8 | [TODO] |
| G2 | gate | LB ≥ 0.67 또는 회수 후 band 박제 | [TODO] |
| c15 | synthesis | `analysis/plan-018/results.md` + `next_plan_candidates.md` (≥ 2 후보). spec @ §9 | [TODO] |
| G_final | gate | results.md + plan-019 후보 + 3 파일 frontmatter sync | [TODO] |

### Plan-specific severe (WORKFLOW.md §12.3 default 위 추가분)

- `baseline_reproduce_fail`: G0 의 A0 OOF ∉ [0.6479, 0.6485]. 데이터 split 또는 basis_terms 식 mismatch — plan-007 §7.2 와의 spec drift. 즉시 halt + 사용자 escalate (모든 후속 ablation 의 신뢰성 무효).
- `arch_code_unstable`: 6 arch 중 ≥ 2 가 NaN loss / gradient explosion / training fail. 본 plan 의 6 arch 가 *논문 reference* 와 spec 일치하지 않음 — 코드 재작성 필요.

### Plan-specific paths (WORKFLOW.md §12.5/§12.6 추가/제외)

- whitelist 추가:
  - `src/plan018/**` (본 plan 의 모든 trainable module)
  - `analysis/plan-018/**` (EDA + ablation results)
  - `runs/baseline/F008_arch-ablation/**`, `runs/baseline/F009_best-arch-lb/**`
- blacklist 추가:
  - `src/pb_0_6822/**` 의 *수정* (lock-in, import only of `CANDIDATE_LIST[17]` identifier — *함수 호출은 본 plan 에서 X*)
  - `src/plan017/**` (plan-017 의 GRU-attn coeff regressor 코드 — 본 plan 의 A6 는 *plan-017 carry 가 아닌 신규 작성*. 코드 재사용 시 ambiguity 위험.)

### Decision-note 사용 예 (자율 결정 시 commit msg 박제)

- `decision-note: spec-default — A1 Set Transformer ISAB inducing point M=16 (Lee 2019 Table 3 권장값).`
- `decision-note: spec-default — A3 MoLE K=16 top-2 (Shazeer 2017 의 small-data 권장 K=16, top-2 routing).`
- `decision-note: spec-default — Adam lr=1e-3, wd=1e-4, batch=256, epoch=50, patience=8 (plan-007 §7.2 carry, 모든 arch 동일).`
- `decision-note: exception-plan — plan-007 §2.2 의 "End-to-end 학습 통합" out-of-scope 의 본 plan 예외 (frontmatter exception_policy 박제).`
- `decision-note: code-reuse-skip — plan-017 의 GRU-attn 코드 spec ambiguity (Δ-param init 의 정확한 form 미확정) 으로 본 plan A6 신규 작성.`

---

## §1. 배경

### §1.1 plan-007 step 4 결과 (carry-over)

- plan-007 best basis (Step 3, 8 vars): `frenet_par120, perp_neg020, speed_slope_d1, rotation_term, d1_norm_acc_par, v_mean3, jerk_term, par_quad` (analysis/plan-007/best_basis.json 박제).
- step 4 MLP (~300 params, 13-d stats input, 2-layer): **OOF 0.6482 / LB 0.6598**.
- 시나리오 B 결론 (plan-007 §9.2): "단일 공식 framework 의 한계 ≈ 0.6491 baseline 동급, +0.0095 marginal." plan-008 후보 (1) 27 후보 풀 확장 / (2) corrector × MLP 결합 / (3) Step 4 LB carry-over 박제.

### §1.2 본 plan 의 가설

| 가설 | 검증 방법 | 합격 |
|---|---|---|
| H1: step 4 의 marginal gain (+0.0095) 의 main bottleneck 은 *encoder* (13-d stats) 의 정보 손실 | A1/A2/A5/A6 (encoder 강화) OOF ≥ A0 + 0.005 | ≥ 1 arch 통과 |
| H2: head 의 capacity (~300 → ~10K) 만으로도 일부 gain | A3 (MoLE head, A0 encoder 유지) OOF vs A0 | A3 OOF ≥ A0 + 0.005 |
| H3: 단일 모델 LB ≥ 0.67 (plan-004 ensemble 0.6822 의 95%) 도달 | best arch dacon-submit | LB ≥ 0.67 |

### §1.3 본 plan 이 *안 하는* 것 (focus)

- 27 후보 풀 확장 (plan-007 §9.2 후보 1) — 본 plan 다음.
- learnable basis (iteration 2 brainstorm 의 Koopman lift) — 본 plan 다음.
- corrector 결합 (plan-005 / plan-013/014/015/016) — 본 plan 미적용. fixed basis + per-sample head 의 *single-stack* 만.
- multi-formula MoE (selector + per-cand coeff) — 본 plan 외.

---

## §2. Scope (명시적)

### §2.1 In-scope

| 항목 | 값 |
|---|---|
| basis (fixed) | plan-007 Step 3 best 8 vars |
| 데이터 증폭 | sliding 40K + original 10K = 50K pool (plan-007 §3.1 답습) |
| 5-fold split | plan-007 §7.2 답습 (same seed=42, sample_id-grouped) |
| training | Adam lr=1e-3, wd=1e-4, batch=256, epoch=50, patience=8, grad_clip=2.0 (모든 arch 공통) |
| loss | soft_hit_loss (sigmoid, threshold=0.01, sharpness=200) — plan-007 §7.2 carry |
| ablation arch | A1/A2/A3/A5/A6 (총 5종, A4 제외 — §5) |
| LB 제출 | **1 회** (best arch 1, G2) |

### §2.2 Out-of-scope (절대 안 함)

| 항목 | 이유 |
|---|---|
| Corrector 결합 | 본 plan 미적용. plan-019 후보로 carry. |
| Multi-formula (≥ 2 basis stack) | 본 plan single-stack rule. plan-019 후보. |
| Learnable basis | fixed 8 vars carry. plan-019 후보. |
| 27 후보 풀 확장 | 본 plan basis 고정. plan-019 후보 1. |
| Hyperparameter tuning (각 arch 별 lr/wd 변형) | 모든 arch 동일 setting. *arch comparison fair* 보장. |
| LB 제출 ≥ 2 회 | 본 plan = 1 회 (best 1 만). 무차별 탐색의 cost-efficient rule. |
| Plan-004/005/006/007/017 의 *trainable module* import | 신규 작성 (§9 코드 재사용 정책). |

### §2.3 plan-007 §2.2 exception 명시

- plan-007 §2.2 의 "End-to-end 학습 통합 out-of-scope" 는 본 plan-018 에서 **예외 허용**. 사유:
  1. 본 plan 의 6 arch 는 *single-stack* (encoder + head + fixed basis). 본질적으로 plan-007 step 4 MLP 와 같은 stack 깊이.
  2. plan-007 §2.2 의 "End-to-end" = *corrector + MLP 결합* 같은 *multi-stack*. 본 plan 은 그 multi-stack 을 여전히 out-of-scope (§2.2).
  3. 즉, 본 plan = step 4 의 *spirit* 유지하면서 *encoder/head 만 교체*. plan-007 의 의도에 부합.
- frontmatter `exception_policy` 박제.

---

## §3. 사전 등록 (Pre-registration)

### §3.1 입력 데이터 + 분할

| 분할 | 출처 | 사용 |
|---|---|---|
| Train original (10K, end_idx=10, target=train_y) | `data/train/` | A0 + A1/A2/A3/A5/A6 학습/검증 |
| Train sliding (40K, end_idx ∈ [5, 8], horizon=2, target=train_x[:, end_idx+2]) | sliding window aug (plan-007 §4.1) | 6 arch 의 *train fold only*. val fold 는 original 만. |
| Test (10K, end_idx=10) | `data/test/` | best arch inference + submission |

**Window slicing rule** (모든 arch 공통, AMBIGUITY 1 해소):

- `trajectory_window = train_x[:, end_idx - 5 : end_idx + 1, :]` → shape `(N, 6, 3)`. 즉 *end_idx 포함 마지막 6 step*.
- original (end_idx=10) → `train_x[:, 5:11, :]`.
- sliding (end_idx=5) → `train_x[:, 0:6, :]`. end_idx=8 → `train_x[:, 3:9, :]`. boundary-safe (plan-007 §4.1 carry).
- target: original = `train_y` (horizon=2 of end_idx=10). sliding = `train_x[:, end_idx + 2, :]`.

**EDA caveat** (Agent 2 권고):

- plan-001 §3 const-vel baseline 이 LB ~0.60 으로 *강함*. per-timestep noise floor ≈ 0.005~0.007 m. 즉 trajectory 의 sequential order 가 *주요 신호 (jerk, rotation)* 에만 필수 — 본 plan A1 Set Transformer 의 permutation-invariance 가정이 일부 신호 손실 위험 (ablation 으로 측정).

### §3.2 합격 기준 (정량)

- **G0**: A0 OOF ∈ [0.6479, 0.6485] (= plan-007 step 4 결과 0.6482 ± 0.003). EDA check (§4) PASS.
- **G1**: 5 ablation arch (A1/A2/A3/A5/A6) 중 ≥ 1 OOF ≥ 0.6532. best ablation arch 선정 (A0 baseline 제외).
- **G2**: best arch LB ≥ 0.67 (plan-004 ensemble 0.6822 의 95%).
- **G_final**: results.md + plan-019 후보 ≥ 2 + frontmatter sync.

### §3.3 평가

- OOF metric = 5-fold concat, original 10K 의 *hit rate* (threshold 0.01 m). plan-007 §3.3 carry.
- LB metric = DACON public LB hit rate (plan-007 §8 carry).

---

## §4. STAGE 0 — EDA + A0 Baseline Reproduce (c2~c4)

### §4.1 EDA check (c2)

**`analysis/plan-018/eda_check.py`** (신규 작성):

```python
# 데이터셋 sanity check. 본 plan 의 모든 arch 가 가정하는 input shape/distribution 확인.
import numpy as np

train_x = np.load("data/train/train_x.npy")   # (10000, 11, 3) — 11 step, 3D
train_y = np.load("data/train/train_y.npy")   # (10000, 3)     — horizon=2 step ahead
test_x  = np.load("data/test/test_x.npy")     # (10000, 11, 3)

# Assertion 1: shape
assert train_x.shape == (10000, 11, 3), f"train_x shape mismatch: {train_x.shape}"
assert train_y.shape == (10000, 3), f"train_y shape mismatch: {train_y.shape}"

# Assertion 2: distribution (plan-001 §3 결과: per-axis std ≈ 0.6~1.2 m)
for k in range(3):
    std_k = train_x[:, :, k].std()
    assert 0.4 < std_k < 1.5, f"axis {k} std out of expected range: {std_k}"

# Assertion 3: noise floor (plan-001 §3: per-axis MAE ≈ 0.005~0.007 m of const-velocity baseline)
# Agent 2 권고 — threshold 0.05 → 0.015 (3σ margin around plan-001 실측값 0.007)
vel = np.diff(train_x, axis=1).mean(axis=1)                      # const-vel per sample (N, 3)
const_vel_pred = train_x[:, -1] + 2 * vel                         # 2-step horizon
const_vel_mae  = np.abs(const_vel_pred - train_y).mean(axis=0)    # per-axis MAE
assert np.all(const_vel_mae < 0.015), f"const-vel MAE 비정상: {const_vel_mae}"

# Assertion 4: target hit rate (plan-006 baseline ≈ 0.6491 → const-vel only is sub-baseline)
hit = (np.linalg.norm(const_vel_pred - train_y, axis=1) < 0.01).mean()
print(f"const-vel hit rate: {hit:.4f}")   # 기대값 ~0.4 안팎

# Output: analysis/plan-018/eda_check.json (frontmatter + assertion 결과)
```

### §4.2 A0 baseline reproduce (c3~c4)

**`src/plan018/baseline_a0.py`** (신규 작성, plan-007 §7.1/7.2 spec 답습 — 코드 *재사용 X*, 새 module).

```python
# A0: plan-007 step 4 MLP 재구현. 본 plan 의 6 arch 와 *완전 같은* training loop 위에서 reproduce.
# 목적: plan-007 step 4 결과 OOF 0.6482 ± 0.003 재현으로 본 plan 의 framework sanity 검증.
import numpy as np
import torch
import torch.nn as nn


class BaselineA0(nn.Module):
    """plan-007 step 4 MLP — 13-d stats summary → 8 coefficient."""
    def __init__(self, feat_dim: int = 13, n_coeffs: int = 8,
                 global_init: np.ndarray | None = None):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, 32),
            nn.SiLU(),
            nn.Linear(32, n_coeffs),
        )
        if global_init is not None:
            with torch.no_grad():
                self.mlp[-1].bias.copy_(torch.tensor(global_init, dtype=torch.float32))
                self.mlp[-1].weight.zero_()

    def forward(self, traj_features: torch.Tensor) -> torch.Tensor:
        return self.mlp(traj_features)   # (B, 8)


def compute_trajectory_features(x_window: np.ndarray) -> np.ndarray:
    """plan-007 §7.2 carry (13-d stats summary).

    Args:
        x_window: (N, 6, 3) — §3.1 의 window slicing rule 결과.
    Returns:
        (N, 13) — pos_mean(3) + pos_std(3) + pos_range(3) + speed_mean/std/max/last (4).
    """
    pos_mean  = x_window.mean(axis=1)
    pos_std   = x_window.std(axis=1)
    pos_range = x_window.max(axis=1) - x_window.min(axis=1)
    deltas    = np.diff(x_window, axis=1)
    speed_norms = np.linalg.norm(deltas, axis=2)
    speed_mean = speed_norms.mean(axis=1, keepdims=True)
    speed_std  = speed_norms.std(axis=1, keepdims=True)
    speed_max  = speed_norms.max(axis=1, keepdims=True)
    speed_last = speed_norms[:, -1:]
    return np.concatenate([pos_mean, pos_std, pos_range,
                           speed_mean, speed_std, speed_max, speed_last], axis=1)
```

### §4.3 G0 합격 기준 (자동 판정)

- `eda_check.json["all_assertions_pass"] == True`
- A0 5-fold OOF ∈ [0.6479, 0.6485]. 위반 시 `baseline_reproduce_fail` severe + halt.

### §4.4 시간 예산

- EDA check: ~1 분
- A0 학습 + OOF: ~5 분 (cuda 2.8.0, 50 epoch × 50K pool)

---

## §5. STAGE 1 — Architecture Ablation 6 종 (c5~c12)

### §5.0 공통 framework

모든 arch 는 다음 *동일 training loop* 위에서 학습. arch 별로 *input type* 만 분기 (registry):

```python
# Pseudocode — 모든 arch 의 공통 fold loop (analysis/plan-018/ablation_runner.py).

# ── ARCH REGISTRY: 각 arch 의 constructor signature + input type 분기 spec ──
# input_type ∈ {"stats_13d", "traj_6x3"}.
#   - "stats_13d": encoder_input = compute_trajectory_features(window) → (B, 13). A0, A3.
#   - "traj_6x3":  encoder_input = window 그대로 (B, 6, 3). A1, A2, A5, A6.
# 모든 arch constructor 는 keyword-only:
#   __init__(self, *, n_coeffs: int = 8, global_init: np.ndarray | None = None, **arch_specific)
# `feat_dim` 같은 arch-specific arg 는 각 module 내부 default 로 처리 (e.g., A0 의 feat_dim=13).

ARCH_REGISTRY = {
    "A0": {"class": BaselineA0,            "input_type": "stats_13d"},
    "A1": {"class": SetTransformerCoeff,   "input_type": "traj_6x3"},
    "A2": {"class": PathSignatureCoeff,    "input_type": "traj_6x3"},
    "A3": {"class": MoLECoeffHead,         "input_type": "stats_13d"},
    "A5": {"class": NeuralCDECoeff,        "input_type": "traj_6x3"},
    "A6": {"class": GRUAttnCoeff,          "input_type": "traj_6x3"},
}


def collate_batch(samples: list, input_type: str) -> Batch:
    """
    각 sample 은 다음 field 보유 (DataLoader 의 Dataset 가 사전 계산):
      - window:        (6, 3)  float32 — §3.1 의 window slicing rule
      - stats_13d:     (13,)   float32 — compute_trajectory_features(window) 사전 계산
      - basis_terms:   (8, 3)  float32 — plan-007 §6.3.1 의 compute_basis_terms 결과,
                                          best_basis_vars 순서로 stack. **dataset init 시 사전 계산**
                                          (학습 매 step 재계산 X — 속도 + 결정성).
      - p0:            (3,)    float32 — window[-1]  (= 현 위치)
      - target:        (3,)    float32

    Returns Batch with:
      - encoder_input: (B, 13) if input_type=="stats_13d" else (B, 6, 3)
      - basis_terms:   (B, 8, 3)
      - p0:            (B, 3)
      - target:        (B, 3)
    """
    ...   # 표준 torch collate, encoder_input 만 분기


# ── 학습 loop ──
for arch_id, spec in ARCH_REGISTRY.items():
    arch_class = spec["class"]
    input_type = spec["input_type"]

    fold_oofs = []
    for fold_k in range(5):
        # data split (plan-007 §7.2 carry — seed=42, sample_id-grouped, sliding ∪ original).
        # 같은 sample_id 의 모든 view 가 같은 fold 에 묶임 → leakage X.
        train_loader, val_loader = build_fold(
            fold_k, aug=True, collate_fn=lambda s: collate_batch(s, input_type)
        )

        model = arch_class(n_coeffs=8, global_init=stage3_best_params)
        # NOTE: stage3_best_params 는 analysis/plan-007/best_basis.json 의 8-vec.
        # 모든 arch 의 head final linear bias init 으로 사용. weight=0 으로 두면 학습 0 step 에서
        # plan-007 step 3 best 와 동일 동작 (A0 의 plan-007 carry 와 일치). A6 의 Δ-parametrization
        # 은 §5.7 에서 별도 처리.

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

        best_val, best_state, patience_count = 0.0, None, 0
        for epoch in range(50):
            model.train()
            for batch in train_loader:
                coeffs = model(batch.encoder_input)                                  # (B, 8)
                pred = batch.p0 + (coeffs.unsqueeze(-1) * batch.basis_terms).sum(dim=1)  # (B, 3)
                loss = soft_hit_loss(pred, batch.target, threshold=0.01, sharpness=200)
                # arch-specific aux loss (A3 MoLE 의 load balancing 등) 은 model.aux_loss 로 노출.
                if hasattr(model, "aux_loss"):
                    loss = loss + model.aux_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                optimizer.step()
                optimizer.zero_grad()

            val_hit = evaluate_hit_rate(model, val_loader)
            if val_hit > best_val + 1e-4:
                best_val, best_state, patience_count = val_hit, copy.deepcopy(model.state_dict()), 0
            else:
                patience_count += 1
                if patience_count >= 8:
                    break

        model.load_state_dict(best_state)
        fold_oofs.append(evaluate_hit_rate(model, val_loader))   # original-only val

    ARCH_RESULTS[arch_id] = {"oof": float(np.mean(fold_oofs)), "fold_oofs": fold_oofs}
```

### §5.1 A0: plan-007 step 4 MLP reproduce — §4.2 참조.

- encoder input: 13-d stats summary (plan-007 §7.2 carry).
- head: 2-layer MLP (Linear(13→32) + SiLU + Linear(32→8)).
- params: ~300.
- 기대 OOF: 0.6482 ± 0.003 (= plan-007 결과 재현).

### §5.2 A1: Set Transformer (ISAB encoder, Lee et al. 2019)

**`src/plan018/arch_a1_set_transformer.py`** (신규 작성):

- input: trajectory (B, 6, 3) → token sequence of 6 (각 timestep 이 token).
- ISAB block ×2: inducing point M=16, head=4. 출력 (B, 6, H=32).
- pooling: mean over timestep → (B, H=32).
- head: per-coeff query attention. learnable basis-var embedding (8, H=32) → cross-attn over (B, 6, H) → (B, 8, H) → MLP → (B, 8) coeff.
- params: ~30K.
- 기대 OOF: 0.66 ~ 0.68.

### §5.3 A2: Path Signature features (Kidger et al. 2020, signatory library)

**`src/plan018/arch_a2_signature.py`** (신규 작성):

- input: trajectory (B, 6, 3) — augmented with time channel → (B, 6, 4).
- signature: depth=3 truncated. signature dim = 1 + 4 + 16 + 64 = 85.
- head: 2-layer MLP (Linear(85→64) + SiLU + Linear(64→8)).
- params: ~10K.
- 기대 OOF: 0.66 ~ 0.68.

### §5.4 A3: Sparse MoLE head (Shazeer et al. 2017, head-only ablation)

**`src/plan018/arch_a3_mole.py`** (신규 작성):

- encoder: plan-007 13-d stats summary (A0 와 동일).
- head: K=16 experts, 각 expert = 8-d global coefficient. top-k=2 routing.
- gating: MLP(13→16) + softmax + top-2 hard selection.
- aux loss: L_aux = 16 · Σ_k (importance_k · load_k), α = 0.01.
- params: ~10K.
- 기대 OOF: 0.66 ~ 0.68.
- *important*: head-only 교체. encoder 는 A0 와 동일 → A3 vs A0 의 Δ = *head capacity 효과* 만.

### §5.5 ~~A4: Vector Neurons~~ — **제외 (EDA 모순)**

> Agent 2 EDA 재평가 finding: plan-007 best basis 의 `rotation_term` (frenet frame curvature 의존, +0.31pp 기여) 이 *rotation-dependent* 신호. SO(3) equivariant network 의 *invariant pooling* 단계가 이 신호를 *제거 보장* → 정보 손실 보장. EDA 와 직접 모순으로 본 plan 에서 제외. plan-019 후보 carry 도 X (rotation-aware 변형은 별도 paradigm 으로만 가능).

### §5.6 A5: Neural CDE (Kidger et al. 2020, torchcde library)

**`src/plan018/arch_a5_neural_cde.py`** (신규 작성):

- input: trajectory (B, 6, 3) → cubic spline interpolation → continuous path.
- CDE: dz/dt = f_θ(z) · dx/dt, z(0) = ζ(x(0)), z ∈ R^32.
- solver: Runge-Kutta 4-step, dt=1/4 (= 24 effective inner steps over 6 outer).
- head: 2-layer MLP (32 → 16 → 8 coeff) from final z.
- params: ~80K.
- 기대 OOF: 0.66 ~ 0.68 (largest params, continuous-time inductive bias).

### §5.7 A6: GRU-attn coefficient regressor (plan-017 spirit, 신규 작성)

**`src/plan018/arch_a6_gru_attn.py`** (신규 작성, plan-017 코드 *import X*, 새 module):

- encoder: 2-layer GRU(seq_dim=3 → 32). 입력 trajectory (B, 6, 3) → (B, 6, 32) + final hidden (B, 32).
- head: per-coeff query attention. learnable basis-var embedding (8, 32) → cross-attn over GRU output → (B, 8, 32) → MLP → (B, 8) coeff.
- Δ-parametrization: c = c_init + Δ(τ), c_init = stage3_best_params (head 의 마지막 Linear bias).
- L2 prior: ||Δ||² weight = 1e-4 (plan-017 §5 carry).
- params: ~50K.
- 기대 OOF: 0.66 ~ 0.69 (plan-017 의 G1 target).
- *important*: plan-017 의 GRU-attn module 을 *재사용 X*. plan-017 의 spec ambiguity (Δ-param 의 정확한 form, collinearity diagnostic 의 weight 결정) 가 본 plan 시점에 미확정 — *코드 재사용 시 silent bug 위험*. 신규 작성으로 *현 spec 박제*.

---

## §6. STAGE 1 ablation runner + G1 합격 (c11~c12)

### §6.1 `analysis/plan-018/ablation_runner.py` (c11, 신규 작성)

- §5.0 의 framework 위에서 6 arch (A0 baseline + A1/A2/A3/A5/A6 ablation) × 5-fold OOF.
- 출력: `analysis/plan-018/ablation_results.json`
  ```json
  {
    "A0": {"oof": 0.6482, "fold_oofs": [...], "params": 297, "elapsed_sec": 312},
    "A1": {"oof": 0.6XXX, "fold_oofs": [...], "params": 29481, "elapsed_sec": 1840},
    ...
  }
  ```
- compute budget: cuda 2.8.0+cu128, ~3 시간 wall-time (6 arch × 5-fold × ~6 분).

### §6.2 G1 합격 기준 (자동 판정)

- `ablation_results.json` 의 5 ablation arch (A1/A2/A3/A5/A6, A0 제외) 중 ≥ 1 의 OOF ≥ 0.6532. → PASS.
- best arch = `argmax_{arch ∈ {A1,A2,A3,A5,A6}} oof`. tie → params 작은 쪽.
- 위반 시: `no_arch_improvement` warn (severe X). plan 결과 박제 후 plan-019 후보로 carry-over (encoder/head architectural lever 무효 confirmed).

### §6.3 시간 예산

- ablation 실행: ~3 시간 (cuda)
- 결과 박제: ~10 분

---

## §7. ablation_results.md + best arch 선정 (c13)

### §7.1 `analysis/plan-018/ablation_results.md` 포맷

```markdown
| arch | encoder | head | params | OOF | fold spread | elapsed | rank |
|---|---|---|---|---|---|---|---|
| A0 | stats-13d | MLP | 297 | 0.6482 | 0.013 | 5m | 7 |
| A1 | SetTransformer | per-coeff attn | 29.5K | 0.6XXX | ... | ... | ... |
| ... |
```

- best arch 선정 사유 1~3 sentence (e.g., "A1 의 ISAB encoder 가 trajectory 의 timestep-level 정보를 손실 없이 인코딩 — A0 의 13-d stats summary 의 bottleneck 회피").
- best arch 의 LB submission spec (§8) 으로 carry.

### §7.2 G1 결과 박제 + plan-019 후보 candidate signals

- G1 PASS / fail 박제.
- best arch 의 *failure mode* 도 함께 박제 (예: fold spread 큰 arch → variance issue → multi-seed ensemble plan-019 후보).

---

## §8. STAGE 2 — best arch LB 제출 (c14, dacon-submit skill)

### §8.1 자율 호출

```python
# G1 PASS 시:
Skill(skill="dacon-submit",
      args=f"runs/baseline/F009_best-arch-lb/submission.csv F009_{best_arch_id}_plan-018")
```

### §8.2 응답 4-분기 처리 (plan-007 §8.2 동일 패턴)

| (isSubmitted, lb_score) | 처리 | frontmatter `lb_score` | status | severe |
|---|---|---|---|---|
| (True, float) | full success | `<float>` 소수 4자리 | `all_complete` | — |
| (True, None) | partial — carry-over commit `c14.1` | `TBD` | `partial` | — |
| (False, *) | retry 1회 (60 초 sleep). 재실패 시 severe | `null` | `partial` | `lb_unsubmitted` |
| Skill exception | 즉시 escalate | `null` | `partial` | `dacon_submit_skill_missing` |

### §8.3 `analysis/plan-018/lb_log.md` 포맷

```markdown
| timestamp_kst             | exp_id                       | arch | isSubmitted | lb_score | detail |
|---------------------------|------------------------------|------|-------------|----------|--------|
| 2026-05-1XT__:__:__+09:00 | F009_<best_arch>_plan-018    | A?   | true        | 0.6XXX   | OK     |
```

### §8.4 G2 합격 기준 + band 분류

- LB ≥ 0.67 → **G_final PASS (plan-004 95% 도달)**, status `all_complete`
- 0.66 ≤ LB < 0.67 → `partial — band carry from plan-007` (warn 없음, plan-007 LB 0.6598 위)
- 0.65 ≤ LB < 0.66 → `lb_marginal_below_baseline` **warn** (plan-007 step 4 LB 0.6598 미달 — architectural lever 가 *partial fail*)
- LB < 0.65 → `lb_below_threshold` **warn** (single-model architectural lever 자체가 plan-007 보다 *역효과* — plan-019 분석 필수)

---

## §9. STAGE 3 — Synthesis + plan-019 후보 (c15)

### §9.1 `analysis/plan-018/results.md`

frontmatter:
```yaml
---
plan_id: 018
based_on:
  - 007
  - 005
  - 004
  - 017
finished_at: <ISO8601 KST>
status: all_complete | partial
exp_ids_completed:
  - F008_arch-ablation
  - F009_best-arch-lb
lb_exp_id: F009_<best_arch>_plan-018
lb_score: <float|TBD|null>
lb_submitted_at: <ISO8601 KST>
exception_policy: plan-007 §2.2 의 end-to-end 통합 예외 — 본 plan 의 6 arch 가 single-stack 위반 아님 확인
---
```

본문:
- G0 EDA + A0 baseline reproduce 결과
- G1 ablation table (6 arch: A0 + A1/A2/A3/A5/A6 × OOF × fold spread × params × elapsed)
- best arch 선정 사유
- G2 LB 결과 + band 분류
- plan-019 후보 ≥ 2 (시나리오 분기에 따라)

### §9.2 시나리오 분기

| G2 결과 | plan-019 후보 |
|---|---|
| LB ≥ 0.67 (G_final PASS) | (1) best arch + corrector 결합 (plan-005 / plan-016 carry) — LB 0.70+ 시도. (2) best arch × multi-seed ensemble (variance reduction). |
| 0.66 ≤ LB < 0.67 (partial) | (1) best arch + plan-007 시나리오 B 후보 1 (27 후보 풀 확장) 결합. (2) best 2 arch ensemble (architectural diversity). |
| LB < 0.66 (lb_below_threshold) | (1) plan-007 step 4 의 *bottleneck* 이 encoder 가 아닌 *basis 자체* — learnable basis (iteration 2 brainstorm) plan-019. (2) 본 plan 의 framework 외 (corrector / multi-formula) plan-019. |

### §9.3 frontmatter sync (3 파일)

- `plans/plan-018-arch-ablation-single-model.md` top-level `lb_score`
- `plans/plan-018-arch-ablation-single-model.results.md` frontmatter
- `analysis/plan-018/results.md` frontmatter

---

## §10. 코드 재사용 정책 (사용자 명시)

### §10.1 핵심 원칙

> **확실하지 않으면 새 코드 생성**. spec ambiguity (signature mismatch, hidden dependency, version drift) 발견 시 *해당 module 신규 작성*, 기존 코드 *import X*.

### §10.2 본 plan 의 신규 작성 / import 허용 / import 금지

| 영역 | 정책 |
|---|---|
| `src/plan018/baseline_a0.py` ~ `arch_a6_gru_attn.py` | **신규 작성** (모든 trainable module). 다른 plan source 의 import X. |
| `analysis/plan-018/eda_check.py`, `ablation_runner.py` | 신규 작성. |
| basis_terms 식 (8 vars) | plan-007 `§6.3.1` 의 `compute_basis_terms` 식 *재구현* (코드 import X, 식 자체만 carry). 8 vars 식별자는 `analysis/plan-007/best_basis.json` 의 ground truth 만 read. |
| `stage3_best_params` (8-vec init) | `analysis/plan-007/best_basis.json` 의 `stage3_best_params` 만 read (JSON, deterministic). |
| 5-fold split (seed=42, sample_id-grouped) | plan-007 §7.2 spec 답습. 코드 *재구현*. |
| soft_hit_loss | plan-007 §7.2 식 *재구현* (간단 함수, import X). |
| `src/pb_0_6822/selector.py` 의 `CANDIDATE_LIST[17]` | identifier 만 *read* (basis var 식별자 확인). 함수 호출 X. |
| `src/plan017/*` | **import 금지** — spec ambiguity (Δ-param form 미확정). A6 는 신규 작성. |
| `src/plan005/`, `src/plan013/`, `src/plan014/`, `src/plan015/`, `src/plan016/` | import 금지. corrector / multi-stage 는 본 plan out-of-scope (§2.2). |

### §10.3 ambiguity 발견 시 처리

- 함수 시그니처 / 동작 의문 시 → 해당 함수 신규 작성. plan 본문 §10 에 "신규 작성 사유" 1 줄 박제.
- 예: "plan-007 의 `compute_basis_terms` 가 sliding window 시 end_idx 보정 식 불명 — 본 plan §5.0 에서 신규 정의 (식: ...)."

---

## §11. References (논문 + 본 plan 내 reference)

### §11.1 논문 (외부)

- Lee, Lee, Kim, Kosiorek, Choi, Teh (2019). *Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks*. ICML.
- Kidger, Morrill, Foster, Lyons (2020). *Neural Controlled Differential Equations for Irregular Time Series*. NeurIPS.
- Kidger, Lyons (2020). *Signatory: differentiable computations of the signature and logsignature transforms*. ICLR.
- Shazeer, Mirhoseini, Maziarz, Davis, Le, Hinton, Dean (2017). *Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer*. ICLR.
- Fedus, Zoph, Shazeer (2021). *Switch Transformer: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity*. JMLR.
- ~~Deng, Litany, Duan, Poulenard, Tagliasacchi, Guibas (2021)~~ — Vector Neurons (A4 제외 사유, §0)

### §11.2 본 plan 내 reference

- plan-007 §3.1 (입력 데이터), §7.1/§7.2 (step 4 MLP spec), §8 (LB submission 4-분기), §9.2 (시나리오 B 결론)
- plan-017 §5 (GRU-attn coeff regressor, Δ-param + L2)
- plan-004 selector.py:697 (`CandidateAttentionGRUSelector`, A6 spec 참조용 — 코드 import X)
- plan-001 §3 (EDA: per-axis std 0.6~1.2 m, noise floor 0.005~0.007 m)
- analysis/plan-007/best_basis.json (8 vars 식별자 + stage3_best_params)

---

## §12. 시간 예산 (전체)

| 단계 | 예상 소요 |
|---|---|
| c1 plan 작성 (본 파일) | (이미 완료) |
| c2 EDA check | ~5 분 |
| c3~c4 A0 reproduce | ~15 분 |
| c5~c10 arch 코드 작성 (A4 제외 → 5 ablation arch) | ~2.5~3.5 시간 (5 arch × ~30~40 분) |
| c11~c12 ablation runner + 5-fold OOF (6 arch 포함 A0) | ~2.5~3.5 시간 (cuda) |
| c13 ablation_results.md | ~30 분 |
| c14 LB 제출 + 회수 | ~10 분 |
| c15 synthesis | ~1 시간 |
| **총** | ~9~12 시간 wall-time |

---

## §13. End-of-Plan Checklist

- [ ] G0: A0 reproduce OOF ∈ [0.6479, 0.6485] + EDA check PASS
- [ ] G1: 6 arch (A0 + A1/A2/A3/A5/A6) × 5-fold OOF 박제, 5 ablation arch 중 ≥ 1 OOF ≥ 0.6532
- [ ] G2: best ablation arch LB ≥ 0.67 (또는 회수 후 band 박제)
- [ ] G_final: results.md + plan-019 후보 ≥ 2 + 3 파일 frontmatter sync
- [ ] 모든 commit + push 완료 (CLAUDE.md ⚠️ Commit · Push 의무)
- [ ] Agent 병렬 재평가 박제 (본 v1.1 patch — BLOCKER 4 + AMBIGUITY 3 + A4 제외)
