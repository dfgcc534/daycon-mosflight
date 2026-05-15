---
plan_id: 019
version: 1
date: 2026-05-15 (Asia/Seoul)
status: draft
based_on:
  - 007 (Step 4 MLP OOF 0.6482, LB 0.6598 — 본 plan baseline)
  - 005 (oracle 0.7188 — 본 plan ceiling reference)
  - 004 (ensemble LB 0.6822 — single-model ceiling 돌파 target)
  - brainstorm-iter-1~5 (5-iteration /loop, 6 candidates: meta-EBIP+ICNN / EBIP / meta-EBIP / Learnable Basis+MoLE / DEQ / ICNN)
scope: 본 5-iteration brainstorm 의 *ambitious path* — **meta-EBIP + ICNN hybrid** (brainstorm ranking #1)
       의 progressive ablation. plan-018 결과 *무관* (사용자 명시) — plan-007 step 4 가 baseline.
       3 stage breakdown: (S1) EBIP base / (S2) + ICNN convex / (S3) + meta adaptation.
       LB > 0.70 도달이 G_final (= plan-005 oracle 0.7188 의 97%, plan-004 ensemble 0.6822 위 +0.018).
       brainstorm candidates #2 DEQ / #3 meta-EBIP / #4 Learnable Basis+MoLE 는 plan-020 carry.
exp_ids:
  - F014_ebip-base           # S1 EBIP base (energy = bilinear anchor + g_θ small MLP)
  - F015_ebip-icnn           # S2 + ICNN convex g_θ
  - F016_meta-ebip-icnn      # S3 + meta adaptation (FOMAML inner loop)
lb_score: null
exception_policy: plan-007 §2.2 "End-to-end 학습 통합 out-of-scope" 의 **예외 plan** (plan-018 의 예외 plan 정책 carry).
                  본 plan 의 paradigm = energy-based implicit prediction — single-stack 의 일종 (encoder + energy + coefficient).
                  multi-stack (corrector + selector + 등 ≥ 3 stage) 은 여전히 out-of-scope.
---

# plan-019 v1 — Meta-EBIP + ICNN Hybrid (single-model ceiling 돌파 path)

## §0. 한 줄 목적

> 본 5-iteration `/loop` brainstorm 의 ranking #1 paradigm **meta-EBIP + ICNN hybrid** 으로 plan-007 step 4 spirit (per-sample coefficient regression on fixed 8 basis) 을 *energy-based implicit prediction* 으로 reformulate. progressive ablation 3 stage (S1 EBIP base → S2 + ICNN convex → S3 + meta adaptation) 으로 *각 component 의 marginal gain 측정*. 단일 모델 LB > 0.70 도달 시 G_final PASS (= plan-005 oracle 0.7188 의 97%, plan-004 ensemble 0.6822 +0.018). brainstorm 의 나머지 5 candidates (DEQ / meta-EBIP / Learnable Basis+MoLE / EBIP base / ICNN Convex Energy) 중 본 plan 미적용은 plan-020 후보 carry.

---

## §0.5 Quick Reference (autonomous loop 매 turn 읽는 section)

### 본 plan 의 task essence — "energy-based implicit prediction 의 progressive 3-stage ablation"

- plan-007 step 4 의 *explicit* form `pred = p0 + Σ c(τ)·B(τ)` 의 *implicit* reformulation:
  ```
  pred = argmin_p [ ||p - p0 - c(τ)·B(τ)||²  +  λ · g_θ(p, τ) ]
                    └──── bilinear anchor ────┘   └ energy correction ┘
                          (step 4 spirit 박제)        (learnable universal)
  ```
- 3 stage progressive ablation:
  - **S1 EBIP base**: g_θ = small MLP (~600 params), `unrolled gradient descent (5-step)` on p
  - **S2 + ICNN convex**: g_θ = Input Convex Neural Network (Amos 2017), `1-step Newton` (closed-form)
  - **S3 + meta adaptation**: c(τ) 를 *FOMAML inner loop* (1-step SGD on c, self-supervised pretext) 로 sample-별 adapt
- 각 stage 의 marginal gain 측정 = "어떤 component 가 가장 critical 인가" 의 직접 답.

### Brainstorm carry — 본 plan 의 paradigm 선정 사유

- 5-iter brainstorm (iter 1~5, `/loop` 결과) 의 6 candidates ranking:
  1. **meta-EBIP + ICNN hybrid** ⭐⭐ (본 plan 채택, 예상 LB 0.72~0.74)
  2. DEQ (예상 LB 0.71~0.73, plan-020 carry)
  3. meta-EBIP (예상 LB 0.72~0.74, 본 plan S3 가 sub-set)
  4. Learnable Basis + Sparse MoLE (예상 LB 0.71~0.72, plan-020 carry)
  5. EBIP base (예상 LB 0.70~0.72, 본 plan S1 = 이것)
  6. ICNN Convex Energy (예상 LB 0.69~0.71, 본 plan S2 = 이것)
- 즉 본 plan 의 3 stage 가 #5, #6, #1 을 직접 cover. #2 / #3 / #4 는 plan-020 carry.

### 합격 기준 (G-gate sequence)

- **G0** (baseline): plan-007 step 4 A0 baseline 재현. OOF ∈ [0.6479, 0.6485]. 위반 시 `baseline_reproduce_fail` severe (plan-018 §4 carry).
- **G1** (S1 EBIP base): 5-fold OOF ≥ 0.66 (= step 4 baseline 0.6482 + 0.012, marginal gain 박제). 위반 시 `ebip_no_gain` warn — 결과 박제 후 S2 진행 (warn-only).
- **G2** (S2 + ICNN convex): 5-fold OOF ≥ 0.68. 위반 시 `icnn_no_gain` warn.
- **G3** (S3 + meta adaptation): 5-fold OOF ≥ 0.70 ⭐. 위반 시 `meta_adaptation_no_gain` warn.
- **G4** (LB 제출): G1/G2/G3 중 best variant 1 만 dacon-submit. **LB > 0.70** → G_final PASS. 0.69 ≤ LB ≤ 0.70 → `partial — band carry`. LB < 0.69 → `lb_below_target` warn.
- **G_final**: results.md + plan-020 후보 ≥ 2 + 3 파일 frontmatter sync.

LB 제출 = **총 1회** (best variant 만, plan-018 spirit carry — DACON daily 5 limit 내 cost-efficient rule).

### G-gates (commit 단위 milestone)

- G0: A0 baseline reproduce + EDA check (plan-018 §4 carry)                  [TODO]
- G1: S1 EBIP base OOF ≥ 0.66                                                  [TODO]
- G2: S2 + ICNN convex OOF ≥ 0.68                                              [TODO]
- G3: S3 + meta adaptation OOF ≥ 0.70 ⭐                                       [TODO]
- G4: best variant LB > 0.70                                                   [TODO]
- G_final: results.md + plan-020 후보 + frontmatter sync                       [TODO]

### Commit chain (next-up)

| # | type | spec section | status |
|---|---|---|---|
| c1 | docs | `plans/plan-019-*.md` 본문 (본 파일) | [DONE] (3772665) |
| c1.1 | docs | v1.1 spec amendments (user feedback Critical/High/Medium 3건) | [DONE] (015ca9e) |
| c2 | code | `analysis/plan-019/eda_check.py` — 데이터 sanity (plan-018 §4.1 spec carry). 신규 작성 | [TODO] |
| c3 | code | `src/plan019/baseline_a0.py` — plan-007 step 4 MLP 재구현 (plan-018 §4.2 spec carry). 신규 작성 | [TODO] |
| c4 | exp | A0 5-fold OOF reproduce | [TODO] |
| G0 | gate | A0 OOF ∈ [0.6479, 0.6485] | [TODO] |
| c5 | code | `src/plan019/ebip_base.py` — EBIP base (energy = bilinear + g_θ small MLP, unrolled 5-step Adam). spec @ §5 | [TODO] |
| c6 | exp | F014: S1 EBIP base 5-fold OOF | [TODO] |
| G1 | gate | S1 OOF ≥ 0.66 | [TODO] |
| c7 | code | `src/plan019/ebip_icnn.py` — Input Convex NN g_θ + 1-step Newton (Amos 2017). spec @ §6 | [TODO] |
| c8 | exp | F015: S2 EBIP + ICNN 5-fold OOF | [TODO] |
| G2 | gate | S2 OOF ≥ 0.68 | [TODO] |
| c9 | code | `src/plan019/meta_ebip_icnn.py` — FOMAML inner loop + ICNN energy. spec @ §7 | [TODO] |
| c10 | exp | F016: S3 meta-EBIP + ICNN 5-fold OOF | [TODO] |
| G3 | gate | S3 OOF ≥ 0.70 ⭐ | [TODO] |
| c11 | sub-lb | best variant dacon-submit + lb_log + frontmatter. spec @ §8 | [TODO] |
| G4 | gate | LB > 0.70 (또는 회수 후 band 박제) | [TODO] |
| c12 | synthesis | `analysis/plan-019/results.md` + `next_plan_candidates.md` (≥ 2 후보). spec @ §9 | [TODO] |
| G_final | gate | results.md + plan-020 후보 + 3 파일 frontmatter sync | [TODO] |

### Plan-specific severe (WORKFLOW.md §12.3 default 위 추가분)

- `baseline_reproduce_fail`: G0 의 A0 OOF ∉ [0.6479, 0.6485]. plan-018 §4 carry. 즉시 halt.
- `ebip_unrolled_unstable`: S1 학습 중 NaN loss / unrolled gradient explosion / fold 의 ≥ 3 가 NaN. unrolled depth T 를 5 → 3 로 줄이고 1회 retry. 재실패 시 severe.
- `icnn_constraint_violation`: S2 의 ICNN weight 가 non-negativity 제약 위반 (softplus parametrization fail). 1회 retry. 재실패 시 severe.
- `meta_inner_loop_divergent`: S3 의 FOMAML inner loop 가 sample 의 ≥ 30% 에서 발산 (||c_τ - c_meta||² > 10). inner lr η 를 0.01 → 0.003 로 줄이고 1회 retry. 재실패 시 severe.

### Plan-specific paths (WORKFLOW.md §12.5/§12.6 추가/제외)

- whitelist 추가:
  - `src/plan019/**` (본 plan 의 모든 trainable module — 신규 작성)
  - `analysis/plan-019/**` (EDA + stage 결과)
  - `runs/baseline/F014_ebip-base/**`, `runs/baseline/F015_ebip-icnn/**`, `runs/baseline/F016_meta-ebip-icnn/**`
- blacklist 추가:
  - `src/pb_0_6822/**` 의 수정 (lock-in)
  - `src/plan017/**`, `src/plan018/**` 의 import (plan-018 §10 carry — 신규 작성 원칙)

### Decision-note 사용 예 (자율 결정 시 commit msg 박제)

- `decision-note: spec-default — S1 EBIP unrolled depth T=5, inner lr=0.1, λ=0.5 (brainstorm iter 3 spec).`
- `decision-note: spec-default — S2 ICNN hidden 32, 2 layers, softplus parametrization (Amos 2017 §3.2).`
- `decision-note: spec-default — S3 FOMAML inner lr η=0.01, 1-step inner SGD (Nichol 2018 Reptile 권장값).`
- `decision-note: spec-default — S3 self-supervised pretext = window[-2] reconstruction (next-step prediction).`
- `decision-note: exception-plan — plan-007 §2.2 의 end-to-end 통합 out-of-scope 의 예외 (plan-018 §2.3 답습).`
- `decision-note: brainstorm-carry — 5-iter /loop brainstorm 의 ranking #1 paradigm 채택. #2/#3/#4 는 plan-020 carry.`

### ⚠️ Spec amendments (user feedback 검토 후 박제, c2~c12 구현 시 본문 spec 보다 *우선* 적용)

- `decision-note: spec-amendment — S3 의 basis_terms_prev 는 반드시 horizon=1 (window[-2] → window[-1], +40ms) 로 계산. 최종 basis_terms 는 horizon=2 (+80ms) — 두 basis 의 시간 스케일이 다름. basis term 이 t² (가속도), t³ (jerk) 비선형이므로 horizon mismatch 시 trajectory 가 물리적으로 깨짐. §7.3 collate_fn 의 dataset pre-compute 시 두 horizon 의 basis_terms 분리 저장 필수.`
- `decision-note: spec-amendment — S2 ICNN 의 §6.1 의 1-step Newton with diagonal Hessian (H_inv = 1/(2+λ)·I) 은 사실상 constant-LR GD — ICNN convex advantage 실종. unrolled GD T=3~5 step 으로 변경 (ICNN convex 보장으로 발산 X). 또는 Amos 2017 의 OptNet 기반 implicit differentiation (선택, complexity 증가). 본 plan default = unrolled GD T=3.`
- `decision-note: spec-amendment — c_dim=13 (handcrafted stats) 는 plan-011 결과 (P1.ID TrajectoryCNNEncoder 64-dim 이 유일한 +0.0050 gain) 위반의 information bottleneck. coeff_mlp 와 icnn 의 conditioning feature 를 13d → 77d (13 handcrafted + 64 CNN encoded) 로 확장. plan-011 의 TrajectoryCNNEncoder 코드는 import X (§10 정책) — 신규 작성 (src/plan019/cnn_encoder.py, plan-011 §P1.ID spec 만 carry). ICNN convexity 는 p (3-d output) 에 대해서만, c (77-d conditioning) 는 non-convex 허용.`

---

## §1. 배경

### §1.1 plan-007 step 4 의 한계 (carry-over)

- plan-007 step 4 MLP (~300 params, 13-d stats encoder → 8 coefficient) = **OOF 0.6482, LB 0.6598**.
- 시나리오 B (plan-007 §9.2): "단일 공식 framework 의 한계 ≈ 0.6491 baseline 동급, +0.0095 marginal."
- plan-005 oracle = **0.7188** (best of 27 candidates, sample-별 best). 단일 공식 ceiling 과의 gap = 0.07 → 본 plan 의 *측정 대상*.

### §1.2 본 plan 의 핵심 가설

| 가설 | 검증 방법 | 합격 |
|---|---|---|
| H1: step 4 의 marginal gain (+0.0095) 의 main bottleneck 은 *explicit form* 의 expressivity 제약 — implicit reformulation (energy minimization) 으로 universal correction 가능 | S1 EBIP base OOF ≥ 0.66 | G1 PASS |
| H2: EBIP 의 training stability 문제 (g_θ non-convex) 는 ICNN convex 제약으로 해결, 천장 약간 하향하되 안정성 ↑ | S2 + ICNN OOF ≥ 0.68 | G2 PASS |
| H3: per-sample coefficient 의 *FOMAML inner-loop adaptation* 으로 천장이 oracle 근접 | S3 + meta OOF ≥ 0.70 | G3 PASS |
| H4: best variant 단일 모델 LB > 0.70 (plan-004 ensemble 위 +0.018) | dacon-submit | G4 PASS |

### §1.3 본 plan 이 *안 하는* 것 (focus)

- DEQ (brainstorm #2) — plan-020 후보. infinite-depth implicit layer 는 본 plan 의 5-step unrolled 보다 ambitious, 본 plan 범위 외.
- Learnable Basis + Sparse MoLE (#4) — plan-020 후보. *basis 자체의 learnable* 변환은 step 4 spirit 의 일부 위반 (basis 변경) — 본 plan basis 고정 8 vars.
- 27 후보 풀 확장 — plan-007 §9.2 의 후보 1, 본 plan basis 고정.
- corrector 결합 (plan-005 / plan-016) — multi-stack 으로 본 plan out-of-scope. plan-020 후보.

---

## §2. Scope (명시적)

### §2.1 In-scope

| 항목 | 값 |
|---|---|
| basis (fixed) | plan-007 Step 3 best 8 vars (analysis/plan-007/best_basis.json) |
| 데이터 증폭 | sliding 40K + original 10K = 50K pool (plan-018 §3.1 carry) |
| 5-fold split | plan-018 §3.1 carry (seed=42, sample_id-grouped) |
| window slicing | `train_x[:, end_idx-5 : end_idx+1, :]` → (N, 6, 3) (plan-018 §3.1 carry) |
| training | Adam lr=1e-3, wd=1e-4, batch=256, epoch=50, patience=8, grad_clip=2.0 (plan-018 §2.1 carry — 모든 stage 공통) |
| loss | `soft_hit_loss(pred, target, threshold=0.01, sharpness=200)` (plan-007 §7.2 carry) |
| paradigm | meta-EBIP + ICNN hybrid (3 stage progressive) |
| LB 제출 | **1 회** (best variant, G4) |

### §2.2 Out-of-scope (절대 안 함)

| 항목 | 이유 |
|---|---|
| DEQ / Learnable Basis + MoLE / pure meta-EBIP without ICNN | brainstorm #2/#3/#4, plan-020 후보 carry |
| Corrector 결합 | multi-stack — plan-020 후보 |
| 27 후보 풀 확장 / multi-formula | basis 고정 8 vars |
| Hyperparameter sweep (각 stage 별 lr/wd 변형) | 모든 stage 동일 setting. *progressive gain comparison fair* 보장 |
| LB 제출 ≥ 2 회 | 본 plan 1 회 (best variant), plan-018 spirit carry |
| Plan-004/005/006/007/016/017/018 의 trainable module import | 신규 작성 (§10 코드 재사용 정책, plan-018 §10 carry) |

### §2.3 plan-007 §2.2 exception 명시 (plan-018 §2.3 carry)

본 plan 의 EBIP variant 들은 *single-stack end-to-end* (encoder + energy g_θ + coefficient head). plan-007 §2.2 의 "End-to-end 학습 통합 out-of-scope" 의 예외 — 본 plan = plan-007 step 4 의 *spirit* (per-sample coefficient on fixed basis) 의 직접 generalization. plan-018 §2.3 정책 그대로 carry.

---

## §3. 사전 등록 (Pre-registration)

### §3.1 입력 데이터 + 분할 (plan-018 §3.1 carry)

| 분할 | 출처 | 사용 |
|---|---|---|
| Train original (10K, end_idx=10) | `data/train/` | A0 + S1/S2/S3 학습/검증 |
| Train sliding (40K, end_idx ∈ [5, 8], horizon=2) | sliding aug (plan-007 §4.1) | train fold only |
| Test (10K, end_idx=10) | `data/test/` | best variant inference + submission |

Window slicing rule: `trajectory_window = train_x[:, end_idx - 5 : end_idx + 1, :]` → (N, 6, 3). target: original = `train_y`, sliding = `train_x[:, end_idx + 2, :]`.

### §3.2 합격 기준 (정량)

- **G0**: A0 OOF ∈ [0.6479, 0.6485].
- **G1**: S1 EBIP base 5-fold OOF ≥ 0.66.
- **G2**: S2 + ICNN convex 5-fold OOF ≥ 0.68.
- **G3**: S3 + meta adaptation 5-fold OOF ≥ 0.70 ⭐.
- **G4**: best variant LB > 0.70.
- **G_final**: results.md + plan-020 후보 ≥ 2 + frontmatter sync.

### §3.3 평가

- OOF metric = 5-fold concat, original 10K 의 hit rate (threshold 0.01 m) — plan-007 §3.3 carry.
- LB metric = DACON public LB hit rate.

---

## §4. STAGE 0 — A0 Baseline Reproduce (c2~c4, plan-018 §4 carry)

### §4.1 EDA check

`analysis/plan-019/eda_check.py` (신규 작성). plan-018 §4.1 spec 의 4 assertion (shape / distribution / noise floor / const-vel hit rate) carry. 코드 *재구현* (import X — plan-018 §10 정책).

### §4.2 A0 baseline reproduce

`src/plan019/baseline_a0.py` (신규 작성). plan-007 step 4 MLP 의 직접 재구현 (plan-018 §4.2 spec carry). 본 plan 의 모든 stage 의 training loop 가 *동일 framework* 위에서 동작하는지 검증.

### §4.3 G0 합격 (자동 판정)

- A0 OOF ∈ [0.6479, 0.6485]. 위반 시 `baseline_reproduce_fail` severe.

---

## §5. STAGE 1 — EBIP Base (c5~c6, F014)

> ⚠️ **§5 구현 시 §0.5 의 spec-amendment 박제 우선 적용**:
> - c_dim=13 → 77 (13 handcrafted + 64 CNN encoded, plan-011 P1.ID TrajectoryCNNEncoder 신규 작성).
> - §5.2 의 `feat_dim: int = 13` 은 *handcrafted only* 의미. 실제 forward 의 conditioning 은 `torch.cat([traj_features_13d, cnn_encoded_64d], dim=1)` = 77d.

### §5.1 Energy formulation

```python
# pred = argmin_p E(p; τ)
#
# E(p; τ) = ||p - anchor(τ)||²  +  λ · g_θ(p, encoder_features(τ))
# where:
#   anchor(τ) = p0 + Σ_i c_i(τ) · B_i(τ)     # step 4 spirit anchor
#   c(τ)      = small MLP encoder → 8-dim coefficient (A0 spec carry, 13-d stats input)
#   g_θ       = small MLP (3 + 13 → 32 → 32 → 1), p 와 encoder features 의 함수
#   λ         = learnable scalar (init 0.5)
```

### §5.2 Implementation — `src/plan019/ebip_base.py`

```python
import numpy as np
import torch
import torch.nn as nn


class EBIPBase(nn.Module):
    """Energy-Based Implicit Prediction — bilinear anchor + small MLP correction.

    forward(traj_features, p0, basis_terms) → pred (B, 3)
      where pred = argmin_p E(p; ...), via unrolled gradient descent (T=5 steps).
    """
    def __init__(self, *, feat_dim: int = 13, n_coeffs: int = 8,
                 global_init: np.ndarray | None = None,
                 hidden: int = 32, unroll_T: int = 5):
        super().__init__()
        # coefficient encoder (A0 carry)
        self.coeff_mlp = nn.Sequential(
            nn.Linear(feat_dim, 32), nn.SiLU(),
            nn.Linear(32, n_coeffs),
        )
        if global_init is not None:
            with torch.no_grad():
                self.coeff_mlp[-1].bias.copy_(
                    torch.tensor(global_init, dtype=torch.float32))
                self.coeff_mlp[-1].weight.zero_()

        # energy correction (small MLP, p ∈ R^3 + features ∈ R^feat_dim → scalar)
        self.energy_mlp = nn.Sequential(
            nn.Linear(3 + feat_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 1),
        )

        self.log_lambda = nn.Parameter(torch.tensor(0.0))   # λ = exp(log_lambda), init 1.0
        self.unroll_T = unroll_T
        self.inner_lr = 0.1   # gradient descent step size on p

    def forward(self, traj_features: torch.Tensor, p0: torch.Tensor,
                basis_terms: torch.Tensor) -> torch.Tensor:
        """
        Args:
            traj_features: (B, feat_dim)  — A0 의 13-d stats summary
            p0:            (B, 3)
            basis_terms:   (B, 8, 3)
        Returns:
            pred:          (B, 3)
        """
        # 1) anchor 계산
        coeffs = self.coeff_mlp(traj_features)                           # (B, 8)
        anchor = p0 + (coeffs.unsqueeze(-1) * basis_terms).sum(dim=1)    # (B, 3)

        # 2) energy minimization — unrolled gradient descent on p
        lam = torch.exp(self.log_lambda)
        p = anchor.detach().clone().requires_grad_(True)
        for t in range(self.unroll_T):
            energy = (
                ((p - anchor) ** 2).sum(dim=1).mean()                    # anchor 항
                + lam * self.energy_mlp(
                    torch.cat([p, traj_features], dim=1)).squeeze(-1).mean()
            )                                                            # g_θ 항
            grad_p = torch.autograd.grad(energy, p, create_graph=self.training)[0]
            p = p - self.inner_lr * grad_p
        return p

    # NOTE: unrolled gradient descent through autograd.grad(create_graph=True) 는
    #       메모리 O(T). T=5 면 5x 메모리. unstable 시 unroll_T=3 fallback (plan-specific severe).
```

### §5.3 학습

- §3.1 의 5-fold split + sliding aug pool 위에서 학습.
- loss = soft_hit_loss(pred, target) — pred 는 위 forward 의 결과.
- Adam lr=1e-3, wd=1e-4, batch=256, epoch=50, patience=8 (val_hit on original-only val fold).

### §5.4 산출

- `runs/baseline/F014_ebip-base/checkpoint.pt` — best fold 의 state_dict
- `runs/baseline/F014_ebip-base/oof_predictions.npz` — 5-fold concat OOF
- `analysis/plan-019/s1_ebip_base.json` — OOF + fold_oofs + elapsed

### §5.5 G1 합격 (자동 판정)

- S1 5-fold OOF ≥ 0.66. 위반 시 `ebip_no_gain` warn (severe X, S2 진행).

### §5.6 시간 예산

- 학습: ~30~40 분 (cuda 2.8.0, 50 epoch × 50K pool × T=5 unroll = 5x A0 cost).

---

## §6. STAGE 2 — EBIP + ICNN Convex Energy (c7~c8, F015)

> ⚠️ **§6 구현 시 §0.5 의 spec-amendment 박제 우선 적용**:
> - **§6.1 의 EBIPICNN.forward 의 1-step Newton with diagonal Hessian → unrolled GD T=3 로 변경** (ICNN convex 보장이라 발산 X, 진짜 minimum 도달 가능). 1-step diagonal Newton 은 사실상 constant-LR GD — ICNN advantage 실종.
> - c_dim 도 §5 동일하게 77d (handcrafted 13 + CNN 64). ICNN convexity 는 *p (3-d)* 에만 적용, c 는 non-convex 허용.

### §6.1 ICNN spec (Amos et al. 2017, Input Convex Neural Network)

ICNN 의 핵심: input p 에 대해 *globally convex* — weight 의 non-negativity 제약 (`softplus` 로 reparametrize).

```python
class ICNNEnergy(nn.Module):
    """Input Convex NN — g_θ(p, c) is convex in p.

    c (conditioning, traj_features) 는 *non-convex 변수* — 학습 시 free, p 는 convex.
    Amos 2017 eq 4 의 fully-input-convex variant.
    """
    def __init__(self, *, p_dim: int = 3, c_dim: int = 13, hidden: int = 32, n_layers: int = 2):
        super().__init__()
        self.W_p = nn.ModuleList([nn.Linear(p_dim, hidden, bias=False) for _ in range(n_layers + 1)])
        # W_z: non-negative weight (softplus param)
        self._W_z_raw = nn.ParameterList([
            nn.Parameter(torch.randn(hidden, hidden) * 0.01) for _ in range(n_layers)
        ])
        self.W_c = nn.ModuleList([nn.Linear(c_dim, hidden) for _ in range(n_layers + 1)])
        self.b   = nn.ParameterList([nn.Parameter(torch.zeros(hidden)) for _ in range(n_layers + 1)])
        # final output: scalar
        self.W_out_p = nn.Linear(p_dim, 1, bias=False)
        self._W_out_z_raw = nn.Parameter(torch.randn(hidden) * 0.01)
        self.W_out_c = nn.Linear(c_dim, 1)

    def _W_z(self, idx: int) -> torch.Tensor:
        return torch.nn.functional.softplus(self._W_z_raw[idx])   # ≥ 0 (convexity)

    def _W_out_z(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self._W_out_z_raw)

    def forward(self, p: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """p: (B, 3), c: (B, 13). Returns (B,) scalar energy."""
        z = torch.nn.functional.silu(self.W_p[0](p) + self.W_c[0](c) + self.b[0])
        for l in range(len(self._W_z_raw)):
            z = torch.nn.functional.silu(
                z @ self._W_z(l).T + self.W_p[l + 1](p) + self.W_c[l + 1](c) + self.b[l + 1]
            )
        out = (z * self._W_out_z()).sum(dim=1, keepdim=True) \
              + self.W_out_p(p) + self.W_out_c(c)
        return out.squeeze(-1)


class EBIPICNN(nn.Module):
    """EBIP base 의 energy_mlp 를 ICNNEnergy 로 교체. argmin = 1-step Newton (closed-form)."""
    def __init__(self, *, feat_dim: int = 13, n_coeffs: int = 8,
                 global_init: np.ndarray | None = None,
                 icnn_hidden: int = 32, icnn_layers: int = 2):
        super().__init__()
        self.coeff_mlp = nn.Sequential(
            nn.Linear(feat_dim, 32), nn.SiLU(),
            nn.Linear(32, n_coeffs),
        )
        if global_init is not None:
            with torch.no_grad():
                self.coeff_mlp[-1].bias.copy_(
                    torch.tensor(global_init, dtype=torch.float32))
                self.coeff_mlp[-1].weight.zero_()
        self.icnn = ICNNEnergy(p_dim=3, c_dim=feat_dim,
                               hidden=icnn_hidden, n_layers=icnn_layers)
        self.log_lambda = nn.Parameter(torch.tensor(0.0))

    def forward(self, traj_features: torch.Tensor, p0: torch.Tensor,
                basis_terms: torch.Tensor) -> torch.Tensor:
        coeffs = self.coeff_mlp(traj_features)
        anchor = p0 + (coeffs.unsqueeze(-1) * basis_terms).sum(dim=1)
        lam = torch.exp(self.log_lambda)

        # E(p; c) = ||p - anchor||² + λ · g_icnn(p, c)
        # ∇²_p E = 2·I + λ · ∇²_p g_icnn(p, c)
        # ∇_p E |_{p=anchor} = 2·(anchor - anchor) + λ · ∇_p g_icnn(anchor, c) = λ · ∇_p g_icnn(anchor, c)
        # 1-step Newton: p* ≈ anchor - (∇²_p E)⁻¹ · (∇_p E |_{anchor})
        #
        # ICNN 의 ∇²_p g_icnn 은 p.s.d. (convexity 보장). 단, 작은 batch 에서 Hessian
        # explicit 계산은 cost. simpler choice: damped Newton — fixed step Hessian approx.
        # 본 spec: 1-step Newton with diagonal Hessian approx (효율).
        p = anchor.detach().clone().requires_grad_(True)
        energy = self.icnn(p, traj_features).sum()
        grad_p = torch.autograd.grad(energy, p, create_graph=self.training)[0]   # (B, 3)
        # diagonal Hessian: ∂²_p g ≈ ε⁻² · (g(p+ε) - 2g(p) + g(p-ε))   per dim
        # 간소화: H ≈ (1 + λ) · I  (constant approx, 학습 안정성 우선)
        H_inv = 1.0 / (2.0 + lam)
        pred = anchor - lam * H_inv * grad_p
        return pred
```

### §6.2 학습

- §5.3 spec 동일 (Adam lr=1e-3, wd=1e-4, batch=256, epoch=50, patience=8).
- 추가 제약: ICNN weight 의 non-negativity assertion (epoch 마다 1 회) — `assert (torch.nn.functional.softplus(self._W_z_raw[l]) >= 0).all()`. 위반 시 `icnn_constraint_violation` warn.

### §6.3 산출

- `runs/baseline/F015_ebip-icnn/checkpoint.pt`
- `runs/baseline/F015_ebip-icnn/oof_predictions.npz`
- `analysis/plan-019/s2_ebip_icnn.json`

### §6.4 G2 합격

- S2 5-fold OOF ≥ 0.68. 위반 시 `icnn_no_gain` warn (severe X, S3 진행).

### §6.5 시간 예산

- 학습: ~20~25 분 (cuda, 1-step Newton 으로 unroll 없음 → S1 보다 빠름).

---

## §7. STAGE 3 — Meta-EBIP + ICNN (FOMAML inner loop) (c9~c10, F016)

> ⚠️ **§7 구현 시 §0.5 의 spec-amendment 박제 우선 적용 — Temporal Horizon Mismatch (Critical)**:
> - `basis_terms_prev` = `compute_basis_terms(window[:, :5], horizon=1)` — +40ms 예측용 basis (inner loop 의 self-supervised pretext 가 window[-2] → window[-1] 이므로 horizon=1 필수).
> - `basis_terms` (최종 anchor 계산용) = `compute_basis_terms(window, horizon=2)` — +80ms 예측용 basis (target=train_y 의 horizon 과 일치).
> - **두 basis 가 다른 horizon** — basis term 의 t² (acc), t³ (jerk) 비선형성 때문에 inner loop 에서 학습된 c_τ 를 *그대로* horizon=2 anchor 에 쓰면 trajectory 깨짐. dataset pre-compute 시 두 horizon 의 basis_terms 를 *분리 저장* 필수 (§7.3).
> - c_dim 도 §5/§6 동일하게 77d (handcrafted 13 + CNN 64).

### §7.1 FOMAML spec (Finn et al. 2017, first-order MAML)

각 sample 의 coefficient c 를 *task-specific parameter* 로 reframe:

```
outer training (over all samples):
  학습: c_meta (meta-coefficient via coeff_mlp), ICNN g_θ, λ
  loss: L_hit (pred_τ, target_τ) — pred_τ 는 아래 inner loop 의 결과

inner adaptation (per sample τ, both training & inference):
  1. c_τ ← c_meta(τ) - η · ∇_c L_unsup(c; τ)
     L_unsup(c; τ) = ||window[-2] - reconstruct(c, window[:-1])||²
                     # next-step self-supervised pretext
     reconstruct(c, window) = window[-1] + Σ c_i · B_i(window[:-1])
                     # window[-1] 을 anchor 로, c 와 basis_terms(window[:-1]) 로 next-step 예측
  2. pred_τ ← argmin_p [||p - p0 - c_τ · B(window)||² + λ · g_icnn(p, traj_features)]
                                                       # S2 의 EBIPICNN forward, c 를 c_τ 로 대체
```

- *first-order* MAML: outer gradient 에서 inner-loop 의 second-order term 무시 (FOMAML, Finn 2017 §5.2).
- 메모리 = S2 와 동등 (inner loop 가 differentiable 이지만 first-order 만).
- inner step 수 = 1 (Reptile 권장).

### §7.2 Implementation — `src/plan019/meta_ebip_icnn.py`

```python
class MetaEBIPICNN(nn.Module):
    """meta-EBIP + ICNN — FOMAML inner loop adaptation + ICNN convex energy.

    S2 EBIPICNN 의 super-set. inner_steps=0 이면 S2 와 동일.
    """
    def __init__(self, *, feat_dim: int = 13, n_coeffs: int = 8,
                 global_init: np.ndarray | None = None,
                 inner_lr: float = 0.01, inner_steps: int = 1, **icnn_kwargs):
        super().__init__()
        self.coeff_mlp = nn.Sequential(
            nn.Linear(feat_dim, 32), nn.SiLU(),
            nn.Linear(32, n_coeffs),
        )
        if global_init is not None:
            with torch.no_grad():
                self.coeff_mlp[-1].bias.copy_(
                    torch.tensor(global_init, dtype=torch.float32))
                self.coeff_mlp[-1].weight.zero_()
        self.icnn = ICNNEnergy(p_dim=3, c_dim=feat_dim, **icnn_kwargs)
        self.log_lambda = nn.Parameter(torch.tensor(0.0))
        self.inner_lr = inner_lr
        self.inner_steps = inner_steps

    def _self_supervised_loss(self, c: torch.Tensor, window: torch.Tensor,
                              basis_terms_prev: torch.Tensor) -> torch.Tensor:
        """Next-step reconstruction loss for inner-loop adaptation.

        window: (B, 6, 3). 사용:
          - anchor = window[:, -2]  (= 직전 step 위치)
          - target = window[:, -1]  (= 마지막 step 위치, *next-step* 의미)
          - basis_terms_prev: basis_terms computed on window[:, :-1]  (B, 8, 3)
        """
        pred_recon = window[:, -2] + (c.unsqueeze(-1) * basis_terms_prev).sum(dim=1)
        return ((pred_recon - window[:, -1]) ** 2).sum(dim=1).mean()

    def forward(self, traj_features: torch.Tensor, p0: torch.Tensor,
                basis_terms: torch.Tensor, window: torch.Tensor,
                basis_terms_prev: torch.Tensor) -> torch.Tensor:
        # 1) c_meta
        c_meta = self.coeff_mlp(traj_features)   # (B, 8)

        # 2) FOMAML inner adaptation
        c_tau = c_meta
        for _ in range(self.inner_steps):
            L_unsup = self._self_supervised_loss(c_tau, window, basis_terms_prev)
            grad_c = torch.autograd.grad(L_unsup, c_tau, create_graph=False)[0]
            #                                              ^^^^^^^^^^^^^^^^
            #              first-order MAML: inner grad 의 outer-backprop 차단
            c_tau = c_tau - self.inner_lr * grad_c

        # 3) S2 의 EBIPICNN forward — c 를 c_tau 로 대체
        anchor = p0 + (c_tau.unsqueeze(-1) * basis_terms).sum(dim=1)
        lam = torch.exp(self.log_lambda)

        p = anchor.detach().clone().requires_grad_(True)
        energy = self.icnn(p, traj_features).sum()
        grad_p = torch.autograd.grad(energy, p, create_graph=self.training)[0]
        H_inv = 1.0 / (2.0 + lam)
        pred = anchor - lam * H_inv * grad_p
        return pred
```

### §7.3 학습 — collate_fn 확장

S3 의 입력은 S1/S2 보다 *window + basis_terms_prev* 가 추가. collate_fn 에 다음 field 추가:

- `window:            (B, 6, 3)`
- `basis_terms_prev:  (B, 8, 3)`  — `compute_basis_terms(window[:, :5], horizon=1)` ⚠️ **horizon=1 필수** (§0.5 amendment, §7 박스)
- `basis_terms:       (B, 8, 3)`  — `compute_basis_terms(window, horizon=2)` (anchor 계산, target=train_y 의 horizon 과 일치)

두 basis 모두 dataset init 시 *분리 저장* 사전 계산 (plan-018 §5.0 의 dataset pre-compute 정책 carry, 학습 매 step 재계산 X).

`compute_basis_terms(x, horizon)` 의 horizon 분기 spec — plan-007 §6.3.1 의 식 (carry, 재구현) 에 `horizon` argument 추가:
- horizon=1: end_idx 의 *다음 step* 예측용 — t¹ (속도), t² (가속도/2), t³ (jerk/6) 의 1-step 값.
- horizon=2: end_idx 의 *2-step 후* 예측용 — t¹×2, t²×4/2, t³×8/6 (즉 horizon 배수에 따른 비선형 scaling).
- horizon mismatch 시 trajectory 가 *물리적으로 깨짐* — basis term 의 비선형성이 horizon 에 의존.

### §7.4 산출

- `runs/baseline/F016_meta-ebip-icnn/checkpoint.pt`
- `runs/baseline/F016_meta-ebip-icnn/oof_predictions.npz`
- `analysis/plan-019/s3_meta_ebip_icnn.json`

### §7.5 G3 합격

- S3 5-fold OOF ≥ 0.70 ⭐. 위반 시 `meta_adaptation_no_gain` warn.

### §7.6 시간 예산

- 학습: ~25~30 분 (cuda, FOMAML inner 1-step 추가, S2 보다 ~20% 느림).

---

## §8. STAGE 4 — Best Variant LB 제출 (c11)

### §8.1 자율 호출

```python
# G3 종료 후, best variant 선정:
# best_stage = argmax over {"S1": S1.oof, "S2": S2.oof, "S3": S3.oof}
# (참고: G1/G2/G3 가 모두 warn 인 경우 — best 가 < 0.66 일 수 있음. submit 진행, band 박제.)
Skill(skill="dacon-submit",
      args=f"runs/baseline/F{14+best_stage_idx}_*/submission.csv "
           f"F{14+best_stage_idx}_{best_stage_name}_plan-019")
```

### §8.2 응답 4-분기 처리 (plan-018 §8.2 동일 패턴)

| (isSubmitted, lb_score) | 처리 | frontmatter `lb_score` | status | severe |
|---|---|---|---|---|
| (True, float) | full success | `<float>` | `all_complete` | — |
| (True, None) | partial — carry-over commit `c11.1` | `TBD` | `partial` | — |
| (False, *) | retry 1회 (60 초 sleep). 재실패 시 severe | `null` | `partial` | `lb_unsubmitted` |
| Skill exception | 즉시 escalate | `null` | `partial` | `dacon_submit_skill_missing` |

### §8.3 `analysis/plan-019/lb_log.md` 포맷

```markdown
| timestamp_kst | exp_id | best_stage | isSubmitted | lb_score | detail |
|---|---|---|---|---|---|
| ... | F0XX_<stage>_plan-019 | S? | true | 0.7XXX | OK |
```

### §8.4 G4 합격 + band 분류

- **LB > 0.70** → G_final PASS, status `all_complete`. plan-005 oracle 0.7188 의 97% 도달, plan-004 ensemble 0.6822 위 +0.018.
- 0.69 ≤ LB ≤ 0.70 → `partial — band carry`. plan-004 근접 도달, ceiling 돌파 미달.
- 0.68 ≤ LB < 0.69 → `lb_marginal_above_plan018` warn — plan-018 ceiling 위 미달.
- LB < 0.68 → `lb_below_target` warn — ambitious paradigm 의 *실측 ceiling* 박제.

---

## §9. STAGE 5 — Synthesis + plan-020 후보 (c12)

### §9.1 `analysis/plan-019/results.md`

frontmatter:
```yaml
---
plan_id: 019
based_on:
  - 007
  - 005
  - 004
  - brainstorm-iter-1~5
finished_at: <ISO8601 KST>
status: all_complete | partial
exp_ids_completed:
  - F014_ebip-base
  - F015_ebip-icnn
  - F016_meta-ebip-icnn
lb_exp_id: F0XX_<best_stage>_plan-019
lb_score: <float|TBD|null>
lb_submitted_at: <ISO8601 KST>
exception_policy: plan-007 §2.2 의 end-to-end 통합 예외 — 본 plan 의 EBIP variant 가 single-stack 확인
---
```

본문:
- G0 baseline reproduce 결과
- G1/G2/G3 progressive ablation table (S1/S2/S3 OOF, fold spread, elapsed, params)
- 각 component 의 marginal gain 분석:
  - S1 vs A0 = "explicit → implicit reformulation 의 gain"
  - S2 vs S1 = "convex 제약의 stability vs ceiling 의 trade-off"
  - S3 vs S2 = "per-sample meta adaptation 의 추가 gain"
- G4 LB 결과 + band 분류
- plan-020 후보 ≥ 2 (시나리오 분기)

### §9.2 시나리오 분기

| G4 결과 | plan-020 후보 |
|---|---|
| LB > 0.70 (G_final PASS) | (1) **plan-020 = best variant + corrector freeze (plan-005/016 결합)** — LB 0.72+ 시도. (2) DEQ variant (brainstorm #2) 대신 meta-EBIP+ICNN base 위에서 *27 후보 풀 확장* — single-formula × multi-candidate 결합. |
| 0.68 ≤ LB ≤ 0.70 | (1) **plan-020 = Learnable Basis + MoLE (brainstorm #4) 단독 시도** — basis 자체 확장으로 천장 추가. (2) DEQ (brainstorm #2) 시도 — infinite-depth implicit layer. |
| LB < 0.68 | (1) brainstorm 의 *체계적 review* — 표현력 paradigm 자체 reset. (2) plan-005 corrector 결합 path (multi-stack 허용). |

### §9.3 frontmatter sync (3 파일)

- `plans/plan-019-meta-ebip-icnn-hybrid.md` top-level `lb_score`
- `plans/plan-019-meta-ebip-icnn-hybrid.results.md` frontmatter
- `analysis/plan-019/results.md` frontmatter

---

## §10. 코드 재사용 정책 (plan-018 §10 carry)

### §10.1 핵심 원칙

> **확실하지 않으면 새 코드 생성**. 다른 plan 의 trainable module 의 *직접 import 금지*. 사양 (식, 정책) 만 spec 으로 carry 하여 *재구현*.

### §10.2 본 plan 의 신규 작성 / 허용 / 금지

| 영역 | 정책 |
|---|---|
| `src/plan019/baseline_a0.py` ~ `meta_ebip_icnn.py` | **신규 작성**. plan-018 / plan-017 / plan-007 의 trainable module 직접 import X. |
| `analysis/plan-019/eda_check.py` | 신규 작성. plan-018 §4.1 spec 의 4 assertion *재구현*. |
| basis_terms 식 (8 vars) | plan-007 `§6.3.1` 의 compute_basis_terms 식 재구현. import X. |
| `stage3_best_params` (8-vec init) | `analysis/plan-007/best_basis.json` 의 `stage3_best_params` 만 read (JSON, deterministic). |
| 5-fold split (seed=42, sample_id-grouped) | plan-018 §3.1 spec 답습. 코드 재구현. |
| soft_hit_loss | plan-007 §7.2 식 재구현. |
| Window slicing (`train_x[:, end_idx-5:end_idx+1]`) | plan-018 §3.1 spec 답습. |
| `src/plan017/**`, `src/plan018/**` | **import 금지** — 신규 작성 원칙. |
| `src/plan005/**`, `src/plan013~016/**` | import 금지. corrector / multi-stage 는 본 plan out-of-scope (§2.2). |

### §10.3 ambiguity 발견 시 처리

- 함수 시그니처 / 동작 의문 시 → 해당 함수 신규 작성. plan 본문 §10 에 사유 1 줄 박제.
- 예: "plan-018 의 collate_batch 의 basis_terms_prev field 신규 추가 — plan-018 §5.0 spec 에 미정의 (본 plan §7.3 carry 위해 신규 추가)."

---

## §11. References

### §11.1 논문 (외부)

- LeCun, Chopra, Hadsell, Ranzato, Huang (2006). *A Tutorial on Energy-Based Learning*. MIT Press.
- Du & Mordatch (2019). *Implicit Generation and Generalization in Energy-Based Models*. NeurIPS.
- Florence, Lynch, Zeng, Ramirez, Wahid, Downs, Wong, Lee, Mordatch, Tompson (2021). *Implicit Behavioral Cloning*. CoRL.
- Amos, Xu, Kolter (2017). *Input Convex Neural Networks*. ICML.
- Chen, Shi, Mazumdar, Kolter (2020). *Optimal Control via Neural Networks: A Convex Approach*. ICLR.
- Finn, Abbeel, Levine (2017). *Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks*. ICML.
- Nichol, Achiam, Schulman (2018). *On First-Order Meta-Learning Algorithms*. arXiv 1803.02999 (Reptile).

### §11.2 본 plan 내 reference

- plan-007 §3.1 (입력 데이터), §6.3.1 (compute_basis_terms), §7.1/§7.2 (step 4 MLP spec), §8 (LB submission 4-분기)
- plan-018 §2.3 (plan-007 §2.2 exception), §3.1 (window slicing), §4.1 (EDA check), §4.2 (A0 spec), §5.0 (collate_batch contract), §8.2 (LB 4-분기), §10 (코드 재사용 정책)
- plan-005 oracle 0.7188 (analysis/plan-005/results.md)
- analysis/plan-007/best_basis.json (8 vars + stage3_best_params)
- brainstorm /loop iteration 1~5 (대화 transcript) — 6 candidates ranking + paradigm 평가

---

## §12. 시간 예산 (전체)

| 단계 | 예상 소요 |
|---|---|
| c1 plan 작성 (본 파일) | (이미 완료) |
| c2 EDA check | ~5 분 |
| c3~c4 A0 reproduce | ~10~15 분 |
| c5 S1 EBIP base 코드 | ~30 분 작성 |
| c6 S1 5-fold OOF | ~30~40 분 (cuda) |
| c7 S2 ICNN 코드 | ~40 분 작성 |
| c8 S2 5-fold OOF | ~20~25 분 (cuda) |
| c9 S3 meta-EBIP+ICNN 코드 | ~40 분 작성 |
| c10 S3 5-fold OOF | ~25~30 분 (cuda) |
| c11 best variant LB 제출 + 회수 | ~10 분 |
| c12 synthesis | ~1 시간 |
| **총** | ~5~6 시간 wall-time |

---

## §13. End-of-Plan Checklist

- [ ] G0: A0 reproduce OOF ∈ [0.6479, 0.6485] + EDA check PASS
- [ ] G1: S1 EBIP base 5-fold OOF ≥ 0.66
- [ ] G2: S2 + ICNN convex 5-fold OOF ≥ 0.68
- [ ] G3: S3 + meta adaptation 5-fold OOF ≥ 0.70 ⭐
- [ ] G4: best variant LB > 0.70 (또는 회수 후 band 박제)
- [ ] G_final: results.md + plan-020 후보 ≥ 2 + 3 파일 frontmatter sync
- [ ] 모든 commit + push 완료 (CLAUDE.md ⚠️ Commit · Push 의무)
- [ ] brainstorm carry — plan-020 후보 = (1) DEQ / (2) Learnable Basis + MoLE / (3) corrector freeze 결합
