---
plan_id: 017
version: 1.1 (plan-review-master iter 1 fix 10건 — voxel window ±2cm BLOCKER fix + 9 AMBIGUITY 정합/명시화)
date: 2026-05-15 (Asia/Seoul)
status: spec
based_on:
  - 013 (LB 미산출, OOF 0.6381, submission analysis/plan-013/submission.csv)
  - 014 (LB 0.6628, OOF 0.6425, submission runs/baseline/plan014_g5_phase4/submission_best.csv)
  - 015 (= 014 deterministic same, drop rule per)
  - 016 (LB 0.6638 G1, OOF 0.6452, submission runs/baseline/plan016_g1_path_a/submission.csv)
followed_by: []
scope: paradigm-shift 결정점 도달 전 *low-cost stage 1 batch* — 두 액션 (#5 ensemble / #4 hit-aware voxel CE head). ablation 없이 *최대한 단순한 버전*. 각 stage 의 OOF/LB measured 후 사용자에게 paradigm-shift (#1 plan-004 2-stage / #2+#3 Trajectory-CLIP+Regime bias) 결정 위임.
exp_ids:
  - H057_g0_preflight
  - H058_g1_ensemble
  - H059_g2_voxel_ce
  - H060_g_final_synthesis
lb_score: null
baseline_lb: 0.6638  # plan-016 G1
baseline_oof: 0.6452 # plan-016 G1
---

# plan-017 v1 — Low-Cost Stage 1 Batch (Ensemble + Voxel CE Head)

## §0. 한 줄 목적

> **plan-016 G1 (LB 0.6638) baseline 위 *low-cost* 두 paradigm 적용 — (G1) 3 plan submission 좌표 mean ensemble, (G2) 5×5×5 voxel CE corrector head. 각 OOF/LB measured 후 사용자 paradigm-shift 결정점 (= G_final) 까지 진행.**

---

## §0.5 Quick Reference

### G-gates

- G0: preflight — 3 submission file 존재 verify + Plan017VoxelCEHead module smoke + baseline reproduce  [TODO]
- G1: 3-plan ensemble (plan-013/014_15/016_G1 좌표 mean) submission + dacon-submit 1회  [TODO]
- G2: 5×5×5 voxel CE head, 5-seed × 5-fold (plan-016 G1 carry config), OOF + dacon-submit 1회  [TODO]
- G_final: results.md + paradigm-shift 결정점 user confirm — #1 plan-004 2-stage 또는 #2+#3 CLIP+Regime bias  [TODO]

### Target

- baseline LB = **0.6638** (plan-016 G1 carry).
- G1 ensemble pass = LB Δ ≥ 0 (non-strict, tie 도 fail 아님; §3.1 와 단어 통일).
- G2 voxel CE pass = OOF Δ > 0 vs G1 baseline 0.6452 (positive). LB submission 사용자 confirm.
- G_final = 두 stage 결과 summary + 사용자 paradigm-shift 결정 anchor 박제.

### Commit chain

| # | type | spec section | status |
|---|---|---|---|
| c1 | docs | v1 draft — plan-017 spec (low-cost stage 1) | [DONE] 0566934 |
| c1.1 | docs | **v1.1 spec patch — plan-review-master iter 1 fix 10건.** (1) §1.2 voxel window ±2.5cm → ±2cm BLOCKER fix. (2) §2.1 G2 변경/보존 명세 정합 (anchor codebook 무력화 명시 + confound caveat). (3) §0.5 G1 pass criterion Δ>0 → Δ≥0 통일. (4) §4.1 (d) 2cm coverage measure 추가. (5) §6.1 voxel_idx_to_offset numpy/torch 양 variant 명시. (6) §6.1 sample_weight dtype/device 명시 (torch.Tensor, requires_grad=False, dtype=float32). (7) §5.2 submission save schema inline. (8) §4.1 (c) smoke input dim (B=16, seq_len=6, feature_dim=9) inline. (9) §7.2 LB band threshold inline (plan-016 외부 의존 제거). (10) §4.3 재사용 module signature inline + cascade 위험 박제. v1 → v1.1 | [TODO] |
| c2 | code+exp | STAGE 0 (G0) — preflight + Voxel CE module smoke | [TODO] |
| c3 | exp | STAGE 1 (G1) — 3-plan ensemble + dacon-submit | [TODO] |
| c4 | code+exp | STAGE 2 (G2) — Voxel CE head 5-seed × 5-fold + dacon-submit | [TODO] |
| c5 | docs+sync | STAGE 3 (G_final) — results.md + frontmatter sync + paradigm-shift 결정점 | [TODO] |

### plan-specific severe

- (없음, default 만)

### plan-specific paths

- whitelist 추가: (없음)
- blacklist 추가: 외부 plan source code 의 *변경* — `src/pb_0_6822/plan014_paradigm.py`, `plan015_train.py`, `plan016_ensemble.py` 등은 *추가 only* (default 동작 보존, 기존 호출 path bit-identical). 새 head/loss 는 신규 module `plan017_voxel_ce.py` 에 위치.

### autonomous decision-note 박제 룰

- dacon-submit 전 *모든 케이스* 사용자 confirm (feedback memory: `feedback_dacon_submit_confirmation.md`). plan spec 의 "각 stage dacon-submit 1회" 은 *허용된 budget* 의미.
- 코드 재사용 시 cascade 효과 사전 검토 (feedback memory: `feedback_code_reuse_correctness.md`).

---

## §1. 배경 / 동기

plan-014/015/016 의 corrector paradigm (F0=plan-006 frenet_par120_perp_neg020 + BiGRU corrector + 7 anchor codebook K=9) 가 LB 0.6638 (plan-016 G1) plateau. plan-016 §11 paradigm-shift 후보 박제:
- ① low-cost stage 1 (본 plan): #5 ensemble + #4 hit-aware voxel CE
- ② paradigm-shift: #1 plan-004 2-stage corrector / #2+#3 Trajectory-CLIP + Regime bias

본 plan = ① 의 *측정만*. 사용자가 ② 결정하기 전 *low-cost evidence* 박제.

### §1.1 Ensemble (G1) — framework-disjoint 결합 가설

- plan-013 submission (analysis/plan-013/submission.csv, OOF 0.6381, plan-004 framework path simplified)
- plan-014/015 best (runs/baseline/plan014_g5_phase4/submission_best.csv, LB 0.6628)
- plan-016 G1 best (runs/baseline/plan016_g1_path_a/submission.csv, LB 0.6638)

3 submission 의 framework 가 *partially disjoint* — plan-013 = plan-004 framework simplified; plan-014/015/016 = corrector paradigm. 좌표 mean 시 uncorrelated error 부분 reduce 기대.

가설: ensemble LB ≥ 0.6638 (+ ε), where ε ~ +0.005 (uncorrelated error mean의 typical effect).

### §1.2 Voxel CE Head (G2) — hit metric 직접 정렬

plan-016 G2 (Path B monitor=val_loss) 의 measured 결론: train objective (hybrid_combined_loss) ↔ eval metric (hit@1cm) misalignment → val_loss 감소해도 hit 안 늘어남.

해결: **discrete classification** 위 hit metric 직접 정렬.
- Voxel grid: F0_pred 중심 **±2cm 범위 (= ±2 voxels × 1cm width)**, 5×5×5 = 125 voxel, voxel width = 0.01m (1cm). axis 별 5 levels = `[-0.02, -0.01, 0, 0.01, 0.02]` m (§6.2.A 와 일치).
- Voxel index = argmin || voxel_center - y_true ||₂ 위 cross-entropy 학습.
- Forward predict: argmax 위 voxel_center → 3D offset → F0_pred + offset.
- 1cm voxel width = hit@1cm threshold 의 *natural alignment* (1cm 안 prediction = correct voxel argmax).

가설: voxel CE 가 plan-016 G1 BiGRU regression head 대비 +0.003~0.005 OOF 회수 가능.

---

## §2. Scope

### §2.1 In-scope

| 항목 | 값 |
|---|---|
| Baseline | plan-016 G1 (5-seed × 5-fold = 25 models, F0 frozen, BiGRU h=128, 7 anchor K=9, boundary_weight_on, monitor=val_hit) |
| G1 변경 | 3 submission 좌표 mean (training 없음, no head/loss change) |
| G2 변경 | corrector head **교체 (cls_head[K] + reg_head[K*3*tanh*REG_SCALE] → voxel_cls_head[125] softmax)** + 새 loss CE(voxel_idx). **anchor codebook (K=9) 도 forward path 에서 무력화** — voxel paradigm 은 F0 + voxel offset 만 사용, anchor 호출 없음. 단일 변경 아니나 voxel CE paradigm 의 *분리 불가능한 cohesive change* 로 spec 박제. confound caveat 는 G_final 결과 해석 시 명시. |
| G2 보존 | F0, BiGRU encoder, 5-fold scheme, multi-seed list, monitor=val_hit |
| 평가 | OOF (5-fold concat) + LB (사용자 confirm 후 dacon-submit) |

### §2.2 Out-of-scope

| 항목 | 이유 |
|---|---|
| Ablation 사이 stage (G1 vs G2 vs G1+G2) | 사용자 명시 "ablation 없이 최대한 단순한 버전" |
| F3/F4 formula parity fix | plan-011 paradigm 전용. plan-006 F0 (frenet_par120_perp_neg020) 사용 시 무관. |
| Voxel CE 내부 tuning (voxel size / window / depth) | 단일 spec (5×5×5, ±2.5cm) 만. 후속 fine-tune 은 ② 결정 후 |
| paradigm-shift (#1 / #2 / #3) 구현 | G_final 사용자 결정점 후 후속 plan |

---

## §3. 사전 등록 (Pre-registration)

### §3.1 합격 기준 (per stage)

- **G1 (ensemble)** pass = LB Δ ≥ 0 vs plan-016 G1 LB 0.6638 (positive direction). OOF 산출 불가 (3 submission 의 OOF 가 동일 train set 위 derived, ensemble OOF = 좌표 mean 위 train sample hit 가능 but 가치 낮음 — *LB 직접 측정만*).
- **G2 (voxel CE)** pass = OOF Δ ≥ +0.003 vs plan-016 G1 OOF 0.6452. LB Δ pass = +0.003 vs 0.6638.
- 둘 다 pass → positive direction confirmed
- 둘 다 fail → paradigm-shift (#1 / #2+#3) 필수성 강화

### §3.2 OOF aggregation (G2)

plan-016 §5.2 carry: 5 seed × 5 fold → per-fold seed-mean → 5-fold concat → hit@1cm.

### §3.3 DACON quota

- 남은 quota: 3 (5/일 - 2 사용 with plan-016 G1+G2).
- G1 ensemble: 1 submit.
- G2 voxel CE: 1 submit.
- 남은 1 = ② paradigm-shift 후속 plan 용 carry.

### §3.4 exp_id

- H057_g0_preflight
- H058_g1_ensemble
- H059_g2_voxel_ce
- H060_g_final_synthesis

---

## §4. STAGE 0 (c2, G0) — preflight [TODO]

### §4.1 산출물

- `analysis/plan-017/preflight.py` — 3 task:
  - (a) 3 submission file 존재 + row count = 10001 (header + 10000 sample) verify.
  - (b) plan-016 G1 baseline reproduce check — 직접 reproduce 안 함 (이미 plan-016 G0/G1 박제). artifact load 만.
  - (c) Plan017VoxelCEHead module smoke — input `seq (B=16, seq_len=6, feature_dim=9)` 위 forward → logits shape `(16, 125)`, voxel_idx range ∈ [0, 125) verify, `voxel_ce_loss(logits, y, f0_pred)` 의 `loss.item()` finite, `loss.backward()` no error. (seq_len=6 = plan-014/016 carry, end_idx=10 의 [5..10] 6 step.)
  - (d) **Voxel grid 2cm window coverage measure** — 10K train sample 위 `||y - F0||₂` 분포 산출, `frac(||y-F0||₂ ≤ 0.02)` (= voxel grid 안 정확 mapping 비율) + p50/p95/p99 quantile 박제. coverage < 95% 시 clamp 효과 caveat 박제 (§6.2.B clamp behavior).
- `analysis/plan-017/preflight.json`
- registry row `H057_g0_preflight`

### §4.2 G0 합격

- (a) 3 file 존재 (10000 row + header = 10001 line), row count 일치
- (b) plan-016 G1 artifact 로드 OK (`analysis/plan-016/g1_path_a.json` 의 OOF=0.6452, LB=0.6638 carry)
- (c) Voxel CE smoke: forward (B=16, seq_len=6, feature_dim=9) → logits shape (16, 125), voxel_idx shape (16,) ∈ [0, 125), loss.item() finite, backward.step() no error
- (d) Voxel coverage measure: `frac(||y - F0||₂ ≤ 0.02m) ≥ 0.90` (90% 이상 coverage 시 clamp 효과 marginal 판정).

### §4.3 Code reuse safety check (§3.4 박제, code 작성 의무)

- `src/pb_0_6822/plan014_paradigm.py` *수정 안 함*. import only.
- `src/pb_0_6822/plan016_ensemble.py` *수정 안 함*. import only.
- 신규: `src/pb_0_6822/plan017_voxel_ce.py` — VoxelCEHead + loss + ensemble runner adapter.
- 재사용 module signature (inline 박제, 외부 read 없이 작성 가능):
  - `Plan014BiGRUEncoder(input_dim=9, hidden=128, num_layers=2, dropout=0.1)`: `forward(seq: torch.Tensor of shape (B, seq_len, 9)) → torch.Tensor of shape (B, 256)` (last-step bidir concat).
  - `_boundary_weight(F0_pred_init: np.ndarray of shape (N, 3), target: np.ndarray of shape (N, 3)) → np.ndarray of shape (N,)`: mask = (0.005 < ‖F0−y‖ < 0.015), sw = where(mask, 3.0, 1.0).
  - `Plan014F0Function()`: callable `f0_function(X: np.ndarray of shape (N, 11, 3)) → np.ndarray of shape (N, 3)` (numpy F0 prior).
  - `stable_hash_fold(sid: str) → int ∈ [0, 5)`: sha256 + salt='plan-014-v1'. plan-014 carry.
- cascade 위험 점검 (사전 박제):
  - voxel CE 도입 시 hybrid_combined_loss (plan-014/016 ring CE + huber + hinge) 와 *공존 안 함*. plan-017 voxel CE loss 만 사용 — anchor/reg/hinge term 모두 forward path 에 없음.
  - boundary_weight 사용 시 voxel_ce_loss 안에 sample-wise 곱 — gradient flow 보존 (sample_weight 은 torch tensor on logits.device, requires_grad=False).
  - test pred 산출 시 (X_test=None default) 기존 plan-016 ensemble runner 의 동작 보존 (X_test 미사용 시 cascade 없음).

---

## §5. STAGE 1 (c3, G1) — 3-plan ensemble [TODO]

### §5.1 산출물

- `analysis/plan-017/g1_ensemble.py` — 3 submission 좌표 mean 산출.
- `runs/baseline/plan017_g1_ensemble/submission.csv` — final ensemble submission.
- `analysis/plan-017/g1_ensemble.json` — sample_ids 일치 verify, mean shape (10000, 3), per-source-submission L2 distance to mean (variance proxy).
- registry row `H058_g1_ensemble`.

### §5.2 spec

3 submission 좌표 mean (모든 file 의 schema = `id,x,y,z` header + 10000 row, sample_submission.csv 순서):
```python
import pandas as pd
import numpy as np

sub_a = pd.read_csv("analysis/plan-013/submission.csv")
sub_b = pd.read_csv("runs/baseline/plan014_g5_phase4/submission_best.csv")
sub_c = pd.read_csv("runs/baseline/plan016_g1_path_a/submission.csv")

# id 정렬 invariant: sample_submission.csv 위 sort 동일 (각 plan 이미 그 순서)
for s in (sub_b, sub_c):
    assert (s["id"].values == sub_a["id"].values).all(), "id mismatch"

# 좌표 mean
mean_xyz = (sub_a[["x","y","z"]].values
          + sub_b[["x","y","z"]].values
          + sub_c[["x","y","z"]].values) / 3.0
ids = sub_a["id"].tolist()

out = pd.DataFrame({
    "id": ids,
    "x": [f"{v:.6f}" for v in mean_xyz[:, 0]],
    "y": [f"{v:.6f}" for v in mean_xyz[:, 1]],
    "z": [f"{v:.6f}" for v in mean_xyz[:, 2]],
})
out.to_csv("runs/baseline/plan017_g1_ensemble/submission.csv", index=False)
```

dacon-submit 1회 (사용자 confirm 후, feedback memory `feedback_dacon_submit_confirmation` 박제 의무).

### §5.3 G1 합격

- LB Δ ≥ 0 vs 0.6638 → positive ensemble effect.
- LB Δ < 0 → ensemble effect negative (uncorrelated error 가정 falsified, plan-013 의 낮은 prediction quality 가 결합 시 pull-down).

### §5.4 Code reuse safety check

- 외부 plan submission file *읽기만*. 절대 *수정 / 덮어쓰기 안 함*.
- id column 정렬 일치 verify (`assert` 명시) — sample_submission.csv 의 row 순서 가 모든 plan 의 submission 에서 동일하다는 invariant.

---

## §6. STAGE 2 (c4, G2) — Voxel CE head 5-seed × 5-fold [TODO]

### §6.1 산출물

- `src/pb_0_6822/plan017_voxel_ce.py` — 신규 module (양 variant numpy + torch 모두 박제):
  - `class Plan017VoxelCEHead(nn.Module)` — encoder + voxel cls head (125 class).
  - `def voxel_grid_centers_np() → np.ndarray of shape (125, 3)`: F0 relative offset, axis 별 [-0.02, -0.01, 0, 0.01, 0.02] m.
  - `def y_to_voxel_idx(y: np.ndarray of shape (N, 3), f0_pred: np.ndarray of shape (N, 3)) → np.ndarray of shape (N,) int64`: clamp 후 axis-wise base-5 인덱싱.
  - `def voxel_idx_to_offset_np(idx: np.ndarray of shape (N,)) → np.ndarray of shape (N, 3)`: numpy variant (CSV write 용).
  - `def voxel_idx_to_offset_torch(idx: torch.Tensor of shape (N,), device) → torch.Tensor of shape (N, 3)`: torch variant (forward predict 용, autograd 호환).
  - `def voxel_ce_loss(logits: torch.Tensor (B, 125), y: torch.Tensor (B, 3), f0_pred: torch.Tensor (B, 3), sample_weight: torch.Tensor (B,) or None) → torch.Tensor scalar`: CE on voxel_idx. sample_weight = torch.Tensor on logits.device, requires_grad=False, dtype=float32.
  - `def hybrid_predict_voxel(seq, encoder, voxel_head, f0_pred_detached)` — encoder(seq) → voxel_head(h) → argmax → voxel_idx_to_offset_torch → F0 + offset.
  - `def train_one_fold_voxel(cfg, fold_id, X_train, Y_train, X_val, Y_val, f0_function, X_test=None)` — plan014_paradigm.train_one_fold 의 voxel 변형. 동일 cfg dataclass 사용. monitor=val_hit (val_hit 산출 = hybrid_predict_voxel 후 hit@1cm). early stop, sample_weight (=boundary_weight) 동일 적용.
  - `def run_multiseed_kfold_voxel(ids_train, X_train, Y_train, ids_test, X_test, config_base, seeds, f0_function)` — plan016_ensemble.run_multiseed_kfold 의 voxel 변형. OOF aggregation = 좌표 mean over seeds → 5-fold concat → hit@1cm (§5.2 carry).
- `analysis/plan-017/g2_voxel_ce.py` — 5-seed × 5-fold 학습 + OOF + test ensemble + submission 산출.
- `runs/baseline/plan017_g2_voxel_ce/submission.csv`
- `analysis/plan-017/g2_voxel_ce.json`
- registry row `H059_g2_voxel_ce`.

### §6.2 spec

#### §6.2.A Voxel grid

```python
VOXEL_WIDTH = 0.01   # 1cm (hit threshold)
VOXEL_DEPTH = 5      # ±2.5cm per axis (= ±2 voxels + center)
VOXEL_TOTAL = 125    # 5³
HALF_RANGE = (VOXEL_DEPTH - 1) / 2  # 2.0
# axis grid: [-2, -1, 0, 1, 2] × VOXEL_WIDTH = [-0.02, -0.01, 0, 0.01, 0.02]
# voxel_idx = (ix + 2) * 25 + (iy + 2) * 5 + (iz + 2)   where ix ∈ {-2..2}
```

#### §6.2.B Voxel label (y → voxel_idx)

```python
def y_to_voxel_idx(y, f0_pred):
    """y (N, 3), f0_pred (N, 3). Returns (N,) int ∈ [0, 125).
    voxel_idx = nearest voxel center to (y - f0_pred). Out-of-range → clamp to nearest edge."""
    offset = y - f0_pred                                    # (N, 3)
    voxel_ijk = np.round(offset / VOXEL_WIDTH).astype(int)  # axis-wise nearest integer
    voxel_ijk = np.clip(voxel_ijk, -2, 2)                   # clamp ±2 (= ±2cm)
    voxel_idx = (voxel_ijk[:, 0] + 2) * 25 + (voxel_ijk[:, 1] + 2) * 5 + (voxel_ijk[:, 2] + 2)
    return voxel_idx
```

> *Note*: y - F0 의 norm 이 > 2cm 인 sample 은 *clamp* 됨 — voxel grid 가 cover 못함. plan-014 G0 oracle 0.8248 (E0b Frenet-ortho) 의 *radius 1cm* 와 비교 시 2cm window 가 ~95% sample 을 cover 추정 (검증 필요, G0 preflight 에서 measure).

#### §6.2.C Voxel CE loss

```python
import torch.nn.functional as F

def voxel_ce_loss(logits, y, f0_pred, sample_weight=None):
    """logits (B, 125), y (B, 3), f0_pred (B, 3). Returns scalar."""
    voxel_idx = y_to_voxel_idx(y.cpu().numpy(), f0_pred.cpu().numpy())   # (B,) int
    voxel_idx_t = torch.from_numpy(voxel_idx).to(logits.device)
    ce_per_sample = F.cross_entropy(logits, voxel_idx_t, reduction="none")  # (B,)
    if sample_weight is not None:
        ce_per_sample = ce_per_sample * sample_weight
    return ce_per_sample.mean()
```

> boundary_weight (plan-014 E6 carry) 사용 가능 — `sample_weight = _boundary_weight(F0_train, Y_train)` (plan-016 G1 spec 동일).

#### §6.2.D Forward predict

```python
def hybrid_predict_voxel(seq, encoder, voxel_head, f0_pred_detached, temperature=None):
    """temperature 무시 (voxel CE는 argmax 만). Returns (B, 3) world frame pred."""
    h = encoder(seq)                                    # (B, 256)
    logits = voxel_head(h)                              # (B, 125)
    argmax_idx = logits.argmax(dim=1)                   # (B,)
    offset = voxel_idx_to_offset_tensor(argmax_idx)     # (B, 3) torch
    return f0_pred_detached + offset
```

#### §6.2.E Train loop

plan-016 G1 carry config (5 seed × 5 fold = 25 models, K=9 anchor *unused* in voxel paradigm — anchor logit 대신 voxel argmax). monitor=val_hit (hit@1cm 직접 monitor).

> *주의*: anchor codebook 은 *voxel head 와 함께 무력화*. anchor 가 forward path 에 안 들어감 = pure voxel-only paradigm. *plan-016 G1 의 K=9 anchor + boundary_weight + bigru encoder* 만 carry 하되 anchor 는 forward 안 씀.

### §6.3 G2 합격

- OOF Δ ≥ +0.003 vs plan-016 G1 OOF 0.6452 (= OOF ≥ 0.6482) → OOF pass.
- LB Δ ≥ +0.003 vs 0.6638 (= LB ≥ 0.6668) → LB pass.
- 둘 다 pass → positive (G6 target 0.6678 근접).
- 한 쪽 만 → marginal.
- 둘 다 Δ < 0 → negative_drop, paradigm-shift (#1 / #2+#3) 필수.

### §6.4 Code reuse safety check

- `plan014_paradigm` 의 `Plan014BiGRUEncoder` 재사용 — `input_dim=9` (plan-016 G1 baseline 동일), encoder.forward(seq) → (B, 256) 시그너처 보존.
- `plan014_paradigm._boundary_weight` 재사용 — sample-wise weight 산출, sample_weight 인자로 voxel_ce_loss 전달.
- `plan014_paradigm.run_kfold_oof` / `train_one_fold` 는 *재사용 안 함* — voxel paradigm 이 forward path 가 다름. 신규 `train_one_fold_voxel`, `run_multiseed_kfold_voxel` 작성.
- `plan016_ensemble.run_multiseed_kfold` 의 OOF aggregation 패턴 (좌표 mean over seeds → 5-fold concat → hit@1cm) 만 *복제* — voxel 변형에서 동일 logic 적용.
- F0 = plan-006 `Plan014F0Function()` — 변경 없음.
- 5-fold split = `pp.stable_hash_fold` (plan-014 carry) — 변경 없음.
- seed list = plan-016 G1 carry [20260514..20260518].

---

## §7. STAGE 3 (c5, G_final) — synthesis + paradigm-shift 결정점 [TODO]

### §7.1 산출물

- `plans/plan-017-low-cost-stage1.results.md` (신규)
- `plans/plan-017-low-cost-stage1.md` frontmatter sync (status=G_final_complete, lb_score, followed_by=[018])
- registry append H060_g_final_synthesis
- §0.5 sync c5 [TODO]→[DONE]

### §7.2 합격 기준

- 2 stage 결과 박제 (G1 ensemble LB Δ, G2 voxel CE OOF+LB Δ).
- band 분류 (inline 박제, plan-016 §0.5 carry):
  - LB ≥ 0.68 → **plan-004 LB 정조준 달성** (positive-top)
  - 0.66 ≤ LB < 0.68 → positive
  - 0.65 ≤ LB < 0.66 → partial
  - LB < 0.65 → negative
- **paradigm-shift 결정점** = 사용자 confirm:
  - 후보 A: #1 plan-004 2-stage corrector (selector + boundary corrector).
  - 후보 B: #2 Trajectory-CLIP + KNN-Augmented 27-pool + #3 486-entry regime bias (병합 plan-018).
  - 후보 C: 기타 (#4/#5/#6 등 plan-016 §11.4 후보 중 사용자 선택).

### §7.3 paradigm-shift 결정 anchor

- G1 ensemble LB > 0.6638 → ensemble path 가 cheap +ε 회수. paradigm-shift cost 낮춰도 됨.
- G1 ensemble LB ≤ 0.6638 → ensemble 가치 무효. paradigm-shift 필수.
- G2 voxel CE OOF/LB Δ > +0.003 → loss-metric alignment 가설 *measured true*. plan-018 후속 voxel CE 확장 (depth/window grid) 매력적.
- G2 OOF Δ < 0 → corrector paradigm 내 head reformation 한계. paradigm-shift (#1 또는 #2+#3) 필수.

---

## §N+4. 변경 이력

- v1 (2026-05-15) — draft. ablation 없이 simplest version. F3/F4 fix drop (plan-006 F0 paradigm 무관).

---

## §N+5. 참조

- `plans/plan-016-corrector-stabilization.results.md` §4 — paradigm-shift candidates (shift-1 ~ shift-5)
- `notes/new-ideas.md` §D.1 (5×5×5 Voxel CE), §D.3 (Trajectory-CLIP)
- `notes/drone-insights.md` §📌 Element A (486-entry regime bias)
- `notes/prior-ideas.md` §2 (State-conditional anchor), §4 (Physics regularizer)
- `src/pb_0_6822/plan014_paradigm.py` — Plan014BiGRUEncoder, _boundary_weight, Plan014F0Function, stable_hash_fold (재사용, 변경 없음)
- `src/pb_0_6822/plan016_ensemble.py` — run_multiseed_kfold OOF aggregation pattern (복제용 reference)
- analysis/plan-016/g1_path_a.json — plan-016 G1 baseline (OOF 0.6452 / LB 0.6638) carry
- `feedback_dacon_submit_confirmation.md` (memory) — dacon-submit user confirm 의무
- `feedback_code_reuse_correctness.md` (memory) — 코드 재사용 cascade 검토 의무
