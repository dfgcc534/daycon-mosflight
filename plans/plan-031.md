---
plan_id: plan-031
status: complete
based_on: plan-030
followed_by: plan-032 (fine-distill + reverse-pretrain 후보, 사용자 선택)
title: PB training procedure carry (multi-phase + pairwise + prior) over plan-030 input axis
data_window: train_labels.csv 전체 (plan-024/29/30 carry)
fold_pin: stable_fold_id(sid, 5) (plan-024 carry, MD5)
horizon: 80ms (2 step × DT=40ms)
g3_oof_hit_1cm: 0.6397
g3_band: STRONG
g1_oof_hit_1cm: 0.6450
lift_vs_plan_030: +0.0103
lift_vs_F0: +0.0077
---

# plan-031 — PB training procedure carry

## §0. 한 줄 목적

plan-030 G3 FAIL (hit_1cm 0.6294 < F0 0.6320) 의 root cause = **single-phase training 의 logit magnitude 신장 실패** (`plan-030.results.md` §2.3). `analysis/plan-029/paradigm_root_cause.md` 의 진단 확정. plan-031 = **PB selector 의 multi-phase training procedure 를 plan-030 input axis 위에 carry** 하여 PB 0.6511 ↔ plan-030 0.6294 의 **gap -0.0217 회복 + plan-024 ceiling 0.6387 + F0 +0.005 안전 margin** 달성.

## §0.5 Quick Reference

| 항목 | 값 |
|---|---|
| paradigm | GRU-attention residual injection (plan-030 carry) + **PB multi-phase training (신규)** |
| input axis (carry from plan-030) | GRU input 97 + attention-K/V H+5 + attention-Q 64 + head sample 51 + slim7 7 (전체 carry, 변경 X) |
| **신규 training axis** | **pre + fine 2-phase** + **pairwise margin loss** + **regime/class prior** + **head MLP slimming** |
| pre phase | 15 epoch, lr=7e-4 cosine, soft CE only |
| fine phase | 35 epoch, lr=2e-4 cosine, soft CE + pairwise margin 0.12 + regime_prior strength 0.65 + class_prior strength 0.45 |
| head MLP slimming | head_hidden 384 → **196** (over-param 완화, plan-030 §2 분석 carry) |
| evaluation | 5-fold OOF stable_fold_id, hit_1cm + hit_1p5cm |
| 합격 기준 | G3 OOF hit_1cm ≥ **0.6360 PASS** (F0 +0.004). 0.6387+ STRONG (plan-024 ceiling). 0.65+ EXCELLENT (PB level). |

### Commit chain (예정)

| commit | spec section | status |
|---|---|---|
| c0 spec | §0 ~ §6 | [DONE] |
| c1 pairwise_loss | §3.1 | [DONE] `analysis/plan-031/pairwise_loss.py` |
| c2 regime_class_prior | §3.2 | [DONE] `analysis/plan-031/prior_loss.py` |
| c3 head slimming | §3.3 | [DONE] `analysis/plan-031/model.py` GRUNetX3 (head_hidden 196) |
| c4 train multi-phase | §3.4 | [DONE] `analysis/plan-031/train.py` (pre 15ep + fine 35ep) |
| c5 smoke | §4 | [DONE] `tests/test_plan031_smoke.py` (20/20 green) |
| c6 G1 1-fold | §4 | [DONE] hit_1cm = **0.6450** (PASS > 0.6330) |
| c7 G3 5-fold OOF | §4 | [DONE] hit_1cm = **0.6397** ✅ **STRONG** (≥ 0.6387) |
| c8 ablation (input axis drop) | §5 | [DEFERRED → plan-032 후보] G3 STRONG 도달로 우선순위 낮음, plan-032 spec 단계의 옵션 |
| c9 results | §5 | [DONE] `plans/plan-031.results.md`, frontmatter sync |

---

## §1. Failure attribution (plan-030 G3 FAIL 원인 박제)

plan-030 의 G3 hit_1cm = 0.6294 (FAIL_regression) 의 5 layer root cause (`plans/plan-030.results.md` §2 carry):

| layer | 진단 | plan-031 fix |
|---|---|---|
| 1. fold-0 outlier | G1 0.6436 (fold-0) vs G3 0.6294 (5-fold mean). fold 1~4 mean 0.6260 가 진짜 expected | G1 PASS threshold 상향 (F0 + 0.001 = 0.6330) |
| 2. head MLP over-param | 382 dim 의 sample_summary broadcast 65% dominant → anchor discrimination dilute | head_hidden 384 → **196 slim** |
| 3. score_std 신장 실패 | last epoch 2.7 (logit-scale) vs label sharpness 1/τ_cls=1000 → top1_acc 12.3% | **pairwise margin loss** (margin 0.12) 로 logit gap 직접 강제 + **pre/fine 2-phase** (fine 단계 lr 1/3.5 로 sharpening) |
| 4. 잔차 (a)/(b) net negative | input feature 만으로 -0.0022 worse | input axis carry 유지 (단독 ablation c8 으로 PB 결합 후 기여도 측정) |
| 5. paradigm_root_cause.md 진단 확정 | main lever = training procedure | **본 plan 의 main axis** (multi-phase + pairwise + prior) |

---

## §2. Architecture carry (plan-030 input axis 그대로)

**모든 plan-030 input 박제 그대로 carry** (변경 X):
- GRU input per step: seq 95 + 잔차 (a) XY/Z 2 = **97D × 7**
- attention-K/V per step: GRU output H + 잔차 (a) 5 = **(H+5) × 7** (K=V tied)
- attention-Q per anchor: anchor_spec 9 + par/perp/dist 3 + interactions 10 + 잔차 (b) 35 + slim 7 = **64D × 14**
- head MLP sample summary: GRU hidden[-1] H + 51 = **H+51**
- head MLP per-anchor concat: attention context + sample_summary broadcast + slim 7

**유일 변경 (over-param 완화)**:
- head_hidden: 384 → **196** (Linear(382, 196) → SiLU → Dropout 0.08 → Linear(196, 1))
- 파라미터 수: head MLP `382 × 384 + 384 + 384 × 1 + 1 = 147,073` → `382 × 196 + 196 + 196 × 1 + 1 = 75,069` (= 51% 감소)

GRU/attention/anchor 처리 모두 plan-030 GRUNetX2 carry.

---

## §3. Training spec (신규 — PB multi-phase carry)

### §3.1 Pairwise margin loss (`analysis/plan-031/pairwise_loss.py`)

```python
def pairwise_margin_loss(
    score: torch.Tensor,           # (B, K=14) raw logit
    gt_anchor_idx: torch.Tensor,   # (B,) int — argmin distance anchor (label-derived)
    margin: float = 0.12,
) -> torch.Tensor:
    """gt anchor 의 logit 이 other K-1 anchor 의 logit 보다 최소 margin 만큼 크도록 강제.

    loss = mean over batch [ mean over j≠gt of max(0, score[gt] - score[j] - margin) ]
    Wait, this is wrong direction. 정의:
    loss = mean over batch [ mean over j≠gt of max(0, margin - (score[gt] - score[j])) ]
        = mean( margin - (score[gt] - score[other]) )_+ averaged over (other ≠ gt)

    logit gap 강제 = "gt logit - other logit >= margin" → violation = max(0, margin - gap).
    """
    B, K = score.shape
    gt_score = score.gather(1, gt_anchor_idx.unsqueeze(1))    # (B, 1)
    gap = gt_score - score                                     # (B, K), gap[b, gt] = 0
    violation = torch.clamp(margin - gap, min=0.0)             # (B, K)
    # zero out gt index (gt vs gt 비교 제외)
    mask = torch.ones_like(violation)
    mask.scatter_(1, gt_anchor_idx.unsqueeze(1), 0.0)
    loss = (violation * mask).sum(dim=1) / (K - 1)             # (B,)
    return loss.mean()
```

gt_anchor_idx = `argmin_k ||R_wfn^T @ (gt - F0) - ANCHORS_A6[k]||` (label-derived hardest anchor).

### §3.2 Regime / class prior loss (`analysis/plan-031/prior_loss.py`)

```python
def regime_class_prior_loss(
    score: torch.Tensor,                # (B, K)
    regime_anchor_prior: torch.Tensor,  # (B, K) regime r 의 P(anchor=k) (train-fold lookup, plan-029 carry)
    class_prior_global: torch.Tensor,   # (K,) train-fold 전체 의 P(anchor=k)
    regime_strength: float = 0.65,
    class_strength: float = 0.45,
) -> torch.Tensor:
    """KL divergence 가중 합. log P(model) vs (regime_prior + class_prior) weighted target.

    log_probs = log_softmax(score, dim=-1)                              # (B, K)
    prior_target = regime_strength * regime_anchor_prior + class_strength * class_prior_global[None, :]
    prior_target = prior_target / prior_target.sum(dim=-1, keepdim=True)  # normalize
    loss = -(prior_target * log_probs).sum(dim=-1).mean()                 # CE to prior
    return loss
```

가중치 (plan-004 carry): regime_strength=0.65, class_strength=0.45.

### §3.3 Model spec (`analysis/plan-031/model.py`)

```python
class GRUNetX3(GRUNetX2):
    """plan-030 GRUNetX2 carry. 유일 변경 = head_hidden 384 → 196 (over-param 완화)."""

    def __init__(self, head_hidden: int = 196, **kwargs):
        super().__init__(head_hidden=head_hidden, **kwargs)
```

기존 GRUNetX2 의 __init__ 가 head_hidden 인자 받음 → 그냥 default 196 으로 호출.

### §3.4 Multi-phase training (`analysis/plan-031/train.py`)

**Phase A — pre (15 epoch)**:
- lr = 7e-4, cosine T_max=15 (warmup 5 ep + cosine 10 ep)
- loss = **soft CE only** (plan-030 carry)
- optimizer = AdamW (wd=1e-4)
- grad_clip = 1.0
- 목적: 기본 GRU + attention representation 학습

**Phase B — fine (35 epoch)**:
- lr = **2e-4** (= 7e-4 / 3.5), cosine T_max=35 (warmup 없음, pre 의 last lr 에서 시작)
- loss = `0.5 * soft_CE + 0.3 * pairwise_margin(margin=0.12) + 0.2 * regime_class_prior(regime=0.65, class=0.45)`
- optimizer = AdamW (wd=1e-4) — pre 의 optimizer state carry (continue)
- grad_clip = 1.0
- 목적: logit sharpening + anchor discrimination + prior 보강

**total 50 epoch** (= 15 pre + 35 fine), plan-030 X1 carry 와 동일 epoch budget.

---

## §4. 합격 기준 + Gate

| gate | 검사 | PASS band | severity |
|---|---|---|---|
| G0 (data) | finite + max_class_ratio < 0.95 | 100% | severe halt |
| G1 (smoke) | pytest test_plan031_smoke.py | 20+ green | warn |
| G2 (1-fold) | fold-0 hit_1cm | **> 0.6330** (F0 +0.001, plan-030 G1 0.6436 의 conservative 60% target) | warn |
| G3 (OOF 5-fold) | 5-fold OOF hit_1cm | **≥ 0.6360 PASS**, 0.6387+ STRONG, 0.65+ EXCELLENT | severe halt if < F0 0.6320 |
| G_final | G3 PASS band + paired permutation p<0.005 vs F0 | PASS | results 박제 + plan-032 spec 진입 |

**G1 threshold 상향 (0.6290 → 0.6330)**: plan-030 의 fold-0 outlier 학습 — single fold PASS 가 5-fold PASS 보장 안 함. conservative threshold.

**G3 FAIL 시 fallback**:
- (i) fine phase lr 추가 감소 (2e-4 → 1e-4) — 1 ablation commit
- (ii) pairwise margin 증가 (0.12 → 0.20) — 1 ablation commit
- (iii) regime/class prior strength 증가 (0.65/0.45 → 0.80/0.60) — 1 ablation commit
- (iv) plan-032 spec 진입 (fine-distill + reverse-pretrain + GPU batch 4096)

---

## §5. Evaluation spec

- metric: hit_1cm (world frame Euclidean < 0.01m), hit_1p5cm
- statistic: paired permutation 10000 resample vs F0, p < 0.005
- artifact: `analysis/plan-031/results_g1.json`, `results_g3.json` + `.npz` (oof_pred + oof_probs)
- report: `plans/plan-031.results.md`

---

## §6. Deferred (plan-032 이후 후보)

| 항목 | 사유 | 예상 lift |
|---|---|---|
| fine-distill (weight 0.55, temp 0.07) | teacher distill 효과, plan-004 carry | +0.005~0.010 |
| reverse-pretrain (BiLSTM 역방향) | sequence representation 보강 | low-medium |
| GPU batch=4096 + hidden=48 (PB 정확 모사) | small hidden + large batch dynamics | medium |
| boundary corrector 2-stage | plan-004 §2.2 carry | +0.0094 (absolute) |
| input axis ablation (잔차 a/b 단독 기여) | PB carry 후 net contribution 측정 | informative |

---

## §7. References

- `analysis/plan-029/paradigm_root_cause.md` — main carrier = training procedure 진단
- `plans/plan-030.results.md` — G3 FAIL_regression 분석 5 layer
- `src/pb_0_6822/selector.py` — PB selector (pairwise + prior + distill source)
- `analysis/plan-030/model.py` — GRUNetX2 (carry)
- `analysis/plan-030/train.py` — train pipeline (carry, 2-phase 분리만 추가)
- `analysis/plan-030/{residual,query,head_summary}_builder.py` — input axis (carry)

decision-note: plan-031 spec 시작 박제. plan-030 input axis carry 유지 + PB training procedure 의 4 lever (multi-phase / pairwise / prior / head slimming) 추가. fine-distill + reverse-pretrain + GPU batch 4096 + boundary corrector 는 §6 deferred (plan-032 후보) 로 분리 — 단일 plan 의 변수 다수 동시 변경 회피.
