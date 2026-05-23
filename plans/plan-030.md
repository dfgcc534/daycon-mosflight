---
plan_id: plan-030
status: written
based_on: plan-029
title: GRU-attention residual injection (3-pair residual block + Q=anchor / K=GRU step)
data_window: train_labels.csv 전체 (plan-024/29 carry)
fold_pin: stable_fold_id(sid, 5) (plan-024 carry, MD5)
horizon: 80ms (2 step × DT=40ms)
---

# plan-030 — GRU-attention residual injection

## §0. 한 줄 목적

plan-029 의 0.6316 regression 을 **3-pair 잔차 (raw / F0 / anchor) block + Q=anchor channel / K=GRU step 정합** 으로 fix. F0 baseline 0.6320 안전 초과 + plan-024 honest ceiling 0.6387 회복.

## §0.5 Quick Reference

| 항목 | 값 |
|---|---|
| paradigm | GRU-attention, anchor residual K=14 (plan-024/29 carry). `final = F0 + R_wfn @ (probs @ ANCHORS_A6)` |
| GRU input (per step, T=7) | seq 95 + 잔차 (a) XY/Z 2 = **97D × 7** |
| attention-K/V (per step, T=7) | GRU output H + 잔차 (a) 5 (raw bottleneck-bypass) = **(H+5) × 7** |
| attention-Q (per anchor, K=14) | anchor_spec 9 + par/perp/dist 3 + interactions 10 + 잔차 (b) 35 + slim 7 = **64D × 14** |
| head MLP sample summary | GRU hidden[-1] H + Bz/Tz 2 + macro 8 + A1 3 + A6 3 + A10 Pct 9 + A12 3 + plan-021 macro 9 + L4 14 = **H + 51** |
| head MLP per-anchor concat | attention context per anchor + slim 7 per anchor |
| K (anchors) | 14 (ANCHORS_A6, plan-022 BCC codebook) |
| GRU hidden | 196 (plan-029 X1 carry) |
| training | 50ep cosine + AdamW lr=7e-4 + batch=64 + soft CE (plan-029 X1 carry, single phase) |
| evaluation | 5-fold OOF stable_fold_id, hit_1cm + hit_1p5cm world frame Euclidean |
| 합격 기준 | G3 OOF hit_1cm ≥ **0.6360** (F0 +0.004, band: PASS). 0.6387+ STRONG. |
| time horizon 정합 | DT=40ms, HORIZON=2 (80ms). 잔차 align F0_pred(t) ↔ raw(t+2). |

### Commit chain (예정)

| commit | spec section | TODO |
|---|---|---|
| c0 spec | §0 ~ §6 | DONE (본 commit) |
| c1 residual builder | §2.1 | [TODO] `analysis/plan-030/residual_builder.py` — 3-pair 잔차 (a) 5coord×7step + (b) 5coord×7step×K 산출 |
| c2 query builder | §2.2 | [TODO] `analysis/plan-030/query_builder.py` — 64D per anchor (slim 7 carry from plan-029) |
| c3 head summary builder | §2.3 | [TODO] `analysis/plan-030/head_summary.py` — sample-only 51D 묶음 |
| c4 model | §3 | [TODO] `analysis/plan-030/model.py` — GRUNetX2 (Q=anchor / K=GRU+잔차 / head=context+sample+slim) |
| c5 train | §4 | [TODO] `analysis/plan-030/train.py` — 5-fold OOF (plan-029 X1 carry) |
| c6 smoke | §5 | [TODO] `tests/test_plan030_smoke.py` — pytest green + finite + max_class_ratio<0.95 |
| c7 G1 1-fold | §5 | [TODO] hit_1cm > F0 −0.003 |
| c8 G3 OOF | §5 | [TODO] OOF 5-fold, hit_1cm ≥ 0.6360 PASS band |
| c9 results | §6 | [TODO] `plans/plan-030.results.md` 박제 |

---

## §1. Input architecture (component 결정 표)

### §1.1 paradigm 표준 멘탈 모델

attention 표준: **Q 는 anchor 별 검증 신호** (K=14 query), **K/V 는 sequence step 별 sample state** (T=7 keys). plan-024 carry.

`final = F0 + R_wfn @ (probs @ ANCHORS_A6)` (anchor residual paradigm).

### §1.2 3-pair 잔차 정의 (baseline 필수)

7 step (t=−8 ~ t=−2) × 5 coord ([XY norm, Z signed, Frenet along/across/vert]) × 2 pair (잔차 (c) drop):

| pair | sample-variant | anchor-variant | dim | input 위치 |
|---|---|---|---|---|
| (a) raw(t+2) − F0_pred(t) | ✓ | ✗ | 7 × 5 = 35 per sample | **GRU input: XY norm + Z signed 2D 만** (Frenet 3축은 seq[16:19] redundant), **attention-K/V: 5D 모두** (raw bottleneck-bypass) |
| (b) raw(t+2) − anchor_world_k(t) | ✓ | ✓ | 7 × 5 = 35 per anchor | **attention-Q (per anchor)** |
| ~~(c) F0 − anchor~~ | — | — | — | **drop** (Frenet 3축 = anchor_spec coord redundant, world XY/Z 는 Bz/Tz 2 와 거의 derivable, N_z sign 1만 신규 — 신호 약함) |

**시간 정합 (필수)**: 각 step t 의 F0_pred(t) 는 그 시점에서 +2 step (80ms) 후 위치 예측 (`analysis/plan-024/cand_builder.py:45-46` DT=0.040, HORIZON=2). task horizon 과 동일.

### §1.3 component 결정 표

| component | 결정 | 근거 |
|---|---|---|
| anchor_embed (14, 8) learnable | **제거** | plan-024 carry 잔재, Q 가 sample-variant 면 불필요 |
| seq_builder 95D × 7 step | **seq → GRU input** | sequential raw, GRU 본연의 자리 |
| GRU output (B, T=7, H) | **attention-K/V (all step)** | step-wise matching 살림, last hidden 압축 안 함 |
| par/perp/dist normalized 3D (sample×anchor) | **attention-Q (per anchor)** | plan-024/29 carry, anchor channel 핵심 신호 |
| **residual block — 2-pair × 5 coord × 7 step** | (a) raw−F0 → **GRU input 2D + attention-K/V 5D**, (b) raw−anchor 35D per-anchor → **attention-Q** | 5 coord = [XY norm, Z signed, Frenet along/across/vert]. (c) drop |
| interactions 10D (anchor·res/v/a/EWMA, corner×turn, sign-agree, physics-extrap, anchor·Δz, A3 BCC adj 2) | **attention-Q (per anchor)** | plan-024/29 carry. 비선형 결합 (dot/sign/physics) 은 잔차 raw 와 별개 신호 |
| plan-029 extension **slim 7D** (D.regime_anchor_prob 1 + B.cos 1 + F.2 5) | **attention-Q + head MLP 둘 다** | prior + alignment + step velocity dot. D.regime_prob 가 score 직결되도록 bottleneck 회피. A.dist 5 / A.tangent 3 = 잔차 (b) 와 redundant 라 drop |
| base EWMA(α=0.3) res Frenet 3D | **drop** | seq[65:68] 에 동일 EWMA(0.3) 있고 GRU 가 압축, target-aligned 아니라 bottleneck-bypass motivation 약함 |
| macro_stat 8D (speed/accel/jerk/turn mean+std) | **head MLP sample summary** | trajectory-level summary, sample-only |
| A1 STA/LTA 3D (EWMA α=0.5/0.1 ratio per Frenet axis) | **head MLP sample summary** | change-detection ratio, sample-only |
| A2 Multi-window 60D (144→60 trimmed) | **drop** | dim 부담 크고 효과 불명 |
| A6 wingbeat-jitter std 3D | **head MLP sample summary** | micro-vibration 수준, sample-only |
| A10 **Pct-rolling 9D 만** (idx 0-8: pct_{20,50,80}(rolling_std(‖v‖, w∈{3,5,7}))) | **head MLP sample summary** | rolling 통계만 유지, Peak 3D (jerk_count/sign_flip/sharp_turn) drop |
| A12 v_autocorr Pearson lag {1,2,3} 3D | **head MLP sample summary** | velocity self-similarity per lag, sample-only |
| plan-021 macro_stat 9D (path/straight/slope/cv/turn/accel/turn_vol/linear_resid/jerk_vol) | **head MLP sample summary** | trajectory geometry summary, path-related 신규 신호 |
| plan-021 EWMA 27D (L1 Frenet 9D × 3α∈{0.1,0.3,0.5}) | **drop** | GRU 가 multi-scale EWMA 자체 학습 가능, dim 부담 |
| plan-025 block ④ seq 8-stat 760D (95ch × 8 stat) | **drop** | GRU 가 seq 직접 처리, 760D redundant + dim 폭증 |
| base last-step v/acc/res Frenet 9D | **drop** (GRU hidden[-1] 로 대체) | raw last-step 9D = GRU hidden[-1] 의 weak version |
| **GRU hidden[-1]** (H dim, non-linear sample summary) | **head MLP sample summary** | sequence 전체 압축. head 에 sample-invariant bias 역할 (attention context 와 별도) |
| Bz/Tz 2D (R_wfn[:,2,2] + R_wfn[:,2,0]) | **head MLP sample summary** | Frenet→world 회전 z-axis 정보, anchor world position 추론 직결 |
| regime 18D one-hot | **drop** | D.regime_anchor_prob (slim 7D 의 일부) 가 이미 regime-conditioned prior 박았으므로 redundant |
| A5 WAP last-step 5D (wing action pattern) | **drop** | micro-action 신호, baseline 에서 우선순위 낮음 |
| A8 f0_conf 2D (residual norm + step spread) | **drop** | 잔차 (a) 35D 가 더 풍부한 raw 정보, A8 은 압축본이라 redundant |
| plan-021 L1 trajectory flatten 99D (11 step × pos/vel/acc 9) | **drop** | GRU input 의 seq 95D 안에 L1 Frenet 이미 포함, flatten 박을 이유 없음 |
| plan-021 L2 F0 residual Frenet flatten 21D (7 step × 3축) | **drop** | 잔차 (a) 35D 의 Frenet 3축 부분과 완전 동일, 부분집합 |
| plan-021 L4 soft hit flatten 14D (7 step × [σ((R_HIT−d)/τ), σ((R_HIT_LOOSE−d)/τ)] = 2) | **head MLP sample summary** | target metric (hit_1cm) 의 soft 버전, score 학습 가속 inductive bias |

### §1.4 input 위치별 누적 dim 요약

| 위치 | 구성 | dim |
|---|---|---|
| GRU input per step | seq 95 + 잔차 (a) XY/Z 2 | 97 (× 7 step) |
| attention-K/V per step | GRU output H + 잔차 (a) 5 | H + 5 (× 7 step) |
| attention-Q per anchor | anchor_spec 9 + par/perp/dist 3 + interactions 10 + 잔차 (b) 35 + slim 7 | 64 (× 14 anchor) |
| head sample summary | GRU hidden[-1] H + (Bz/Tz 2 + macro 8 + A1 3 + A6 3 + A10 Pct 9 + A12 3 + plan-021 macro 9 + L4 14) | H + 51 |
| head per-anchor concat | attention context (per anchor, from attention output) + slim 7 per anchor | (attention dim) + 7 (× 14 anchor) |

---

## §2. Builder spec

### §2.1 `analysis/plan-030/residual_builder.py` (c1 commit)

```python
def build_residuals(
    X: np.ndarray,                    # (N, 11, 3) world raw observations (t=-10 ~ 0)
    R_wfn: np.ndarray,                # (N, 3, 3) Frenet basis
    anchors: np.ndarray,              # (14, 3) ANCHORS_A6 Frenet
    f0_baseline_fn: Callable,         # plan-020 baseline f0 식
    t_range: tuple[int, int] = (2, 9),  # t=-8 ~ t=-2 (7 step), align with raw(t+2)
) -> dict:
    """
    Returns:
        residual_a: (N, 7, 5) — raw(t+2) - F0_pred(t), 5 coord [XY_norm, Z_signed, Frenet along/across/vert]
        residual_a_gru: (N, 7, 2) — XY_norm + Z_signed 만 (GRU input concat 용)
        residual_b: (N, K=14, 7, 5) — raw(t+2) - anchor_world_k(t), per anchor
    """
```

5 coord 분해:
- XY_norm = `√(Δx² + Δy²)`
- Z_signed = `Δz`
- Frenet along/across/vert = `R_wfn^T @ Δ` (3축)

### §2.2 `analysis/plan-030/query_builder.py` (c2 commit)

```python
def build_query(
    cand_feat_150: np.ndarray,        # (N, K=14, 150) plan-024 cand_builder 결과 (par/perp/dist 3 + interactions 10 + anchor_spec 9 + 기타 128)
    residual_b: np.ndarray,           # (N, K=14, 7, 5) from residual_builder
    extension_slim7: np.ndarray,      # (N, K=14, 7) D.regime_prob 1 + B.cos 1 + F.2 5 (plan-029 carry, slim)
) -> np.ndarray:
    """Returns: query (N, K=14, 64) = anchor_spec 9 + par/perp/dist 3 + interactions 10 + 잔차(b) flatten 35 + slim 7"""
```

cand_feat_150 에서 필요한 22D (anchor_spec 9 + par/perp/dist 3 + interactions 10) 만 추출 + 잔차(b) flatten 35 + slim 7 = 64D.

### §2.3 `analysis/plan-030/head_summary.py` (c3 commit)

```python
def build_head_summary(
    cand_feat_150: np.ndarray,        # plan-024 cand_builder 결과 (ctx 일부 추출)
    plan021_macro9: np.ndarray,       # plan-021 macro_stat 9
    soft_hit_L4: np.ndarray,          # plan-021 L4 14
) -> np.ndarray:
    """Returns: head_summary (N, 51) = Bz/Tz 2 + macro 8 + A1 3 + A6 3 + A10_pct 9 + A12 3 + plan-021 macro 9 + L4 14"""
```

cand_feat_150 의 ctx 부분에서 Bz/Tz/macro 8/A1/A6/A10/A12 추출. A10 은 12D 중 첫 9D 만 (Pct-rolling). plan-021 macro 9 + L4 14 는 별도 build.

---

## §3. Model spec (`analysis/plan-030/model.py`, c4 commit)

```python
class GRUNetX2(nn.Module):
    """
    Q = anchor channel (per K=14), K/V = GRU step (per T=7), head = context + sample summary + slim per-anchor.
    """
    def __init__(self, seq_in_dim=97, query_in_dim=64, head_summary_dim=51, slim7_dim=7,
                 hidden=196, attn_dim=128, head_hidden=384, dropout=0.08):
        # GRU: 97 -> 196 (bidirectional 안 함, plan-029 carry)
        # K/V projection: GRU output (B, T, H) concat 잔차 (a) 5 -> (B, T, H+5) -> linear (H+5, attn_dim)
        # Q projection: query 64 -> attn_dim
        # attention: scaled dot-product single head -> (B, K, attn_dim)
        # head: concat [attention context (B, K, attn_dim), sample_summary (B, H+51) broadcast, slim7 (B, K, 7)]
        #       -> MLP (head_hidden) -> 1 -> (B, K) score
        # softmax over K -> probs -> final = F0 + R_wfn @ (probs @ ANCHORS_A6)

    def forward(self, seq, residual_a_gru, residual_a_kv, query, head_summary, slim7) -> torch.Tensor:
        ...
```

architecture 하이퍼파라미터 (plan-029 X1 carry):
- GRU hidden = 196, 1 layer, no bidirection
- attention single head dot-product, attn_dim = 128
- head MLP hidden = 384, 2 layer + SiLU + dropout 0.08
- softmax temperature = 1.0

---

## §4. Training spec (`analysis/plan-030/train.py`, c5 commit)

plan-029 X1 carry (단순 single-phase):

```python
optimizer = AdamW(lr=7e-4, weight_decay=1e-5)
scheduler = SequentialLR([
    LinearLR(start_factor=0.1, end_factor=1.0, total_iters=5),  # warmup 5 epoch
    CosineAnnealingLR(T_max=45)
])
loss = soft_CE(probs, soft_label_tau)  # plan-022 build_soft_label_with_tau (τ=0.001)
epochs = 50
batch = 64
nan_to_num + clip both input and grad
```

5-fold OOF stable_fold_id(sid, 5) (plan-024 carry MD5).

**plan-004 multi-phase / pairwise / prior / distill / reverse-pretrain 은 본 plan 미포함** — `§7 deferred` 로 plan-031 이후 후보.

---

## §5. 합격 기준

### Gate 정의

| gate | 검사 | PASS band | FAIL → action |
|---|---|---|---|
| G0 (data) | 5-fold split + soft_label finite + residual finite | 100% finite, max_class_ratio < 0.95 | data debug |
| G1 (smoke) | pytest test_plan030_smoke.py | 19+ test green | code fix |
| G2 (1-fold) | fold-0 hit_1cm | > F0 −0.003 (= 0.6290) | architecture debug |
| G3 (OOF 5-fold) | 5-fold OOF hit_1cm | **≥ 0.6360 PASS** (F0 +0.004), 0.6387+ STRONG (plan-024 ceiling), 0.65+ EXCELLENT | regression → root cause |
| G_final | G3 PASS band 이상 + paired permutation p < 0.005 vs F0 | PASS | results 박제 + plan-031 spec 진입 |

### Severity

- **severe (halt)**: max_class_ratio ≥ 0.95 (mode collapse), NaN inflood (>1% 샘플), G3 < F0 (회귀)
- **warn (continue)**: G2 < F0 −0.003 (1-fold noise, OOF 확인 필요), training divergence (loss spike but recovered)

---

## §6. Evaluation spec

- **metric**: hit_1cm (world frame Euclidean distance < 0.01m), hit_1p5cm (< 0.015m)
- **statistic**: paired permutation test (10000 resample) vs F0 baseline, p < 0.005 한쪽 검정
- **artifact**: `analysis/plan-030/results_baseline.json` (OOF metric + per-fold metric + ANCHORS_A6 사용 확인), `analysis/plan-030/oof_baseline.npz` (probs + final pred)
- **report**: `plans/plan-030.results.md` (G_final 통과 후)

---

## §7. Deferred (plan-031 이후 후보)

| 항목 | 사유 | 예상 lift |
|---|---|---|
| PB multi-phase training (pre + fine + freeze_fine + epoch_plus) | plan-004 carry 의 main lever (paradigm_root_cause.md) | +0.015~0.020 |
| pairwise margin loss (margin 0.12) | anchor discrimination 강제 | high |
| regime/class prior (strength 0.65 / 0.45) | inductive bias | medium |
| fine-distill (weight 0.55, temp 0.07) | teacher distill | medium |
| reverse-pretrain (BiLSTM 역방향) | low-medium | low-medium |
| GPU batch=4096 + hidden=48 (PB 정확 모사) | small hidden + large batch dynamics | medium |
| boundary corrector 2-stage | plan-004 §2.2 (+0.0094 absolute) | high |
| 이전 step 14 anchor softmax CE scalar | seq[26:40] vocab F 와 거의 redundant | low |
| anchor vocab F 압축 신호 (F top1 prob, entropy 외 다른 측면) | weak | low |

---

## §8. References

- `analysis/plan-024/seq_builder.py` — seq 95D × 7 step build
- `analysis/plan-024/cand_builder.py` — cand 150D 4 묶음 (par/perp/dist + interactions + anchor_spec + ctx)
- `analysis/plan-029/anchor_query_extend.py` — extension 15 (slim 7 추출 source)
- `analysis/plan-022/anchors.py` — `ANCHORS_A6` 14 BCC Frenet codebook
- `analysis/plan-022/selector_only_model.py` — `build_soft_label_with_tau`
- `analysis/plan-021/build_input.py` — macro_stat 9 (`_macro_stat_9d`) + L4 (`_build_L2_L4`)
- `analysis/plan-020/baseline_f0.py` — F0 deterministic baseline
- `src/pb_0_6822/selector.py` — `fit_regime_bins`, `assign_regimes`, `stable_fold_id`
- `notes/fe_axis_24_25_26_27_29.md` — 5-axis FE classification (본 plan 의 component 선정 source)
- `analysis/plan-029/paradigm_root_cause.md` — gap -0.02 root cause (training procedure)

decision-note: plan-030 spec 사용자 directive component-by-component 결정 → §1 표 박제. training procedure 는 plan-029 X1 carry, multi-phase/pairwise/prior/distill 는 §7 deferred (plan-031 후보).
