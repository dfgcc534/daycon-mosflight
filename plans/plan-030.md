---
plan_id: plan-030
status: complete
based_on: plan-029
followed_by: plan-031 (PB training procedure carry — main carrier per paradigm_root_cause.md)
title: GRU-attention residual injection (3-pair residual block + Q=anchor / K=GRU step)
data_window: train_labels.csv 전체 (plan-024/29 carry)
fold_pin: stable_fold_id(sid, 5) (plan-024 carry, MD5)
horizon: 80ms (2 step × DT=40ms)
g3_oof_hit_1cm: 0.6294
g3_band: FAIL_regression
g1_oof_hit_1cm: 0.6436
---

# plan-030 — GRU-attention residual injection

## §0. 한 줄 목적

plan-029 의 0.6316 regression 을 **3-pair 잔차 (raw / F0 / anchor) block + Q=anchor channel / K=GRU step 정합** 으로 fix. F0 baseline 0.6320 안전 초과 + plan-024 honest ceiling 0.6387 회복.

**self-label scope 명료화**: 본 plan 은 **paradigm shift 가 아닌 input feature axis patch over plan-029 X1 carry**. model architecture (GRU 196 + single-head attention 128 + head MLP 384 2 layer) 와 training procedure (50ep cosine + AdamW + soft CE) 는 plan-029 X1 그대로 carry, *신규 기여 = residual_builder + query/head_summary 재구성 + model forward 의 residual_a_kv concat path*. PB multi-phase training (plan-004 main carrier) 은 §7 deferred 로 plan-031 후보.

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

| commit | spec section | status |
|---|---|---|
| c0 spec | §0 ~ §6 | [DONE] 본 commit (rev1 = plan-review-master 5 iter 결과 박제) |
| c1 residual builder | §2.1 | [DONE] `analysis/plan-030/residual_builder.py` — 3-pair 잔차 (a) 5coord×7step + (b) 5coord×7step×K 산출 |
| c2 query builder | §2.2 | [DONE] `analysis/plan-030/query_builder.py` — 64D per anchor (slim 7 carry from plan-029) |
| c3 head summary builder | §2.3 | [DONE] `analysis/plan-030/head_summary.py` — sample-only 51D 묶음 |
| c4 model | §3 | [DONE] `analysis/plan-030/model.py` — GRUNetX2 (Q=anchor / K=GRU+잔차 / head=context+sample+slim) |
| c5 train | §4 | [DONE] `analysis/plan-030/train.py` — 5-fold OOF (plan-029 X1 carry) |
| c6 smoke | §5 | [DONE] `tests/test_plan030_smoke.py` — 20/20 pytest green |
| c7 G1 1-fold | §5 | [DONE] fold-0 hit_1cm = **0.6436** (PASS > 0.6290) |
| c8 G3 OOF + c9 results | §5/§6 | [DONE] OOF 5-fold hit_1cm = **0.6294** — **FAIL_regression** (< F0 0.6320). `plans/plan-030.results.md` 박제. § 5 fallback rule → plan-031 PB training procedure carry escalate |
| c10~c13 ablation | §5 fallback | [DEFERRED → plan-031] main lever (training procedure) 부재 상태에서 input axis ablation 은 정보 가치 낮음. plan-031 PB carry 후 재진행 |

---

## §1. Input architecture (component 결정 표)

### §1.1 paradigm 표준 멘탈 모델

attention 표준: **Q 는 anchor 별 검증 신호** (K=14 query), **K/V 는 sequence step 별 sample state** (T=7 keys). plan-024 carry.

`final = F0 + R_wfn @ (probs @ ANCHORS_A6)` (anchor residual paradigm).

### §1.2 3-pair 잔차 정의 (baseline 필수)

7 step (t=−8 ~ t=−2) × 5 coord ([XY norm, Z signed, Frenet along/across/vert]) × 2 pair (잔차 (c) drop).

**lift evidence**: plan-029 의 cand 165D 에는 (a) raw−F0 의 step 별 시계열이 **부재** (`cand_builder.py` 의 base 12D 안에 last-step res_last 3 만 carry, EWMA(0.3) 3 압축본 1개). plan-004 / plan-021 의 step 별 raw F0 잔차 (L2 21D = 7 step × 3축) 가 PB selector 의 핵심 신호였으나 plan-024/29 paradigm 으로 carry 안 됨. 본 plan 은 잔차 (a) 35D + (b) 35D per anchor 로 그 gap 메움 → expected lift +0.005 ~ +0.01.

**risk disambiguation (paradigm_root_cause.md cite 와의 misalign)**: `analysis/plan-029/paradigm_root_cause.md` 는 plan-004 PB 와 plan-029 E2 의 -0.0202 gap 의 *main carrier* 를 **training procedure** (multi-phase + pairwise + prior + distill + reverse-pretrain) 로 진단. 본 plan 은 *input feature axis* 만 fix (training 은 plan-029 X1 carry, multi-phase 는 §7 deferred 로 분리). 따라서 본 plan 의 +0.005~0.01 lift 는 main carrier 부재로 *PASS band 0.6360 경계선* 위험 — G3 fail 시 §7 deferred 의 PB training procedure 도입 (= plan-031) 으로 진행 필요. lift 의 *상한* 은 plan-021 L2 carry (= 잔차 (a) 의 Frenet 부분집합) 이 PB selector 학습에 미친 단위 기여 추정치 (~+0.01) 에서 유래.

| pair | sample-variant | anchor-variant | dim | input 위치 |
|---|---|---|---|---|
| (a) raw(t+2) − F0_pred(t) | ✓ | ✗ | 7 × 5 = 35 per sample | **GRU input: XY norm + Z signed 2D 만** (Frenet 3축은 seq[16:19] redundant), **attention-K/V: 5D 모두** (raw bottleneck-bypass) |
| (b) raw(t+2) − anchor_world_k(t), `anchor_world_k(t) = F0_pred(t) + R_wfn @ anchors[k]` | ✓ | ✓ | 7 × 5 = 35 per anchor | **attention-Q (per anchor)** |
| ~~(c) F0 − anchor~~ | — | — | — | **drop** (Frenet 3축 = anchor_spec coord redundant, world XY/Z 는 Bz/Tz 2 와 거의 derivable, N_z sign 1만 신규 — 신호 약함) |

**시간 정합 (필수)**: 각 step t 의 F0_pred(t) 는 그 시점에서 +2 step (80ms) 후 위치 예측 (`analysis/plan-024/cand_builder.py:45-46` DT=0.040, HORIZON=2). task horizon 과 동일.

**step ordering ↔ seq align rule**:
- 잔차 (a) 의 **valid 7 step wall-clock window** = t = {−8, −7, ..., −2} (raw(t+2) 가 관측 도메인 t∈[−10,0] 안에 있는 모든 step).
- 단 seq_builder 의 step ordering = t_wall = {−6, −5, ..., 0} (`analysis/plan-024/seq_builder.py` `t_range=(4, 11)` carry).
- **concat align**: GRU input / attention-K/V 의 step i (i=0..6) 의 wall-clock = (−6 + i). 잔차 (a)[i] 의 wall-clock 도 동일하게 (−6 + i) 로 reindex (= 잔차_a_raw_t_eq_minus_8_plus_j 의 j = i + 2).
- valid: i = 0..4 (wall t = −6..−2) → 잔차 (a) 값 채움.
- invalid: i = 5, 6 (wall t = −1, 0) → raw(t+2) ∈ {+1, +2} 미관측 → **zero-pad** (or `nan_to_num(0.0)` 동치).
- 잔차 (b) 도 동일 align rule 적용 (step ordering = seq 와 동일, 마지막 2 step zero-pad).

**R_wfn 시점**: `R_wfn` 은 sample 의 end_idx=10 (= t_wall=0) 기준 Frenet basis 1개 `(N, 3, 3)` (`analysis/plan-024/cand_builder.py:245` carry, step-invariant). 모든 7 step 의 Frenet 분해 (잔차 (a) Frenet 3축, 잔차 (b) Frenet 3축, anchor_world_k(t) 변환) 에 동일 R_wfn 사용 (step 별 local R_wfn 미사용).

### §1.3 component 결정 표

**G3 fail 시 ablation 분리 우선순위** (component 동시 변경이 다수라 단일 G3 OOF 만으로 원인 분리 어려움): §7 deferred 의 PB training procedure carry 와 별개로, G3 fail 시 ablation 1차 우선순위 = (i) 잔차 (a) drop (GRU input+K/V 둘 다 X) → input 단독 ablation, (ii) 잔차 (b) drop (Q per-anchor 35D X) → query 단독 ablation, (iii) slim 7 drop (Q + head 둘 다 X) → extension ablation, (iv) head sample_summary drop (sample-only 51D X) → head 단독 ablation. 각 ablation 은 단일 commit, single fold 1-fold 비교 (G2 비용).


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
| head MLP input per anchor (= 최종 MLP input) | attention context (per anchor) + sample_summary (broadcast K=14 times) + slim 7 per anchor | attn_dim 128 + (H+51=247) + 7 = **382 per anchor** (§3 master spec) |

---

## §2. Builder spec

### §2.1 `analysis/plan-030/residual_builder.py` (c1 commit)

```python
def build_residuals(
    X: np.ndarray,                    # (N, 11, 3) float, world raw observations (t_wall = -10 ~ 0)
    R_wfn: np.ndarray,                # (N, 3, 3) float, end_idx=10 (t_wall=0) Frenet basis (step-invariant)
    anchors: np.ndarray,              # (K=14, 3) float, ANCHORS_A6 Frenet codebook
    f0_baseline_fn: Callable,         # = analysis.plan_020.baseline_f0.f0_baseline (`analysis/plan-020/baseline_f0.py:24-37`)
                                       # signature: f0_baseline(x: np.ndarray (N, T, 3), end_idx: int) -> np.ndarray (N, 3)
                                       # end_idx = 0-based last index (= T-1 일 때 마지막 3 step 사용).
                                       # 내부적으로 x[:, end_idx-2], x[:, end_idx-1], x[:, end_idx] 만 read → T ≥ 3, end_idx ≥ 2 필수.
                                       # 반환 dtype = x dtype 따름 (입력 np.float32 이면 출력도 np.float32, world frame XYZ).
    t_wall_range: tuple[int, int] = (-6, 1),  # seq align: step ordering = t_wall {-6, ..., 0} (i=0..6, length 7)
) -> dict:
    """
    Returns dict with keys:
        "residual_a":     (N, 7, 5) float32 — raw(t+2) - F0_pred(t), 5 coord [XY_norm, Z_signed, Frenet along/across/vert].
                          i=0..4 (t_wall=-6..-2) 값 채움, i=5,6 (t_wall=-1,0) zero-pad (raw(t+2)∈{+1,+2} 미관측).
        "residual_a_gru": (N, 7, 2) float32 — XY_norm + Z_signed 만 (GRU input concat 용, Frenet 3축은 seq[16:19] redundant 라 drop).
        "residual_b":     (N, K=14, 7, 5) float32 — raw(t+2) - anchor_world_k(t), per anchor.
                          anchor_world_k(t) = F0_pred(t) + R_wfn @ anchors[k]  (anchors[k] = Frenet codebook, R_wfn = end_idx=10 frame).

    Downstream consumer mapping (§4 train.py / Dataset collate 책임):
        dict["residual_a_gru"]  →  GRU input concat: seq_97 = concat([seq_95, residual_a_gru], -1)  →  model.forward(seq=seq_97, ...)
        dict["residual_a"]      →  model.forward(residual_a_kv=dict["residual_a"], ...)  (5coord 그대로 K/V concat 용)
        dict["residual_b"]      →  query_builder input  →  query 64D → model.forward(query=query, ...)
    """
    # 7 step loop (seq align):
    # for i in range(7):
    #     t_wall = -6 + i                                    # {-6, ..., 0}
    #     t_idx = t_wall + 10                                # X 의 absolute index, {4, ..., 10}
    #     # F0_pred(t) — plan-021 _build_L2_L4 carry 호출 패턴
    #     sub_x = X[:, t_idx-2:t_idx+1, :]                   # (N, 3, 3) — 3 step context
    #     F0_pred_t = f0_baseline_fn(sub_x, end_idx=2)       # (N, 3) world XYZ at t_wall
    #     # raw(t+2)
    #     t_target_idx = t_idx + 2
    #     if t_target_idx <= 10:
    #         raw_t2 = X[:, t_target_idx, :]                 # (N, 3) world
    #         delta_a = raw_t2 - F0_pred_t                   # (N, 3) world Δ
    #         delta_b_per_k = raw_t2 - (F0_pred_t[:, None, :] + np.einsum("nij,kj->nki", R_wfn, anchors))  # (N, K, 3)
    #     else:
    #         delta_a = np.zeros((N, 3))                     # zero-pad
    #         delta_b_per_k = np.zeros((N, K, 3))
    #     # 5 coord 분해 (R_wfn = end_idx=10 step-invariant)
    #     residual_a[:, i, 0]   = np.linalg.norm(delta_a[:, :2], axis=1)              # XY_norm
    #     residual_a[:, i, 1]   = delta_a[:, 2]                                       # Z_signed
    #     residual_a[:, i, 2:5] = np.einsum("nij,nj->ni", R_wfn.transpose(0,2,1), delta_a)  # Frenet along/across/vert
    #     # residual_b 동일 분해 per anchor (batched einsum over K=14 axis):
    #     residual_b[:, :, i, 0]   = np.linalg.norm(delta_b_per_k[:, :, :2], axis=-1)                              # (N, K) XY_norm
    #     residual_b[:, :, i, 1]   = delta_b_per_k[:, :, 2]                                                        # (N, K) Z_signed
    #     residual_b[:, :, i, 2:5] = np.einsum("nij,nkj->nki", R_wfn.transpose(0,2,1), delta_b_per_k)              # (N, K, 3) Frenet
    # residual_a_gru = residual_a[:, :, :2]                                          # XY_norm + Z_signed only
    # 초기화 정책: residual_a / residual_a_gru / residual_b 모두 `np.zeros(..., dtype=np.float32)` 로 init.
    # invalid step (i=5,6, t_target_idx > 10) 은 별도 처리 없이 zero 유지 (zero-pad 자연 effect).
```

5 coord 분해 식 (R_wfn = end_idx=10, step-invariant):
- `XY_norm = √(Δx² + Δy²)` (world frame, 1D)
- `Z_signed = Δz` (world frame, 1D)
- `Frenet along/across/vert = R_wfn^T @ Δ` (3D)

### §2.2 `analysis/plan-030/query_builder.py` (c2 commit)

```python
def build_query(
    cand_feat_150: np.ndarray,        # (N, K=14, 150) plan-024 cand_builder 결과
    residual_b: np.ndarray,           # (N, K=14, 7, 5) from residual_builder
    extension_slim7: np.ndarray,      # (N, K=14, 7) — upstream 에서 cand_ext_165 (shape (N, K=14, 165)) 의 col [159, 158, 160:165] 추출:
                                       #   D.regime_anchor_prob (cand_ext col 159, 1D) +
                                       #   B.cos                (cand_ext col 158, 1D) +
                                       #   F.2 anchor·v_frenet  (cand_ext col 160:165, 5D) = 7D total
                                       # extract: extension_slim7 = cand_ext_165[:, :, [159, 158, 160, 161, 162, 163, 164]]
                                       # `analysis/plan-029/anchor_query_extend.py:103-138` ordering carry
) -> np.ndarray:
    """Returns: query (N, K=14, 64) per anchor.

    cand_feat_150 slicing (plan-024 cand_builder.py 묶음 ordering carry):
      [0:3]    par/perp/dist          (3D, sample × anchor)
      [3:12]   anchor_spec            (9D, anchor-only broadcast)
      [12:140] ctx 128D               (sample-only broadcast — query 에서 미사용, head_summary 에서 추출)
      [140:150] interactions          (10D, sample × anchor)

    query 64D 구성:
      anchor_spec      = cand_feat_150[:, :, 3:12]          # (N, K, 9)
      par_perp_dist    = cand_feat_150[:, :, 0:3]           # (N, K, 3)
      interactions     = cand_feat_150[:, :, 140:150]       # (N, K, 10)
      residual_b_flat  = residual_b.reshape(N, K, 35)       # (N, K, 35)  — 7 step × 5 coord flatten
      slim7            = extension_slim7                    # (N, K, 7)
      query = concat([anchor_spec, par_perp_dist, interactions, residual_b_flat, slim7], axis=-1)
            # (N, K=14, 9+3+10+35+7 = 64)
    """
```

### §2.3 `analysis/plan-030/head_summary.py` (c3 commit)

```python
def build_head_summary(
    cand_feat_150: np.ndarray,        # (N, K=14, 150) plan-024 cand_builder 결과 (ctx 부분 추출, anchor 차원은 [, 0, ] 1개만 사용 — broadcast)
    plan021_macro9: np.ndarray,       # (N, 9) plan-021 _macro_stat_9d
    soft_hit_L4: np.ndarray,          # (N, 14) plan-021 _build_L2_L4 의 L4 7step × 2 flatten
) -> np.ndarray:
    """Returns: head_summary (N, 51) — sample-only summary (anchor 차원 없음, head 에서 broadcast).

    cand_feat_150 ctx slicing (plan-024 cand_builder.py L397-409 ordering carry):
      ctx_internal_idx     absolute_idx_in_150     content
      [ 0:12]              [12:24]                 base 12 (v_last 3 + a_last 3 + res_last 3 + EWMA(0.3) res 3)
      [12:20]              [24:32]                 macro 8 (plan-024 cand_builder 자체 macro_stat — plan-021 macro_9 와 별개)
      [20:22]              [32:34]                 Bz/Tz 2
      [22:40]              [34:52]                 regime 18 one-hot
      [40:43]              [52:55]                 A1 STA/LTA 3
      [43:103]             [55:115]                A2 multiwindow 60
      [103:108]            [115:120]               A5 WAP 5
      [108:111]            [120:123]               A6 wingbeat-jitter 3
      [111:113]            [123:125]               A8 f0_conf 2
      [113:125]            [125:137]               A10 Pct+Peak 12 (idx 0-8 Pct-rolling = absolute [125:134], idx 9-11 Peak = absolute [134:137])
      [125:128]            [137:140]               A12 v_autocorr 3

    head_summary 51D 구성 (anchor 차원 [, 0, ] 으로 1개만, sample 단위):
      macro_8       = cand_feat_150[:, 0, 24:32]            # (N, 8)
      Bz_Tz         = cand_feat_150[:, 0, 32:34]            # (N, 2)
      A1            = cand_feat_150[:, 0, 52:55]            # (N, 3)
      A6            = cand_feat_150[:, 0, 120:123]          # (N, 3)
      A10_pct       = cand_feat_150[:, 0, 125:134]          # (N, 9)  — Pct-rolling only (Peak 3D drop)
      A12           = cand_feat_150[:, 0, 137:140]          # (N, 3)
      head_summary  = concat([Bz_Tz, macro_8, A1, A6, A10_pct, A12, plan021_macro9, soft_hit_L4], axis=-1)
                    # (N, 2+8+3+3+9+3+9+14 = 51)
    """
```

**Note**: `cand_feat_150` 의 ctx [12:140] 는 14 anchor 행이 모두 같은 broadcast 값 (`analysis/plan-024/cand_builder.py:409`). anchor 차원 0 만 추출 = 모든 anchor 동일 의미.

---

## §3. Model spec (`analysis/plan-030/model.py`, c4 commit)

```python
class GRUNetX2(nn.Module):
    """
    Q = anchor channel (per K=14), K/V = GRU step (per T=7), head = context + sample summary + slim per-anchor.
    F0 = numpy frozen (no gradient) — final = F0 + R_wfn @ (probs @ ANCHORS_A6), R_wfn 도 frozen torch tensor.
    학습 가능 path = GRU + K/V proj + Q proj + attention + head MLP. probs 만 backprop.
    """
    def __init__(self, seq_in_dim=97, query_in_dim=64, head_summary_dim=51, slim7_dim=7,
                 hidden=196, attn_dim=128, head_hidden=384, dropout=0.08, K=14):
        # GRU: seq_in_dim=97 (= seq 95 + residual_a_gru 2) -> hidden=196, 1 layer, no bidirection (plan-029 carry)
        # K/V projection: GRU output (B, T, H) concat residual_a_kv (B, T, 5) -> (B, T, H+5) -> Linear(H+5, attn_dim)
        # Q projection: query (B, K, query_in_dim=64) -> Linear(64, attn_dim)
        # attention: scaled dot-product single head, attn_scores = Q @ K^T / sqrt(attn_dim), softmax over T
        #            attn_context = softmax(attn_scores) @ V   -> (B, K, attn_dim)
        # head MLP input concat (per anchor):
        #     attn_context        : (B, K, attn_dim)
        #     sample_bias         : sample_summary_full (B, H+head_summary_dim).unsqueeze(1).expand(-1, K, -1) → (B, K, H+51)
        #                            where sample_summary_full = concat([gru_hidden_last (B, H), head_summary (B, 51)], -1)
        #     slim7               : (B, K, 7)
        #     head_input          : concat([attn_context, sample_bias, slim7], -1) → (B, K, attn_dim + H + 51 + 7)
        #                          = (B, 14, 128 + 196 + 51 + 7) = (B, 14, 382)
        # head MLP: Linear(382, head_hidden=384) -> SiLU -> Dropout(0.08) -> Linear(head_hidden, 1) -> squeeze(-1) → (B, K=14) score (logit-scale)
        # softmax over K: probs = softmax(score / temperature, dim=K), temperature = 1.0 (logit-scale, label τ_cls 와 별개 — §4 참조)
        # final: world_pred = F0_pred (B, 3) + (R_wfn @ (probs @ ANCHORS_A6).unsqueeze(-1)).squeeze(-1)
        #        where R_wfn (B, 3, 3) frozen, ANCHORS_A6 (K=14, 3) Frenet codebook frozen

    def forward(
        self,
        seq: torch.Tensor,              # (B, T=7, 97) — seq 95 + residual_a_gru 2 concatenated by upstream
        residual_a_kv: torch.Tensor,    # (B, T=7, 5) — for K/V concat (raw bottleneck-bypass)
        query: torch.Tensor,            # (B, K=14, 64) — anchor channel query (from query_builder)
        head_summary: torch.Tensor,     # (B, 51) — sample-only summary (from head_summary builder)
        slim7: torch.Tensor,            # (B, K=14, 7) — plan-029 ext slim per-anchor (head MLP per-anchor concat)
        F0_pred: torch.Tensor,          # (B, 3) — frozen F0 baseline world pred at task target horizon (= t_wall=0 시점에서 +2 step (80ms) 후 위치, `f0_baseline(X, end_idx=10)` 결과)
        R_wfn: torch.Tensor,            # (B, 3, 3) — frozen Frenet basis (end_idx=10)
        ANCHORS_A6: torch.Tensor,       # (K=14, 3) — frozen Frenet codebook
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns: (world_pred (B, 3), probs (B, K=14))"""
        # 1. GRU forward: gru_out (B, T, H), gru_hidden_last = gru_out[:, -1, :] (B, H)
        # 2. K/V build: kv_raw = cat([gru_out, residual_a_kv], -1) (B, T, H+5); K = V = Linear(H+5, attn_dim)(kv_raw)
        #    NOTE: K, V tied (single projection 결과를 K, V 양쪽 사용) — 채택 사유:
        #      (a) attn_dim 128 small + single head 환경에서 parameter 절약,
        #      (b) anchor selection 의 attention 역할이 "step 별 GRU+잔차 상태와 anchor 의 affinity 점수" 산출 → value 가 key 와 동일 representation 이어도 충분 (transformer 의 lookup 역할과 다름).
        #    untied K/V 는 §7 deferred (parameter 2x lift 가치 작음 추정).
        # 3. Q build: Q = Linear(64, attn_dim)(query) (B, K=14, attn_dim)
        # 4. Attention: attn_scores = Q @ K.transpose(-2,-1) / sqrt(attn_dim); attn_w = softmax(attn_scores, dim=-1)
        #               attn_context = attn_w @ V (B, K=14, attn_dim)
        # 5. Head MLP: per above
        # 6. softmax + anchor residual final
        ...
```

architecture 하이퍼파라미터 (plan-029 X1 carry):
- GRU hidden = 196, **2 layer dropout 0.10, no bidirection** (plan-029 X1 carry — `analysis/plan-029/model.py:69-72`)
- attention single head scaled dot-product, attn_dim = 128
- head MLP input dim = attn_dim + H + 51 + 7 = **382**, hidden = 384, 2 layer + SiLU + dropout 0.08
- softmax temperature = 1.0 (score 는 logit-scale). **score magnitude ↔ label τ_cls 결합**: label distribution 의 sharpness 가 τ_cls=0.001 (distance scale ≈ 1mm) 이므로 model probs 가 label 과 정합하려면 logit-score 의 표준편차가 약 `1/τ_cls = 1000` scale 까지 자율 키워야 함. 학습 초기 score variance ≈ N(0, 1) 에서 gradient flow 가 logit magnitude 를 ~1000x 까지 신장하는 *long warmup* 가 필요 — 본 plan 의 warmup 5 epoch + cosine 45 epoch 가 그 신장에 충분한지 G2 (1-fold loss curve) 에서 확인 필수.
- F0 = numpy frozen (no gradient) — gradient path 는 attention/head 만

---

## §4. Training spec (`analysis/plan-030/train.py`, c5 commit)

plan-029 X1 carry (단순 single-phase):

```python
optimizer = AdamW(lr=7e-4, weight_decay=1e-5)
scheduler = SequentialLR([
    LinearLR(start_factor=0.1, end_factor=1.0, total_iters=5),  # warmup 5 epoch
    CosineAnnealingLR(T_max=45)
])
# loss: cross-entropy between model probs (softmax over K=14, temperature=1.0, logit-scale)
#       and soft label distribution (plan-022 build_soft_label_with_tau, τ_cls=0.001 distance-scale)
#       loss = -Σ_k soft_label_k * log(model_probs_k)
#       (model softmax temperature 와 label τ_cls 는 별개 scale — model 은 logit, label 은 거리 기반.
#        둘은 *학습 가능 결합* 으로, model 이 label 의 sharpness 를 학습해서 distribution 정합)
loss = soft_CE(model_probs, soft_label_tau)
epochs = 50
batch = 64
nan_to_num + clip both input and grad
```

5-fold OOF stable_fold_id(sid, 5) (plan-024 carry MD5).

**Dataset `__getitem__` 입력 schema (train.py 의 data pipeline)**:

각 sample 의 raw input 은 다음 (sample-level) artifact 에서 load — 모두 *upstream caching* 결과 (한 번 빌드 후 npz/parquet 캐시):

| artifact | source | shape | dtype |
|---|---|---|---|
| `X` (raw observation) | `data/train.csv` (또는 plan-024 carry preprocess) | (11, 3) | float32, world XYZ |
| `R_wfn` (Frenet basis) | `analysis/plan-024/cand_builder.py:245` 의 build 단계에서 산출 (end_idx=10) → npz cache | (3, 3) | float32 |
| `F0_pred_t_eq_0` (final 용) | `analysis/plan-020/baseline_f0.py:f0_baseline(X, end_idx=10)` **batch precompute** 1회 호출 후 npz cache (N samples 전체 batch (N, 11, 3) → (N, 3) 한 번에) — Dataset 은 idx-th slice 만 reference, single-sample reshape 불필요. **task target horizon = t_wall=+2 (80ms 후) 예측** | (3,) | float32, world XYZ |
| `seq_95` (per step) | `analysis/plan-024/seq_builder.py` build → npz cache | (7, 95) | float32 |
| `cand_feat_150` | `analysis/plan-024/cand_builder.py` build → npz cache | (14, 150) | float32 |
| `cand_ext_165` | `analysis/plan-029/anchor_query_extend.py` build → npz cache | **(14, 165)** | float32 |
| `plan021_macro9` | `analysis/plan-021/build_input.py:_macro_stat_9d` → npz cache | (9,) | float32 |
| `soft_hit_L4_flat` | `_build_L2_L4(...)[1].reshape(14)` → npz cache | (14,) | float32 |
| `gt_world` (label) | `data/train_labels.csv` 의 next +80ms position | (3,) | float64, world |
| `ANCHORS_A6` (frozen) | `analysis/plan-022/anchors.py` ANCHORS_A6 → model `register_buffer` | (14, 3) | float32, Frenet |

Dataset `__getitem__(idx)` 는 위 sample-level artifact 의 idx-th 행 + `build_residuals` 호출 결과 (dict 의 residual_a/residual_a_gru/residual_b) 를 **개별 torch tensor 로 unpack** 하여 dict 반환:
```python
{
    "seq_97":         torch.from_numpy(np.concatenate([seq_95, residual_a_gru], -1)).float(),  # (7, 97)
    "residual_a_kv":  torch.from_numpy(residual_a).float(),                                    # (7, 5)
    "query_64":       torch.from_numpy(query_builder(cand_feat_150, residual_b, slim7)).float(),# (14, 64)
    "head_summary_51":torch.from_numpy(head_summary).float(),                                  # (51,)
    "slim7":          torch.from_numpy(slim7).float(),                                         # (14, 7)
    "F0_pred":        torch.from_numpy(F0_pred_t_eq_0).float(),                                # (3,)
    "R_wfn":          torch.from_numpy(R_wfn).float(),                                         # (3, 3)
    "soft_label":     torch.from_numpy(soft_label).float(),                                    # (14,)
}
```
DataLoader `collate_fn=torch.utils.data.default_collate` (dict 형 자동 stack), 별도 custom collate 불필요.

**R_wfn degenerate sample 처리 (G0 정책)**:
- norm=0 (정지 sample, v_world≈0) 또는 NaN R_wfn 발생 시 → upstream `cand_builder.py` 가 fallback identity matrix 사용 (plan-024 carry). 추가 처리 X.
- 최종 numerical safety: Dataset 단 `np.nan_to_num(R_wfn, nan=0.0, posinf=1e3, neginf=-1e3)` + Frenet 분해 결과 동일 처리.
- G0 검사: residual finite 100% — fail 시 (= NaN 발생) 해당 sample drop (severe halt 아닌 sample-level skip, log 박제).

---

**numpy ↔ torch 변환 책임 요약 (위 Dataset schema 와 일치)**:
- `residual_builder.build_residuals(...)` 반환 dict 의 numpy arrays → Dataset `__getitem__` 단계에서 `torch.from_numpy(arr).float()` 변환 + 단일 sample slice.
- `R_wfn` (N, 3, 3) np.float32 → Dataset 단에서 `torch.from_numpy(R_wfn).float()` → DataLoader collate 후 (B, 3, 3) torch.float32.
- `ANCHORS_A6` (K=14, 3) np.float32 → model `__init__` 단계에서 `self.register_buffer('ANCHORS_A6', torch.from_numpy(anchors).float())` (frozen, gradient X). model.forward 에서 buffer access.
- `F0_pred` (N, 3) np.float32 → Dataset 단에서 sample 별 torch.float32 변환 + collate.
- device: train.py main 에서 `model.to(device)` + DataLoader pin_memory + `batch.to(device, non_blocking=True)`. cuda 가용 시 자동, CPU only 도 동작.

**build_soft_label_with_tau 호출 (label 측)**:
```python
from analysis.plan_022.selector_only_model import build_soft_label_with_tau
# signature: (gt: (N,3) float64 world, R_wfn: (N,3,3), pred_F0_world: (N,3) float64, anchors: (K=14,3), tau_cls: float) -> (N, K) float32
soft_label = build_soft_label_with_tau(gt_world, R_wfn, pred_F0_world, ANCHORS_A6, tau_cls=0.001)  # (N, 14)
```

**soft_CE loss 식 (numerical stability)**:
```python
import torch.nn.functional as F
log_probs = F.log_softmax(score / temperature, dim=-1)             # (B, K=14), numerically stable (logsumexp 내장)
loss = -(soft_label * log_probs).sum(dim=-1).mean()                # scalar
# (raw log(probs) 호출 금지 — probs=0 시 -inf. log_softmax 가 logsumexp 로 안정성 확보)
```

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
- **fallback rule — softmax temp ↔ label τ_cls 신장 실패**: G2 단계 **last epoch (= 50ep 끝) fold-0 valid set** 의 logit-score std < 100 (= 1/τ_cls 의 1/10 미만) 측정 시 → follow-up commit 으로 **learnable temperature scalar τ_model** (`nn.Parameter(torch.tensor(1.0))` 도입, softmax(score/τ_model) 학습) 추가. (= score magnitude 자연 신장 실패 시 명시적 learnable scale 로 우회).
- **fallback rule — G3 borderline (0.6320 ≤ hit_1cm < 0.6360)**: §1.3 ablation 4단 (i~iv) 을 follow-up commit c10~c13 으로 진행 → 각 ablation 의 단일 fold 결과로 어느 component drop 이 main lift 인지 isolation. G3 PASS 도달 못 하면 plan-031 spec 진입 (= PB training procedure carry).

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
