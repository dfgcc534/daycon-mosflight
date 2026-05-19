# 모기 궤적 예측: 미실험 아이디어 카탈로그

> plan-001 ~ plan-022 실행 이후 **아직 실험되지 않은** 후보만 모은 통합본.
> 원본: `new-ideas.md` + `prior-ideas.md` + `mosquito-trajectory-ideas.md` + `optiver_lanl_missing_details.md` (통합 후 삭제됨).
> 현 baseline:
> - **plan-004** PB_0.6822 framework full-fit LB **0.6822**
> - **plan-006** Variant E (physics_bias + soft averaging) LB **0.6692**
> - **plan-020** C05 per-regime F0 hit@1cm **0.6503** (Δ +0.0183)
> - **plan-021** A LGBM corrector hit@1cm **0.6488** (Δ +0.0168)
> - **plan-022** A6_bcc14_τ001 corrector-free hit@1cm **0.6528** (Δ +0.0208)

데이터 shape: 11 step × 40ms regular grid, +80ms horizon, ~10K samples, Hit Rate 평가.

---

## 미실험 후보 ROI 요약

| Tier | 아이디어 | 출처 | 예상 ΔOOF | Cost |
|---|---|---|---|---|
| ★★★ | Trajectory-CLIP latent KNN (dual-encoder InfoNCE) | new D.3 | +0.010~0.020 | 중 |
| ★★★ | KNN-augmented 27-pool (coord-KNN candidates) | new B.1.1 | +0.005~0.015 | 소 |
| ★★★ | Variance-aware 27-cand MDN | new C.1' | +0.005~0.010 | 소~중 (F3/F4 fix 후) |
| ★★ | 3-channel selector ensemble (pos/vel/acc) | Learning-IMM 2025 | +0.003~0.006 | 1.5d + 학습×3 |
| ★★ | Multi-parse input (raw/SG/EMA) | MTP 2022 | +0.002~0.004 | 0.5d |
| ★★ | Physics conservation jerk regularizer | CPhy-ML 2024 | +0.001~0.003 | 0.5d + 학습×4 |
| ★★ | Path × accel reparameterization (28 후보) | PTNet 2021 | +0.001~0.003 | 2d (risk) |
| ★★ | EB shrinkage grid search | EB-Trajectory 2022 | +0.001 | 학습×6 |
| ★★ | Path Signatures (signatory) input alt | new C.2 | +0.002~0.008 | 1d |
| ★★ | SE(3) Lie group corrector module | new B.2 | +0.002~0.005 | 소~중 |
| ★ | 5×5×5 Voxel CE narrow-window head | new D.1 + plan-017 G2 revisit | +0.005~0.012 | 소~중 |
| ★ | PointNet 점군 패러다임 전환 | mosquito Main 7 | 불확실 | 중 |
| ★ | TTA (Z축 회전 / Y축 flip) | mosquito Supp 6 | +0.002~0.005 | 0.5d |
| ★ | Hard-sample fine-tune (OHEM 정신) | engineering | +0.001~0.003 | 0.5d |
| ★ | Cascade 2-stage corrector | engineering | +0.001~0.004 | 1d |
| ★ | VQ-Trajectory codebook | new D.2 | +0.002~0.005 | 중 (ensemble only) |
| ★ | IRM / Domain-adversarial | mosquito Supp 4 | 불확실 | 중 |
| ★ | GMM (NN) mode-seeking output | mosquito Main 2 | F3/F4 parity 선행 필요 | 중 |
| Skip | FNO/FEDformer | new A.2 | ~0 (N=11 fatal) | 대 |
| Skip | Neural ODE | new A.3 | ~0 (regular grid) | 대 |
| Skip | Koopman | new B.3 | ~0 (data-rich 위배) | 대 |

---

## ★★★ Tier — main path 후보 (대형 paradigm shift)

### 1. Trajectory-CLIP / Dual-Encoder InfoNCE — `D.3`

**개념**: Encoder A (past 11-step) + Encoder B (future +80ms displacement) → InfoNCE: same-sample (past, future) pair 가까이, 다른 sample pair 멀게. 학습 후 Encoder B drop, Encoder A latent 위 KNN retrieval.

**현 task 쓸모**:
- ✅ InfoNCE > heuristic SupCon. "미래 변위 유사 의 quantization" 문제를 same-sample positive pair 로 우아하게 해결.
- ✅ Encoder A latent = "past 가 *predictive of similar future*" 인 sample 끼리 모임 = retrieval mechanism 본질 정렬.
- ✅ 학습 후 Encoder B drop → 추론 cost = SupCon-KNN 와 동일.
- ⚠️ Single-positive InfoNCE 의 multimodality 한계: 같은 past 가 좌/우 분기 미래 (Lévy) 갖는 경우 latent 가 mean displacement 로 collapse 가능. 완화책: 추론 시 k=5~10 nearest retrieval + 후보 pool 추가 (B.1.1 결합).

**구현 spec**:
- Encoder A: CNN1D(64) on (B, T=11, F) → latent z_a (128-dim)
- Encoder B: MLP on Δ (B, 3) → z_b (128-dim)
- InfoNCE temp τ=0.07, batch 256~512
- 학습 후 Faiss IndexFlatIP on z_a (cosine = L2-norm 후 inner-product)
- 추론: query z_a → top-k=5~10 train sample → 각 Δ 가 후보 pool element

**합격 기준**: fold-0 OOF ≥ 0.6500 (plan-011 ID 0.6450 + 0.005). selector_logit 분포에서 CLIP-KNN 후보의 avg log-prob ≥ Frenet 후보의 60%.

**미실험 사유**: plan-022 까지 모든 retrieval 계열 미진입 (Frenet anchor sweep 만).

---

### 2. KNN-Augmented 27-Candidate Pool — `B.1.1`

**개념**: 기존 plan-008 의 selector framework + 27-candidate pool **골격 보존**. `selector.make_candidates()` 에 coord-level KNN-displacement 후보 k=3~5 개 추가. Selector 가 Frenet vs KNN 후보 선택을 자연스럽게 학습.

**구체 spec**:

```python
# src/pb_0_6822/knn_candidates.py (신규)
# Step 1: train pool 구축 (한 번만, fold 별 separate index)
#   - 모든 train sample 의 11-step trajectory 를 (v_last frame) 정규화
#     normalize: p0=0 (last position), v_last → +x axis 회전 (yaw only)
#   - 정규화 trajectory flatten 33-dim
#   - +80ms 정답 displacement (3-dim) 도 same frame 정규화
#   - Faiss IndexFlatL2 로 33-dim 색인

def make_knn_candidates(x, end_idx, faiss_index, train_targets, k=5):
    """x:(B,T,3), end_idx:보통 10 → (B,k,3) raw frame candidate"""
    p0 = x[:, end_idx]
    v_last = x[:, end_idx] - x[:, end_idx-1]
    R = build_rotation(v_last)               # v_last → +x
    x_norm = (x - p0[:, None]) @ R
    query = x_norm.flatten(start_dim=1)      # (B, 33)
    _, knn_idx = faiss_index.search(query, k)
    delta_norm = train_targets[knn_idx]      # (B, k, 3)
    delta_raw = delta_norm @ R.transpose(-2, -1)
    return p0[:, None] + delta_raw

# Step 3: 기존 27-pool 에 append
#   knn_cands = make_knn_candidates(x, 10, faiss_idx, train_tgt, k=5)
#   candidates = torch.cat([candidates, knn_cands], dim=1)  # (B, 27+k, 3)
# Selector head output dim 27 → 32 로 확장.
```

**Hyperparam grid**:
- k ∈ {3, 5, 8}
- distance ∈ {L2, DTW (fastdtw)} — DTW 는 cost ↑↑ 이므로 L2 먼저
- query 정규화: yaw only (pitch over-rotate 위험)
- train target leakage 방지: fold 별 separate Faiss index
- displacement clustering pre-check: K-means k=5~10 silhouette ≥ 0.3 만 통과

**합격 기준**:
- (a) fold-0 OOF ≥ 0.6500
- (b) selector_logit 분포에서 KNN 후보 avg log-prob ≥ Frenet 후보의 60%

**Risk**:
- displacement clustering silhouette < 0.3 → fatal hypothesis 확정 → 후보 폐기 (early kill)
- 27 → 32 pool 확장 시 plan-004 checkpoint state_dict 부분 load 필요 (frozen GRU + new head)

**구현 비용**: 1~2 day.

**미실험 사유**: plan-008 candidate pool greedy set-cover expansion 은 시도됐으나 KNN retrieval-based 후보 자체는 미진입.

---

### 3. Variance-Aware 27-Cand MDN — `C.1'`

**개념**: 출력을 (x,y,z) 좌표 대신 **selector 의 27 mode 위 per-candidate variance σ_k 학습** + mixture density argmax. argmax(π_k · N(μ_k, σ_k)) 으로 mean 회귀 회피.

**왜 27 위에서 하나**:
- plan-008 의 27-cand selector + argmax = 이미 *discrete* MDN. argmax(π_k) over 27 fixed modes.
- 연속 MDN 의 marginal 가치 = 27 mode 가 cover 못 하는 영역 도달. 그러나 27 위에 σ 학습이 minimal-change variant.
- 기존 골격 보존하면서 mode-seeking 도입.

**선행 조건**: plan-011 F4 (LearnableSingleCandidate) 의 F3/F4 parity bug fix 필요 (OOF 0.0980/0.0322 catastrophic).

**구현 spec**:
- selector logits → softmax → π_k (27 mode 분포)
- 각 후보 위에 per-cand σ_k head (3-dim, diagonal cov)
- loss: NLL(Σ_k π_k · N(target | μ_k, σ_k))
- inference: argmax(π_k / σ_k³) — density mode

**Risk**: MDN mode collapse 표준 failure. N=11 + 10K + 27-mode → π_k 가 한 component 로 collapse 흔함. mode collapse monitoring 필수 (entropy(π) ≥ 1.0 nat).

**미실험 사유**: plan-011 F3/F4 parity bug 으로 axis 미진입.

---

## ★★ Tier — paper-derived 보조 trick

### 4. 3-Channel Selector Ensemble (Learning-IMM 2025)

**핵심**: Position / velocity / acceleration **세 채널을 별도 Transformer/GRU 로 병렬 예측** → 학습된 transition probability 로 융합.

**대회 적용**:
```python
channel_pos = [x, y, z deltas]
channel_vel = [speed, prev_speed_ratio, turn_cos]
channel_acc = [acc_norm, jerk, curvature]

selector_pos = AttnGRU(channel_pos) → logits_pos
selector_vel = AttnGRU(channel_vel) → logits_vel
selector_acc = AttnGRU(channel_acc) → logits_acc

logits = w_p · logits_pos + w_v · logits_vel + w_a · logits_acc
```

가벼운 버전: 같은 selector 를 input feature subset 3 종으로 재학습 → seed ensemble 처럼 OOF 평균.

**예상 이득**: +0.003~+0.006.

**Cost**: 1.5d + 학습 ×3. channel별 hidden 32~48 로 underfitting 회피.

**미실험 사유**: plan-003 residual GRU 가 ablation 했으나 *3-channel split* 형태는 아님.

---

### 5. Multi-Parse Input (MTP 2022)

**핵심**: 11 frame 좌표를 **3 종으로 사전 가공** 후 selector 에 통과, logit 평균.

```python
x_raw = trajectory
x_sg  = savgol_filter(trajectory, 5, 2, axis=time)
x_ema = ema_smooth(trajectory, alpha=0.6)
logits = (selector(x_raw) + selector(x_sg) + selector(x_ema)) / 3
```

**예상 이득**: +0.002~+0.004.

**Cost**: 0.5d (학습은 raw 만 해도 됨, parse 는 inference-time augmentation). 학습 시 random parse 선택을 augmentation 으로 쓰면 효과 ↑.

**미실험 사유**: smoothing-based augmentation 미진입.

---

### 6. Physics Conservation Jerk Regularizer (CPhy-ML 2024)

**핵심**: Tiny corrector 출력 `Δx` 가 kinematically implausible 한 경우 penalty.

```python
typical_jerk_step = 0.004  # 4mm — train fold 99-quantile 로 보정
delta_jerk = compute_jerk_norm(delta, recent_acc, recent_jerk)
physics_penalty = max(0, delta_jerk - typical_jerk_step) ** 2
loss = boundary_weighted_loss + λ · physics_penalty
```

`λ ∈ {0.1, 0.3, 1.0, 3.0}` OOF tuning.

**예상 이득**: +0.001~+0.003.

**Risk**: λ 큰 경우 corrector 0 collapse (zero-init 이라 이미 보수적). typical_jerk_step 은 train fold 99-quantile 로 fold-consistent 하게.

**미실험 사유**: plan-015 feature expansion (jerk) 은 input 측이고 loss-side regularizer 는 미진입.

---

### 7. Path × Accel Reparameterization (PTNet 2021)

**핵심**: 27 candidate 을 **7 path × 4 accel = 28** 로 재구성. 후보 의미가 path-acceleration 평면에 정렬 → selector 학습 쉬워짐.

```python
paths = [
    (par=1.0, perp=0.0),    # 직진
    (par=0.9, perp=±0.2),   # 약좌/우선
    (par=0.7, perp=±0.4),   # 큰 좌/우선
    (par=1.1, perp=0.0),    # 가속 직진
    (par=0.5, perp=0.0),    # 감속 직진
]
accels = [-0.3, 0.0, 0.3, 0.6]
```

**예상 이득**: +0.001~+0.003 (oracle 유지 시).

**Cost**: 2d 전면 재작성. **다른 trick 다 끝낸 후 시도**. oracle hit rate 27 대비 떨어지면 path/accel 수 증가, 그래도 안 되면 27 그대로.

**미실험 사유**: plan-022 anchor layout sweep 은 7 layout × 3 τ 인데 path×accel 평면 분해는 아님.

---

### 8. EB Shrinkage Grid Search

**핵심**: PB_0.6822 `candidate_regime_bias()` 의 `shrink=18.0` 고정값 → grid search.

```python
def candidate_regime_bias(candidates, target, regimes, regime_count, shrink=18.0):
    alpha = float(np.sum(mask) / (np.sum(mask) + shrink))
# shrink ∈ {6, 12, 18, 30, 50, 100}, regime_prior_strength 와 2D grid 권장.
```

**예상 이득**: +0.001 (existing 메커니즘 하이퍼튜닝).

**Cost**: 학습 ×6~36 (regime bias 만 다시 계산, selector 재학습 불필요할 가능성).

**미실험 사유**: plan-007 CMA-ES 가 다른 hyperparam 중심, regime_bias shrink/strength 2D grid 는 미진입.

---

### 9. Path Signatures (signatory) Input Alternative — `C.2`

**개념**: 11-step trajectory 의 iterated integral 을 fixed-size signature feature (truncation level k 까지) 로 추출.

**현 task 쓸모**:
- ✅ N=11 + truncation k=2~3 → 39~120 features = fixed-size. CNN1D 와 차원 비교 가능.
- ✅ Lead-lag 변환 추가 시 LiDAR jittering noise robustness 일부 확보 (낮은 order 한정).
- ⚠️ "적분이라 노이즈에 강함" 은 order 1~2 에서만 참. Order 3+ = noise amplification.
- ⚠️ Signature 는 temporal ordering 의 일부 정보 손실 — 마지막 위치 p0 별도 concat 필수.
- ✅ `signatory` lib 안정적, 구현 1d.

**Verdict**: plan-012/021 In axis 의 alt sub-experiment slot. 단독 main 부족, **CNN-encoder + signature concat** hybrid 가 ROI 최대화.

**미실험 사유**: plan-011 In axis CNN 64-dim 가 best 였으나 signature 와 비교/병합 미진입.

---

### 10. SE(3) Lie Group Corrector Module — `B.2`

**개념**: 각 시점간 움직임을 SE(3) 변환으로 정의 → se(3) 공간으로 log-매핑 → 선형화된 속도/회전 벡터에 최근 가중치 → exp 매핑 으로 +80ms 변환 추정.

**현 task 쓸모**:
- ⚠️ Tangent 에서 유도한 SE(3) = 본질적으로 Frenet frame 과 동등. plan-006 frenet_par120_perp_neg020 (LB 0.6692) 이 이미 SE(3) 외삽의 단순화 버전.
- ✅ Frenet → SE(3) marginal 확장 = explicit angular velocity smoothing (se(3) angular component).
- ⚠️ Lévy flight 급선회 = log-linearization smooth 가정 위배.
- ✅ Implementation 가벼움 (`pypose.SE3`, `liegroups`).

**Verdict**: plan-021 의 corrector 위 보조 module 로 가치. Frenet 외삽의 anisotropic perp scaling 을 SE(3) angular weighting 으로 대체.

**예상 ΔOOF**: +0.002~+0.005.

**미실험 사유**: plan-020 17 후보 중 R 항목으로 deterministic 평가는 됐으나 corrector 보조 module 형태 미진입.

---

## ★★ Tier B — Optiver/LANL feature engineering (plan-021 LGBM follow-up)

> plan-021 LGBM 170D (hit@1cm 0.6488, partial PASS — 1.5cm 미달) 위에 얹을 미반영 12 항목.
> 출처: Optiver Realized Volatility 2021 1st (nyanp) + LANL Earthquake 2019 1st (Philipp Singer).

### Stage 1 — cost trivial 즉시 추가 (기대 +0.005~0.013)

**A4. LGBM + GRU ensemble**: plan-021 의 sub-exp A (LGBM 170D) + B (GRU 134D) 단순 평균 또는 rank-blend. weight grid {0.3, 0.5, 0.7} CV. 두 paradigm 의 magnitude/direction 보완. 기대 +0.003~+0.008.

**A6. WAP-style composite physical feature**: `|v|²·κ` (원심력), `|j|/|a|` (jerk-to-accel ratio), `½|v|²` (운동 E), `|v|·τ` (out-of-plane drift). Frenet decomposition 의 cross-channel 물리적 곱 부재 보완. ~6-8 feature.

**B3. STA/LTA ratio**: short-term avg (최근 3 step) / long-term avg (전체 11 step). L6 EWMA 의 `α=0.5/α=0.1` ratio = 정확히 STA/LTA. 9 channel × 3 ratio = 27D. 비용 0 (이미 EWMA 27D 있음, division 만).

**B4. Peak counts**: `|jerk|` local maxima 개수, `velocity_x` sign-flip 개수, turning_angle 큰 변화 횟수. L5 macro 가 percentile 만 — count statistic 부재 (percentile/count 직교). ~6-8 D.

### Stage 2 — 본격 추가 (기대 +0.010~0.020)

**A1. Multi-window nested aggregation grid**: trajectory 11점을 `[전체 11], [뒤 7], [뒤 5], [뒤 3]` 4 sub-window 로 나누고 각 window 에서 `{mean, std, max, min, sum_sq, slope, last-first}` 7 통계. Frenet (p,v,a) 9-channel × 7-stat × 3-subwindow → ~135 D (중복 trim 후 ~60D). L6 EWMA 의 exponential decay 와 직교 (직사각형 sub-window × 다양한 통계 grid).

**B2. Percentile-of-rolling-std**: 2단 nested aggregation — 안쪽 `std(x, window=w)` 시간열 → 바깥쪽 `pct_{20,50,80}` 적용. axis × `pct_{20,50,80}(std(speed_mag, w={3,5,7}))` = 27D. "변동성의 변동성" — saccade burst 강도의 일관성. Singer LANL top-20 importance 반복.

**C2. Multi-output joint (axis correlation)**: plan-021 의 21 LGBM regressor (7 anchor × 3 axis) 가 완전 독립 학습 → axis 간 correlation 학습 X (caveat 5 박제됨). multi-output GBDT (lightgbm multi-output 또는 catboost MultiRMSE) 또는 axis-pair joint feature `x·y, y·z, x·z, |xy|, |yz|`. F0 잔차의 방향성 회복.

### Stage 3 — paradigm shift (기대 +0.005~+0.012)

**A3. Nearest-neighbor target-mean encoding (nyanp signature)**: trajectory 의 motion signature (평균 속도 + 평균 곡률 + 평균 z) latent embedding → fold-out-of-fold 에서 같은 cluster trajectory 의 target offset 평균을 feature. k-NN (k=20-50) target-mean encoding, fold-internal leakage-safe. "비슷한 motion pattern → 비슷한 F0 보정량" cluster prior. **leakage 방지 까다로움**.

**A2. Cross-sample aggregation**: 같은 regime cluster 안의 다른 trajectory feature 평균과의 차이 (regime-mean macro_stat z-score). fold-wise OOF 로 regime-mean 계산. 경계 sample vs 전형 sample 의 F0 신뢰도 차이.

**C1. Log-target / robust loss**: target = 3D vector, magnitude long-tail (saccade 직후 큰 offset). target → (방향, log(1+|magnitude|)) 분해 또는 LGBM `objective='huber'/'quantile'`. cruise (90%) + saccade (10%) 균형 학습. **1cm hit + 1.5cm hit 둘 다 PASS (partial PASS 극복) 후보**.

**B1. Massive feature count (170D → 600D+)**: Singer LANL 1st 가 ~1000D hand-crafted, LGBM `feature_fraction=0.3-0.5` 로 redundant 견딤. S1~S4 + A1~A6 누적 시 자동 달성. `feature_fraction`, `lambda_l2` 조정 필수.

**B7. Hand-crafted ≫ 자동 FE 검증 박제**: catch22 viable subset (~8D) 한번 돌려보고 hand-crafted 와 CV 비교 박제. paradigm 확신 + caveat 박제.

### Optiver/LANL 12 항목 ranking

| Rank | 항목 | 기대 Δhit@1cm | 비용 |
|---|---|---|---|
| 1 | A4 LGBM+GRU ensemble | +0.003~0.008 | trivial |
| 2 | A1 Multi-window stat grid | +0.005~0.010 | 중 (+60D) |
| 3 | C2 Multi-output joint | +0.003~0.006 | 작 |
| 4 | B1 Massive 170D → 600D+ | +0.002~0.005 | 작 |
| 5 | A3 NN target-mean | +0.002~0.006 | 큼 (leakage-safe) |
| 6 | B2 Pct-of-rolling-std | +0.002~0.004 | 작 (+27D) |
| 7 | C1 Log-target / Huber | +0.001~0.003 | 작 |
| 8 | A2 Cross-sample agg | +0.001~0.003 | 중 (OOF) |
| 9 | B3 STA/LTA ratio | +0.001~0.003 | trivial |
| 10 | B4 Peak counts | +0.001~0.002 | trivial |
| 11 | A6 WAP composite | +0.001~0.003 | trivial |
| 12 | B7 Hand vs auto 박제 | 0 (methodology) | 작 |

→ plan-021 LGBM 0.6488 → 잠재 0.66~0.68 권 (plan-004 0.6806 침투 가능 영역).

**Reference**:
- Optiver 2021 1st (nyanp): kaggle.com/competitions/optiver-realized-volatility-prediction/writeups/nyanp-1st-place-solution
- LANL 2019 1st (Singer): medium.com/@ph_singer/1st-place-in-kaggle-lanl-earthquake-prediction-competition

---

## ★ Tier — diversity / engineering trick

### 11. 5×5×5 Voxel CE Narrow-Window Head — `D.1` revisit

**plan-017 G2** 에서 3,375-class voxel CE 가 −0.0121 FAIL. 그러나 **5×5×5 = 125 class** (selector best candidate 위 ±2.5cm window) 는 미실험.

- ✅ Hit-rate ↔ argmax voxel 직접 정렬.
- ✅ Multimodal softmax 자연 처리.
- ✅ CE loss 안정.
- ⚠️ window 가 hyperparameter — plan-006 CV 외삽의 99 percentile error 측정 후 조정.
- ✅ selector coarse pick 유지 + voxel head 를 corrector regression head **대체**.

**예상 ΔOOF**: +0.005~+0.012.

---

### 12. PointNet 점군 패러다임 — mosquito Main 7

11 개 3D 좌표를 *작은 점군* 으로 입력. 시간 순서 무시, 공간 형태 학습.
- 짧은 sequence (11) 에서 RNN 한계 회피.
- 노이즈 강한 LiDAR 강건.
- ⚠️ 시간 정보 무시 → +80ms 외삽엔 mathematically ill-posed. **시간 PE concat** 또는 PointNet++ 의 temporal grouping 필요.

**미실험 사유**: 모든 plan 이 sequence-paradigm. spatial-paradigm 미진입.

---

### 13. TTA — Z-rot / Y-flip — mosquito Supp 6

추론 시 입력을 여러 각도로 변형 → 모델 통과 → 원래 좌표계 복원 후 평균.
- **Z축 (상향, 중력)**: 건드리면 안 됨.
- **X-Y 평면 회전 + Y축 Flip**: 물리적으로 동일한 비행 패턴 → 안전한 증강.

**예상 이득**: +0.002~+0.005. **Cost**: 0.5d.

**미실험 사유**: 학습 시 augmentation 은 일부 plan 에서 시도, **inference-time TTA** 는 미진입.

---

### 14. Hard-Sample Fine-Tune (OHEM 정신)

학습 중간에 OOF error 큰 sample 만 추출 → 가중치 ↑ 로 short epoch 추가 학습.

**예상 이득**: +0.001~+0.003. **Cost**: 0.5d.

**Risk**: hard sample = 진짜 outlier (LiDAR 노이즈) 일 수 있어 overfit. OHEM 비율 ≤ 20% 권장.

---

### 15. Cascade 2-Stage Corrector

Stage 1 corrector 출력을 Stage 2 corrector 의 input residual 로 재사용. 잔차의 잔차 학습.

**예상 이득**: +0.001~+0.004. **Cost**: 1d.

**Risk**: Stage 2 가 noise 학습 시 LB 하락. zero-init + early stop 필수.

---

### 16. VQ-Trajectory Codebook — `D.2`

VQ-VAE 구조 — 11-step trajectory → encoder → K=256 codebook 중 nearest 로 quantize.

- ✅ Maneuver primitive discovery (interpretability ↑).
- ✅ O(1) lookup.
- ⚠️ VQ-VAE 학습 난이도 (commitment loss + EMA codebook + dead-entry reset).
- ⚠️ K=256 < 10K = 정보 손실. 같은 codebook 매핑 sample 의 displacement 평균 → multimodal collapse.
- ⚠️ 연속 latent + Faiss 10K retrieval (Trajectory-CLIP) >> 이산 latent + 256-cell lookup.

**Verdict**: Trajectory-CLIP 의 strict downgrade. **plan-013 ensemble diversity member** (이산 prototype vs 연속 retrieval inductive bias 차이) 로만 가치.

**예상 ΔOOF**: +0.002~+0.005 (ensemble only).

---

### 17. IRM / Domain-Adversarial Training — mosquito Supp 4

실내/야외 등 환경이 섞인 데이터 대응. 11개 시점 분산/속도 통계로 환경 클러스터링 후 멀티태스크 학습 또는 domain-adversarial discriminator.

**미실험 사유**: domain shift 분석 자체가 plan 단계에서 부재. plan-005 진단에 regime 단위 stratification 만 있고 domain 단위 절단 없음.

---

### 18. GMM (Continuous NN) Mode-Seeking Output — mosquito Main 2

단일 좌표 대신 GMM 파라미터 (π_k, μ_k, Σ_k) 출력. Inference 시 density mode 선택.

**선행 조건**: plan-011 F3/F4 parity bug fix 필요. **C.1' (★★★ 3) 가 minimal-change variant** — 27 위에서 σ 학습.

**Verdict**: F4 LearnableSingleCandidate 의 multi-modal 일반화. 연속 MDN 단독은 ★★ (mode collapse risk), C.1' 27-위 variance 는 ★★★.

---

## Skip 확정 (왜 안 하는지 박제)

### S1. FNO / FEDformer — `A.2`
- ❌ N=11 → FFT 무의미. Nyquist 까지 6 frequency bin, amplitude estimate variance 폭주.
- ❌ Wingbeat aliasing (모기 300-700Hz × 25Hz 샘플링) 11 sample 로 phase 복원 불가.
- ❌ 10K × 11 = operator learning 필요 scale 의 1/10~1/100.
- **Skip 확정**: 데이터 길이가 fatal.

### S2. Neural ODE — `A.3`
- ❌ regular 40ms grid → Neural ODE main advantage (irregular timestamp) 무효.
- ❌ 11-step initial trajectory → ODE underdetermined.
- ❌ Lévy flight 급선회 = discrete event → ODE smooth dynamics 가정 위배.
- ⚠️ "Spline polynomial-explosion" 은 이미 plan-006 Frenet 외삽이 해결.
- **Skip 확정** (primary). ensemble smoothing prior 로는 미미.

### S3. Koopman Operator — `B.3`
- ❌ N=11 → 10 transitions 으로 d×d A (d=64~128) = severely underdetermined.
- ⚠️ Global Koopman = 평균값 회귀. Sample-conditioned = GRU/LSTM simpler linear form.
- ⚠️ 2-step horizon 에서 numerical advantage 무시 가능.
- **Skip 확정**: 데이터-풍부 linear regime 가정 위반.

---

## 부분 평가됨 (plan-020 17 후보 안에 deterministic 평가 포함, 본격 적용은 미진입)

| 항목 | plan-020 평가 결과 | 본격 후속 가능 형태 |
|---|---|---|
| N. KNN trajectory matching | 17후보 중 deterministic 형태 | **B.1.1 (★★★ 2)** 으로 격상 |
| S. IMM-KF physics filter | 17후보 중 평가 | plan-013 ensemble member (★★) |
| R. SE(3) Lie group | 17후보 중 deterministic | **B.2 (★★ 10)** corrector 보조 |
| JJ. Lévy flight prior | 17후보 중 평가 | 분포 출력 형태 결합 (NN 와) |
| X. MoE / behavior classification | 17후보 중 평가 | regime-conditional routing 확장 |

→ **plan-020 통과 후보 (C05 per-regime F0)** 외에는 본격 ensemble/corrector path 진입 안 됨. 잔여 후보로 보존.

---

## 진행 roadmap (제안)

**Phase A — retrieval paradigm 진입 (★★★)**:
1. **B.1.1 coord-KNN candidates** → 27-pool 확장 (small-cost 진입)
2. **D.3 Trajectory-CLIP latent KNN** → InfoNCE encoder 학습 + latent retrieval
3. F3/F4 parity fix → **C.1' variance-aware 27-MDN**

**Phase B — paper-derived 보조 trick (★★)**:
4. 3-channel selector ensemble
5. Multi-parse input (raw/SG/EMA)
6. EB shrinkage 2D grid
7. SE(3) corrector module (corrector path 보조)
8. Path Signatures (In axis alt)
9. CPhy-ML jerk regularizer

**Phase B' — plan-021 LGBM follow-up (★★ Tier B)**:
4'. A4 LGBM+GRU ensemble (trivial 즉시)
5'. A1 Multi-window stat grid + B2 Pct-of-rolling-std
6'. C2 Multi-output joint (axis correlation)
7'. C1 Log-target / Huber (partial PASS 극복)
8'. A3 NN target-mean encoding (nyanp signature)

**Phase C — engineering / ensemble diversity (★)**:
10. 5×5×5 Voxel CE narrow-window
11. TTA (Z-rot, Y-flip)
12. PointNet 점군 paradigm
13. VQ-Trajectory ensemble member
14. Hard-sample fine-tune
15. Cascade 2-stage corrector
16. PTNet path × accel reparameterization
17. IRM / Domain-adversarial

**핵심 근거**: plan-001~022 까지 corrector path 의 candidate pool 자체가 Frenet family 27 variant + anchor sweep 에 갇혀 있음 (plan-011/020/022 모두 동일 framework). **retrieval-based pool 확장 (B.1.1, D.3)** 과 **mode-seeking 출력 (C.1')** 이 구조적 ceiling 돌파 1순위.
