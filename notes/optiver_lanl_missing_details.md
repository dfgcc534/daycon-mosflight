# Optiver / LANL 1st place 디테일 중 plan-021 LGBM 미반영 항목

**출처**: Optiver Realized Volatility 2021 (nyanp 1st) + LANL Earthquake 2019 (Philipp Singer 1st) + 두 솔루션 공통 패턴.

**기준 baseline (plan-021 LGBM v1.3 170D)**:
- L1 Frenet trajectory 99D + L2 F0 residual seq 21D + L4 F0 soft hit seq 14D + L5 macro stat 9D + L6 EWMA 27D
- 21 LGBM regressor 독립 학습 (7 anchor × 3 axis offset), 5-fold OOF
- 성능: hit@1cm 0.6488 (+0.0168 PASS), hit@1.5cm 0.8070 (+0.0037), partial PASS

---

## A. Optiver Realized Volatility 1st (nyanp) — 5 항목

### A1. Multi-window nested aggregation grid

> 비유: 카메라로 풍경 찍을 때 *광각/표준/망원/클로즈업* 4장 다 찍어서 모델에 보여줌. 같은 풍경이지만 줌 배율마다 다른 정보.

- **무엇**: trajectory 11점을 `[전체 11], [뒤 7], [뒤 5], [뒤 3]` 4 sub-window 로 나누고, 각 window 에서 `{mean, std, max, min, sum_sq, slope, last-first}` 7 통계 전부 뽑음. 단일 raw 채널 → ~28 derived feature.
- **plan-021 에 왜 없나**: L6 EWMA 가 비슷하지만 *exponential decay* 만. *직사각형 sub-window × 다양한 통계* grid 는 별개 정보 — 두 형태는 직교.
- **적용**: Frenet (p,v,a) 9-channel × 7-stat × 3-subwindow → ~135 D (중복 trim 후 ~60D).
- **효과**: tree split 깊이 절감. nyanp 가 이 grid 로만 base 8 → derived 200+ feature.

---

### A2. Cross-sample aggregation

> 비유: 시험 점수 80점이 잘 본 건지 못 본 건지는 *같은 반 평균* 비교해야 앎. 상대 위치가 중요.

- **무엇**: 같은 regime cluster 안의 *다른 trajectory* 의 feature 평균과의 차이 = "이 trajectory 가 regime 안에서 특이한가?"
- **plan-021 에 왜 없나**: trajectory 들이 *완전 독립* 으로 처리. 다른 sample 과 비교하는 feature 부재.
- **적용**: fold-wise OOF 로 regime-mean macro_stat 계산 → 각 sample 의 deviation (z-score).
- **효과**: regime 안의 *경계 sample* vs *전형 sample* 의 F0 신뢰도 차이 직접 측정.

---

### A3. Nearest-neighbor target-mean (latent ordering 복원)

> 비유: 모르는 문제 앞에서 *비슷한 유형* 의 평균 정답률 보고 baseline 잡음. 단 *같은 시험* 의 답은 못 보고 *과거 시험* 답만 (leakage 방지).

- **무엇**: trajectory 의 *motion signature* (평균 속도 + 평균 곡률 + 평균 z) 로 latent embedding → fold-out-of-fold 에서 *같은 cluster trajectory* 의 target offset 평균을 feature.
- **plan-021 에 왜 없나**: 완전 부재. nyanp 의 *signature trick* — private LB 결정적 차이.
- **적용**: k-NN (k=20-50) target-mean encoding, fold-internal leakage-safe.
- **효과**: "비슷한 motion pattern → 비슷한 F0 보정량" 이라는 cluster prior 를 LGBM 이 그냥 받음. F0 의 cluster-specific systematic bias 직접 보정.
- **주의**: leakage 방지 까다로움.

---

### A4. LGBM + GRU ensemble

> 비유: 의사 진료 시 *내과 + 외과* 둘 다 보면 blind spot 보완.

- **무엇**: LGBM 예측 + GRU 예측의 *단순 평균* 또는 *rank-blend*.
- **plan-021 에 왜 없나**: sub-exp A (LGBM 170D) vs B (GRU 134D) 를 *paradigm 비교* 목적 독립 평가만. 합치지 않음.
- **적용**: trivial — 이미 두 model 다 학습됨, 평균만 추가. weight grid {0.3, 0.5, 0.7} CV 비교.
- **효과**: LGBM = magnitude, GRU = direction 서로 보완. Optiver 1st 가 정확히 이 ensemble. 기대 +0.003~0.008.

---

### A6. WAP-style composite physical feature

> 비유: 키 175cm + 몸무게 70kg 둘 다 넣는 것보다 *BMI = 몸무게/키²* 가 더 강한 신호. 둘의 물리적 조합이 의미.

- **무엇**: raw 보다 *물리적 의미 있는 조합*. 우리 task analog:
  - `|v|² · κ` = 원심력 analog
  - `|j|/|a|` = jerk-to-accel ratio = 급격한 변화 정도
  - `½|v|²` = 운동 에너지 analog
  - `|v| · τ` = out-of-plane drift speed
- **plan-021 에 왜 없나**: Frenet decomposition 은 *축 분리* 만. cross-channel 물리적 곱 없음.
- **적용**: ~6-8 composite feature 추가.
- **효과**: tree 가 *변수 곱* 학습하려면 깊은 split 2회 — 미리 곱해서 주면 1회로 끝.

---

## B. LANL Earthquake 1st (Philipp Singer) — 5 항목

### B1. Massive feature count (~1000D)

> 비유: 색연필 12색보다 *120색 세트* 가 표현력 넓음. 안 쓰는 색 많아도 필요한 색이 있을 확률 ↑.

- **무엇**: Singer 가 hand-crafted ~1000+ feature. LGBM `feature_fraction=0.3-0.5` 로 각 트리가 30-50% 만 사용 → redundant 해도 학습 OK.
- **plan-021 에 왜 없나**: 170D — Singer recipe 의 1/5-1/6. LGBM 의 feature-견딤 강점 활용 부족.
- **적용**: 추천 S1~S4 + A1~A6 누적 시 260D, 추가로 +500~800D 까지 부담 없음 (학습 시간 ~2배).
- **효과**: feature 폭 자체로 *interaction 학습 깊이* 보완. 단, `feature_fraction`, `lambda_l2` 조정 필수.

---

### B2. Percentile-of-rolling-std (cross-scale nested distributional)

> 비유: 학교 점수 평균만 보는 게 아니라 *반별 점수 표준편차의 80 분위수* 도 봄 — 분포의 분포.

- **무엇**: 2단 nested aggregation. 안쪽 `std(x, window=w)` 시간열 → 바깥쪽 `pct_{20,50,80}` 적용.
- **plan-021 에 왜 없나**: L5 macro 의 단일 percentile, L6 EWMA 의 단일 mean — 둘 다 1단.
- **적용**: per axis × `pct_{20,50,80}(std(speed_mag, w={3,5,7}))` = 3 × 3 × 3 = 27D.
- **효과**: "변동성의 변동성" — saccade burst 강도의 일관성 vs 들쭉날쭉 신호. Singer LANL 에서 top-20 importance 반복.

---

### B3. STA/LTA ratio

> 비유: 평소 심박 60, 지금 90 → baseline 대비 1.5배 burst. 절대값보다 *비율* 이 의미.

- **무엇**: Short-term avg (최근 3 step) / Long-term avg (전체 11 step). 지진계 표준 trigger indicator.
- **plan-021 에 왜 없나**: L6 EWMA 가 α∈{0.1, 0.3, 0.5} 3개를 *individual* 로 넣지만 *ratio* 안 만듦. EWMA(α=0.5)/EWMA(α=0.1) = 정확히 STA/LTA.
- **적용**: 9 channel × 3 ratio = 27D 추가. 비용 0 (이미 EWMA 27D 있음, division 만).
- **효과**: 절대 magnitude 보다 *baseline 대비 burst 비율* 이 saccade 같은 transient event 검출에 직접적.

---

### B4. Peak counts

> 비유: 일주일간 *화낸 횟수* 가 성격 지표 — 평균값보다 count 가 의미.

- **무엇**: `|jerk|` local maxima 개수, `velocity_x` sign-flip 개수, turning_angle 큰 변화 횟수.
- **plan-021 에 왜 없나**: L5 macro 가 *percentile* 만 — count statistic 부재. percentile = "큰 값이 있다" / count = "몇 번 있었나" 는 직교.
- **적용**: ~6-8 count feature. S3 saccade 추천과 통합.
- **효과**: percentile = 분포 위치 / count = 분포 횟수 — 두 정보 직교. Singer LANL 에서 peak_count_50 단일 top-10 importance.

---

### B7. Hand-crafted ≫ 자동 FE 검증 박제

> 비유: 수제 만두 vs 공장 만두 *직접 비교* 안 하면 의문 남음.

- **무엇**: Singer 가 tsfresh 자동 FE 시도 → hand-crafted 가 strict 하게 우세 박제. paradigm 확신용.
- **plan-021 에 왜 없나**: tsfresh / catch22 자동 FE 시도 자체 없음. hand-crafted 만 채택, 비교 없음.
- **적용**: catch22 viable subset (~8D) 한번 돌려보고 hand-crafted 와 CV 비교 박제.
- **효과**: paradigm 확신 + caveat 박제. 자동 FE 가 실제로 약함을 직접 보이면 hand-crafted 선택 정당성 부각.

---

## C. 두 솔루션 공통 + plan-021 미반영 — 2 항목

### C1. Log-target / robust loss

> 비유: 키 데이터에서 NBA 선수 포함 시 *꼬리* 가 김. MSE 학습 시 꼬리 sample 이 dominate. log(키) 변환 또는 Huber loss 로 균형 학습.

- **무엇**: target offset = 3D vector, magnitude long-tail (saccade 직후 큰 offset).
  - target → (방향 vector, log(1+|magnitude|)) decomposition
  - 또는 loss → Huber loss (큰 error robust)
- **plan-021 에 왜 없나**: target 그대로 MSE — saccade sample (10%) dominate, cruise sample (90%) 작은 보정량 학습 희석.
- **적용**: lightgbm `objective='huber'` 또는 `objective='quantile'` 시도.
- **효과**: cruise + saccade 균형 학습. 1cm hit + 1.5cm hit 둘 다 PASS 하려면 필수. partial PASS (1cm only) 극복 후보.

---

### C2. Multi-output joint (axis correlation)

> 비유: 21명 의사가 *독립* 진단 → x,y,z 추천이 서로 모순. *왼쪽-위로 함께 가야 하는데* 한 명은 "왼쪽" 다른 명은 "위" 라고만 함. 21명이 *회의* 해야 합의된 진단.

- **무엇**: plan-021 의 21 LGBM regressor (7 anchor × 3 axis offset) 가 *완전 독립* 학습. **caveat 5 박제**: "axis 간 correlation 학습 X".
- **plan-021 에 왜 없나**: 본인이 *이미 박제한 미해결*. 인지는 했으나 해결 안 함.
- **적용**:
  - multi-output GBDT (lightgbm multi-output mode, 또는 catboost MultiRMSE)
  - axis-pair joint feature: `x·y, y·z, x·z, |xy|, |yz|`
- **효과**: F0 잔차의 *방향성* 은 sample 마다 일관 (saccade 후엔 항상 진행방향+반대편위) — 21 regressor 가 따로 학습하면 이 방향성 못 잡음.
- **비용**: 작 (multi-output mode 또는 axis-product feature 5-6개).

---

## 종합 — 우선순위 ranking

| Rank | 항목 | 기대 Δhit@1cm | 비용 | 비고 |
|---|---|---|---|---|
| 1 | **A4 LGBM+GRU ensemble** | +0.003~0.008 | trivial | 즉시 가능, paradigm 비교 후 follow-up |
| 2 | **A1 Multi-window stat grid** | +0.005~0.010 | 중간 (+60D) | feature scaling 의 핵심 |
| 3 | **C2 Multi-output joint** | +0.003~0.006 | 작 | caveat 5 의 self-acknowledged gap 해결 |
| 4 | **B1 Massive 170D → 600D+** | +0.002~0.005 | 작 | S1~A6 누적 시 자동 달성 |
| 5 | **A3 NN target-mean** | +0.002~0.006 | 큼 (leakage-safe) | nyanp signature trick |
| 6 | **B2 Pct-of-rolling-std** | +0.002~0.004 | 작 (+27D) | A1 grid 와 함께 |
| 7 | **C1 Log-target / Huber** | +0.001~0.003 | 작 | partial PASS 극복 후보 |
| 8 | **A2 Cross-sample agg** | +0.001~0.003 | 중간 (OOF) | regime cluster deviation |
| 9 | **B3 STA/LTA ratio** | +0.001~0.003 | trivial | EWMA division 만 |
| 10 | **B4 Peak counts** | +0.001~0.002 | trivial | S3 saccade 와 통합 |
| 11 | **A6 WAP composite** | +0.001~0.003 | trivial | 6-8 feature |
| 12 | **B7 Hand vs auto 박제** | 0 (methodology) | 작 | catch22 비교 |

**Stage 1 즉시 적용 (cost trivial)**: A4 + B3 + A6 + B4 → ~+0.005~0.013 기대
**Stage 2 본격 추가**: A1 + B2 + C2 → ~+0.010~0.020 기대
**Stage 3 paradigm shift**: A3 + A5(per-regime bagging) + C1 → ~+0.005~0.012 기대

→ plan-021 LGBM 0.6488 → 잠재 0.66~0.68 권 진입 (plan-004 0.6806 침투 가능 영역).

---

## Reference

- Optiver Realized Volatility 1st (nyanp): https://www.kaggle.com/competitions/optiver-realized-volatility-prediction/writeups/nyanp-1st-place-solution-nearest-neighbors
- LANL Earthquake 1st (Philipp Singer): https://medium.com/@ph_singer/1st-place-in-kaggle-lanl-earthquake-prediction-competition-15a1137c2457
- nyanp discussion: https://www.kaggle.com/competitions/optiver-realized-volatility-prediction/discussion/274970
- Optiver Trading at Close 1st (hyd, 2024 후속): https://www.kaggle.com/competitions/optiver-trading-at-the-close/writeups/hyd-1st-place-solution
- plan-021 spec: `plans/plan-021-frenet-corrector-input-augment.md`
- plan-021 results: `plans/plan-021-frenet-corrector-input-augment.results.md`
