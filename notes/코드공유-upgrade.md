# plan-005 진단 결과 + 핵심 직관 (upgrade motivation)

이하 두 upgrade idea (연속 heatmap, 학습 후보) 의 *motivation* 은 plan-005 의 정량 진단. 핵심 finding 과 그로부터 도출된 5가지 직관 박제.

## 핵심 수치 (`analysis/plan-005/results.md`)

| Metric | 값 | 시사점 |
|---|---|---|
| **Raw oracle** | **0.7188** | 27 후보의 ceiling — selector 완벽해도 못 넘음 |
| **Post-correction oracle** | **0.7111** | ⚠️ corrector 가 oracle 을 *0.77pp 떨어뜨림* (14/18 regime negative) |
| **Selector hit (soft)** | **0.6599** | oracle 까지 5.1pp gap |
| **Selector top-1 ranking** | **12.6%** | 진짜 best 후보를 1순위로 picking 12.6%만 — soft 평균이 cover |
| **GRU marginal contribution** | **+0.0052** | noise floor (±0.005) 근방 — 거의 기여 없음 |
| **Regime marginal contribution** | **+0.0029** | noise floor 미만 — 거의 기여 없음 |
| **GRU intervention helped/hurt** | **0.565 / 0.435** | 거의 50:50 (무작위적 개입) |
| **PB vs B001 baseline** | 0.6599 vs 0.5787 | PB win 965 / loss 153 (6.3:1) — framework 자체는 정당 |

## STAGE별 finding

| STAGE | 핵심 finding |
|---|---|
| **1 Oracle** | corrector 가 oracle 을 *낮춤* — family-aware loss 가 best 후보를 옮기는 부작용 |
| **2 Selector** | top-K {1: 12.6%, 3: 21.8%, 5: 28.2%} — ranking 능력 매우 약함, soft 평균이 noise cover |
| **3 Corrector** | cap saturation 3.6% (cap 충분). binormal 변위 = parallel 의 1/7 (z 축 보정 거의 0) |
| **4 Component** | GRU/regime 기여 noise floor 근방, helped/hurt ≈ 50:50 (무작위 개입) |
| **5 Failure** | worst-100 의 49% 가 regime 10~14 (high-speed × high-curvature) 집중 |

## 5가지 직관

### 1. Selector 는 picker 가 아니라 distributor

```
Top-1 ranking 정확도: 12.6%   ← "진짜 best 를 1순위로 picking"
Soft hit rate:        66.0%   ← "최종 좌표가 정답 1cm 안"
```

비유 — 27지선다: 학생이 정답을 1번으로 마킹하는 비율 12.6%, 그런데 *확신도 분포의 가중평균 좌표* 가 정답 영역에 떨어지는 비율 66%. **selector 는 "정답을 안다" 가 아니라 "정답이 어느 *영역* 인지 좁힐 줄 안다".** 27 개 후보가 비슷한 좌표 부근에 있어 *평균이 정답 영역에 들어감*.

### 2. GRU 와 regime 둘 다 *장식*

- GRU 기여 +0.0052, regime 기여 +0.0029 — 둘 다 노이즈 floor (±0.005) 근방
- 개입했을 때 helped/hurt ≈ 50:50 → 거의 무작위적 개입

→ **PB framework 의 "지능" 90% 는 *손으로 짠 27 개 물리 식* 에서 나오고, 신경망/regime 은 장식에 가깝다.**

### 3. Corrector 의 역설 — 평균을 최적화하면 best 가 망가진다

기대: corrector 가 27 후보를 더 정답에 가깝게 → oracle ↑
실제: corrector 적용 후 oracle 0.7188 → 0.7111 (0.77pp ↓)

비유 — 사격 훈련: 27 명에게 *평균 탄착점이 정중앙* 되도록 보정값 적용 → 평균은 정중앙, 그러나 *이미 정중앙 맞히던 1명* 은 보정값 때문에 옆으로 밀려남. **Soft prediction 최적화를 위한 *부수 효과*.**

### 4. 진짜 ceiling 은 후보 다양성

```
1.0000  절대 ceiling
0.7188  Raw oracle (27 후보의 한계)  ★ 진짜 ceiling
0.7111  Post-corr oracle
0.6599  Selector + corrector (실제 PB)
0.5787  B001 linear baseline
```

selector/corrector 를 완벽하게 만들어도 0.7188 위로 못 감. 그 위로 가려면 **후보를 더 많이/잘 만들어야** 함. worst-100 의 49% 가 regime 10~14 (high-speed) 집중 → *고속 비행 모기의 정답 영역을 27 후보가 못 덮음*.

### 5. *Simple is better* — 이 framework 은 과설계

- GRU 기여 ≈ 0
- Regime 기여 ≈ 0
- Corrector 가 oracle 까지 *떨어뜨림*
- 그래도 B001 보다 +8pp 잘함

**결론**: PB 가 B001 을 이기는 이유는 *27 개 후보의 다양성* + *soft averaging* 두 가지뿐. 나머지 정교함은 *비용만 들이고 효과 미미*. **Galton 황소 무게 추측 (군중 평균이 전문가보다 정확) 의 trajectory prediction 버전.**

## 이 진단이 가리키는 upgrade 방향

| 방향 | 동기 metric | 이 문서에서의 대응 |
|---|---|---|
| **후보 다양성 ↑** | Raw oracle 0.7188 ceiling + worst-100 의 49% high-speed regime 집중 | **Idea 2 (데이터 기반 N 후보 학습)** — 사람 직관 한계 돌파, drone 확장 시 재학습만 |
| **Regime prior 단순화** | regime 기여 +0.0029 (noise floor 미만), 18 이산 격자가 marginal 정보만 제공 | **Idea 1 (연속 heatmap)** — 격자 → 부드러운 kernel, feature 수 늘면 자연 대응 |
| **GRU 제거 / arch 교체** | GRU 기여 +0.0052 (noise floor), top-1 12.6% | plan-006 후보 1 (selector 단순화 / TCN·Transformer 교체) — 본 문서 out-of-scope |
| **Corrector loss 재설계** | oracle 0.77pp 손실 (14/18 regime negative) | plan-006 후보 2 — 본 문서 out-of-scope |

→ 본 문서의 두 idea (연속 heatmap, 학습 후보) 는 **후보 다양성 강화 + regime 단순화** 두 anchor 를 *파이프라인의 다른 단계* 에서 동시 공격. 4가지 ablation 조합으로 분리 측정 가능 (마지막 섹션 참조).

---

# PB_0.6822 Upgrade Idea: 이산 regime 표 → 연속 heatmap

## 현재 구조 (PB_0.6822 코드공유.ipynb)

```
1. (speed, curvature, speed_slope) feature 추출 (최근 4~6 프레임)
2. 분위수 컷으로 18 regime 분류 (3 × 3 × 2)
3. [18 regime × 27 후보] 표 계산 — empirical Bayes shrinkage 포함
4. 각 sample → 자기 regime의 행을 lookup → 27차원 bias
5. 최종 점수 = gru_logits + 0.65 × physics_bias + 0.45 × regime_bias[regime]
```

**단점**:
- 분위수 컷 경계에서 *discrete jump* (sample 두 개가 비슷한 feature인데 다른 regime이면 bias 점프)
- Sample 분포가 불균등할 때 작은 regime이 noisy → shrinkage로 보완하지만 *수동 hyperparameter* (`shrink=18`)
- Quantile cutoff, regime count, shrink coefficient 등 *여러 hyperparameter*
- Feature 수가 늘면 격자 셀 폭발 (현재 3×3×2=18, 4축이면 3×3×2×3=54, 5축이면 162…)

---

## 제안: 연속 heatmap (kernel-weighted bias)

이산 표 대신 **(speed, curvature, slope) 연속 공간 위의 부드러운 함수**로 bias를 표현.

### 핵심 수식

```
bias(x, c) = [Σ_i K(x - x_i) · hit_metric_i(c) + λ · global_metric(c)]
           / [Σ_i K(x - x_i) + λ]
```

- `x`: query sample의 feature (speed, curve, slope)
- `x_i`: train sample들의 feature
- `K`: Gaussian kernel — `K(d) = exp(-(d/bandwidth)²)`
- `λ`: pseudo-count (sparse 영역에서 global prior로 후퇴)
- `hit_metric`: `log(hit + 1e-4) - 18 × mean_distance`

마지막에 centering 적용해 framework 일관성 유지:
```
bias = bias - bias.mean(axis=candidates)
```

### 왜 sample 개수가 자동 처리되는가

| Feature 공간 영역 | Σ K (kernel sum) | 효과 |
|---|---|---|
| Sample 밀집 | Σ K >> λ | local 통계 dominant |
| Sample 희소 | Σ K ~ λ | local과 global blend |
| 데이터 없음 | Σ K ≈ 0 | global prior dominant |

→ **현재 코드의 empirical Bayes shrinkage가 kernel weighting 안에 자연 흡수**. `α = n/(n+18)` 식 별도로 안 필요.

---

## 장점

| 측면 | 이산 표 | 연속 heatmap |
|---|---|---|
| Quantization artifact | bin 경계 jump | 없음 (연속 부드러움) |
| Sample density 처리 | 명시적 shrinkage | **자동** (kernel weighting) |
| Hyperparameter | quantile cuts + shrink | **bandwidth 하나만** |
| 차원 확장성 | 격자 셀 폭발 | 자연 (feature 수 ↑ 시 부드럽게 대응) |
| Paper figure | 18-row 이산 grid | **3D 연속 surface** (paper 머니샷급) |
| 코드 복잡도 | regime fit/assign/shrink 3 함수 | kernel 1 함수 |

## 단점

| 측면 | 영향 |
|---|---|
| 추론 비용 | O(1) lookup → O(N_train) kernel sum |
| Bandwidth 선택 | 새 hyperparameter (CV로 1~2값 시도) |
| 차원의 저주 | feature ≥5D 가면 KDE 신중 (현재 3D는 안전) |
| 안정성 | bandwidth 너무 작으면 high variance |

---

## 구현 스케치

```python
def kernel_regime_bias(
    train_features,    # [N, 3]  in (speed, curve, slope), normalized
    train_cands,       # [N, 27, 3]
    train_y,           # [N, 3]
    query_features,    # [M, 3]
    bandwidth: float = 0.3,
    lam: float = 20.0,
):
    err = np.linalg.norm(train_cands - train_y[:, None, :], axis=2)  # [N, 27]
    hit_metric = np.log((err <= R_HIT).astype(float) + 1e-4) - 18.0 * err

    global_metric = hit_metric.mean(axis=0)  # [27]

    d = np.linalg.norm(query_features[:, None, :] - train_features[None, :, :], axis=2)  # [M, N]
    w = np.exp(-(d / bandwidth) ** 2)

    numer = w @ hit_metric + lam * global_metric[None, :]
    denom = w.sum(axis=1, keepdims=True) + lam

    bias = numer / denom
    bias = bias - bias.mean(axis=1, keepdims=True)
    return bias.astype(np.float32)
```

→ 기존 `fit_regime_bins` + `assign_regimes` + `candidate_regime_bias` 세 함수가 **이 한 함수로 통합**.

---

## 추론 비용 완화 (선택)

`O(N_train)` 부담을 줄이려면:

1. **Train sample을 대표점으로 압축**: k-means centroid K=200 정도로 축약 → O(K) per query
2. **KD-tree로 nearest neighbor만 kernel 계산**: 멀리 있는 sample은 자동 무시
3. **Precompute grid + 추론 시 보간**: query는 grid lookup + bilinear, train 시점 한 번만 grid 계산

가장 단순한 시작점은 *그대로* 돌려보고 속도 문제 있을 때 1~3번 적용.

---

## 시각화 (paper figure 후보)

3D feature 공간이라 한 장에 못 그리니 **2D marginal heatmap 27장** (후보별):

```
                후보 c의 P(hit) heatmap
       curvature
          ▲
       높음│ ░░░▒▒▒▓▓
          │ ░░▒▒▒▓▓▓     <- (speed, curve) 평면
          │ ░▒▒▒▒▓▓▓        slope는 marginal/평균
          │ ▒▒▒▒▓▓░░
       낮음│ ▒▒▓▓▓░░░
          └───────────────► speed
            느림        빠름
```

또는 **family별 통합** (base, acc, frenet, turn, jerk, latency 6장)으로 압축.

권장 3-set:
1. **Raw hit rate heatmap** (centering 전, 절대 비교)
2. **Centered bias heatmap** (framework이 실제 쓰는 값)
3. **Sample density heatmap** (Σ K of train samples)

---

## 기대 효과

| 항목 | 추정 |
|---|---|
| Hit rate (모기 단일) | +0.0 ~ +0.5 pp (marginal) |
| 코드 단순화 | regime fit/assign/shrinkage 제거 |
| Hyperparameter 수 | 줄어듦 (quantile + shrink → bandwidth) |
| **Paper figure** | **매우 강함** — 연속 surface가 핵심 contribution figure |
| **Drone 확장 시** | **결정적** — feature 수 늘면 격자보다 훨씬 깔끔 |

---

## 실행 체크리스트

- [ ] `kernel_regime_bias` 구현
- [ ] 기존 `regime_table` 호출부를 새 함수로 교체 (학습/추론 양쪽)
- [ ] Bandwidth ablation (0.2, 0.3, 0.4)
- [ ] λ ablation (10, 20, 50)
- [ ] Hit rate 1fold 비교: 기존 vs 제안
- [ ] 시각화 3-set 생성 (raw / centered / density)
- [ ] 추론 속도 측정 → 필요 시 KD-tree 또는 centroid 압축
- [ ] Drone domain 이식 시점에 재검증 (feature 수 늘었을 때 진가 발휘)

---

## 한 줄 정리

**이산 18-regime 표 + 명시적 shrinkage → 연속 (speed, curve, slope) 공간의 kernel-weighted heatmap.** Sample density가 *kernel weighting 안에 자연 흡수*되어 shrinkage 코드 제거, hyperparameter 하나(bandwidth)로 축소, *연속 surface heatmap*이라는 paper-grade visualization 공짜로 따라옴. Hit rate 이득은 모기 단일 도메인에선 marginal하지만 **drone 확장 시 feature 수 증가에 자연 대응**한다는 점에서 *구조적 가치*가 큼.

---

# PB_0.6822 Upgrade Idea 2: 사람이 손으로 만든 27 후보 → 데이터로 학습한 N 후보

## 현재 구조

```python
CANDIDATES = [
    CandidateSpec("p0_2d1",        d1=2.00, par=0.00, perp=0.00, d2=0.0, jerk=0.0, time_scale=1.00),
    CandidateSpec("acc_2d1_050",   d1=2.00, par=0.50, perp=0.50, ...),
    CandidateSpec("frenet_best",   d1=1.98, par=0.96, perp=-0.08, ...),
    ...                                                              # 27개
]                                                                    ↑
                                                              사람이 직관으로 결정
```

→ 6차원 계수 `(d1, d2, par, perp, jerk, time_scale)` 의 *27개 조합을 사람이 손으로 선택*. Candidate spec 변경은 코드 수정 + 재실행이 필요한 *manual loop*.

**단점**:
- 6차원 계수 공간을 인간 직관으로 탐색 불가 (효율적 조합을 빠뜨릴 가능성)
- 27이라는 숫자가 *부드러운 hyperparameter* (늘리기 어렵고 줄이기 모호)
- 새 도메인(drone)으로 옮기면 *처음부터 손으로 재설계* 필요
- Train 데이터 분포가 바뀌면 (예: 추가 데이터 들어오면) 손 후보가 *서서히 stale*

---

## 제안: 데이터 기반 candidate 학습

train 데이터로부터 **N개의 (d1, d2, par, perp, jerk, time_scale) 조합을 자동 학습**. 학습 후 N개 LEARNED_CANDIDATES가 *고정 상수*로 박혀, **추론 시에는 현재 27 손 후보와 완전히 동일하게 작동**.

```python
# 학습 단계 (train time만)
LEARNED_CANDIDATES = optimize_coefficients_from_data(train_data, n=N)

# 추론 단계 — 현재 코드와 같은 형태
candidates = make_candidates(x, end_idx, formulas=LEARNED_CANDIDATES)
```

→ **drop-in replacement**. `make_candidates` 호출부 외에는 코드 변경 0.

### Framework 철학과 정합

| 측면 | 영향 |
|---|---|
| 학습 시 | DL 사용 (계수 선택만) |
| **추론 시** | **고정 계수, DL 안 씀** |
| 후보 출력 | 결정적, 물리적 해석 가능 |
| 노이즈 외우기 risk | 매우 낮음 (학습 변수가 6×N개 뿐) |

→ "DL은 추론에 끼어들지 않는다" 라는 원 framework 철학 보존.

---

## 4가지 학습 방법

### A. K-means in coefficient space (가장 단순)

```python
# 1. 각 train sample에 대해 *오차 최소인 최적 계수* 분석적으로 풀어냄
optimal_coefs = solve_per_sample_best_coefs(train_x, train_y)   # [N_train, 6]

# 2. 6차원 공간에서 K-means
from sklearn.cluster import KMeans
km = KMeans(n_clusters=N).fit(optimal_coefs)

# 3. Centroid가 학습된 후보
LEARNED_CANDIDATES = [CandidateSpec(f"learned_{i}", *c) for i, c in enumerate(km.cluster_centers_)]
```

**장점**: 미분 불필요, 간단, 빠름
**단점**: per-sample 최적 계수 풀이가 closed-form이 아닐 수 있음

### B. Differentiable coefficient optimization (가장 강력)

```python
class CandidateLearner(nn.Module):
    def __init__(self, n: int):
        super().__init__()
        self.coefs = nn.Parameter(torch.randn(n, 6) * 0.3)   # 학습됨

    def make_candidates(self, x, end_idx):
        # 현재 make_candidates와 동일한 수식, 단 coefs가 학습 가능
        ...

# 학습 loop
for batch in loader:
    cands = learner.make_candidates(batch.x, batch.end_idx)   # [B, N, 3]
    err = (cands - batch.y[:, None, :]).norm(dim=2)            # [B, N]
    # Soft min을 통해 미분 가능하게
    soft_min_err = -torch.logsumexp(-err / tau, dim=1).mean()
    soft_min_err.backward()
    opt.step()
```

**장점**: 직접 hit rate 최적화, end-to-end
**단점**: tau hyperparameter, mode collapse 위험

### C. Anchor learning (MultiPath/CoverNet 스타일)

```python
# 1. Train trajectory의 +80ms 좌표를 trajectory-relative 좌표계로 정규화
relative_targets = normalize_to_local_frame(train_y, train_x)

# 2. K-means → N anchor 좌표
anchors = KMeans(N).fit(relative_targets).cluster_centers_

# 3. Anchor → 6-coef 역변환 (선형 회귀)
LEARNED_CANDIDATES = inverse_solve_coefs_from_anchors(anchors)
```

**장점**: 자율주행에서 입증된 표준, 안정적
**단점**: 좌표 → 계수 역변환이 underdetermined일 수 있음

### D. Discrete combinatorial optimization

```python
# 1. 6-coef를 grid에 두기 (예: 각 축 10단계)
candidate_pool = generate_grid(d1=10, par=10, perp=10, ...)   # 10^6 후보

# 2. Greedy로 N개 선택 — oracle hit rate 최대화
selected = []
for _ in range(N):
    best = max(candidate_pool, key=lambda c: oracle_hit_with(selected + [c], train_y))
    selected.append(best)
```

**장점**: 명시적 oracle 최적화, 해석 가능
**단점**: 계산 비싸다, grid 해상도 trade-off

---

## 장점

| 측면 | 손 후보 (27) | 학습 후보 (N) |
|---|---|---|
| 6차원 공간 탐색 | 인간 직관 한계 | 데이터 기반 자동 |
| Oracle hit rate | baseline | 같은 N에서 ↑ 가능성 |
| N 조정 | 어려움 (코드 수정) | hyperparameter grid search |
| Domain transfer | 손으로 재설계 | 재학습만 |
| 새 데이터 들어오면 | 손 후보 stale | 재학습 |
| Paper contribution | 평범 | **"data-driven physics anchor learning"** |

## 단점

| 측면 | 영향 |
|---|---|
| **Mode collapse** | N개가 비슷한 곳에 모일 위험. Diversity regularization 필요 |
| **Train/val 누수** | 계수 학습이 train만 보도록 CV discipline 엄수 |
| **N 선택** | 새 hyperparameter (grid search) |
| **해석 가능성** | 학습된 계수는 "이름이 없음". 후처리로 family 클러스터링 후 라벨 부여 |
| **Oracle gap이 너무 작아질 위험** | 학습 후보가 너무 oracle에 가까우면 selector 학습이 단순화되어 generalization ↓. 적당한 diversity 유지 필요 |

---

## 구현 스케치 (Method B - Differentiable)

```python
import torch
import torch.nn as nn
import math

class LearnableCandidates(nn.Module):
    """N개의 (d1, d2, par, perp, jerk, time_scale)를 학습 가능 파라미터로 두는 모듈."""

    def __init__(self, n: int):
        super().__init__()
        # 초기화: 현재 27 손 후보 근처에서 약간의 jitter
        init = torch.tensor([
            [spec.d1, spec.d2, spec.par, spec.perp, spec.jerk, spec.time_scale]
            for spec in base.CANDIDATES[:n]
        ], dtype=torch.float32)
        if init.shape[0] < n:
            extra = init[:1].repeat(n - init.shape[0], 1) + torch.randn(n - init.shape[0], 6) * 0.1
            init = torch.cat([init, extra], dim=0)
        self.coefs = nn.Parameter(init)
        self.n = n

    def forward(self, p0, d1, d2, acc_par, acc_perp, jerk, horizon: int = 2):
        # current make_candidates와 동일 수식, 단 self.coefs가 학습됨
        v_scale = horizon / 2.0
        acc_scale = (horizon / 2.0) ** 2
        cands = []
        for i in range(self.n):
            c = self.coefs[i]   # [6]: d1, d2, par, perp, jerk, time_scale
            ts = c[5]
            cands.append(
                p0
                + c[0] * v_scale * ts * d1
                + c[1] * v_scale * ts * d2
                + c[2] * acc_scale * (ts ** 2) * acc_par
                + c[3] * acc_scale * (ts ** 2) * acc_perp
                + c[4] * acc_scale * (ts ** 2) * jerk
            )
        return torch.stack(cands, dim=1)   # [B, N, 3]


def train_candidate_learner(model, train_loader, opt, tau: float = 0.005,
                            diversity_weight: float = 0.1, epochs: int = 50):
    for epoch in range(epochs):
        for batch in train_loader:
            cands = model(batch.p0, batch.d1, batch.d2, batch.acc_par, batch.acc_perp, batch.jerk)
            err = (cands - batch.y.unsqueeze(1)).norm(dim=2)   # [B, N]

            # (1) Soft-min loss: 가장 가까운 후보의 오차 최소화
            soft_min = -torch.logsumexp(-err / tau, dim=1).mean()

            # (2) Diversity regularization: 계수 간 거리 최대화
            coef_dist = torch.cdist(model.coefs, model.coefs)
            diversity_loss = -coef_dist[~torch.eye(model.n, dtype=bool)].mean()

            loss = soft_min + diversity_weight * diversity_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
```

학습 끝나면 `model.coefs.detach().cpu().numpy()` 를 `CandidateSpec` 리스트로 변환해 박음.

---

## Idea 1과의 관계 — 직교성

두 upgrade가 *파이프라인의 서로 다른 단계*를 건드림:

| 질문 | 단계 | 현재 | 제안 |
|---|---|---|---|
| 어떤 후보를 만들 것인가 | Candidate generation | 손 27개 | **학습 N개** (Idea 2) |
| 각 후보를 regime별로 어떻게 평가할 것인가 | Scoring (regime 부분) | 18×27 이산 표 | **연속 heatmap** (Idea 1) |

→ **독립 적용 가능 → 4가지 ablation 조합**:

| 후보 | Regime | 명칭 |
|---|---|---|
| 손 27 | 이산 18 | baseline (현재) |
| 손 27 | 연속 heatmap | Idea 1 단독 |
| 학습 N | 이산 18 | Idea 2 단독 |
| **학습 N** | **연속 heatmap** | **둘 다 (최대 변경)** |

각각 단독 ablation으로 **두 contribution의 기여를 분리 측정** 가능 → paper structure에 깔끔.

---

## 잠재적 함정

| 함정 | 대응 |
|---|---|
| Mode collapse | Diversity regularization (`diversity_loss`), 또는 furthest-first init |
| Oracle gap 소실 | N을 적당히 (모기는 20~40 추천), oracle hit이 100% 가까이 가지 않도록 |
| Train 과적합 | CV로 N과 hyperparameter 결정, val hit rate 모니터링 |
| 학습 후보의 *물리적 해석 손실* | 학습 후 6-coef 공간에서 K-means → family 라벨 부여 (자동 family 분류) |
| Latency family 같은 *특수 구조* 누락 | 초기화에 손 후보 일부를 포함 + frozen subset 옵션 |

---

## 기대 효과

| 항목 | 추정 |
|---|---|
| Oracle hit rate | 같은 N에서 손 후보 대비 **+1~3 pp** 가능 |
| Selector hit rate | **+0.5~1.5 pp** (oracle 상승의 일부 흡수) |
| 코드 단순화 | 손 후보 수정 loop 제거 → 학습 한 번 |
| Hyperparameter | N, tau, diversity_weight 추가 (3개) |
| **Domain transfer 비용** | **결정적 ↓** — drone으로 옮길 때 *재학습만* 필요 |
| **Paper contribution** | "Data-driven physics candidate discovery" — MultiPath/CoverNet의 *coordinate anchor*를 *physics formula anchor*로 변환한 새 angle |

---

## 실행 체크리스트

- [ ] Method A (K-means) 먼저 — 가장 적은 구현 비용으로 *학습 후보가 손 후보보다 나은지* 빠른 검증
- [ ] Method A 결과가 promising 하면 Method B (differentiable) 구현
- [ ] N ablation (10, 20, 27, 40, 60)
- [ ] Diversity weight ablation (0.0, 0.05, 0.1, 0.3)
- [ ] Tau ablation (0.003, 0.005, 0.01)
- [ ] Initialization 비교: random vs 손 후보 seed
- [ ] Frozen subset 실험: latency family를 frozen으로 두고 나머지만 학습
- [ ] 학습된 계수의 6-coef 시각화 (PCA 2D plot, family 클러스터링)
- [ ] **Idea 1과 조합 ablation 4가지** (orthogonal contribution 입증)
- [ ] Drone domain 이식 시 *재학습 비용 측정*

---

## 한 줄 정리

**사람이 6차원 계수 공간을 직관으로 탐색해 27개를 손으로 고른 것을, *train 데이터로 N개를 자동 학습*으로 대체.** 학습 후 계수는 *고정 상수*로 박혀 추론 시 DL 추가 비용 0, framework 철학(DL이 추론에 안 끼어듦)도 보존. Idea 1(연속 heatmap)과 *파이프라인의 다른 단계*를 건드리므로 **독립적/직교적으로 적용 가능**, 4가지 ablation 조합으로 각 contribution을 분리 입증할 수 있어 paper structure에도 깔끔. 모기 단일에선 hit rate 이득이 중간 정도지만 **domain transfer (drone) 시 *재학습만으로 후보 재설계*가 끝난다는 점에서 구조적 가치가 큼**.

---

# 두 Idea의 통합 시각화

```
Trajectory (11 또는 5 프레임 + sliding window 옵션)
   │
   ▼
[STAGE 1: 후보 생성]                   ← Idea 2 (학습된 N 물리식)
   ┌──────────────────────────┐
   │ LEARNED_CANDIDATES[N]    │
   │  (train 시 학습됨)        │
   │  추론 시 고정             │
   └──────────────────────────┘
   → N개 후보 좌표 [B, N, 3]
   │
   ▼
[STAGE 2: 점수 매김]
   ├─ physics_bias × 0.65  (전체 평균, 고정)
   ├─ ★ 연속 heatmap regime bias × 0.45  ← Idea 1 (kernel-weighted)
   └─ gru_logits (attn_gru per-sample)
   → 최종 score [B, N]
   │
   ▼
[STAGE 3: Tiny correction]              (변경 없음)
   Boundary MLP delta 적용
   │
   ▼
[STAGE 4: 최종 선택]                    (변경 없음)
   margin gate → argmax 또는 soft
```

→ **두 변경 모두 *기존 단계의 한 함수 교체*** 로 끝나고, 다른 단계는 그대로 사용. 점진적 도입 + 분리 ablation에 매우 친화적인 구조.

---

# 추가 실험 후보 (plan-007 후속, 독립 검토)

본 섹션 = 위 두 idea (연속 heatmap, 학습 후보) 와 *별개* 로, plan-006 4-way LB 회수 (full=0.6806, A=0.6796, B=0.6704, E=0.6692) 이후 식별된 lift 후보를 LB 단위 ROI 순으로 박제. **C001~C007 각각이 독립적으로 시도 가능** (의존/직교 관계는 마지막 표 참조). plan-007 이미 채택된 A1 (후보 다양화) = C001.

## C001~C007 실험 후보표

| 실험 ID | 이름 | 한 줄 설명 | 변경 위치 | baseline | 예상 ΔLB | 작업량 |
|---|---|---|---|---|---|---|
| **C001** | candidate-diversify-35 | 27 후보 → 35+ 후보 확장 (radial / ballistic / ema family 추가). worst regime 16/17/10 (high-speed × high-curvature, hit < 0.42) 커버. | `src/pb_0_6822/selector.py` `CANDIDATES` 에 8개 추가 + corrector retrain | Variant A (0.6796) | **+0.02 ~ +0.04** | 0.5d |
| **C002** | corrector-residual-redesign | TinyCorrectionNet → deeper residual block / multi-task loss. plan-005 `failure_b001` 영역 회복 + oracle-drop (0.7188 → 0.7111) 해소. | `src/pb_0_6822/boundary.py` TinyCorrectionNet 교체 + 5-fold 재학습 | full (0.6806) | **+0.01 ~ +0.025** | 1d |
| **C003** | lightgbm-rank-stacking | GRU `final_ctx` + `event_ctx` + candidate physics feature → LightGBM LambdaRank listwise head. NN selector 의 stacking 또는 대체. n=10k 에 sample-efficient. | 신규 `src/ranking/lgbm_rank.py` + plan-005 `corrected_*.npz` 재사용 | full (0.6806) | **+0.005 ~ +0.02** | 0.5d |
| **C004** | set-transformer-cands | 27 후보를 set 으로 보고 self-attention → 후보 간 상호작용 학습 (현재 head 는 후보별 독립 logit). "다른 26개 대비 outlier" 판별 가능. | `selector.py` head 에 set-attn block 추가 + 재학습 | full (0.6806) | **+0.005 ~ +0.015** | 1d |
| **C005** | bi-gru-encoder | unidirectional GRU → bidirectional. full history 활용. 거의 free lift. | `selector.py` `nn.GRU(..., bidirectional=True)` + dropout 0.08 → 0.12 | full (0.6806) | **+0.003 ~ +0.008** | 0.2d |
| **C006** | tcn-encoder-swap | GRU encoder → TCN (dilated 1D conv). 시계열 압축기만 교체, listwise head 유지. T=20 에서 GRU 와 비등하거나 약간 우위 (Bai et al. 2018). | `selector.py` encoder block 교체 + 재학습 | full (0.6806) | **±0.005** | 0.5d |
| **C007** | transformer-encoder-swap | GRU encoder → self-attention encoder (2-layer, 4-head). n=10k + T=20 에서 GRU 우위 가능성 큼 — *negative result* 박제도 가치 있음. | `selector.py` encoder block 교체 + warmup/dropout tuning | full (0.6806) | **±0.005** | 1~2d |
| **C008** | corrector-do-no-harm-gate | TinyCorrectionNet 에 gate head 추가: `correction_final = sigmoid(MLP(features)) × correction_raw`. **asymmetric loss** 로 `raw_hit && corrected_miss` 에 λ=5~10 penalty. plan-005 `corrector_decomp` 의 [0.005, 0.010) band 에서 발견된 *destructive 패턴* (2594 samples 중 -203 hits) 직접 차단. C002 의 *concrete sub-instance*. | `boundary.py` TinyCorrectionNet 에 gate Linear head 추가 + loss 함수 + 5-fold 재학습 | full (0.6806) | **+0.015 ~ +0.025** | 0.5~1d |
| **C009** | corrector-band-aware-loss | gate 추가 없이 loss 만 교체: `L = L2 × (1 + α × is_already_hit) × (1 + β × is_destructive_move)`. C008 의 *lite version* (구조 변경 없음, loss 만). C008 보다 lift 작지만 위험도/작업량도 작음 — C008 의 fallback. | `boundary.py` loss 함수만 수정 + 5-fold 재학습 | full (0.6806) | **+0.005 ~ +0.010** | 0.2d |
| **C010** | corrector-frenet-anisotropic-loss | loss 를 Frenet local frame 변환 후 `w_par=1.0, w_perp=1.0, w_bi=0.1` 비대칭 weight. plan-005 direction breakdown (parallel 0.0451, perp 0.0214, **binormal 0.0064**) 기반 — binormal capacity 낭비 회수. C008/C009 과 **직교** (stack 가능). | `boundary.py` loss 에 frenet rotation 추가 + 5-fold 재학습 | full (0.6806) | **+0.003 ~ +0.005** | 0.3d |
| **D001** | corrector-oracle-simulation | **사전 진단 (재학습 0)**: plan-005 `corrected_oof.npz` + raw scores 로 *perfect gate* 시뮬레이션 — "destructive samples 를 모두 skip 했을 때" 의 OOF 상한 계산. C008/C009 의 **go/no-go anchor**: < 0.66 이면 ROI 작아서 C001 우선; ≥ 0.67 이면 C008 진행. | `analysis/plan-007/oracle_gate.py` 신규 (재학습 없음) | OOF 0.6524 (E_corrected) | 진단 only, lift 없음 | 0.1d |

## 의존 / 직교 관계

| 묶음 | 관계 | 비고 |
|---|---|---|
| C001 + C002 | **직교** | 후보 쪽 vs corrector 쪽. lift 누적 가능 (`+0.03 ~ +0.06`). plan-007 의 A1+A2 와 동일. |
| C001 + C003 | **직교** | C001 의 35 후보 위에 C003 의 lgbm head 얹기 가능. |
| C002 + C003 | **직교** | corrector 가 좌표 보정, lgbm 이 ranking. 독립. |
| C004 vs C003 | **상충** | 둘 다 ranking head 교체 — 둘 중 하나만. |
| C005 / C006 / C007 | **상호 배타** | encoder swap 3선택지 중 1. |
| C001 + (C005 or C006) | **직교** | encoder swap + 후보 다양화 동시 가능. |
| C001 + Idea 1 (연속 heatmap) | **직교** | C001 = candidate 단계, Idea 1 = regime scoring 단계. |
| C001 vs Idea 2 (학습 후보) | **상충** | 둘 다 candidate 단계 — 손 추가 vs 데이터 학습. Idea 2 가 *재현 가능 손 추가* 의 일반화. |
| C002 vs C008 / C009 | **상충 (C008/C009 ⊂ C002)** | C008/C009 = C002 의 concrete sub-instance. C002 generic redesign 대신 C008 부터 시도 권장 (motivation metric 명확). |
| C008 + C010 | **직교** | gate head (when to correct) + frenet weight (how to correct) — 독립. stack 시 lift `+0.018 ~ +0.030` 추정. |
| C009 + C010 | **직교** | loss 항만 합치면 됨 — 가장 cheap 한 stacking. |
| D001 → C008 | **선행 진단** | D001 oracle 상한 < 0.66 이면 C008/C009/C010 모두 skip, C001 로 회귀. |
| C008 vs C009 | **상충 (C009 ⊂ C008)** | C009 = C008 의 lite version (gate 없이 loss 만). C008 실패 시 C009 fallback 또는 C008 의 ablation. |

## 권장 우선순위 (LB-단위 ROI / 작업량 기준)

1. **C001** — plan-007 이미 채택. 가장 명확한 lift, 가장 cheap.
2. **D001 → C008** — D001 (사전 oracle 진단, 0.1d, 재학습 0) 이 ≥ 0.67 이면 C008 (do-no-harm gate) 진행. corrector lift +0.0274 (전체 main engine) + plan-005 corrector_decomp 의 destructive band 명확 → ceiling 큼. C002 generic 대신 C008 부터 (motivation metric 구체적).
3. **C003** — sample-efficient, plan-005 산출 재사용으로 0.5d 만에 시도. NN 외 다른 inductive bias.
4. **C010** — frenet anisotropic loss. C008 과 stack 가능, 거의 free.
5. **C005** — bi-GRU. 거의 free, 위험 없음. 1순위 묶음에 옵션 추가 가능.
6. C009 — C008 의 fallback (gate 학습 실패 시).
7. C004 / C006 / C007 — C001~C003 의 lift 가 saturate (< +0.005pp) 한 *이후* 검토.

## 평가 메트릭 기반 fit 분석 — encoder swap (C005~C007) 이 낮은 우선순위인 이유

- 메트릭: hit_rate@threshold (discrete listwise ranking). encoder 의 시계열 압축 품질은 *간접 영향*.
- 4-way LB 분해: GRU 자체의 LB contribution = +0.0104pp (full=0.6806 − A=0.6796 의 sibling 비교 또는 A=0.6796 − E=0.6692). encoder 를 swap 해서 얻을 marginal lift 는 이 +0.0104pp 의 *일부* — 즉 ceiling 이 낮음.
- 반면 C001~C003 은 *다른 lift component* (candidate diversity, corrector, ranking head) 를 건드리므로 ceiling 이 +0.02pp 이상.

## Idea 1/2 와의 통합 — 6가지 ablation 셀

기존 2축 (Idea 1 ✕ Idea 2) × 본 섹션의 candidate-diversification (C001) 으로 확장 시:

| candidate 단계 | regime scoring | ablation 명칭 |
|---|---|---|
| 손 27 (현재) | 이산 18 (현재) | baseline (full = 0.6806) |
| 손 27 + **C001 35+** | 이산 18 | C001 단독 |
| 손 27 | 연속 heatmap (Idea 1) | Idea 1 단독 |
| 학습 N (Idea 2) | 이산 18 | Idea 2 단독 |
| 손 27 + C001 | 연속 heatmap | C001 + Idea 1 |
| 학습 N (Idea 2) | 연속 heatmap | 둘 다 (최대 변경) |

→ C001 은 *손 후보 일부 확장* 이고 Idea 2 는 *손 후보 → 데이터 학습* 의 일반화이므로, **C001 을 Idea 2 의 stepping stone** 으로 위치시킬 수 있음. 즉:
- 단기: C001 (손으로 8개 추가, 작업량 0.5d) 로 candidate diversification 의 LB lift 가치 확인.
- 중기: lift 가 +0.02pp 이상 확인되면 Idea 2 (학습 후보 N=35~50) 로 일반화 — paper contribution 가치 동반.

## 한 줄 정리

**C001 (후보 다양화) → D001 사전 진단 → C008 (do-no-harm gate) → C003 (LightGBM ranker stacking) 순으로 plan-007/008/009 후보 박제**. C008/C009/C010 은 plan-005 corrector_decomp 의 destructive [0.005, 0.010) band (2594 samples, -203 hits) 를 직접 겨냥 — C002 generic redesign 의 *concrete instance* 들. encoder swap (C005~C007) 은 ceiling 낮아 후순위. Idea 1 (연속 heatmap) / Idea 2 (학습 후보) 와 직교/일반화 관계로 묶이므로, C001 의 LB lift 검증 결과가 Idea 2 의 ROI 판단 anchor 가 됨.
