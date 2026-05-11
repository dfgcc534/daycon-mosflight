# plan-006 후보 (plan-005 진단 결과 기반)

각 후보는 plan-005 의 *근거 metric* + *예상 ROI* + *작업 범위* + *선행 조건* 4 항목 박제.

**우선순위 (descending ROI×evidence)**: 1 (selector simplify) > 2 (corrector loss) > 3 (high-speed regime) > 4 (bias re-tuning).

---

## 후보 1: Selector simplify / arch 교체 (소량 marginal contribution + 약한 ranking)

- **근거 metric**: 두 진단이 같은 방향:
    1. `selector_decomp.json["top_k"]` = {1: 0.1262, 3: 0.2183, 5: 0.2815}: selector 가 *진짜 best candidate* 를 1순위로 picking 12.6% — 27 cands 중 best 를 fluent 하게 ranking 하는 능력 매우 약함.
    2. `component_contribution.json["marginal_contribution"]`: gru contribution = soft +0.0052, regime = soft +0.0029. **둘 다 §N+3 #9 noise floor (±0.005) 근방** — gru/regime 의 추가 정보가 noise 에 가까움.
    3. intervention 의 helped/hurt rate: gru = 0.565/0.435, regime = 0.509/0.491 — 거의 50:50 (무작위적 개입). plan §8.6: marginal_contribution.gru < 0.02 → 'selector arch 교체 / 단순화' 조건 충족.
- **예상 ROI**: 두 path 후보:
    - (a) **단순화** (drop gru, physics-only baseline): wall-time 절감 (선택자 학습 ~20min → 0min), hit 거의 동일 (max -0.5pp). LB score 회복은 다른 후보로.
    - (b) **arch 교체** (TCN / Transformer): top-1 ranking 개선 → soft hit 향상 (top-1 12.6% → 25%+ 가능성, soft 0.660 → 0.680+ 추정). risk: TCN ~2× train time.
- **작업 범위**: plan-006 STAGE 1 — (a) physics-only baseline 측정 → (b) arch ablation (attn_gru vs TCN vs hierarchical-TCN vs latent-physics-TCN; selector.py 에 이미 모두 구현됨). ensemble size, hidden dim grid.
- **선행 조건**: `analysis/plan-005/selector_decomp.json` (top_k + per_regime_hit) + `component_contribution.json` (variants + marginal + intervention).

---

## 후보 2: Corrector loss 재설계 (best-of-27 oracle 보존 제약)

- **근거 metric**: `oracle_summary.json["corrector_oracle_gain"]` = -0.0077 (corrector 가 best-of-27 oracle 을 0.77pp 떨어뜨림). per-regime: 14/18 regime 에서 negative gain. corrector 의 family-aware loss 가 *best 후보를 옮기는* 부작용.
- **예상 ROI**: oracle ceiling 회복 → soft hit 추가 향상 (best-of-27 0.71 회복 시 selector 의 ranking 능력에 대한 *상한* 도 함께 들림). 현 corrector 의 selector-soft 효과 (+0.89pp) 는 trade-off — pareto frontier 재탐색 필요.
- **작업 범위**: plan-006 STAGE 2 — corrector loss 에 *oracle 보존 항* 추가 (best-cand 의 residual 만 별도 weight). cap=0.006 은 충분 (saturation 3.6%) 이라 cap 변경은 불필요. apply_scale=1.0 의 soft tuning 만 후보.
- **선행 조건**: `analysis/plan-005/oracle_summary.json` (per-regime 4/13/16/17 만 + 인 finding) + `corrector_decomp.json` (cap_saturation 3.6% 박제).

---

## 후보 3: High-speed regime adaptive candidates

- **근거 metric**: `failure_b001.json["worst_regime_counts"]` = {'10': 8, '11': 6, '12': 5, '13': 19, '14': 11}. worst-100 의 49 (~50%) 이 regime 10-14 (high-speed × high-curvature × high-fatigue). `oracle_summary.json["per_regime"]` 도 regime 13/14/16/17 의 raw oracle 이 0.33-0.62 로 가장 낮음.
- **예상 ROI**: high-speed regime 에 specialized cand template 추가 → oracle 0.33→0.50 가능성 (해당 regime 의 N 합 ~2400 / 10000 = 24%, 전체 hit 기여 ~+1.5-2.5pp 추정). risk: 27 → 35+ cand 로 selector 부담 ↑.
- **작업 범위**: plan-006 STAGE 3 — `selector.CANDIDATES` 에 high-speed regime 전용 spec 추가 (예: longer time_scale, larger jerk amplitude). selector 재학습 1회 + boundary 재학습 1회 (~30min wall-time).
- **선행 조건**: `analysis/plan-005/failure_b001.json` (worst-100 regime 분포) + `analysis/plan-004/regime_distribution.json` (regime 정의).

---

## 후보 4: Bias 가중치 (physics 0.65, regime 0.45) 재튜닝 — LOW priority

- **근거 metric**: `component_contribution.json["marginal_contribution"]`: regime contribution soft = +0.0029 (noise floor 근방). regime intervention 의 helped/hurt = 0.509/0.491 (~50:50). regime prior 의 가치는 *현재 spec 에서* 거의 없음 — 가중치 조정 효과도 작을 것.
- **예상 ROI**: cheap (학습 X, scoring 만; ~5min). prior 가중치 grid (4 × 3 = 12 setting) → soft hit 향상 추정 ±0.005 (noise floor 와 동급). 후보 1-3 보다 ROI 작지만 sanity-check 가치는 있음.
- **작업 범위**: plan-006 STAGE 4 (선택) — selector 재학습 X. plan-004 ens_prior (이미 박제) 의 physics/regime 분리 후, prior 가중치 grid 적용 + 12 setting soft hit 측정. corrector_oof.npz 도 그대로 활용.
- **선행 조건**: `analysis/plan-005/component_contribution.{json,md}` + `oof_selector_scores.npz` (ens_prior).

---

## 추가 가능 후보 (낮은 우선순위)

- **5. Binormal family 추가**: STAGE 3 의 binormal mean=0.0064 (parallel 의 1/7) 가 작아 ROI 낮음. *각 sample 의 z-축 trajectory variance* 가 충분히 클 때만 의미.
- **6. Per-fold corrector OOF**: 본 plan 의 corrector full-fit 은 약한 leakage 존재 (§N+3 #1). per-fold 재학습 시 wall-time ~50min, 진단 정확도 미세 향상.
- **7. selector 의 confidence threshold gating**: 본 plan margin_hist 의 p10 = (작음) 인 sample 만 별도 fallback (linear B001) → loss 줄이기. cheap 시도 가치.
