# plan-002 results — Cubic spline interpolation baseline

**status**: all_complete (4 LB scores 회수 완료)
**finished_at (CV portion)**: 2026-05-10 KST
**4 LB submissions submitted_at**: 2026-05-10T05:09~05:12 KST
**4 LB scores collected_at**: 2026-05-10T14:11 KST (사용자 dacon.io 직접 확인)

---

## §1. 종합 표 — B001 + S001~S004 (5 entries)

| exp_id | method | hyperparam | cv_mean_eucl ± std | per-axis MAE [x, y, z] | hit@0.10 | runtime (s) | lb_score |
|---|---|---|---|---|---|---|---|
| **B001_linear-2pt** (plan-001) | polyfit | w=2, d=1 | **0.01294 ± 0.00058** | [0.0070, 0.0071, 0.0050] | 0.9923 | 0.8 | **0.60** (plan-001) |
| S001_cspline-natural-full | cspline | w=11, BC=natural | 0.01742 ± 0.00071 | [0.0096, 0.0096, 0.0066] | 0.9842 | 5.4 | **0.4932** |
| S002_cspline-notaknot-full | cspline | w=11, BC=not-a-knot | 0.05370 ± 0.00282 | [0.0277, 0.0288, 0.0235] | 0.8815 | 5.3 | 0.1204 |
| S003_cspline-window-grid | cspline | per-axis [(5,nat),(5,nat),(4,nat)] | 0.01740 ± 0.00071 | [0.0096, 0.0096, 0.0066] | 0.9842 | 226.8 | 0.4926 |
| S004_smoothing-spline-tuned | smoothing | k=3, s=[1e-4, 1e-4, 1e-4] | 0.03322 ± 0.00270 | [0.0191, 0.0176, 0.0115] | 0.9506 | 17.1 | 0.2178 |

CV-best of S001~S004 = **S003** (0.01740). LB-best of S001~S004 = **S001** (0.4932; S003 와 ΔLB=0.0006 tie).
4 변형 모두 B001 floor (CV 0.01294 / LB 0.60) 미달.

---

## §2. per-experiment 분석

### S001 — natural BC, full 11-pt window

CV mean_eucl 0.01742 — B001 floor 대비 +0.0045 worse, 모든 fold 에서 강하게 worse (paired Δ 분포 [+0.0043 ~ +0.0048], sign=1.00). Natural BC ("끝 곡률 0") 가정이 외삽을 *flat* 으로 만들지만, 11점 전체 보간이 oldtimesteps 노이즈를 fit 끝까지 끌고 와 boundary 곡률을 왜곡. 결과: B001 등속 외삽 (oldtimesteps 무시) 보다 *systematically* 부정확.

### S002 — not-a-knot BC, full 11-pt window

CV mean_eucl 0.05370 — B001 대비 +0.041 (3× worse), per-axis MAE 도 3× 영역. Not-a-knot 은 boundary 의 cubic 을 그대로 외삽 영역으로 연장하므로 노이즈가 amplify. hit@0.10 도 0.88 까지 떨어짐. 본 plan 의 *의도된 worst case* — 어떤 정상 데이터도 이 BC 로 외삽 + 노이즈를 같이 쥐고 가면 flat 가정보다 훨씬 부정확함.

### S003 — per-axis (window × bc_type) grid

CV mean_eucl 0.01740 — S001 과 동급. 12-cell grid (window ∈ {4,5,7,11} × BC ∈ {natural,not-a-knot,clamped}) 의 5-fold inner CV 가 5 outer fold 에서 동일하게 [(5,nat),(5,nat),(4,nat)] 선택. 즉 **clamped 가 한 번도 안 뽑힘**. H2 의 핵심 prediction (clamped 의 chord-derivative ≈ B001 등속 외삽) 은 정량적으로 *노이즈 데이터에서는 clamped 가 chord-derivative noise 까지 같이 외삽해 natural 보다 worse* 라는 결과로 부분 refuted. 작은 window + natural 이 fit 영역의 노이즈를 적게 가져오는 한편 boundary 곡률 0 가정이 외삽 안정.

### S004 — smoothing spline tuned

CV mean_eucl 0.03322 — 본 plan 의 가장 유망 후보였으나 *worst-but-S002*. Inner CV 가 모든 axis 에서 s=1e-4 선택 (s_grid {0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1} 중 lower-end 인접). s=0 (interpolation) 보다 약간의 smoothing 이 inner CV val MAE 를 줄이지만, 이는 fit 영역 거리 metric 에서의 이득. 외삽 영역 (t=+80) 에서는 boundary 곡률 fit 정확도가 더 중요해 smoothing 이 손해. 즉 inner CV 의 axis MAE 가 t=+80 의 mean_eucl 의 좋은 proxy 가 아님 (selection bias).

---

## §3. paired comparison — same-fold Δ vs B001

`Δ_fold[i] = S00x.fold_means[i] - B001.fold_means[i]` (음수 = S00x 가 우수). 5 fold 동일 split (kfold_split deterministic, seed=42, plan-001 §3.1 와 100 % 동등).

| exp_id | fold 0 | fold 1 | fold 2 | fold 3 | fold 4 | mean Δ | sign 일관성 |
|---|---|---|---|---|---|---|---|
| B001 (ref) | 0.01371 | 0.01201 | 0.01259 | 0.01313 | 0.01326 | — | — |
| S001 | +0.00465 | +0.00429 | +0.00439 | +0.00475 | +0.00430 | **+0.00448** | 1.00 (전부 worse) |
| S002 | +0.04376 | +0.03834 | +0.03812 | +0.04290 | +0.04065 | **+0.04076** | 1.00 |
| S003 | +0.00462 | +0.00428 | +0.00437 | +0.00473 | +0.00427 | **+0.00446** | 1.00 |
| S004 | +0.02081 | +0.01773 | +0.01880 | +0.02445 | +0.01958 | **+0.02027** | 1.00 |

모든 4 변형 × 모든 5 fold 에서 B001 보다 strictly worse. mean Δ ≫ B001 fold-σ (0.00058). 즉 noise 영역 *훨씬 위* — 단일 plan 의 4 spline 변형 그 어떤 것도 등속 외삽 baseline 을 넘지 못함을 *결정적으로* 박제.

---

## §4. S003 의 axis 별 chosen 분해

5 outer fold 모두 동일:

| axis | chosen (window, bc_type) |
|---|---|
| x | (5, natural) |
| y | (5, natural) |
| z | (4, natural) |

12-cell grid 중 clamped 는 어떤 axis/fold 에서도 1위가 아님. natural BC 가 dominate. 짧은 window (4~5) 가 11pt 보다 우월 — oldtimesteps 노이즈 누적이 외삽 정확도를 해친다는 plan-001 §1.3 의 핵심 통찰을 cspline 분기에서 재확인.

---

## §5. S004 의 axis 별 chosen + s_grid 곡선

5 outer fold 의 chosen s_per_axis = [1e-4, 1e-4, 1e-4] 4 개, [1e-3, 1e-4, 1e-4] 1 개. 즉 axis-x 가 한 fold 에서만 1e-3 으로 약간 더 smoothing. Final full-train re-tune = [1e-4, 1e-4, 1e-4] 3 axes 동일.

s=0 (interpolation) 은 inner CV 에서 axis MAE 가 약간 더 큼 (smoothing 의 이득 있음). 하지만 그 이득이 *외삽 metric 까지 carry over 안 됨* — H3 refuted.

s_grid axis MAE 곡선은 `runs/baseline/S004_smoothing-spline-tuned/summary.json` 의 `full_train_grid_errors` 에 박제.

---

## §6. H1 / H2 / H3 verdict

- **H1**: 11pt natural / not-a-knot 보간 → polyfit-11pt 급 (0.03~0.05) — **CV·LB 모두 confirmed**. CV: natural 0.0174 (0.02 미만으로 약간 더 좋음), not-a-knot 0.0537 (예측 범위 안). LB: natural 0.4932 / not-a-knot 0.1204 — 둘 다 B001 LB 0.60 미달, not-a-knot 은 LB 에서 4× worse (외삽 폭주가 hit_rate 에 결정적).
- **H2**: window × BC grid 가 B001 근접 — **CV·LB 모두 partially refuted**. CV 0.0174 / LB 0.4926 영역 (S001 동급, ΔLB=0.0006) 에서 멈춤. clamped 가 한 번도 안 뽑힘. small-window 의 이득은 있으나 등속 polyfit 의 CV·LB floor 모두 넘지 못함.
- **H3**: smoothing 이 노이즈 흡수로 B001 위협 — **CV·LB 모두 refuted**. CV 0.0332 / LB 0.2178 — fit-영역 axis MAE 의 이득이 외삽 영역 mean_eucl 에도, LB hit_rate tail 에도 transfer 안 됨. **본 plan 의 가장 강한 가설 reject** (smoothing prior 자체가 외삽 task 와 부적합).

---

## §7. CV ↔ LB 상관 분석 (점수 회수 완료)

5 점 (B001 + S001~S004) (cv_mean_eucl, lb_score):

| exp | CV mean_eucl (↓) | LB hit_rate (↑) | CV rank | LB rank |
|---|---|---|---|---|
| B001 | 0.01294 | 0.6000 | 1 | 1 |
| S003 | 0.01740 | 0.4926 | 2 | 3 |
| S001 | 0.01742 | 0.4932 | 3 | 2 |
| S004 | 0.03322 | 0.2178 | 4 | 4 |
| S002 | 0.05370 | 0.1204 | 5 | 5 |

**Spearman ρ(CV ↑ ↔ LB ↓) = +0.90** (n=5).
- d_i (rank diff) = [0, +1, −1, 0, 0] → Σd² = 2
- ρ = 1 − 6·Σd² / (n·(n²−1)) = 1 − 12/120 = **0.90**
- p-value (Student's t with df=3): t = ρ·√(3/(1−ρ²)) = 0.90·√(3/0.19) ≈ 3.58 → p ≈ 0.037 (양측). n=5 에서는 신뢰성 한계, 단 effect size 큼.

**점수 도착 전 3 prior 시나리오 verdict**:
- 비례 시나리오 → **거의 적중** (rank 일치 외 S001/S003 만 미세 swap, ΔLB=0.0006 = noise tie).
- 반전 시나리오 (S004 LB 회복) → **반박** (LB 0.2178 — S001/S003 의 절반).
- 약-상관 시나리오 → **반박** (4 LB 가 0.12 ~ 0.49 범위로 강하게 분리).

**해석**:
- **CV mean_eucl 이 LB hit_rate 의 신뢰성 있는 proxy**. 향후 plan 은 CV 만으로 우선순위 결정 가능 → LB 슬롯 절약 (1일 5회 한정 자원).
- 단, S001 vs S003 의 ΔCV=0.00002 / ΔLB=0.0006 는 *둘 다 noise 영역* — 5e-3 미만 차이는 LB 도 random.
- not-a-knot (S002) 의 LB 0.1204 가 가장 큰 outlier — 외삽 폭주가 hit_rate 의 *전체 분포* 까지 망가뜨림 (cv mean 만 보면 3× worse 인데 LB 는 5× worse).

**§N+3 #8 caveat 확인**: LB 차이 ≤ 0.005 영역은 noise — S001/S003 의 ΔLB=0.0006 이 정확히 그 영역. tie 처리 정당.

---

## §8. submission 결과

4 LB 제출 모두 isSubmitted=True (api Success). lb_log @ `analysis/plan-002/lb_log.md`.

| order | exp_id | submitted_at (KST) | api response | lb_score |
|---|---|---|---|---|
| 1 | S004 | 2026-05-10T05:09 | Success | 0.2178 |
| 2 | S003 | 2026-05-10T05:11 | Success | 0.4926 |
| 3 | S001 | 2026-05-10T05:11 | Success | 0.4932 |
| 4 | S002 | 2026-05-10T05:12 | Success | 0.1204 |

Budget: 4/5 일일. 1 슬롯 contingency (미사용). Carry-over: closed (2026-05-10T14:11 KST 사용자가 dacon.io 대회 페이지 에서 4 점수 확인 후 server agent 에 전달, 일괄 갱신 완료).

---

## §9. 다음 plan 후보 (enumeration only — CV-LB ρ=+0.90 박제 반영)

CV 가 LB proxy 로 신뢰 가능 → 다음 plan 은 *CV 만 보고도* 우선순위 결정 가능. LB 슬롯 절약하고 더 많은 ablation 가능.

1. **Kalman / Savitzky-Golay 입력 평활 → polyfit**: smoothing spline (S004) 이 "post-hoc 평활" 이라 H3 refute 의 원인 추정. 입력 측 평활 + 작은-window polyfit 이 다른 inductive bias. *S004 의 LB 0.2178 (CV 0.0332 시점에 예측 가능했던 수준)* 과 비교 후, 입력 측 평활이 CV 0.013 영역으로 들어오는지 확인.
2. **Velocity model**: t=0 에서의 instantaneous velocity 추정 + 등속 외삽 (B001 의 일반화). 6/8/10-pt polyfit derivative 와 비교. B001 LB 0.60 floor 를 *명시적으로 노리는* 1순위 후보.
3. **Ensemble (B001, S003, S004)**: CV-mean_eucl 측 약간의 다양성 + LB hit_rate 측 다른 tail. 단순 평균 vs CV-weighted. 단 CV-LB ρ=+0.90 으로 ensemble 의 LB upside 가 작을 수도 (CV 비례 → 가중평균이 단순 winner 못 이김).
4. **Neural seq2seq (LSTM / 1D-Transformer)**: 11pt × 3-axis 입력으로 +80 ms 출력. small data — 강한 augmentation + light model 필요.
5. **Per-axis combination of B001 + smoothing axis-wise**: B004 의 generalization. axis 별 best of {polyfit, cspline, smoothing}. 단 S003 가 이미 axis-별 grid 를 한 결과 *clamped 도 안 뽑힘* — 여기 axis 별 method 도 polyfit dominate 가능성 큼.
6. **Hit-radius probing**: 1 LB 슬롯 사용해 hit_rate 반경/분모 추정. *deprioritize* — CV-LB ρ 0.90 박제로 metric 추정 가치가 줄어듦.
