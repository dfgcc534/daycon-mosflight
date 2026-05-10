# plan-003 results — Residual GRU Lean Baseline + Component Ablation Grid + Winning-Components Combined

작성: 2026-05-11 (KST). status: **partial** (carry-over: lb_score 회수 대기).

## §1. 종합 표

| exp_id | method | key component | cv_mean_eucl ± std | per-axis MAE (x,y,z) | hit@0.05 | hit@0.10 | lb_score | total train (s) |
|---|---|---|---:|---|---:|---:|---:|---:|
| **B001_linear-2pt** | polyfit (w=2,d=1) | reference (closed-form floor) | 0.012941 ± 0.000584 | 0.00705, 0.00710, 0.00504 | 0.9571 | 0.9923 | **0.60** (plan-001) | 0.8 |
| S001_cspline-natural-full | cspline (w=11, natural) | reference (cspline floor) | 0.017418 ± 0.000713 | 0.00960, 0.00960, 0.00660 | 0.9302 | 0.9842 | 0.4932 | 5.4 |
| **R001_baseline-residual-gru** | gru-residual | rel + GRU(64,2,0.1) huber + linear | **0.013383 ± 0.000718** | 0.00738, 0.00734, 0.00516 | 0.9602 | 0.9935 | (= R006) | 34.3 |
| R002_physics-features | gru-residual | + physics (vel/acc/jerk/κ) input_dim=13 | 0.015157 ± 0.000499 | 0.00817, 0.00807, 0.00636 | 0.9590 | 0.9929 | (미제출) | 57.2 |
| R003_ema-extrapolate | gru-residual | baseline_type=ema (α=0.5) | 0.014038 ± 0.000976 | 0.00759, 0.00770, 0.00561 | 0.9595 | 0.9936 | (미제출) | 80.8 |
| R004_wingbeat-oscillation | gru-residual | + wingbeat FFT (n_bins=3) input_dim=12 | 0.013476 ± 0.000684 | 0.00733, 0.00740, 0.00529 | 0.9593 | 0.9936 | (미제출) | 34.5 |
| R005_loss-mse | gru-residual | loss_type=mse | 0.013388 ± 0.000580 | 0.00740, 0.00736, 0.00514 | 0.9604 | 0.9934 | (미제출) | 27.7 |
| **R006_combined-winners** | gru-residual | winning=0 → R001 비트 동일 사본 | 0.013383 ± 0.000718 | 0.00738, 0.00734, 0.00516 | 0.9602 | 0.9935 | **carry-over pending** | 34.3 (R001 학습 재사용, 추가 0) |

(hit@0.20, hit@0.50 모두 = 1.0000, 표 단순화 위해 생략. submission.csv 의 dacon API 응답 = `{isSubmitted: True, detail: Success}`.)

## §2. per-experiment 분석

### §2.1 R001_baseline-residual-gru (lean baseline)

- 상태: complete. started_at 2026-05-10T23:56:34+09:00, duration 37.4s.
- CV mean_eucl = 0.013383 ± 0.000718.
- per-axis MAE = (x 0.00738, y 0.00734, z 0.00516) m.
- hit_rate (4 radii): @0.05=0.9602, @0.10=0.9935, @0.20=1.0, @0.50=1.0.
- fold_best_val_mean_eucl = [0.01439, 0.01230, 0.01290, 0.01369, 0.01364].
- fold_best_epoch = [1, 10, 1, 4, 17] — *대부분 1~10 epoch 내 수렴*. 잔차 분포가 좁아 GRU 가 빠르게 plateau, early-stop 발동 빈번.
- best ckpt path: `runs/baseline/R001_baseline-residual-gru/ckpt/fold{0..4}.pt` (.gitignore).
- train_device: `cuda:0` (RTX/Hopper-class GPU 추정 device_count=2 환경).
- 특이사항: NaN/Inf 0건, training divergence 0건. 최소 epoch 으로 plateau 도달은 *baseline (linear extrapolation) 이 이미 strong* 하다는 시그니처.

### §2.2 R002_physics-features (H1 검증)

- 단일 변경 (vs R001): feature_components → [relative, physics] (input_dim=3 → 13).
- physics: per-axis vel(3) + acc(3) + jerk(3) + curvature(1).
- 결과: cv = 0.015157 ± 0.000499. **R001 대비 paired Δ = +0.001775** (5 fold 모두 양수, sign agreement 5/5). *non-winning*.
- fold_best_epoch = [..., 14, 16, 17] — R001 보다 학습 길게 진행 (입력 차원 ↑ → 표현력 학습 시도).
- **H1 (physics features 가 GRU residual 학습 표현력 보강) 잠정 기각** — caveat #5 박제: jerk magnitude (~78 m/s³) raw 입력이 SNR 악화로 noise dominant feature 가 되어 GRU 가 overfitting / 학습 안정성 저해 가능. normalize 적용 시 결과 달라질 가능성 존재 (별도 plan).

### §2.3 R003_ema-extrapolate (H2 검증)

- 단일 변경 (vs R001): baseline_type linear → ema (α=0.5).
- 결과: cv = 0.014038 ± 0.000976. **R001 대비 paired Δ = +0.000655** (4/5 fold 양수, fold 2 만 +0.0027 outlier). *non-winning*.
- fold_best_epoch = [37, 37, 7, 48, 41] — *훨씬 더 오래 학습* (linear baseline 의 잔차 분포보다 EMA baseline 의 잔차 분포가 더 어렵게 수렴).
- **H2 (EMA baseline 이 더 unbiased → 잔차 분포 좁아짐) 잠정 기각** — *반대로 학습이 더 어려워졌음*. EMA 가 closed-form linear 보다 본 데이터에서 *더 noisy 한 baseline* 가 되어 GRU 가 더 큰 잔차를 학습해야 함. caveat #6 박제: α=0.5 fixed (별도 plan 에서 sweep).

### §2.4 R004_wingbeat-oscillation (H3 검증)

- 단일 변경 (vs R001): feature_components → [relative, wingbeat] (input_dim=3 → 12).
- wingbeat: per-axis FFT magnitude n_bins=3 (DC + 1st + 2nd harmonic) → 9 features (sequence-level broadcast).
- 결과: cv = 0.013476 ± 0.000684. **R001 대비 paired Δ = +0.000093** (3/5 fold 양수, margin tiny). *non-winning, but tightest*.
- fold_best_epoch = [1~20] — R001 와 유사한 빠른 수렴.
- **H3 (wing-beat oscillation pattern 이 비행 상태 implicit class) 잠정 기각** — 단 margin (Δ=+9.3e-5) 이 fold-σ (~0.00072) 의 ~13% 영역 → noise 로 분류. caveat #4: "wingbeat" 라벨은 실제 wing-beat 가 아닌 11pt 저주파 oscillation 패턴 의미 (Nyquist=12.5Hz < 모기 wingbeat 수백 Hz 이라 aliasing).

### §2.5 R005_loss-mse (H4 검증)

- 단일 변경 (vs R001): loss_type huber → mse.
- 결과: cv = 0.013388 ± 0.000580. **R001 대비 paired Δ = +0.000005** (사실상 동등). *non-winning, near-tie*.
- fold_best_epoch = R001 와 거의 동일.
- **H4 (Huber outlier robustness 가 본 데이터에 작용) 검증: Huber prior 기각** — caveat #7 박제: PyTorch HuberLoss δ=1.0 default 가 본 데이터의 ~수 mm 잔차 분포에서는 *전 영역 quadratic* (≈ MSE) 이라 사실상 Huber↔MSE 동등. δ를 0.001~0.01 으로 작게 잡았으면 다른 결과 가능 (별도 plan).

### §2.6 R006_combined-winners (H5 검증, winning=0 분기)

- 단일 변경 (vs R001): exp_id 만 (winning=0 → R001 비트 동일 사본).
- 결과: cv = 0.013383 ± 0.000718 (= R001 정확히 동일).
- **H5 (combined-additive 효과) 검증 자체가 *trivial***: ablation winning 0개로 R006 = R001, interaction effect 측정 불가.
- caveat #14 + #15 + #16 박제: winning 기준 (paired mean Δ < 0) 이 너무 보수적이라 marginal positive Δ (R004 +9.3e-5, R005 +5e-6) 모두 non-winning 으로 분류 → R006 가 trivial 분기 진입. 별도 plan 에서 *strict mode* (|Δ| ≥ fold-σ + 부호 일관성 ≥ 4/5) 또는 *완화 mode* (Δ < +noise_margin, 예: noise_margin=0.0002) 검증 필요.

## §3. paired comparison

### §3.1 B001 vs R001~R006 (same-fold Δ_fold = R00x.fold - B001.fold)

| exp | fold0 Δ | fold1 Δ | fold2 Δ | fold3 Δ | fold4 Δ | mean Δ | sign agreement |
|---|---:|---:|---:|---:|---:|---:|---:|
| R001 | +0.000681 | +0.000284 | +0.000309 | +0.000557 | +0.000378 | **+0.000442** | 5/5 |
| R002 | +0.001913 | +0.002234 | +0.002422 | +0.002252 | +0.002261 | +0.002216 | 5/5 |
| R003 | +0.000691 | +0.000581 | +0.002996 | +0.000804 | +0.000414 | +0.001097 | 5/5 |
| R004 | +0.000486 | +0.000328 | +0.000459 | +0.000794 | +0.000608 | +0.000535 | 5/5 |
| R005 | +0.000411 | +0.000449 | +0.000430 | +0.000557 | +0.000386 | +0.000447 | 5/5 |
| R006 | +0.000681 | +0.000284 | +0.000309 | +0.000557 | +0.000378 | +0.000442 | 5/5 |

**관찰**: 모든 R00x 가 B001 보다 *5/5 fold 모두 더 큼* (mean_eucl 기준). 즉 **closed-form B001 이 ablation 전 영역에서 paired-floor**. 이는 plan-001 의 결론을 강화 — "잔차 ~수 mm 영역에서 GRU 의 학습 가치는 음수" 라는 강한 증거. paired Δ 모두 임계값 +0.005 이내 → §3.2 C 합격 기준 통과.

### §3.2 R001 vs R002~R005 (winning 식별 근거)

| exp | mean Δ vs R001 | fold-σ (R001=0.000718) 의 배수 | sign agreement (R00x ≥ R001) | winning? |
|---|---:|---:|---:|:-:|
| R002 | +0.001775 | 2.5× | 5/5 | no |
| R003 | +0.000655 | 0.9× | 4/5 (fold 0 동등) | no |
| R004 | +0.000093 | 0.13× (noise) | 3/5 | no |
| R005 | +0.000005 | 0.007× (sub-noise) | 3/5 | no |

**관찰**: R002 만 fold-σ 의 2.5× 영역으로 *유의한* 악화. R003 도 mean 기준 fold-σ 영역 (~1×) 으로 *주관적* 으로 noise margin 너머. R004/R005 는 fold-σ 의 13%/0.7% 영역으로 *명백히 noise*. 본 plan 의 보수적 winning 기준 (Δ < 0 strict) 이 R004/R005 의 marginal 결과를 *false-non-winning* 으로 처리할 가능성 — caveat #15 박제.

### §3.3 R001 vs R006 (combined 효과 측정)

- R006 = R001 비트 동일 사본 → fold Δ 모두 0.0, mean Δ = 0.0.
- **H5 verification trivial** — interaction effect 추정 불가능. *별도 plan 에서 strict-mode winning 으로 R004/R005 winning 처리해 보고 R006 학습* 필요.

## §4. winning trace + R006 자동 생성

상세 내용: `analysis/plan-003/winning_trace.md`.

요약:

```
winning = {R002: F, R003: F, R004: F, R005: F}, count = 0
→ R006 config = R001 비트 동일 (exp_id 만 R006_combined-winners)
→ R006 = R001 ckpt + summary 직접 cp (학습 skip)
→ R006.cv = 0.013383 = R001.cv ≤ R001.cv + 0.001 → fallback = False
→ lb_exp_id = R006_combined-winners
```

## §5. H1~H5 검증/기각 (CV 축 종합)

- **H1 (physics features)** — *기각*. paired Δ +0.001775 (5/5 fold). 가능 원인: jerk magnitude SNR 악화 (caveat #5).
- **H2 (EMA baseline)** — *기각*. paired Δ +0.000655 (4/5 fold). 가능 원인: α=0.5 fixed 가 본 데이터의 *velocity smoothness 분포* 와 mismatch (caveat #6).
- **H3 (wing-beat)** — *기각, 단 margin tiny (noise)*. paired Δ +0.000093 (fold-σ 의 13%). caveat #4 박제 — "wingbeat" 라벨이 실제 모기 wingbeat 가 아닌 저주파 oscillation 임을 강조.
- **H4 (loss MSE / Huber prior 약함)** — *Huber prior 기각*. paired Δ +0.000005 (사실상 동등). PyTorch δ=1.0 default 가 본 데이터의 mm-단위 잔차 영역에서 quadratic 영역에 머물러 효과 없음 (caveat #7).
- **H5 (combined-additive)** — *trivial 분기 (winning=0)*. R006 = R001 비트 동일이라 interaction 측정 불가. caveat #14 + #15 + #16 박제 — 본 plan 의 strict winning 기준이 marginal R004/R005 를 false-non-winning 으로 처리한 것이 H5 검증 차단.

## §6. lb_exp_id 위치 (carry-over pending)

- lb_exp_id = R006_combined-winners (= R001 비트 동일).
- API 응답 (2026-05-11T00:08 KST): `{isSubmitted: True, detail: Success}`. score 회수는 dacon.io 페이지 수동 확인 (DACON API 가 get_score 미지원).
- **CV 위치**: R006 CV 0.013383 vs B001 CV 0.012941 (Δ = +0.000442) — neural model 이 closed-form floor 를 *넘지 못함*.
- **LB 비교 prior**: B001 LB = 0.60, S001 LB = 0.4932. CV ↔ LB ρ = +0.90 (plan-002 분석). R006 LB 가 0.60 보다 작으면 *neural 모델 자체가 closed-form floor 미달* 증거 (CV-LB 일관성 가정).
- **expected LB (CV-LB extrapolation)**: B001 (CV=0.0129, LB=0.60) 대비 R006 CV=0.0134 → 단순 비례로 LB ≈ 0.55~0.59 영역 추정. 단 plan-002 의 S001~S004 가 이 비례를 벗어난 사례 (smoothing spline S004: CV=0.033, LB=0.22 — 격차 큼) 가 있어 *LB 회수 후 비교 필수*.

## §7. 학습 안정성

- 6 exp 모두 NaN/Inf 0건 (loss + val_mean_eucl). training divergence 0건.
- early-stop 발동 분포:
  - R001 fold_best_epoch [1, 10, 1, 4, 17] — *최저 1 epoch* (잔차가 너무 작아 GRU 가 즉각 plateau).
  - R002 [..., 14, 16, 17] — physics features 표현 학습 시도로 약간 더 오래.
  - R003 [37, 37, 7, 48, 41] — EMA baseline 의 잔차가 더 어렵게 수렴.
  - R004/R005 R001 와 유사 패턴.
- CUDA OOM 0건, GPU device_count=2 환경에서 device=cuda:0 만 사용 (decision-note v6).

## §8. submission 결과

- `runs/baseline/R001_baseline-residual-gru/submission.csv` (10000 rows, schema OK) — c13 생성, c13 테스트.
- `runs/baseline/R006_combined-winners/submission.csv` (10000 rows, schema OK, R001 과 비트 동일 — winning=0 cp 분기).
- DACON 자율 제출 1회: `dacon-submit` skill via Skill tool, fallback path 미사용.
- API 응답: `{isSubmitted: True, detail: Success}`.
- LB 점수: **carry-over pending** (사용자가 dacon.io 페이지에서 회수 후 별도 commit 으로 박제).

## §9. 다음 plan 후보 (enumeration only)

local 의사결정 권한. server 가 우선순위 정하지 않음:

1. **R001~R005 ablation 의 LB 신호 회수 (= 5 LB 제출 plan)** — 본 plan 의 CV-LB ranking divergence 측정 (caveat #10/#13).
2. **Strict-mode winning 기준 + R006 재학습** — `noise_margin = -fold_σ` 또는 `|Δ| ≥ fold-σ + 부호 일관성 ≥ 4/5` 로 R004 winning 처리 → R006 = ["relative", "wingbeat"] 학습 (input_dim=12) → CV/LB 회수 (caveat #14/#15).
3. **GRU hyperparameter sweep** — hidden ∈ {32, 128, 256}, layers ∈ {1, 3}, dropout ∈ {0.0, 0.3, 0.5}, lr ∈ {3e-4, 3e-3} grid + lr scheduler.
4. **Architecture 비교** — TCN, Transformer, MLP-residual.
5. **Physics features normalization** — z-score per-feature 또는 per-axis log-scale (caveat #5 의 jerk SNR 문제 해결 시도).
6. **EMA α sweep** — α ∈ {0.1, 0.3, 0.7, 0.9} (caveat #6).
7. **Huber δ sweep** — δ ∈ {0.001, 0.01, 0.1, 1.0} (caveat #7).
8. **Ensemble (R00x + B001)** — closed-form floor 와 neural delta 의 weighted mean.
9. **Kalman / Savitzky-Golay 입력 평활** — noise reduction prior.
10. **TTA inference (X-Y rotation, Y-flip)** — 본 plan out-of-scope, 후처리만으로 검증.
11. **Hit-rate aware loss** — hit@1cm 직접 최적화 (huber 대신 hinge 류 + radius 기반 reward).
12. **Interaction effect 분리 측정 (R002+R004 vs R002 only vs R004 only)** — caveat #14.

## §10. 의사결정 anchor (caveat 모음)

- caveat #1: paired Δ 가 fold-σ 보다 작으면 noise. 본 plan 의 R004/R005 가 그 영역.
- caveat #5: jerk normalize 미적용 (lean baseline) — H1 기각의 confound.
- caveat #6: EMA α=0.5 fixed — H2 기각의 confound.
- caveat #7: Huber δ=1.0 default — H4 검증 시 Huber↔MSE 동등 가능성 사전 박제, 결과로 확인.
- caveat #14: H5 의 additive 가정 — winning=0 trivial 분기로 검증 불가.
- caveat #15: 본 plan winning 기준 (Δ < 0 strict) 의 보수성 — *별도 plan strict mode* 가 필요한 가장 직접적 결론.
- caveat #16: combined fallback false (R006 = R001) 자체가 informative — *interaction 검증 필수성* 의 강한 신호.

본 plan 의 *주요 정보 산출*: (a) closed-form B001 이 paired-floor 라는 강한 신호 (5/5 fold 모두 GRU > B001), (b) marginal winning 기준의 false-non-winning 영역 발견 (R004 fold-σ의 13%, R005 0.7%), (c) physics/EMA 의 clear non-winning (별도 plan 의 normalize / α-sweep 가 의미 있음), (d) Huber prior 의 *데이터별 검증 결과* (default δ 영역에선 무효).
