---
plan_id: a-002
version: 1
date: 2026-05-26 (Asia/Seoul)
status: all_complete
lane: a
inspired_by:
  - a-001 (KR002 input-yaw = 프로젝트 LB 신기록 0.6818. 핵심 발견 = **CV-LB 괴리** (입력 yaw OOF neutral +0.0024인데 LB +0.0060). KR002 가 본 plan baseline)
  - notes/LB_0.6780 코드공유.ipynb (Kalman CV 필터 + aux W multi-σ. 본 plan = 그 Kalman 이 *산출하고 버리는* 부산물(innovation·filtered state·CA)을 입력 feature 로 회수)
  - 020 (F0 baseline 0.6320 floor + hit_1cm/1p5cm metric 정의)
code_reuse:
  - module: src/io.py
    symbols: [load_all_samples, load_labels]
    reason: 데이터 loader. X (N,11,3), y (N,3).
  - module: src/pb_0_6822/selector.py
    symbols: [stable_fold_id]
    reason: MD5 5-fold split (plan-a-001/OOF 비교 호환).
  - module: analysis/plan-020/baseline_f0.py
    symbols: [R_HIT, R_HIT_LOOSE, f0_baseline]
    reason: hit metric + F0 floor 비교.
  - module: analysis/plan-a-001/kalman.py
    symbols: [kalman_predict, DT, T_PRED, SIGMA_OBS_MAIN, SIGMA_PROC_MAIN]
    reason: CV/CA 필터 본체. 본 plan 은 **수정 없이 import** + 내부 노출 변형(kalman_with_internals)을 plan-a-002 신규 모듈에 별도 작성 (plan-a-001 repro 불변 보존).
  - module: analysis/plan-a-001/yaw.py
    symbols: [yaw_angle, rotate_xy, inverse_rotate_xy]
    reason: yaw 좌표계 회전 (KR002 lever). 신규 feature 의 벡터 채널도 동일 frame 으로 회전.
  - module: analysis/plan-a-001/features.py
    symbols: [build_seq_t3, build_scalar_40d, compute_noise, normalize_seq]
    reason: KR002 입력 파이프라인. 신규 채널/스칼라를 여기에 concat 확장.
  - module: analysis/plan-a-001/model.py
    symbols: [GRUModelMultiAux]
    reason: GRU+F/W multi-aux. n_channels/scal_dim 만 신규 입력 dim 으로 확장 (구조 동일).
  - module: analysis/plan-a-001/losses.py
    symbols: [loss_combo, loss_aux_euclid]
    reason: combo(euclid+0.3 softhit) + aux. 변경 없음.
  - module: analysis/plan-a-001/run_oof.py
    symbols: [main]
    reason: 5-fold OOF runner. flag(--innov/--filtered-v/--cv-ca/--filtered-yaw) 추가 확장.
exp_ids:
  - KR003_kalman-derived-feats
  - KR004_filtered-yaw-frame
---

# plan-a-002 — Kalman 부산물 입력 feature (innovation · filtered velocity · CV/CA 불일치)

## §0. 한 줄 목적

> **plan-a-001 KR002 (Kalman CV 잔차 GRU + 입력 yaw 회전, 프로젝트 LB 신기록 0.6818) 위에, Kalman 필터가 *이미 계산하고 버리는* 부산물 3종 — (a) per-step innovation (등속예측 대비 측정 surprise = maneuver/모델불일치), (b) filtered velocity (denoise 속도), (c) CV vs CA 외삽 불일치 (가속 모델링이 forecast 를 얼마나 바꾸나) — 을 입력 feature 로 회수**해 잔차 GRU 에 *baseline 실패·maneuver 신호*를 명시 주입한다. Kalman gain/covariance 는 선형 시불변 필터에서 측정 독립 → **전 샘플 동일 (per-sample 정보량 0)** 이므로 의도적 제외 (제외 자체가 정보이론적 결론). 모든 신규 feature 는 *관측창* 산출 → leakage 없음. **plan-a-001 CV-LB 괴리 발견 (입력 yaw OOF neutral·LB+0.006) 을 전제로**, OOF 는 *no-regression sanity* 로만 쓰고 진짜 verdict 는 LB (사용자 confirm gated) 로 둔다.

---

## §0.5 Quick Reference (autonomous loop 매 turn 읽는 section)

| 항목 | 값 |
|---|---|
| paradigm | Kalman CV 잔차 회귀 (plan-a-001 carry — anchor/selector 아님) |
| baseline exp | **KR002** (input-yaw on, LB 0.6818, OOF 0.6663). 본 plan 모든 exp 의 비교 기준 |
| data | `load_all_samples` X (N,11,3), y (N,3). horizon +80ms (2 step × DT=40ms) |
| 신규 feature (a) innovation | per-step `innov(t)=z(t)−H·x̂(t|t−1)` (관측 t=1..10), (N,11,3) → **yaw 회전 후 GRU seq 채널 +3** (t=0 zero-pad) |
| 신규 feature (b) filtered v | per-step filtered velocity `state[:,1]` (N,11,3) → **yaw 회전 후 GRU seq 채널 +3** |
| 신규 feature (c) CV/CA 불일치 | `kalman('CA')−kalman('CV')` at +80ms (N,3) → **yaw 회전 3D + norm 1D = scalar +4** |
| 입력 dim 변화 | seq 9→**15** (KR003), scalar 40→**44**. (KR004 도 동일 dim, θ source 만 변경) |
| KR003 변경 | KR002 + (a)+(b)+(c) 동시 (frame θ = raw v_last yaw, KR002 그대로) |
| KR004 변경 | KR003 + **θ = yaw_angle(filtered v_last)** (frame 안정화 — denoise 속도로 heading 산출). LB lever(input-yaw) 증폭 가설 |
| 제외 (정보 0) | **Kalman gain K / covariance P** — 선형 TI 필터에서 측정 독립 = 전 샘플 동일 상수 |
| model/loss/ensemble | GRUModelMultiAux + combo(euclid+0.3 softhit) + aux F/W λ0.3. 2cfg(A: lr5e-4·do0.3 / B: lr1e-3·do0.1, plan-a-001 KR002 carry)×5fold stable_fold_id×3seed×200ep. **전부 KR002 carry** |
| metric | OOF hit_1cm (world Euclid<0.01m) uncalibrated headline + hit_1p5cm. paired permutation 10k vs KR002 |
| compare | KR002 OOF 0.6663 / LB 0.6818 · F0 0.6320 · Kalman-alone 0.5964 |
| 합격 기준 | **G_kalman (KR003)**: OOF hit_1cm ≥ **0.6653** (KR002 −0.001, no-regression PASS) / ≥+0.002 & p<0.05 = positive / <0.6643 = FAIL_regression(정보, halt X). **G_frame (KR004)**: Δ vs KR003 보고 (band: positive/neutral/negative). LB 는 §6 사용자 gated |

### Commit chain (예정)

| commit | spec | status |
|---|---|---|
| c0 spec | §0~§7 (본 파일) | [DONE] (plan-review-master BLOCKER 5→0) |
| c1 kalman internals | §4.1 `analysis/plan-a-002/kalman_features.py` — `kalman_with_internals(X,model)` → (pred, innov_seq (N,T,3), filtered_v (N,T,3)); `cv_ca_disagreement(X)` → (N,3) | [DONE] (smoke green, leakage-safe) |
| c2 feature 확장 | §4.2 `features_ext.py` — KR002 seq/scalar 위에 innov(+3)·filtered_v(+3) seq 채널, cv/ca(+4) scalar. yaw 회전 적용 (flag 별) | [DONE] (seq 9/12/15·scalar 44) |
| c3 runner flag | §4.3 `run_oof.py` (plan-a-002 자체 orchestrator) — `--innov --filtered-v --cv-ca --filtered-yaw` flag, n_channels 동적. | [DONE] |
| c4 smoke | §5 `tests/test_plan_a002_smoke.py` — import + 1f1s1e finite + leakage assert | [DONE] (7 passed) |
| c5 G1 | §5 KR003 1-fold 1-seed full-ep — finite & ≥ KR002 1-fold (−tol) | [DONE] (KR003 0.6762 vs KR002base 0.6767, Δ−0.0005 PASS) |
| c6 KR003 full | §5 2cfg×5fold×3seed OOF → `results_kr003.json/.npz` | [DONE] (0.6667, no_regression_PASS) |
| c7 KR004 full | §5 동일 budget + `--filtered-yaw` → `results_kr004.json/.npz` | [DONE] (0.6652, neutral) |
| c8 results + merge | §5 `plan-a-002-...results.md` + §0.5 sync + lane-a worktree→main merge | [DONE] |

### G-gates

- G0: c1~c4 인프라 + smoke green + leakage assert  **[DONE]** (7 pass, t_pred-invariant)
- G1: KR003 1-fold 1-seed hit_1cm finite & ≥ KR002 1-fold − 0.005 (신규 채널이 학습 안정성 안 깨뜨림 sanity)  **[DONE]** (0.6762 ≥ 0.6717)
- G_kalman (G2): KR003 full OOF band 판정 (vs KR002 0.6663 + paired permutation)  **[DONE]** (0.6667, Δ+0.0004 p=0.83 → no_regression_PASS)
- G_frame (G3): KR004 full OOF Δ vs KR003 + paired permutation  **[DONE]** (0.6652, Δ−0.0015 p=0.20 → neutral)
- G_final: 양 exp results 박제 + §0.5 sync + main merge  **[DONE]**

### Plan-specific 주의 (CV-LB 괴리)

- plan-a-001 에서 입력 yaw 회전은 OOF neutral(+0.0024 ns)인데 LB +0.0060 였다. → **OOF Δ<threshold 라도 FAIL 아님**; G_kalman 은 *no-regression* 만 hard 요구하고, neutral/positive 모두 LB 후보 자격. **OOF 만으로 feature 폐기 금지** (CV-LB 괴리 박제).
- 신규 feature 는 전부 **저차원(seq+6채널, scalar+4) + 물리기반** — CV-LB 괴리 환경에서 overfit/발산 위험 최소화 원칙 (고차원 flatten 회피).

---

## §1. 배경

plan-a-001 결과 (results.md):
- **KR001 (노트북 재현)**: OOF hit_1cm 0.6639 (EXCELLENT, F0 +0.0319), LB 0.6758.
- **KR002 (입력 yaw 회전)**: OOF 0.6663 (KR001 +0.0024, p=0.32 **neutral**), **LB 0.6818 (프로젝트 신기록, KR001 +0.0060)**.
- 핵심 발견 = **CV-LB 괴리**: 입력 yaw 회전이 OOF 에선 통계적 0 인데 test LB 에선 명확한 양 lift. train 5-fold OOF 가 입력 회전의 일반화 이득을 과소평가.

**미사용 자원**: KR002 의 Kalman CV 필터는 `kalman_predict` 가 +80ms 외삽값 (N,3) *만* 반환하고, 그 과정에서 계산한 풍부한 내부 신호를 버린다:
1. **innovation** `innov(t)=z(t)−H·x̂(t|t−1)` — 매 관측 스텝의 "등속예측 대비 측정 surprise". 큰 innovation = 모기가 등속에서 벗어남(maneuver). CV 필터가 *가장 틀리는 순간*을 직접 가리킴 → 모델이 예측할 미래 잔차의 선행지표.
2. **filtered velocity** `state[:,1]` — noise 제거된 속도. (i) feature 로, (ii) yaw 각도 산출원으로 (raw v_last 는 noisy → frame 흔들림).
3. **CV vs CA 불일치** — 등속(CV)·등가속(CA) 두 필터 외삽의 차 = "이 샘플에 가속 모델링이 얼마나 중요한가" = 곡률/maneuver 강도. 노트북 aux W (multi-σ) 가 model-disagreement 의 일종을 이미 맛봄.

**제외 — gain/covariance (정보이론적 결론)**: 선형 시불변 Kalman 에서 covariance P 와 gain K 는 `P=FPFᵀ+Q`, `K=P[:,0]/S` 로 **측정값 z 에 의존하지 않는다** → P_t·K_t 가 모든 샘플에서 t 만의 결정함수 = **전 샘플 완전 동일**. feature 로 넣으면 상수 컬럼 (per-sample 정보량 0). 적응형 σ (sample 별 noise 연동) 로 바꾸면 정보가 생기나 그건 별 필터 설계 → 본 plan out-of-scope.

**전략 (CV-LB 괴리 대응)**: 신규 feature 는 전부 저차원·물리기반 (maneuver/noise/baseline-실패 신호). OOF 는 sanity(no-regression), 진짜 검증은 LB. plan-003 → plan-a-001 의 교훈 (baseline 품질·metric 정렬·출력 clamp 가 부호를 뒤집음) 위에, "baseline 의 자기진단 신호"를 더하는 단계.

## §2. 가설

- **H1 (innovation)**: per-step innovation 시퀀스는 CV 필터가 깨지는 maneuver 를 직접 표지 → 미래 잔차 예측 개선. KR003 OOF ≥ KR002 (최소 neutral), LB 양 가능.
- **H2 (filtered v + CV/CA)**: denoise 속도 채널 + CV/CA 불가속 불일치가 가속·곡률 샘플 식별을 보강 → 보조 lift.
- **H3 (filtered-yaw frame, KR004)**: raw v_last 대신 *filtered* v_last 로 yaw 각도를 산출하면 frame heading 이 안정화 → KR002 의 LB lever(입력 yaw 회전)를 *증폭*. OOF 는 neutral 일 수 있으나 (CV-LB 괴리), LB 에서 양 가설.
- **메타 (gain 제외)**: gain/covariance 를 feature 에서 빼는 것은 휴리스틱이 아니라 선형 TI 필터의 측정독립성에서 따라오는 *정당한* 제외 (넣어도 정보 0).

## §3. 실험 목록

### KR003_kalman-derived-feats
- **type**: feature 추가 (vs KR002 단일 패러다임 carry)
- **baseline**: KR002 (input-yaw on)
- **변경 변수**: 입력에 (a) innovation seq +3채널, (b) filtered velocity seq +3채널, (c) CV/CA 불일치 scalar +4 동시 추가. (a)(b)(c) 벡터분은 KR002 와 동일 yaw frame (θ=raw v_last) 으로 회전. model/loss/ensemble/calibration·frame θ source 는 전부 KR002 동일. **(a)(b)(c) 동시 투입 = 의도적 설계** — §0.5 의 "저차원·물리기반 Kalman 부산물 묶음" 단일 가설 cluster (CV-LB 괴리 환경 overfit 회피 위해 개별 분해보다 일괄 검증 우선). per-channel attribution 은 §실패분기·deferred 진단(아래)으로 분리.
- **config/경로**: `run_oof.py --input-yaw --innov --filtered-v --cv-ca`
- **기대 runtime**: KR002 ≈ 700s (GPU L40S) + 입력 dim 소폭 증가. CPU 시 seed 3→1 자동감소(decision-note carry).
- **성공 기준**: OOF hit_1cm ≥ 0.6653 (KR002 −0.001 no-regression PASS). finite, NaN/Inf 0, leakage assert green.
- **실패 분기**: < 0.6643 (clear regression) → leave-one-out 1-fold 진단 (innov/filtered-v/cv-ca 각각 제거)로 회귀 유발 채널 식별 → 해당 채널 drop. severe 아님 (정보).
- **attribution (deferred/optional)**: G_kalman 이 movement 보이면 leave-one-out 3종 1-fold 진단. **단 CV-LB 괴리로 OOF attribution 은 약한 신호** 명시 — 1-fold 진단은 방향 힌트일 뿐.

### KR004_filtered-yaw-frame
- **type**: frame source 단일 변경 (vs KR003)
- **baseline**: KR003
- **변경 변수**: yaw 각도 θ 를 `yaw_angle(raw v_last)` → **`yaw_angle(filtered v_last)`** (Kalman 마지막 스텝 filtered velocity). 타깃·입력 회전 모두 이 θ 사용. 그 외 KR003 동일. **단일변수 전제**: θ source 1곳만 교체 (frame 의 정의만 바뀌고 feature 집합 불변).
- **config/경로**: `run_oof.py --input-yaw --innov --filtered-v --cv-ca --filtered-yaw`
- **기대 runtime**: KR003 동일.
- **성공 기준**: Δ = KR004 − KR003 OOF hit_1cm 보고. positive Δ≥+0.002 & p<0.05 / neutral / negative. (CV-LB 괴리로 neutral 도 LB 후보)
- **실패 분기**: Δ ≤ −0.002 → filtered θ 가 raw heading 신호를 깎음 (informative 음 band). frame 은 KR003(raw θ) 로 환원.

## §4. 서버 작업 순서 (모듈 spec)

### §4.1 kalman_features.py (c1)
- `kalman_with_internals(X, model='CV', dt=DT, t_pred=T_PRED, sigma_obs=SIGMA_OBS_MAIN, sigma_proc=SIGMA_PROC_MAIN)` → `(pred (N,3), innov_seq (N,T,3), filtered_v (N,T,3))`. (default = plan-a-001 import 상수 — frontmatter code_reuse.) 초기 공분산 `P0` 는 인자 아님 — plan-a-001 `kalman_predict` 내부와 **동일 초기화 박제** (CV pred 재현 일관).
  - plan-a-001 `kalman.py` 의 행렬 setup (F/Q/F_pred) **재사용** (**import 만** — 코드 박제 금지, plan-a-001 repro 불변 보존; helper 가 private 이면 plan-a-001 에 export 1줄 추가 후 import). loop 는 축별 predict→update 1 cycle/step: predict `x̂(t|t−1)=F·x̂(t−1|t−1)` 후 `innov(t)=z(t)−H·x̂(t|t−1)` (z(t)=관측위치 `X[:,t,j]`, H=위치 관측행렬, **update 전** 예측위치 대비 잔차), 이어 update: `S=H·P(t|t−1)Hᵀ+R` (R=σ_obs²), gain `K(t)=P(t|t−1)Hᵀ/S(t)`, `x̂(t|t)=x̂(t|t−1)+K(t)·innov(t)`, posterior `P(t|t)=(I−K(t)H)P(t|t−1)`, 다음 step `P(t+1|t)=F·P(t|t)·Fᵀ+Q`. **선형 TI → K(t)·P(t) 측정 독립 = 전 샘플 동일 → loop 1회 precompute 후 재사용** (§1 gain-제외 논거의 구현 귀결; innov·state 만 per-sample). 축별 `innov` 를 `innov_seq[:,t,j]` 에 저장, update 후 `state[:,1]` (CV velocity) 를 `filtered_v[:,t,j]` 에 저장.
  - **t=0**: innov 미정의(첫 update 전) → `innov_seq[:,0,:]=0`. filtered_v[:,0,:] = 0 (**plan 의 t=0 pad 규약** — t=0 은 첫 update 전이라 정의상 pad, plan-a-001 내부 init 과 무관; innov t=0 zero-pad 와 일관, leakage 무관).
  - **leakage 안전**: loop 는 관측 t=1..T-1 만 (T=11 → 관측 10 step; t=0 zero-pad 로 innov_seq/filtered_v = (N,11,3)). +t_pred 외삽 스텝은 pred 에만 반영, internals 에 미포함.
- `cv_ca_disagreement(X, dt=DT, t_pred=T_PRED, sigma_obs=SIGMA_OBS_MAIN, sigma_proc=SIGMA_PROC_MAIN)` → `(N,3)`: +80ms 외삽 차 `kalman_ca_predict(X) − kalman_predict(X,'CV')` (`kalman_ca_predict` = `kalman_features.py` 내 신규 helper, 아래 CA 산출식 closed-form; `cv_ca_disagreement` 가 호출). **CV 항** = plan-a-001 `kalman_predict` (model='CV', 검증분) 그대로. **CA 항** = **실행 시 확인: plan-a-001 `kalman_predict` 가 `model='CA'` 를 이미 완전 지원** (kalman.py L41-54 의 F_CA·Q_CA branch) → `kalman_predict(X,'CA')` **직접 재사용** (신규 구현 불필요; 아래 CA 산출식 = kalman.py CA branch 문서화). σ_MAIN·t_pred 동일. plan-a-001 repro 불변 (kalman.py 미수정). decision-note: review-phase 는 source-blind 라 plan-a-001 *plan 텍스트*(CV만 서술) 기준 "신규 구현"으로 spec 했으나, 실코드에 CA 존재 → 재사용으로 정정.
  - **CA 산출식** (축별 1D, step dt=DT): 등가속 운동학 `pos+=v·dt+½·a·dt²`, `v+=a·dt`, `a` 일정 → 전이 `F_CA=[[1,dt,½dt²],[0,1,dt],[0,0,1]]`, 관측 `H=[1,0,0]` (위치만, 관측 노이즈 `R=σ_obs²`=SIGMA_OBS_MAIN — CV 와 동일), 프로세스 노이즈 `Q=σ_proc²·diag(0,0,1)` (acc 채널 구동, σ_proc=SIGMA_PROC_MAIN — plan-a-001 σ carry; CV(vel-driven)↔CA(acc-driven) 차원 상이하나 cv_ca 는 *상대* disagreement 신호라 절대 스케일 민감도 낮음, 재튜닝 §6 out-of-scope), 초기 `P0=diag(σ_obs², P_big, P_big)` (acc 초기분산 diffuse, P_big=1e3), `a` 초기값=0. `x̂_last`=[pos,vel,acc]_last 는 위 `kalman_with_internals` 와 **동형 predict→update 점화** (F_CA·H·Q·P0, 관측 10 step) 로 산출 — 위치만 관측(H=[1,0,0])이라 vel·acc 가 gain 점화로 채워짐 (CV 와 동일 update 식, state 3D). +80ms 외삽 = `H·F_CA(τ=t_pred)·x̂_last` = `pos_last + v_last·t_pred + ½·a_last·t_pred²` → `(N,3)`.
- backward-compat: plan-a-001 `kalman.py` 는 **수정 X** (KR001/KR002 repro 보존). 본 모듈은 별도.

### §4.2 features_ext.py (c2)
- `build_seq_ext(X, *, innov, filtered_v, theta, input_yaw)` → (N,11,C). KR002 `build_seq_t3` 결과(9채널 = [rel(3), v(3), a(3)], KR002 carry; rotate_xy 는 각 triplet 의 xy 만 회전·**z 보존**)에:
  - `--innov` 시 innov_seq(yaw 회전, +3) concat.
  - `--filtered-v` 시 filtered_v(yaw 회전, +3) concat.
  - rel/v/a 의 input_yaw 회전은 KR002 로직 그대로. 신규 벡터 채널도 **동일 θ 로 rotate_xy** (frame 일관).
- `build_scalar_ext(...)` → KR002 scalar_40d 에 `--cv-ca` 시 cv_ca 불일치 3D 차벡터 `rotate_xy(θ)` 3D + L2 norm 1D = +4 concat. **norm = ‖CA−CV‖₂ (x,y,z 3D 전체; rotate_xy 는 L2 불변 → 회전 전/후 동일값, 회전 순서 무관).**
- **채널 순서 고정 규약**: seq = `[KR002 9채널 | innov 3 | filtered_v 3]`, scalar = `[KR002 40 | cv_ca 회전3 + norm1]` (flag off 시 해당 블록만 생략, 나머지 순서 불변). scaler·model n_channels 자동추론·audit 표가 이 고정 순서 공유 → flag 조합별 index mismatch 회피.
- 채널/스칼라 명 → rotate|invariant 분류표를 `results_kr00X.json` 에 박제 (frame 일관 사후 audit).

### §4.3 run_oof.py 확장 (c3)
- 신규 flag `--innov --filtered-v --cv-ca --filtered-yaw`. 기존 `--input-yaw` carry.
- `--filtered-yaw` 시 θ = `yaw_angle(filtered_v[:, -1, :])` (kalman_with_internals 산출), 아니면 KR002 처럼 raw v_last.
- per-fold StandardScaler 는 확장된 채널/스칼라 dim 에 자동 적용. inverse_rotate_xy 로 world 복원 (θ source 와 일관).
- model n_channels/scal_dim 은 입력 dim 에서 자동 추론 (GRUModelMultiAux 구조 불변).

## §5. 합격 기준 + Gate

| gate | 검사 | PASS band | severity |
|---|---|---|---|
| G0 | import + smoke (1f1s1e finite) + **leakage assert** (구체형: innov_seq·filtered_v 는 관측창 `X[:, :11]` 만의 함수 → `t_pred` 를 임의값으로 바꿔도 둘이 불변임을 `assert allclose` [외삽 step 은 pred 에만 반영, internals 미포함]; 미래 +80ms 관측은 X 에 부재 → 구조적 무참조) | green | severe halt if import/NaN/leakage |
| G1 | KR003 1-fold 1-seed full-ep hit_1cm | ≥ **KR002 동일 fold0/seed0/cfgA 1-fold** − 0.005 (G1 에서 KR002 baseline 을 동일 fold0·seed0·cfgA 1회 재실행해 임계 산출 — plan 본문 자족; 학습 안정 sanity) | warn |
| G_kalman | KR003 full OOF hit_1cm | **≥0.6653 no-regression PASS** / ≥+0.002&p<0.05 positive / <0.6643 FAIL_regression(정보) | <0.6643 = 정보, halt X |
| G_frame | KR004 OOF Δ vs KR003 | ≥+0.002&p<0.05 positive / \|Δ\|<0.002 neutral / ≤−0.002 negative | warn |
| G_final | 양 exp results 박제 + §0.5 sync + main merge | 완료 | — |

- statistic: paired permutation 10000 resample (KR003 vs KR002, KR004 vs KR003), p<0.05.
- artifact: `analysis/plan-a-002/results_kr003.json/.npz`, `results_kr004.json/.npz`, `plan-a-002-...results.md`.
- NaN/Inf/divergence 0 의무. cuda OOM 시 batch 256→128→64 자동 감소.
- **CV-LB 괴리 박제 의무**: results 에 OOF Δ 와 함께 "LB 검증 필요(사용자 gated)" 명시. OOF neutral 을 negative 결론으로 박제 금지.

## §6. Out of scope

- **Kalman gain/covariance feature** — 측정 독립(전 샘플 동일) 정보량 0. 제외 자체가 §1 결론.
- per-step CV/CA 불일치 시퀀스 (관측창 전체) — 본 plan 은 +80ms 외삽 불일치 scalar 만. 시퀀스화는 후속.
- 적응형 σ Kalman (sample 별 noise 연동) — gain 에 정보 부여하나 별 필터 설계.
- Kalman σ 대규모 재튜닝 (plan-a-001 MAIN σ carry).
- anchor/selector paradigm ensemble.
- **autonomous DACON LB 제출** — quota 사용자 명시 confirm 필요 (특히 OOF Δ<threshold 시 더더욱). 본 plan headline = OOF. **단 CV-LB 괴리 때문에 best OOF config(KR003 또는 KR004)의 LB 제출을 *사용자 confirm 후* 강력 권장** (입력 yaw 처럼 OOF-neutral·LB-positive 가능성). 제출 = 별 turn, 사용자 승인 gated.

## §7. 참조

- `plans/plan-a-001-kalman-residual-gru-repro.results.md` — KR001/KR002 OOF·LB, CV-LB 괴리.
- `analysis/plan-a-001/kalman.py` — CV/CA 필터 본체 (innov/filtered/CA 산출 지점).
- `analysis/plan-a-001/{yaw,features,model,losses,run_oof}.py` — KR002 파이프라인.
- `analysis/plan-020/baseline_f0.py` — F0 floor + hit metric.
- `WORKFLOW.md §4` — lane mutex + worktree→main merge (lane a 2번째 plan).

decision-note: spec-default — plan-a-002 = lane a 2번째 plan, baseline=KR002(LB record). exp prefix=KR carry (KR003/KR004). Kalman gain/covariance 제외 = 측정독립 정보0 (정보이론). 신규 feature 전부 관측창 산출(leakage 무) + 저차원 물리기반(CV-LB 괴리 대응). LB 제출 = 사용자 confirm gated (out-of-scope-by-default, but CV-LB 괴리로 권장). KR004 = θ source 만 filtered-v 로 단일 변경.
