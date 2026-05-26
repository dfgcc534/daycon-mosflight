---
plan_id: a-003
version: 1
date: 2026-05-26 (Asia/Seoul)
status: all_complete
lane: a
inspired_by:
  - a-002 (KR003 Kalman 부산물 feature = LB 0.6854 현 record. 본 plan baseline. OOF-neutral·LB-positive = CV-LB 괴리 2번째 사례)
  - a-001 (KR002 입력 yaw 회전 = 회전 불변성→sample efficiency, OOF-neutral·LB +0.0060. 본 plan augmentation 의 메커니즘 선행 evidence)
  - 021 (Frenet 입력 회전 input-augment positive band, "invariance→sample efficiency↑" — 대칭 활용 lever 의 계보)
  - notes/fe_axis_24_25_26_27_29.md (Augmentation 축 정의 — 반사/노이즈)
code_reuse:
  - module: analysis/plan-a-002/run_oof.py
    symbols: [main, train_one, run_config]
    reason: KR003 OOF+test 파이프라인. train_one 에 augmentation 만 in-loop 추가 (flag default off → KR003/KR004 repro 불변). 그 외 전부 carry.
  - module: analysis/plan-a-002/features_ext.py
    symbols: [build_seq_ext, build_scalar_ext]
    reason: 15ch seq / 44 scalar + 채널명(반사 대상 `_y`/`cvca_y` 식별). 변경 없음.
  - module: analysis/plan-a-002/kalman_features.py
    symbols: [kalman_with_internals, cv_ca_disagreement]
    reason: 부산물 feature. 변경 없음.
  - module: analysis/plan-a-001/{kalman,yaw,model,losses}.py
    symbols: [kalman_predict, rotate_xy, inverse_rotate_xy, yaw_from_last_step, GRUModelMultiAux, loss_combo, loss_aux_euclid]
    reason: paradigm 본체. 변경 없음.
exp_ids:
  - KR008_reflect-noise-aug
  - KR009_reflect-only
---

# plan-a-003 — 반사 + 노이즈 augmentation (KR003 위 대칭/정칙화 lever)

## §0. 한 줄 목적

> **KR003 (Kalman-잔차 GRU + 입력 yaw 회전 + Kalman 부산물 feature, LB 0.6854 현 record) 위에, 입력 yaw 회전이 "회전 불변성→sample efficiency" 로 LB +0.0060 을 낸 것과 *동일 메커니즘* 으로, train-time augmentation 2종 — (a) 반사(yaw frame y→−y, 좌↔우회전 대칭, yaw 정규화가 못 잡는 새 DOF), (b) 측정노이즈 jitter(robust 학습) — 을 추가**해 train/test 일반화 갭(CV-LB 괴리)을 직접 공략한다. **고차 물리 모델(CTRV/고차 Kalman)은 GRU 가 raw seq 로 이미 학습하므로 명시적 out-of-scope** — 남은 lever 는 동역학이 아니라 대칭/정칙화. OOF 는 sanity(no-regression), 진짜 verdict 는 LB (사용자 confirm gated).

---

## §0.5 Quick Reference (autonomous loop 매 turn 읽는 section)

| 항목 | 값 |
|---|---|
| paradigm | Kalman CV 잔차 회귀 (plan-a-002 carry — anchor/selector 아님) |
| baseline exp | **KR003** (input-yaw+innov+filtered_v+cv_ca, LB 0.6854, OOF 0.6667). 본 plan 비교 기준 |
| lever | **train-time augmentation** (feature 아님): 반사 + 노이즈. model/loss/feature/ensemble 전부 KR003 동일 |
| (a) 반사 augment | yaw frame 에서 `_y` 채널(rel/v/a/innov/fv_y, idx 1,4,7,10,13) + scalar cvca_y(idx 41) + 타깃 tgt_main/F/W 의 y 부호 반전. per-sample p=0.5 online (train-only). x/z·magnitude·cos 불변 |
| (b) 노이즈 augment | 표준화 seq 입력에 `N(0, σ_aug)` 가산 (per-batch, train-only). σ_aug=0.10 (표준화 단위, 고정 — grid out-of-scope) |
| KR008 변경 | KR003 + (a)+(b) 동시 (묶음, 주 실험) |
| KR009 변경 (optional/deferred) | KR003 + (a) 반사 단독 (KR008 positive 시 attribution용; quota·noise-limited 라 deferred) |
| 불변 | val/test = 원본(aug train-only). aug off = KR003 bit-identical repro 의무 |
| out-of-scope | 고차 물리 모델(CTRV/고차 Kalman/multi-model), σ_aug grid, 반사확률 grid, calibration |
| metric | OOF hit_1cm (uncalibrated) + hit_1p5cm. paired permutation 10k vs KR003 |
| compare | KR003 OOF 0.6667 / **LB 0.6854** · KR002 LB 0.6818 · F0 0.6320 · Kalman-alone 0.5964 |
| 합격 기준 | **G_aug (KR008)**: OOF hit_1cm ≥ **0.6657** (KR003 −0.001 no-regression PASS) / <0.6647 = FAIL_regression(정보). **G_lb**: KR008 LB vs KR003 0.6854 (사용자 gated) = 진짜 verdict |

### Commit chain (예정)

| commit | spec | status |
|---|---|---|
| c0 spec | §0~§7 (본 파일) | [DONE] |
| c1 augmentation | §4.1 `run_oof.py` train_one 확장 — `--reflect-aug --noise-aug`, 반사(채널명 `_y` 자동 식별)·노이즈 in-loop. default off | [DONE] (aug-off bit-identical) |
| c2 smoke | §5 `tests/test_plan_a003_smoke.py` — 반사 항등성(2회=원본)·aug off==KR003 bit-identical·1f1s1e finite | [DONE] (4 pass; off=0.6637) |
| c3 G1 | §5 KR008 1-fold 1-seed full-ep — finite & ≥ KR003 1-fold − 0.005 | [DONE] (0.6757 vs 0.6762, Δ−0.0005 PASS) |
| c4 KR008 full + submission | §5 2cfg×5fold×3seed OOF + `--predict-test` → `results_kr008.json/.npz` + `submission_kr008.csv` | [DONE] (0.6671 no_regression_PASS) |
| c5 results + (LB 제출 user-gated) + merge | §5 `plan-a-003-...results.md` + §0.5 sync + lane-a worktree→main merge | [DONE] |

### G-gates

- G0: c1~c2 인프라 + smoke green + 반사 항등성 + aug-off repro 불변  **[DONE]** (pytest 4 pass, aug-off 0.6637 bit-identical, aug-on finite)
- G1: KR008 1-fold 1-seed hit_1cm finite & ≥ KR003 1-fold − 0.005 (aug 가 학습 안정성 안 깨뜨림 sanity)  **[DONE]** (0.6757 ≥ 0.6712)
- G_aug (G2): KR008 full OOF band 판정 (vs KR003 0.6667 + paired permutation). no-regression hard 요구, neutral/positive 모두 LB 후보  **[DONE]** (0.6671, Δ+0.0004 p=0.87 → no_regression_PASS)
- G_lb (G3): KR008 LB 제출 (사용자 confirm gated) vs KR003 0.6854 — **진짜 verdict**  **[PENDING]** (submission_kr008.csv 준비, 사용자 confirm 대기)
- G_final: results 박제 + §0.5 sync + main merge  **[DONE]**

### Plan-specific 주의 (CV-LB 괴리)

- 입력 yaw·Kalman 부산물 둘 다 OOF-neutral·LB-positive. → **augmentation 도 OOF Δ<threshold 라도 FAIL 아님**; G_aug 는 no-regression 만 hard, verdict 는 LB. **OOF 만으로 augmentation 폐기 금지**.
- 우린 LB 노이즈 floor(±0.002~0.003, n_test=10000) 근처 + 수확 체감 구간(lever 효과 +0.006→+0.0036). KR008 LB Δ 가 noise floor 내면 inconclusive 박제 (강제 positive 주장 금지).

---

## §1. 배경

plan-a-002 결과: KR003 (Kalman 부산물 feature) OOF 0.6667, **LB 0.6854 = 프로젝트 record** (KR002 0.6818 +0.0036). 핵심 = **CV-LB 괴리 2번째 사례** — OOF Δ+0.0004(ns)인데 LB +0.0036. plan-a-001 입력 yaw(OOF +0.0024 ns·LB +0.0060)와 동일 패턴.

**관찰**: 지금까지 LB 를 올린 lever 는 *물리 모델 정교화*가 아니라 **대칭/불변성 활용** 이었다.
- 입력 yaw 회전 = 회전 불변성을 좌표 정규화로 부여 → sample efficiency↑ → LB +0.0060 (최대 lever).
- Kalman 부산물 = baseline 자기진단 신호 주입 → LB +0.0036.
- 고차 물리 모델(CA, cv_ca)은 +0.0036 에 그침 — GRU 가 raw rel/v/a 로 비선형 동역학을 이미 학습하므로 CTRV/고차 Kalman 은 *중복 모델링* (무의미, 사용자 확인).

**미활용 대칭 — 반사**: yaw frame 정규화는 *회전* DOF 만 제거한다. 모기 비행은 heading 축 기준 좌우(좌회전↔우회전) chirality 가 없으므로 **y→−y 반사도 유효 대칭** 인데 아직 안 쓴다 (회전 ≠ 반사, 중복 아님). 반사된 (입력, 타깃) 쌍은 동등한 유효 샘플 → online 반사 augmentation 으로 유효 데이터를 대칭 orbit 으로 2배 → test 일반화↑. 이는 입력 yaw 가 +0.0060 을 낸 *바로 그 메커니즘* (invariance→sample efficiency)의 직계 후속.

**미활용 정칙화 — 노이즈 jitter**: 관측은 측정 noise 를 포함(noise_poly2/savgol 추정). train 입력에 소량 noise 를 주입하면 noise robust 학습 → train 분포 과적합 완화 → test 일반화↑ (CV-LB 괴리 직격). 표준 denoising-style 정칙화.

## §2. 가설

- **H1 (반사)**: heading 축 반사 대칭을 online augmentation 으로 부여하면 sample efficiency↑ → test 일반화 이득. OOF 는 neutral 일 수 있으나(yaw 정규화로 이미 회전은 정규화됨, 반사는 추가 DOF) **LB 양** 가능. 입력 yaw 와 동일 메커니즘의 다음 수.
- **H2 (노이즈)**: 측정 noise jitter 가 robust 학습 → train/test gap 축소 → LB 양 보조.
- **메타 (고차 물리 무의미)**: GRU 가 raw seq 로 동역학을 이미 학습 → CTRV/고차 Kalman 은 중복. 남은 lever 는 대칭/정칙화 축 (본 plan).
- **메타 (수확 체감)**: lever 효과 +0.006→+0.0036 감소 중. KR008 LB Δ 는 noise floor(±0.002~0.003) 내일 수 있음 — 그 경우 inconclusive 박제.

## §3. 실험 목록

### KR008_reflect-noise-aug
- **type**: train-time augmentation 추가 (vs KR003 단일 패러다임 carry)
- **baseline**: KR003 (input-yaw+innov+filtered_v+cv_ca)
- **변경 변수**: train_one 학습 루프에 (a) 반사 p=0.5 online + (b) 노이즈 σ_aug=0.10 동시 추가. model/loss/feature/ensemble/frame θ 전부 KR003 동일. val/test 변환 없음.
- **config/경로**: `run_oof.py --innov --filtered-v --cv-ca --input-yaw --reflect-aug --noise-aug --predict-test --exp KR008 --compare-to results_kr003.npz`
- **기대 runtime**: KR003 ≈ 613s (GPU L40S) + aug in-loop 소폭. CPU 시 seed 3→1 자동감소.
- **성공 기준**: OOF hit_1cm ≥ 0.6657 (KR003 −0.001 no-regression PASS). finite, NaN/Inf 0, aug-off repro 불변 검증 green.
- **실패 분기**: < 0.6647 (clear regression) → 반사/노이즈 1-fold 개별 제거 진단. severe 아님(정보). **LB verdict 우선** — OOF regression 이어도 LB 는 §6 사용자 gated.

### KR009_reflect-only (optional, deferred)
- **type**: augmentation 분해 (vs KR008)
- **baseline**: KR003
- **변경 변수**: 반사 단독 (노이즈 off). σ_aug 경로 제거.
- **조건부 실행**: KR008 LB 가 KR003 record 갱신 시에만, 반사 단독 기여 attribution 용. **단 LB noise floor·quota 때문에 default deferred** (분해는 noise-limited — plan-a-002 에서 박제). 실행 시 별 turn·사용자 confirm.

## §4. 서버 작업 순서 (모듈 spec)

### §4.1 run_oof.py train_one 확장 (c1)
- 신규 flag `--reflect-aug --noise-aug` (둘 다 default off → KR003/KR004 bit-identical repro 보존). `train_one` 에 `reflect_aug: bool, noise_aug: float, reflect_idx_seq, reflect_idx_scal` 인자 추가, `run_config`·`main` 이 전달.
- **반사 대상 index 산출 (main)**: `build_seq_ext`/`build_scalar_ext` 가 반환한 `seq_names`/`scal_names` 에서 `name.endswith("_y")` 인 채널 index 수집 → `reflect_idx_seq` (= [1,4,7,10,13] for 15ch), `scal_names` 의 `"cvca_y"` index → `reflect_idx_scal` (= [41] for 44). 타깃 y = index 1 (고정). **채널명 자동 식별** (하드코딩 회피, flag 조합 변해도 robust).
- **반사 in-loop** (train batch 마다): `mask = (rand(batch) < 0.5)`; mask 행에 대해 `seq[mask][:,:,reflect_idx_seq] *= -1`, `scal[mask][:,reflect_idx_scal] *= -1`, `tgt_main[mask][:,1] *= -1`, `tgt_F[mask][:,1] *= -1`, `tgt_W[mask][:,1] *= -1`. (반사는 부호 반전이라 항등성: 2회 = 원본.)
- **노이즈 in-loop**: 표준화된 seq batch 에 `seq = seq + noise_aug * randn_like(seq)` (σ_aug=0.10, 표준화 단위). scal 은 미적용 (magnitude 라 noise 의미 약함 — seq 만). val/test batch 미적용.
- **순서**: 노이즈 → 반사 (또는 무관, 둘 다 부호/가산 독립). seed 고정 재현성 위해 aug RNG 는 torch generator (train seed 종속).
- backward-compat: `--reflect-aug`/`--noise-aug` 둘 다 off 면 train_one 은 plan-a-002 와 bit-identical (KR003 repro 검증 = G0).

### §4.2 반사 대상 채널 audit (c1)
- 반사 channel 분류표(seq `_y` idx, scal cvca_y idx, target y)를 `results_kr008.json` 에 박제 → 반사 대칭 일관성 사후 audit. magnitude/cos/norm 채널이 반사 대상에 잘못 포함되지 않았는지 확인 (반사 불변이어야 함).

### §4.3 smoke (c2)
- `tests/test_plan_a003_smoke.py`: (1) 반사 항등성 — 같은 batch 2회 반사 = 원본 (`allclose`). (2) **aug-off repro** — `--reflect-aug`/`--noise-aug` 없이 1f1s1e 출력이 plan-a-002 train_one 과 동일(seed 고정 bit-identical 또는 hit 동률). (3) aug-on 1f1s1e finite·NaN 0. (4) 반사 대상 index 가 `_y`/cvca_y 만 (magnitude 미포함).

## §5. 합격 기준 + Gate

| gate | 검사 | PASS band | severity |
|---|---|---|---|
| G0 | import + smoke + 반사 항등성 + aug-off KR003 repro 불변 | green | severe halt if import/NaN/repro 깨짐 |
| G1 | KR008 1-fold 1-seed full-ep hit_1cm | ≥ KR003 1-fold − 0.005 (학습 안정 sanity) | warn |
| G_aug | KR008 full OOF hit_1cm | **≥0.6657 no-regression PASS** / ≥+0.002&p<0.05 positive / <0.6647 FAIL_regression(정보) | <0.6647 = 정보, halt X |
| G_lb | KR008 LB vs KR003 0.6854 | 사용자 gated 제출. Δ≥+noise floor = 신기록 / floor 내 = inconclusive / Δ<0 = 음 | — (verdict) |
| G_final | results 박제 + §0.5 sync + main merge | 완료 | — |

- statistic: paired permutation 10000 resample (KR008 vs KR003), p<0.05.
- artifact: `analysis/plan-a-002/results_kr008.json/.npz` (run_oof 위치 carry), `submission_kr008.csv`, `plan-a-003-...results.md`.
- **NaN/Inf/divergence 0 의무**. aug-off repro 불변 의무 (KR003 회귀 방지).
- **CV-LB 괴리 박제 의무**: results 에 OOF Δ + LB(사용자 gated) 병기. OOF neutral 을 negative 결론으로 박제 금지. LB Δ 가 noise floor 내면 inconclusive 명시 (강제 positive 금지).

## §6. Out of scope

- **고차 물리 모델** (CTRV/등가속 이상 고차 Kalman/multi-model 융합) — GRU 가 raw seq 로 이미 학습, 중복 무의미 (사용자 명시). 본 plan 핵심 배제.
- σ_aug grid / 반사확률 grid (단일 변수 — 고정값. tuning 은 후속).
- calibration ((1,0.95,1)) — 별 lever.
- 노이즈 aug 를 raw X 에 적용(Kalman 재계산 per-epoch) — 비용 과다, 본 plan 은 표준화 seq 가산으로 근사.
- KR009 반사 단독 분해 — default deferred (LB noise·quota; KR008 record 갱신 시에만 사용자 confirm 후).
- **autonomous DACON LB 제출** — quota 사용자 명시 confirm 필요 ([[feedback-dacon-submit-confirmation]]). 본 plan headline = OOF, 단 CV-LB 괴리상 KR008 LB 제출 사용자 confirm 후 권장 (KR003 record 갱신 가능성).
- anchor/selector paradigm ensemble (별 방향, plan-a-002 §5 에서 noise-limited 박제).

## §7. 참조

- `plans/plan-a-002-kalman-derived-features.results.md` — KR003 OOF 0.6667·LB 0.6854, CV-LB 괴리 2번째.
- `plans/plan-a-001-kalman-residual-gru-repro.results.md` — KR002 입력 yaw OOF-neutral·LB +0.0060 (augmentation 메커니즘 선행).
- `analysis/plan-a-002/run_oof.py` — train_one/run_config/main (augmentation in-loop 추가 지점) + `--predict-test`.
- `analysis/plan-a-002/{features_ext,kalman_features}.py` — 15ch seq·44 scalar·채널명(`_y`/cvca_y 반사 식별).
- `notes/fe_axis_24_25_26_27_29.md` — Augmentation 축 정의 (반사/노이즈).
- `WORKFLOW.md §4` — lane mutex + worktree→main merge (lane a 3번째 plan).

decision-note: spec-default — plan-a-003 = lane a 3번째, baseline=KR003(LB record). exp prefix=KR carry (KR008/KR009). 고차 물리 모델 명시 배제(사용자). lever = 대칭(반사)+정칙화(노이즈) augmentation, train-only, aug-off repro 불변. 반사 = yaw 정규화 미포착 DOF (회전≠반사). verdict=LB. KR009 분해 deferred(noise-limited). σ_aug/p 고정(grid out-of-scope).
