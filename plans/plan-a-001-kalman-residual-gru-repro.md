---
plan_id: a-001
version: 1
date: 2026-05-26 (Asia/Seoul)
status: draft
lane: a
inspired_by:
  - notes/LB_0.6780 코드공유.ipynb (재현 대상 — Kalman 잔차 + GRU+F+W + yaw + calibration, LB 0.6780)
  - 003 (residual-GRU lean baseline — linear extrap 잔차 GRU 가 LB 0.5688 로 baseline 0.60 보다 *퇴보*. 본 plan 의 대조군: baseline 품질(칼만) + metric-aware loss(softhit) + 출력 clamp(tanh×2cm) 가 부호를 뒤집는지 검증)
  - 020 (F0 baseline 0.6320/0.8033 — 비교 floor)
  - 022 (anchor paradigm best OOF 0.6528 — 프로젝트 현 최고, STRONG band 기준)
  - 021 (Frenet 입력 회전 input-augment 가 positive band, "invariance→sample efficiency↑" — KR002 입력 yaw 회전의 선행 evidence, 단 4-lever 묶음이라 단독 기여 불명)
code_reuse:
  - module: src/io.py
    symbols: [load_all_samples, load_labels]
    reason: 프로젝트 데이터 loader. X (N,11,3), y (N,3).
  - module: src/pb_0_6822/selector.py
    symbols: [stable_fold_id]
    reason: MD5 5-fold split (plan-020~031 carry, OOF 비교 호환).
  - module: analysis/plan-020/baseline_f0.py
    symbols: [R_HIT, R_HIT_LOOSE, f0_baseline]
    reason: hit_1cm/1p5cm metric 정의 + F0 floor 비교.
exp_ids:
  - KR001_notebook-repro
  - KR002_input-yaw-rot
g_repro_oof_hit_1cm: null
g_yaw_delta: null
---

# plan-a-001 — Kalman-Residual GRU 노트북 재현 + 입력 yaw 회전 ablation

## §0. 한 줄 목적

> **`notes/LB_0.6780 코드공유.ipynb` (Kalman CV 잔차 → GRU+F+W multi-aux → yaw 좌표계 타깃 → per-axis calibration, LB 0.6780) 파이프라인을 프로젝트 데이터/5-fold 위에 *그대로 이식*해 OOF hit_1cm 을 박제(KR001)하고, 그 위에 *입력 seq 도 yaw 회전*하는 단일 변수 ablation(KR002)으로 입력 회전의 순수 기여를 측정**한다. plan-003 residual-GRU 가 LB 0.5688 로 baseline 보다 퇴보한 것과 대조 — "칼만 baseline + metric-aware loss + 출력 clamp" 조합이 잔차-GRU 의 부호를 양(+)으로 뒤집는지 확인.

---

## §0.5 Quick Reference (autonomous loop 매 turn 읽는 section)

| 항목 | 값 |
|---|---|
| paradigm | **Kalman CV 잔차 회귀** (anchor/selector paradigm 아님 — plan-003 계보의 강화판) |
| data | `load_all_samples` X (N,11,3), `load_labels` y (N,3). horizon +80ms (2 step × DT=40ms) |
| baseline (잔차 기준점) | Kalman CV (σ_obs=0.3mm, σ_proc=1.0 — 노트북 carry) |
| 타깃 | `rotate_xy(y − kalman, θ)` (yaw 좌표계 잔차). aux F = `rotate_xy(y − last, θ)`, aux W = `rotate_xy(y − kalman_altσ, θ)` |
| model | 단방향 GRU(h=64, l=1) + scalar concat → main + aux F/W head, main `tanh×2cm` clip |
| loss | main `combo = euclid + 0.3·softhit(1cm)`; aux F/W `euclid`, λ=0.3 |
| ensemble | 2 config (A: lr5e-4 do0.3 / B: lr1e-3 do0.1) × **5-fold stable_fold_id** × 3 seed 평균 |
| calibration | per-axis α — OOF-fit 보고 + 노트북 하드코드 (1,0.95,1) cross-check (overfit-risk flag) |
| KR002 단일 변경 | **입력 seq(rel/v/a)도 `rotate_xy(θ)` 로 yaw 회전** (magnitude scalar 는 회전불변이라 그대로). 그 외 KR001 과 동일 |
| metric | OOF hit_1cm (world Euclid < 0.01m), hit_1p5cm. paired permutation 10k vs F0 / vs KR001 |
| compare floor | F0 0.6320 · plan-022 best 0.6528 · 노트북 OOF 0.6625(자기 split) |
| 합격 기준 | **G_repro**: KR001 OOF hit_1cm ≥ **0.6320 PASS** (F0 floor), 0.6528+ STRONG, 0.6600+ EXCELLENT. **G_yaw**: Δ(KR002−KR001) ≥ **+0.002** & p<0.05 = positive lift |

### Commit chain (예정)

| commit | spec | status |
|---|---|---|
| c0 spec | §0~§7 (본 파일) | [DONE] |
| c1 kalman + yaw utils | §4.1 `analysis/plan-a-001/kalman.py`, `yaw.py` | [TODO] |
| c2 features | §4.2 `features.py` (seq 11×9 + scalar 40 + noise 추정) | [TODO] |
| c3 model + loss | §4.3 `model.py` (GRUModelMultiAux), `losses.py` (combo/softhit/aux) | [TODO] |
| c4 OOF runner | §4.4 `run_oof.py` (5-fold stable_fold_id + 2cfg + 3seed + calibration + `--input-yaw` flag) | [TODO] |
| c5 smoke | §5 `tests/test_plan_a001_smoke.py` (import + 1-fold 1-seed 1-epoch finite) | [TODO] |
| c6 G1 validation | §5 1-fold 1-seed full-epoch — hit_1cm > Kalman-alone 확인 | [TODO] |
| c7 KR001 full repro | §5 2cfg×5fold×3seed OOF → `results_kr001.json/.npz` | [TODO] |
| c8 KR002 input-yaw | §5 동일 budget + `--input-yaw` → `results_kr002.json/.npz` | [TODO] |
| c9 results + merge | §5 `plan-a-001-...results.md` frontmatter sync + **main merge** (§4 lane lifecycle) | [TODO] |

### G-gates

- G0: c1~c5 인프라 + smoke green                                  [TODO]
- G1: 1-fold 1-seed hit_1cm finite & > Kalman-alone (잔차 GRU 신호 sanity) [TODO]
- G_repro (G2): KR001 full OOF hit_1cm band 판정                    [TODO]
- G_yaw (G3): KR002 full OOF Δ vs KR001 + paired permutation        [TODO]
- G_final: results 박제 + §0.5 sync + worktree → main merge          [TODO]

---

## §1. 배경

두 갈래 검증 (사용자 narrative):

1. **이 방법의 프로젝트 OOF 확인**: `notes/LB_0.6780 코드공유.ipynb` 는 anchor/selector paradigm 과 *완전히 다른 계보* — 칼만 CV 로 +80ms 직진 외삽 후 **그 잔차만 GRU 로 회귀**, yaw 좌표계 타깃 + F/W 보조 head + per-axis calibration 으로 **LB 0.6780** (역대 19 제출 중 최고, plan-004 PB_0.6822 계열 0.6806 다음). 노트북 자체 OOF = 0.6625. 본 plan 은 이 파이프라인을 프로젝트 데이터·`stable_fold_id` 5-fold 위에 이식해 **프로젝트 비교계(OOF hit_1cm)** 로 박제한다.

2. **plan-003 대조 — 부호 역전 가설**: plan-003 R001 (linear extrap 잔차 GRU + Huber) 은 OOF euclidean 은 baseline 과 동률이었으나 LB 0.60 → **0.5688 로 퇴보** (잔차 GRU 가 hit@1cm 에 *음(−) 기여*). 진단(이 세션 분석): linear 2점 외삽이 측정노이즈를 증폭 → 잔차 타깃이 노이즈 → GRU 가 jitter 만 추가 → 경계점이 1cm 밖으로 밀림. 노트북은 (a) 칼만 평활 baseline → 잔차가 *학습가능 구조신호*, (b) softhit → loss 가 metric 정렬, (c) tanh×2cm → 출력 clamp 로 jitter 손해 차단. 본 plan 은 이 조합이 잔차 GRU 의 부호를 양(+)으로 뒤집는지 검증.

3. **입력 yaw 회전 (KR002)**: 노트북은 *타깃만* yaw 회전하고 입력 seq 는 world 그대로 둔다 (입력의 속도채널이 heading 을 이미 담아 모델이 self-rotate 가능). plan-021 은 Frenet 으로 *입력*을 회전(input-augment)해 positive band ("invariance→sample efficiency↑") 를 냈으나 4-lever 묶음이라 단독 기여 불명. KR002 = 입력 seq 도 yaw 회전하는 **단일 변수 ablation** → 입력 회전의 순수 기여를 깨끗이 분리 측정. yaw 는 Frenet 의 직진 degeneracy 가 없어 더 robust (수평만 회전, 수직 보존).

## §2. 가설

- **H1 (재현)**: 노트북 파이프라인을 프로젝트 5-fold 에 이식하면 OOF hit_1cm ≥ F0 0.6320 (필수), plausibly ≥ plan-022 0.6528. 미달이면 paradigm 이 프로젝트 데이터/split 으로 transfer 안 됨 또는 이식 버그.
- **H2 (부호 역전)**: 칼만 baseline + softhit + tanh clip 조합은 잔차 GRU 를 baseline(Kalman-alone) 대비 *상승*시킨다 (plan-003 의 퇴보와 반대). G1 에서 hit_1cm > Kalman-alone 으로 1차 확인.
- **H3 (입력 회전)**: 입력 seq 도 yaw 회전하면 OOF 가 소폭 상승(+) 한다. 단 핵심 이득(출력 정렬)은 타깃 회전이 이미 흡수했으므로 lift 는 작을 것(Δ ≈ +0.000~+0.005). 음(−) 이면 입력의 절대 heading 신호 상실이 회전 정렬 이득을 상회한다는 반증.

## §3. 실험 목록

### KR001_notebook-repro
- **type**: full-stack 재현 (잔차 회귀 paradigm)
- **baseline (잔차 기준)**: Kalman CV (σ_obs=0.3e-3, σ_proc=1.0). Kalman-alone OOF hit_1cm 도 별도 박제.
- **변경 변수**: (vs 프로젝트 기존) paradigm 전체 신규. 노트북 cell 6~33 이식.
- **config/경로**: `analysis/plan-a-001/run_oof.py` default (input-yaw off)
- **기대 runtime**: 2cfg × 5fold × 3seed × 200ep. GPU(cuda) ~1.5h 추정. CPU 시 seed 3→1 자동 감소(decision-note) + 재추정.
- **성공 기준**: OOF hit_1cm ≥ 0.6320 (G_repro PASS). finite, NaN/Inf 0.
- **실패 분기**: < 0.6320 시 — (i) Kalman σ_obs mini-grid {0.1,0.3,1.0}mm, (ii) 입력 정규화/seq 채널 정의 audit, (iii) 수렴 부족이면 epoch↑/lr 점검. severe 가 아니라 *정보* (paradigm transfer 실패 박제).

### KR002_input-yaw-rot
- **type**: 단일 변수 ablation (vs KR001)
- **baseline**: KR001
- **변경 변수**: **입력 seq 의 rel/v/a 3개 벡터 묶음에 `rotate_xy(θ)` 적용** (z 보존). magnitude 기반 scalar(speed/acc/turn_cos 등)는 회전불변 → 변경 없음. θ = KR001 과 동일(마지막 속도 yaw). 그 외 model/loss/ensemble/calibration 전부 KR001 동일.
- **config/경로**: `run_oof.py --input-yaw`
- **기대 runtime**: KR001 과 동일.
- **성공 기준**: Δ = KR002 − KR001 OOF hit_1cm. positive band Δ ≥ +0.002 & paired permutation p<0.05.
- **실패 분기**: Δ ≤ −0.002 → 입력 회전이 절대 heading 신호를 깎음 (informative, 음 band 박제). |Δ|<0.002 → neutral (타깃 회전이 이미 이득 흡수 확정).

## §4. 서버 작업 순서 (모듈 이식 spec)

### §4.1 kalman.py / yaw.py (c1)
- `kalman_predict(X, model='CV', dt=0.04, t_pred=0.08, sigma_obs=0.3e-3, sigma_proc=1.0)` — 노트북 cell 7 vectorized CV 필터 그대로. 각 축 독립, t_pred 외삽 → (N,3).
- `yaw_angle(v)=atan2(v_y,v_x)`, `rotate_xy(vec,θ)` (x,y 회전 z 보존), `inverse_rotate_xy(vec,θ)` — cell 13. 항등성 assert.

### §4.2 features.py (c2)
- `build_seq_t3(X)` → (N,11,9): ch0-2 rel(=X−X[-1]), ch3-5 v(zero-pad t0), ch6-8 a(zero-pad t0,1). cell 22.
- `build_scalar_feats(X, noise_p, noise_s, noise_loo)` 21D + `build_tier3_extra(X)` 19D = 40D. cell 11/16. long-tail log1p.
- noise 추정: `noise_poly2`, `noise_savgol`, `noise_loo_spline` (cell 10). LOO spline 은 캐시(`analysis/plan-a-001/noise_cache.npz`).
- `--input-yaw` 시 build_seq_t3 출력의 rel/v/a 각 triplet 에 rotate_xy(θ) 적용 (KR002 only).

### §4.3 model.py / losses.py (c3)
- `GRUModelMultiAux(n_channels=9, scal_dim=40, hidden=64, layers=1, fc_hidden=128, p, aux_dims=[3,3], main_out_scale_cm=2.0)` — cell 24. main `tanh×2cm`.
- `loss_combo = euclid + 0.3·softhit(beta=0.002, thr=0.01)`; `loss_aux_euclid`. cell 18/20.

### §4.4 run_oof.py (c4)
- 5-fold = `stable_fold_id(sid,5)` (KFold shuffle 대신 — 프로젝트 OOF 호환).
- 각 fold: per-fold StandardScaler(seq 채널별 + scalar), 2 config(A/B) × 3 seed 학습, val 예측은 `inverse_rotate_xy` 로 world 복원 → OOF 누적.
- test 예측은 본 plan out-of-scope (OOF 만; DACON LB 미제출 — §6).
- calibration: OOF 에서 per-axis α grid {0.85..1.05 step .025} fit → headline 은 **uncalibrated**, calibrated 는 add-on (overfit-risk flag) + 노트북 (1,0.95,1) cross-check.

## §5. 합격 기준 + Gate

| gate | 검사 | PASS band | severity |
|---|---|---|---|
| G0 | import + smoke (1f1s1e finite) | green | severe halt if import/NaN |
| G1 | 1-fold 1-seed full-ep hit_1cm | **> Kalman-alone** (잔차 GRU 양 기여 sanity) | warn |
| G_repro | KR001 full OOF hit_1cm | **≥0.6320 PASS** / ≥0.6528 STRONG / ≥0.6600 EXCELLENT | <0.6320 = FAIL_transfer (정보, halt X) |
| G_yaw | KR002 OOF Δ vs KR001 | **≥+0.002 & p<0.05 = positive** / \|Δ\|<0.002 neutral / ≤−0.002 negative | warn |
| G_final | 양 exp results 박제 + §0.5 sync + main merge | 완료 | — |

- statistic: paired permutation 10000 resample (KR001 vs F0, KR002 vs KR001), p threshold 0.05.
- artifact: `analysis/plan-a-001/results_kr001.json`, `results_kr002.json` + `.npz`(oof_pred), `plan-a-001-...results.md`.
- **NaN/Inf/divergence 0 의무**. cuda OOM 시 batch 256→128→64 자동 감소.

## §6. Out of scope

- DACON LB 제출 (quota 소모 — 사용자 명시 confirm 필요, 본 plan 은 OOF 만). 
- anchor/selector paradigm 과의 ensemble.
- boundary corrector 2-stage (plan-004 계열).
- Kalman σ 대규모 grid (실패 분기에서만 mini-grid).
- 입력+타깃 동시 회전 외 다른 좌표계(Frenet 등) — KR002 는 yaw 단독.
- calibration grid 의 본격 tuning (overfit-risk; headline 은 uncalibrated).

## §7. 참조

- `notes/LB_0.6780 코드공유.ipynb` — 이식 원본 (cell 6~33).
- `plans/plan-003-residual-gru-grid.results.md` — linear-extrap 잔차 GRU LB 0.5688 (대조군).
- `plans/plan-021-frenet-corrector-input-augment.results.md` — 입력 프레임 회전 positive band 선행 evidence.
- `analysis/plan-020/baseline_f0.py` — F0 0.6320 + hit metric.
- `WORKFLOW.md §4` — lane mutex + worktree→main merge 규약 (본 plan = lane a 첫 plan).

decision-note: spec-default — plan-a-001 = lane 형식 첫 plan (WORKFLOW.md §4 신규 규약). exp prefix=KR (Kalman-Residual). 5-fold = stable_fold_id (노트북 KFold 대신 프로젝트 OOF 호환). headline metric = uncalibrated OOF hit_1cm (calibration overfit-risk 회피). DACON 제출 out-of-scope (quota confirm 정책). KR002 단일 변경 = 입력 seq yaw 회전만.
