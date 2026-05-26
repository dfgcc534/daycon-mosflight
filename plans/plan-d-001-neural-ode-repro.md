---
plan_id: d-001
version: 1
date: 2026-05-26 (Asia/Seoul)
status: all_complete
lane: d
inspired_by:
  - notes/[LB_0.6+] Neural ODE 기반 예측모델.ipynb (재현 대상 — pos+vel 상태계 위 학습 가속도장 + RK4 단일스텝 + 학습 damping + local/global bias, 자칭 LB 0.6+)
  - notes/ideas.md §S2 (Neural ODE "Skip 확정" 분석 — regular grid·underdetermined·Lévy 급선회 smooth-dynamics 위배. 본 plan = 그 서류 기각을 *실측 OOF* 로 검증/반증)
  - 020 (CTRA 0.5070 · CTRV 0.5207 · Singer 0.5951 — 고정 물리 적분기 전부 F0 0.6320 미달. "Neural ODE F0 미시도 ★★" 로 명시 지목. 본 plan = 그 학습형 일반화 실행)
  - a-001 (LB_0.6780 노트북 재현 plan — 동일한 "코드공유 노트북 → 프로젝트 데이터/5-fold 이식" 프로토콜. frame util `yaw.py`, feature util `features.py` 재사용 원천)
  - c-001 (F0 잔차 GRU OOF 0.6622 — "고정 물리 + 잔차 보정" 패러다임. 본 plan 의 대조 paradigm: Neural ODE 는 잔차 대신 *물리 자체를 학습*)
code_reuse:
  - module: src/io.py
    symbols: [load_all_samples, load_labels]
    reason: 프로젝트 데이터 loader. X (N,11,3), y (N,3). 노트북의 data/cache/*.npy 더미 fallback 대체.
  - module: src/pb_0_6822/selector.py
    symbols: [stable_fold_id]
    reason: MD5 5-fold split (plan-020~a-001~c-001 carry, OOF 비교 호환 — 노트북 KFold(shuffle,seed42) 대신).
  - module: analysis/plan-020/baseline_f0.py
    symbols: [R_HIT, R_HIT_LOOSE, f0_baseline]
    reason: hit_1cm/1p5cm metric 정의 + F0 floor 0.6320 비교.
  - module: analysis/plan-a-001/yaw.py
    symbols: [yaw_angle, rotate_xy]
    reason: 로컬 yaw 프레임 θ + 회전행렬 R 구성 (노트북 extract_features 의 R/theta 출력 재구현 building block).
  - module: analysis/plan-a-001/features.py
    symbols: [build_seq_t3, build_scalar_feats]
    reason: 속도/가속도/jerk seq + scalar feature 산출 building block (24D 재구성 §4.2).
exp_ids:
  - NODE001_notebook-repro
---

# plan-d-001 — Neural ODE 쇼케이스 노트북 재현 (학습 가속도장 + RK4)

## §0. 한 줄 목적

> **`notes/[LB_0.6+] Neural ODE 기반 예측모델.ipynb` (pos+vel 6D 상태계 위 가속도장을 MLP 로 학습 → RK4 단일스텝 적분, 학습 damping `-d·v` + local/global bias, Huber+Soft-Hit(1cm)+가속도 정규화 loss) 파이프라인을 프로젝트 데이터/`stable_fold_id` 5-fold 위에 *그대로 이식*해 OOF hit_1cm 을 박제(NODE001)** 한다. 노트북이 의존하는 외부 `model.utils.extract_features` (프로젝트 부재) 는 노트북 markdown 명세대로 **재구현**. ideas.md §S2 "Skip 확정" 과 plan-020 "미시도 ★★" 로 *판정만 충돌하고 한 번도 실행 안 된* Neural ODE 를 실측 OOF 로 결론낸다.

---

## §0.5 Quick Reference (autonomous loop 매 turn 읽는 section)

| 항목 | 값 |
|---|---|
| paradigm | **Neural ODE** — 학습 가속도장 `a_θ(pos,vel,latent,θ,speed)` + RK4 단일스텝 적분 (anchor/selector·잔차-GRU 아님; 물리 *자체를 학습*) |
| data | `load_all_samples` X (N,11,3), `load_labels` y (N,3). horizon +80ms = `dt_physical 0.08` 단일 적분과 정합 |
| 상태계 | local frame: `init_pos=0`, `init_vel = (diffs@R)[:,-1]/0.04`. `dpos=vel`, `dvel = -damping·vel + a_θ` |
| model | `SimpleNeuralODEModel(input_dim=24, latent_dim=64)`: backbone(Lin24→64+LN+GELU+ResBlock) → latent; `SimpleAccelerationField`(in=3+3+64+2=72 [pos3+vel3+latent64+**θ1+speed1**, `+2`=θ·speed 스칼라이지 heading-2D 아님] → 64 → ResBlock → 3); `learned_damping`(3, init 0.1), `local_bias`(3), `global_bias`(3) — **노트북 cell 8 그대로** |
| 적분 | RK4 1-step, `dt_physical=0.08`. `pred_global = p_last + R·(pos+local_bias) + global_bias`. RK4 4단계 가속도 `_last_accels` 보관 (정규화용) |
| loss | `soft_hit + 126.309·huber(δ=0.001026) + 1e-4·accel_reg`. soft_hit=`(1-σ(-(d-0.011178)·332.259)).mean()`. **상수 전부 노트북 cell 10 그대로** |
| features (재구현) | 24D 로컬 스케일드 feature (속도·가속도·jerk·heading·통계량 — **dim→식 정본 표 = §4.2**). **외부 extract_features 부재 → markdown 의미 매칭 재구현** (byte-exact 불가, decision-note) |
| 학습 | AdamW lr=4e-3 wd=1e-3, batch 256, epochs 15, **5-fold 전부** (노트북은 fold0 break — OOF 위해 전 fold). 1 seed default |
| fold split | `stable_fold_id(sid,5)` (노트북 KFold 대신 — 프로젝트 OOF 호환) |
| metric | OOF hit_1cm (world Euclid < `R_HIT`=0.01m), hit_1p5cm. paired permutation 10k vs F0 |
| compare floor | F0 0.6320 · c-001 F0-잔차 0.6622 · a-001 KR002 0.6639(OOF) · 현 LB record 0.6854 |
| 합격 기준 | **G_repro**: NODE001 OOF hit_1cm ≥ **0.6320 PASS** (F0 floor; 노트북 자칭 LB 0.6+ 와 정합), ≥0.6622 STRONG (잔차 paradigm 동급), ≥0.6854 EXCELLENT (record 돌파). <0.6320 = FAIL_transfer (정보 — ideas.md §S2 기각 실측 확증, severe 아님) |

### Commit chain (예정)

| commit | spec | status |
|---|---|---|
| c0 spec | §0~§7 (본 파일) | [DONE] |
| c1 data + frame | §4.1 `analysis/plan-d-001/data.py` (load+diffs+p_last), `frame.py` (θ·R from yaw.py) | [DONE] |
| c2 features | §4.2 `features.py` — `extract_features` 24D 재구현 + train-fold mean/std scaling stats | [DONE] |
| c3 model + loss | §4.3 `model.py` (`SimpleNeuralODEModel`+RK4+damping, cell 8 그대로), `losses.py` (softhit+huber+accel_reg, cell 10 그대로) | [DONE] |
| c4 OOF runner | §4.4 `run_oof.py` (5-fold stable_fold_id + per-fold scaling + full train loop + world hit_1cm) | [DONE] |
| c5 smoke | §5 `tests/test_plan_d001_smoke.py` (import + 1f1s1e finite, NaN/Inf 0) | [DONE] |
| c6 G1 validation | §5 1-fold 1-seed full-epoch — val hit_1cm finite & final-epoch val hit_1cm ≥ epoch1 값 (비단조 noise 허용, crisp 판정) (학습신호 sanity) | [DONE] |
| c7 NODE001 full repro | §5 5-fold×1seed×15ep OOF → `results_node001.json/.npz` | [DONE] |
| c8 results + merge | §5 `plan-d-001-...results.md` frontmatter sync + **worktree→main 자율 merge** (§4 lane lifecycle) | [DONE] |

### G-gates

- G0: c1~c5 인프라 + smoke green (import + 1f1s1e finite)            [DONE] (frame self-check + smoke 3 pass + runner --smoke 63.02%)
- G1: 1-fold 1-seed full-epoch val hit_1cm finite & final-epoch val hit_1cm ≥ epoch1 값 (비단조 noise 허용, crisp 판정)   [DONE] (fold0 ep1 63.02%→final 63.07% PASS; peak ep3 66.63% overfitting 관찰)
- G_repro (G2): NODE001 full OOF hit_1cm band 판정 (≥0.6320 PASS)    [DONE] (**0.6330 PASS**, vs F0 Δ+0.0010 p=0.79 동률; H2 잔차 paradigm 미달)
- G_final: results 박제 + §0.5 sync + worktree→main merge            [DONE]

---

## §1. 배경

**Neural ODE 는 이 프로젝트에서 "판정만 두 번, 실행은 0번"** 인 미해결 카드다 (이 세션 git 추적 확정 — 최초 언급 2026-05-14, plan-005(05-12) 이후):

1. **ideas.md §S2 "Skip 확정"**: ❌ regular 40ms grid (불규칙 timestamp 강점 무효) ❌ 11-step 초기궤적 → ODE underdetermined ❌ Lévy 급선회 = 이산 이벤트 → smooth-dynamics 가정 위배. *분석 기각.*
2. **plan-020 "미시도 ★★"**: 고정 물리 적분기 — CTRA(OOF 0.5070)·CTRV(0.5207)·Singer(0.5951) — 가 **전부 F0 0.6320 미달**. 사람이 손으로 고른 힘 법칙이 모기 erratic 비행에 안 맞음 → "힘 법칙을 *학습*하는 Neural ODE 가 다음 후보" 로 명시 지목.

두 판정이 *충돌* 한다 ("하지 마" vs "해볼 만"). 본 plan 은 노트북이라는 *구체적 구현체* 가 생긴 김에 **실측 OOF 로 결론** 낸다. 노트북은 자칭 `[LB_0.6+]` 이고 본문에 "교육·공유용 쇼케이스 베이스라인 — 무거운 feature·복잡 감쇄항 배제" 라 명시 → *상한이 아니라 하한* 을 보는 실험.

**개념 위치 (newcomer 요약)**: F0 = 등가속도 ODE 를 손으로 푼 닫힌해(`p₀+1.98v+1.20a_par`). a-001/c-001 = 고정 물리(Kalman CV·F0) **뼈대 + GRU 잔차 보정**. Neural ODE 는 분업을 뒤집어 **가속도장 자체를 신경망이 학습 + RK4 로 적분** (잔차 보정기 없음). 즉 *F0 의 학습형 일반화* (신경망이 "가속도=상수" 만 내면 정확히 F0 로 환원).

---

## §2. 가설

- **H1 (재현)**: 노트북 파이프라인을 프로젝트 5-fold 에 이식하면 OOF hit_1cm ≥ F0 0.6320 (자칭 LB 0.6+ 와 정합). 미달이면 (a) 쇼케이스 단순화(stripped feature)가 프로젝트 데이터에서 floor 미달, 또는 (b) `extract_features` 재구현이 원본 의미를 못 살림, 또는 (c) ideas.md §S2 의 기각 사유(underdetermined/급선회)가 실측으로 옳음 — 분기 진단 §3.
- **H2 (paradigm 대조)**: "학습 물리(Neural ODE)" 가 "고정 물리+잔차(c-001 F0-GRU 0.6622)" 와 동급 이상이면 → 잔차 패러다임 외 대안 존재. 미달이면 → 이 프로젝트에서 *잔차 보정 > 순수 학습 물리* 재확인 (plan-020 의 "고정 물리 다 졌다" 의 학습형까지 확장).
- **H3 (§S2 반증/확증)**: regular grid·11-step·급선회가 정말 Neural ODE 를 무력화하는가? OOF ≥ 0.6320 이면 §S2 기각은 과했던 것(반증), < 0.6320 이면 §S2 가 옳았던 것(확증). 어느 쪽이든 박제 = 카드 종결.

---

## §3. 실험 목록

### NODE001_notebook-repro
- **type**: full-stack 재현 (Neural ODE paradigm, 신규).
- **baseline (비교 floor)**: F0 0.6320 (`f0_baseline`, plan-020). 추가 cross-ref: c-001 0.6622, a-001 0.6639, LB record 0.6854.
- **변경 변수**: (vs 프로젝트 기존) paradigm 전체 신규. 노트북 cell 2~14 이식 (단 cell 12 3D 시각화 = 분석용, OOF 무관 → out-of-scope).
- **config/경로**: `analysis/plan-d-001/run_oof.py` default. OOF json/npz 는 `analysis/plan-d-001/` (runs/ 미사용).
- **기대 runtime**: 5-fold × 1 seed × 15 epoch, batch 256. GPU(cuda) ~20–40min 추정 (RK4 4× accel-field eval/step). CPU 시 그대로 진행(단일스텝이라 가벼움), epoch 시간 재추정.
- **성공 기준**: OOF hit_1cm ≥ 0.6320 (G_repro PASS). finite, NaN/Inf 0.
- **실패 분기** (severe 아님 — 정보):
  - < 0.6320: (i) `extract_features` 24D 재구성 audit (속도 init `diffs_local[:,-1]/0.04` 단위·R 회전 방향 `diffs@R` vs `R·pred` 일관성 검증), (ii) damping init 0.1 이 80ms 에서 과감쇠인지 (`exp(-0.1·0.08)≈0.992` → 미미, 무해 확인), (iii) 수렴 부족이면 epoch 15→40 / lr 점검, (iv) loss 상수(332.259/126.309/0.011178/0.001026)가 프로젝트 좌표 단위(m)와 정합인지. (v) **H3 교란 격리** — feature 재구현 미스(b) vs §S2 paradigm-한계(c) 분리: §4.2 24D 의 d0–11(검증된 vL/aL/jL 물리량)만 유지하고 **d12–23 을 0 으로 채워(input_dim=24 불변, model arch 그대로)** 1-fold mini-run OOF 를 측정해 full-24D 와 비교 → 큰 격차면 d12–23 통계 feature 재구성 결함(b, audit 회수 가능), 물리량만으로도 미달이면 §S2 확증(c)에 무게. 각 분기 결과 results.md 박제.
  - NaN/Inf: damping·bias gradient 폭주 또는 R 특이 → severe halt 후 진단.

---

## §4. 서버 작업 순서 (모듈 이식 spec)

### §4.1 data.py / frame.py (c1)
- `data.py`: `load_all_samples`/`load_labels` 로 X (N,11,3), y (N,3). `diffs = X[:,1:]-X[:,:-1]` → (N,10,3). `p_last = X[:,-1,:]` → (N,3). 노트북 data/cache npy 더미 fallback 은 제거 (실데이터만).
- `frame.py`: θ = `yaw_angle(v_last)` (마지막 step 속도; `v_last = diffs[:,-1]/0.04`). R (N,3,3) = z축 yaw 회전행렬 (xy 회전, z 보존) — `rotate_xy` 와 항등성 cross-check (`(diffs@R)` 가 local frame, `R·local` 가 world 복원). **노트북 forward 의 `diffs@R` (global→local) vs `einsum('nij,nj->ni',R,·)` (local→global) 부호 일관성 assert** (**확정: R = local→global yaw 회전행렬, global→local = Rᵀ = `vec@R`. 노트북 cell8 `diffs@R`(in)·`einsum('nij,nj->ni',R,·)`(out) 가 이 방향으로 내부 일관** — assert 는 이 방향 검증, code_reuse correctness).
- `speed = ||v_last||` (N,) → accel-field 의 `speed` 입력.

### §4.2 features.py — extract_features 재구현 (c2)
- ⚠️ **외부 `model.utils.extract_features` 는 프로젝트 부재 (작성자 사유 모듈)**. 노트북 호출 계약만 관측됨: `extract_features(X[,mean,std])` → 11-tuple `(ft, df, p_last, theta, _, _, _, R, speed, mean_stats, std_stats)`. 사용 위치는 index 0,1,2,3,7,8,9,10 뿐 (4,5,6 = `_` 미사용 → 재현 불요). **사용 slot 반환 계약** (전부 `torch.float32`, model device): `ft`(0)=(N,24), `df`(1)=(N,10,3), `p_last`(2)=(N,3), `theta`(3)=(N,) [accel-field 가 dim==1 시 unsqueeze], `R`(7)=(N,3,3), `speed`(8)=(N,), `mean_stats`(9)/`std_stats`(10)=(24,).
- **24D `ft` 재구성 — dim→식 정본 표** (markdown 명세 "속도·가속도·jerk·heading·이동 std, 로컬 정렬·스케일"). 표기: local frame = world 벡터를 `@R` (=Rᵀ·, §4.1) 로 회전. 단계량 `vL[t]=(diffs@R)[t]/0.04` (t=0..9), `aL[t]=vL[t]−vL[t−1]` (t=1..9), `jL[t]=aL[t]−aL[t−1]` (t=2..9), `XL[t]=(X−p_last)@R` (t=0..10). `‖·‖`=L2.

  | dim | 식 |
  |---|---|
  | 0–2 | `vL[-1]` (마지막 step 속도, local) |
  | 3 | `‖vL[-1]‖` |
  | 4–6 | `aL[-1]` |
  | 7 | `‖aL[-1]‖` |
  | 8–10 | `jL[-1]` |
  | 11 | `‖jL[-1]‖` |
  | 12–13 | `(sinθ, cosθ)`, θ=`yaw_angle(v_last)` (§4.1) |
  | 14 | `mean_t ‖vL[t]‖` (평균 speed) |
  | 15–17 | `std_t((diffs@R)[t])` 축별 (x,y,z 표준편차) |
  | 18–20 | `mean_t vL[t]` (평균 속도벡터) |
  | 21 | turn-cos `⟨vL[-2],vL[-1]⟩ / (‖vL[-2]‖·‖vL[-1]‖)` (곡률 proxy; 분모 0 시 1.0) |
  | 22 | z-range `max_t XL[t]_z − min_t XL[t]_z` |
  | 23 | path-length `Σ_t ‖diffs[t]‖` |

  합 = 3+1+3+1+3+1+2+1+3+3+1+1+1 = **24**. 전부 diffs/X 로부터 직접 산출 → **self-contained** (`build_seq_t3`/`build_scalar_feats` 는 동일 vL/aL/jL 산출 시 *저수준 helper 로만* 선택 사용 가능, a-001 채널 layout 미채택). **decision-note**: 원본 외부 모듈의 정확 24D 구성 불가지 → 위 표는 markdown 의미 매칭 *재구성* (byte-exact 아님). 위 표 그대로 `results_node001.json` 에 박제 → 사후 audit.
- **scaling**: `mean_stats,std_stats` = train fold ft 의 축별 mean/std (**본 plan 의 모든 std/표준편차 = `ddof=0` 모분산 — dim15–17 포함, numpy/torch default 차이 제거**). `extract_features(X_train)` (stats 인자 없음) = stats 계산 모드, `extract_features(X, mean, std)` = `(ft-mean)/(std+1e-8)` 적용 모드 (**eps=1e-8 0-division 가드** — 상수 feature 축(z-range/path-length 등) Inf 방지 → §5 "NaN/Inf 0 의무" 충족). val/test 는 train fold stats 재사용 (leakage 0).

### §4.3 model.py / losses.py (c3)
- `model.py`: `ResBlock`, `SimpleAccelerationField`, `SimpleNeuralODEModel` — **노트북 cell 8 그대로** (구조 변경 0). `dt_physical=0.08`, RK4 단일스텝, `learned_damping` (3,) init `[0.1,0.1,0.1]` (= vel 3축 **per-axis elementwise** 계수, batch broadcast; `dvel = -learned_damping*vel + a` 의 `*` 는 elementwise), `local_bias`/`global_bias` zeros. forward(features, diffs, p_last, theta, speed, R) 시그니처 보존. `_last_accels` (RK4 4단계 가속도) 정규화용 노출.
- `losses.py`: `loss = soft_hit + 126.309·huber + 1e-4·accel_reg`. `soft_hit=(1-sigmoid(-(d-0.011178)*332.259)).mean()`, `huber=F.huber_loss(pred,y,delta=0.001026)`, `accel_reg = mean_k( mean_batch( Σ_axis a_k² ) )` = `sum(a.pow(2).sum(-1).mean() for a in _last_accels)/4` (노트북 cell10: per-sample 축-합 → batch-mean → RK4 4단계 평균). **상수 전부 노트북 cell 10 그대로** (좌표 단위 m 가정 — 실패 분기 (iv) 에서만 재점검).

### §4.4 run_oof.py (c4)
- 5-fold = `stable_fold_id(sid,5)` (노트북 KFold(shuffle,seed42) 대신 — 프로젝트 OOF 호환).
- 각 fold: train fold 로 `extract_features` stats 선계산 → AdamW(lr=4e-3,wd=1e-3) batch256 epochs15 학습 (전 fold; 노트북 fold0 break 제거). **매 epoch 종료 시 val hit_1cm 계산·로깅** (G1 의 epoch1·final-epoch 비교용 — runner 가 per-epoch val eval pass 보유). val 예측 → world 좌표 그대로 (forward 가 이미 global 출력) → OOF 누적.
- OOF hit_1cm = `mean(||oof_pred - y|| < R_HIT)`. hit_1p5cm = `< R_HIT_LOOSE`. paired permutation 10k vs F0 (`f0_baseline`).
- test 예측·DACON 제출 = out-of-scope (§6). seed 1 default (쇼케이스 = ensemble 없음); 시간 여유 시 3-seed 평균은 add-on (headline 은 1-seed).

## §5. 합격 기준 + Gate

| gate | 검사 | PASS band | severity |
|---|---|---|---|
| G0 | import + smoke (1f1s1e finite) | green | severe halt if import/NaN |
| G1 | 1-fold 1-seed full-ep val hit_1cm | finite & final-epoch val hit_1cm ≥ epoch1 값 (비단조 noise 허용, crisp 판정) (학습신호 sanity) | warn |
| G_repro | NODE001 full OOF hit_1cm | **≥0.6320 PASS** / ≥0.6622 STRONG / ≥0.6854 EXCELLENT | <0.6320 = FAIL_transfer (정보, halt X) |
| G_final | results 박제 + §0.5 sync + main merge | 완료 | — |

- statistic: paired permutation 10000 resample (NODE001 vs F0), p threshold 0.05.
- artifact: `analysis/plan-d-001/results_node001.json` (+ feature-dim 구성표 + R 부호 audit) + `.npz`(oof_pred), `plan-d-001-...results.md`.
- **NaN/Inf/divergence 0 의무**. cuda OOM 시 batch 256→128→64 자동 감소.

## §6. Out of scope

- DACON LB 제출 (quota 소모 — 사용자 명시 confirm 필요, 본 plan 은 OOF 만).
- multi-step 적분 (2+ RK4 step / dt 분할) — 노트북은 80ms 단일스텝, 그대로.
- ensemble / multi-seed 본격 (1-seed headline; 3-seed 는 add-on).
- 입력 yaw 회전·Frenet 등 좌표계 ablation (a-001 영역 — isolation, 본 plan 미포함).
- byte-exact `extract_features` (외부 모듈 부재 — markdown 의미 재구성, §4.2).
- cell 12 RK4 경로 3D 시각화 (분석용, OOF 무관).
- architecture sweep (latent_dim·ResBlock·damping init 등 — 재현 후 별도 plan).

## §7. 참조

- `notes/[LB_0.6+] Neural ODE 기반 예측모델.ipynb` — 이식 원본 (cell 2~14; 8=model, 10=loss/train, 12=시각화는 skip).
- `notes/ideas.md §S2` — Neural ODE "Skip 확정" 분석 (본 plan = 실측 검증 대상).
- `plans/plan-020-f0-structural-search.md` (line 167 "Neural ODE F0 미시도 ★★") + `analysis/plan-020/results_deterministic.md` (CTRA/CTRV/Singer < F0).
- `plans/plan-a-001-kalman-residual-gru-repro.md` — 코드공유 노트북 재현 프로토콜 템플릿 + `yaw.py`/`features.py` 원천.
- `plans/plan-c-001-f0-residual-gru.results.md` — F0 잔차 GRU 0.6622 (대조 paradigm).
- `analysis/plan-020/baseline_f0.py` — F0 0.6320 + hit metric (R_HIT=0.01).
- `WORKFLOW.md §4` — lane mutex + worktree→main merge (본 plan = lane d 첫 plan).

decision-note: spec-default — plan-d-001 = lane d 첫 plan. exp prefix=NODE (Neural ODE). 5-fold=stable_fold_id (노트북 KFold 대신, OOF 호환). 전 fold 학습 (노트북 fold0 break 제거 — OOF 필요). model/loss 상수 = 노트북 cell 8/10 그대로 이식. extract_features = 외부 모듈 부재로 markdown 의미 재구성 (byte-exact 불가, 구성표 박제 audit). 1-seed headline (쇼케이스 ensemble 없음). DACON 제출·시각화·ablation·arch sweep = out-of-scope. G_repro FAIL(<F0) 은 severe 아닌 정보 (ideas.md §S2 기각 실측 = 본 plan 의 valid outcome).
