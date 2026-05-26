---
plan_id: d-001
version: 1
date: 2026-05-26 (Asia/Seoul)
status: draft
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
| paradigm | **Neural ODE** — 학습 가속도장 `a_θ(pos,vel,latent,heading,speed)` + RK4 단일스텝 적분 (anchor/selector·잔차-GRU 아님; 물리 *자체를 학습*) |
| data | `load_all_samples` X (N,11,3), `load_labels` y (N,3). horizon +80ms = `dt_physical 0.08` 단일 적분과 정합 |
| 상태계 | local frame: `init_pos=0`, `init_vel = (diffs@R)[:,-1]/0.04`. `dpos=vel`, `dvel = -damping·vel + a_θ` |
| model | `SimpleNeuralODEModel(input_dim=24, latent_dim=64)`: backbone(Lin24→64+LN+GELU+ResBlock) → latent; `SimpleAccelerationField`(in=3+3+64+2 → 64 → ResBlock → 3); `learned_damping`(3, init 0.1), `local_bias`(3), `global_bias`(3) — **노트북 cell 8 그대로** |
| 적분 | RK4 1-step, `dt_physical=0.08`. `pred_global = p_last + R·(pos+local_bias) + global_bias`. RK4 4단계 가속도 `_last_accels` 보관 (정규화용) |
| loss | `soft_hit + 126.309·huber(δ=0.001026) + 1e-4·accel_reg`. soft_hit=`(1-σ(-(d-0.011178)·332.259)).mean()`. **상수 전부 노트북 cell 10 그대로** |
| features (재구현) | 24D 로컬 스케일드 (속도/가속도/jerk 벡터+크기, heading sin/cos, 이동 std/축, 평균 speed). **외부 extract_features 부재 → markdown 의미 매칭 재구현** (byte-exact 불가, decision-note) |
| 학습 | AdamW lr=4e-3 wd=1e-3, batch 256, epochs 15, **5-fold 전부** (노트북은 fold0 break — OOF 위해 전 fold). 1 seed default |
| fold split | `stable_fold_id(sid,5)` (노트북 KFold 대신 — 프로젝트 OOF 호환) |
| metric | OOF hit_1cm (world Euclid < `R_HIT`=0.01m), hit_1p5cm. paired permutation 10k vs F0 |
| compare floor | F0 0.6320 · c-001 F0-잔차 0.6622 · a-001 KR002 0.6639(OOF) · 현 LB record 0.6854 |
| 합격 기준 | **G_repro**: NODE001 OOF hit_1cm ≥ **0.6320 PASS** (F0 floor; 노트북 자칭 LB 0.6+ 와 정합), ≥0.6622 STRONG (잔차 paradigm 동급), ≥0.6854 EXCELLENT (record 돌파). <0.6320 = FAIL_transfer (정보 — ideas.md §S2 기각 실측 확증, severe 아님) |

### Commit chain (예정)

| commit | spec | status |
|---|---|---|
| c0 spec | §0~§7 (본 파일) | [TODO] |
| c1 data + frame | §4.1 `analysis/plan-d-001/data.py` (load+diffs+p_last), `frame.py` (θ·R from yaw.py) | [TODO] |
| c2 features | §4.2 `features.py` — `extract_features` 24D 재구현 + train-fold mean/std scaling stats | [TODO] |
| c3 model + loss | §4.3 `model.py` (`SimpleNeuralODEModel`+RK4+damping, cell 8 그대로), `losses.py` (softhit+huber+accel_reg, cell 10 그대로) | [TODO] |
| c4 OOF runner | §4.4 `run_oof.py` (5-fold stable_fold_id + per-fold scaling + full train loop + world hit_1cm) | [TODO] |
| c5 smoke | §5 `tests/test_plan_d001_smoke.py` (import + 1f1s1e finite, NaN/Inf 0) | [TODO] |
| c6 G1 validation | §5 1-fold 1-seed full-epoch — val hit_1cm finite & epoch 단조 개선 (학습신호 sanity) | [TODO] |
| c7 NODE001 full repro | §5 5-fold×1seed×15ep OOF → `results_node001.json/.npz` | [TODO] |
| c8 results + merge | §5 `plan-d-001-...results.md` frontmatter sync + **worktree→main 자율 merge** (§4 lane lifecycle) | [TODO] |

### G-gates

- G0: c1~c5 인프라 + smoke green (import + 1f1s1e finite)            [TODO]
- G1: 1-fold 1-seed full-epoch val hit_1cm finite & epoch 단조 개선   [TODO]
- G_repro (G2): NODE001 full OOF hit_1cm band 판정 (≥0.6320 PASS)    [TODO]
- G_final: results 박제 + §0.5 sync + worktree→main merge            [TODO]

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
  - < 0.6320: (i) `extract_features` 24D 재구성 audit (속도 init `diffs_local[:,-1]/0.04` 단위·R 회전 방향 `diffs@R` vs `R·pred` 일관성 검증), (ii) damping init 0.1 이 80ms 에서 과감쇠인지 (`exp(-0.1·0.08)≈0.992` → 미미, 무해 확인), (iii) 수렴 부족이면 epoch 15→40 / lr 점검, (iv) loss 상수(332.259/126.309/0.011178/0.001026)가 프로젝트 좌표 단위(m)와 정합인지. 각 분기 결과 results.md 박제.
  - NaN/Inf: damping·bias gradient 폭주 또는 R 특이 → severe halt 후 진단.

---

## §4. 서버 작업 순서 (모듈 이식 spec)

### §4.1 data.py / frame.py (c1)
- `data.py`: `load_all_samples`/`load_labels` 로 X (N,11,3), y (N,3). `diffs = X[:,1:]-X[:,:-1]` → (N,10,3). `p_last = X[:,-1,:]` → (N,3). 노트북 data/cache npy 더미 fallback 은 제거 (실데이터만).
- `frame.py`: θ = `yaw_angle(v_last)` (마지막 step 속도; `v_last = diffs[:,-1]/0.04`). R (N,3,3) = z축 yaw 회전행렬 (xy 회전, z 보존) — `rotate_xy` 와 항등성 cross-check (`(diffs@R)` 가 local frame, `R·local` 가 world 복원). **노트북 forward 의 `diffs@R` (global→local) vs `einsum('nij,nj->ni',R,·)` (local→global) 부호 일관성 assert** (R 이 어느 방향인지 확정 후 박제 — code_reuse correctness).
- `speed = ||v_last||` (N,) → accel-field 의 `speed` 입력.

### §4.2 features.py — extract_features 재구현 (c2)
- ⚠️ **외부 `model.utils.extract_features` 는 프로젝트 부재 (작성자 사유 모듈)**. 노트북 호출 계약만 관측됨: `extract_features(X[,mean,std])` → 11-tuple `(ft, df, p_last, theta, _, _, _, R, speed, mean_stats, std_stats)`. 사용 위치는 index 0,1,2,3,7,8,9,10 뿐 (4,5,6 = `_` 미사용 → 재현 불요).
- **24D `ft` 재구성** (markdown 명세 "속도·가속도·jerk·heading θ·이동 std, 로컬 정렬·스케일"): local frame 회전 후 — 속도벡터+크기(4), 가속도벡터+크기(4), jerk벡터+크기(4), heading (sin,cos)(2), 평균 speed(1), step 변위 std/축(3), 평균 속도벡터(3), curvature/turn-cos(1), z-range(1), path-length(1) = **24**. `build_seq_t3`/`build_scalar_feats` (a-001) building block 재사용. **decision-note**: 정확한 원본 24D 구성 불가지 → markdown 의미 매칭 *재구성* (byte-exact 아님). 구성 표(dim→의미)를 `results_node001.json` 에 박제 → 사후 audit.
- **scaling**: `mean_stats,std_stats` = train fold ft 의 축별 mean/std. `extract_features(X_train)` (stats 인자 없음) = stats 계산 모드, `extract_features(X, mean, std)` = `(ft-mean)/std` 적용 모드. val/test 는 train fold stats 재사용 (leakage 0).

### §4.3 model.py / losses.py (c3)
- `model.py`: `ResBlock`, `SimpleAccelerationField`, `SimpleNeuralODEModel` — **노트북 cell 8 그대로** (구조 변경 0). `dt_physical=0.08`, RK4 단일스텝, `learned_damping` init `[0.1,0.1,0.1]`, `local_bias`/`global_bias` zeros. forward(features, diffs, p_last, theta, speed, R) 시그니처 보존. `_last_accels` (RK4 4단계 가속도) 정규화용 노출.
- `losses.py`: `loss = soft_hit + 126.309·huber + 1e-4·accel_reg`. `soft_hit=(1-sigmoid(-(d-0.011178)*332.259)).mean()`, `huber=F.huber_loss(pred,y,delta=0.001026)`, `accel_reg=mean_k(mean(||a_k||²))`. **상수 전부 노트북 cell 10 그대로** (좌표 단위 m 가정 — 실패 분기 (iv) 에서만 재점검).

### §4.4 run_oof.py (c4)
- 5-fold = `stable_fold_id(sid,5)` (노트북 KFold(shuffle,seed42) 대신 — 프로젝트 OOF 호환).
- 각 fold: train fold 로 `extract_features` stats 선계산 → AdamW(lr=4e-3,wd=1e-3) batch256 epochs15 학습 (전 fold; 노트북 fold0 break 제거). val 예측 → world 좌표 그대로 (forward 가 이미 global 출력) → OOF 누적.
- OOF hit_1cm = `mean(||oof_pred - y|| < R_HIT)`. hit_1p5cm = `< R_HIT_LOOSE`. paired permutation 10k vs F0 (`f0_baseline`).
- test 예측·DACON 제출 = out-of-scope (§6). seed 1 default (쇼케이스 = ensemble 없음); 시간 여유 시 3-seed 평균은 add-on (headline 은 1-seed).

## §5. 합격 기준 + Gate

| gate | 검사 | PASS band | severity |
|---|---|---|---|
| G0 | import + smoke (1f1s1e finite) | green | severe halt if import/NaN |
| G1 | 1-fold 1-seed full-ep val hit_1cm | finite & epoch 단조 개선 (학습신호 sanity) | warn |
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
