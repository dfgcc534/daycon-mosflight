---
plan_id: a-004
version: 1
date: 2026-05-26 (Asia/Seoul)
status: draft
lane: a
inspired_by:
  - a-003 (KR008 LB 0.6862 record. 단일 회귀 head — 본 plan baseline. lever ROI 수확 체감)
  - mode_selectability.py (★ 결정 evidence: 학습 거친 모드 GBM 86~95% 선택 가능 ≠ 14 고정 anchor 13%≈chance. selection 은 벽 아님. 단 뭉툭 centroid 후보는 KR008 정밀도 미달 → 정밀 per-sample head 필요)
  - ceiling_verify.py / roi_diagnostic.py (단일 점예측·corrector·고정 anchor selector = ROI 0. 천장은 *선택가능한 multimodal* 에 있음)
  - 022-031 (anchor/selector paradigm — 14 후보 selector 0.6528 정체. 본 plan 은 정밀 학습 head 로 selector 문제 우회 시도)
code_reuse:
  - module: analysis/plan-a-002/run_oof.py
    symbols: [main, train_one, run_config, paired_perm, hit_rate, hit_mask]
    reason: KR008 OOF+test+aug 파이프라인. multi-head 분기만 추가 (n_heads=1 default → KR008 bit-identical).
  - module: analysis/plan-a-001/model.py
    symbols: [GRUModelMultiAux]
    reason: GRU trunk + head 구조. **수정 X** — plan-a-004 신규 model_mh.py 가 trunk 재사용하고 head 만 K-way + selector 확장.
  - module: analysis/plan-a-001/losses.py
    symbols: [loss_combo, loss_euclid, loss_softhit, LAMBDA_AUX]
    reason: combo loss. WTA/MCL wrapper + selector CE 신규 추가.
  - module: analysis/plan-a-002/{kalman_features,features_ext}.py
    symbols: [kalman_with_internals, cv_ca_disagreement, build_seq_ext, build_scalar_ext]
    reason: KR008 입력 feature (selector·CV/CA seed 후보용). 변경 없음.
exp_ids:
  - KR010_mcl2-decisive
  - KR011_gen-axis
  - KR012_route-K-axis
---

# plan-a-004 — Multi-Hypothesis 후보: 생성·선택 axis 탐색 (KR008 단일 회귀 → K-head)

## §0. 한 줄 목적

> **KR008 (Kalman-잔차 GRU, LB 0.6862 record, 단일 회귀 head) 의 출력 head 를 K개 정밀 per-sample head + selector 로 확장**해, 단일 회귀가 *평균-함정*에 빠지는 multimodal 샘플(미래 좌/우 분기 등)을 K개 후보로 분리·선택함으로써 +0.01 (LB ≥ 0.696) 시도. **핵심 = "후보를 어떻게 생성(generation)하고 어떻게 선택(routing)하나"의 넓은 탐색 axis** (§3). `mode_selectability.py` 가 *선택은 벽이 아님(거친 모드 95% 예측가능)* 을 입증한 위에서, 뭉툭 centroid 대신 *정밀 학습 head* 로 천장을 넘는지 검증. trunk·입력·Kalman 잔차 틀·고차물리-배제 전부 KR008 carry. verdict = LB (CV-LB 괴리, 사용자 gated).

---

## §0.5 Quick Reference (autonomous loop 매 turn 읽는 section)

| 항목 | 값 |
|---|---|
| paradigm | Kalman CV 잔차 회귀 (plan-a-002/003 carry) + **multi-hypothesis 출력 head** |
| baseline exp | **KR008** (반사+노이즈 aug, LB 0.6862, OOF 0.6671). 단일 head. 본 plan 비교 기준 |
| 확장 지점 | `model.py` 의 `z(fc_hidden//2) → head_main Linear(→3)` 를 **K개 head Linear(→3) + selector Linear(→K)** 로. GRU trunk·MLP 불변 |
| 후보 = | K개 회귀 head 출력 (anchor 아님). 각 head = KR008 만큼 정밀한 per-sample 예측, 자기 모드 특화 |
| 학습 | WTA/MCL (샘플별 정답 최근접 head 만 main grad) + selector CE(승자 예측) + aux F/W carry |
| 추론/제출 | selector argmax → 그 head 예측 1개 제출 (LB) |
| ★ 핵심 불확실성 | 단일 GRU 가 *이미* feature 로 soft mode-conditioning → MoE hard-routing 우위는 미확정. **G1 에서 fail-fast** |
| 탐색 axis | **생성(G1~G7) × 선택(R1~R3) × K(2~4)** — §3. 단계적 (cheap decisive 먼저) |
| metric | OOF: oracle@K, realized-hit(selector-picked), hit_1p5cm. + paired permutation vs KR008. LB(gated) |
| compare | KR008 OOF 0.6671 / **LB 0.6862** · mode_sel: 거친모드 oracle@2 0.59/@8 0.74 (정밀head 가 이걸 올려야) |
| 합격 기준 | **G1_decisive (KR010)**: oracle@2 ≥ **0.69** (KR008 +0.02, headroom 존재) & realized ≥ KR008−0.005. **미달 = KILL** (단일GRU 가 이미 conditional 최적, 정보). **G_mh (KR011/12)**: realized-hit > KR008 + paired p. **G_lb**: best LB vs 0.6862 (gated, +0.01 목표) |

### Commit chain (예정)

| commit | spec | status |
|---|---|---|
| c0 spec | §0~§7 (본 파일) | [TODO] |
| c1 multi-head model + WTA loss | §4.1 `analysis/plan-a-004/model_mh.py` (GRUMultiHead: K head + selector), `losses_mcl.py` (WTA combo + selector CE) | [TODO] |
| c2 runner | §4.2 `run_oof_mh.py` — KR008 파이프라인 + `--n-heads --gen --route` flag. n_heads=1 → KR008 bit-identical | [TODO] |
| c3 smoke | §5 `tests/test_plan_a004_smoke.py` — n_heads=1 repro·WTA loss·selector·1f1s1e finite | [TODO] |
| c4 G1_decisive (KR010) | §5 2-head MCL 1f full-ep — **oracle@2 vs KR008 headroom 판정** (KILL gate) | [TODO] |
| c5 G_mh 생성·선택 axis (KR011/12) | §5 G1 통과 시만 — 생성×선택×K sweep, best realized-hit OOF | [TODO] |
| c6 results + (LB gated) + merge | §5 `plan-a-004-...results.md` + §0.5 sync + lane-a merge | [TODO] |

### G-gates

- G0: c1~c3 인프라 + smoke green + n_heads=1 KR008 repro 불변
- **G1_decisive (KR010, ★ fail-fast)**: 2-head MCL 1-fold full-ep. **oracle@2(best-of-2 학습 head) ≥ 0.69** (KR008 0.6671 +0.02 = 정밀 head 가 mode_sel 의 거친 oracle 0.59 를 유의 초월?) & realized-hit ≥ KR008 − 0.005. **미달 → KILL** (단일 GRU 가 conditional 최적 = MoE 무용, 정보 박제, plan 조기 종료).
- G_mh (G2): G1 통과 시 — 생성×선택×K axis sweep, best realized-hit OOF vs KR008 + paired permutation.
- G_lb (G3): best config LB (사용자 gated) vs KR008 0.6862. +0.01(≥0.696) 목표, noise floor 내면 inconclusive.
- G_final: results 박제 + §0.5 sync + main merge.

### Plan-specific 주의

- **fail-fast 의무**: G1_decisive 가 KILL band 면 c5(breadth) 진입 금지 — 단일 GRU 가 이미 conditional 최적이란 강한 정보. mode_selectability 가 "선택 가능" 은 보였으나 "정밀 head 가 단일 GRU 초월" 은 미검증이라 G1 이 진짜 게이트.
- **CV-LB 괴리**: OOF realized-hit 가 KR008 과 neutral 이어도 LB 후보 (yaw·부산물 전례). 단 +0.01 은 noise floor(±0.003) 훨씬 위라 OOF 에서도 분명한 oracle headroom 없으면 LB 도 어려움 — G1 oracle 이 1차 필터.
- 고차 물리 모델(CTRV/고차 Kalman) 도입 금지 (사용자 명시). 단 *기존* CV/CA 필터(cv_ca) 를 후보 seed 로 쓰는 건 허용 (신규 물리 아님).

---

## §1. 배경

plan-a-003 종료: KR008 LB 0.6862 = record, 단일 회귀 head. ROI 진단으로 미세 lever·corrector·고정 anchor selector·단순 ensemble 전부 ROI≈0 확정 (`roi_diagnostic.py`, `ceiling_verify.py`).

**전환점 — `mode_selectability.py` (사용자 제안 "few hit-optimized candidates" 검정)**: 14 고정 anchor 선택은 chance(13%) 였으나, **학습 거친 모드(k-means K=2~8)는 GBM 86~95% 선택 가능** (chance 대비 Δ+0.17~+0.54). 즉 *"어느 모드인지"는 입력 feature 에 있다* — 미세 분할(14)이 묻은 구조를 거친 묶음이 드러냄. **selection 은 벽이 아니다.** 단 거친 centroid 후보(모드당 글로벌 offset 1개)는 KR008 의 per-sample 정밀도를 못 따라가 realized-hit 0.59~0.65 < 0.6671.

**가설의 핵심**: 뭉툭 centroid 를 **정밀 per-sample 학습 head** 로 바꾸면 (각 head = KR008 급 정밀 + 모드 특화), 단일 회귀가 *euclid loss 로 양갈래 평균(=둘 다 miss)* 을 내는 multimodal 샘플을 K head 가 각 갈래로 commit + 95% selector 가 선택 → 그 평균-함정 샘플 회수 → +0.01 가능.

**미해결 핵심 리스크 (정직)**: KR008 단일 GRU 는 *이미* 속도/회전 feature 로 soft mode-conditioning 한다 — 조건부 평균이 feature 로 갈리면 단일 head 도 모드별 예측을 낸다. MoE 의 hard-routing+전문화가 그 soft conditioning 을 *유의하게* 넘는지는 경험적이며 (충분 capacity 단일모델 ≈ MoE 흔함), **G1_decisive 가 이를 cheap 하게 판정** (oracle@2 headroom 없으면 즉시 KILL).

## §2. 가설

- **H1 (multi-head oracle headroom)**: K개 정밀 학습 head 의 oracle(best-of-K) 가 KR008 단일 head 를 유의 초월(≥+0.02 @ K=2). multimodal 샘플에서 head 가 분화하므로. 미달 = 단일 GRU 가 이미 conditional 최적 (KILL).
- **H2 (selectability → realized)**: mode_sel 의 95% 선택가능성이 정밀 head 에도 유지 → realized-hit(selector-picked) 가 oracle 의 큰 부분 회수 → KR008 초월. selector 가 head 분화에 맞춰 학습되므로 post-hoc GBM(95%) 보다 유리.
- **H3 (생성 mechanism 차이)**: emergent-MCL / hybrid(KR008+deviation) / supervised-mode / MDN 중 무엇이 oracle·realized 우월한지 axis 탐색.
- **메타 (CV-LB)**: OOF realized 가 neutral 이어도 LB 후보. 단 +0.01 은 oracle headroom 이 OOF 에서 분명해야 LB 도 가능.

## §3. 탐색 axis (후보 생성 × 선택 — 넓게)

**§3.A 생성 axis (G — K개 후보를 *어떻게 만드나*)** — 본 plan 핵심 탐색:

| id | 생성 방식 | 후보 정밀도 | 비고 |
|---|---|---|---|
| **G1 emergent-MCL** | K개 동일구조 head, WTA loss 로 self-organize (모드 비지정) | per-sample 정밀 | 기본·최단순. KR010 |
| **G2 hybrid-anchored** | head-0 = KR008(frozen or warm) "직진 default", head-1..K−1 = 학습 "deviation" head (residual-of-residual) | 정밀 | KR008 record 보존 + 대안만 추가, 최저위험 |
| **G3 supervised-mode** | r_yaw k-means(K) 라벨 → head_k 를 해당 cluster 샘플로 학습 (모드 지정) | 정밀 | mode_sel 직계, 라벨 noise 위험 |
| **G4 motion-seed** | CV·CA(기존 cv_ca) 외삽을 2 seed 후보 + 각 GRU refine. (신규 물리 X — 기존 필터만) | 중~정밀 | 물리 prior 분화 |
| **G5 MDN** | head = K개 (mean,logvar,weight) 혼합가우시안, NLL loss | 정밀+불확실성 | weight 가 곧 selector |
| (G6 anchor-revival) | 14 기하 anchor + KR008 을 후보 pool, 개선 selector | 고정 anchor 정밀↓ | mode_sel 상 열위 — 후순위 ablation |
| (G7 ensemble-pool) | KR001~008 예측을 고정 후보 pool + selector | correlated, oracle 0.69 | roi_diag 상 저oracle — 후순위 |

**§3.B 선택 axis (R — K개 중 *어떻게 1개 고르나*)**:

| id | 선택 방식 | 비고 |
|---|---|---|
| **R1 joint-selector-head** | trunk z → Linear(→K) softmax, WTA 승자를 CE 로 학습. 추론 argmax | 기본. head 분화와 공동학습 |
| **R2 post-hoc-GBM** | 학습 후 feature→best-head GBM (mode_sel 의 95% 방식) | selector 분리, 진단 비교용 |
| **R3 confidence** | MDN weight (G5) 또는 head 간 spread/엔트로피 | gating 자연 |

**§3.C 용량/파라미터 axis**: K ∈ {2,3,4}, target param {residual-to-Kalman(KR008 동일) / residual-to-KR008(G2)}, WTA {hard / top-2 soft}, diversity reg {off/on}.

**탐색 grid 는 combinatorial → 단계 실행** (§3.D): cheap decisive(G1×R1×K2) 먼저 → 통과 시 생성 axis(G1/G2/G3/G5) × 선택(R1/R2) × K{2,3} sweep. G4/G6/G7 은 시간 여유 시 ablation.

## §3' 실험 목록

### KR010_mcl2-decisive (G1_decisive, fail-fast)
- **type**: 최단순 multi-head (vs KR008 단일 head)
- **baseline**: KR008
- **변경 변수**: 출력 head 1→2 (G1 emergent-MCL), WTA combo loss + R1 joint selector. K=2. 그 외 trunk/입력/aug 전부 KR008 동일.
- **config/경로**: `run_oof_mh.py --innov --filtered-v --cv-ca --input-yaw --reflect-aug --noise-aug 0.10 --n-heads 2 --gen mcl --route joint --gate g1 --exp KR010`
- **성공 기준 (KILL gate)**: oracle@2 (best-of-2 head, OOF 1-fold) ≥ **0.69** & realized-hit ≥ KR008 1-fold − 0.005. 미달 → **plan KILL** (단일 GRU conditional 최적, 정보 박제).
- **실패 분기**: oracle@2 < 0.69 → MoE 무용 결론, c5 진입 금지, results 에 "단일 GRU = conditional 최적" 박제 후 G_final(축소).

### KR011_gen-axis (G_mh, G1 통과 시만)
- **변경 변수**: 생성 mechanism {G1 mcl / G2 hybrid-anchored / G3 supervised-mode / G5 MDN} × K{2,3}, R1 고정. full OOF.
- **성공 기준**: best generation 의 realized-hit > KR008 0.6671 + paired permutation p<0.05 (또는 CV-LB 괴리상 neutral+LB 후보).

### KR012_route-K-axis (G_mh, 보강)
- **변경 변수**: best generation 위 선택 axis {R1 joint / R2 post-hoc-GBM / R3 confidence} × K{2,3,4}. realized-hit 최대화.
- **산출**: best config → `--predict-test` submission (LB §6 gated).

## §4. 서버 작업 순서 (모듈 spec)

### §4.1 model_mh.py / losses_mcl.py (c1)
- `GRUMultiHead(n_channels, scal_dim, ..., n_heads=K, gen='mcl', selector=True)`: plan-a-001 `GRUModelMultiAux` trunk(GRU+MLP→z) **그대로** + `z → heads: ModuleList([Linear(fc//2,3)]×K)` (각 `tanh×2cm`) + `selector: Linear(fc//2,K)`. aux F/W head carry. **n_heads=1 → KR008 과 bit-identical** (단일 head, selector 미사용 경로).
  - G2 hybrid: head_0 = KR008 가중치 load(frozen/warm), head_1..K−1 = residual-of-(KR008 pred). G3 supervised: forward 동일, 학습 시 라벨 routing. G5 MDN: head = Linear(fc//2, K*7)(mean3+logvar3+w1).
- `loss_mcl(preds_K, target, selector_logits, *, mode='wta')`: WTA = 샘플별 `argmin_k euclid(pred_k,tgt)` 승자에 combo loss + 비승자 0 (or top-2 soft) + selector CE(승자 라벨). MDN = mixture NLL. λ_sel 가중.

### §4.2 run_oof_mh.py (c2)
- plan-a-002 `run_oof.py` 복제 확장 (또는 import): `--n-heads --gen --route` flag. train_one 의 forward/loss 를 multi-head 분기. **n_heads=1 → run_oof.py 와 동일 결과(repro 검증)**.
- OOF 산출: per-sample K 후보 + selector 확률 저장 → oracle@K, realized-hit(argmax selector), per-head 사용률. `--predict-test` 시 selector-picked 제출.
- baseline KR008 npz(`results_kr008.npz`) 와 paired permutation.

### §4.3 smoke (c3)
- `tests/test_plan_a004_smoke.py`: (1) n_heads=1 → KR008 bit-identical(smoke 0.6637). (2) WTA loss finite·grad. (3) selector CE 작동. (4) K=2 1f1s1e finite + oracle@2 ≥ realized (정의상). (5) hybrid head-0 freeze 시 KR008 가중치 로드 확인.

## §5. 합격 기준 + Gate

| gate | 검사 | PASS band | severity |
|---|---|---|---|
| G0 | import+smoke + n_heads=1 KR008 repro | green | severe halt if NaN/repro 깨짐 |
| **G1_decisive** | KR010 oracle@2 (1-fold) | **≥0.69 & realized≥KR008−0.005 = PASS** / <0.69 = **KILL**(정보) | KILL halt (c5 금지) |
| G_mh | KR011/12 full OOF realized-hit | > KR008 0.6671 + p<0.05 / neutral(LB 후보) | warn |
| G_lb | best LB vs 0.6862 | 사용자 gated. ≥+0.01=목표달성 / +noise floor=inconclusive | verdict |
| G_final | results+sync+merge | 완료 | — |

- statistic: paired permutation 10000 (vs KR008), oracle@K / realized-hit / per-head 사용률 박제.
- artifact: `analysis/plan-a-004/{model_mh,losses_mcl,run_oof_mh}.py`, `results_kr0XX.json/.npz`, `plan-a-004-...results.md`.
- **NaN/Inf 0. n_heads=1 repro 불변 의무.** WTA dead-head(한 head 미사용) 모니터 — 발생 시 top-2 soft 전환.
- **CV-LB·정직 박제**: oracle headroom 없으면 KILL 정직 기록 (MoE 강행 금지). LB Δ noise floor 내면 inconclusive.

## §6. Out of scope

- 고차 물리 모델(CTRV/고차 Kalman/multi-model 융합) — 사용자 명시 배제. 기존 CV/CA seed(G4)만 허용.
- trunk(GRU/MLP) 구조 변경 — head/selector 만. (trunk 변경은 별 plan.)
- 새 입력정보/데이터 확장 — 별 방향(데이터 문제).
- G6 anchor-revival / G7 ensemble-pool 본격화 — mode_sel·roi_diag 상 열위라 후순위 ablation only.
- autonomous DACON 제출 — quota 사용자 confirm gated ([[feedback-dacon-submit-confirmation]]). best OOF realized config 만 §6 권장.
- K>4 / trunk-per-head(완전 분리 MoE) — 용량 폭증, 후속.

## §7. 참조

- `plans/plan-a-003-reflect-noise-augment.results.md` — KR008 LB 0.6862, §7 통계, 천장.
- `analysis/plan-a-002/mode_selectability.py` — 거친 모드 95% 선택가능(선택≠벽), 거친 centroid 미달 → 정밀 head 동기.
- `analysis/plan-a-002/{ceiling_verify,roi_diagnostic}.py` — 천장·ROI(다른 lever ROI 0).
- `analysis/plan-a-001/model.py` — GRU trunk + head (확장 지점).
- `analysis/plan-a-002/run_oof.py` — KR008 파이프라인 (multi-head 분기 base).
- `WORKFLOW.md §4 / §12.10` — lane mutex + sync-then-ff merge (lane a 4번째 plan).

decision-note: spec-default — plan-a-004 = lane a 4번째(사용자 지정), baseline=KR008(LB record). exp prefix=KR(KR010~, KR009 skip 후 monotonic). 핵심 = 후보 생성×선택 axis 넓게(§3) + fail-fast(G1_decisive oracle@2≥0.69 KILL). 고차물리 배제 유지(기존 CV/CA seed 만 G4). MoE 우위 미확정 → G1 이 진짜 게이트. verdict=LB gated.
