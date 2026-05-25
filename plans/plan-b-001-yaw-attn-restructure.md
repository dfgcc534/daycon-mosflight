---
plan_id: plan-b-001
version: 1.0
date: 2026-05-26 (Asia/Seoul)
status: written
track: B (notebook-inspired — anchor-selector 골격 위 LB_0.6780 노트북 lever 이식)
inspired_by:
  - plan-030 (FAIL_regression OOF 0.6294 < F0 0.6320. + 실측 §2.1/2.2/2.3 = head over-param·raw 잔차 net-zero·logit 안 sharp = attention 약점 evidence. 본 plan 이 직격)
  - plan-024 (honest ceiling 0.6387, 3-seed ensemble +0.0010 §5.14)
  - plan-022 (ANCHORS_A6 BCC K=14 codebook + build_soft_label_with_tau τ=0.001)
  - plan-020 (F0 baseline hit@1cm 0.6320, R_HIT 0.01 metric)
  - notes/LB_0.6780 코드공유.ipynb (sub_24 LB 0.6780 — yaw frame / log1p / noise feat / softhit loss / 3-seed / Tier3 / Kalman CV baseline)
code_reuse:
  - module: analysis/plan-030/residual_builder.py
    symbols: [build_residuals]
    reason: 잔차 (a)/(b) 산출 골격 carry. frame Frenet→yaw, 5-coord→직교 3-coord 로 개작.
  - module: analysis/plan-030/query_builder.py
    symbols: [build_query, extract_slim7_from_cand_ext_165]
    reason: per-anchor query 골격. 잔차(b) 35D 제거 (→ attention bias 로 이동), yaw projection 으로 개작. slim7 추출 (D.regime_anchor_prob 1 + B.cos 1 + F.2 5) 도 carry.
  - module: analysis/plan-024/seq_builder.py
    symbols: [build]
    reason: GRU input 의 **seq 95채널 × 7step** 원천 (plan-029/030 carry). 95D = anchor_vocab + torsion + frenet local 등 raw 시계열. 본 plan 미개작 (그대로 GRU input 의 95채널). 잔차(a) yaw 3 만 concat.
  - module: analysis/plan-030/head_summary.py
    symbols: [build_head_summary]
    reason: sample summary 골격. Bz/Tz drop + log1p + noise + Tier3 추가.
  - module: analysis/plan-030/model.py
    symbols: [GRUNetX2]
    reason: GRU-attention backbone. attention 재구조 (F1 bias / F2 KV / F3 head) 로 GRUNetX3 신규 파생.
  - module: analysis/plan-030/train.py
    symbols: []
    reason: 5-fold OOF 학습 루프 골격. softhit loss 추가 + 3-seed 추가 + 2-arm baseline.
  - module: analysis/plan-022/anchors.py
    symbols: [ANCHORS_A6]
    reason: K=14 BCC codebook (frame-agnostic 격자 — 재fit 불필요, R_yaw 로 decode 만).
  - module: analysis/plan-022/selector_only_model.py
    symbols: [build_soft_label_with_tau]
    reason: soft label 산식 (R_wfn → R_yaw, F0 → baseline 인자 교체).
  - module: analysis/plan-020/baseline_f0.py
    symbols: [f0_baseline, R_HIT, R_HIT_LOOSE]
    reason: F0 baseline arm + hit metric.
  - module: notes/LB_0.6780 코드공유.ipynb
    symbols: [kalman_predict]
    reason: cell 7 Kalman CV → analysis/plan-b-001/kalman_cv.py 로 port (Kalman baseline arm).
  - module: src/io.py
    symbols: [load_all_samples, load_labels]
    reason: data loader.
  - module: src/pb_0_6822/selector.py
    symbols: [stable_fold_id]
    reason: 5-fold MD5 split (plan-030 carry).
exp_ids:
  - B001_f0-baseline
  - B002_kalman-baseline
scope: |
  plan-030 cross-attention anchor-selector 골격 유지 + LB_0.6780 노트북 6-lever 이식 + plan-030 실측 attention 약점 3종 수술.
  변경 = (C1) 좌표 frame Frenet→yaw (degeneracy 제거, 직교 3축, Bz/Tz drop) · (C2) log1p · (C3) noise feature (poly2+savgol) ·
  (①) softhit metric-aligned loss (soft CE + 0.3·softhit on decoded world) · (②) 3-seed ensemble · (③) Tier3 feature ·
  (F1) 잔차(b) → attention bias (Q flatten 제거) · (F2) 잔차(a) KV 제거 (GRU input 잔존) · (F3) head broadcast bias 축소.
  baseline = 2-arm ablation (B001 F0 / B002 Kalman). 그 외 모두 동일.
  carry-fix: GRU hidden=196 2-layer · single-head attn=128 · K=14 · 50ep cosine+warmup5 AdamW lr7e-4 batch64 · 5-fold stable_fold_id.
  OUT: ④ binary flags · ⑤ (칼만은 baseline arm 으로만, 추가 aux 아님) · ⑥ 2-config · ⑦ calibration · ⑧ multi-task aux head · ⑨ tanh clip (selector hull bounded) · DACON LB 제출 · PB multi-phase training (plan-031 A-track) · climb-angle 보강 feature (ablation 보류).
lb_score: null
band: null
---

# plan-b-001 v1 — Yaw-frame Anchor-selector + Attention Restructure (notebook-levers port)

## §0. 한 줄 목적

> **plan-030 의 OOF 0.6294 FAIL 을, ㈀ 후보 frame 의 Frenet degeneracy 를 안정 yaw frame 으로 제거하고 ㈁ plan-030 이 실측한 attention 3대 약점(§2.1 head over-param / §2.2 raw 잔차 net-zero / §2.3 logit 안 sharp)을 *잔차(b) → attention bias* 재구조로 직격하며 ㈂ LB_0.6780 노트북에서 검증된 metric-aligned softhit loss · noise/log1p/Tier3 feature · 3-seed 를 이식해 회복**한다. baseline 은 F0(established floor) vs Kalman(노이즈-필터 잔차) **2-arm ablation** 으로 분리 검증. anchor-selector paradigm(K=14 BCC + soft CE) 과 backbone(GRU196 + single-head attn) 은 plan-030 carry.
>
> **핵심 가설**: plan-030 attention 이 약했던 이유는 *가장 변별력 높은 신호(잔차 b = 후보×step 정합도 텐서)를 query 에 flatten 해 묻어버리고, anchor-무관 신호(잔차 a)로 key 를 만들어* attention 이 정합을 *재발견* 해야 했기 때문. 잔차(b) 를 attention bias 로 직접 주입하면 "14후보 × seq 압축" 본연의 정합이 작동한다.

---

## §0.5 Quick Reference (autonomous loop 매 turn 읽는 section)

| 항목 | 값 |
|---|---|
| paradigm | GRU-attention anchor-selector K=14 (plan-030 carry). `final = baseline + R_wfy @ (probs @ ANCHORS_A6)` |
| **frame (C1)** | **yaw** (θ=atan2(v_y,v_x) of v_last; z=world-vertical native). Frenet `R_wfn` → `R_yaw`. degenerate normal/binormal 제거 |
| anchor codebook | ANCHORS_A6 (frame-agnostic BCC, **재fit 불필요** — R_yaw 로 decode·soft-label 만) |
| residual coords | 직교 3축 `[forward, lateral, vertical=world-z]` (plan-030 5-coord 중복 제거). Bz/Tz **drop** |
| **attention (F1/F2/F3)** | Q=static anchor 정체성(~29D, 잔차b 제거) · K=V=`kv_proj(gru_out)` (잔차a 제거) · **attn_logits += λ·Linear₃(잔차b_yaw[k,t])** · head broadcast bias 축소 |
| GRU input | seq 95 + 잔차(a) yaw 3 = **98D × 7 step** |
| attention bias | 잔차(b) yaw (B,14,7,3) → learnable Linear(3→1) → (B,14,7) additive bias (scale λ learnable) |
| **loss (①)** | `soft_CE(probs, q_τ=0.001) + 0.3·softhit(decoded_world, y)`; softhit=σ((d−0.01)/0.002).mean() |
| **ensemble (②)** | 3-seed/fold, decoded world pred 평균 |
| feature (C2/C3/③) | head_summary: log1p(long-tail) + noise(poly2+savgol) + Tier3(cum_path·rolling speed×DT·a·DT²/j·DT³ 단위통일) |
| **baseline arm** | **B001=F0** (0.6320) / **B002=Kalman CV** (σ=0.3mm/1.0). 그 외 전부 동일 |
| training (carry) | 50ep cosine+warmup5 · AdamW lr7e-4 wd1e-4 · batch64 · grad_clip1.0 · K=14 · 5-fold stable_fold_id |
| evaluation | 5-fold OOF, hit_1cm + hit_1p5cm (world frame Euclidean, R_HIT=0.01) |
| **합격 기준 (G3)** | OOF hit_1cm ≥ **0.6360** (F0+0.004, PASS). **0.6387+ STRONG**. baseline 무관, 프로젝트 reference 고정 |
| time horizon | DT=40ms, HORIZON=2 (80ms). 잔차 align baseline_pred(t) ↔ raw(t+2) (plan-030 carry) |

### G-gate sequence (합격 기준)

- G0: 인프라 — kalman_cv + yaw_frame + 개작 builder/model/train + smoke green (NaN/Inf 0, gradient flow OK).
- **G0.5 (baseline 측정, 본 plan 신규)**: B001 F0 / B002 Kalman 의 **standalone OOF hit_1cm** 박제 (no-residual = baseline 만). 모순 증거(노트북 train: Kalman 0.5964 < F0 0.6320) 확인용. severe 아님 — 정보 박제.
- G1: 1-fold smoke 양 arm hit_1cm finite + PASS threshold(>0.6290) 통과.
- G3: 5-fold OOF 양 arm hit_1cm. PASS ≥ 0.6360, STRONG 0.6387+.
- G_final: results.md 박제 + §0.5 [TODO]→[DONE] sync + best arm 식별.

### Commit chain (next-up)

| # | type | spec section | status |
|---|---|---|---|
| c0 | spec | §0 ~ §6 (본 commit) | [TODO] |
| c1 | code | `analysis/plan-b-001/kalman_cv.py` — 노트북 cell7 Kalman CV port + 외삽. spec @ §2.1 | [TODO] |
| c2 | code | `analysis/plan-b-001/yaw_frame.py` — `yaw_angle`, `to_yaw`/`from_yaw` (R_yaw, z 보존), degenerate fallback θ=0. spec @ §2.2 | [TODO] |
| c3 | code | `analysis/plan-b-001/residual_builder.py` — 잔차(a) yaw 3-coord (GRU input용) + 잔차(b) yaw 3-coord (bias용, no Q flatten). spec @ §2.3 | [TODO] |
| c4 | code | `analysis/plan-b-001/query_builder.py` — static anchor 정체성 ~29D (anchor_spec + yaw par/perp/dist + interactions + slim7, **잔차b 제거**). spec @ §2.4 | [TODO] |
| c5 | code | `analysis/plan-b-001/noise_estimator.py` (poly2+savgol) + `tier3.py` (cum_path/rolling/disp 단위통일) + `head_summary.py` (Bz/Tz drop, log1p, noise/Tier3 통합). spec @ §2.5 | [TODO] |
| c6 | code | `analysis/plan-b-001/model.py` — `GRUNetX3` (F1 잔차b bias + F2 KV=gru_out + F3 head 축소). spec @ §3 | [TODO] |
| c7 | code | `analysis/plan-b-001/train.py` + `run_oof.py` — softhit loss + 3-seed + 2-arm baseline dispatch. spec @ §4 | [TODO] |
| c8 | test | `tests/test_planb001_smoke.py` — builder shape + model finite + gradient + yaw 항등성 + kalman R-Hit assert. spec @ §5 | [TODO] |
| G0 | gate | smoke green + 기존 tests backward-compat | [TODO] |
| c9 | exp G0.5+G1 | baseline standalone OOF (G0.5) + 1-fold smoke 양 arm (G1). spec @ §5 | [TODO] |
| c10 | exp G3 | 5-fold OOF B001 + B002 → `results_g3_{f0,kalman}.json/npz`. spec @ §5 | [TODO] |
| c11 | docs | `analysis/plan-b-001/results.md` + `plans/plan-b-001-yaw-attn-restructure.results.md` (frontmatter best arm/hit/band) | [TODO] |
| G_final | gate | 위 완료 + §0.5 sync | [TODO] |

### Plan-specific severe (WORKFLOW §12.3 default 위 추가)

- `nn_numerical`: 학습 loss/grad NaN·Inf → dtype/lr/grad_clip 점검. 자동 복구 X.
- `yaw_identity_fail`: `from_yaw(to_yaw(v,θ),θ)` max err ≥ 1e-10 → 회전 정의 버그.
- `kalman_repro_fail`: Kalman CV train R-Hit 가 노트북 0.5964 와 |Δ| ≥ 0.005 (동일 데이터 가정 시) → port 버그. (데이터 split 차이로 일부 허용 — warn 우선, 0.02 초과 시 severe.)
- `softhit_no_gradient`: softhit 항이 probs→ANCHORS decode gradient path 끊김 (smoke grad norm 0) → decode 미분 경로 버그.
- `baseline_floor_regression` (warn, severe X): G0.5 에서 arm 의 standalone hit_1cm 가 frontmatter 기대(F0 0.6320 / Kalman ~0.5964) 대비 |Δ|≥0.01 → 데이터/baseline 산식 점검 + 박제 (G3 진입 차단 X).

### G3 fail 시 ablation 우선순위 (단일 G3 만으로 6-lever 분리 불가)

1. **F1 off** (잔차b 를 Q 로 복귀 = plan-030 방식) → attention 재구조 단독 효과.
2. **C1 off** (Frenet 복귀) → frame 단독 효과.
3. **① off** (softhit λ=0) → loss 정렬 단독 효과.
4. **C2/C3/③ off** (feature 묶음 복귀) → feature 단독 효과.
- 각 1-fold 비교 (G1 비용). ② (seed) 는 분산만 변경, ablation 후순위.

### Decision-note 사용 예 (자율 결정 시 commit msg 박제)

- `decision-note: spec-default — yaw θ = atan2(v_last_y, v_last_x), v_last = x[end]−x[end-1]. ‖v_xy‖<1e-9 시 θ=0 (world-x = forward) fallback`
- `decision-note: spec-default — ANCHORS_A6 codebook 값 불변 (frame-agnostic BCC). soft-label·decode 만 R_yaw 사용. 재fit X`
- `decision-note: spec-default — 잔차b bias = learnable Linear(3→1) per (k,t) + learnable scalar λ (init 1.0); softmax over step 전 attn_logits 에 가산`
- `decision-note: spec-default — softhit β=0.002, λ_softhit=0.3 (노트북 combo 가중). soft_CE 는 euclid 대체 (anchor pull 항)`
- `decision-note: spec-default — 3-seed = SEED + fold*10 + s (s=0,1,2). decoded world pred 평균 후 hit 계산`
- `decision-note: spec-default — noise = poly2(2차 다항 잔차 std) + savgol(w=5,p=2 잔차 std), log1p. LOO spline 은 cost 커서 미도입 (out-of-scope)`
- `decision-note: spec-default — head_in = attn_context(128) + Linear(sample_summary→64) + slim7. gru_hidden_last 는 KV(=gru_out)에 이미 인코딩되어 head 에서 drop (F3)`
- `decision-note: spec-default — Kalman baseline = CV model, σ_obs=0.3mm σ_proc=1.0 (노트북 grid best), +80ms 외삽`

---

## §1. Input architecture (component 결정 표)

### §1.1 paradigm 멘탈 모델 (plan-030 carry + 본 plan 수정)

attention 표준: **Q = anchor 별 *static 정체성*** (K=14 query), **K/V = sequence step 별 sample state** (T=7 keys), **attention bias = 후보×step 정합도 (잔차 b)**. 

`final = baseline + R_wfy @ (probs @ ANCHORS_A6)` — `R_wfy` = yaw→world 역회전, `baseline` ∈ {F0, Kalman}.

### §1.2 C1 — yaw frame (Frenet 교체)

- `R_yaw`: xy 평면만 heading θ 만큼 회전, z=world-vertical 보존. plan-021 `build_frenet_basis_3d` **미사용** (degenerate normal/binormal 가 plan-030 후보 의미 비일관성의 원인).
- 잔차·anchor decompose 전부 yaw. plan-030 의 world-z bolt-on (residual `Z signed`, head `Bz/Tz`) 는 yaw vertical 축에 **자연 흡수 / drop** (yaw z-row 항상 [0,0,1] → Bz/Tz 정보 0).
- 5-coord `[XY norm, Z signed, Frenet 3]` → 직교 3-coord `[forward, lateral, vertical]`.

### §1.3 F1/F2/F3 — attention 재구조 (plan-030 §2.1/2.2/2.3 직격)

| 약점 (plan-030 실측) | 수술 | 위치 |
|---|---|---|
| §2.3 logit 안 sharp — 변별력 최대 신호(잔차 b)가 Q 에 flatten 돼 step 정렬 파괴 | **F1**: 잔차(b) (14,7,3) → learnable Linear(3→1) → (14,7) attention bias 가산. Q 에서 잔차b 35D 제거 | model.forward / query_builder |
| §2.2 raw 잔차 net-zero — 잔차(a)가 GRU input·KV 이중 등장, tied KV 가 활용 못함 | **F2**: KV = `kv_proj(gru_out)` only. 잔차(a)는 GRU input(seq98)에만 잔존 | model.forward |
| §2.1 head 382D over-param → cross-fold 분산 (G1→G3 −0.0142) | **F3**: head broadcast bias 축소 (gru_last drop, summary projection 64). context 변별력은 F1 이 보강 | model.forward / head_summary |

### §1.4 input 위치별 dim 요약

| 위치 | 구성 | dim |
|---|---|---|
| GRU input / step | seq 95 + 잔차(a) yaw 3 | 98 (×7) |
| K=V / step | kv_proj(gru_out H=196) | 128 (×7) |
| Q / anchor | anchor_spec 9 + yaw par/perp/dist 3 + interactions 10 + slim7 = ~29 → q_proj | 128 (×14) |
| attention bias | Linear₃(잔차b_yaw[k,t]) · λ | (14,7) scalar |
| head / anchor | attn_context 128 + proj(sample_summary)64 + slim7 = ~199 | → head_hidden 256 → 1 |

(정확 micro-dim 은 executor default — §0.5 decision-note 범위. dim 폭증 회피가 F3 의도.)

---

## §2. Builders (server 작업 — c1~c5)

### §2.1 kalman_cv.py (c1)
노트북 cell7 `kalman_predict` 1:1 port. CV model, F/F_pred/Q 행렬, 축 독립 벡터화, σ_obs=0.3mm σ_proc=1.0, +80ms(T_PRED=0.08) 외삽. `kalman_train`/`kalman_test` (N,3). train R-Hit assert (kalman_repro_fail gate).

### §2.2 yaw_frame.py (c2)
`yaw_angle(v)=atan2(v[:,1],v[:,0])`; `to_yaw(vec,θ)` (forward/lateral/vert), `from_yaw(vec,θ)` (역). 항등성 smoke (yaw_identity_fail). degenerate fallback θ=0.

### §2.3 residual_builder.py (c3)
plan-030 `build_residuals` 개작. baseline 인자화 (F0 또는 Kalman). 잔차(a)=raw(t+2)−baseline(t) → yaw 3-coord (N,7,3, GRU input용). 잔차(b)=raw(t+2)−anchor_world_k(t), anchor_world_k(t)=baseline(t)+R_wfy@ANCHORS_A6[k] (R_wfy = **단일 v_last 기준 yaw 역회전, step-invariant** — §0.5 θ=atan2(v_last) 정합; step별 θ(t) 아님) → yaw 3-coord (N,14,7,3, bias용). **step align**: step i=0..6 의 wall-clock t=(−6+i); 잔차는 raw(t+2) 필요 → i=5,6 (t=−1,0) 은 raw(+1,+2) 미관측이라 **zero-pad** (plan-030 carry).

### §2.4 query_builder.py (c4)
static anchor 정체성만 (anchor_spec 9 + yaw par/perp/dist 3 + interactions 10 + slim7 7 = **29D**) — **모두 plan-030 query_builder carry**: anchor_spec=anchor world coord 파생 9, interactions=anchor·(res/v/a) dot+sign+physics 10쌍, slim7=`extract_slim7_from_cand_ext_165` (D.regime_anchor_prob 1 + B.cos 1 + F.2 5). 본 plan 개작 = **잔차(b) 35D 제거**(F1) + par/perp/dist 등 projection 을 yaw 로 재계산. (anchor_spec/interactions 의 정확 산식은 pinned 모듈 그대로 재사용.)

### §2.5 noise_estimator.py + tier3.py + head_summary.py (c5)
- **noise (C3)** — 관측 궤적 X (N, T_obs=11, 3), 시간축 t=arange(11):
  - `poly2`: 각 축 j 위치를 t 에 대해 **2차 다항 fit** (`np.polyfit(t, X[:,:,j], deg=2)` 후 평가) → residual = X − fit; 3축 residual 합친 std → 스칼라/샘플.
  - `savgol`: `savgol_filter(X, window_length=5, polyorder=2, axis=time)` → residual = X − smoothed; 3축 평균 std → 스칼라/샘플.
  - 둘 다 `log1p`. (LOO spline 미도입 — §6.)
- **tier3 (③)** — 전부 **world frame, m 단위** 통일 (notebook 방식). `disp=diff(X,axis=time)` (N,10,3), `v=disp/DT`, `a=diff(v)/DT`, `j=diff(a)/DT` (DT=0.040):
  - `cum_path` = Σ‖disp‖ (누적 경로장, m).
  - `roll_mean`,`roll_std` = ‖v‖ 의 window=3 rolling mean × DT (m) 의 mean·std (2 스칼라).
  - `a_unit`,`j_unit` = `‖a·DT²‖` mean, `‖j·DT³‖` mean (m, 2 스칼라).
  - 출력 = 5 스칼라, long-tail 은 log1p.
- **head_summary**: plan-030 carry 에서 **Bz/Tz drop**, macro/A1/A6/A10/A12/plan-021 long-tail magnitude 에 **log1p**(C2), 위 noise(2)·tier3(5) 통합. sample_summary → Linear proj 64 (F3 compact).

---

## §3. Model (c6 — GRUNetX3)

plan-030 `GRUNetX2` 파생. 변경점:
1. **F1**: `forward` 에 `residual_b_yaw (B,K,T,3)` 인자 추가. `bias = self.resb_proj(residual_b_yaw).squeeze(-1) * self.lambda_bias` (Linear(3,1), λ learnable). `attn_logits = einsum(q,kv)/√d + bias`.
2. **F2**: `kv = kv_proj(gru_out)` (잔차a 인자 제거). seq_dim=98.
3. **F3**: head_in = `cat[attn_context, sample_proj(64), slim7]`; `gru_hidden_last` drop. head_hidden 256. head_mlp → (B,14) anchor logit → **`softmax(dim=K=14)` → probs** (= selector 분포; attention 의 step-softmax(dim=T=7) 와 별개 2-축 softmax).
4. decode: `R_wfn` → `R_wfy` (yaw→world), `baseline_pred` 인자 (F0/Kalman). ANCHORS_A6 buffer 그대로. **decode 경로(`probs @ ANCHORS_A6` → `from_yaw` → `+ baseline_pred`) 전체 torch tensor 연산** (from_yaw torch mirror, numpy 금지) — softhit gradient path 보존.
smoke: shape/finite/probs 분포/gradient flow + softhit decode gradient (softhit_no_gradient).

---

## §4. Training (c7)

- loss = `soft_CE(probs, q_τ) + 0.3·softhit(decoded_world, y)`.
  - `soft_CE = −Σ_k q_τ[k]·log(probs[k])`, batch **mean** reduction.
  - `q_τ` = build_soft_label_with_tau(gt, R_yaw, baseline, ANCHORS_A6, τ=0.001) → 반환 = (N, K=14) **확률분포 (행합=1)**. 산식: 각 샘플의 yaw-frame 잔차 `to_yaw(gt−baseline, θ)` 와 14 anchor 간 거리 `d_k` → `q = softmax(−d_k / τ)` (τ=0.001 = sharpness, 작을수록 nearest anchor one-hot 근접).
  - `softhit = σ((‖decoded_world − y‖ − 0.01)/0.002).mean()`. gradient = probs→(probs@ANCHORS)→from_yaw→decoded → hit metric 직접 최적화.
- 3-seed (②): fold 별 s=0,1,2, decoded world pred 평균 → OOF/test.
- 2-arm: B001 baseline=F0, B002 baseline=Kalman. config dispatch (`--baseline {f0,kalman}`). 그 외 hyperparam·seed·fold 동일.
- carry: 50ep cosine+warmup5, AdamW lr7e-4 wd1e-4, batch64, grad_clip1.0, 5-fold stable_fold_id.

---

## §5. Gates / Server 작업 순서

1. c1~c5 builder + c6 model + c7 train + c8 test → **G0** (smoke green, backward-compat).
2. **G0.5**: 양 arm standalone baseline OOF hit_1cm 박제 (residual=0). 모순 증거 확인 (정보).
3. **G1**: 1-fold smoke 양 arm hit_1cm finite + >0.6290.
4. **G3**: 5-fold OOF 양 arm. PASS ≥0.6360, STRONG 0.6387+. fail → §0.5 ablation 우선순위.
5. **G_final**: results.md (양 arm + best 식별) + §0.5 sync.

서버는 위 외 작업 안 함 (DACON 제출·PB training·climb feature 미수행).

---

## §6. Out of scope

④ binary flags · ⑤ Kalman aux(baseline arm 으로만) · ⑥ 2-config · ⑦ calibration · ⑧ multi-task aux head · ⑨ tanh clip · DACON LB 제출 · PB multi-phase training(plan-031 A-track) · climb-angle/v_z 보강 feature(ablation 보류) · ANCHORS 재fit(frame-agnostic 라 불필요).

## §7. 참조

- `plans/plan-030-*.md` + `.results.md` (FAIL 0.6294, §2.1/2.2/2.3 attention 실측 약점)
- `analysis/plan-029/paradigm_root_cause.md` (training procedure = A-track carrier; 본 plan 은 input/frame/attention axis = B-track)
- `notes/LB_0.6780 코드공유.ipynb` (lever 출처)
- `analysis/plan-022/anchors.py`, `analysis/plan-020/baseline_f0.py`
