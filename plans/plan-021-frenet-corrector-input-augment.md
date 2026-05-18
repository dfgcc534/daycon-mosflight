---
plan_id: 021
version: 1
date: 2026-05-18 (Asia/Seoul)
status: draft
based_on:
  - 020 (F0 baseline 0.6320 / 0.8033 + 5-fold stable_fold_id MD5 + C05 per-regime winner finding)
  - 006 (F0 산식 frenet_par120_perp_neg020 — d1=1.98 / par=1.20 / perp=-0.20)
  - 004 (사용자 명시 carry: recent_temporal_physics_features + observation_environment_features — LGBM 전용 9D)
followed_by: []
scope: F0 잔차 corrector — input augment 4 lever (① Frenet trajectory ② F0 residual sequence Frenet ③ F0 soft hit sequence ④ soft label) + dual head (7-anchor Frenet-orthogonal classifier + 7×3 reg offset) + 2 sub-exp 독립 (A: LGBM + 9D macro stat + 27D EWMA / B: 단일방향 GRU). pass criterion paired Δ ≥ +0.005 둘 다. 27-pool 통합 / LB / BMA / dacon-submit = out-of-scope.
exp_ids:
  - Z021_A_lgbm
  - Z021_B_gru
lb_score: null
band: null
---

# plan-021 v1 — Frenet Corrector with Input Augment (Frenet + F0 residual + soft hit + soft label, dual head)

## §0. 한 줄 목적

> **F0 의 paired Δ ≥ +0.005 *둘 다* (hit@1cm + hit@1.5cm) 통과** = corrector NN/tree 의 *input MI 부족* root cause 를 4 lever input augment 로 직접 공략. 4 lever:
> 1. **Frenet input** — trajectory 를 Frenet 좌표계 변환 (산식 §6.1.1: `R_wfn^T @ (traj − origin)`, R_wfn columns = [t̂, n̂, b̂]) → translation + rotation invariance
> 2. **F0 residual sequence (Frenet)** — 7 past sub-window 의 F0 잔차 `(pred_F0_t − actual_t) @ R_wfn`
> 3. **F0 soft hit sequence** — 7 past sub-window 의 `sigmoid((R − d)/τ)` at R ∈ {0.01, 0.015}
> 4. **soft label** — classifier target = `softmax(−dist(ANCHORS_FRENET, residual_true)/τ_cls)` (7-class soft prob, hard one-hot 회피)
>
> 위 4 lever 위에 **dual head** (7-anchor classifier + 7×3 regression offset) — Frenet-orthogonal anchor codebook `{origin, ±t̂, ±n̂, ±b̂} × 0.005m`. classifier soft CE + regression smooth_hit_loss 결합.
>
> **2 sub-exp 독립 측정** (ensemble 아님, 같은 input/output spec 위 model 만 다름):
> - **A: LGBM** — flat 170D (= 134D 공통 + 9D macro statistic + 27D EWMA)
> - **B: 단일방향 GRU** — sequence (11, 9) + flat 35D = 134D 공통
>
> **pass criterion**: A 또는 B 중 *최소 1 sub-exp* 가 paired Δ ≥ +0.005 *둘 다* 통과 = G3 PASS.
>
> **out-of-scope**: plan-020 의 C05 per-regime F0 와의 ensemble / 27-pool 통합 / LB 측정 / DACON submit / BMA. 전부 follow-up.

---

## §0.5 Quick Reference (autonomous loop 매 turn 읽는 section)

### 합격 기준 (G-gate sequence)

- **G0**: 4 module (build_input / dual_head_model / run_oof / cma_es_fit_carry) import + smoke + tests green. F0 baseline 0.6320 / 0.8033 reproduce sanity (plan-020 carry). 위반 시 `infra_drift` severe.
- **G1**: F0 baseline 5-fold concat OOF — hit@1cm ∈ [0.6315, 0.6325] AND hit@1.5cm ∈ [0.8028, 0.8038] (plan-020 carry exact). 위반 시 `f0_reproduce_drift` severe.
- **G2.A**: sub-exp A LGBM 5-fold OOF 완료. metric finite. 위반 시 `lgbm_numerical` severe.
- **G2.B**: sub-exp B 단일방향 GRU 5-fold OOF 완료. metric finite + val_hit > 0.10 + train_hit − val_hit < 0.10. 위반 시 `gru_no_signal` / `gru_overfit` warn.
- **G3 (paradigm-level)**: A 또는 B 중 **≥ 1 sub-exp 가 paired Δ ≥ +0.005 *둘 다*** 통과 → PASS. 0 통과 = `all_negative` warn (severe X, negative finding 박제 후 G_final 진입).
- **G_final**: results.md + best 박제 + follow-up plan 후보 ≥ 2건 박제 + 3-file frontmatter sync.

### G-gates

- G0: STAGE 0 인프라 [TODO]
- G1: STAGE 1 F0 baseline reproduce [TODO]
- G2.A: STAGE 2 sub-exp A LGBM 측정 [TODO]
- G2.B: STAGE 3 sub-exp B GRU 측정 [TODO]
- G3: STAGE 4 paradigm-level finding [TODO]
- G_final: STAGE 5 results + 3-file sync [TODO]

### Commit chain (next-up)

| # | type | spec section | status |
|---|---|---|---|
| c1 | docs | `plans/plan-021-frenet-corrector-input-augment.md` 본문 v1 작성 (v1.1 plan-review-master 5-iter 자동 fix BLOCKER 0 도달, A/B/C 모두 "잘 드러남" 안정 수렴) | [DONE] |
| c2 | code | `analysis/plan-021/build_input.py` (Frenet basis + L1/L2/L4/L5/L6 + ANCHORS_FRENET + soft label, smoke 8/8 pass — Frenet orthonormality 2.7e-7 / roundtrip 9.1e-9 / F0 sub-window 호환 / 170D concat) | [DONE] |
| c3 | code | `analysis/plan-021/dual_head_model.py` (LgbmDualHead + GRUDualHead + soft_ce_loss + smooth_hit_loss + tau_for_epoch, smoke LGBM train hit 0.875 / GRU forward+backward OK / schedule 정확) | [DONE] |
| c4 | code | `analysis/plan-021/run_oof.py` (5-fold OOF runner, sub-exp A/B dispatch + F0 baseline anchor, G1 reproduce sanity ✓) | [DONE] |
| c5 | test | `tests/test_plan021_smoke.py` (11 pytest: import + Frenet orth + roundtrip + L1/L2/L4/L5/L6 shape+finite + soft_label sum=1 + LGBM/GRU forward + losses+backward + tau schedule + G1 carry) | [DONE] |
| G0 | gate | smoke + tests green — 11/11 pytest pass ✓ | [DONE] |
| c6 | exp G1 | F0 baseline 5-fold OOF reproduce → exact 0.6320 / 0.8033 (plan-020 carry 정확 일치, 0.8s CPU). `analysis/plan-021/baseline_carry.json` 박제. | [DONE] |
| G1 | gate | F0 hit@1cm = **0.6320** ∈ [0.6315, 0.6325] AND hit@1.5cm = **0.8033** ∈ [0.8028, 0.8038] ✓ | [DONE] |
| c7 | exp G2.A | A LGBM 5-fold OOF (v1.3 conceptual fix 후): hit@1cm 0.6488 (Δ +0.0168 ✓), hit@1.5cm 0.8070 (Δ +0.0037 ✗ < +0.005). pass_both=False (partial). CPU 334s. | [DONE] |
| G2.A | gate | A metric finite ✓ (lgbm_numerical 미발동) | [DONE] |
| c8 | exp G2.B | sub-exp B GRU 5-fold OOF → `analysis/plan-021/results_gru.{json,md}` | [TODO] |
| G2.B | gate | B metric finite + val_hit > 0.10 + overfit guard | [TODO] |
| c9 | analysis | paradigm-level finding (A vs B 비교 + 4 lever marginal 가치) → `analysis/plan-021/paradigm_analysis.{json,md}` | [TODO] |
| G3 | gate | ≥ 1 sub-exp paired Δ ≥ +0.005 둘 다 | [TODO] |
| c10 | docs | results.md + frontmatter sync + follow-up plan 후보 ≥ 2 | [TODO] |
| G_final | gate | 3-file sync + §0.5 [TODO]→[DONE] + follow-up ≥ 2 박제 | [TODO] |

### Plan-specific severe (WORKFLOW.md §12.3 default 위 추가분)

- `f0_reproduce_drift`: G1 reproduce 가 plan-020 hard evidence 0.6320 / 0.8033 ±0.0005 밖. 추출/fold split/F0 산식 carry 버그 의심 → halt.
- `lgbm_numerical`: A LGBM classifier 또는 regression 출력 NaN/Inf. soft label CE 또는 anchor target 산출 버그 의심.
- `gru_no_signal`: B GRU **val_hit@1cm (5-fold concat OOF) < 0.10** (= F0 baseline 0.6320 의 1/6, "최소 학습 시그널 floor" 임계 — 7-class random uniform `1/7 ≈ 0.143` 과는 다른 task-specific floor). normalization / loss 버그 의심.
- `gru_overfit`: B GRU train_hit − val_hit > 0.10. regularization 부족.
- `frenet_basis_degenerate`: ‖v_last‖ < 1e-9 또는 ‖a_⊥‖ < 1e-9 sample 비율 > 5% — Frenet basis 정의 불가 sample 다수. `a_⊥ = a − (a · t̂) · t̂` (= 가속도 a 의 tangent-수직 성분, perpendicular 방향). fallback (world frame use: `R_wfn ← I_3`, origin ← x[end_idx]) 또는 sample 제외 박제.
- `soft_label_collapse`: classifier 가 anchor 0 (origin) 만 selecting (π_0 평균 > 0.95). soft CE temperature 너무 sharp 의심.
- `all_negative`: A + B 모두 paired Δ < +0.005 *둘 다*. → `all_negative` warn 박제 + G_final 진입 (paradigm-level evidence — input augment lever 의 한계).

### Plan-specific paths (WORKFLOW.md §12.5/§12.6)

- whitelist 추가:
  - `analysis/plan-021/**`
  - `tests/test_plan021_smoke.py`
  - `runs/baseline/Z021_*/` (GRU ckpt — `.gitignore` 적용)
- blacklist 추가:
  - plan-001~020 산출 자동 변경 (`runs/baseline/{B,S,R,P,D,E,F,H,Z020}*/**`, `analysis/plan-{001..020}/**`)
- 참조 (read-only):
  - `analysis/plan-020/baseline_oof.{json,md}` — F0 0.6320 / 0.8033 carry source
  - `analysis/plan-020/baseline_f0.py` — F0 산식 (numpy + torch) carry
  - `src/pb_0_6822/selector.py` L185 stable_fold_id (fold split carry)
  - `src/pb_0_6822/selector.py` L297-360 recent_temporal_physics_features + observation_environment_features (사용자 명시 carry, LGBM 전용)
  - `src/io.py` load_all_samples + load_labels

### Decision-note 사용 예 (자율 결정 시 commit msg 박제)

- `decision-note: spec-default — Frenet basis = build_frenet_basis_3d(traj, end_idx=10), columns = [t̂, n̂, b̂]. ‖v‖<1e-9 fallback = world frame use.`
- `decision-note: spec-default — F0 residual sub-window = 7 past windows (t=4..10), F0 산식 default coef (1.98, 1.20, -0.20) + horizon=2 step.`
- `decision-note: spec-default — F0 soft hit τ_loss = 0.001m (1mm, anchor 0.5cm 의 1/5 scale, plan-020 schedule final phase 동일).`
- `decision-note: spec-default — anchor = Frenet-orthogonal 7-Way × 0.005m radius (origin + ±t̂/±n̂/±b̂). 단일 codebook 고정 (bake-off X).`
- `decision-note: spec-default — soft label τ_cls = 0.001m (1mm, anchor scale 의 1/5).`
- `decision-note: spec-default — LGBM: n_estimators=500, lr=0.05, num_leaves=63, multi-output 7-class softmax classifier + 21 regressor.`
- `decision-note: spec-default — GRU: hidden=64, layers=1, bidir=False, dropout=0.1, epochs=50, batch=256, Adam lr=1e-3.`
- `decision-note: spec-default — multi-seed = 3 seeds [20260518, 20260519, 20260520], best-on-train selection. **scope: GRU (sub-exp B) 전용** — LGBM (sub-exp A) 는 결정적이라 single seed (§6.2 spec-default).`
- `decision-note: spec-default — 2 sub-exp 독립 (ensemble 아님). 결과 비교는 paradigm-level finding 만.`

---

## §1. 배경

### §1.1 plan-020 의 finding 과 본 plan 의 의도

| plan-020 결과 | 본 plan-021 응답 |
|---|---|
| **C05 per-regime F0 단독 PASS** (Δ +0.0183 / +0.0053) | C05 는 *coefficient discrete partition* 으로 sample 분할 — corrector 와 결합 시 sample 수 부담 |
| N1/N2/N5 NN coef regression 모두 paired Δ 둘 다 fail | NN 의 input MI 부족 root cause (plan-014/016 5.4% recovery ceiling 과 동일) |
| paradigm-level 결론: F0 산식 위 *계수 lever* 가 본질 | 본 plan = **input MI 강화** lever 로 다른 paradigm 시도 |

### §1.2 사용자 narrative — 4 lever 의 직접 input MI 공략

plan-014/016/017/020 의 4 plan 모두 NN input 에 *F0 의 sample-conditional 정보* echo 안 함:
- world frame raw trajectory 만 input
- F0 의 prediction / residual / hit accuracy 가 input 측 부재
- → NN 이 "이 sample 에서 F0 가 어디로 빗나가는지" 학습 source 없음

본 plan 의 4 lever = **F0 의 sample-conditional 정보를 input 에 직접 echo**:
1. Frenet trajectory (frame 정렬, sample efficiency ↑)
2. F0 residual sequence (방향 + 크기 vector, 7 past sub-window)
3. F0 soft hit sequence (정확도 graded scalar, 7 × 2 R)
4. soft label classifier target (anchor 거리 기반 soft prob, classifier collapse 회피)

### §1.3 dual head (anchor classifier + reg offset)

mean-regression trap (single regression head 가 residual 평균에 끌려가는 함정) 회피:
- 7 anchor Frenet-orthogonal (origin + ±t̂/±n̂/±b̂ × 0.005m)
- classifier 가 "어느 anchor 방향" 선택 (soft prob)
- regression head 가 "cluster 내 ±0.5cm 미세 offset" 회귀

본 plan-021 의 input 강화로 classifier collapse 위험 ↓ 기대 (input 에 F0-relative signal 풍부).

### §1.4 2 sub-exp 독립 (LGBM vs 단일방향 GRU)

| sub-exp | 본질 | 검증 가설 |
|---|---|---|
| **A LGBM** | tree + flat 170D (macro stat + EWMA 추가) | "통계 aggregates + EWMA + tree 가 134D 위 더 효과적인가" |
| **B 단일방향 GRU** | sequence (11, 9) + flat 35D | "sequence learning 이 134D 위 효과적인가" |

두 sub-exp 의 결과 = *같은 input 강화 lever 위 model 의 paradigm-level 비교*. ensemble 아님 (사용자 명시).

---

## §2. Scope (명시적)

### §2.1 In-scope

| 항목 | 값 |
|---|---|
| sub-exp 개수 | **2** (A LGBM + B 단일방향 GRU) |
| Fold split | `stable_fold_id(str(sample_id), 5)` (plan-020 carry, MD5) |
| Input lever | 4 (Frenet + F0 residual + F0 soft hit + soft label) |
| Head | dual (7-anchor classifier soft + 7×3 reg offset, ±0.5cm) |
| 평가 metric | hit@1cm + hit@1.5cm, sample-level paired Δ vs F0 |
| Pass criterion | paired Δ ≥ +0.005 *둘 다*, A 또는 B 중 ≥ 1 PASS = G3 PASS |
| Multi-seed | 3 seeds [20260518..20260520], best-on-train (GRU 만 — LGBM 은 single seed) |
| 결과 박제 | A + B 각 hit/Δ + paradigm-level finding (4 lever 의 marginal 가치) |

### §2.2 Out-of-scope (절대 안 함)

| 항목 | 이유 |
|---|---|
| plan-020 의 C05 per-regime F0 와 ensemble | 본 plan = independent corrector paradigm 측정 우선 |
| LB 제출 (dacon-submit) | DACON 5회 quota 보존, follow-up plan |
| BMA / IMM mixture | 단독 측정 완료 후 conditional |
| plan-004 27-pool 통합 | follow-up plan-021.1 또는 plan-022 |
| A + B ensemble | 사용자 명시: 2 sub-exp 독립, ensemble 안 함 |
| 다른 anchor codebook (Absolute / K-Means) | 사용자 명시: Frenet-orthogonal 단일 codebook |
| anchor radius / K 변경 | 단일 0.005m × 7-Way 고정 |

---

## §3. 사전 등록 (Pre-registration)

### §3.1 Fold split

| 분할 | 값 |
|---|---|
| folds | 5 |
| fold 할당 | `stable_fold_id(str(sample_id), 5)` (plan-020 carry, MD5 32-bit prefix mod 5) |
| seed | fold split deterministic (no seed) |
| F0 baseline OOF | plan-020 baseline_oof.json 의 *집계 metric* 만 carry (0.6320 / 0.8033 sanity). **paired Δ 산출용 per-sample F0 pred 는 run-time 에 `bf.f0_baseline(X, end_idx=10)` 호출로 재계산** (deterministic numpy, < 1s) — baseline_oof.json 에는 array 박제 X. |

### §3.2 합격 기준 (정량)

- **G0**: 4 모듈 import + smoke + tests green
- **G1**: F0 reproduce hit@1cm ∈ [0.6315, 0.6325] AND hit@1.5cm ∈ [0.8028, 0.8038]
- **G2.A**: A LGBM 5-fold OOF hit metric finite
- **G2.B**: B GRU 5-fold OOF hit metric finite + val_hit > 0.10 + train_hit − val_hit < 0.10
- **G3**: A 또는 B 중 ≥ 1 sub-exp paired Δ ≥ +0.005 *둘 다*
  - 0 통과 시 → `all_negative` warn 박제 후 G_final 직진

### §3.3 평가 점수

| metric | 식 | 비교 |
|---|---|---|
| hit@1cm | `mean(‖final_world − gt‖₂ ≤ 0.01)` | F0 baseline 0.6320 |
| hit@1.5cm | `mean(‖final_world − gt‖₂ ≤ 0.015)` | F0 baseline 0.8033 |
| paired Δ | sample-level: `mean_i(1{‖pred_cand_i − gt_i‖ ≤ R} − 1{‖pred_F0_i − gt_i‖ ≤ R})`. 5-fold concat OOF. **`pred_cand_i` = sub-exp 단수 (A 또는 B 각각)** — ensemble 아님. A/B 각자 별 paired Δ 산출. | +0.005 임계 적용 |
| fold variance | per-fold metric (5 개) std | **informational only** (gate binding 없음 — `nn_overfit`/`gru_overfit` warn 은 train-val gap 으로 별도 측정 §3.2 G2.B) |

### §3.4 Anchor 정의 (단일 codebook, 사용자 명시 — bake-off 안 함)

```python
ANCHORS_FRENET = np.array([
    (  0.000,   0.000,   0.000),  # ch 0: origin (= F0 그대로)
    (+ 0.005,   0.000,   0.000),  # ch 1: +t̂  (앞 0.5cm)
    (- 0.005,   0.000,   0.000),  # ch 2: -t̂  (뒤 0.5cm)
    (  0.000, + 0.005,   0.000),  # ch 3: +n̂  (회전 안쪽 0.5cm)
    (  0.000, - 0.005,   0.000),  # ch 4: -n̂  (회전 바깥 0.5cm)
    (  0.000,   0.000, + 0.005),  # ch 5: +b̂  (평면 위 0.5cm)
    (  0.000,   0.000, - 0.005),  # ch 6: -b̂  (평면 아래 0.5cm)
], dtype=np.float32)  # (7, 3) — Frenet coords
```

각 anchor 는 model 의 **output channel index** 로 hardcoded (input 에 박제 안 함 — single codebook 고정이라 redundant).

---

## §4. STAGE 0 — 인프라 (G0)

### §4.1 모듈 layout

```
analysis/plan-021/
├── build_input.py              # 4 lever input 빌드 + Frenet basis + macro stat + EWMA
├── dual_head_model.py          # LGBM dual head + GRU dual head + soft CE / smooth_hit loss
├── run_oof.py                  # 5-fold OOF runner (sub-exp A/B dispatch)
├── paradigm_analysis.py        # c9 G3 — A vs B + 4 lever marginal
├── results_lgbm.{json,md}
├── results_gru.{json,md}
├── paradigm_analysis.{json,md}
└── results.md                  # G_final synthesis
```

### §4.2 module top-level export 보장 (smoke test lock-in)

| symbol | module | type |
|---|---|---|
| `build_frenet_basis_3d` | build_input | `Callable[[np.ndarray, int], np.ndarray]` ((N,T,3), end_idx → (N,3,3) = stack [t̂, n̂, b̂] columns). 산식 §4.2.1 참조. |
| `to_frenet` | build_input | `Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]` (vec, R, origin → frenet vec) |
| `build_input_common` | build_input | `Callable[[np.ndarray, Callable], dict]` (X (N,T,3), `f0_baseline_fn` injected → {"L1": (N,11,9), "L2": (N,7,3), "L4": (N,7,2), **"R_wfn": (N,3,3), "origin": (N,3), "pred_F0_world": (N,3)**}). origin (= x[end_idx]) 은 L1 의 Frenet trajectory invariance 용, **pred_F0_world (= F0 의 80ms 미래 예측, v1.3 박제) 은 anchor codebook 의 reference 위치**. LGBM extras (L5/L6 36D) 는 `build_input_lgbm_extra` 별도, `run_oof_lgbm` 의 `X_lgbm = concat([L1.reshape(N,99), L2.reshape(N,21), L4.reshape(N,14), lgbm_extra(36)], axis=1)` 총 170D. |
| `build_input_lgbm_extra` | build_input | `Callable[[np.ndarray], np.ndarray]` (X → (N, 9+27=36) = macro stat 9D + EWMA 27D) |
| `build_soft_label` | build_input | `Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]` (gt (N,3), R_wfn (N,3,3), **pred_F0_world (N,3)** → (N, 7) soft prob). **v1.3 박제**: anchor reference = pred_F0_world (= F0 의 80ms 미래 예측). 산식: `residual_true_frenet = einsum('nij,nj->ni', R_wfn.transpose(0,2,1), gt - pred_F0_world); dist = ‖ANCHORS_FRENET[None] − residual_true_frenet[:,None]‖; q = softmax(−dist / 0.001, axis=1)`. |
| `ANCHORS_FRENET` | build_input | `np.ndarray` shape (7, 3) |
| `LgbmDualHead` | dual_head_model | sklearn-style class — fit / predict_proba / predict_offset |
| `GRUDualHead` | dual_head_model | `nn.Module` — forward returns (logits, reg_offset) |
| `soft_ce_loss` | dual_head_model | `Callable[[Tensor, Tensor], Tensor]` (logits (B, K), q (B, K) soft prob → scalar loss = `-(q * log_softmax(logits)).sum(dim=1).mean()`) |
| `smooth_hit_loss` | dual_head_model | `Callable[[Tensor, Tensor, float, bool], Tensor]` (pred (B, 3), gt (B, 3), tau, use_boundary → scalar loss). 산식 §7.3.1 참조. |
| `run_oof_lgbm`, `run_oof_gru` | run_oof | `Callable` |

→ 위 export 중 하나라도 AttributeError 시 G0 `infra_drift` severe.

### §4.2.1 `build_frenet_basis_3d` 산식 (self-contained)

```python
def build_frenet_basis_3d(x: np.ndarray, end_idx: int) -> np.ndarray:
    """x shape (N, T, 3), end_idx = T-1. returns (N, 3, 3) — columns = [t̂, n̂, b̂].

    산식 (sample-wise):
      v_last = x[:, end_idx] - x[:, end_idx-1]                          # (N, 3)
      v_prev = x[:, end_idx-1] - x[:, end_idx-2]
      a      = v_last - v_prev                                           # (N, 3)
      
      t̂ = v_last / max(‖v_last‖, 1e-9)                                   # tangent (앞)
      a_⊥ = a - (a · t̂) · t̂                                              # tangent-수직 성분
      n̂ = a_⊥ / max(‖a_⊥‖, 1e-9)                                         # normal (회전 안쪽)
      b̂ = t̂ × n̂                                                          # binormal (위)
      
      return stack([t̂, n̂, b̂], axis=-1)  # (N, 3, 3) — columns = basis vectors

    Fallback (frenet_basis_degenerate):
      - ‖v_last‖ < 1e-9 → R_wfn = I_3 (world frame use)
      - ‖a_⊥‖ < 1e-9 (직선 운동) → 임의 n̂ 선택:
          z_world = (0, 0, 1)
          if |t̂ · z_world| > 0.99:   z_world ← (1, 0, 0)    # collinearity 임계 정량
          n̂ = (z_world − (z_world · t̂) · t̂) / ‖...‖
          b̂ = t̂ × n̂
      - 둘 다 발생 sample 비율 > 5% → severe escalate
    """
```

산식 invariant (smoke test catch):
- `R @ R.T ≈ I_3` (orthonormality, ±1e-6)
- `det(R) ≈ +1` (right-handed)
- `R[:, :, 0] ≈ v_last / ‖v_last‖` (t̂ column 0)

### §4.3 F0 산식 carry (plan-020 baseline_f0.py reuse)

```python
# analysis/plan-021/build_input.py 상단
import sys
from pathlib import Path
import importlib.util

_PLAN020_DIR = Path(__file__).resolve().parent.parent / "plan-020"
spec = importlib.util.spec_from_file_location("baseline_f0", _PLAN020_DIR / "baseline_f0.py")
bf = importlib.util.module_from_spec(spec); spec.loader.exec_module(bf)
# bf.f0_baseline, bf.D1, bf.PAR, bf.PERP, bf.R_HIT, bf.R_HIT_LOOSE
```

plan-020 의 `f0_baseline(x, end_idx)` 그대로 사용. 산식 변경 X (paired Δ anchor 일관성).

### §4.4 tests (c5)

- 4 모듈 import (AttributeError 0건)
- `build_frenet_basis_3d` shape (N, 3, 3) + orthonormality (R @ R.T ≈ I)
- `to_frenet` round-trip (`world → frenet → world` 일치)
- `build_input_common` 출력 shape 일치 ((N,11,9), (N,7,3), (N,7,2))
- `build_input_lgbm_extra` 출력 shape (N, 36) + finite
- `build_soft_label` 출력 (N, 7) sum=1
- F0 reproduce sanity (plan-020 baseline_oof.json 와 비교)

---

## §5. STAGE 1 — F0 baseline reproduce (c6, G1)

### §5.1 실행

plan-020 의 baseline_oof.json 그대로 carry (재실행 불필요):

```python
import json
baseline = json.loads((REPO_ROOT / "analysis/plan-020/baseline_oof.json").read_text())
f0 = baseline["f0_baseline"]
assert 0.6315 <= f0["hit_1cm_5fold_concat"] <= 0.6325
assert 0.8028 <= f0["hit_1.5cm_5fold_concat"] <= 0.8038
```

산출: `analysis/plan-021/baseline_carry.json` (plan-020 결과 + 본 plan dataset 매칭 검증).

### §5.2 G1 합격 (자동)

- plan-020 baseline_oof.json 의 metric 그대로 통과 (이미 검증됨)
- 위반 시 `f0_reproduce_drift` severe (= plan-020 environment drift)

---

## §6. STAGE 2 — Sub-exp A LGBM (c7, G2.A)

### §6.1 Input spec (170D)

| 채널 | dim | source | sample-wise? |
|---|---|---|---|
| L1 Frenet trajectory | 11 × 9 = **99** | per-step `[p, v, a]` Frenet, 산식 §6.1.1 | ✓ |
| L2 F0 residual sequence | 7 × 3 = **21** | 7 past sub-window F0 잔차 Frenet, 산식 §6.1.2 | ✓ |
| L4 F0 soft hit sequence | 7 × 2 = **14** | `sigmoid((R − d_t)/τ_loss)` at R={0.01, 0.015}, τ_loss=0.001m, 산식 §6.1.3 | ✓ |
| L5 macro statistic | **9** | 6D recent_temporal + 3D observation_env (self-contained 식 §6.1.4) | ✓ |
| L6 EWMA | 9 × 3 = **27** | Frenet `[p, v, a]` per step → EWMA last value at α ∈ {0.1, 0.3, 0.5}, 식 §6.1 ewma_last | ✓ |
| **total** | **170** | | |

#### §6.1.1 L1 산식 (Frenet trajectory 11 × 9)

```python
# x shape (N, 11, 3), end_idx = 10
R_wfn = build_frenet_basis_3d(x, end_idx=10)   # (N, 3, 3) — §4.2.1
origin = x[:, end_idx]                          # (N, 3)

# per-step 9D: [px, py, pz, vx, vy, vz, ax, ay, az] in Frenet, displacement units
L1 = np.zeros((N, 11, 9), dtype=np.float32)
for t in range(11):
    pos_world = x[:, t]
    v_world = x[:, t] - x[:, t-1]  if t >= 1 else np.zeros_like(pos_world)
    v_prev_world = x[:, t-1] - x[:, t-2]  if t >= 2 else np.zeros_like(pos_world)
    a_world = v_world - v_prev_world  if t >= 2 else np.zeros_like(pos_world)
    
    # Frenet 변환 (R_wfn^T 회전 + origin 평행이동)
    pos_frenet = einsum('nij,nj->ni', R_wfn.transpose(0, 2, 1), pos_world - origin)
    v_frenet   = einsum('nij,nj->ni', R_wfn.transpose(0, 2, 1), v_world)
    a_frenet   = einsum('nij,nj->ni', R_wfn.transpose(0, 2, 1), a_world)
    
    L1[:, t, 0:3] = pos_frenet
    L1[:, t, 3:6] = v_frenet
    L1[:, t, 6:9] = a_frenet
```

단위: displacement (v, a 가 Δt 분할 없음, plan-020 baseline_f0 carry 와 통일).

#### §6.1.2 L2 산식 (F0 residual sequence, Frenet)

```python
# 7 past sub-window: t ∈ {4, 5, 6, 7, 8, 9, 10}
# 각 sub-window 마다 F0 적용 → past observation `x[t]` 와의 잔차
# horizon = 2 step (= 0.080 s, plan-020 baseline_f0 default 와 통일)
#
# F0 input window = (x[t-4], x[t-3], x[t-2]) — 3 점 (last_idx = t-2 in sub-window)
# F0 output target time = (t-2) + horizon = t
# 즉 F0 가 t-2 시점에서 80ms 앞 예측 → t 시점 위치 예측
# actual_t = x[:, t]

L2 = np.zeros((N, 7, 3), dtype=np.float32)
residual_t_world_list = []  # §6.1.3 L4 가 reuse — d_t = ‖residual_t_world‖ 산출용
# f0_baseline_fn = plan-020 bf.f0_baseline (signature `(x, end_idx) -> pred (N, 3)`,
#                  horizon 2 step (=80ms) hardcoded in 산식 — 계수 1.98 ≈ 2 그 자체.
#                  parameter X. sub-window 3 점 호환 — last 3 obs 만 사용해 산출).
#
# ★ sub-window 호환성 verify (c2 build_input.py smoke test 의무 추가 항목):
#   assert bf.f0_baseline(x[:, 0:3, :], end_idx=2).shape == (N, 3)
#   assert np.isfinite(bf.f0_baseline(x[:, 0:3, :], end_idx=2)).all()
#   → 위 assertion fail 시 plan-020 f0_baseline 가 end_idx<10 호환 안 한다는 의미 →
#     `f0_baseline_3pt(p0, p1, p2)` helper 를 plan-021 안에 직접 작성 (산식: §4.3 numpy 그대로).
for i, t in enumerate(range(4, 11)):
    sub_x = x[:, t-4:t-1, :]                                    # shape (N, 3, 3)
    pred_t_world = f0_baseline_fn(sub_x, end_idx=2)              # (N, 3) — last_idx=2 → predict t (= 2+2)
    actual_t_world = x[:, t]                                      # (N, 3)
    residual_t_world = pred_t_world - actual_t_world             # (N, 3)
    residual_t_world_list.append(residual_t_world)                # ★ L4 cache
    # Frenet 변환 (현재 sample 의 R_wfn 사용 — sub-window 별 frame 아님)
    L2[:, i] = einsum('nij,nj->ni', R_wfn.transpose(0, 2, 1), residual_t_world)
```

`f0_baseline_fn` = `build_input_common` 의 인자로 plan-020 `bf.f0_baseline` 주입.

#### §6.1.3 L4 산식 (F0 soft hit sequence)

```python
# L2 의 7 past sub-window 와 동일 t loop. d_t = ‖residual_t_world‖ (world frame norm).
TAU_LOSS = 0.001  # 1mm

L4 = np.zeros((N, 7, 2), dtype=np.float32)
for i, t in enumerate(range(4, 11)):
    # L2 산출 시 이미 계산된 residual_t_world reuse
    d_t = np.linalg.norm(residual_t_world_list[i], axis=1)       # (N,)
    L4[:, i, 0] = sigmoid((0.010 - d_t) / TAU_LOSS)              # R=1cm
    L4[:, i, 1] = sigmoid((0.015 - d_t) / TAU_LOSS)              # R=1.5cm

sigmoid(z) = 1.0 / (1.0 + np.exp(-z))
```

#### §6.1.4 L5 산식 (macro statistic 9D, self-contained)

last 6 timestep (`start = max(0, end_idx - 5)` = 5, `pts = x[:, 5:11]` 6 점) 위 계산:

```python
pts = x[:, 5:11, :]                                              # (N, 6, 3)
v = np.diff(pts, axis=1)                                          # (N, 5, 3) — step deltas
speeds = np.linalg.norm(v, axis=2)                                # (N, 5)
current_speed = speeds[:, -1:] + 1e-9                             # (N, 1)
mean_speed = speeds.mean(axis=1, keepdims=True) + 1e-9            # (N, 1)
path = speeds.sum(axis=1, keepdims=True)                          # (N, 1)
disp = np.linalg.norm(pts[:, -1] - pts[:, 0], axis=1, keepdims=True)  # (N, 1)
straightness = disp / (path + 1e-9)                               # (N, 1)
speed_slope = (speeds[:, -1:] - speeds[:, :1]) / mean_speed       # (N, 1)
speed_cv = speeds.std(axis=1, keepdims=True) / mean_speed         # (N, 1)
v0, v1 = v[:, :-1], v[:, 1:]                                       # (N, 4, 3) each
turn_cos = (v0 * v1).sum(axis=2) / (np.linalg.norm(v0, axis=2) * np.linalg.norm(v1, axis=2) + 1e-9)
turn_accum = (1.0 - np.clip(turn_cos, -1, 1)).mean(axis=1, keepdims=True)
turn_volatility = (1.0 - np.clip(turn_cos, -1, 1)).std(axis=1, keepdims=True)
acc = np.diff(v, axis=1)                                          # (N, 4, 3)
acc_norm = np.linalg.norm(acc, axis=2)                            # (N, 4)
accel_slope = (acc_norm[:, -1:] - acc_norm[:, :1]) / (acc_norm.mean(axis=1, keepdims=True) + 1e-9)
linear_pred = x[:, end_idx-1] + (x[:, end_idx-1] - x[:, end_idx-2])
linear_resid = np.linalg.norm(x[:, end_idx] - linear_pred, axis=1, keepdims=True) / mean_speed
jerk = np.diff(acc, axis=1)                                       # (N, 3, 3)
jerk_vol = np.linalg.norm(jerk, axis=2).std(axis=1, keepdims=True) / mean_speed

# 9D = recent (6D) + observation_env unique (3D)
L5 = np.concatenate([
    path / current_speed,    # [0] normalized path length
    straightness,            # [1] disp/path
    speed_slope,             # [2] 속도 변화 추세
    speed_cv,                # [3] 속도 variability
    turn_accum,              # [4] 회전 누적
    accel_slope,             # [5] 가속도 변화 추세
    turn_volatility,         # [6] 회전 안정성
    linear_resid,            # [7] linear extrap 잔차 normalized
    jerk_vol,                # [8] jerk volatility normalized
], axis=1).astype(np.float32)  # (N, 9)
```

이 9D 가 LGBM 전용 (GRU 미사용 — sequence GRU 가 자체 학습).

EWMA 식:
```python
def ewma_last(seq: np.ndarray, alpha: float) -> np.ndarray:
    """seq (N, T, D) → last EWMA value (N, D). s_t = α·x_t + (1-α)·s_{t-1}."""
    s = seq[:, 0]
    for t in range(1, seq.shape[1]):
        s = alpha * seq[:, t] + (1 - alpha) * s
    return s
```

### §6.2 Model spec (LGBM dual head)

```python
class LgbmDualHead:
    def __init__(self, n_estimators=500, lr=0.05, num_leaves=63):
        # classifier: 7-class softmax (multi_logloss objective)
        self.clf = LGBMClassifier(
            n_estimators=n_estimators, learning_rate=lr, num_leaves=num_leaves,
            objective="multiclass", num_class=7, verbose=-1,
        )
        # regression: 21 booster (7 anchor × 3 axis), each scalar
        self.reg = [LGBMRegressor(
            n_estimators=n_estimators, learning_rate=lr, num_leaves=num_leaves,
            objective="regression", verbose=-1,
        ) for _ in range(21)]

    def fit(self, X, soft_label_q, residual_targets):
        # X: (N, 170), q: (N, 7) soft prob, residual_targets: (N, 7, 3)
        # classifier — argmax of soft label as hard target with sample_weight = q
        hard_target = soft_label_q.argmax(axis=1)
        weights = soft_label_q.max(axis=1)
        # ★ single-class fallback: hard_target unique count < 7 시 LGBM multi-class fit 실패 위험.
        # → 누락된 class index 마다 1개의 dummy sample 주입 (sample_weight=0 → 학습 기여 X,
        #    classes_ 만 7-class 확보). predict_proba 시 자연스럽게 0 column 으로 나옴.
        unique_classes = set(np.unique(hard_target).tolist())
        missing_classes = set(range(7)) - unique_classes
        if missing_classes:
            X_dummy = np.zeros((len(missing_classes), X.shape[1]), dtype=X.dtype)
            target_dummy = np.array(sorted(missing_classes), dtype=hard_target.dtype)
            weight_dummy = np.zeros(len(missing_classes), dtype=weights.dtype)
            X = np.concatenate([X, X_dummy], axis=0)
            hard_target = np.concatenate([hard_target, target_dummy])
            weights = np.concatenate([weights, weight_dummy])
        self.clf.fit(X, hard_target, sample_weight=weights)
        # regression — per-(anchor, axis) booster, target = anchor 별 residual offset
        for k in range(7):
            for axis in range(3):
                self.reg[k*3+axis].fit(X, residual_targets[:, k, axis])

    def predict(self, X):
        # logits via probabilities — **7-class zero-pad guard** (fold 내 미관측 class 처리)
        probs_raw = self.clf.predict_proba(X)                # (N, len(self.clf.classes_))
        probs = np.zeros((X.shape[0], 7), dtype=np.float32)  # full 7-channel
        for col_idx, class_label in enumerate(self.clf.classes_):
            probs[:, int(class_label)] = probs_raw[:, col_idx]
        # missing class column 은 0 (분포 mass 0) — softmax-renormalize 불필요 (학습 시
        # 미관측이라 inference 도 그 anchor 선택 안 됨이 합리적 prior)
        reg_offset = np.stack([
            np.stack([self.reg[k*3+axis].predict(X) for axis in range(3)], axis=1)
            for k in range(7)
        ], axis=1)  # (N, 7, 3)
        reg_offset = np.tanh(reg_offset) * 0.005  # bounded ±0.5cm
        return probs, reg_offset
```

**hyperparam (default)**:
- n_estimators=500, lr=0.05, num_leaves=63
- single seed (LGBM 결정적 — multi-seed 안 함)

**LGBM 의 soft CE 우회 (self-label vs 실제 구현 정합 명시)**:
- 본 plan §0/§4.2 의 "soft CE" 는 GRU 측 *full* 구현 (continuous q distribution 위 `-(q · log_softmax(logits)).sum(1).mean()`).
- LGBM 측은 sklearn API 의 multi-class classifier 가 *hard label + sample_weight* 만 지원 → **`argmax(q) + sample_weight=max(q)` 우회** (위 `fit` 코드).
- 즉 lever ④ "soft label" 은 LGBM 에선 *partial* (sample weighting 만), GRU 에선 *full* (continuous CE) 적용. 두 sub-exp 결과 차이 분석 시 본 우회 효과도 paradigm-level finding 의 component 로 박제.

### §6.3 학습 target

```python
# soft label (classifier target) — §4.2 build_soft_label 산식과 동일
# v1.3 박제: anchor reference = pred_F0_world (F0 의 80ms 미래), NOT origin (= last point).
# world → frenet: R_wfn^T @ (gt − pred_F0_world)
residual_true_frenet = np.einsum('nij,nj->ni', R_wfn.transpose(0, 2, 1), gt - pred_F0_world)  # (N, 3) Frenet
dist_to_anchors = np.linalg.norm(
    ANCHORS_FRENET[None, :, :] - residual_true_frenet[:, None, :], axis=-1
)  # (N, 7)
soft_label_q = softmax(-dist_to_anchors / 0.001, axis=1)  # (N, 7), τ_cls=0.001

# regression target (anchor 별 offset)
# raw target = true residual − anchor (unclipped, can be > ±0.005m for samples far from any anchor).
# inference 의 reg_offset 은 tanh*0.005 로 ±0.5cm clamp 라 unclamped target ↔ clamped output 불일치 가능.
# spec-default policy: **target 도 ±0.005m 로 clip** — anchor-local 영역만 학습, anchor 밖 sample 은
# anchor 쪽 saturation 효과 그대로 (classifier 가 다른 anchor 선택 책임).
residual_targets_raw = residual_true_frenet[:, None, :] - ANCHORS_FRENET[None, :, :]  # (N, 7, 3) unclipped
residual_targets = np.clip(residual_targets_raw, -0.005, 0.005)                       # (N, 7, 3) clipped
```

### §6.4 Inference (5-fold OOF)

```python
for k in range(5):
    train_idx = folds != k
    val_idx = folds == k
    model = LgbmDualHead()
    model.fit(X[train_idx], q[train_idx], residual_targets[train_idx])
    probs_val, reg_offset_val = model.predict(X[val_idx])

    # final pred (Frenet → world, v1.3 conceptual fix)
    # anchor reference = pred_F0_world (F0 의 80ms 미래), NOT origin.
    # → final_world = pred_F0_world + R_wfn @ Frenet mixture (±0.5cm anchor 보정).
    combined = ANCHORS_FRENET[None, :, :] + reg_offset_val   # (N_val, 7, 3) Frenet
    final_frenet = (probs_val[:, :, None] * combined).sum(axis=1)  # (N_val, 3) Frenet
    final_world = np.zeros((N, 3), dtype=np.float32)         # MINOR: pre-alloc
    final_world[val_idx] = einsum('nij,nj->ni', R_wfn[val_idx], final_frenet) + pred_F0_world[val_idx]
```

### §6.5 산출 (`results_lgbm.json`)

```json
{
  "candidate": "A_lgbm",
  "n_samples": 10000,
  "hit_1cm": 0.XX,
  "hit_1.5cm": 0.XX,
  "delta_1cm": +0.XX,
  "delta_1.5cm": +0.XX,
  "hit_1cm_per_fold": [...],
  "hit_1.5cm_per_fold": [...],
  "fold_variance_1cm": 0.0XX,
  "fold_variance_1.5cm": 0.0XX,
  "pass_both": true/false
}
```

### §6.6 G2.A 합격 기준 (자동)

- metric finite (no NaN/Inf)
- 위반 시 `lgbm_numerical` severe

### §6.7 시간 예산

- LGBM CPU (n_estimators=500 × 22 booster × 5 fold) ≈ ~10-20 min

---

## §7. STAGE 3 — Sub-exp B 단일방향 GRU (c8, G2.B)

### §7.1 Input spec (134D)

| 채널 | shape | source |
|---|---|---|
| seq (sequence input) | (B, 11, 9) | L1 Frenet trajectory |
| flat (concat) | (B, 35) | L2 (21) + L4 (14) flatten |

LGBM 의 L5 macro stat + L6 EWMA 미사용 (GRU 가 sequence 안에서 자체 학습).

### §7.2 Model spec (GRU dual head)

```python
class GRUDualHead(nn.Module):
    def __init__(self, seq_dim=9, hidden=64, flat_dim=35, dropout=0.1):
        super().__init__()
        self.gru = nn.GRU(input_size=seq_dim, hidden_size=hidden, num_layers=1,
                          batch_first=True, bidirectional=False)
        self.dropout = nn.Dropout(dropout)
        # head input = seq_hidden (64) + flat (35) = 99
        self.clf_head = nn.Linear(hidden + flat_dim, 7)
        self.reg_head = nn.Linear(hidden + flat_dim, 21)  # 7 × 3

    def forward(self, seq, flat):
        # seq: (B, 11, 9), flat: (B, 35)
        out, _ = self.gru(seq)
        seq_hidden = self.dropout(out[:, -1, :])      # (B, 64) — last-step single direction
        combined = torch.cat([seq_hidden, flat], dim=-1)  # (B, 99)
        logits = self.clf_head(combined)              # (B, 7)
        reg_raw = self.reg_head(combined).view(-1, 7, 3)
        reg_offset = torch.tanh(reg_raw) * 0.005      # bounded ±0.5cm
        return logits, reg_offset
```

### §7.3 학습 spec

| 항목 | 값 |
|---|---|
| Optimizer | Adam |
| Learning rate | 1e-3 |
| Batch size | 256 |
| Epochs | 50 |
| Dropout | 0.1 |
| Weight decay | 1e-4 |
| Device | cuda:1 (가용 시) / cpu fallback |
| Seed list | [20260518, 20260519, 20260520] |
| Seed aggregation | 각 fold k, 각 seed 별: epoch 0~49 학습 + `train_hit_history[epoch]` 누적 + early-stop (10 epoch plateau) → `best_epoch_per_seed = argmax(train_hit_history)` + checkpoint 저장. 3 seed 중 `best_seed = argmax_seed(train_hit_history[best_epoch_per_seed])` → 해당 (seed, best_epoch) checkpoint 만 val_k 위 평가. 5-fold val_k concat = OOF. (의사 코드: `best_seed, best_epoch = argmax_{seed, epoch ≤ early_stop_epoch}(train_hit_history[seed, epoch])`. val metric 미사용 — selection bias 회피.) |
| Early stop | train_(not k) hit plateau 10 epoch |
| Loss | `α · CE(softmax(logits), q) + β · smooth_hit(final, gt; R=0.01) + (β/2) · smooth_hit(final, gt; R=0.015)`, α=β=1.0 |
| smooth_hit τ schedule | annealed: epoch [0, 15) τ=0.003, [15, 30) τ=0.001, [30, 50) τ=0.0003 + boundary weighting (= §7.3.1 `tau_for_epoch` 의 0..14 / 15..29 / 30..49 와 정확 일치) |
| soft CE τ_cls | 0.001m (anchor scale 의 1/5) |
| Fold-internal training | 5-fold |

### §7.3.0 GRU 학습 loop 의 `final` 합성 (§6.4 LGBM inference 와 동일 식 재진술)

**책임 위치**: GRU `forward(seq, flat)` 는 `(logits, reg_offset)` 만 반환 (R_wfn/origin 은 forward signature 부재). final_world 산출 + loss 계산은 **training loop helper** (예: `compute_loss_and_pred(model, seq, flat, R_wfn, origin, q, gt, epoch)`) 에서 수행. R_wfn / origin 은 batch 별 tensor 로 dataloader 가 전달 (build_input_common 반환 dict 의 R_wfn, origin 을 fold 별 tensor 화 후 batch sampling).

```python
# GRU forward → (logits, reg_offset). 학습/평가 공통 final 산출 (training loop 안):
probs = torch.softmax(logits, dim=1)                                # (B, 7)
ANCHORS = torch.from_numpy(ANCHORS_FRENET).to(device)               # (7, 3), buffer
combined = ANCHORS[None, :, :] + reg_offset                         # (B, 7, 3) — Frenet
final_frenet = (probs[:, :, None] * combined).sum(dim=1)            # (B, 3) Frenet
# Frenet → world (v1.3 conceptual fix — anchor reference = pred_F0_world)
# R_wfn columns = [t̂, n̂, b̂] → world = R_wfn @ frenet_vec (transpose NO — §6.4 와 정합).
final_world = torch.einsum('nij,nj->ni', R_wfn, final_frenet) + pred_F0_world
# loss
loss = soft_ce_loss(logits, q)                                       # classifier
     + smooth_hit_loss(final_world, gt, tau=tau_for_epoch(ep)[0], use_boundary=tau_for_epoch(ep)[1])
```

LGBM inference (§6.4) 와 산식 동일, torch tensor + autograd 만 차이.

### §7.3.1 smooth_hit_loss 산식 (self-contained)

```python
def smooth_hit_loss(pred: Tensor, gt: Tensor, tau: float, use_boundary: bool = False) -> Tensor:
    """L = − mean_i [ w_i · ( sigmoid((R_main − d_i)/τ) + 0.5 · sigmoid((R_loose − d_i)/τ) ) ].

    pred, gt: (B, 3). 둘 다 world frame.
    tau: 0.003 → 0.001 → 0.0003 annealed per epoch (schedule §7.3 table).
    use_boundary: True (epoch 31-50) 시 boundary weighting 적용.
    """
    R_MAIN = 0.010
    R_LOOSE = 0.015
    d = (pred - gt).norm(dim=1)                                              # (B,)
    sh_main = torch.sigmoid((R_MAIN - d) / tau)                              # (B,)
    sh_loose = torch.sigmoid((R_LOOSE - d) / tau)                            # (B,)
    per_sample = sh_main + 0.5 * sh_loose
    if use_boundary:
        # boundary weight: R_main 경계 부근 sample 비중 강화 (d.detach() 로 weight grad 차단)
        d_detach = d.detach()
        w = 1.0 + 5.0 * torch.exp(-(((R_MAIN - d_detach) / 0.001) ** 2))     # (B,)
        per_sample = w * per_sample
    return -per_sample.mean()
```

τ schedule (per epoch, `tau_for_epoch(epoch)`):
- epoch 0..14  → τ = 0.003,  use_boundary = False
- epoch 15..29 → τ = 0.001,  use_boundary = False
- epoch 30..49 → τ = 0.0003, use_boundary = True

### §7.4 산출 (`results_gru.json`)

```json
{
  "candidate": "B_gru",
  "n_samples": 10000,
  "hit_1cm": 0.XX,
  "hit_1.5cm": 0.XX,
  "delta_1cm": +0.XX,
  "delta_1.5cm": +0.XX,
  "hit_1cm_per_fold": [...],
  "hit_1.5cm_per_fold": [...],
  "train_hit_log": {fold: {seed: train_hit, best_seed: ...}},
  "pass_both": true/false
}
```

### §7.5 G2.B 합격 기준

- metric finite
- val_hit > 0.10 (= F0 baseline 0.6320 / 6, **task-specific 최소 학습 시그널 floor**. 7-class random uniform 0.143 보다 낮은 conservative 임계 — §0.5 `gru_no_signal` 정의 carry, hit@1cm 의 random floor 가 R=0.01 m 거리 chance 라 매우 낮음.)
- train_hit − val_hit < 0.10 (overfit guard, 미달 시 `gru_overfit` warn)

### §7.6 시간 예산 (cuda:1)

- 5 fold × 3 seed × 50 epoch × 8000 sample / 256 batch ≈ ~30-50 min

---

## §8. STAGE 4 — Paradigm-level finding (c9, G3)

### §8.1 산출 표

`analysis/plan-021/paradigm_analysis.md`:

```markdown
| sub-exp | hit@1cm | Δ_1cm | hit@1.5cm | Δ_1.5cm | pass 둘 다 ≥ +0.005 |
|---|---|---|---|---|---|
| F0 baseline | 0.6320 | — | 0.8033 | — | — |
| A LGBM (170D + EWMA + macro) | 0.XX | +0.XX | 0.XX | +0.XX | ✓/✗ |
| B GRU (134D + sequence) | 0.XX | +0.XX | 0.XX | +0.XX | ✓/✗ |
```

### §8.2 A vs B 직접 비교

| 비교 axis | A LGBM | B GRU | 차이 의미 |
|---|---|---|---|
| Δ_1cm | +0.XX | +0.XX | tree+aggregate vs seq learning |
| Δ_1.5cm | +0.XX | +0.XX | |
| fold variance | 0.0XX | 0.0XX | LGBM 안정성 vs GRU 분산 |
| best_seed 일치 | — | seed log | seed sensitivity |

### §8.3 4 lever 의 marginal 가치 (paradigm-level inference)

본 plan 의 4 lever 는 *동시 적용* 측정 (lever 별 ablation X — out-of-scope per §2.2). 따라서:
- A 또는 B 가 pass → 4 lever *전체* 가 paradigm-level 효과 입증
- 둘 다 fail → 4 lever 의 *조합 한계* 박제 (paradigm 측면 negative finding)

lever 별 marginal 가치 ablation = follow-up plan 으로 carry.

### §8.4 G3 합격

- A 또는 B 중 ≥ 1 paired Δ ≥ +0.005 *둘 다* → PASS
- 0 통과 → `all_negative` warn 박제 후 G_final 진입

---

## §9. STAGE 5 — Results + frontmatter sync (c10, G_final)

### §9.1 3-file frontmatter sync

- `plans/plan-021-frenet-corrector-input-augment.md` top-level frontmatter
- `plans/plan-021-frenet-corrector-input-augment.results.md`
- `analysis/plan-021/results.md`

세 파일 모두 다음 필드 동시 갱신:
- `status: all_complete` (또는 `partial` if G2.A 또는 G2.B fail)
- `band: positive / marginal / negative` (G3 winner 의 paired Δ 기준)
- `best_sub_exp: A_lgbm / B_gru / 없음` (G3 winner 단수 — A/B 둘 다 pass 시 Δ_combined tie-break)
- `best_hit_1cm: <float>`, `best_hit_1.5cm: <float>`

#### §9.1.1 overall best 단수 선정 (A + B 둘 다 pass 시)

1. pass criterion 둘 다 통과 sub-exp 만 candidates
2. candidates 중 *가장 큰 Δ_combined = Δ_hit@1cm + 0.5·Δ_hit@1.5cm* 후보 1개 = winner
3. tie 시 hit@1cm 우선, 그 다음 fold variance (smaller)
4. candidates 빈 경우 → `best_sub_exp = "없음"`, `band = negative`

### §9.2 results.md 필수 항목

- F0 baseline (G1 carry plan-020)
- A LGBM 5-fold OOF + paired Δ (full table)
- B GRU 5-fold OOF + paired Δ (full table)
- A vs B 직접 비교 (§8.2)
- 4 lever 의 (조합) paradigm-level 효과
- decision-note 박제 list
- follow-up plan 후보 ≥ 2
- caveats

### §9.3 G_final 합격 기준

- 3-file sync 완료
- §0.5 commit chain c1~c10 모두 [DONE]
- results.md 필수 항목 모두 박제
- follow-up plan 후보 ≥ 2건 박제

---

## §N+1. results.md 필수 항목

(plan-020 results.md format 참조)

- plan_id, version, date, status, band, best_sub_exp
- F0 baseline measured (G1 carry)
- A LGBM × 2 metric × 5-fold concat 표
- B GRU × 2 metric × 5-fold concat 표
- A vs B 직접 비교 (학습 방식 / input dim / paradigm 차이)
- 4 lever 의 paradigm-level 효과 (조합)
- decision-note 박제 list
- follow-up plan 후보 (post-G_final)

---

## §N+2. 통계 함정 & caveats

1. **Frenet basis numerical degeneracy**: ‖v_last‖ < 1e-9 (정지 sample) 또는 ‖a_⊥‖ < 1e-9 (직선 운동) → t̂ 또는 n̂ 정의 불가. fallback = world frame use + flag indicator. >5% sample 발생 시 `frenet_basis_degenerate` severe.

2. **soft label collapse**: classifier 의 π_0 (origin anchor) 평균 > 0.95 → mode 분리 시그널 부족, mean-regression trap 재발. soft CE τ_cls=0.001 너무 sharp 의심 → τ_cls 키우는 ablation (0.001 → 0.003).

3. **F0 residual sequence 의 sub-window 의존**: 7 past sub-window 의 F0 predictions 가 *현재 step* 의 F0 prediction 분포와 다를 가능성 (sub-window 가 *4..10* 의 11점 trajectory 안 위치 변동). 본 plan 은 sub-window 5+ point 마지막 만 사용해 *minimum subset* lock-in. 그래도 distribution shift 존재 — caveats 박제.

4. **F0 soft hit τ_loss 와 anchor 0.5cm scale 정합**: τ_loss=0.001m (anchor radius 의 1/5) — anchor 경계 부근 sample 의 soft hit 변동이 학습 신호로 직접 작용. τ 너무 작으면 hard hit 비슷 → graded 정보 손실.

5. **LGBM multi-output regression bias**: 21 LGBM regressor 독립 학습 → axis 간 correlation 학습 X. F0 residual 의 *방향성* 학습 한계 가능. multi-output GBDT (단일 model) 또는 axis 결합 학습 follow-up.

6. **GRU 단일방향 한계**: bidirectional 안 사용. trajectory 가 *모두 과거* 라 future leak 없지만, *역방향* context (예: "이 sample 의 미래 motion" 의 backward pass) 활용 X. bidirectional GRU ablation = follow-up.

7. **2 sub-exp 독립의 paradigm 한계**: A 와 B 가 *같은 input 강화 lever* 위 model 만 다름. 둘 다 fail 시 → "4 lever 자체의 한계" vs "model 의 한계" 분리 어려움. lever 별 ablation 이 진정한 진단 (follow-up).

8. **EWMA α multi-scale 의 hyperparam 과적합 위험**: α ∈ {0.1, 0.3, 0.5} 3-scale 자체가 hyperparam — fold 마다 best α 다를 가능성. fold-internal α 선택 ablation 미시도.

9. **paired Δ 측정의 dataset 의존**: F0 baseline 의 fold 별 hit 변동 (0.625~0.640) → paired Δ 도 fold 마다 다름. fold variance 박제 의무.

10. **DACON LB submit out-of-scope**: 본 plan 의 G3 PASS = 5-fold OOF 위 paradigm-level 결론만. LB 측정은 follow-up plan-022 (가칭) 의 27-pool 통합 후 carry.

---

## §N+3. 변경 이력

- v1 (2026-05-18): 초안 — 4 lever input augment (Frenet + F0 residual + F0 soft hit + soft label) + dual head (7-anchor classifier + 7×3 reg) + 2 sub-exp 독립 (LGBM 170D / GRU 134D). pass criterion paired Δ ≥ +0.005 둘 다.
- v1.2 (2026-05-18): plan-review-master 5-iter 자동 fix (BLOCKER 0 도달, 37 fix). 산식·시그너처·경계·단위 self-contained 강화.
- v1.3 (2026-05-18): **conceptual fix** — anchor codebook 의 reference 가 잘못된 origin (= x[end_idx], last observed) 으로 정의되어 있어 c7 G2.A 첫 actual run 시 A LGBM hit@1cm = 0.06 (F0 0.6320 보다 -57pp). 원인 = final pred 식이 `x[end_idx] + Frenet ±0.5cm` 로 F0 의 80ms 외삽 (= ~2cm) 누락. **fix**: anchor reference 를 *pred_F0_world* (= F0 산출 80ms 미래 위치) 로 변경 — final pred = `pred_F0_world + R_wfn @ Σ_k π_k · (anchor_k + reg_offset_k)`. spec §4.2 (build_input_common 반환 dict + build_soft_label signature) / §6.3 (residual_true_frenet 산식) / §6.4 (LGBM final pred) / §7.3.0 (GRU final pred) 일괄 수정 + L1 의 origin (= last point, invariance 용) 와 anchor reference (= pred_F0_world) 분리 명시. 11/11 pytest 통과 후 c7 relaunch.

---

## §N+4. 참조

- `plans/plan-020-f0-structural-search.md` — F0 baseline 0.6320 / 0.8033 + 5-fold stable_fold_id + C05 per-regime winner finding
- `plans/plan-020-f0-structural-search.results.md` — 17 후보 ablation 결과 + paradigm-level finding
- `analysis/plan-020/baseline_oof.{json,md}` — G1 carry source
- `analysis/plan-020/baseline_f0.py` — F0 산식 carry (numpy + torch)
- `src/pb_0_6822/selector.py` L185 `stable_fold_id` — fold split carry
- `src/pb_0_6822/selector.py` L297-360 `recent_temporal_physics_features` + `observation_environment_features` — 사용자 명시 carry (LGBM 전용 9D macro stat)
- `src/io.py` `load_all_samples` + `load_labels` — data carry
- `CLAUDE.md` — autonomous execution policy
- `WORKFLOW.md` — plan/results/registry convention
- memory `feedback_user_proposal_isolation.md` — 사용자 명시 외 다른 plan 의 carry 자동 incorporate 금지 (본 plan-021 design 의 baseline 원칙)
