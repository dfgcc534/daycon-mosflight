---
plan_id: 025
version: 1.0
date: 2026-05-21 (Asia/Seoul)
status: draft
based_on:
  - 024 (cross-attn anchor-vocab G2 FAIL hit_1cm 0.6370, -0.0158 vs plan-022. 사후분석 4축 root-cause 박제: ①sample-weight expansion effective N 14× gap [가장 큰 단일 원인], ②static anchor × cross-attn mismatch, ③16 lever FE 245D redundancy [v6 ablation +0.0003], ④2M params overfit plateau val_loss ≈ log14. plan-024 v5 A7 Learnable embed top1_acc +0.0162 partial signal carry. plan-024 poss 3 input aug +0.0135 lift carry.)
  - 022 (A6_bcc14_tau001 winner OOF hit_1cm 0.6528 / hit_1.5cm 0.8104 / Δ_1cm +0.0208. LGBM sample-weight expansion = effective rows N×K=140k vs cross-attn N=10k 의 14× gap reference. 14 BCC anchor + τ=0.001 sharp soft label paradigm carry.)
  - 021 (170D LGBM input carry. F0 baseline 0.6320 / 0.8033 anchor.)
  - 020 (F0 baseline + 5-fold stable_fold_id MD5)
inspired_by:
  - 024 §5.10 사용자 통찰 "model overpowered + low-info input → overfit" + §10 plan-025 영역 "LGBM sample-weight expansion mimic" 명시 carry
  - plan-022 sub_v1 LgbmSelectorOnly._build_pointwise_frame (N×K row expansion + group-by-sample) reference 패턴
code_reuse:
  - module: analysis/plan-024/model.py
    symbols: [CandidateAttentionGRUSelector]
    reason: cross-attn GRU 아키텍처 carry. E0 (plan-024 reproduce control) 및 E1~E3 의 backbone 동일. forward signature 만 (sample_feat, cand_feat) → pointwise 변형 확장.
  - module: analysis/plan-024/cand_builder.py
    symbols: [build_cand_feat]
    reason: 14 BCC anchor cand_feat 22D (= position 3D + radial 1D + sample-broadcast-context 18D) builder carry. 단 plan-024 의 cand 150D (= 22D + ctx 128D broadcast) 가 아닌 *순수* 22D 만 사용 (input redundancy 회피, §2.2 single variable).
  - module: analysis/plan-024/seq_builder.py
    symbols: [build_seq_feat]
    reason: T=7 sequence feat 95D 의 *Tier S+A 제거 형* — plan-021 carry 170D 와 정합. 단 plan-024 의 16 lever FE max 미적용 (v6 ablation +0.0003 redundancy 박제 후 carry 차단).
  - module: analysis/plan-022/anchors.py
    symbols: [ANCHORS_A6]
    reason: 14 BCC anchor 좌표 (= plan-024 carry 정확 동일).
  - module: analysis/plan-022/selector_only_model.py
    symbols: [LgbmSelectorOnly, build_soft_label_with_tau]
    reason: G1 plan-022 carry reproduce + E1 의 pointwise target q_ik (=build_soft_label_with_tau τ=0.001 결과) ground-truth.
  - module: analysis/plan-021/build_input.py
    symbols: [build_frenet_basis_3d, to_frenet, build_input_common, build_input_lgbm_extra]
    reason: 170D LGBM input pipeline (plan-022 §6.1 동일 carry). E1/E2/E3 의 seq_feat backbone.
  - module: analysis/plan-020/baseline_f0.py
    symbols: [f0_baseline, D1, PAR, PERP, R_HIT, R_HIT_LOOSE]
    reason: F0 baseline injection + paired Δ anchor (plan-022/023/024 모두 carry).
  - module: src/io.py
    symbols: [load_all_samples, load_labels]
    reason: data loader.
  - module: src/pb_0_6822/selector.py
    symbols: [stable_fold_id]
    reason: 5-fold stable split (plan-020/021/022/023/024 carry, 변경 불가).
followed_by:
  - plan-026 (가칭): anchor pool dynamic 화 — static 14 BCC → sample-conditional anchor (plan-004 27 physics pattern 복원 위에 14 BCC). plan-024 사후분석 root-cause #2 (static anchor × cross-attn mismatch) 직접 fix.
  - plan-027 (가칭): ensemble (plan-022 LGBM + plan-025 best variant). plan-024 §10 영역.
  - plan-028 (가칭, lower priority): anchor radius 확장 (0.5cm → 0.7~1.0cm) + F0 baseline ML 화. oracle 0.7928 → 0.85+ 추정.
scope: 단일 변수 = **effective N 메커니즘** (10k → ~140k) + **anchor identity capacity** (A7 learnable embedding). 4 model variant (E0 control + E1/E2/E3 expansion lever) × 1 anchor (14 BCC) × 1 input (170D plan-021 carry) × 1 τ_cls (0.001 plan-022 winner). E1 = pointwise K-independent forward (LGBM mimic 최직접). E2 = row expansion + listwise CE (hybrid). E3 = E1+A7 learnable anchor embedding. anchor layout sweep (plan-023 영역) / 16 lever FE (plan-024 fail mode) / corrector reg head / dynamic anchor (plan-026) / ensemble (plan-027) / DACON LB submit / sample-conditional anchor / 1-fold long-diag = out-of-scope.
exp_ids:
  - Z025_E0_plan024_reproduce
  - Z025_E1_pointwise_expansion
  - Z025_E2_row_expansion_listwise
  - Z025_E3_E1_plus_anchor_embed
# exp_id ↔ variant 매핑: frontmatter `Z025_E{n}_<short>` ↔ 본문 variant_key `E{n}_<short>`. best_variant frontmatter 박제 시 variant_key 사용 (e.g. `E1_pointwise_expansion`).
lb_score: null
band: null
---

# plan-025 v1 — Cross-attn Effective-N Expansion Mimic + Learnable Anchor Embedding

## §0. 한 줄 목적

> **plan-024 사후분석 #1 root-cause (LGBM sample-weight expansion 의 effective N=N×K=140k vs cross-attn 의 N=10k 의 14× gradient signal gap) 을 cross-attention GRU 학습에서 직접 재현하여, plan-024 의 G2 FAIL (0.6370) 이 *arch lever 자체 fail* 인지 *expansion 부재의 결과* 인지를 분리 측정한다.** 3 expansion variant (E1 pointwise / E2 row-expansion-listwise / E3 E1+A7 learnable anchor embed) head-to-head + E0 plan-024 v1.1-rev2 control. **단일 변수 = effective N 메커니즘 + anchor identity capacity**. anchor 14 BCC + input 170D + τ_cls=0.001 = plan-022 winner carry 그대로 (= plan-024 의 16 lever FE 와 그 외 모든 lever 차단). **G2 PASS criterion = hit_1cm ≥ 0.6528 (plan-022 winner 회복)** — 회복 시 arch lever 가 expansion 부재로 mask 되었던 것 확정, 미회복 시 arch lever 자체 fail 확정 후 plan-026 anchor dynamic 으로 분기.

---

## §0.5 Quick Reference (autonomous loop 매 turn 읽는 section)

### 합격 기준 (G-gate sequence)

- **G0**: 4 module (model_pointwise / expansion_runner / anchor_embed / pytest) import + smoke + tests green (≥ 10/10). 위반 시 `infra_drift` severe.
- **G1**: F0 baseline 5-fold concat OOF — hit@1cm ∈ [0.6315, 0.6325] AND hit@1.5cm ∈ [0.8028, 0.8038] (plan-020/021/022/023/024 carry exact). plan-022 winner reproduce hit@1cm ∈ [0.6523, 0.6533] AND hit@1.5cm ∈ [0.8099, 0.8109] (plan-022 carry). 위반 시 `f0_reproduce_drift` 또는 `plan022_carry_drift` severe.
- **G2.E0**: plan-024 v1.1-rev2 reproduce — hit_1cm ∈ [0.6360, 0.6385] (= plan-024 results §2 OOF 0.6370 ± per-fold std 0.0034). 위반 시 `plan024_reproduce_drift` severe (E1~E3 측정 무의미 → halt).
- **G2.E1**: pointwise K-independent forward — hit_1cm finite + max_class_ratio < 0.95 (mode collapse 차단, plan-022 §3.3 동일). q_pred_ik distribution finite (NaN/Inf 차단). 위반 시 `pointwise_numerical` severe.
- **G2.E2**: row expansion listwise CE — 동일 무결성 gate. 위반 시 `rowexp_numerical` severe.
- **G2.E3**: E1 best + A7 anchor embed — anchor embed L2 norm finite (gradient explode 차단). 위반 시 `embed_diverge` severe.
- **G3 (paradigm-level)**: max(E1, E2, E3) hit_1cm ≥ **0.6528** (plan-022 winner) → PASS. 0.6528 ≤ best < 0.6628 = band positive (arch lever salvageable). best ≥ 0.6628 (plan-024 G3) = band strong_positive. best < 0.6528 = band negative (`expansion_no_recovery` warn 박제 → plan-026 dynamic anchor 분기).
- **G_final**: results.md (12 항목) + best variant 박제 (E# + hit_1cm + gap_ranking + top1_acc) + plan-024 v1 0.6370 / plan-022 0.6528 head-to-head 표 + arch lever 분리 결론 + follow-up plan 후보 ≥ 2건 박제 + 3-file frontmatter sync.

### G-gates

- G0: STAGE 0 인프라 + 10/10 pytest                              [TODO]
- G1: STAGE 1 F0 + plan-022 carry reproduce                      [TODO]
- G2.E0: plan-024 reproduce control                              [TODO]
- G2.E1: pointwise expansion                                     [TODO]
- G2.E2: row expansion listwise                                  [TODO]
- G2.E3: E1 + A7 anchor embed                                    [TODO]
- G3: paradigm — band 판정 (positive / strong_positive / negative) [TODO]
- G_final: results + 3-file sync + follow-up                     [TODO]

### Commit chain (next-up)

| # | type | spec section | status |
|---|---|---|---|
| c1 | docs | `plans/plan-025-expansion-mimic-anchor-embed.md` v1 작성 (본 commit) | [TODO] |
| c1.5 | docs | plan-review-master 5-iter 자동 fix (BLOCKER+AMB 0 수렴 목표). 잔여 MINOR 허용. | [TODO, optional pre-G0] |
| c2 | code | `analysis/plan-025/model_pointwise.py` — CandidateAttentionGRUSelector 의 forward 를 (B, K) → (B*K,) scalar q_ik regression head 로 변형. spec @ §5.1 | [TODO] |
| c3 | code | `analysis/plan-025/model_row_expansion.py` — batch shape (B*K, D) + group-by-sample softmax + listwise CE. spec @ §5.2 | [TODO] |
| c4 | code | `analysis/plan-025/anchor_embed.py` — `nn.Parameter(14, 8)` learnable anchor embedding (plan-024 v5 A7 carry, init=0.02). spec @ §5.3 | [TODO] |
| c5 | code | `analysis/plan-025/expansion_runner.py` — 5-fold OOF runner + 4 variant CLI (E0/E1/E2/E3) + per-fold timing. spec @ §6 | [TODO] |
| c6 | test | `tests/test_plan025_smoke.py` — 10 pytest (4 variant forward + soft label q_ik consistency + anchor embed shape + group-by-sample softmax sum=1 + F0 carry + samples/anchor floor + mode collapse guard + per-batch invariants) | [TODO] |
| G0 | gate | smoke + tests green 10/10 | [TODO] |
| c7 | exp G1 | F0 + plan-022 carry reproduce → `analysis/plan-025/baseline_carry.json` 박제 (dataset_hash + F0 + plan-022 winner 3 metric) | [TODO] |
| G1 | gate | F0 + plan-022 carry hit_1cm/1.5cm 둘 다 tolerance 통과 | [TODO] |
| c8 | exp G2.E0 | plan-024 v1.1-rev2 reproduce — control variant. hit_1cm 0.6370 ± 0.0015 sanity. `results_E0.json` 박제. | [TODO] |
| G2.E0 | gate | plan-024 0.6370 reproduce ✓ (drift < 0.0015) | [TODO] |
| c9 | exp G2.E1 | pointwise K-independent forward — main root-cause fix variant. 5-fold OOF + per-fold variance + soft_CE + top1_acc + gap_ranking + max_class_ratio. `results_E1.json` 박제. | [TODO] |
| G2.E1 | gate | metric finite ✓ + max_class_ratio < 0.95 ✓ + soft_CE finite | [TODO] |
| c10 | exp G2.E2 | row expansion listwise — hybrid variant. 동일 metric set. `results_E2.json` 박제. | [TODO] |
| G2.E2 | gate | metric finite ✓ + max_class_ratio < 0.95 ✓ | [TODO] |
| c11 | exp G2.E3 | E1 + A7 anchor embed — capacity lever 추가. 동일 metric set + anchor_embed_L2_norm 박제. `results_E3.json` 박제. | [TODO] |
| G2.E3 | gate | metric finite ✓ + anchor embed L2 norm finite ✓ | [TODO] |
| c12 | analysis | 4 variant × 5-fold metric 표 + plan-024 v1 (0.6370) / plan-022 (0.6528) head-to-head + per-fold variance + soft_CE - log(14) deviation 표 + gap_ranking 비교 + arch lever 분리 결론 → `paradigm_analysis.{json,md}` | [TODO] |
| G3 | gate | band 판정 (max(E1,E2,E3) vs 0.6528 / 0.6628) | [TODO] |
| c13 | docs | `analysis/plan-025/results.md` 12 항목 + `plans/plan-025-*.results.md` pair + follow-up plan-026/027/028 박제 + 3-file frontmatter sync (status=all_complete, band=… , best_variant=…) | [TODO] |
| G_final | gate | 3-file sync ✓ + §0.5 c1~c13 모두 [DONE] ✓ + follow-up 2+ 박제 ✓ | [TODO] |

### Plan-specific severe (WORKFLOW.md §12.3 default 위 추가분)

- **`plan024_reproduce_drift`**: G2.E0 의 plan-024 v1.1-rev2 reproduce 가 hit_1cm 0.6370 ± 0.0015 밖. control 자체 실패 시 E1~E3 결과 해석 무의미 → halt + telegram alert. 원인 추적: random seed / lib version / fold split MD5 / cand_feat hash.
- **`pointwise_numerical`** / **`rowexp_numerical`**: E1/E2 의 forward 또는 backward 가 NaN/Inf. 원인: pointwise q_ik scalar regression 의 sigmoid saturation 또는 row expansion 의 group-by-sample softmax overflow. halt + 원인 분석 후 hyperparam fix.
- **`embed_diverge`**: E3 의 `nn.Parameter(14, 8)` L2 norm > 10 (= init 0.02 의 500×). gradient explode. halt + lr / weight_decay 재조정.
- **`expansion_no_recovery`** (warn only, halt 아님): G3 의 max(E1,E2,E3) hit_1cm < 0.6528. arch lever 자체 fail 확정 → band negative + plan-026 dynamic anchor 분기. results.md §5 에 분기 결론 박제 후 G_final 진입.

### Plan-specific paths (WORKFLOW.md §12.5/§12.6 default 위 추가/제외)

- whitelist 추가: `analysis/plan-025/**/*` (본 plan 산출 영역)
- whitelist 추가: `tests/test_plan025_smoke.py`
- blacklist 추가: `analysis/plan-024/**/*` (plan-024 reproduce 시 *read-only* import, code 수정 금지 — drift 추적 가능성 보존)
- blacklist 추가: `analysis/plan-022/**/*` (plan-022 carry symbols *read-only*)

### Decision-note 사용 예 (자율 결정 시 commit msg 박제)

- `decision-note: spec-default — pointwise regression head 의 loss = MSE (대안 BCE 미선택, q_ik ∈ [0,1] regression 의 sklearn convention)`
- `decision-note: spec-default — anchor embed init σ=0.02 (plan-024 v5 carry, Transformer default)`
- `decision-note: spec-default — input augmentation σ=0.05 미적용 (§2.2 single variable 원칙, plan-024 poss 3 의 +0.0135 lift 는 plan-025 G3 PASS 후 plan-026 영역으로 분리)`
- `decision-note: spec-default — E1 pointwise 의 inference 시 14 q_ik 를 softmax-renormalize 후 weighted avg (q_ik 가 학습 시 sigmoid scalar 라 sum ≠ 1, plan-022 selector_only_model.predict_oof 패턴 carry)`

---

## §1. 배경 / plan-024 인계

### §1.1 plan-024 G2 FAIL 의 4축 사후분석 (results.md §5 박제)

| 축 | 진단 | 증거 | plan-025 처리 |
|:--|:--|:--|:--|
| ① effective N 14× gap | LGBM sample-weight expansion = N×K=140k effective row vs cross-attn = N=10k. 14× gradient signal 차이. | plan-024 results.md §5(f) val_loss 2.5736 ≈ log(14) − 0.066 = near-uniform softmax plateau | **본 plan G2.E1/E2/E3 의 핵심 fix** |
| ② static anchor × cross-attn mismatch | 14 BCC anchor 가 모든 sample 에 정적 → cross-attn query 의 sample-conditionality 부재 → attention degenerate | plan-024 results.md §5.3 + §5.10 (e) 미시도 sweet spot | **plan-026 영역, 본 plan out-of-scope** |
| ③ 16 lever FE 245D redundancy | v6 ablation: LGBM 에 245D 먹여도 +0.0003. plan-021 carry 170D 가 도메인 포화. | plan-024 results.md §5.9 (v6 finding) | **본 plan input = 170D plan-021 carry, 245D 차단** |
| ④ 2M params + overfit | model overpowered. best ep 35 → 발산. input aug poss 3 +0.0135 lift = overfit fix lever. | plan-024 results.md §5.10 (a)(d) | **본 plan input aug out-of-scope (§2.2 single variable). best ep tracking + early stop 만 유지.** |

본 plan 은 **축 ①만** isolation 으로 측정. 다른 3 축은 plan-026/027/028 분기.

### §1.2 LGBM sample-weight expansion 의 수식적 정의 (= plan-022 §6.2 carry)

plan-022 의 `LgbmSelectorOnly._build_pointwise_frame` 패턴:

```
input: X ∈ R^{N×D}, anchors ∈ R^{K×3}, target ∈ R^{N×3}, τ ∈ R+
step 1: q_target = build_soft_label_with_tau(target, anchors, τ)  # (N, K), softmax of -‖t-a‖²/τ
step 2: 펼침
    X_expand   = repeat(X, K, axis=0)        # (N*K, D)
    A_expand   = tile(anchor_feat, N, ax=0)  # (N*K, A_dim)  # anchor 좌표 + radial
    y_expand   = q_target.reshape(N*K,)       # (N*K,)
    w_expand   = ones(N*K,) * K               # sample_weight = K (per-sample 합 = K, 동등 가중)
step 3: LGBM 학습
    pred_scalar_ik = LGBM.fit(concat([X_expand, A_expand]), y_expand, w_expand)
step 4: inference
    pred (N, K) = pred_scalar_ik.reshape(N, K)
    pred_softmax = softmax(pred / τ) along axis=1   # 또는 그냥 sum 후 normalize
    final = pred_softmax @ anchors   # (N, 3)
```

**핵심**: gradient signal 의 *unit* 이 (sample, anchor) pair 1개. LGBM tree 의 split 결정이 140k 개 row 의 loss 에 의존 → effective N 14×.

cross-attn 의 plan-024 default:
```
input: X ∈ R^{N×D}, anchor_feat ∈ R^{K×A}
forward: logits ∈ R^{N×K} = cross_attn(X, anchor_feat)
loss: CE(softmax(logits), q_target)   # listwise, sum over K within sample
gradient unit: sample 1개 (K=14 logit 의 joint softmax 의 sum loss).
```

→ cross-attn 의 effective N = N. **plan-025 가 이 gap 을 메우는 3 variant**.

---

## §2. Scope (명시적)

### §2.1 In-scope

| 항목 | 값 |
|:--|:--|
| anchor | 14 BCC (= plan-022 A6, plan-024 carry, plan-023 미사용 — 직접 head-to-head 우선) |
| τ_cls (soft label) | 0.001 (= plan-022 winner, plan-024 carry) |
| input | 170D plan-021 carry (build_input_common + build_input_lgbm_extra). plan-024 의 16 lever FE 245D 미적용. |
| selector backbone | CandidateAttentionGRUSelector (plan-024 carry) hidden=384 default, 단 E0 만 plan-024 v1 hyperparam 정확 carry |
| fold split | 5-fold stable_fold_id MD5 (plan-020 carry, 변경 불가) |
| model variant | E0 control / E1 pointwise / E2 row-listwise / E3 E1+A7 = 4 variant |
| anchor identity | E0/E1/E2 = plain anchor coord + radial. E3 = + nn.Parameter(14, 8) learnable embed concat |
| training budget | E0~E3 동일 — epoch 100 max + early stop patience 8 (val_loss) + lr 7e-4 const + weight_decay 0.02. plan-024 default carry. |
| metric set | OOF hit_1cm / hit_1.5cm / Δ_F0 / Δ_F0_1.5 / gap_ranking / top1_acc / max_class_ratio / soft_CE / dist_match_KL (= plan-024 §1 동일) |
| compute | 5-fold CPU (E2 의 row expansion = batch K× 효과로 ~14× wall time 예상) |

### §2.2 Out-of-scope (절대 안 함)

| 항목 | 이유 |
|:--|:--|
| anchor layout sweep (B4_fib50 등) | plan-023 영역, 본 plan single variable 침범 |
| 16 lever FE 245D (Multi-window/STA-LTA/WAP/wingbeat 등) | plan-024 v6 ablation +0.0003 redundancy 박제 후 차단 — single variable + input 도메인 포화 |
| corrector reg head | plan-021 selector-only carry, corrector slot = plan-027 ensemble 영역 |
| dynamic anchor (sample-conditional) | plan-026 영역, root-cause #2 분기 |
| ensemble (LGBM + cross-attn) | plan-027 영역 |
| DACON LB submit | G3 PASS 후 plan-027 또는 별 plan 영역 (memory feedback_dacon_submit_confirmation 박제) |
| input augmentation Gaussian noise σ=0.05 (plan-024 poss 3) | §2.1 single variable 원칙 — overfit fix lever 는 G3 PASS 후 plan-026 영역으로 분리. 단 E1/E2/E3 가 G3 fail 시 plan-026 의 0순위 lever 후보. |
| hidden width sweep (128/256/384 등) | plan-024 v4/v7 ablation 박제 후 effect ≈ 0 확정, carry 차단 |
| τ_cls scan (0.003/0.005 등) | plan-022 winner τ=0.001 carry, single variable |
| 1-fold long-diagnose | plan-024 §5.10 영역. 본 plan 은 5-fold OOF 정식 측정만. |

### §2.3 단일 변수 원칙의 정확한 정의

**plan-025 의 단일 변수 = effective N 메커니즘** (E0 = 10k / E1·E2 = 140k / E3 = 140k + 112 learnable params).

E3 의 anchor embed (=second lever) 가 *동시 변경* 으로 보일 수 있으나, E3 는 E1 위에 *orthogonal lever 추가* 의 head-to-head 측정 — E1 single 결과와 E3 단독 비교로 lever decomposition 가능 (plan-022 §2 의 "한 변수 + orthogonal head-to-head" 패턴 carry).

---

## §3. 사전 등록 (Pre-registration)

### §3.1 Fold split

5-fold stable_fold_id MD5 (plan-020 carry):
```
fold_id = int(md5(f"{sample_id}_{seed}").hexdigest()[:8], 16) % 5
seed = "plan020_stable_v1"  # plan-020/021/022/023/024 동일
```

OOF concat = 5 fold validation prediction 의 sample_id 순 concatenation. 길이 = N = 10000.

### §3.2 합격 기준 (G-gate 정량 정의)

| Gate | 정량 조건 | severe (위반 시) |
|:--|:--|:--|
| G0 | pytest ≥ 10/10 pass AND import error 0 | `infra_drift` |
| G1 | F0 hit_1cm ∈ [0.6315, 0.6325] AND hit_1.5cm ∈ [0.8028, 0.8038] AND plan-022 winner hit_1cm ∈ [0.6523, 0.6533] AND hit_1.5cm ∈ [0.8099, 0.8109] | `f0_reproduce_drift` / `plan022_carry_drift` |
| G2.E0 | hit_1cm ∈ [0.6360, 0.6385] (plan-024 0.6370 ± per-fold std 0.0034 × 1.5 ≈ 0.005, 단 단방향 0.0015 으로 strict) | `plan024_reproduce_drift` |
| G2.E1 | metric finite (NaN/Inf 0) AND max_class_ratio < 0.95 AND soft_CE finite | `pointwise_numerical` |
| G2.E2 | metric finite AND max_class_ratio < 0.95 AND group-by-sample softmax sum=1 invariant 통과 | `rowexp_numerical` |
| G2.E3 | metric finite AND anchor embed L2 norm < 10 (init 0.02 의 500×) | `embed_diverge` |
| G3 (band 판정) | max(E1, E2, E3) hit_1cm 의 위치:<br>- ≥ 0.6628 → **band strong_positive** (plan-024 G3 영역, arch lever 강력 회복)<br>- ≥ 0.6528 → **band positive** (plan-022 winner 동등 회복, arch lever salvageable)<br>- < 0.6528 → **band negative** (`expansion_no_recovery` warn, arch lever 자체 fail 확정, plan-026 분기) | `expansion_no_recovery` (warn only) |
| G_final | 3-file frontmatter sync (plan / results.md / .results.md pair) + §0.5 c1~c13 [DONE] + follow-up plan ≥ 2건 박제 + results.md ≥ 11 항목 | `g_final_incomplete` |

### §3.3 평가 점수

primary: **OOF hit_1cm** (= 5-fold concat prediction 의 1cm hit rate, plan-022/024 carry).

secondary (informational):
- OOF hit_1.5cm
- Δ_F0_1cm = hit_1cm − F0_hit_1cm
- Δ_F0_1.5cm = hit_1.5cm − F0_hit_1.5cm
- **gap_ranking** = oracle_1cm − argmax_hit (selector ranking 능력. plan-024 v1 = 0.1934, plan-009 = 0.1080 reference)
- **top1_acc** = argmax 후 ground truth anchor 와 일치율 (plan-024 v1 = 0.1227, plan-022 = 0.1707 reference)
- max_class_ratio = `probs_all.mean(axis=0).max()` (mode collapse 진단)
- soft_CE = cross entropy on q_target (= plan-024 §1 carry). log(14) ≈ 2.639 = uniform reference.
- dist_match_KL = KL(probs_mean ‖ q_target_mean)

### §3.4 per-fold variance 박제

per-fold hit_1cm std ≥ 0.005 (plan-024 v1 의 0.0034 의 1.5×) 시 `high_variance` warn 박제 (band 판정 영향 없음, 단 results.md 에 박제).

---

## §4. STAGE 0 — 인프라 (c2~c6 + G0)

### §4.1 module 작성 (c2/c3/c4/c5)

| module | symbol | 책임 |
|:--|:--|:--|
| `model_pointwise.py` | `PointwiseSelector` | plan-024 backbone 변형. forward(X, A) → q_ik scalar (B*K,). loss = MSE(q_ik, q_target_ik). |
| `model_row_expansion.py` | `RowExpansionSelector` | batch (B*K, D) → group-by sample_idx → softmax within group → listwise CE on q_target. |
| `anchor_embed.py` | `LearnableAnchorEmbed(K=14, D_embed=8)` | `nn.Parameter(K, D_embed)` init=0.02. forward(anchor_idx) → embed (B, K, D_embed). E3 only. |
| `expansion_runner.py` | `run_variant(E#, fold)` | 5-fold OOF runner. CLI: `--variant E0/E1/E2/E3 --fold 0..4`. 산출: `results_E#_fold{N}.json`. |

### §4.2 pytest (c6)

10 test 최소:
1. import 4 module without error
2. PointwiseSelector forward shape: (B, D), (K, A) → (B*K,) scalar
3. RowExpansionSelector forward shape: (B*K, D+A) → group-by softmax (B, K) sum=1 per row
4. LearnableAnchorEmbed shape: K=14, D_embed=8 → embed (B, 14, 8). norm finite.
5. soft label q_ik (= build_soft_label_with_tau τ=0.001) sum=1 per sample
6. F0 carry: f0_baseline 호출 → hit_1cm = 0.6320 ± 0.0005
7. samples/anchor floor: 10000 / 14 ≈ 714 sample/anchor > 100 minimum (plan-022 §3.3 carry)
8. mode collapse guard: random init forward 의 probs_all.mean(0).max() < 0.5 (init 시 < 0.5 면 학습 후 < 0.95 가능성 ↑)
9. per-batch invariant: pointwise variant 의 (B*K,) reshape → (B, K) 와 RowExpansion 의 (B*K, ...) → group-by 결과의 sample_idx 일치
10. anchor 14 BCC 좌표 invariant: ‖a‖ = 0.005m exact, np.unique == 14

### §4.3 산출물

- `analysis/plan-025/__init__.py`
- 위 4 module
- `tests/test_plan025_smoke.py`

### §4.4 종료 조건 (G0)

- 10/10 pytest pass
- import error 0
- 위반 시 `infra_drift` severe halt

---

## §5. STAGE 1~4 — Variant 사양 (c8~c11)

### §5.1 E0 — plan-024 v1.1-rev2 reproduce (control, c8 + G2.E0)

**목적**: drift 측정 + 모든 후속 variant 의 baseline.

**구성**:
- model: plan-024 `CandidateAttentionGRUSelector` 정확 carry (= analysis/plan-024/model.py import)
- input: 170D plan-021 carry (NOT plan-024 의 245D — single variable + redundancy 박제 후 차단)
- hyperparam: hidden=384, dropout 0.10/0.15, lr 7e-4 const, weight_decay 0.02, batch=32, epoch 22 (= plan-024 v1 carry), patience 8

**예상 결과**: hit_1cm 0.6370 ± 0.0015 (= plan-024 results §2 OOF 동일).

**제외**: 245D input 차이로 plan-024 v1 와 정확 일치 안 할 가능성. **G2.E0 tolerance = 0.6360~0.6385** (per-fold std × 1.5).

**산출**: `results_E0.json` (= plan-024 §1 13 metric 동일 schema).

**의의**: G2.E0 가 통과해야 E1/E2/E3 의 lift/drop 비교 가능. drift 시 halt.

### §5.2 E1 — Pointwise K-Independent Forward (c9 + G2.E1)

**목적**: LGBM sample-weight expansion 의 effective N 14× gap 직접 fix (root-cause #1).

**구성**:
- 학습 stage:
  - batch sampling: 매 step 에서 N 개 sample → (X_b ∈ R^{B×D}, q_target_b ∈ R^{B×K})
  - forward: 각 (sample i, anchor k) pair 마다 scalar pred_ik = MLP(concat([X_i, anchor_feat_k])). 동시에 B×K=B*14 forward.
  - loss: MSE(pred_ik, q_target_ik) over all B*K pair. **gradient unit = 1 (sample, anchor) pair** → effective N = N*K = 140k epoch 당.
- inference stage:
  - 각 sample 의 14 pred_ik 수집 → softmax-renormalize (학습 시 sigmoid scalar 라 sum ≠ 1) → weighted average with anchor coord → 3D pred.
- backbone: plan-024 의 CandidateAttentionGRUSelector 의 seq_encoder (GRU + Frenet pool) 부분 그대로 carry. 단 cross-attn cand attention 제거 → 각 anchor 가 독립 scalar 출력으로 MLP head 통과.

**아키텍처 변경 (vs E0)**:
```
E0: seq_feat (B, T=7, F=170) → GRU → ctx (B, 128) → cross_attn(ctx, cand_feat (K, A=22)) → logits (B, K) → softmax → CE
E1: seq_feat (B, T=7, F=170) → GRU → ctx (B, 128) → broadcast (B, K, 128) + cand_feat (B, K, A=22) → MLP(150 → 64 → 1) → pred (B, K) → sigmoid → MSE on q_target
```

**핵심 lever**: pointwise loss → 매 (sample, anchor) pair 의 grad 가 *독립*. 14 pair 가 sum-coupled (softmax CE) 가 아니라 개별 grad signal. LGBM tree split 의 single-row gain 패턴과 등가.

**예상 결과 (가설)**:
- arch lever 가 expansion 부재로 mask 되었다면: hit_1cm 0.6528 ± 0.005 (plan-022 winner 회복)
- arch lever 자체 fail 이면: hit_1cm 0.6370 ± 0.005 (plan-024 와 유사)

**산출**: `results_E1.json` + per-fold timing (E2 wall-time 예측 기준).

### §5.3 E2 — Row Expansion + Listwise CE (c10 + G2.E2)

**목적**: pointwise 의 *gradient unit 독립화* 와 listwise CE 의 *softmax 정상화* 의 hybrid.

**구성**:
- batch shape: (B*K, D+A) — 각 (sample, anchor) row 를 batch element 로 펼침.
- forward: scalar logit_ik = MLP(concat([X_i, anchor_feat_k])). shape (B*K,).
- group-by sample_idx (B*K → B groups of K) → softmax within group → probs_ik (B*K,).
- loss: CE(probs_ik, q_target_ik) 단 grad 가 B*K element-wise (= sample-grouped softmax 의 chain rule).
- gradient unit: softmax 의 K-group joint chain → group 내 K element 가 coupled, 단 batch 의 *cross-sample* element 는 독립 → effective batch B*K로 SGD step 의 noise/signal ratio 가 *B 만* 보는 E0 대비 K 배 향상.

**아키텍처 변경 (vs E1)**:
- E1 의 sigmoid → group softmax (sample 내 normalize).
- E1 의 MSE → CE (listwise).

**의의**: pointwise (E1) 가 sigmoid 의 sum ≠ 1 의 unphysicality 를 inference 시 renormalize 로 풀지만, E2 는 학습 시 softmax 정상화 → q_target 분포와 직접 align.

**예상 결과**: E1 와 E2 의 차이가 "pointwise sigmoid + renorm" vs "listwise softmax" 의 *학습 분포 align* 효과. E2 가 ≥ E1 이면 listwise 가 본질, < E1 이면 pointwise 의 gradient unit 독립이 본질.

**산출**: `results_E2.json` + group-by-sample softmax sum=1 invariant check log.

### §5.4 E3 — E1 + A7 Learnable Anchor Embedding (c11 + G2.E3)

**전제**: E1 / E2 중 hit_1cm 가 더 높은 *winner variant* 위에 A7 lever 추가. 단순화 위해 E1 기반으로 spec 명시 (E2 가 winner 시 c11 spec 만 swap, decision-note 박제).

**목적**: plan-024 v5 의 A7 finding (top1_acc 0.1227 → 0.1389, +0.0162) carry. anchor identity capacity 부족 (cand_dim 22D 중 anchor-discriminating 22D 만, 128D ctx 는 broadcast) 의 직접 fix.

**구성**:
- 추가 module: `nn.Parameter(K=14, D_embed=8)` init=Normal(0, 0.02) (Transformer convention).
- 학습: standard backprop, weight_decay 0.02 (= 다른 param 과 동일).
- forward 시: anchor_idx (0..13) lookup → embed (B, K, 8) → cand_feat 와 concat → (B, K, 30) = 22+8.
- 추가 params: 14 × 8 = 112 (= plan-024 v5 carry).

**의의**:
- E1 의 expansion 효과 위에 anchor 14개 *차별화 capacity* 추가.
- 두 lever 가 *orthogonal* (one is data leverage, other is param capacity) → 둘이 add 또는 multiply lift 측정.

**예상 결과**:
- E1 hit_1cm + ≈ 0.005 (plan-024 v5 의 top1_acc 0.0162 lift 가 hit_1cm 로 일부 transfer 가정)
- 또는 0 (plan-024 v5 의 hit_1cm lift +0.0002 가 expansion 환경에서도 동일하면)

**산출**: `results_E3.json` + anchor embed final L2 norm 박제.

---

## §6. expansion_runner.py 사양 (c5)

### §6.1 CLI

```bash
python -m analysis.plan-025.expansion_runner \
    --variant {E0|E1|E2|E3} \
    --fold {0..4|all} \
    --output_dir analysis/plan-025/runs/{variant}/
```

`--fold all` = 5-fold sequential. 산출:
- `results_{variant}_fold{N}.json` per fold
- `results_{variant}.json` = OOF concat metric

### §6.2 5-fold OOF concat

plan-022 `LgbmSelectorOnly.predict_oof` 패턴 carry:
1. fold k=0..4: train on fold ≠ k, predict on fold == k
2. concat 5 prediction (sample_id 순)
3. metric on concat = OOF metric

### §6.3 산출 schema

```json
{
  "variant": "E1",
  "metrics": {
    "hit_1cm": 0.65xx,
    "hit_1.5cm": 0.80xx,
    "delta_f0_1cm": 0.0xxx,
    "delta_f0_1.5cm": 0.0xxx,
    "gap_ranking": 0.0xxx,
    "top1_acc": 0.1xxx,
    "max_class_ratio": 0.0xxx,
    "soft_ce": 2.xxx,
    "dist_match_kl": 0.00xx,
    "oracle_1cm": 0.7928,
    "argmax_hit": 0.xxxx
  },
  "per_fold": [{"fold": k, "hit_1cm": .., "time_sec": ..}, ...],
  "total_time_sec": ...,
  "config": {"variant": ..., "hidden": 384, "lr": 7e-4, ...},
  "dataset_hash": "b91502db94fab67d"
}
```

---

## §7. paradigm_analysis (c12)

### §7.1 4 variant × 5-fold metric 표

| variant | hit_1cm | hit_1.5cm | gap_ranking | top1_acc | soft_CE | max_class_ratio | time |
|:--|--:|--:|--:|--:|--:|--:|--:|
| E0 (plan-024 reproduce) | 0.637? | 0.809? | 0.193? | 0.123? | 2.57? | 0.105? | ~167s |
| E1 (pointwise expansion) | ? | ? | ? | ? | ? | ? | ~140s estimate |
| E2 (row listwise) | ? | ? | ? | ? | ? | ? | ~700s estimate (B*K batch) |
| E3 (E1 + A7) | ? | ? | ? | ? | ? | ? | ~155s estimate |

### §7.2 plan-022 / plan-024 head-to-head 표

| plan | model | input | OOF hit_1cm | Δ vs plan-022 |
|:--|:--|:--|--:|--:|
| plan-022 winner | LGBM + sample-weight expansion (effective 140k) | 170D | 0.6528 | — |
| plan-024 v1.1-rev2 | cross-attn (effective 10k) + 245D | 245D | 0.6370 | −0.0158 |
| plan-025 E0 (reproduce) | cross-attn (effective 10k) + 170D | 170D | 0.637? | ≈ −0.016 (예상) |
| plan-025 E1/E2/E3 | cross-attn + expansion (effective 140k) + 170D | 170D | ?? | ?? |

### §7.3 arch lever 분리 결론

3 가능 branch:
- **branch A (band strong_positive)**: max(E1,E2,E3) ≥ 0.6628 → arch lever 가 LGBM 의 14× data leverage 위에서 *추가 lift* 가능. plan-027 ensemble + plan-026 dynamic anchor 동시 lever.
- **branch B (band positive)**: 0.6528 ≤ max(E1,E2,E3) < 0.6628 → arch lever 가 expansion 부재로 mask. plan-022 winner 동등 회복. plan-024 의 G2 FAIL 의 80~100% 가 root-cause #1. plan-026 dynamic anchor 가 추가 +0.005~0.01 lift 가능성 ↑.
- **branch C (band negative)**: max(E1,E2,E3) < 0.6528 → arch lever 자체 fail 확정. root-cause #1 외에 #2 (static anchor mismatch) 또는 다른 hidden lever 가 본질. plan-026 dynamic anchor 가 *유일* 회복 lever 후보.

### §7.4 soft_CE deviation 박제

soft_CE - log(14) (= 2.639) 표:
- plan-024 v1: 2.566 - 2.639 = -0.073 (near-uniform plateau)
- E0: 예상 ≈ -0.073
- E1/E2/E3 의 deviation 이 더 negative (선명 분포) 이면 expansion 효과 = "uniform 탈출" 의 명시 증거.

---

## §8. results.md 필수 항목 (c13, §N+2)

11~13 항목 (plan-024 results.md §1~§11 schema carry):

1. 한 줄 결론 + band 판정
2. §1 OOF metric table (4 variant × 8 metric)
3. §2 per-fold variance
4. §3 G-gate 결과
5. §4 plan-024 v1 vs plan-022 head-to-head
6. §5 fail/success 원인 분석 (band 별 분기)
7. §6 measurable headroom (oracle_1cm, argmax_hit, gap_ranking)
8. §7 LB-OOF gap 비교 (informational, LB 미회수)
9. §8 ablation slot (plan-026 영역 lever 후보)
10. §9 comparison table (plan-022/023/024/025)
11. §10 follow-up plan 후보 (plan-026/027/028)
12. §11 paths & artifacts
13. §12 commit chain final state (§0.5 sync)

---

## §9. 작업량 총 회계

### §9.1 commit chain

- c1 (spec) + c1.5 (optional plan-review) = 1~2 commit
- c2~c6 (module + test) = 5 commit
- G0 = 1 sync commit (또는 c6 안에 흡수)
- c7 (G1) = 1 commit
- c8~c11 (E0~E3 OOF) = 4 commit
- c12 (paradigm) = 1 commit
- c13 (results + 3-file sync) = 1 commit

→ 총 12~14 commit (= plan-022/023/024 range 내)

### §9.2 compute budget

| variant | 5-fold OOF wall time 예상 | 근거 |
|:--|--:|:--|
| E0 (plan-024 reproduce) | ~170s | plan-024 v1 의 167s carry |
| E1 (pointwise) | ~140s | sigmoid scalar 출력 (CE softmax 없음) 으로 약간 빠름 |
| E2 (row expansion) | ~700s | batch B*K=32*14=448, K× wall time |
| E3 (E1 + A7) | ~155s | E1 + nn.Parameter (overhead ~ 5%) |

→ 4 variant 총 ~1200s ≈ 20분 (CPU). compute halt risk 낮음.

### §9.3 cell / task / unit

- variant cell: 4 (E0/E1/E2/E3)
- per-variant fold: 5
- 총 measurement unit: 4 × 5 = 20

---

## §N+3. 통계 함정 & caveats

### caveat #1: E1 의 sigmoid scalar 의 inference renormalize 가 *학습-inference distribution mismatch* 만들 수 있음

E1 학습 = sigmoid scalar (각 q_ik 가 independent [0,1]). Inference = 14 sigmoid 출력의 softmax renormalize. 학습 시 본 적 없는 변환.

**완화**: E2 의 softmax 학습이 비교 control. E1 vs E2 의 metric 차이가 이 distribution mismatch 효과의 직접 측정.

### caveat #2: E2 의 row expansion 의 batch_size 의미 변화

E0 batch=32 = 32 sample. E2 batch=32 = 32 (sample, anchor) row = ~2.3 sample. → 학습 dynamics (lr schedule, momentum) 가 다른 unit 에서 작동.

**완화**: E2 의 batch=448 (= 32 * 14) 로 일치화 — sample-level batch 32 유지. CLI option `--batch_size_unit {sample|row}` 박제.

### caveat #3: per-fold std 0.0034 와 E0~E3 의 lift 측정 가능성

plan-024 v1 의 per-fold std = 0.0034. 5-fold OOF concat std ≈ 0.0015 추정 (CLT). E1/E2/E3 의 lift 0.005 검출은 std 의 3× 이내 → seed noise 위에서 측정 가능. 단 0.005 미만 lift 는 *불검출*.

**완화**: lift > 0.010 만 "real signal" 로 박제. < 0.010 lift = "borderline" 박제 후 plan-026 영역에서 재측정.

### caveat #4: A7 anchor embedding 의 14 × 8 = 112 params 의 micro-capacity 가 overfit 의 minor 추가

plan-024 v5 의 안 됨이 hit_1cm 0.6372 (v1 0.6370 의 +0.0002) — capacity 효과 거의 0. E1/E2 의 expansion 환경에서도 비슷할 risk.

**완화**: E3 vs E1 의 hit_1cm Δ < 0.003 = 무효 박제 (= seed noise band 내). plan-026 영역으로 carry.

### caveat #5: 14 BCC anchor 고정의 implicit limit (= plan-024 root-cause #2 의 carry)

본 plan 이 root-cause #1 만 fix → root-cause #2 (static anchor) 가 잔존. branch C (band negative) 발생 시 root-cause #2 가 본질일 가능성 박제.

### caveat #6: 170D plan-021 carry 가 plan-024 245D 의 *informative subset* 인 가설

v6 ablation (LGBM + 245D → +0.0003) 은 LGBM 환경. cross-attn 환경에서도 245D 가 redundant 인지 미측정. 본 plan 은 *170D 가정* 으로 진행, 245D 는 plan-026 영역.

### caveat #7: E1 의 pointwise MSE vs BCE 선택

q_target ∈ [0,1] regression. plan-022 LGBM 은 default regression (MSE). 단 sigmoid + BCE 가 saturation 영역 (q ≈ 0 또는 1) 에서 더 안정. **default = MSE** (plan-022 carry), BCE 는 caveat 으로만 박제 후 plan-026 영역.

### caveat #8: epoch budget plan-024 v1 carry (22) 의 *under-fit* risk

plan-024 v1 best ep 35 (long-diag, 1-fold) vs spec ep 22 (5-fold). 본 plan 의 ep 22 도 잠재 under-fit risk. 단 patience 8 early stop 으로 mitigate.

**완화**: per-fold best ep + final ep 박제. best ep > 18 (= 0.82 × 22) 시 `epoch_undertrain` warn 박제 (단 halt 아님).

### caveat #9: max_class_ratio = q_true.mean 자연 분포 의 mirror (plan-022 §12 post-G_final 박제)

plan-022 의 A8 ablation 박제: `max_class_ratio` 가 *mode collapse* 가 아니라 *q_true.mean 의 mirror* 일 가능성. 본 plan 도 동일. **추가 metric**: `dist_match_KL` 과 `top1_acc` 로 collapse 진단 분리 측정.

### caveat #10: gap_ranking 의 해석 (plan-024 §6 carry)

gap_ranking = oracle_1cm − argmax_hit = selector ranking 능력. plan-024 v1 = 0.1934 (oracle 0.7928 − argmax 0.5994). plan-025 의 expansion 이 ranking 능력 회복 시 gap_ranking ↓ 예상. ranking 회복 정도 = arch lever 의 *진짜* 가치.

---

## §N+4. 변경 이력

- v1 (2026-05-21): 초안. plan-024 사후분석 4축 (results.md §5) + 사용자 명시 "expansion mimic 우선" choice 박제. 단일 변수 = effective N 메커니즘. 4 variant E0/E1/E2/E3. anchor 14 BCC + input 170D + τ=0.001 carry.

---

## §N+5. 참조

- `plans/plan-024-cross-attention-anchor-vocab.md` v1.1-rev2 + results.md §5 (4축 사후분석 anchor)
- `plans/plan-022-corrector-free-anchor-layout-sweep.md` (sample-weight expansion 패턴 + 14 BCC + τ=0.001 winner)
- `plans/plan-021-frenet-corrector-input-augment.md` (170D LGBM input carry)
- `plans/plan-020-f0-structural-search.md` (F0 baseline + 5-fold stable_fold_id)
- `WORKFLOW.md §1~§12` (전체 plan/results/registry/autonomous protocol)
- `CLAUDE.md` (autonomous execution policy)
- `analysis/plan-024/results.md` §10 (follow-up plan-025/026/027 박제)
