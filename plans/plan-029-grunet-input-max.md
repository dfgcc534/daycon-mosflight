---
plan_id: 029
version: 1.0
date: 2026-05-22 (Asia/Seoul)
status: complete
best_cell: X1
best_hit_1cm: 0.6316
best_hit_1p5cm: 0.8039
based_on:
  - 022 (winner A6_bcc14_tau001 OOF 0.6531 / 0.8108. K=14 BCC + τ=0.001. selector-only LGBM 170D baseline)
  - 024 (cross-attention GRU selector. honest ceiling = 0.6387 3-seed (§5.14) / 0.6375 ± 0.0004 4-way OOF plateau (§5.13). under-converged/over-reg/anchor-identity 가설 모두 기각/부분 기각. **본질적 fail diagnosis (사용자, 2026-05-22)**: plan-024 `query = query_mlp(cand_feat)` 의 cand_feat 150D 중 sample × anchor *interaction* channel 은 13D (par/perp/dist 3 + interactions 10) 만, 나머지 137D 는 sample-broadcast ctx 128D + anchor-only spec 9D = *각 축에서 invariant heavy*. attention 의 query 가 anchor 별로 sample-specific 신호 부족 → attention discrimination 학습 자체 불가. plan-024 의 모든 plateau (0.6370~0.6387) 는 이 sample-invariant query 위에서의 ceiling. 본 plan = paradigm framework 유지 + **핵심 4 lever**: (a) **query enrichment** (cand_feat 에 sample × anchor interaction channel ≥15 추가 — past trajectory point displacement to anchor + anchor·v 시계열 + regime × anchor cross + cosine sim), (b) **anchor embedding 학습** (learnable 14×8, init=randn×0.1, query + key 양쪽 broadcast), (c) **key anchor-conditional** (per-anchor key projection 또는 broadcast add → key 가 anchor 별 다른 sequence representation), (d) **보조: head raw skip 차단** (head_in = event_ctx 196D only — PB framework carry 의 cand_feat raw 직통 제거, attention path effect isolation))
  - 025 (LGBM + 후보 concat + seq 압축 → mode collapse → F0 복사. block ③ LGBM tree categorical split evidence. 본 plan = plan-025 1080D 중 cand_feat 150D (block ②③) 만 *cross-attention query* 로 사용. block ① 170D + block ④ 760D 는 head skip / GRU input 둘 다 사용 X (out-of-scope, plan-030 후보))
  - 020 (F0 baseline 0.6320/0.8033 + stable_fold_id MD5)
inspired_by:
  - 사용자 (2026-05-22): "plan-026, 027 은 gru-attention" — abandoned LGBM 026/027 의 통합 재발행. paradigm-level 검증 1회 plan.
  - plan-025 paradigm mismatch finding: block ③ 22D per-anchor 가 LGBM 에서 self-prediction trigger 였지만 GRU-attention 에서는 query 의 anchor identity 로 정상 작동 예상.
code_reuse:
  - module: analysis/plan-024/model.py (worktree-plan-024-combo branch 의 c2 cherry-pick 필요)
    symbols: [CandidateAttentionGRUSelectorCarry]
    reason: 본 plan 의 backbone (GRU + query_mlp + cross-attention computation 만 carry). hidden=384 → 196 변경. `backbone.head` (hardcoded `Linear(2H+cand_dim=542, hidden=196)`) 는 **사용 X** — 본 plan head_in 이 event_ctx (196D) only 이므로 backbone.head 의 542D input dim 과 mismatch 이기도 하고, 더 본질적으로 backbone.head 가 raw cand_feat 150D 를 concat 받는 PB framework carry 자체가 *본 plan 이 제거하려는 paradigm-confound source*. wrapper `GRUNetX1` 안에서 backbone 의 forward 일부 (GRU + query_mlp + cross-attention) 만 재사용 (또는 subclass + `self.head=nn.Identity()` overwrite). `CrossAttentionAnchorSelector` outer wrapper (FWD wrapping) 는 import 안 함 (FWD off).
  - module: analysis/plan-024/cand_builder.py
    symbols: [build]
    reason: cand_feat 150D 산출. plan-025 build_feat_1080 의 source.
  - module: analysis/plan-024/seq_builder.py
    symbols: [build]
    reason: seq 95D × 7 step. GRU encoder input source (= plan-025 block ④ 의 raw 형태, 8-stat 압축 전).
  - module: analysis/plan-024/anchor_vocab.py
    symbols: [build]
    reason: seq_builder internal.
  - module: analysis/plan-024/torsion_calc.py
    symbols: [build]
    reason: seq_builder internal.
  - module: analysis/plan-024/quantile_carry.py
    symbols: [build, QuantileCarry]
    reason: fold-leakage 차단 quantile carry (omega_p90, jerk_p90).
  - module: analysis/plan-024/multiwindow_trim_build.py
    symbols: [load_trim]
    reason: 144→60 trim index.
  - module: analysis/plan-024/feature_weighted_dropout.py (worktree-plan-024-combo cherry-pick 필요)
    symbols: [FeatureWeightedDropout]
    reason: plan-024 의 input dropout lever. 본 plan training schedule 재설계의 일부.
  - module: analysis/plan-025/build_feat_1080.py
    symbols: [build_feat_1080, BLOCK_DIMS]
    reason: import 만 carry (G0 smoke test 정합성 검증용). 본 plan 은 1080D 직접 사용 X — head skip 전체 제거 후 head_in = event_ctx (196D) only 이므로 block ① 170D + block ④ 760D source 자체 unused. plan-030 후보 (head expansion lever 부활 시 source).
  - module: analysis/plan-022/anchors.py
    symbols: [ANCHORS_A6]
    reason: K=14 BCC anchor codebook (plan-022 winner carry).
  - module: analysis/plan-022/selector_only_model.py
    symbols: [build_soft_label_with_tau]
    reason: soft label 산식.
  - module: analysis/plan-021/build_input.py
    symbols: [build_frenet_basis_3d, build_input_common, build_input_lgbm_extra]
    reason: 170D plan-022 carry input pipeline (block ①).
  - module: analysis/plan-020/baseline_f0.py
    symbols: [f0_baseline, D1, R_HIT, R_HIT_LOOSE]
    reason: F0 baseline + hit metric.
  - module: src/io.py
    symbols: [load_all_samples, load_labels]
    reason: data loader.
  - module: src/pb_0_6822/selector.py
    symbols: [stable_fold_id, fit_regime_bins, assign_regimes]
    reason: 5-fold split + regime assignment.
supersedes_abandoned:
  - 026 (LGBM block ablation, user intent mismatch)
  - 027 (LGBM 3-way ensemble, user intent mismatch)
followed_by:
  - plan-030 (가칭, paradigm-distinct lever 전환 우선 — F0 ML / corrector / KNN. plan-029 G3 = regression 결과 따라 single-lever ablation 우선순위 낮음 — 4 lever 모두 적용해도 regression 이므로 단독 ablation 도 유사 예상)
scope: plan-024 cross-attention GRU selector paradigm framework 유지 + **핵심 4 lever (attention path 강화 중심)**: (a) **query enrichment** — `analysis/plan-029/anchor_query_extend.py` 신규 wrapper 로 plan-024 cand_builder 150D 위에 **N_new=15 channel** (5 group: A.dist 5 + A.tangent_proj 3 + B.cos 1 + D.regime_anchor_prob 1 + F.2 multi-step anchor·v 5) 추가 → cand_ext (B, 14, 165). 확장 후 dim = 165. (b) **anchor embedding 학습** — `nn.Parameter(K=14, d_embed=8)` learnable, **init = randn × 0.1** (사용자 결정, plan-024 v5 의 0.02 carry 대비 5× 증가 — lever (c) 환경 보정). query 와 key 양쪽 broadcast concat. (c) **key anchor-conditional** — key (B, T=7, hidden) → (B, K=14, T=7, hidden) anchor 별 sequence representation. 식 = `key_anchor[b,k,t,:] = key[b,t,:] + linear_proj(anchor_embed[k,:])` (broadcast add 권장 default). attn_logits = einsum("bkh,bkth->bkt", query, key_anchor) / sqrt(hidden). (d) **보조 head raw skip 차단** — head_in = event_ctx (196D) only (h_final_bc / cand_ext / block1_bc / block4_bc 모두 head 에 부재). PB framework carry 의 raw cand_feat 150D 직통 (paradigm-confound) 제거 → attention path effect isolation. 보조 sub-decision: hidden 196 (사용자 명시), batch 64 (effective step 4×). FWD off (§5.4 기각, outer wrapper 미import). GRU encoder input = raw seq (B, 7, 95). head = `Linear(196, 1)` 단순. training schedule = 50 epoch fixed (early stop disabled, §5.1 under-conv 기각 후 cosine annealing 완주 lever), lr=7e-4 + SequentialLR([LinearLR warmup 5 ep, CosineAnnealingLR T_max=45 ep]), AdamW (wd=1e-4), GRU dropout=0.10, gradient_clip=1.0, soft cross-entropy loss. K=14 BCC + τ_cls=0.001 fix (plan-022 carry). 5-fold stable_fold_id. ensemble / DACON LB / corrector / F0 ML / augmentation (plan-024 §5.10 poss 3 carry) / head raw skip 부활 (plan-030 후보) / h_final_bc head 추가 (plan-030 후보) / head 2-layer MLP (plan-030 후보) = out-of-scope.
exp_ids:
  - Z029_X1_gru_h196
lb_score: null
band: regression
---

# plan-029 v1 — GRU-attention Input Max (hidden=196, 1080D + raw seq)

## §0. 한 줄 목적

> **plan-024 의 fail 진단 (사용자, 2026-05-22): query 가 sample invariant 했기 때문 — cand_feat 150D 중 sample × anchor interaction 13D 만, 137D 가 sample-broadcast 또는 anchor-only invariant. attention 의 query 가 anchor 별 sample-specific 신호 부족 → discrimination 학습 자체 불가**. 본 plan = paradigm framework 유지 + **attention path 강화 4 lever**: (a) query enrichment (cand_feat 에 sample × anchor interaction channel 추가), (b) anchor embedding 학습 (query + key 양쪽), (c) key anchor-conditional projection, (d) 보조 head raw skip 차단. attention path 가 *진짜* 학습되도록 query/key 양쪽에 anchor 정보 풍부히 박은 후 attention output 단독으로 score → softmax → soft CE. **abandoned plan-026 + plan-027 의 통합 재발행** (사용자 plan-026/027 GRU-attention 의도 합의).
>
> **paradigm rationale** (사용자 진단 + plan-024 results.md §5.1+§5.8+§5.9+§5.10+§5.13+§5.14 종합):
> 1. **plan-024 의 본질적 fail (사용자, 2026-05-22)**: `query = query_mlp(cand_feat)` 에서 cand_feat (B, K=14, 150) 의 channel 구성 = par/perp/dist 3D + anchor spec 9D + ctx broadcast 128D + interactions 10D. sample × anchor *interaction* (b,k 마다 모두 다른 값) = 3 + 10 = **13D 만**. ctx 128D 는 anchor 무관 (K 축 broadcast 동일), anchor spec 9D 는 sample 무관 (B 축 broadcast 동일). attention 의 query 가 anchor 별 differential 정보 13D 위에서만 작동 → attention scoring 의 discrimination 학습 자체 불가. plan-024 의 v1~v5 / poss 1~3 / long-diag / combo / 3-seed *모든 variant* 가 이 sample-invariant query 위에서 학습 → 모든 ablation 이 0.6370~0.6387 plateau 에 묶인 *진짜 root cause*. CPU under-converged §5.1 / over-reg §5.4 / anchor identity A7 §5.8 등 plan-024 가 시도한 가설은 모두 *query 약함 위에서의 보조 lever 효과* 측정이었음 → 모두 plateau 안 noise.
> 2. **본 plan 의 핵심 lever (4 axis 동시)**:
>     - (a) **query enrichment** (N_new=15 박제, 사용자 확정): 신규 `analysis/plan-029/anchor_query_extend.py` 에서 plan-024 cand_builder 150D 위에 *sample × anchor interaction channel 15개* 추가 → cand_ext (B, 14, 165). 5 group: **A.dist** (past 5 step t=5..9 의 anchor world distance norm, 5 ch) + **A.tangent_proj** (past 3 step t=8..10 의 Frenet 0-axis projection, 3 ch) + **B.cos** (anchor_dir vs velocity cosine sim, 1 ch) + **D.regime_anchor_prob** (`P(gt=k | regime[b])` train-fold empirical lookup table, fold-leakage 차단, 1 ch) + **F.2 multi-step anchor·v** (t∈{5..9} 의 anchor·v_frenet 시계열, 5 ch). plan-024 ④ interactions 10 ch (single t=10 만) 의 *시간 axis 확장* 이 핵심 novel — novel ≈67% (sub-agent 실측). 사용자 진단 ("query sample invariant") 의 직접 fix.
>     - (b) **anchor embedding 학습**: `nn.Parameter(K=14, d_embed=8)` learnable, **init = randn × 0.1** (사용자 결정, plan-024 v5 의 0.02 carry 대비 5× 증가 — lever (c) 환경 보정). query 와 key 양쪽 broadcast concat 또는 add. plan-024 v5 (§5.8) 는 query 에만 broadcast concat + init 0.02 → "부분 기각 (hit_1cm 변화 X)" 였으나 본 plan 은 *key 에도 동시 적용* + query enrichment 환경 + init 5× → 환경 자체 다름.
>     - (c) **key anchor-conditional**: key (B, T=7, hidden=196) 에 anchor embedding broadcast add → key_anchor (B, K=14, T=7, hidden). 식: `key_anchor[b,k,t,:] = key[b,t,:] + anchor_proj(anchor_embed[k,:])`. attn_logits = einsum("bkh,bkth->bkt", query, key_anchor) / sqrt(196). 즉 key 가 anchor 별로 다른 sequence representation. plan-024 attention 식은 key 가 anchor 무관 (B, T, hidden) — 본 plan 은 key 도 anchor-conditional.
>     - (d) **보조 head raw skip 차단**: head_in = event_ctx (B, 14, 196) only. h_final_bc / cand_feat / block1_bc / block4_bc 모두 head 에 부재. head = `Linear(196, 1)` 단순. PB framework carry 의 raw cand_feat 직통 제거 → attention path 가 score 의 main 신호.
> 3. plan-025 (1080D LGBM) mode collapse = block ③ 22D 가 LGBM tree categorical split 으로 anchor identity memorize. 본 plan head 는 attention output 단독 → cand_feat / block ③ 가 head 에 *raw 부재* → self-prediction trigger 경로 물리적 부재. lever (d) 가 이 mechanism 차단.
>
> **단일 cell (paradigm-level 검증 1회 plan, 4 lever 동시 적용)**:
> - **X1** = GRU(hidden=196) + cross-attention(query=cand_ext 165D + anchor_embed 8D = 173D, key=GRU out 196D + anchor_key_proj(anchor_embed 8D) broadcast = (B, K, T, 196)) + head=Linear(event_ctx 196D → 1)
> - training schedule: epoch=50 fixed (no early stop), lr=7e-4 SequentialLR([warmup 5 ep, cosine 45 ep]), AdamW (wd=1e-4), GRU dropout=0.10, gradient_clip=1.0, batch=64
> - **head 단독 param ≈ 197** (Linear 196×1 + bias). attention path 강화 lever (a)(b)(c) 가 model capacity 의 main carrier.
>
> **pass criterion (G3)** (paradigm = attention path 강화의 진짜 효과 검증):
> - **PASS**: hit_1cm > 0.6528 → 4 lever 동시 적용 attention 강화 가 plan-022 LGBM floor 회복. plan-024 plateau 0.6387 의 +0.014 lift = query enrichment + anchor embedding + key conditional 의 합산 효과 입증.
> - **partial_above_p024**: 0.6387 < hit_1cm ≤ 0.6528 → attention 강화 가 plan-024 ceiling 위로 lift 단 LGBM floor 미달. 4 lever 중 일부 contribution. follow-up = single-lever ablation (query enrichment only / anchor embed only / key conditional only) 분해.
> - **partial_below_p024**: 0.6320 ≤ hit_1cm ≤ 0.6387 → attention 강화 가 plan-024 plateau 위로 lift 없음. *사용자 진단 (query sample invariant) 자체가 fail 의 root cause 가 아니었을* 가능성 — 다른 paradigm-distinct lever (F0 ML / corrector / KNN) 로 전환.
> - **regression**: hit_1cm < 0.6320 → 4 lever 동시 적용 가 *오히려 학습 destabilize*. attention path 강화가 model 안정성 해친 결과. follow-up = sub-lever ablation (각각 단독 효과 측정).
>
> **out-of-scope**: ensemble (plan-030 후보) / DACON LB submit / boundary corrector / F0 ML / anchor layout 변경 / τ_cls 변경 / hidden ≠ 196 sweep / batch ≠ 64 / **head raw skip 부활** (= plan-024 carry head 또는 1472D head 둘 다 plan-030 후보) / **h_final_bc head 추가** (= optional lever, plan-030 후보) / **single-lever ablation 분해** (= H1 결과 후 plan-030).

---

## §0.5 Quick Reference (autonomous loop 매 turn 읽는 section)

### 합격 기준 (G-gate sequence)

- **G0**: 5 module (model / run_oof / train / tests + plan-024 cherry-pick 9 file) import + smoke + tests green. plan-022 / plan-024 / plan-025 carry import 정상. 위반 시 `infra_drift` severe.
- **G1**: F0 baseline + plan-022 winner reproduce (plan-025 baseline_carry.json carry 또는 재산출). hit_1cm F0 ∈ [0.6315, 0.6325] AND plan-022 ∈ [0.6523, 0.6533]. 위반 시 `f0_reproduce_drift` / `plan022_reproduce_drift` severe.
- **G2.X1**: X1 cell 5-fold OOF metric finite + `max_class_ratio < 0.95`. mode collapse 표시 = `max_class_ratio ∈ [0.05, 0.1]` (near 1/K=0.071) — paradigm mismatch evidence. 위반 시 `numerical` severe / `mode_collapse` warn.
- **G3 (paradigm)**: PASS (>0.6528 — attention 단독 LGBM floor 회복) / partial_above_p024 (0.6387~0.6528 — attention 단독 plan-024 ceiling 위) / partial_below_p024 (0.6320~0.6387 — attention 단독은 plan-024 plateau 내, raw skip 이 lift source 였던 evidence) / regression (<0.6320 — attention 단독은 F0 미달, raw signal essential 결론) 판정 (§0 criterion).
- **G_final**: results.md + 3-file frontmatter sync + follow-up plan-030 (가칭) 박제.

### G-gates

- G0: STAGE 0 인프라 + plan-024 cherry-pick (model.py + feature_weighted_dropout.py) [DONE] 19 pytest green, 3.67s
- G1: STAGE 1 F0 + plan-022 winner reproduce [DONE] F0 0.6320 / p022 0.6531 tight band ✓
- G2.X1: X1 5-fold OOF [DONE] hit_1cm 0.6316, max_class_ratio 0.1328 < 0.95, all grad_ep5 > 1e-4
- G3: STAGE 3 paradigm 판정 [DONE] **regression** (0.6316 < F0 0.6320)
- G_final: STAGE 4 results + 3-file sync [DONE]

### Commit chain (next-up)

| # | type | spec section | status |
|---|---|---|---|
| c1 | docs | `plans/plan-029-grunet-input-max.md` v1 작성 | [DONE] |
| c2 | chore | plan-024 추가 cherry-pick from `worktree-plan-024-combo` (commit 915dd26): `model.py` + `feature_weighted_dropout.py`. 기존 plan-025 cherry-pick (anchor_vocab/cand_builder/seq_builder/torsion_calc/quantile_carry/multiwindow_trim_build + json + __init__) 외 추가 2 file. | [DONE] |
| c3 | code | `analysis/plan-029/anchor_query_extend.py` — 신규 wrapper. plan-024 `cand_builder.build()` 호출 후 sample × anchor interaction channel **15개** 추가 (5 group: A.dist 5 + A.tangent_proj 3 + B.cos 1 + D.regime_anchor_prob 1 + F.2 multi-step anchor·v 5). 출력 shape (B, K=14, 165). **signature**: `build(X, R_wfn, pred_F0_world, anchors, f0_baseline_fn, regimes, quantile_carry, multiwindow_trim_path, regime_count, regime_anchor_table: np.ndarray \| None = None)`. 마지막 arg = D channel 의 train-fold lookup table, shape `(regime_count, K)` float32 row-sum=1 (fold-leakage 차단, train.py 에서 fold-별 산출 후 inject). | [DONE] |
| c4 | code | `analysis/plan-029/model.py` — 신규 `GRUNetX1` class. 4 lever (a)(b)(c)(d) 통합 구현. plan-024 `CandidateAttentionGRUSelectorCarry` 의 GRU + query_mlp 는 **architecture template carry only** (class import 아닌 design pattern 재사용, dim 본 plan 자체 — query_mlp input 173 ≠ plan-024 150), instance 는 본 plan 안에서 fresh 생성. backbone.head + FWD wrapper class 둘 다 import X. **forward signature = `forward(self, seq, cand_ext) -> score`** (cand_ext 외부 사전 산출 후 주입, model 내부 anchor_query_extend.build 호출 X). **반환 = score (B,K) 단일 tensor** (tuple 분기 없음). **신규 design**: (i) `self.anchor_embed = nn.Parameter(torch.randn(14, 8) * anchor_embed_init_scale)` (anchor identity learnable embedding, default `anchor_embed_init_scale=0.1` 사용자 확정); (ii) `self.anchor_key_proj = nn.Linear(8, 196)` (anchor embedding → key dim); (iii) forward: input cand_ext (B,K,165) 외부 산출본 → `query_in = cat([cand_ext, anchor_embed.broadcast(B,K,8)], dim=-1)` (B,K,173), `query = query_mlp(query_in)` (B,K,196); (iv) `key = gru(seq).out` (B,T,196), `key_anchor = key.unsqueeze(1) + anchor_key_proj(anchor_embed).unsqueeze(0).unsqueeze(2)` (B,K,T,196) — broadcast add; (v) `attn_logits = einsum("bkh,bkth->bkt", query, key_anchor) / sqrt(196)`, `attn = softmax(dim=-1)`, `event_ctx = einsum("bkt,bkth->bkh", attn, key_anchor)` (B,K,196); (vi) **head = `Linear(196, 1)` 단순**: `score = head(event_ctx).squeeze(-1)` (B, K). hidden=196, GRU dropout=0.10. | [DONE] |
| c5 | code | `analysis/plan-029/train.py` — PyTorch 5-fold OOF training loop (epoch=50 fixed, lr=7e-4 SequentialLR[warmup 5 ep + cosine T_max=45 ep], AdamW wd=1e-4, GRU dropout=0.10, gradient_clip=1.0, batch=64, soft cross-entropy loss, `model.train()` 명시). | [DONE] |
| c6 | code | `analysis/plan-029/run_oof.py` — orchestrator + G1 reproduce + 5-fold concat OOF + final metric. CLI `--cell X1` 또는 `--g1`. | [DONE] |
| c7 | test | `tests/test_plan029_smoke.py` — 15+ pytest (import / cand_ext shape (B, 14, 165) / cand_ext sample×anchor 차이 assertion / regime_anchor_table fold-leakage / anchor_embed shape (14,8) + init scale ∈ [0.05, 0.15] + requires_grad / query_in shape (B,14,173) / key_anchor shape (B,14,7,196) / attn_logits shape (B,14,7) / attn row-sum=1 / event_ctx shape (B,14,196) / head Linear(196,1) shape (B,14) / forward end-to-end / soft label sum=1 / Frenet→world 식 / no raw skip in head / anchor_embed gradient). | [DONE] |
| G0 | gate | smoke + tests green (예상 < 300s) | [DONE] |
| c8 | exp G1 | G1 carry verification: plan-025 baseline_carry.json load + tight band assert (§5.1). file 부재 또는 hash drift 시 fallback = train.py `--cell G1` 별도 commit 으로 재산출 후 `baseline_carry.json` 박제 | [DONE] |
| G1 | gate | F0 hit ∈ tight band ✓ AND plan-022 winner hit ∈ tight band ✓ | [DONE] |
| c9 | exp G2.X1 | X1 5-fold OOF (4 lever 동시) 학습. 예상 runtime: CPU **7-15 min (~420-900s, §6.2 추정)**. plan-024 167s 의 ~2.4× (batch 256→64 4× step + epoch 22→50 2.27× + hidden 196 0.27× FLOPs). key_anchor (B,K,T,H) FLOPs 는 K expansion 가 plan-024 einsum 의 K 합산 이미 포함 이라 ~0.51× (감소). cache miss 추가 1-2×. `results_X1.json` + `train_X1.log` 박제 + per-epoch anchor_embed grad norm trajectory 박제. | [DONE] |
| G2.X1 | gate | metric finite + max_class_ratio < 0.95. 또한 epoch 50 fully trained 검증 (early stop disabled) + anchor_embed gradient norm > 0 (학습 진행 검증) | [DONE] |
| c10 | analysis | X1 결과 + paired Δ vs F0 + paired Δ vs plan-022 winner + 14-anchor oracle 회수율 + mode collapse 진단 + **anchor_embed cosine similarity matrix (K=14)** (anchor 별 학습된 embedding 의 differentiation 진단) → `paradigm_analysis.{json,md}` | [DONE] |
| G3 | gate | paradigm 판정 (PASS / partial_above_p024 / partial_below_p024 / regression) | [DONE] |
| c11 | docs | 3-file frontmatter sync + `analysis/plan-029/results.md` + `plans/plan-029-*.results.md` pair + follow-up plan-030 (가칭, single-lever ablation 분해 우선) 박제 | [DONE] |
| G_final | gate | 3-file sync + §0.5 c1~c11 [DONE] | [DONE] |

### Plan-specific severe

- `infra_drift`: plan-024 cherry-pick 또는 plan-025 carry module import 실패.
- `f0_reproduce_drift` / `plan022_reproduce_drift`: G1 reproduce tight band 위반.
- `numerical`: PyTorch forward / backward NaN/Inf.
- `mode_collapse` (warn): max_class_ratio ∈ **[0.05, 0.10)** (1/K=0.0714 근방). H3 임계 (> 0.10) 와 align. paradigm mismatch finding 박제 후 G2 계속.
- `mode_collapse_attention` (warn): epoch 5 시점 anchor_embed grad norm ≤ 1e-4 (warmup 종료 직후 cold start). lever (b)(c) 학습 무효 진단 — paradigm_analysis 박제 후 G2 계속, G3 partial_below_p024 가능성 시그널.
- `fold_leakage_violation` (severe halt): regime_anchor_table 산출이 test-fold gt 사용 시. test (`test_regime_anchor_table_fold_leakage`) 가 잡으면 G0 단계 halt. silent 통과 시 G2 결과 invalid → paradigm 판정 신뢰성 0.
- `model_capacity_overflow`: GPU/CPU OOM 또는 학습 시간 > 30 min.
- `plan024_cherry_pick_missing`: c2 cherry-pick 후 import 실패 → halt.

### Plan-specific paths

- whitelist:
  - `analysis/plan-029/**`
  - `tests/test_plan029_smoke.py`
  - `analysis/plan-024/{model.py, feature_weighted_dropout.py}` — **c2 cherry-pick 단계 유일 plan-024 path 수정 허용** (add only)
- blacklist: `analysis/plan-{001..028}/**` (read-only import 예외)

### Decision-note (사용자 결정 + 본문 미박제 핵심 5건만)

- `decision-note: 사용자 결정 (2026-05-22) — anchor_embed init = randn(14,8) * 0.1. plan-024 v5 §5.8 의 0.02 carry 대비 5×. lever (c) key 환경 보정 (anchor_key_proj 의 anchor bias 가 GRU out norm 1~3 의 ≥5% scale).`
- `decision-note: 사용자 결정 (2026-05-22) — N_new=15 (5 group: A.dist 5 + A.tangent_proj 3 + B.cos 1 + D.regime_anchor_prob 1 + F.2 multi-step anchor·v 5). 사용자 EWQ 예시 무시. 식 정의 §3.4.1.`
- `decision-note: spec-default — FWD off. plan-024 §5.4 v3 noise (+0.0003) — outer wrapper 자체 import X.`
- `decision-note: spec-default — NaN/Inf 처리 = torch.nan_to_num(input, nan=0.0, posinf=1e3, neginf=-1e3) before forward + anchor_query_extend.build 마지막 numpy nan_to_num. double safety.`
- `decision-note: spec-default — model.train() 명시 호출 (§6.1 fold loop epoch 진입 직전). GRU dropout self.training 분기 + future-safe.`

---

## §1. 배경

### §1.1 plan-024/025 finding 과 본 plan 의 응답

| Plan | Best | hit_1cm | hit_1p5cm | Finding |
|:--|:--|--:|--:|:--|
| plan-022 | A6_bcc14_tau001 | 0.6528 | 0.8104 | LGBM 170D selector floor (winner) |
| plan-023 | B4_fib50_tau001 | 0.6532 | 0.8108 | anchor large-N marginal (+0.0004) |
| plan-024 v1 (single-seed) | cross-attention h384 | 0.6370 | 0.8092 | spec default. v2/v3/v4/v5 5-fold OOF 모두 0.6370~0.6375 plateau |
| plan-024 §5.13 (4-way honest) | poss 1/2/3/combo | **0.6375 ± 0.0004** | — | 4-way 5-fold OOF plateau (range 0.0007 noise) |
| plan-024 §5.14 (3-seed ens) | combo h128+aug × 3 | **0.6387** | 0.8096 | **paradigm honest ceiling** (variance reduction +0.0010) |
| plan-024 §5.10 (1-fold lucky) | poss 3 h384+aug | 0.6505 | — | 1-fold best epoch tracking 의 lucky catch (5-fold 시 0.6374) |
| plan-024 §5.9 v6 | LGBM + plan-024 230D input | 0.6531 | — | **input lever 자체 carry-redundant (+0.0003)** |
| plan-025 | C1 LGBM 1080D | 0.6320 | 0.8033 | mode collapse → F0 복사 (block ③ LGBM tree categorical split) |
| plan-026 (abandoned) | A2 no-block③ | 0.6509 | 0.8118 | LGBM block ablation finding (사용자 의도 mismatch) |
| plan-027 (abandoned) | E3 weighted | 0.6529 | 0.8118 | LGBM ensemble negative (사용자 의도 mismatch) |

본 plan 의 응답:
- **plan-024 진단 재정의 (사용자, 2026-05-22)**: query 가 sample invariant — cand_feat 150D 중 sample×anchor interaction 13D 만, 137D 가 sample-broadcast (ctx 128D) + anchor-only (spec 9D). attention discrimination 학습 불가가 0.6370~0.6387 plateau 의 진짜 root cause. plan-024 의 §5.1/§5.4/§5.8 기각된 가설들은 모두 *sample-invariant query 위에서의 보조 lever 측정* 이었음.
- **본 plan 의 4 lever (attention 강화)**:
  - (a) query enrichment — `anchor_query_extend.build` (B, 14, **165** = 150+15), 5 group (A.dist 5 + A.tangent_proj 3 + B.cos 1 + D.regime_anchor_prob 1 + F.2 multi-step anchor·v 5). novel ≈67% vs plan-024 ④ (sub-agent 실측)
  - (b) anchor embedding 학습 — `nn.Parameter(14, 8)` learnable, query + key 양쪽 broadcast
  - (c) key anchor-conditional — key_anchor (B,K,T,196) = gru_out broadcast + anchor_key_proj(anchor_embed)
  - (d) 보조 head raw skip 차단 — head_in = event_ctx only (PB framework carry 의 paradigm-confound 제거)
- **abandoned plan-026 + plan-027 통합 재발행** (사용자 plan-026/027 GRU-attention 의도 합의).

### §1.2 paradigm 가설

plan-024 honest ceiling 0.6387 (3-seed) / 0.6375 ± 0.0004 (4-way OOF plateau) base. **사용자 진단 (query sample invariant)** = plan-024 plateau 의 *진짜 root cause* 가설. 본 plan 4 lever 가 이 root cause 를 해결하는지 검증.

- **H1 (강, 핵심)**: 4 lever (query enrichment + anchor embed + key conditional + head raw skip 차단) 동시 적용 → hit_1cm ≥ 0.6528 (plan-022 LGBM floor 회복). plan-024 plateau 0.6387 의 +0.014 lift = attention path 강화 의 합산 효과. **사용자 진단이 옳았을 경우의 PASS criterion**.
- **H1a (보조)**: hit_1cm > 0.6387 (plan-024 honest ceiling 초과) — attention 강화 가 *어떤 lift* 라도 만들었는지. partial PASS.
- **H2 (약)**: hit_1cm > 0.6531 (plan-022/023 winner 초과) — attention paradigm 이 LGBM anchor-selector ceiling 위 추가 lift.
- **H3 (강)**: max_class_ratio > 0.10 (mode collapse 미발생). plan-025 LGBM 의 1/K=0.0714 uniform 와 *질적으로 다름*. plan-024 v1 가 0.1047. head raw skip 차단 (lever d) 으로 block ③ self-prediction trigger 경로 물리적 부재 → H3 PASS 강하게 예상.
- **H4 (보조, 사용자 진단 자체 검증)**: anchor_embed 학습된 cosine similarity matrix 가 anchor 별 *differential* — 14 anchor 가 학습 후 distinct embedding 갖는지. similarity matrix 의 off-diagonal mean < 0.5 (분리 잘 됨) / ≥ 0.5 (collapse 됨). anchor_embed 가 학습 안 됐으면 (gradient 0) lever (b)(c) 효과 0 — H1 FAIL 의 root cause 진단.

판정:
- **H1 PASS** → 사용자 진단 옳음 (query sample invariant 가 plan-024 plateau root cause). attention 강화 4 lever 가 진짜 lift mechanism. follow-up = plan-030 single-lever ablation 분해 (어느 lever 가 main contribution?).
- **H1 FAIL + H1a PASS** → attention 강화 가 부분 효과. follow-up = (i) ablation 분해 + (ii) augmentation (plan-024 §5.10 poss 3) + (iii) 3-seed ensemble 추가.
- **H1 FAIL + H1a FAIL + H3 PASS** → attention 강화 4 lever 전체 무효. 사용자 진단 자체 falsify — query sample invariant 가 root cause 아님. plan-024 plateau 가 paradigm 본질 한계. follow-up = paradigm-distinct (F0 ML / corrector / KNN).
- **H1 FAIL + H3 FAIL** (mode collapse) → 4 lever 동시 적용 가 attention 학습 destabilize. anchor_embed 가 학습 못함 (H4 진단). follow-up = sub-lever 단독 효과 측정 (lever 하나씩 추가).
- **regression** (hit_1cm < 0.6320) → 4 lever 동시 적용 가 model 안정성 해친 결과. follow-up = lever (a) only / (b) only / (c) only / (d) only 4 단독 cell 재실험.

### §1.3 baseline anchor

- **G1.a F0** (plan-020 carry): 0.6320 / 0.8033. 모든 paired Δ anchor.
- **G1.b plan-022 winner** (carry from plan-025 baseline_carry.json 또는 재산출): 0.6531 / 0.8108. paradigm ceiling reference.

---

## §2. 가설 검증 paradigm (4 lever 통합 원칙)

본 plan = paradigm framework (cross-attention GRU selector) 유지 + **attention path 강화 4 lever 동시 적용** (단일 cell X1 = 4 lever 통합). 사용자 진단 ("plan-024 의 query 가 sample invariant 였음") 의 직접 응답이므로 4 lever 가 *논리적으로 함께 들어가야* 진단 falsify 가능 — 1축 단일 변수 원칙 *의도적 위반* (사용자 plan-026/027 GRU-attention 통합 의도 + 본 plan paradigm-level 검증 1회 plan).

| 축 | 변경 | 본 plan 변수 |
|:--|:--|:--|
| Anchor codebook | K=14 BCC fix | ✗ (carry) |
| τ_cls | 0.001 fix | ✗ (carry) |
| Soft label 산식 | `build_soft_label_with_tau` | ✗ (carry) |
| 5-fold split | `stable_fold_id` | ✗ (carry) |
| F0 baseline | `f0_baseline` | ✗ (carry) |
| GRU encoder input | raw seq (B, 7, 95) from seq_builder | ✗ (carry) |
| **Paradigm framework** | cross-attention GRU selector (plan-024 framework 유지) | ✗ (carry) |
| **Lever (a) Query enrichment** | cand_feat 150D → **165D** (`anchor_query_extend.py` 신규 wrapper, sample × anchor interaction channel **15개** 추가, 5 group). 사용자 진단 ("query sample invariant") 의 직접 fix | **✓ 핵심 lever 1** |
| **Lever (b) Anchor embedding 학습** | `nn.Parameter(K=14, d_embed=8)` learnable. query + key 양쪽 broadcast | **✓ 핵심 lever 2** |
| **Lever (c) Key anchor-conditional** | key (B,T,196) → key_anchor (B,K,T,196) broadcast add anchor_embed. attention 식 modify | **✓ 핵심 lever 3** |
| **Lever (d) Head raw skip 차단** | head_in = event_ctx (196D) only. head = Linear(196, 1). PB framework carry 의 raw cand_feat 직통 (paradigm-confound) 제거 | **✓ 보조 lever 4** |
| GRU hidden | 384 → 196 (사용자 명시) | sub-decision |
| Batch size | 256 → 64 (effective step 4×) | sub-decision |
| anchor_embed_dim | 0 → **8** (학습, init 0.1) | lever (b) sub-param |
| FWD | off (§5.4 기각) | ✗ (carry) |
| Training schedule | 50 epoch fixed + cosine + warmup 5 | 보조 schedule lever |

본 plan = **4 lever 동시 적용** (단일 cell X1). plan-024 §5.5 "다중 lever bottleneck 분해 불가" caveat 인지 — H1 PASS 시 plan-030 = single-lever ablation (a/b/c/d 각각 단독) follow-up.

---

## §3. 사전 등록 (Pre-registration)

### §3.1 Fold split (plan-020/021/022/025 carry)

- 5-fold rotating, `stable_fold_id(sample_id_str, n_folds=5)`. MD5 deterministic.
- N=10000 samples → per-fold test ≈ 2000, train ≈ 8000.
- dataset_hash 일치 (plan-025 baseline_carry.json 의 hash carry).

### §3.2 합격 기준

| Gate | 합격 |
|:--|:--|
| G0 | 12+ pytest green + plan-024 cherry-pick 9 file import OK + anchor_query_extend forward smoke |
| G1.a | F0 hit_1cm ∈ [0.6315, 0.6325] AND hit_1p5cm ∈ [0.8028, 0.8038] |
| G1.b | plan-022 winner hit_1cm ∈ [0.6523, 0.6533] AND hit_1p5cm ∈ [0.8099, 0.8109] |
| G2.X1 | metric finite + max_class_ratio < 0.95 + epoch 50 fully trained ✓ + **anchor_embed gradient norm trajectory** (epoch 5/25/50 박제) + **epoch 5 시점 grad norm > 1e-4** (warmup 종료 직후 학습 시작 검증, cold start 회피) + max_class_ratio ∈ [0.05, 0.10) 시 `mode_collapse` warn 박제 후 계속 |
| **G3** | PASS > 0.6528 / partial_above_p024 0.6387 < hit ≤ 0.6528 / partial_below_p024 0.6320 ≤ hit ≤ 0.6387 / regression < 0.6320 |
| G_final | 3-file sync + §0.5 c1~c11 [DONE] + follow-up 1+ 건 |

### §3.3 평가 점수

- **primary**: `hit_1cm` = mean(D1(pred, gt) ≤ 0.01). 5-fold concat OOF.
- **secondary**: `hit_1p5cm`, `top1_acc` (argmax probs vs gt anchor label), `max_class_ratio` (mode collapse 진단).
- **paired Δ**: vs F0 (G1.a) + vs plan-022 winner (G1.b).
- **14-oracle 회수율**: `best_hit_1cm / 0.7928`.

### §3.4 Model spec (X1 cell, 4 lever 통합)

#### §3.4.1 anchor_query_extend.build 내부 식 (cand_ext 15 ch 산출)

```python
# R_wfn shape (N, 3, 3) columns = [t̂, n̂, b̂] (plan-021 carry).
# ANCHORS_A6 shape (K=14, 3) Frenet coord (plan-022 carry).
# world ← frenet: einsum("nij,kj->nki", R_wfn, ANCHORS_A6) — anchor 별 world 변환
# frenet ← world: einsum("nij,nj->ni", R_wfn.transpose(0,2,1), v_world)

# ---- anchor_world (N, K=14, 3) ----
anchor_world = pred_F0_world[:, None, :] + np.einsum("nij,kj->nki", R_wfn, ANCHORS_A6)

# ---- A.dist (5 ch) — past 5 step (t=5..9) × ||anchor_world - X[:,t,:]|| ----
diff   = anchor_world[:, :, None, :] - X[:, None, 5:10, :]                           # (N, K, 5, 3)
A_dist = np.linalg.norm(diff, axis=-1).astype(np.float32)                            # (N, K, 5)

# ---- A.tangent_proj (3 ch) — past 3 step (t=8..10) Frenet t̂-axis (index 0) ----
# sign convention: past_disp_w = (관측점 - anchor) displacement. t̂-axis projection > 0 = 관측점이 anchor 의 진행 방향 앞쪽, < 0 = 뒤쪽.
# A.dist (L290) 는 norm 이라 sign 무관, A.tangent (L294) 는 sign-aware projection — 두 식의 sign 차이는 의도된 design.
past_disp_w = X[:, None, 8:11, :] - anchor_world[:, :, None, :]                      # (N, K, 3, 3)
R_t         = np.transpose(R_wfn, (0, 2, 1))                                         # (N, 3, 3) world→Frenet
past_disp_f = np.einsum("nij,nkpj->nkpi", R_t, past_disp_w)                          # (N, K, 3, 3) Frenet
A_tangent   = past_disp_f[..., 0].astype(np.float32)                                 # (N, K, 3) — t̂ axis

# ---- B.cos (1 ch) — cos(anchor_dir_w, vel_w), normalized + eps ----
anchor_dir_w = np.einsum("nij,kj->nki", R_wfn, ANCHORS_A6)                           # (N, K, 3) = anchor_world - F0
vel_w        = (X[:, 10, :] - X[:, 9, :])[:, None, :]                                # (N, 1, 3)
num          = (anchor_dir_w * vel_w).sum(axis=-1)                                   # (N, K)
den          = np.linalg.norm(anchor_dir_w, axis=-1) * np.linalg.norm(vel_w, axis=-1) + 1e-9
B_cos        = (num / den).astype(np.float32)[:, :, None]                            # (N, K, 1)

# ---- D.regime_anchor_prob (1 ch) — train-fold lookup, fold-leakage 차단 ----
D = regime_anchor_table[regimes][:, :, None].astype(np.float32)                      # (N, K, 1)

# ---- F.2 multi-step anchor·v (5 ch) — t∈{5..9} ANCHORS_A6 · v_t_frenet ----
v_w_seq = X[:, 6:11, :] - X[:, 5:10, :]                                              # (N, 5, 3) world
v_f_seq = np.einsum("nij,ntj->nti", R_t, v_w_seq)                                    # (N, 5, 3) Frenet
F2      = np.einsum("kj,ntj->nkt", ANCHORS_A6, v_f_seq).astype(np.float32)           # (N, K, 5)

# ---- concat 15 ch on top of plan-024 cand_builder 150D ----
cand_base = cand_builder.build(X, R_wfn, pred_F0_world, ANCHORS_A6, f0_baseline_fn,
                                regimes, quantile_carry, ...)                        # (N, K, 150)
extra = np.concatenate([A_dist, A_tangent, B_cos, D, F2], axis=-1)                   # (N, K, 15)
cand_ext = np.concatenate([cand_base, extra], axis=-1)                               # (N, K, 165)
cand_ext = np.nan_to_num(cand_ext, nan=0.0, posinf=1e3, neginf=-1e3)                 # safety net
```

#### §3.4.2 build_regime_anchor_lookup (D channel source)

```python
def build_regime_anchor_lookup(
    gt_train: np.ndarray,           # (N_tr, 3) world
    regimes_train: np.ndarray,      # (N_tr,) int ∈ [0, regime_count)
    ANCHORS_A6: np.ndarray,         # (K=14, 3) Frenet
    R_wfn_train: np.ndarray,        # (N_tr, 3, 3)
    F0_train: np.ndarray,           # (N_tr, 3) world
    regime_count: int = 18,
    laplace: float = 1.0,
) -> np.ndarray:                    # (regime_count, K) float32, row-sum = 1
    R_t = np.transpose(R_wfn_train, (0, 2, 1))
    gt_res_f = np.einsum("nij,nj->ni", R_t, gt_train - F0_train)                     # (N_tr, 3) Frenet
    dist = np.linalg.norm(ANCHORS_A6[None, :, :] - gt_res_f[:, None, :], axis=-1)    # (N_tr, K)
    gt_anchor = dist.argmin(axis=1)                                                  # (N_tr,)
    table = np.full((regime_count, ANCHORS_A6.shape[0]), laplace, np.float32)        # Laplace smoothing
    np.add.at(table, (regimes_train, gt_anchor), 1.0)
    table /= table.sum(axis=1, keepdims=True)                                        # row-sum = 1
    return table
```

#### §3.4.3 GRUNetX1 forward path (model.py)

```python
# §3.4.3 forward path (단일 cell X1)
B = batch_size; K = 14; T = 7; H = 196; D_EMBED = 8; N_new = 15

# cand_ext + seq 가 input
seq      = seq_builder.build(X, R_wfn, ANCHORS_A6, f0_baseline, quantile_carry)      # (B, 7, 95)
# cand_ext (B, 14, 165) 은 §3.4.1 산출 결과

# === lever (a) query enrichment: cand_ext (B, K, 165) 가 forward 의 input ===
# === lever (b) anchor embedding 학습 (init scale 0.1) ===
# self.anchor_embed = nn.Parameter(torch.randn(14, 8) * anchor_embed_init_scale)  # default 0.1
anchor_embed_bc = anchor_embed.unsqueeze(0).expand(B, -1, -1)                        # (B, 14, 8)
query_in = concat([cand_ext, anchor_embed_bc], dim=-1)                               # (B, 14, 173)

# === query projection (Linear(D_EMBED+165 → H) param 화 권장) ===
# self.query_mlp = Sequential(Linear(173 → hidden), GELU, Linear(hidden → hidden))
query    = query_mlp(query_in)                                                       # (B, 14, 196)

# === GRU encoder (key source) ===
out, h   = GRU(input_size=95, hidden=H, num_layers=2, dropout=0.10, batch_first=True)(seq)  # out (B, T, H)

# === lever (c) key anchor-conditional ===
# self.anchor_key_proj = nn.Linear(D_EMBED, hidden)  # D_EMBED=8, hidden=196
anchor_key_bias = anchor_key_proj(anchor_embed)                                      # (14, 196)
key_anchor      = out.unsqueeze(1) + anchor_key_bias.unsqueeze(0).unsqueeze(2)       # (B, K=14, T=7, H=196)

# === Cross-attention (key anchor-conditional 위에, value=key_anchor 단순화) ===
attn_logits = einsum("bkh,bkth->bkt", query, key_anchor) / sqrt(H)                   # (B, 14, 7)
attn        = softmax(attn_logits, dim=-1)                                           # (B, 14, 7)
event_ctx   = einsum("bkt,bkth->bkh", attn, key_anchor)                              # (B, 14, 196)

# === lever (d) head raw skip 차단 — head_in = event_ctx only ===
# self.head = nn.Linear(hidden, 1)  # hidden 변경 시 param 화 필수
score    = head(event_ctx).squeeze(-1)                                               # (B, 14)
probs    = softmax(score, dim=-1)                                                    # (B, 14)
```

**c4 구현 가이드 (hardcoded literal 회피)**: 위 code 의 `Linear(173 → 196)` / `Linear(8, 196)` / `Linear(196, 1)` literal 은 *식 예시*. 실제 c4 source 는 `Linear(D_EMBED + cand_in_dim, hidden)` / `Linear(D_EMBED, hidden)` / `Linear(hidden, 1)` 로 param 화 — hidden ≠ 196 sweep (§9 plan-030 후보) 시 silent bug 회피.

#### §3.4.4 4 lever 의 함의

- (a) query enrichment: §0 paradigm rationale 참조 (cand_ext 15 ch 가 sample × anchor interaction novel ~67%, plan-024 ④ 와 중복 ~33%).
- (b) anchor embedding: query + key 양쪽 broadcast. plan-024 §5.8 v5 (query 만, init 0.02) 와 환경 다름.
- (c) key anchor-conditional: key_anchor (B,K,T,H) broadcast add. plan-024 미시도 axis.
- (d) head raw skip 차단: head_in = event_ctx (196) only. PB framework carry 의 raw cand_feat 직통 (paradigm-confound) 제거 → attention path 가 score main carrier.

**param budget**:
- GRU(95→196, 2-layer): ~343K
- anchor_embed (14, 8): 112
- anchor_key_proj (8→196): 1,764
- query_mlp (165+8=173 → 196 → 196): ≈ 173×196 + 196 + 196×196 + 196 ≈ 73K (N_new=15 박제)
- head (196 → 1): 197
- **total ≈ 418K** (N_new=15). plan-024 hidden=384 backbone total ~3.5M 의 ~12%. attention path 강화 lever 가 capacity 의 main carrier.

### §3.5 Training schedule (X1)

| Hparam | 값 | 사유 |
|:--|--:|:--|
| epochs | **50 fixed** | plan-024 §5.10 long-diag best ep=35 + 안전 마진 |
| early_stopping | **disabled** | plan-024 §5.1 v2 (patience 999) 와 동일 axis |
| optimizer | AdamW | 표준 |
| lr | 7e-4 | attention 표준 + plan-024 carry |
| lr_schedule | `SequentialLR([LinearLR warmup 5 ep, CosineAnnealingLR T_max=45 ep])` total=50 | warmup 5 + cosine 45 (T_max=45 = 50-5) |
| weight_decay | 1e-4 | AdamW 표준 |
| GRU dropout | 0.10 | plan-024 carry (`nn.GRU(num_layers=2, dropout=0.10)`) |
| head_dropout | **N/A (제거)** | head = `Linear(196, 1)` 단순. dropout 무의미 (단일 layer projection) |
| FWD (input dropout) | **off** | plan-024 §5.4 기각. outer wrapper 미import |
| gradient_clip | 1.0 | attention 안정 |
| batch_size | 64 | plan-024 256 의 1/4 |
| random_state | 20260522 | 본 plan layer (모든 fold 동일 seed) |
| τ_cls | **0.001** | plan-022 carry |
| K (anchor count) | **14 BCC** | plan-022 carry. ANCHORS_A6 |
| loss | **soft cross-entropy** | `loss = -(soft_q * log_softmax(score)).sum(-1).mean()` over K=14. plan-022/024 carry |
| **anchor_embed_dim** | **8** (학습) | lever (b). `nn.Parameter(14, 8)`, **init = `randn * 0.1`** (사용자 결정, plan-024 v5 의 0.02 carry 대비 5× — lever (c) anchor_key_proj 의 GRU out scale (norm 1~3) 대비 ≥5% visible). query + key 양쪽 broadcast |
| **anchor_key_proj** | `Linear(8, 196)` | lever (c). anchor_embed → key dim. key_anchor = out + anchor_key_proj(embed).broadcast |
| **N_new (query enrichment)** | **15 channel** (사용자 확정, sub-agent 권장 = A.dist 5 + A.tangent 3 + B.cos 1 + D.regime_prob 1 + F.2 anchor·v 5) | lever (a). `anchor_query_extend.py` 의 추가 sample × anchor interaction channel. cand_ext dim = 165 |

### §3.6 Loss 식 (soft cross-entropy)

```python
log_probs = log_softmax(score, dim=-1)             # (B, 14) — log_softmax 가 softmax→log 보다 수치 안정 (plan-024 model.py smoke 의 log(q_pred+1e-12) 대비 개선)
soft_q    = build_soft_label_with_tau(gt, R_wfn, F0, ANCHORS_A6, tau_cls=0.001)  # (B, 14), row-sum = 1
loss      = -(soft_q * log_probs).sum(dim=-1).mean()      # H(soft_q, p_model) = soft cross-entropy
```

- K-axis sum → batch mean. plan-022/024 (run_oof.py:167) 동일 reduction + 식 (`-(q_b * torch.log(q_pred + 1e-12)).sum(-1).mean()`).
- "soft cross-entropy" 명명은 plan-024 spec §4.7 carry. KL(q‖p) 대비 entropy H(q) 만큼 offset → gradient 는 동일.
- `score` (logits) 만 사용 — model forward 반환은 **단일 tensor `score` (B, 14)** 로 통일 (tuple 반환 분기 없음). `q_pred = F.softmax(score, dim=-1)` 는 eval 시 별도 계산.

### §3.7 Prediction (eval mode)

```python
# 수식 1 (본 plan)
probs       = softmax(score, dim=-1)                                              # (B, 14)
residual_f  = einsum("bk,kj->bj", probs, ANCHORS_A6)                              # (B, 3) Frenet
residual_w  = einsum("bij,bj->bi", R_wfn_test, residual_f)                        # (B, 3) world
final_pred  = F0_test + residual_w                                                # (B, 3) world
hit_1cm     = (norm(final_pred - gt, dim=-1) <= 0.01).float().mean()

# 수식 2 (plan-024 run_oof.py:287-291 carry, 수학적 등가)
# anchors_world = R_wfn @ ANCHORS_A6 + F0      ;  final_pred = Σ_k probs[k] · anchors_world[k]
# 선형 결합과 affine 변환의 분리/통합. 결과 identical.
```

---

## §4. STAGE 0 — 인프라 (G0)

### §4.1 모듈 layout

```
analysis/plan-029/
├── __init__.py
├── anchor_query_extend.py   ← 신규 wrapper. plan-024 cand_builder.build (B,K,150) + sample × anchor interaction channel 15개 추가 → (B,K,165). lever (a) source (c3)
├── model.py                 ← 신규 `GRUNetX1`. 4 lever (a)(b)(c)(d) 통합. plan-024 GRU+query_mlp 만 carry, 자체 cross-attention 식 (key anchor-conditional), head=Linear(196,1) (c4)
├── train.py                 ← PyTorch 5-fold OOF training (c5)
├── run_oof.py               ← orchestrator + G1 reproduce + CLI (c6)
├── baseline_carry.json      ← G1 박제 (c8)
├── results_X1.json          ← G2.X1 박제 (c9)
├── train_X1.log             ← 학습 진행 log (c9)
├── paradigm_analysis.{json,md}  ← c10 (anchor_embed cosine similarity matrix 포함)
└── results.md               ← c11

analysis/plan-024/            ← c2 추가 cherry-pick from worktree-plan-024-combo
├── model.py                  ← 신규 cherry-pick (GRU + query_mlp 만 사용, backbone.head + FWD wrapper class 둘 다 import X)
└── feature_weighted_dropout.py  ← 신규 cherry-pick (import 만, instantiate X)
(기존 8 file: __init__.py + anchor_vocab.py + cand_builder.py + seq_builder.py + torsion_calc.py + quantile_carry.py + multiwindow_trim_build.py + multiwindow_trim.json — plan-025 c2 cherry-pick 으로 이미 main 존재)

tests/test_plan029_smoke.py   ← 12+ pytest (c7)
```

### §4.2 plan-024 cherry-pick (c2)

```bash
git checkout worktree-plan-024-combo -- analysis/plan-024/model.py analysis/plan-024/feature_weighted_dropout.py
```

commit hash pin: `16b74a1` (= `worktree-plan-024-combo` 의 latest commit affecting `analysis/plan-024/model.py + feature_weighted_dropout.py`. 2026-05-22 기준 확인). c2 cherry-pick 시 `git checkout 16b74a1 -- analysis/plan-024/model.py analysis/plan-024/feature_weighted_dropout.py` 로 SHA 명시 가능. 향후 worktree-plan-024-combo 에 추가 patch 발생 시 본 plan 본문 갱신 필요 (reproducibility 보장).

### §4.3 tests (c7) — 12+ pytest

- `test_imports`: plan-024 model + feature_weighted_dropout (모듈 import 만, FWD wrapper instantiate X) + plan-029 anchor_query_extend + plan-024 cand_builder + plan-022 anchors + plan-021 build_input + plan-020 baseline_f0 모두 import OK.
- `test_anchor_query_extend_shape`: `anchor_query_extend.build(...)` output shape = **(B, 14, 165)**. N_new=15 박제.
- `test_anchor_query_extend_sample_anchor_interaction`: cand_ext 의 새 15 channel (index 150:165) 이 *진짜 sample × anchor specific* — `assert not np.allclose(cand_ext[0, :, 150:], cand_ext[1, :, 150:])` (sample axis 차이) AND `assert not np.allclose(cand_ext[:, 0, 150:], cand_ext[:, 1, 150:])` (anchor axis 차이). silent broadcast bug (b 또는 k 모두 같은 값) 검출.
- `test_regime_anchor_table_fold_leakage`: `regime_anchor_table` 가 train-fold 만 산출 (test-fold gt 미사용). train.py per-fold loop 안 `build_lookup(gt_train, regimes_train)` 만 호출 검증.
- `test_anchor_embed_shape`: `model.anchor_embed.shape == (14, 8)`. `requires_grad == True` (learnable param). **init scale 검증**: `model.anchor_embed.data.std()` ∈ [0.05, 0.15] (init = randn × 0.1 의 std ≈ 0.1, tolerance ±50%).
- `test_query_in_shape`: forward 중간 query_in shape = (B, 14, **173** = 165+8).
- `test_anchor_key_proj_shape`: `anchor_key_proj.weight.shape == (196, 8)`. `anchor_key_proj(anchor_embed).shape == (14, 196)`.
- `test_key_anchor_shape`: forward 중간 key_anchor shape = (B, 14, 7, 196). broadcast add 정합성 (key_anchor[b,k,t,:] == gru_out[b,t,:] + anchor_key_proj(anchor_embed[k,:]) 검증).
- `test_attn_logits_shape`: shape = (B, 14, 7). einsum index 정합.
- `test_attn_row_sum_one`: `attn.sum(dim=-1)` 모두 1 (softmax 검증).
- `test_event_ctx_shape`: shape = (B, 14, 196).
- `test_head_input_dim_196_only`: model.head 의 in_features == 196 (event_ctx only). head 정의가 `nn.Linear(196, 1)` 단순 + bias 1 = total param 197.
- `test_no_raw_skip_in_head`: model forward path 안 head 호출 전 input tensor 가 event_ctx 그 자체 (`id` 또는 shape (B,14,196) 검증). cand_feat / cand_ext / block1 / block4 / h_final_bc 등 다른 tensor 가 head 에 흐르지 않음 — `GRUNetX1.forward` source 검사 또는 hook 으로 입증.
- `test_forward_end_to_end`: dummy input → score (B=4, 14) 정상 + NaN/Inf 부재.
- `test_anchor_embed_gradient`: dummy loss.backward() 후 `model.anchor_embed.grad.norm() > 0` (학습 가능 검증).
- `test_train_mode_dropout_active`: model.train() vs eval() forward 2회 결과 std 차이 (GRU dropout 적용 확인).
- `test_soft_label_sum_one`: build_soft_label_with_tau output row-sum = 1.
- `test_frenet_to_world_inverse`: round-trip Frenet → world → Frenet (tolerance 내).
- `test_soft_ce_loss_nonneg`: soft CE loss ≥ 0.

---

## §5. STAGE 1 — G1 reproduce (c7)

### §5.1 carry from plan-025 baseline_carry.json

```python
import json
prereq_path = "analysis/plan-025/baseline_carry.json"
with open(prereq_path) as f:
    p025_baseline = json.load(f)

F0_hit_1cm = p025_baseline["F0"]["hit_1cm"]      # 0.6320
F0_hit_1p5cm = p025_baseline["F0"]["hit_1p5cm"]  # 0.8033
p022_hit_1cm = p025_baseline["plan022_winner"]["hit_1cm"]    # 0.6531
p022_hit_1p5cm = p025_baseline["plan022_winner"]["hit_1p5cm"]  # 0.8108

assert 0.6315 <= F0_hit_1cm <= 0.6325, f"F0 drift: {F0_hit_1cm}"
assert 0.8028 <= F0_hit_1p5cm <= 0.8038
assert 0.6523 <= p022_hit_1cm <= 0.6533
assert 0.8099 <= p022_hit_1p5cm <= 0.8109
```

plan-025 의 baseline_carry.json 이 이미 reproduce 결과 박제 (main commit e262299). 본 plan 은 carry only (재산출 X).

### §5.2 G1 합격

- carry value 가 tight band ✓ → G1 PASS.
- 재산출 옵션 (decision-note): `--cell G1` CLI 로 본 plan 안에서 새로 reproduce 가능 (drift 발생 시 carry replace).
- **재산출 절차 (fallback)**: (1) F0 = `f0_baseline(X, end_idx=10)` (plan-020 carry, deterministic, no learning) → `err = ||F0 - gt||`, `hit_1cm/_1p5cm` 산출 → tight band assert. (2) plan-022 winner = `selector_only_model` (plan-022 carry — `analysis/plan-022/selector_only_model.py` `selector_only_eval_5fold(X, gt, ANCHORS_A6, tau_cls=0.001, K=14, folds=stable_fold_id, lgbm_hparams=plan-022 winner spec)` 5-fold OOF) → `hit_1cm/_1p5cm` → tight band assert. (3) 산출본 `analysis/plan-025/baseline_carry.json` 형식으로 박제 (key: `F0_hit_1cm`, `F0_hit_1p5cm`, `p022_hit_1cm`, `p022_hit_1p5cm`). (4) 양쪽 모두 tight band 위반 시 `f0_reproduce_drift` 또는 `plan022_reproduce_drift` severe halt.

---

## §6. STAGE 2 — X1 cell 5-fold OOF (c8)

### §6.1 Per-fold loop

```python
# ── 사전 준비 (loop 진입 전) ─────────────────────────────────────
# Dataset-wide load (c6 orchestrator): X shape (N_total, 11, 3) world frame, gt shape (N_total, 3).
X, _   = load_all_samples()                                  # plan-024/025 convention 동일 (src/io.py)
gt     = load_labels()                                        # plan-024/025 convention 동일 (src/io.py)
folds  = stable_fold_id(N_total, n_splits=5, seed=20260522)   # plan-020 carry (MD5 stable)
# MULTIWINDOW_TRIM_PATH: plan-024 cherry-pick 의 multiwindow_trim.json 경로 const
# (plan-025 build_feat_1080.py 의 convention 동일: `str(Path("analysis/plan-024/multiwindow_trim.json"))`)
MULTIWINDOW_TRIM_PATH = str(Path("analysis") / "plan-024" / "multiwindow_trim.json")
N_total = X.shape[0]; K = 14
# regime_count=18 = plan-024 carry: `fit_regime_bins(X, end_idx=10)` 출력 bin 개수 (full dataset 동일, fold 별 미변화).
REGIME_COUNT = 18
oof_pred   = np.zeros((N_total, 3),  dtype=np.float32)
oof_probs  = np.zeros((N_total, K),  dtype=np.float32)
R_wfn_all  = np.zeros((N_total, 3, 3), dtype=np.float32)
F0_all     = np.zeros((N_total, 3),  dtype=np.float32)

for fold in range(5):
    train_idx = np.where(folds != fold)[0]
    test_idx = np.where(folds == fold)[0]
    X_tr, X_te = X[train_idx], X[test_idx]
    gt_tr, gt_te = gt[train_idx], gt[test_idx]

    # Frenet basis + F0
    R_wfn_tr = build_frenet_basis_3d(X_tr, end_idx=10)
    R_wfn_te = build_frenet_basis_3d(X_te, end_idx=10)
    F0_tr = f0_baseline(X_tr, end_idx=10).astype(np.float32)
    F0_te = f0_baseline(X_te, end_idx=10).astype(np.float32)

    # Fold-leakage 차단: train fold quantile
    qc = quantile_carry.build(X_tr, R_wfn_tr)

    # Input feature 산출 (train + test 동일 quantile)
    # D channel 용 fold-leakage-safe lookup table 산출 (train fold gt + regime 만 사용)
    regime_bins_tr = fit_regime_bins(X_tr, end_idx=10)
    regimes_tr = assign_regimes(X_tr, end_idx=10, bins=regime_bins_tr)
    regimes_te = assign_regimes(X_te, end_idx=10, bins=regime_bins_tr)   # train-fold bins inject → fold-leakage 차단
    regime_anchor_table_tr = build_regime_anchor_lookup(
        gt_train=gt_tr, regimes_train=regimes_tr, ANCHORS_A6=ANCHORS_A6,
        R_wfn_train=R_wfn_tr, F0_train=F0_tr,
        regime_count=18, laplace=1.0,
    )                                                                     # (regime_count=18, K=14) row-sum=1, train-fold only
    # lever (a): anchor_query_extend 가 plan-024 cand_builder.build (B,K,150) 호출 후 sample × anchor interaction channel 15개 추가
    cand_ext_tr = anchor_query_extend.build(
                    X_tr, R_wfn_tr, F0_tr, ANCHORS_A6, f0_baseline,
                    regimes=regimes_tr, quantile_carry=qc,
                    multiwindow_trim_path=MULTIWINDOW_TRIM_PATH, regime_count=18,
                    regime_anchor_table=regime_anchor_table_tr)           # (N_tr, 14, 165)
    cand_ext_te = anchor_query_extend.build(
                    X_te, R_wfn_te, F0_te, ANCHORS_A6, f0_baseline,
                    regimes=regimes_te, quantile_carry=qc,                # qc = train-fold quantile (fold-leakage 차단)
                    multiwindow_trim_path=MULTIWINDOW_TRIM_PATH, regime_count=18,
                    regime_anchor_table=regime_anchor_table_tr)           # (N_te, 14, 165), train-fold table inject
    seq_tr = seq_builder.build(X_tr, R_wfn_tr, ANCHORS_A6, f0_baseline, quantile_carry=qc)  # (N_tr, 7, 95)
    seq_te = seq_builder.build(X_te, R_wfn_te, ANCHORS_A6, f0_baseline, quantile_carry=qc)  # (N_te, 7, 95), train-fold qc

    # Soft label
    q_tr = build_soft_label_with_tau(gt_tr, R_wfn_tr, F0_tr, ANCHORS_A6, tau_cls=0.001)

    # Model + Optimizer
    # GRUNetX1: 4 lever 통합. plan-024 GRU + query_mlp 만 carry. backbone.head + FWD wrapper class 둘 다 import X.
    # cand_in_dim = 165 (lever a, N_new=15), anchor_embed_dim=8 (lever b), key_anchor (lever c), head=Linear(196,1) (lever d).
    model = GRUNetX1(seq_dim=95, cand_in_dim=165, hidden=196,
                     anchor_embed_dim=8, anchor_embed_init_scale=0.1, gru_dropout=0.10, K=14)
    optimizer = AdamW(model.parameters(), lr=7e-4, weight_decay=1e-4)
    # LR schedule: warmup 5 epoch (linear 0→lr) + cosine 45 epoch (lr→0). total 50 epoch.
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[
            torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-6, end_factor=1.0, total_iters=5),
            torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=45),
        ],
        milestones=[5],
    )
    
    # Training (epoch=50 fixed, no early stop). model.train() 명시.
    N_tr = len(train_idx)
    for epoch in range(50):
        model.train()
        # per-epoch shuffle (seed = random_state + epoch * 1000 + fold, drop_last=False)
        rng = np.random.default_rng(20260522 + epoch * 1000 + fold)
        perm = rng.permutation(N_tr)
        for b_start in range(0, N_tr, 64):
            idx = perm[b_start : b_start + 64]                          # batch_size 64, last batch < 64 면 그대로 사용
            # numpy → torch (float32) — q_tr / seq_tr / cand_ext_tr 모두 np.ndarray, model 은 torch
            # nan_to_num safety net (decision-note): cand_ext 안 D channel + B.cos eps 등 NaN risk 차단
            seq_batch    = torch.nan_to_num(torch.from_numpy(seq_tr[idx]).float(),      nan=0.0, posinf=1e3, neginf=-1e3)
            cand_batch   = torch.nan_to_num(torch.from_numpy(cand_ext_tr[idx]).float(), nan=0.0, posinf=1e3, neginf=-1e3)
            q_batch      = torch.from_numpy(q_tr[idx]).float()          # (B, 14), row-sum=1 (soft label, NaN risk 없음)
            optimizer.zero_grad()
            score = model(seq_batch, cand_batch)                        # score logits (B, 14), torch
            log_probs = log_softmax(score, dim=-1)
            loss = -(q_batch * log_probs).sum(dim=-1).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()                                                # epoch 단위 step
        # per-epoch anchor_embed grad norm 박제 (G2 검증 + paradigm_analysis)
        log_anchor_embed_grad_norm(epoch, model.anchor_embed.grad)
    
    # Eval — N_te 가 1 batch (memory ~170MB for key_anchor) 면 분할 권장
    model.eval()
    with torch.no_grad():
        probs_te = []
        for e_start in range(0, len(test_idx), 256):                    # eval batch 256 (key_anchor memory ~22MB)
            sb = torch.nan_to_num(torch.from_numpy(seq_te[e_start : e_start + 256]).float(),      nan=0.0, posinf=1e3, neginf=-1e3)
            cb = torch.nan_to_num(torch.from_numpy(cand_ext_te[e_start : e_start + 256]).float(), nan=0.0, posinf=1e3, neginf=-1e3)
            probs_te.append(softmax(model(sb, cb), dim=-1).cpu().numpy())
        probs_te = np.concatenate(probs_te, axis=0)                     # (N_te, 14)
        residual_frenet = (probs_te[:, :, None] * ANCHORS_A6[None, :, :]).sum(axis=1)
        residual_world = np.einsum("nij,nj->ni", R_wfn_te, residual_frenet)
        final_pred = F0_te + residual_world
        oof_pred[test_idx] = final_pred                                 # oof_pred = np.zeros((N_total, 3), np.float32) 사전 alloc
        oof_probs[test_idx] = probs_te                                  # oof_probs = np.zeros((N_total, K=14), np.float32) 사전 alloc
        # R_wfn / F0 fold-별 누적 (OOF aggregate 단계에서 gt_anchor_label 계산 시 사용)
        R_wfn_all[test_idx] = R_wfn_te                                  # (N_total, 3, 3) 사전 alloc
        F0_all[test_idx]    = F0_te                                     # (N_total, 3) 사전 alloc

# Concat OOF metric
err = np.linalg.norm(oof_pred - gt, axis=1)                             # gt = dataset-wide loader 결과 (top-level)
hit_1cm = (err <= 0.01).mean()
hit_1p5cm = (err <= 0.015).mean()
# max_class_ratio: argmax 분포 기준 최빈 anchor 비율 (mode collapse 진단). 1/K = 0.0714, 임계 [0.05, 0.10) → near-uniform/collapse.
top1_argmax = oof_probs.argmax(axis=1)                                  # (N_total,) ∈ [0, K)
max_class_ratio = float(np.bincount(top1_argmax, minlength=14).max() / len(top1_argmax))
# gt_anchor_label = gt → ANCHORS_A6 nearest neighbor (Frenet 좌표). R_wfn_all / F0_all = fold loop 안 누적본.
R_t_all  = np.transpose(R_wfn_all, (0, 2, 1))
gt_res_f = np.einsum("nij,nj->ni", R_t_all, gt - F0_all)
gt_anchor_label = np.linalg.norm(ANCHORS_A6[None, :, :] - gt_res_f[:, None, :], axis=-1).argmin(axis=1)
top1_acc = (top1_argmax == gt_anchor_label).mean()
```

### §6.2 Runtime + Param 예상

**param 추정** (N_new=15 박제):
- GRU(input=95, hidden=196, num_layers=2, batch_first): ≈ 2 × 3 × (95+196+1) × 196 ≈ 343K (~82%)
- query_mlp (Linear 165+8=173→196 + GELU + Linear 196→196): 173×196 + 196 + 196×196 + 196 ≈ **73K**
- anchor_embed (14, 8): 112
- anchor_key_proj (Linear 8→196): 8×196 + 196 ≈ 1.76K
- head (Linear 196→1): 196 + 1 = **197**
- **total ≈ 418K**. plan-024 hidden=384 backbone total ~3.5M 의 ~12%. head 의 dominance 사라짐 → attention path (GRU + query_mlp + key_anchor) 가 model capacity 의 ~99%.

**runtime 추정** (sub-agent 재검증 결과, 2026-05-22):
- plan-024 §2 5-fold OOF 167s base
- plan-029 scaling: 50/22 epoch × 256/64 batch × 196^2/384^2 hidden FLOP scaling = 2.27 × 4 × 0.26 ≈ **2.36× plan-024**
- key_anchor (B,K,T,H) 의 K=14× expansion 가 FLOPs 14× 가 *아님* — plan-024 einsum 도 K 합산 이미 포함. attention block FLOPs 본 plan / plan-024 = (14×7×196) / (14×7×384) ≈ **0.51×** (오히려 감소, hidden 축소 효과)
- 종합: ~2.36× plan-024 = **5-fold total 약 167s × 2.36 ≈ 400s ≈ 7 min CPU** (이전 spec 의 32-37 min 추정은 K=14× 과대 추정에 기반, 폐기)
- 단 key_anchor `(B=64, K=14, T=7, H=196) × 4B = 5.5MB/batch` materialization 의 CPU cache miss + numpy↔torch boundary overhead 가 추가 ~1-2× 가능 → 실측 = 7-15 min 예상 범위
- severe `model_capacity_overflow` 임계 = **30 min total** (예상 7-15 min 의 ~2-4×). 30 min 초과 시 사후 bottleneck 분석

### §6.3 G2.X1 합격

- metric finite ✓ (NaN/Inf X)
- max_class_ratio < 0.95 ✓ (no extreme winner)
- max_class_ratio ∈ [0.05, 0.10) → `mode_collapse` warn 박제 후 계속 (H3 FAIL 판정 input)
- epoch 50 fully trained ✓ (no early stop)
- **anchor_embed gradient norm trajectory** (epoch 5/25/50 박제) — epoch 5 시점 norm > 1e-4 검증 (warmup 종료 직후 cold start 회피, lever (b)(c) effective 학습 시작 진단)
- 위반 (numerical / overflow > 30 min / cherry-pick missing / epoch 5 grad norm ≤ 1e-4) = severe halt

---

## §7. STAGE 3 — Paradigm finding (c9, G3)

### §7.1 X1 결과 표

| Metric | X1 | F0 | plan-022 | plan-024 honest |
|:--|--:|--:|--:|--:|
| hit_1cm | ?.???? | 0.6320 | 0.6531 | 0.6387 |
| hit_1p5cm | ?.???? | 0.8033 | 0.8108 | 0.8096 |
| max_class_ratio | ?.??? | — | 0.1054 | ? |
| oracle 회수율 | ?.??% | — | 82.4% | 80.6% |
| paradigm 위치 | — | — | LGBM floor | 3-seed ceiling (§5.14) |

(plan-024 v1 0.6370 / §5.10 lucky 0.6505 / plan-025 0.6320 mode collapse 비교는 §1.1 참조.)

### §7.2 G3 판정 + Hypothesis 검증

§0 PASS criterion 그대로. H1~H4 + plan-030 follow-up 1 table:

| H | 조건 | PASS 의미 | plan-030 follow-up |
|---|---|---|---|
| **H1** (핵심) | hit_1cm ≥ 0.6528 | 사용자 진단 옳음, 4 lever 합산 효과 | single-lever ablation (a/b/c/d 각 단독) |
| **H1a** | hit_1cm > 0.6387 | 4 lever 가 plan-024 ceiling 위 lift | ablation + augmentation + 3-seed ensemble |
| **H2** | hit_1cm > 0.6531 | attention paradigm 이 LGBM 위 lift | — |
| **H3** | max_class_ratio > 0.10 | mode collapse 미발생 (lever d 차단 효과) | — |
| **H4** | anchor_embed cosine off-diag mean < 0.5 | anchor 별 distinct learnable (lever b/c 효과) | (FAIL 시 lever b/c 식 재설계) |

조합 → follow-up matrix:

| 결과 | plan-030 우선순위 |
|---|---|
| H1 PASS | single-lever ablation 분해 |
| H1 FAIL + H1a PASS | ablation + augmentation + ensemble |
| H1 FAIL + H1a FAIL + H4 PASS | paradigm-distinct (F0 ML / corrector / KNN) — 사용자 진단 falsify |
| H1 FAIL + H4 FAIL | lever (b)(c) 식 재설계 (per-anchor key projection 등) |
| regression (< 0.6320) | lever 단독 cell 4개 재실험 |

### §7.3 paradigm finding 박제

- plan-024 §5.14 honest ceiling 0.6387 대비 lift 분해 (H1 / H1a)
- 사용자 진단 (query sample invariant) valid / falsify 결론 (H1 PASS / H1 FAIL+H4 PASS)
- anchor_embed cosine similarity matrix (K=14) + per-epoch grad norm trajectory (epoch 1/5/25/50)
- H3 mode collapse 진단 (plan-025 LGBM 0.0714 와 비교)

---

## §8. STAGE 4 — G_final (c10)

### §8.1 산출

- `analysis/plan-029/results.md` (11 항목)
- `plans/plan-029-*.results.md` pair
- 3-file frontmatter sync (status=all_complete, band ∈ {positive, partial_above, partial_below, negative}, best_cell=X1, best_hit_1cm, best_delta_1cm vs F0, best_delta_vs_p024_ceiling = best_hit_1cm − 0.6387, **anchor_embed_cosine_offdiag_mean** = H4 진단)
- follow-up plan-030 (가칭) 후보 ≥ 1 건 박제 (§7.4 의 H 시나리오별 lever 우선순위 carry)

### §8.2 G_final 합격

- 3-file sync ✓
- §0.5 c1~c11 모두 [DONE] ✓
- follow-up 1+ 건 박제 ✓

---

## §9. Out of scope

- 단일 cell fix: hidden=196, batch=64, lr=7e-4, anchor_embed_dim=8, N_new=15, τ_cls=0.001, K=14 BCC (모두 sweep 미포함).
- DACON LB submit (별개 결정).
- 별개 paradigm: F0 ML (plan-028 후보), corrector / 2-stage residual regression, anchor layout 변경.
- FWD on (§5.4 기각 lever).
- **plan-030 후보 lever group** (§7.2 H 시나리오 별 우선순위):
  - single-lever ablation (lever a/b/c/d 각 단독)
  - input augmentation σ=0.05 (plan-024 §5.10 poss 3, +0.0135 1-fold)
  - 3-seed ensemble (plan-024 §5.14, +0.0010 variance reduction)
  - head raw skip 부활 (lever d 효과 ablation)
  - per-anchor key projection (lever c 식 재설계)
  - attention bias on logits (lever c memory 단축)
  - head 2-layer MLP (head capacity 부족 진단 시)
  - ensemble / boundary corrector

---

## §10. 참조 (read-only)

- spec: plan-022 / plan-024 / plan-025 carry
- **plan-024 results.md (필수 인용)**:
  - §1 — OOF metric table (v1 0.6370, max_class_ratio 0.1047, top1_acc 0.1227)
  - §2 — per-fold variance (167s, std 0.0034)
  - §5.1 — **under-converged 가설 기각** (v2 patience 999, 171s, 0.6370 동일)
  - §5.4 — over-regularization 기각 (v3 channel drop off, 0.6373 noise)
  - §5.8 — anchor identity capacity 부분 기각 (A7 learnable embedding hit_1cm 변화 X)
  - §5.9 — v6 LGBM + plan-024 230D input = 0.6531 → **input lever carry-redundant (+0.0003)**
  - §5.10 — 3 가능성 ablation, **poss 3 augmentation 1-fold +0.0135 lift** (단 5-fold 0.6374 = plateau)
  - §5.13 — 4-way 5-fold OOF **honest ceiling 0.6375 ± 0.0004 plateau**
  - §5.14 — **3-seed ensemble 0.6387** = paradigm honest best
- carry: plan-020/021/022/024/025 module (frontmatter `code_reuse` 참조)
- abandoned: plan-026 / plan-027 (supersedes_abandoned frontmatter 명시)
- memory: `project_next_plan_direction.md` (2026-05-22 user GRU-attention 의도 + plan-029/030 mapping)
