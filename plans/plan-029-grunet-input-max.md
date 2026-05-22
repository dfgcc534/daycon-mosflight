---
plan_id: 029
version: 1.0
date: 2026-05-22 (Asia/Seoul)
status: written
best_cell: null
best_hit_1cm: null
best_hit_1p5cm: null
based_on:
  - 022 (winner A6_bcc14_tau001 OOF 0.6531 / 0.8108. K=14 BCC + τ=0.001. selector-only LGBM 170D baseline)
  - 024 (cross-attention GRU selector. honest ceiling = 0.6387 3-seed (§5.14) / 0.6375 ± 0.0004 4-way OOF plateau (§5.13). under-converged/over-reg/anchor-identity 가설 모두 기각/부분 기각. 본 plan = paradigm framework 유지 + plan-024 *미시도* lever (head 1472D / hidden 196 / batch 64) 검증)
  - 025 (LGBM + 후보 concat + seq 압축 → mode collapse → F0 복사. block ③ LGBM tree categorical split evidence. 본 plan input = plan-025 의 1080D 그대로, head path 의 GELU+Dropout+mix projection 으로 categorical memorization 메커니즘 부재 기대)
  - 020 (F0 baseline 0.6320/0.8033 + stable_fold_id MD5)
inspired_by:
  - 사용자 (2026-05-22): "plan-026, 027 은 gru-attention" — abandoned LGBM 026/027 의 통합 재발행. paradigm-level 검증 1회 plan.
  - plan-025 paradigm mismatch finding: block ③ 22D per-anchor 가 LGBM 에서 self-prediction trigger 였지만 GRU-attention 에서는 query 의 anchor identity 로 정상 작동 예상.
code_reuse:
  - module: analysis/plan-024/model.py (worktree-plan-024-combo branch 의 c2 cherry-pick 필요)
    symbols: [CandidateAttentionGRUSelectorCarry]
    reason: 본 plan 의 backbone (GRU + query_mlp + cross-attention computation 만 carry). hidden=384 → 196 변경. `backbone.head` (hardcoded Linear(2H+cand_dim, hidden)=542→196) 는 **사용 X** — 본 plan 새 head 가 1472D 입력이므로 dim mismatch. wrapper `GRUNetX1` 안에서 backbone 의 forward 일부만 재사용 (또는 subclass + self.head=Identity). `CrossAttentionAnchorSelector` outer wrapper (FWD wrapping) 는 import 안 함 (FWD off).
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
    reason: 1080D input builder. 본 plan baseline + head skip 의 source.
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
  - plan-030 (가칭, GRU-attention 결과 후속 — F0 ML 또는 corrector 부활)
scope: plan-024 cross-attention GRU selector paradigm framework 유지 + **plan-024 미시도 lever 검증**: (a) head input expansion 1472D (542→1472, plan-025 block ①+④ 930D skip), (b) hidden 196 (plan-024 384 의 0.51×), (c) batch 64 (plan-024 256 의 1/4, effective step 4×). anchor_embed_dim=0 default carry (§5.8 부분 기각). FWD off (§5.4 기각, outer wrapper 미import). GRU encoder input = raw seq (B, 7, 95), cross-attention query = cand_feat (B, 14, 150). training schedule = 50 epoch fixed (early stop disabled, §5.1 under-conv 기각 후 cosine annealing 완주 lever), lr=7e-4 + SequentialLR([LinearLR warmup 5 ep, CosineAnnealingLR T_max=45 ep]), AdamW (wd=1e-4), GRU dropout=0.10, head_dropout=0.15, gradient_clip=1.0, soft cross-entropy loss. K=14 BCC + τ_cls=0.001 fix (plan-022 carry). 5-fold stable_fold_id. ensemble / DACON LB / corrector / F0 ML / augmentation (plan-024 §5.10 poss 3 carry) = out-of-scope.
exp_ids:
  - Z029_X1_gru_h196
lb_score: null
band: null
---

# plan-029 v1 — GRU-attention Input Max (hidden=196, 1080D + raw seq)

## §0. 한 줄 목적

> **plan-024 cross-attention paradigm 의 hyperparameter 재설계** + **plan-025 1080D input 정상 검증**. plan-025 LGBM 의 mode collapse (paradigm mismatch evidence) 가 GRU-attention 위에서는 *paradigm 자연스러운 작동* 으로 회복되는지 검증. **abandoned plan-026 + plan-027 의 통합 재발행** (사용자 plan-026/027 GRU-attention 의도 합의).
>
> **paradigm rationale** (plan-024 results.md §5.1+§5.8+§5.9+§5.10+§5.13+§5.14 종합):
> 1. plan-024 의 honest ceiling = **0.6387 (3-seed ensemble)** / 0.6377 (single-seed combo) / 0.6375 ± 0.0004 (4-way 5-fold OOF plateau). plan-022 carry **−0.0141 미달**. CPU under-converged 가설 §5.1 **기각** (v2 patience 999, 171s, 0.6370 동일), over-regularization 가설 §5.4 **기각**, anchor identity capacity (A7 learnable embed 8D) §5.8 **부분 기각** (hit_1cm 변화 없음). plan-024 §5.10 의 *진짜 단일 lever* = **input augmentation σ=0.05 (poss 3: 1-fold best 0.6505, v1 대비 +0.0135)** 그리고 §5.14 의 *variance reduction lever* = **3-seed ensemble (+0.0010)**. → 본 plan 의 새 lever = (a) **head input expansion (542→1472D, plan-024 미시도)**, (b) **hidden capacity 축소 (384→196)** + (c) batch 256→64 (effective step 4×) + (d) cosine + warmup 5 (long-diag best ep=35 보다 충분히 긴 50 epoch fixed 안에서 lr annealing 완주).
> 2. plan-025 (1080D LGBM) fail = block ③ 22D per-anchor 신호가 LGBM tree 의 sharp categorical split 으로 anchor identity 를 *memorize* (self-prediction trigger). GRU-attention 위에서는 (i) query path 가 anchor 별 dense projection 으로 흐르고 (ii) head path 의 cand_feat raw concat 도 LGBM split 와 달리 *공통 nonlinearity (GELU+Dropout) + 다른 channel 과 mix* 되어 categorical memorization 메커니즘 자체가 부재. 단 plan-024 §5.9 v6 (LGBM + plan-024 230D input) = +0.0003 lift = input lever 자체가 carry 와 **거의 redundant** evidence → 본 plan 의 lift 후보는 input 보다는 **head 차원 확장** 과 **hidden 축소** 의 *plan-024 미시도 axis*.
> 3. plan-024 attention paradigm 의 implicit assumption (anchor identity at query, sequence at key, sample-level ctx broadcast) 이 input 1080D 안에 정확히 박혀 있음 — 본 plan 은 paradigm framework 유지 + head capacity / hidden 만 변경.
>
> **단일 cell (paradigm-level 검증 1회 plan)**:
> - **X1** = GRU(hidden=196) + cross-attention + head_mlp(skip=plan-025 block ①+④)
> - training schedule: epoch=50 fixed (no early stop), lr=7e-4 cosine, AdamW (wd=1e-4), dropout=0.10, gradient_clip=1.0, batch=64
>
> **pass criterion (G3)**:
> - **PASS**: hit_1cm > 0.6528 (= plan-022 winner) → paradigm 정상 검증, plan-024 honest ceiling 0.6387 의 +0.014 돌파 → 새 lever (head 1472D / hidden 196 / batch 64) 중 ≥1 이 0.6387 → 0.6528 lift 의 실질 mechanism 임을 입증
> - **partial_drift_above_p024**: 0.6387 < hit_1cm ≤ 0.6528 → paradigm 새 lever 가 plan-024 ceiling 위로 lift 만들었으나 plan-022 LGBM floor 미달
> - **partial_drift_below_p024**: 0.6320 ≤ hit_1cm ≤ 0.6387 → plan-024 plateau 동일 region. 새 lever 가 paradigm ceiling 못 깸 (plan-024 §5.13 4-way OOF plateau 재현)
> - **regression**: hit_1cm < 0.6320 → paradigm mismatch 본질 (F0 미달)
>
> **out-of-scope**: ensemble (plan-030 후보) / DACON LB submit / boundary corrector / F0 ML / anchor layout 변경 / τ_cls 변경 / hidden ≠ 196 sweep / batch ≠ 64.

---

## §0.5 Quick Reference (autonomous loop 매 turn 읽는 section)

### 합격 기준 (G-gate sequence)

- **G0**: 5 module (model / run_oof / train / tests + plan-024 cherry-pick 9 file) import + smoke + tests green. plan-022 / plan-024 / plan-025 carry import 정상. 위반 시 `infra_drift` severe.
- **G1**: F0 baseline + plan-022 winner reproduce (plan-025 baseline_carry.json carry 또는 재산출). hit_1cm F0 ∈ [0.6315, 0.6325] AND plan-022 ∈ [0.6523, 0.6533]. 위반 시 `f0_reproduce_drift` / `plan022_reproduce_drift` severe.
- **G2.X1**: X1 cell 5-fold OOF metric finite + `max_class_ratio < 0.95`. mode collapse 표시 = `max_class_ratio ∈ [0.05, 0.1]` (near 1/K=0.071) — paradigm mismatch evidence. 위반 시 `numerical` severe / `mode_collapse` warn.
- **G3 (paradigm)**: PASS (>0.6528) / partial_drift_above_p024 (0.6387~0.6528) / partial_drift_below_p024 (0.6320~0.6387) / regression (<0.6320) 판정 (§0 criterion).
- **G_final**: results.md + 3-file frontmatter sync + follow-up plan-030 (가칭) 박제.

### G-gates

- G0: STAGE 0 인프라 + plan-024 cherry-pick (model.py + feature_weighted_dropout.py) [TODO]
- G1: STAGE 1 F0 + plan-022 winner reproduce [TODO]
- G2.X1: X1 5-fold OOF [TODO]
- G3: STAGE 3 paradigm 판정 [TODO]
- G_final: STAGE 4 results + 3-file sync [TODO]

### Commit chain (next-up)

| # | type | spec section | status |
|---|---|---|---|
| c1 | docs | `plans/plan-029-grunet-input-max.md` v1 작성 | [TODO] |
| c2 | chore | plan-024 추가 cherry-pick from `worktree-plan-024-combo` (commit 915dd26): `model.py` + `feature_weighted_dropout.py`. 기존 plan-025 cherry-pick (anchor_vocab/cand_builder/seq_builder/torsion_calc/quantile_carry/multiwindow_trim_build + json + __init__) 외 추가 2 file. | [TODO] |
| c3 | code | `analysis/plan-029/model.py` — 신규 `GRUNetX1` class. plan-024 `CandidateAttentionGRUSelectorCarry` 의 GRU + query_mlp + cross-attention computation 만 carry (instantiate 후 `backbone.head` 는 사용 X — 새 head 가 1472D 입력이므로 backbone.head 의 hardcoded `Linear(2*196+150=542, 196)` 와 dim mismatch). FWD wrapper class `CrossAttentionAnchorSelector` 도 import X (FWD off, `cand_drop_p=seq_drop_p=0` 와 동일 효과). 신규 head: `Linear(1472→384) → GELU → Dropout(0.15) → Linear(384→1)`. hidden=196, anchor_embed_dim=0, GRU dropout=0.10, head_dropout=0.15. | [TODO] |
| c4 | code | `analysis/plan-029/train.py` — PyTorch 5-fold OOF training loop (epoch=50 fixed, lr=7e-4 SequentialLR[warmup 5 ep + cosine T_max=45 ep], AdamW wd=1e-4, GRU dropout=0.10, head_dropout=0.15, gradient_clip=1.0, batch=64, soft cross-entropy loss, `model.train()` 명시). | [TODO] |
| c5 | code | `analysis/plan-029/run_oof.py` — orchestrator + G1 reproduce + 5-fold concat OOF + final metric. CLI `--cell X1` 또는 `--g1`. | [TODO] |
| c6 | test | `tests/test_plan029_smoke.py` — 8+ pytest (import / model forward shape / training step / soft label sum=1 / GRU input shape / cross-attention shape / Frenet→world 식 / plan-025 build_feat_1080 carry). | [TODO] |
| G0 | gate | smoke + tests green (예상 < 300s) | [TODO] |
| c7 | exp G1 | F0 + plan-022 winner reproduce (plan-025 baseline_carry.json 재사용 또는 재산출) → `baseline_carry.json` | [TODO] |
| G1 | gate | F0 hit ∈ tight band ✓ AND plan-022 winner hit ∈ tight band ✓ | [TODO] |
| c8 | exp G2.X1 | X1 5-fold OOF GRU-attention 학습. 예상 runtime: CPU **7-9 min (~420-500s, §6.2 추정)**. plan-024 167s 의 2.5~3× (batch 256→64 4× step + epoch 22→50 2.27× + hidden 196 0.27× FLOPs + head 1472D 의 head-FLOPs 2.72×). `results_X1.json` + `train_X1.log` 박제 + 실측 runtime 비교. | [TODO] |
| G2.X1 | gate | metric finite + max_class_ratio < 0.95. 또한 epoch 50 fully trained 검증 (early stop disabled) | [TODO] |
| c9 | analysis | X1 결과 + paired Δ vs F0 + paired Δ vs plan-022 winner + 14-anchor oracle 회수율 + mode collapse 진단 → `paradigm_analysis.{json,md}` | [TODO] |
| G3 | gate | paradigm 판정 (PASS / partial_drift / regression) | [TODO] |
| c10 | docs | 3-file frontmatter sync + `analysis/plan-029/results.md` + `plans/plan-029-*.results.md` pair + follow-up plan-030 (가칭) 박제 | [TODO] |
| G_final | gate | 3-file sync + §0.5 c1~c10 [DONE] | [TODO] |

### Plan-specific severe

- `infra_drift`: plan-024 cherry-pick 또는 plan-025 carry module import 실패.
- `f0_reproduce_drift` / `plan022_reproduce_drift`: G1 reproduce tight band 위반.
- `numerical`: PyTorch forward / backward NaN/Inf.
- `mode_collapse` (warn): max_class_ratio ∈ **[0.05, 0.10)** (1/K=0.0714 근방 ± tolerance). H3 임계 (> 0.10) 와 정확히 align — uniform 출력 = paradigm mismatch finding 으로 박제, G2 계속 진행.
- `model_capacity_overflow`: GPU/CPU OOM 또는 학습 시간 > 30 min (§6.2 추정 7-9 min 의 ~3.5× 초과 시 spec 가정 위반 — DataLoader I/O / numpy↔torch conversion bottleneck 등 사후 분석 trigger). 30 min 미만이면 정상 진행.
- `plan024_cherry_pick_missing`: c2 cherry-pick 후 model.py / feature_weighted_dropout.py importlib 실패 → halt.

### Plan-specific paths

- whitelist:
  - `analysis/plan-029/**`
  - `tests/test_plan029_smoke.py`
  - `analysis/plan-024/{model.py, feature_weighted_dropout.py}` — **c2 cherry-pick 단계 유일 plan-024 path 수정 허용** (add only)
- blacklist: `analysis/plan-{001..028}/**` (read-only import 예외)

### Decision-note 사용 예

- `decision-note: spec-default — GRU encoder input = raw seq (B, T=7, C=95) from seq_builder.build(). plan-025 block ④ 760D (8-stat 압축) 는 head skip 으로 사용. raw seq 가 GRU 학습 source.`
- `decision-note: spec-default — cross-attention query = cand_feat (B, K=14, 150) from cand_builder.build(). plan-025 block ②③ 의 source. query MLP 입력.`
- `decision-note: spec-default — head skip = concat(h_final_bc 196, event_ctx 196, cand_feat 150, block① 170, block④ stat 760) → MLP → score. block ② ctx 128D 는 cand_feat 안에 포함 (묶음③ slice [12:140]). head_in dim = 196+196+150+170+760 = 1472D.`
- `decision-note: spec-default — anchor_embed_dim=0 (plan-024 v5 default OFF carry). 사용자 명시. plan-024 §5.8 A7 learnable embedding 8D 도 hit_1cm 변화 X (부분 기각) → identity capacity lever 본 plan out-of-scope.`
- `decision-note: spec-default — hidden=196 (사용자 명시). plan-024 384 의 51%. capacity 축소. plan-024 §5.10 poss 1 (h128) 과 §5.11 carry (h384) 사이 unexplored region.`
- `decision-note: spec-default — training schedule = epoch=50 fixed (no early stop). plan-024 §5.1 under-converged 가설 *기각* 후, 50 epoch 은 §5.10 long-diag best ep=35 + 안전 마진 + cosine annealing 완주. lr=7e-4 + SequentialLR([LinearLR warmup 5 ep, CosineAnnealingLR T_max=45 ep]). AdamW (weight_decay=1e-4). GRU dropout=0.10. head_dropout=0.15 (별개 lever). gradient_clip=1.0. batch=64 (plan-024 256 의 1/4, plan-024 미시도 axis). soft cross-entropy loss.`
- `decision-note: spec-default — head_dropout=0.15 + GRU dropout=0.10 *분리 hparam*. §3.5 hparam table 별도 row. plan-024 carry 동일.`
- `decision-note: spec-default — FWD (FeatureWeightedDropout) **off**. plan-024 `CrossAttentionAnchorSelector` outer wrapper import X. plan-024 §5.4 v3 (cand_drop_p=0, seq_drop_p=0) 가 v1 대비 noise (+0.0003) 였으므로 wrapper 자체 제거가 더 simple.`
- `decision-note: spec-default — wrapper class `GRUNetX1` = backbone `CandidateAttentionGRUSelectorCarry` instantiate 후 그 안의 GRU + query_mlp + cross-attention computation 만 호출, `backbone.head` 는 dim mismatch 로 사용 X. 자체 신규 head `Linear(1472→384) → GELU → Dropout(0.15) → Linear(384→1)` 정의. (대안: backbone subclass 후 `self.head = Identity` overwrite — 둘 다 결과 동일, c3 구현 시 자유 선택.)`
- `decision-note: spec-default — attention scaling 1/sqrt(196) ≈ 0.0714 (plan-024 1/sqrt(384) ≈ 0.0510 대비 40% 큰 분모 → 동일 logits scale 일 때 softmax 더 peaked). warmup 5 epoch + lr=7e-4 가 attention 학습 초기 안정 마진. plan-024 carry 와 *수치 차이* 만 있고 식 동일.`
- `decision-note: spec-default — random_state=20260522 (본 plan layer). plan-024 SEED=20260521 reproduce 와 별개. 모든 fold 동일 seed (plan-024 §5.13 carry 와 일관).`
- `decision-note: spec-default — input feature 의 NaN/Inf 처리 = torch.nan_to_num(input, nan=0.0, posinf=1e3, neginf=-1e3) before forward. plan-021/024 의 sigmoid overflow warning 잔재 대응.`
- `decision-note: spec-default — model.train() 명시 호출 (§6.1 fold loop epoch 진입 직전). plan-024 run_oof.py:159 carry 와 동일 — FWD off 라도 GRU dropout=0.10 / head_dropout=0.15 가 self.training flag 분기를 따르므로 train mode 명시 필수.`

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
- **plan-024 honest ceiling 0.6387 인지**: 가설 H1/H2/H3 는 §5.13/§5.14 의 plateau / ceiling 을 base 로 정의 (under-converged 의심 가설 *기각된 후* 의 lever 재선정).
- **새 lever (plan-024 미시도) 검증**: (i) head input expansion 1472D (542→1472, plan-024 모든 variant 가 backbone hardcoded head 542D 만 시도), (ii) hidden 384→196 (plan-024 §5.10 poss 1 의 h128 ↔ §5.11 의 h384 사이 unexplored region), (iii) batch 256→64 (4× effective step). plan-024 §5.10 augmentation lever (1-fold +0.0135) 는 본 plan out-of-scope (사용자 미요청).
- **plan-025 1080D input 정상 활용**: block ① 170D + block ④ 760D head skip + raw cand_feat (block②③) attention 분해 (§0 paradigm rationale 2/3). plan-024 §5.9 v6 의 input carry-redundant finding (+0.0003) 은 LGBM 위 결과 → GRU-attention 위 결과 미검증.
- **abandoned plan-026 + plan-027 통합 재발행** (사용자 plan-026/027 GRU-attention 의도 합의).

### §1.2 paradigm 가설

plan-024 honest ceiling 0.6387 (3-seed) / 0.6375 ± 0.0004 (4-way OOF plateau) base. *기각된 가설* (under-conv §5.1, over-reg §5.4, anchor identity §5.8) 은 H 가설로 재제기 X. *미시도 lever* 만 검증.

- **H1 (강, 핵심)**: GRU-attention 위 plan-024 미시도 lever 조합 (head 1472D / hidden 196 / batch 64) → hit_1cm ≥ 0.6528. plan-024 §5.13 4-way OOF plateau 0.6375 의 +0.0153 lift 가 head capacity 확장 + hidden 축소 + batch 축소 의 *세 미시도 axis 합* 으로 달성 가능한지 검증.
- **H1a (보조)**: hit_1cm > 0.6387 (plan-024 honest ceiling 3-seed 초과). head 1472D / hidden 196 / batch 64 중 ≥1 lever 가 plan-024 4-way plateau 위 *어떤 lift* 라도 만들었는지 진단.
- **H2 (약)**: hit_1cm > 0.6531 (plan-022/023 winner 초과). paradigm-distinct lever 가 LGBM anchor-selector ceiling 위 추가 lift.
- **H3 (강)**: max_class_ratio > 0.10 (mode collapse 미발생). plan-025 LGBM 의 1/K=0.0714 uniform 와 *질적으로 다름*. plan-024 v1 가 0.1047 (mode collapse 아닌 evidence), plan-029 도 동등 이상 예상.

판정:
- H1 PASS → 새 lever ≥1 의 실질 mechanism 확정. plan-030 후속 = 단일-lever ablation 분해 (head / hidden / batch 의 contribution).
- H1 FAIL + H1a PASS → plan-024 ceiling 위 lift 만 있고 plan-022 floor 미달. plan-030 후속 = augmentation (plan-024 §5.10 poss 3 carry) + 3-seed ensemble 추가.
- H1 FAIL + H1a FAIL → plan-024 §5.13 plateau 재현. paradigm 자체 한계 재확정 → F0 ML / corrector 로 전환 (paradigm-distinct lever).
- H3 FAIL (mode collapse, max_class_ratio ∈ [0.05, 0.10)) = paradigm 본질 fail. plan-024 v1 의 0.1047 와 비교 시 매우 unlikely 단 head path 의 cand_feat raw concat 이 self-prediction trigger 재현 risk 존재 (cand_feat 안의 block ③ 22D 가 GRU-attention 안에서도 categorical memorization 일으킬 가능성 — paradigm rationale 2 의 reasoning 검증).

### §1.3 baseline anchor

- **G1.a F0** (plan-020 carry): 0.6320 / 0.8033. 모든 paired Δ anchor.
- **G1.b plan-022 winner** (carry from plan-025 baseline_carry.json 또는 재산출): 0.6531 / 0.8108. paradigm ceiling reference.

---

## §2. 가설 검증 paradigm (한 변수 원칙)

| 축 | 변경 | 단일 변수 |
|:--|:--|:--|
| Anchor codebook | K=14 BCC fix | ✗ (carry) |
| τ_cls | 0.001 fix | ✗ (carry) |
| Soft label 산식 | `build_soft_label_with_tau` | ✗ (carry) |
| 5-fold split | `stable_fold_id` | ✗ (carry) |
| F0 baseline | `f0_baseline` | ✗ (carry) |
| Input source 1080D | plan-025 동일 (+ raw seq 7×95 GRU input) | ✗ (carry) |
| **Paradigm framework** | **cross-attention GRU selector (plan-024 framework 유지)** | ✗ (carry — 본 plan 은 framework 유지) |
| Head input dim | **542D → 1472D** (plan-025 block ①+④ 930D skip 추가, plan-024 미시도) | **✓ 새 lever (a)** |
| GRU hidden | **384 → 196** (plan-024 §5.10 poss 1 h128 ↔ §5.11 carry h384 사이 unexplored) | **✓ 새 lever (b)** |
| Batch size | **256 → 64** (effective step 4×, plan-024 미시도) | **✓ 새 lever (c)** |
| anchor_embed_dim | 0 (plan-024 v5 default OFF, §5.8 부분 기각) | ✗ (carry) |
| FWD | off (plan-024 §5.4 기각) | ✗ (carry) |
| Training schedule | 50 epoch fixed + cosine + warmup 5 (§5.1 under-conv 기각 후 lever) | (보조 — schedule = lever (a)(b)(c) 지원) |

본 plan 의 **3개 새 lever 동시 적용** (단일 cell X1 = 3 lever 묶음). plan-024 §5.5 "16 lever 동시 추가 → bottleneck 분해 불가" caveat 재발 risk 인지 — H1 PASS 시 plan-030 = single-lever ablation (head only / hidden only / batch only) 분해 follow-up. 사용자 plan-026/027 GRU-attention 의도 통합 + 사용자 hidden=196 명시 위에서 X1 = 3 lever 통합 검증 1회 plan.

---

## §3. 사전 등록 (Pre-registration)

### §3.1 Fold split (plan-020/021/022/025 carry)

- 5-fold rotating, `stable_fold_id(sample_id_str, n_folds=5)`. MD5 deterministic.
- N=10000 samples → per-fold test ≈ 2000, train ≈ 8000.
- dataset_hash 일치 (plan-025 baseline_carry.json 의 hash carry).

### §3.2 합격 기준

| Gate | 합격 |
|:--|:--|
| G0 | 8+ pytest green + plan-024 cherry-pick 9 file import OK |
| G1.a | F0 hit_1cm ∈ [0.6315, 0.6325] AND hit_1p5cm ∈ [0.8028, 0.8038] |
| G1.b | plan-022 winner hit_1cm ∈ [0.6523, 0.6533] AND hit_1p5cm ∈ [0.8099, 0.8109] |
| G2.X1 | metric finite + max_class_ratio < 0.95 (no extreme winner) + epoch 50 fully trained ✓. max_class_ratio ∈ [0.05, 0.10) 시 `mode_collapse` warn 박제 후 계속 진행 |
| **G3** | PASS hit_1cm > 0.6528 / partial_drift_above_p024 0.6387 < hit ≤ 0.6528 / partial_drift_below_p024 0.6320 ≤ hit ≤ 0.6387 / regression < 0.6320 |
| G_final | 3-file sync + §0.5 c1~c10 [DONE] + follow-up 1+ 건 |

### §3.3 평가 점수

- **primary**: `hit_1cm` = mean(D1(pred, gt) ≤ 0.01). 5-fold concat OOF.
- **secondary**: `hit_1p5cm`, `top1_acc` (argmax probs vs gt anchor label), `max_class_ratio` (mode collapse 진단).
- **paired Δ**: vs F0 (G1.a) + vs plan-022 winner (G1.b).
- **14-oracle 회수율**: `best_hit_1cm / 0.7928`.

### §3.4 Model spec (X1 cell)

```python
# §3.4.1 forward path (단일 cell X1)
seq         = seq_builder.build(X, R_wfn, ANCHORS_A6, f0_baseline, quantile_carry)   # (B, 7, 95)
cand_feat   = cand_builder.build(X, R_wfn, pred_F0, ANCHORS_A6, f0_baseline, regimes, quantile_carry)  # (B, 14, 150)
feat_1080   = build_feat_1080(X, ANCHORS_A6, f0_baseline, quantile_carry)            # (B*14, 1080) — head skip source

# GRU encoder
out, h      = GRU(input_size=95, hidden=196, num_layers=2, dropout=0.10, batch_first=True)(seq)  # out (B, 7, 196), h (2, B, 196)
h_final     = h[-1]                                                                              # (B, 196)
h_final_bc  = h_final.unsqueeze(1).expand(-1, 14, -1)                                            # (B, 14, 196)

# Cross-attention (anchor query × seq key)
query       = query_mlp(cand_feat)                                                               # (B, 14, 196) — Linear(150→196)+GELU+Linear(196→196)
attn_logits = einsum("bth,bkh->bkt", out, query) / sqrt(196)                                     # (B, 14, 7)
attn        = softmax(attn_logits, dim=-1)                                                       # (B, 14, 7)
event_ctx   = einsum("bkt,bth->bkh", attn, out)                                                  # (B, 14, 196)

# Head MLP — skip connection 강화 (plan-025 block ①+④ 추가). **plan-024 미시도 lever (head input 542→1472).**
# ⚠️ plan-024 `CandidateAttentionGRUSelectorCarry.head` 는 hardcoded `Linear(2*hidden+cand_dim=542, hidden=196)` 임 — backbone instantiate 후 그 head 는 **사용 X**. 새 head 가 1472D 입력이므로 dim mismatch. wrapper `GRUNetX1` 는 backbone 의 GRU + query_mlp + cross-attention 만 carry, 자체 head 정의.
feat_1080_unflat = feat_1080.reshape(B, 14, 1080)                                                # (B, 14, 1080) — sample-major row order (build_feat_1080: np.repeat(block1, K, axis=0))
block1_skip      = feat_1080_unflat[:, 0, 0:170]                                                 # (B, 170) — anchor 무관 generic (block1/4 가 sample-level repeat 이므로 anchor 0 slice 가 모든 anchor 와 동일)
block4_skip      = feat_1080_unflat[:, 0, 320:1080]                                              # (B, 760) — 170+128+22=320 → block4_exp slice
block1_bc        = block1_skip.unsqueeze(1).expand(-1, 14, -1)                                   # (B, 14, 170)
block4_bc        = block4_skip.unsqueeze(1).expand(-1, 14, -1)                                   # (B, 14, 760)
head_in     = concat([h_final_bc, event_ctx, cand_feat, block1_bc, block4_bc], dim=-1)           # (B, 14, 196+196+150+170+760 = 1472)
head        = Linear(1472 → 384) → GELU → Dropout(0.15) → Linear(384 → 1)                       # 새 head, 신규 instantiate. param ≈ 1472×384 + 384 + 384×1 + 1 = 566,145 (head 단독, total 의 ~57%)
score       = head(head_in).squeeze(-1)                                                          # (B, 14)
probs       = softmax(score, dim=-1)                                                             # (B, 14)
```

**block ③ self-prediction trigger 차단 reasoning** (§0 paradigm rationale 2 보강): head_in 의 `cand_feat` (B, 14, 150) 안에 block ③ 22D per-anchor 가 raw 로 포함되지만, (i) LGBM tree 의 sharp categorical split 와 달리 head 의 GELU + Dropout 비선형 + 다른 channel (h_final_bc 196 + event_ctx 196 + block1_bc 170 + block4_bc 760 = 1322D) 과 mixed projection 으로 단일 anchor identity memorization 메커니즘이 *수치적으로 다름*. (ii) head 의 Linear(1472→384) 가 1472D 전체에서 단일 384D bottleneck 으로 압축하므로 anchor 별 raw 22D 가 *독립적* 으로 score 에 매핑되지 않음. 단 H3 검증 (max_class_ratio > 0.10) 으로 사후 확인.

### §3.5 Training schedule (X1)

| Hparam | 값 | 사유 |
|:--|--:|:--|
| epochs | **50 fixed** | plan-024 §5.10 long-diag best ep=35 + 안전 마진. under-converged 가설 §5.1 기각 후 schedule lever (cosine + warmup) 완주 보장 |
| early_stopping | **disabled** | plan-024 §5.1 v2 (patience 999) 와 동일 axis. *early stop noise 회피* + fold-internal val_loss best state 가 hit_1cm best 와 decoupled (§5.12 finding) |
| optimizer | AdamW | 표준 |
| lr | 7e-4 | attention 표준 + plan-024 carry |
| lr_schedule | `SequentialLR([LinearLR warmup 5 epoch, CosineAnnealingLR T_max=45 epoch])` total=50 | warmup 5 + cosine 45 (warmup 이 total 50 epoch 안에 *포함*. T_max=45 = 50-5 으로 lr 가 epoch 50 시점에 ~0 도달) |
| weight_decay | 1e-4 | AdamW 표준 (plan-024 carry 의 0.02 보다 *느슨*. plan-024 §5.4 channel drop off 도 효과 X 였으므로 강한 reg 회피) |
| GRU dropout | 0.10 | plan-024 carry (`nn.GRU(num_layers=2, dropout=0.10)` = layer1→layer2 사이 dropout) |
| head_dropout | 0.15 | plan-024 backbone carry. 새 head Linear(1472→384) 다음 Dropout(0.15) |
| FWD (input dropout) | **off** | plan-024 §5.4 channel drop off (cand_drop_p=0, seq_drop_p=0) 가 v1 0.6370 → v3 0.6373 = noise. `CrossAttentionAnchorSelector` outer wrapper 자체 import X |
| gradient_clip | 1.0 | attention 안정 |
| batch_size | 64 | plan-024 batch 256 의 1/4. step 4× 늘려 effective gradient noise ↑ (plan-024 미시도 lever) |
| random_state | 20260522 | 본 plan layer (모든 fold 동일 seed, plan-024 §5.13 carry 와 일관). data shuffling + model init 둘 다 적용 |
| τ_cls | **0.001** | plan-022 carry (frontmatter scope + §0.5 박제). `build_soft_label_with_tau` 의 인자 |
| K (anchor count) | **14 BCC** | plan-022 carry. ANCHORS_A6 |
| loss | **soft cross-entropy** | 식: `loss = -(soft_q * log_softmax(score)).sum(dim=-1).mean()` over K=14. `score` = model logits. `soft_q` = build_soft_label_with_tau output (sum=1). plan-022 / plan-024 (run_oof.py:167) 동일 식 + 명칭 carry. *수학적으로 KL(q‖p) = soft_CE − H(q)* 단 H(q) 가 model param 무관 → gradient 동일. plan-022/024 carry 명칭 "soft cross-entropy" 사용. |

### §3.6 Loss 식 (soft cross-entropy)

```python
log_probs = log_softmax(score, dim=-1)             # (B, 14) — log_softmax 가 softmax→log 보다 수치 안정 (plan-024 model.py smoke 의 log(q_pred+1e-12) 대비 개선)
soft_q    = build_soft_label_with_tau(gt, R_wfn, F0, ANCHORS_A6, tau_cls=0.001)  # (B, 14), row-sum = 1
loss      = -(soft_q * log_probs).sum(dim=-1).mean()      # H(soft_q, p_model) = soft cross-entropy
```

- K-axis sum → batch mean. plan-022/024 (run_oof.py:167) 동일 reduction + 식 (`-(q_b * torch.log(q_pred + 1e-12)).sum(-1).mean()`).
- "soft cross-entropy" 명명은 plan-024 spec §4.7 carry. KL(q‖p) 대비 entropy H(q) 만큼 offset → gradient 는 동일.
- `score` (logits) 만 사용 — model forward 가 `(q_pred, score)` tuple 반환 시 `score` 만 unwrap (`q_pred = F.softmax(score, dim=-1)` 는 eval 시 별도 계산).

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
├── model.py                 ← 신규 `GRUNetX1` (plan-024 `CandidateAttentionGRUSelectorCarry` 의 GRU + query_mlp + cross-attention 만 carry, backbone.head 미사용, FWD off, 새 head 1472→384→1) (c3)
├── train.py                 ← PyTorch 5-fold OOF training (c4)
├── run_oof.py               ← orchestrator + G1 reproduce + CLI (c5)
├── baseline_carry.json      ← G1 박제 (c7)
├── results_X1.json          ← G2.X1 박제 (c8)
├── train_X1.log             ← 학습 진행 log (c8)
├── paradigm_analysis.{json,md}  ← c9
└── results.md               ← c10

analysis/plan-024/            ← c2 추가 cherry-pick from worktree-plan-024-combo
├── model.py                  ← 신규 cherry-pick
└── feature_weighted_dropout.py  ← 신규 cherry-pick
(기존 8 file: __init__.py + anchor_vocab.py + cand_builder.py + seq_builder.py + torsion_calc.py + quantile_carry.py + multiwindow_trim_build.py + multiwindow_trim.json — plan-025 c2 cherry-pick 으로 이미 main 존재)

tests/test_plan029_smoke.py   ← 8+ pytest (c6)
```

### §4.2 plan-024 cherry-pick (c2)

```bash
git checkout worktree-plan-024-combo -- analysis/plan-024/model.py analysis/plan-024/feature_weighted_dropout.py
```

commit hash carry: `worktree-plan-024-combo` 의 latest (commit 915dd26 또는 그 이후 minor patch).

### §4.3 tests (c6)

- `test_imports`: plan-024 model + feature_weighted_dropout (모듈 import 만, FWD wrapper class instantiate X) + plan-025 build_feat_1080 + plan-022 anchors + plan-021 build_input + plan-020 baseline_f0 모두 import OK.
- `test_model_forward_shape`: dummy `seq (B=4, 7, 95)` + `cand_feat (B=4, 14, 150)` + `feat_1080 (B*14, 1080)` → score (B=4, 14).
- `test_head_in_dim_1472`: `GRUNetX1` forward 중간 head_in tensor shape = (B, 14, 1472). backbone.head 미사용 + 새 head Linear(1472→384) 정합.
- `test_gru_hidden_dim`: GRU encoder hidden = 196 (config).
- `test_anchor_embed_default_off`: model.anchor_embed_dim == 0 (default).
- `test_fwd_not_used`: GRUNetX1 instance 에 FeatureWeightedDropout 모듈 attribute 부재 (FWD off 검증).
- `test_train_mode_dropout_active`: model.train() 모드에서 dropout 적용, model.eval() 모드에서 미적용 (forward 2회 결과 std 차이로 검증).
- `test_soft_label_sum_one`: build_soft_label_with_tau output row-sum = 1.
- `test_frenet_to_world_inverse`: round-trip Frenet → world → Frenet (identity within tolerance).
- `test_soft_ce_loss_nonneg`: soft CE loss ≥ 0 (model prob = uniform 시 loss = log K, 그 외 ≥ 0).
- `test_build_feat_1080_carry`: plan-025 build_feat_1080 output shape (B*14, 1080) + BLOCK_DIMS sum = 1080.

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

---

## §6. STAGE 2 — X1 cell 5-fold OOF (c8)

### §6.1 Per-fold loop

```python
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
    cand_tr = cand_builder.build(X_tr, R_wfn_tr, F0_tr, ANCHORS_A6, f0_baseline,
                                  regimes=assign_regimes(X_tr, end_idx=10, bins=fit_regime_bins(X_tr, end_idx=10)),
                                  quantile_carry=qc)
    seq_tr  = seq_builder.build(X_tr, R_wfn_tr, ANCHORS_A6, f0_baseline, quantile_carry=qc)
    feat_1080_tr = build_feat_1080(X_tr, ANCHORS_A6, f0_baseline, qc)
    # (test 동일 — 코드 생략)

    # Soft label
    q_tr = build_soft_label_with_tau(gt_tr, R_wfn_tr, F0_tr, ANCHORS_A6, tau_cls=0.001)

    # Model + Optimizer
    # GRUNetX1: 자체 head(1472→384→1). backbone.head 미사용 (dim mismatch).
    # FWD off — outer CrossAttentionAnchorSelector wrapper 자체 import X.
    model = GRUNetX1(seq_dim=95, cand_dim=150, hidden=196, anchor_embed_dim=0,
                     gru_dropout=0.10, head_dropout=0.15)
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
    
    # Training (epoch=50 fixed, no early stop). model.train() 명시 (GRU dropout / head_dropout 가 self.training 분기).
    for epoch in range(50):
        model.train()
        for batch in batched(64):
            optimizer.zero_grad()
            score = model(seq_batch, cand_batch, feat_1080_batch)   # score logits (B, 14)
            log_probs = log_softmax(score, dim=-1)
            loss = -(q_batch * log_probs).sum(dim=-1).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()
    
    # Eval
    model.eval()
    with torch.no_grad():
        probs_te = softmax(model(seq_te, cand_te, feat_1080_te), dim=-1).cpu().numpy()  # (N_te, 14)
        residual_frenet = (probs_te[:, :, None] * ANCHORS_A6[None, :, :]).sum(axis=1)
        residual_world = np.einsum("nij,nj->ni", R_wfn_te, residual_frenet)
        final_pred = F0_te + residual_world
        oof_pred[test_idx] = final_pred
        oof_probs[test_idx] = probs_te

# Concat OOF metric
err = np.linalg.norm(oof_pred - gt, axis=1)
hit_1cm = (err <= 0.01).mean()
hit_1p5cm = (err <= 0.015).mean()
max_class_ratio = oof_probs.mean(axis=0).max()
top1_acc = (oof_probs.argmax(axis=1) == gt_anchor_label).mean()
```

### §6.2 Runtime + Param 예상

**param 추정** (head 단독 ~57%):
- GRU(input=95, hidden=196, num_layers=2, batch_first): ≈ 2 × 3 × (95+196+1) × 196 ≈ 343K
- query_mlp (Linear 150→196 + GELU + Linear 196→196): 150×196 + 196 + 196×196 + 196 ≈ 68K
- new head (Linear 1472→384 + GELU + Dropout + Linear 384→1): 1472×384 + 384 + 384 + 1 ≈ **566K (dominant)**
- anchor_embed off
- **total ≈ 977K (~1.0M)**. plan-024 hidden=384 backbone total ~3.5M 의 28%. 단 head share = 57% vs plan-024 의 ~6% — head overfit risk 박제 (plan-030 후속 lever: "head 1472→768→384→1 narrower bottleneck").

**runtime 추정** (plan-024 §2 의 5-fold OOF 167s base):
- plan-024: 22 epoch × batch 256 × hidden 384^2 = baseline
- plan-029: 50 epoch × batch 64 × hidden 196^2 = 50/22 × 256/64 × 196^2/384^2 = 2.27 × 4 × 0.260 ≈ **2.36×** plan-024 FLOPs 단 step 4× (batch 1/4) 이므로 GRU+attention compute scale ≈ 2.36×
- 단 head FLOPs (1472→384) 는 plan-024 (542→384) 의 2.72× 추가 → head 영향 반영 시 ≈ 2.5×~3× plan-024
- → **5-fold total ≈ 167s × 2.5~3 = 420-500s (~7-9 min CPU)**. *NOT 2.5-3.5h*.
- 만약 실측이 30 min/fold (2.5h total) 면 *어떤 추가 bottleneck* (DataLoader I/O / head FLOPs underestimate / numpy↔torch conversion) 이 trigger 된 것 — G2 학습 시 actual runtime 박제 후 사후 분석. 그 자체로 plan-024 §5.1 under-converged 가설 부활 evidence 가 *아님* (architecture 차이).

### §6.3 G2.X1 합격

- metric finite ✓ (NaN/Inf X)
- max_class_ratio < 0.95 ✓ (no extreme winner)
- max_class_ratio ∈ [0.05, 0.10) → `mode_collapse` warn 박제 후 계속 (H3 FAIL 판정 input)
- epoch 50 fully trained ✓ (no early stop)
- 위반 (numerical / overflow > 30 min / cherry-pick missing) = severe halt

---

## §7. STAGE 3 — Paradigm finding (c9, G3)

### §7.1 X1 결과 표

| Metric | X1 | F0 (G1.a) | plan-022 (G1.b) | plan-024 v1 | plan-024 3-seed (§5.14) | plan-024 §5.10 1-fold lucky | plan-025 C1 |
|:--|--:|--:|--:|--:|--:|--:|--:|
| hit_1cm | ?.???? | 0.6320 | 0.6531 | 0.6370 | 0.6387 | 0.6505 | 0.6320 |
| hit_1p5cm | ?.???? | 0.8033 | 0.8108 | 0.8092 | 0.8096 | — | 0.8033 |
| max_class_ratio | ?.??? | — | 0.1054 | 0.1047 | ? | ? | 0.0714 |
| top1_acc | ?.???? | — | 0.1707 | 0.1227 | ? | ? | 0.0879 |
| oracle 회수율 | ?.??% | — | 82.4% | 80.4% | 80.6% | 82.3% | 79.7% |
| runtime | ?s | — | (carry) | 167s | 3×167s | 154s 1-fold | 334s |
| paradigm 위치 | — | — | LGBM floor | spec default | **honest ceiling** | lucky catch | LGBM mode collapse |

### §7.2 G3 판정

- **PASS** (band=positive): hit_1cm > 0.6528 → 새 lever (head 1472D / hidden 196 / batch 64) 중 ≥1 의 실질 mechanism 확정. plan-024 4-way OOF plateau 0.6375 의 +0.0153 lift 달성. follow-up plan-030 = single-lever ablation 분해.
- **partial_drift_above_p024** (band=partial_above): 0.6387 < hit_1cm ≤ 0.6528 → plan-024 3-seed ensemble honest ceiling 위 lift 단 plan-022 LGBM floor 미달. follow-up = plan-024 §5.10 augmentation (poss 3 σ=0.05) + 3-seed ensemble 추가 검증.
- **partial_drift_below_p024** (band=partial_below): 0.6320 ≤ hit_1cm ≤ 0.6387 → plan-024 §5.13 4-way OOF plateau 재현. 새 lever 모두 무효 = paradigm 자체 한계 재확정. follow-up = F0 ML / corrector / KNN retrieval (paradigm-distinct).
- **regression** (band=negative): hit_1cm < 0.6320 → F0 미달 = paradigm mismatch 본질 (head path 의 cand_feat self-prediction trigger 재발 또는 1080D head expansion 의 overfit catastrophic). 사후 mode_collapse 진단 추가.

### §7.3 Hypothesis 검증

- **H1 (강, 핵심)** (hit_1cm ≥ 0.6528): PASS / FAIL
- **H1a (보조)** (hit_1cm > 0.6387 plan-024 honest ceiling): PASS / FAIL
- **H2 (약)** (hit_1cm > 0.6531 plan-022/023 winner 초과): PASS / FAIL
- **H3 (강)** (max_class_ratio > 0.10, mode collapse 미발생): PASS / FAIL

조합 시나리오:
- H1 PASS → 새 lever 실질 mechanism 확정 (plan-030 ablation 분해 follow-up)
- H1 FAIL + H1a PASS → plan-024 ceiling 위 lift 만 (plan-030 = augmentation + 3-seed ensemble)
- H1 FAIL + H1a FAIL → plan-024 §5.13 plateau 재현 (paradigm 한계 재확정)
- H3 FAIL → mode collapse trigger (cand_feat self-prediction 위험 사후 확정)

### §7.4 paradigm finding 박제

- plan-024 §5.13 honest ceiling 0.6387 대비 본 plan 새 lever (head 1472D / hidden 196 / batch 64) 의 lift 분해
- plan-025 mode collapse 의 GRU-attention 위 재현 여부 (H3 결과)
- plan-024 §5.9 v6 input carry-redundant finding 의 GRU-attention 위 재검증 (input 230D → 1080D 추가 lift 가 +0.0003 noise level 인지)
- plan-030 후속 lever 우선순위 결정:
  - H1 PASS → single-lever ablation (head only / hidden only / batch only)
  - H1 FAIL + H1a PASS → augmentation (plan-024 poss 3) + 3-seed ensemble (plan-024 §5.14 carry)
  - H1 FAIL + H1a FAIL → paradigm-distinct lever (F0 ML / corrector / KNN retrieval)

---

## §8. STAGE 4 — G_final (c10)

### §8.1 산출

- `analysis/plan-029/results.md` (11 항목)
- `plans/plan-029-*.results.md` pair
- 3-file frontmatter sync (status=all_complete, band ∈ {positive, partial_above, partial_below, negative}, best_cell=X1, best_hit_1cm, best_delta_1cm vs F0, best_delta_vs_p024_ceiling = best_hit_1cm − 0.6387)
- follow-up plan-030 (가칭) 후보 ≥ 1 건 박제 (§7.4 의 H 시나리오별 lever 우선순위 carry)

### §8.2 G_final 합격

- 3-file sync ✓
- §0.5 c1~c10 모두 [DONE] ✓
- follow-up 1+ 건 박제 ✓

---

## §9. Out of scope

- Ensemble (plan-030 후보)
- DACON LB submit (별개 결정)
- boundary corrector (plan-030 후보)
- F0 baseline ML (plan-028 후보)
- anchor layout 변경 (K=14 BCC fix)
- τ_cls 변경 (0.001 fix)
- hidden ≠ 196 sweep (단일 cell)
- batch ≠ 64, lr ≠ 7e-4 sweep
- corrector / 2-stage residual regression
- anchor_embed_dim ≠ 0 (사용자 명시 OFF)
- **input augmentation σ=0.05 (plan-024 §5.10 poss 3 의 +0.0135 1-fold lever)** — plan-030 후보 (H1 FAIL + H1a PASS 시 우선)
- **3-seed ensemble** (plan-024 §5.14 의 variance reduction +0.0010) — plan-030 후보
- **FWD (FeatureWeightedDropout) on** — plan-024 §5.4 기각 lever
- **head bottleneck 변경 (1472→768→384 등 narrower)** — plan-030 후보 (head overfit 진단 시)

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
