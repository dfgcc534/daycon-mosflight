---
plan_id: 024
version: 1.1-rev2
finished_at: 2026-05-21 (Asia/Seoul)
status: all_complete
band: negative
best_hit_1cm: 0.6387
best_hit_1.5cm: 0.8096
best_delta_1cm: +0.0067
best_delta_1.5cm: +0.0063
best_variant: 3seed_ensemble_combo_h128_aug
gap_ranking: 0.1934
g1_pass: true
g2_pass: false
g3_pass: false
g_final_state: g2_no_improvement_skip
lb_score: null
xattn_no_improvement: true
---

# plan-024 results — Cross-attention Anchor-Vocab Selector G2 FAIL

## 한 줄 결론

plan-024 의 **architecture lever 실패 (band=negative)**. 사용자 통찰 (overfit) 후 6 ablation variant + 3-seed ensemble *정직한 5-fold OOF* 종합:
- **모든 single-seed variant** 5-fold OOF hit_1cm ∈ **[0.6370, 0.6379]** (range 0.0009 noise band 안)
- **3-seed ensemble (마지막 lever)**: **0.6387** = +0.0017 lift vs v1 (paradigm 안 honest best)
- plan-022 carry 0.6528 보다 **-0.0141** 미달 — 모든 시도 paradigm 한계 못 깸
- 1-fold best (0.6490-0.6515) 는 *hit_1cm 기준 best epoch tracking 의 fold-specific lucky catch*
- → plan-024 paradigm 의 honest ceiling **0.6387 (3-seed)** 또는 **0.6377 (single-seed)**, plan-022 carry **-0.0141 ~ -0.0151 미달**

### 5.14 3-seed ensemble (plan-024 paradigm 마지막 lever) — 0.6387

variance reduction 마지막 시도. combo config (hidden 128 + aug σ=0.05) × 3 seeds (20260521/2/3).

per-seed 5-fold OOF:
| seed | hit_1cm |
|:--|--:|
| 20260521 | 0.6377 |
| 20260522 | 0.6389 |
| 20260523 | 0.6393 |
| **avg probs (ensemble)** | **0.6387** |

- vs combo single-seed: **+0.0010** (mild variance reduction lift)
- vs plan-022 carry: **-0.0141** (여전히 미달)
- ensemble hit_1.5cm: 0.8096

**plan-024 paradigm honest best = 0.6387 (3-seed ensemble)**. paradigm 안 *모든 미시도 lever 소진*, 그중 가장 높은 결과. plan-022 carry 0.6528 까지 -0.0141 여전 미달.

### 5.13 4-way 5-fold OOF 정직 평가 — plan-024 paradigm honest ceiling 확정

| variant | config | 5-fold OOF hit_1cm | 1-fold best | gap (1-fold - 5-fold) |
|:--|:--|--:|--:|--:|
| v1 default | spec, h384, drop 0.3/0.2 | 0.6370 | — | — |
| poss 1 (h128 no aug) | h128, drop 0.3/0.2, wd 0.02 | **0.6379** | 0.6490 | **-0.0111** |
| poss 2 (strong reg) | h384, drop 0.5/0.3, wd 0.1 | 0.6372 | 0.6480 | -0.0108 |
| poss 3 (h384 aug) | h384, aug σ=0.05, drop 0.3/0.2 | 0.6374 | 0.6505 | -0.0131 |
| combo (h128+aug) | h128, aug σ=0.05, drop 0.3/0.2 | 0.6377 | 0.6515 | -0.0138 |
| **plan-022 carry** | LGBM, sample-weight expansion | **0.6528** | — | — |

**핵심 메타 finding**:
- 4 variant 모두 5-fold OOF 0.6372-0.6379 plateau (range 0.0007 noise band 안)
- 1-fold best 가 5-fold OOF 보다 *모두* 0.011-0.014 높음 = **systematic lucky catch** (hit_1cm 기준 best epoch tracking = valid set test 정보 사용한 model selection)
- plan-024 paradigm 의 honest ceiling = **0.6375 ± 0.0004** (plan-022 carry 의 -0.0153 미달)
- → 모든 시도 (hyperparam tuning + capacity reduce + augmentation + strong regularization + combo) 5-fold OOF lift 무의미. paradigm 자체 fundamental limit.

### 5.12 combo (poss 1 + poss 3) — best 1-fold 0.6515 단 5-fold OOF 0.6377 (정직)

새 worktree (`worktree-plan-024-combo`) 에서 시도:

**1-fold (epoch 100, hit_1cm 기준 best epoch tracking)** = best 0.6515 @ epoch 70:
- hidden 128 + input aug σ=0.05×std + constant lr + 1 fold
- 1-fold best of all variants. plan-022 carry -0.0013.

**5-fold OOF (patience 10 best val_loss state)** = **0.6377**:
| fold | hit_1cm | val_loss |
|:--|--:|--:|
| 0 | 0.6421 | 2.5648 |
| 1 | 0.6409 | 2.5587 |
| 2 | 0.6356 | 2.5413 |
| 3 | 0.6386 | 2.5642 |
| 4 | 0.6310 | 2.5398 |
| **OOF concat** | **0.6377** | — |

**1-fold 0.6515 vs 5-fold 0.6377 = -0.0138 gap 의 원인**:
- 1-fold best 은 *hit_1cm 기준 best epoch 추적* (= valid set hit metric 으로 model selection = leakage 형태)
- 5-fold OOF 는 *val_loss 기준 best state* (정직, fold-internal val split 만 사용)
- val_loss best epoch ≠ hit_1cm best epoch (decoupled) → 5-fold 가 honest 추정

**최종 conclusion (5-fold OOF 기준)**:
- v1 default 0.6370 → combo 0.6377 = **+0.0007 lift만** (noise band 0.003 안, 사실상 무의미)
- plan-024 paradigm 의 honest 5-fold OOF ceiling ≈ **0.637-0.638**
- plan-022 LGBM carry 0.6528 미달 -0.0151. **모든 시도 (hyperparam + augmentation + capacity reduce) 5-fold OOF lift 무의미**
- 1-fold best epoch tracking 의 0.6515 는 *fold-specific lucky catch*, paradigm 의 진짜 ceiling 아님

## §1 OOF metric table

| metric | plan-024 | plan-022 winner | Δ | 합격 기준 | 결과 |
|:--|--:|--:|--:|:--|:--:|
| **hit_1cm** | **0.6370** | 0.6528 | **−0.0158** | ≥ 0.6528 (G2), ≥ 0.6628 (G3) | ❌ G2 fail |
| **hit_1.5cm** | **0.8092** | 0.8104 | −0.0012 | ≥ 0.8104 (G3) | ❌ |
| Δ_F0 (1cm) | +0.0050 | +0.0208 | −0.0158 | ≥ +0.005 (G3 partial) | △ |
| Δ_F0 (1.5cm) | +0.0059 | +0.0069 | −0.0010 | ≥ +0.005 | ✓ (단 1cm fail) |
| **gap_ranking** | **0.1934** | (LGBM 미측정) | — | ≤ 0.04 (G3) | ❌ |
| oracle_1cm | 0.7928 | (별 metric) | — | — | informational |
| argmax_hit | 0.5994 | (별 metric) | — | — | informational |
| top1_acc | **0.1227** | 0.1707 | −0.048 | ≥ 0.20 (H1) | ❌ |
| max_class_ratio | 0.1047 | 0.1046 | +0.0001 | < 0.95 | ✓ (collapse X) |
| q_true_max | 0.0974 | 0.0974 | 0.0000 | — | (target carry) |
| dist_match_KL | 0.0031 | 0.0022 | +0.0009 | — | informational |
| soft_CE | 2.566 | 2.535 | +0.031 | ≤ 2.50 (H2) | ❌ |

## §2 per-fold variance

| fold | hit_1cm | time |
|:--|--:|--:|
| 0 | 0.6416 | 34s |
| 1 | 0.6370 | 33s |
| 2 | 0.6366 | 36s |
| 3 | 0.6386 | 27s |
| 4 | 0.6310 | 36s |
| **OOF concat** | **0.6370** | **167s** |

per-fold std ≈ 0.0034 (낮은 variance OK). 단 *모든 fold* 가 plan-022 carry 0.6528 미만 — *systematic* underperformance.

## §3 G-gate 결과

| Gate | 조건 | 측정 | 결과 |
|:--|:--|:--|:--:|
| **G0** | 10/10 pytest pass (4.5s) | 10/10 ✓ | ✓ |
| **G1** | F0 carry + plan-022 carry | F0 0.6320 ✓ / plan-022 0.6528 ✓ | ✓ |
| **G2** | hit_1cm ≥ 0.6528 | **0.6370 ❌** | **fail** (`xattn_no_improvement`) |
| G3 | hit_1cm ≥ 0.6628 AND gap_ranking ≤ 0.04 | 0.6370 ❌ / 0.1934 ❌ | fail |
| **G_final** | §0.5 c1~c11 [DONE] AND 3-file sync AND follow-up 3건 | c12/c13 [SKIPPED] (xattn_no_improvement 분기, §0.5 / §8.3 spec) | **pass band=negative** |

## §4 plan-009 fail 패턴 비교

| metric | plan-009 ranking_loss | plan-024 v1.1-rev2 | plan-024 의 fail 양상 |
|:--|--:|--:|:--|
| oof_soft_hit | 0.6482 | 0.6370 | plan-024 더 fail (−0.0112) |
| top1_ranking_acc | 0.0922 | 0.1227 | 약간 회복 단 plan-022 0.1707 미달 |
| gap_ranking | 0.1080 | 0.1934 | plan-024 더 큰 fail (+0.0854) |

**plan-009 와 같은 axis** (`architecture/loss lever 만 변경 → selector ranking 회수 X`) 의 fail 패턴. plan-009 가 *loss lever* fail, plan-024 가 *architecture + FE max lever* fail 단 *gap_ranking 악화* 가 plan-024 가 더 심함 (1.8×).

## §5 fail 원인 분석 — 7가지 가설

### 5.1 ⚠️ CPU under-converged 가설 — **기각** (v2 ablation 결과)
- 초기 의심: 5-fold OOF 학습 시간 = **167s total** (spec §10 GPU 5-7h 가정의 1/100).
- v2 재학습 (patience 3→999, no early stop 강제): hit_1cm = **0.6370 (v1 과 동일)**, time = 171s.
- → patience 변경 효과 없음. 학습은 *정상 수렴*, under-converged 아님.
- 결론: 학습이 167s 짧은 이유 = *2M params + GRU+attention CPU 가 batch 당 0.24s, 22 epoch × 32 batch ≈ 700 step × 0.24s ≈ 168s* 로 합리적 시간. 학습 정상.

### 5.4 ⚠️ channel dropout over-regularization 가설 — **기각** (v3 ablation 결과)
- 초기 의심: ③ ctx 128D p=0.3 channel drop = 평균 38D drop per batch. dim 폭증의 redundancy 가정이 부정확하면 information drop 손해.
- v3 재학습 (cand_drop_p=0, seq_drop_p=0): hit_1cm = **0.6373 (v1 0.6370 와 ~동일)**.
- → channel dropout off 효과 없음. over-regularization 가설 기각.
- 단 top1_acc: 0.1227 → 0.1273 (+0.0046 mild) — channel drop off 가 ranking 능력 약간 ↑ 단 hit rate 변화 없음.

### 5.9 v6 (LGBM + plan-024 input) — **input lever 가 carry 와 redundant 확정**

사용자 통찰 follow: "plan-004 → plan-024 변경 = 후보 + input 만, regime/corrector 영향 작음 입증". → arch (PB cross-attn) carry vs LGBM 짝짓기 비교 시 **LGBM 이 14 BCC 에서 더 잘함** → plan-024 framework 의 input lever 만 LGBM 환경에서 시도.

| variant | model | input | hit_1cm | Δ vs plan-022 |
|:--|:--|:--|--:|--:|
| plan-022 winner | LGBM | 170D base | 0.6528 | — |
| plan-024 v1~v5 | cross-attn | 250D | 0.6370~0.6375 | **−0.0156** |
| **v6** | **LGBM** | **170D + Multi-window 60D = 230D** | **0.6531** | **+0.0003** |

**Multi-window 60D 추가 효과 = +0.0003** — 거의 무효 (Multi-window 가 plan-024 의 가장 큰 단일 lever).

**메타 finding**:
- plan-024 의 16-lever FE max 가 plan-022 carry 와 **대부분 redundant** (information 흡수 측면)
- 4-way ML expert review 의 expected lift 추정 (Multi-window +0.005~+0.010) → 실제 +0.0003 = **over-estimate**
- 외부 reference (Singer LANL ~1000D, nyanp Optiver) 의 lift 추정이 **muflight 도메인에 transfer X**
- plan-021 carry 의 170D (L1+L2+L4+lgbm_extra) 가 이미 *대부분의 information* 잡음, plan-024 의 Multi-window/STA-LTA/WAP/Pct-rolling/v_autocorr 등이 *trajectory macro stat* 차원에서 redundant

→ **fail 의 진짜 분해** (사용자 통찰 + v6 종합):
1. plan-004 → plan-024 변경 = (a) 후보 27 physics → 14 BCC + (b) input FE max
2. (a) 후보 변경의 cost = plan-004 (PB arch + 27 cand) 0.6624 → plan-022 (LGBM + 14 BCC) 0.6528 의 -0.0096 = **후보 변경 자체 cost** (PB arch 의 LGBM 변경으로 부분 회복)
3. (b) FE max input 의 lift = v6 결과 +0.0003 ≈ **0** (carry 와 redundant)
4. plan-024 cross-attn (PB arch carry + 14 BCC) 의 -0.0156 = **arch-후보 mismatch cost** (cross-attn 이 14 static anchor 환경에서 LGBM 보다 못함)

### 5.10 사용자 axis (post-diagnose) — 3 가능성 ablation (1-fold long-diag)

사용자 통찰 "model overpowered + low-info input → overfit" 확정 후, *overfit 회피 lever* 검증:

| variant | config | best hit_1cm | best ep | params | time | vs plan-022 |
|:--|:--|--:|--:|--:|--:|--:|
| long-diag (carry) | h384 + const lr + ep 100 | 0.6495 | 35 | 2M | 142s | -0.0033 |
| **poss 1** (capacity ↓) | h128 + ep 50 + const lr | 0.6490 | **49 last** | 273K | 36s | -0.0038 |
| poss 2 (강한 reg) | h384 + drop 0.5/0.3 + wd 0.1 + ep 100 | 0.6480 | 30 | 2M | 146s | -0.0048 |
| 🏆 **poss 3** (data aug) | h384 + input Gaussian noise σ=0.05×std + ep 100 | **0.6505** | **30** | 2M | 154s | **-0.0023** |

**finding**:
- (a) **poss 3 (input augmentation)** = 가장 효과적. v1 0.6370 → 0.6505 = **+0.0135 lift**, plan-022 carry 의 -0.0023 까지 도달 (약 7× 가까워짐 vs v1 default 의 -0.0158).
- (b) **poss 1 (hidden ↓)** = epoch 49 last 가 best — *더 학습 시 ↑ 가능성*. 단 capacity 부족으로 absolute ceiling 작음.
- (c) **poss 2 (강한 reg)** = lift 없음. drop 0.5 가 *informative channel* 도 drop → 학습 signal 손해.
- (d) **모두 plan-022 carry (0.6528) 미달** — 가장 가까운 poss 3 도 -0.0023.
- (e) **미시도 sweet spot**: poss 1 + poss 3 combo (hidden 128 + aug + epoch 100) — overfit 더 늦게 시작 + best epoch tracking → plan-022 동등 또는 ↑ 가능성.

**plan-024 paradigm 의 새 ceiling 추정**: 1-fold best 0.6505 → 5-fold OOF best 평균 ≈ 0.645-0.652 추정 (fold variance 0.003-0.005). plan-022 carry 0.6528 *동등 가능성* 단 *유의미 lift X*.

### 5.1+5.4+5.6+5.8+5.9+5.10 종합 — **6 variant + 3 가능성 + long-diag = 10 시도 + ablation finding**

| variant | hit_1cm | top1_acc | gap_ranking | soft_CE | time | 핵심 변경 |
|:--|--:|--:|--:|--:|--:|:--|
| v1 spec default | 0.6370 | 0.1227 | 0.1934 | 2.566 | 167s | hidden=384, drop 0.3/0.2, lr 7e-4, wd 0.02 |
| v2 | 0.6370 | 0.1227 | 0.1934 | 2.566 | 171s | + patience 999 (under-converged 가설 기각) |
| v3 | 0.6373 | 0.1273 | 0.1950 | 2.568 | 170s | + channel drop 0/0 (over-reg 가설 기각) |
| v4 PB-default | 0.6375 | 0.1355 | 0.1919 | 2.562 | 78s | hidden=128, lr 1e-3, wd 0.01, drop 0 (PB carry) |
| **v5 A7 embed** | **0.6372** | **0.1389** | **0.1904** | **2.558** | **172s** | + nn.Parameter(14, 8) learnable anchor embedding |

### 5.8 ⚠️ A7 Learnable anchor embedding 가설 — **기각 (v5 결과)**
- 진단 결과 (diagnose_training.json) 의 가장 promising lever 였음 — anchor 14 사이 *차별화 capacity 부족* (cand_dim 150D 중 anchor-discriminating 22D 만, 128D 는 broadcast) 의 직접 fix.
- 식: `nn.Parameter(14, 8) init=0.02`, fwd 후 cand_feat 에 broadcast concat. 14 × 8 = 112 학습 params 추가.
- 결과: hit_1cm 0.6372 (v1 0.6370 vs +0.0002 미미). top1_acc 0.1227 → 0.1389 (+0.0162 mild), soft_CE 2.566 → 2.558 (-0.008 mild). **hit rate 회복 X**.
- → A7 가설 *부분 기각* (top1_acc/soft_CE 약간 ↑ 단 hit_1cm fundamental cap 의 진짜 lever 아님). architecture limit 확정.

모든 variant 가 0.6370~0.6375 = **systematic ~0.6373 ± 0.0003** OOF hit_1cm. plan-022 LGBM 0.6528 보다 **−0.0153 ~ −0.0158** below. *재현 가능* 한 systematic underperformance.

**결론**:
- (a) under-converged 가설 (5.1) **기각** — patience 변경 효과 0.
- (b) over-regularization 가설 (5.4) **기각** — channel drop off 효과 0.
- (c) hyperparam tuning 가설 (5.6) **부분 기각** — PB default carry 도 fail.
- (d) anchor identity capacity 부족 가설 (5.8) **부분 기각** — A7 learnable embedding 도 hit_1cm 변화 X.
- (e) **architecture + FE max lever 자체 inherent fail (5 variant 확정)** — plan-024 의 cross-attention + 14 BCC anchor classification 의 *fundamental ceiling*. plan-009 ranking_loss fail 패턴 의 *architecture-level 재발견*.
- (f) **진단의 deeper finding (diagnose_training.json)**: val_loss 2.5736 ≈ log(14) − 0.066 = uniform softmax 에서 *약간만* 벗어남. 즉 model 이 본질적으로 **near-uniform 출력** 으로 plateau → q_pred · anchors ≈ 0 → pred ≈ pred_F0 → hit ≈ F0 baseline + ε.
- (g) **진짜 lever (plan-025/026/027 영역)**:
  - (i) **LGBM sample-weight expansion mimic** — cross-attn 의 N=10k 학습 vs LGBM 의 N×K=140k effective row 의 14× gap 직접 fix. pointwise (sample, anchor) row expansion 또는 K independent forward.
  - (ii) **anchor pool 변경** — 14 static BCC → sample-conditional dynamic anchor (plan-004 패턴, plan-026 radius 확장 + multi-shell).
  - (iii) **plan-022 LGBM + plan-024 cross-attn ensemble** — 두 paradigm 의 cancel 안 되는 영역 (plan-027). 단 plan-024 base hit 0.6372 + LGBM 0.6528 의 ensemble = 0.65~0.66 가 한계 추정.
  - (iv) **다른 architecture paradigm** — Trajectron-CLIP / KNN retrieval pool / Variance-aware MDN (ideas.md ★★★ Tier, plan-028).

### 5.2 dim 폭증 (caveat #11)
- cand 14×150 = 2100 element + seq 7×95 = 665 element = sample 당 ~2800 element. N=10k → ratio ~3.6 sample/dim.
- LGBM 170D 대비 16× 빡빡. mitigation (dropout 0.10/0.15 + channel drop 0.3/0.2 + weight_decay 0.02 + scale clamp) 부족.

### 5.3 GRU + Cross-attention 의 short-seq 약함 (T=7)
- PB framework `CandidateAttentionGRUSelector` 의 원래 task = T=6 step (plan-004 make_seq_features). plan-024 T=7 도 유사.
- 단 plan-004 의 LB 0.6822 은 27 physics candidate (sample-conditional rich spec) + regime bias 와 함께. plan-024 의 14 BCC (정적 sphere point) + 정적 spec 9D 만 → query 의 sample-conditionality 가 ③ ctx broadcast 와 ④ interactions 만으로 약함.

### 5.4 channel dropout 의 over-regularization
- ③ ctx 128D p=0.3 channel drop = 평균 38D drop per batch. dim 폭증의 *redundancy* 가정이 정확하지 않으면 information drop 손해.

### 5.5 Tier S/A 16 lever 중복 학습 (caveat #16)
- 17 → 16 lever (rev2 의 S4 Fourier PE 제거) 동시 추가 → bottleneck 분해 불가. plan-025 ablation 없이는 어느 lever 가 fail trigger 인지 미분리.

### 5.6 hyperparam tuning 부재 (caveat #2)
- single config 고정 (hidden=384, lr=7e-4, wd=0.02, dropout 0.10/0.15). PB framework default (hidden=128, dropout 0.08) 와의 차이가 fail 원인일 가능성.

### 5.7 sign convention 변경 효과 미검증
- L2 의 sign 통일 (pred-actual → actual-pred) 이 plan-022 LGBM 의 학습 패턴과 mismatch 가능. 단 plan-022 의 selector_only_model.build_soft_label_with_tau 와 정합이라 가설 약함.

## §6 measurable architecture-extractable headroom

| metric | plan-024 측정 | plan-008 carry | meaning |
|:--|--:|--:|:--|
| **oracle_1cm** | **0.7928** | 0.7562 (27 cand) | 14-anchor framework 의 oracle 측정 — 27 cand 보다 높음 |
| **argmax_hit** | 0.5994 | (plan-008 0.7046) | 단순 top-1 픽 의 hit rate |
| **gap_ranking** | **0.1934** | 0.0516 (base) | selector 가 *놓친* 영역 |

**14 anchor oracle 0.7928** (= plan-024 의 *상한*) — 선언:
- plan-024 selector 가 oracle 의 80.4% 만 회수 (0.6370 / 0.7928).
- 만약 gap_ranking 회수 100% 시 OOF hit_1cm = 0.7928 (= LB 0.80+ 가능).
- 0.7928 oracle 은 *14 anchor + corrector-free* 의 본질적 ceiling — plan-026 의 anchor radius 확장 시 ceiling 자체 ↑.

## §7 LB-OOF gap 비교 (informational, LB 미회수)

- plan-004 carry: OOF 0.6624 → LB 0.6806 = **+0.0182**.
- plan-024 OOF 0.6370 (gap +0.0182 가정 시 LB ≈ 0.655) — plan-004 LB 0.6806 floor 미달 추정. dacon-submit skip 정합.

## §8 ablation slot (plan-025 영역)

plan-024 의 16 lever 동시 추가 → bottleneck 분해 plan-025 ablation 으로 위임:
- (A) Tier S 부분 ablation: jerk / ω / saccade / sinusoidal-PE 개별 제거
- (B) Tier A 부분 ablation: Multi-window 60D / STA-LTA / WAP / wingbeat 개별 제거
- (C) regularizer ablation: channel dropout off, learnable scale off
- (D) hidden=384 → 128 (PB default) 또는 256

## §9 comparison table

| plan | model | input dim | OOF hit_1cm | LB | gap_ranking |
|:--|:--|:--|--:|:--|--:|
| plan-022 winner | LGBM (sample-weight expansion) | 170D | 0.6528 | (carry) | (LGBM 미측정) |
| plan-009 (ranking_loss) | Attn-GRU + listwise loss | (PB 170D) | 0.6482 | (carry) | 0.108 |
| **plan-024** | **Cross-attention GRU + FE 245D** | **150+95=245D** | **0.6370** | **null (skip)** | **0.1934** |
| plan-004 PB | Attn-GRU + 27 cand + 18×27 regime | (PB) | 0.6624 | 0.6806 | (carry) |

## §10 follow-up plan 후보 (자동 박제)

### plan-025 — Ablation + 강화 form (가장 시급)
- (A) plan-024 의 16 lever ablation — 항목별 contribution 분해
- (B) **CPU/GPU compute 재검토** (under-converged 의심 fix)
- (C) hyperparam mini-sweep (hidden 128/256/384, lr 5e-4/7e-4/1e-3, dropout off/0.05/0.10, epoch 22 강제 + no early stop variant)
- (D) ideas.md priority 5 (A1 Multi-window vs B3 STA/LTA 분리 측정)
- (E) **path_signature_L2** (A4 v1.1 제외) + **Learnable anchor embedding** (A7) head-to-head

### plan-026 — Anchor radius 확장 + F0 baseline ML 화
- (i) anchor radius 0.5cm → 0.7~1.0cm (geometric hard cap 완화)
- (ii) F0 baseline ML 화 (forward bias 완화)
- (iii) 14 anchor oracle 0.7928 → 0.85+ 추정 (radius 확장 시)

### plan-027 — Ensemble
- plan-022 LGBM + (plan-024 또는 plan-025 의 best variant) 평균. plan-024 v1.1-rev2 단독 fail 단 *다른 inductive bias 의 cancel 안 되는 영역* 위 partial lift 가능성.

### plan-028 (가칭, 신규) — paradigm shift
- ideas.md ★★★ Tier (Trajectron CLIP / KNN pool / MDN). plan-024 fail 이후 architecture lever 더 깊은 검토.

## §11 paths & artifacts

- `analysis/plan-024/results_xattn.json` — G2 OOF metric (167s, CPU)
- `analysis/plan-024/baseline_carry.json` — G1 carry (F0 + plan-022 winner)
- `analysis/plan-024/multiwindow_trim.json` — Multi-window 144→60 trim list
- `analysis/plan-024/g2_run.log` — G2 학습 log (per-fold time + hit rate)
- `plans/plan-024-cross-attention-anchor-vocab.md` v1.1-rev2 — spec 본문 (commit chain c1~c9 [DONE] + c10 [DONE G2 FAIL] + c11~c13 [SKIPPED])
- `plans/plan-024-cross-attention-anchor-vocab.results.md` — pair (status=all_complete, band=negative)

## §12 commit chain final state (§0.5 sync)

| commit | type | state |
|:--|:--|:--|
| c1 | spec v1 | [DONE bd1c4cd] |
| c1.5 ~ c1.5g | spec v1.1 patch + plan-review iter 1-5 + rev2 | [DONE 205a985] |
| c2~c8 | module code | [DONE 6ecbcdb, 0254da3, 915dd26, a915f78] |
| c9 | G1 reproduce | [DONE 52a5462] |
| G1 | gate (F0 + plan-022 carry pass) | ✓ |
| c10 | G2 OOF | **[DONE G2 FAIL, xattn_no_improvement]** |
| G2 | gate (hit_1cm ≥ 0.6528) | ❌ |
| c11 | results.md (본 파일) | [DONE 본 commit] |
| G3 | gate | ❌ (G2 fail → 분기 X) |
| c12 | submission | **[SKIPPED, xattn_no_improvement, §0.5/§8.3 분기]** |
| c13 | dacon-submit | **[SKIPPED]** |
| c14 | 3-file frontmatter sync (band=negative, lb=null) | [DONE 본 commit] |
| G_final | gate (band=negative pass, §8.3 분기) | ✓ |
