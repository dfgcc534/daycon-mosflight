---
plan_id: 024
version: 1.1-rev2
finished_at: 2026-05-21 (Asia/Seoul)
status: all_complete
band: negative
best_hit_1cm: 0.6370
best_hit_1.5cm: 0.8092
best_delta_1cm: +0.0050
best_delta_1.5cm: +0.0059
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

plan-024 의 **architecture lever 실패**. cross-attention GRU + 16-lever FE max input (cand 150D + seq 95D + hidden 384 + per-channel scale + channel dropout) 가 plan-022 LGBM winner (hit_1cm=0.6528) 보다 **−0.0158 below** (OOF hit_1cm=0.6370). plan-009 ranking_loss G1 fail 패턴 재현 — `gap_ranking` 0.1934 (plan-008 base 0.0516 의 3.7× 악화). `xattn_no_improvement` severe → c12/c13 [SKIPPED] → G_final band=negative.

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

### 5.1+5.4 종합 — **architecture + FE max lever 자체의 inherent fail**
v1/v2/v3 모두 ~0.6370 ± 0.0003 = **systematic underperformance vs plan-022 LGBM 0.6528**. *재현 가능* 한 −0.0158 gap. plan-009 의 listwise loss fail 패턴 (architecture lever 자체가 LGBM 보다 약함) 의 *재발견*.

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
