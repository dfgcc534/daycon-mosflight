---
plan_id: 024
version: 1.1-rev2
date: 2026-05-21 (Asia/Seoul)
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
xattn_no_improvement: true
based_on:
  - 022 (winner cell A6_bcc14_tau001: K=14 BCC anchor, τ_cls=0.001, OOF hit_1cm=0.6528 / hit_1.5cm=0.8104, Δ_1cm=+0.0208, Δ_1.5cm=+0.0071, 14-anchor codebook + LGBM sample-weight expansion selector. selector-only paradigm = corrector-free `pred_world = R_wfn · Σ_k q_k · a_k + pred_F0_world`.)
  - 021 (input augment 170D LGBM: L1=Frenet (p,v,a) 11step×9, L2=F0 residual 7step×3, L4=soft hit 7step×2, lgbm_extra 36D = macro_stat 9 + EWMA(p,v,a) 27. build_input_common / build_input_lgbm_extra / build_soft_label / build_frenet_basis_3d / to_frenet)
  - 004 (PB framework CandidateAttentionGRUSelector arch: `forward(seq, cand_feat) → score`. GRU(seq_dim, hidden, num_layers=2, dropout=0.08) → out + h_final, MLP(cand_feat) → query, attention out·query^T/sqrt(hidden), event_ctx = attn·out, head = MLP(concat(final_ctx, event_ctx, cand_feat)) → score. `make_seq_features` 9D/step × 6 step, `make_candidate_features` 32D = par/perp/dist 3 + spec 9 + ctx 9 + interactions 4)
  - 009 (ranking_loss G1 fail finding: oof_soft_hit=0.6482 (carry 0.65 미달), top1_ranking_acc=0.092 (base 0.126 보다 worse), gap_ranking=0.108 (base 0.0516 의 2배). listwise/pairwise loss tuning 만으로는 selector ranking 회수 불가 — *architecture* lever 가 진짜 lever 라는 직접 증거. plan-024 가 정확히 architecture 측 lever.)
  - 008 (gap_ranking 측정 박제: base 27 = 0.0516, extended 25 = 0.1119. main_bottleneck=ranking. plan-009 의 oracle_1cm=0.7562 — corrector-free 위 oracle gap +0.10 잔존.)
inspired_by:
  - 사용자 narrative (2026-05-21): "plan-022 기준으로 FE 최대한 + cross-attention 구조로 구현. LB 최댓값 도전. 다음 lever (radius/F0/ensemble) 는 후일."
  - sub-agent 4-way audit (2026-05-21, v1): Q1+Q5 F0 residual / torsion (silent sign bug + τ_past 분리), Q2+Q3 world frame z-axis (꺾쇠 = R_wfn z-row), Q4 plan-004/022 input audit (macro_stat 9D / L1 EWMA 27D / turn_cos·curvature·direction / regime bin 누락), Q6 ideas.md 통합 (Stage 3 out-of-scope, plan-025 후보).
  - **4-way ML expert review (2026-05-21, v1.1 patch)** — 4 agent (Trajectory prediction / Kaggle-tabular / Physics-informed / Cross-attention-set-prediction) 외부 reference 기반 review. 핵심 발견:
    1. **muflight = mosquito (모기) 확정** (Agent 3, 6 단서): DACON 236716 = "모기 비행 궤적 예측 AI 경진대회", 25Hz × wingbeat 600-800Hz 의도된 aliasing, anchor radius 5mm = body size, Lévy flight 용어. → FE 설계는 mosquito biomechanics 우선.
    2. **cross-confirmed 추가** (≥2 agent): jerk Frenet 3D (A1+A3), saccade binary 1-2D (A1+A3), F3 anchor-projection 14D redundancy 제거 (A1+A2), macro_stat 9D underrepresentation (A2 단독 strong, LANL ~1000D 대비 ~10%).
    3. **single-strong 추가**: angular velocity ω Frenet 3D (A3 Top-1, drone SE(3) 절반 missing), ~~Anchor coord Fourier PE 12D~~ (**v1.1-rev2 사용자 지적으로 제거** — 14 discrete BCC token 에 continuous-PE 부적합, plan-025 lever 측정 오염 회피) + Sinusoidal time PE 4D (A4 seq 측, Vaswani PE 표준, 7-step continuous position 이라 적용 valid), BCC adjacency neighbor pool 2D (A4, Set Transformer ISAB 등가).
    4. **사용자 결정 (옵션 1)**: Tier S (5) + Tier A 옵션 B (10 중 path_signature_L2 + Learnable anchor embedding 제외 = 8) + redundancy 제거 (F3 14D + straightness 1D + axis×forward 1D) + per-channel learnable scale + channel dropout (LGBM feature_fraction NN 등가) + hidden 256 → 384.
    5. **외부 reference URL**: Frenet-Serret tracking 2025 (arxiv 2501.04273), Path Signatures 2025 (arxiv 2506.01815), Trajectron++ (arxiv 2001.03093), PBP (arxiv 2309.03750), Singer LANL writeup (kaggle.com/c/LANL-Earthquake-Prediction/discussion/94390), nyanp Optiver writeup (kaggle.com/c/optiver-realized-volatility-prediction/discussion/274970), Anchor-DETR (Wang 2022), DAB-DETR (Liu 2022), Tancik Fourier features 2020, Set Transformer (Lee 2019), Mellinger-Kumar SE(3) 2011, Aedes mosquito flight (arxiv 1205.5260), mosquito wingbeat (PMC8113239), Lévy flight (PLOS CompBiol 2012, PMC4345481).
code_reuse:
  - module: analysis/plan-021/build_input.py
    symbols: [build_frenet_basis_3d, to_frenet, build_input_common, build_input_lgbm_extra, _macro_stat_9d, _ewma_last]
    reason: L1/L2/L4 + macro_stat 9D + EWMA 27D pipeline 그대로 carry. plan-024 의 seq input + cand_feat ③ ctx broadcast 의 source. **단 sign convention 통일** 위해 L2 의 `pred_t - actual_t` → plan-024 사용 시 negate (= `actual - pred`).
  - module: analysis/plan-022/anchors.py
    symbols: [ANCHORS_A6]
    reason: 14-anchor codebook (BCC: 6 axis ±0.005m + 8 corner ±0.005/√3 m). plan-022 winner cell 의 anchor.
  - module: analysis/plan-022/selector_only_model.py
    symbols: [build_soft_label_with_tau]
    reason: soft label 생성 (output target). plan-024 의 F 묶음 (per past step anchor-vocab) 도 동일 식 (per-step residual 입력) 으로 재호출 — **sign 통일 + τ_past 별도 인자**.
  - module: analysis/plan-020/baseline_f0.py
    symbols: [f0_baseline, D1, PAR, PERP, R_HIT, R_HIT_LOOSE]
    reason: F0 baseline (paired Δ reference) + 1cm/1.5cm hit threshold.
  - module: src/pb_0_6822/selector.py
    symbols: [CandidateAttentionGRUSelector, stable_fold_id, regimes (assign_regimes 등), turn_context_features, motion_terms]
    reason: cross-attention selector arch + 5-fold split + regime 18-bin (cand_feat ③ ctx broadcast).
  - module: src/io.py
    symbols: [load_all_samples, load_labels]
    reason: data loader.
followed_by:
  - plan-025 (가칭): plan-024 G_final 통과 시 ideas.md priority 5 (A1 Multi-window stat / A6 WAP composite / B3 STA/LTA / Multi-Parse Input / B2 Pct-of-rolling-std) 통합 — Stage 3 lever 의 ablation.
  - plan-026 (가칭): anchor radius 확장 (Plan B #3 lever, geometric hard cap) + F0 baseline ML 화 (#4 lever, systematic forward bias) — plan-024 의 cross-attn 위.
  - plan-027 (가칭): plan-022 LGBM + plan-024 cross-attn ensemble (Plan C #5 lever).
scope: plan-022 winner cell A6_bcc14 + τ_cls=0.001 carry 위에서 selector architecture 만 LGBM → CandidateAttentionGRUSelector (cross-attention GRU) 로 교체. FE max input: cand_feat 62D + seq 89D (Stage 1 누락 audit + Stage 2 강화 form). single-config 5-fold OOF + LB 회수. anchor layout sweep / radius 변경 / F0 baseline 변경 / ensemble / ideas.md Stage 3 = out-of-scope.
exp_ids:
  - Z024_xattn_anchor_vocab
lb_score: null
band: null
---

# plan-024 v1 — Cross-attention Anchor-Vocabulary Selector (FE max + PB CandidateAttentionGRUSelector arch on plan-022 winner)

## §0. 한 줄 목적

> **plan-022 winner cell A6_bcc14_tau001 (corrector-free 14-anchor LGBM, OOF hit_1cm=0.6528 / Δ_F0=+0.0208) 위 selector architecture + input FE pipeline 동시 교체 (16 lever, single evaluation run, plan-025 ablation 분리)** — PB framework `CandidateAttentionGRUSelector` (hidden=384)** — anchor coord 가 attention query (cand_feat **150D**) 로 직접 입력 + past seq (95D × 7step) 가 **anchor-vocabulary 로 re-encoding** + **4-way ML expert review (v1.1 patch) 의 cross-confirmed/single-strong finding 모두 박제** — (a) plan-004/022 audit 누락 (macro_stat / L1 EWMA / turn·curv·direction / regime / F0-pred coord / residual angle), (b) world-z 정렬 (t̂_z, b̂_z, Vz, anchor Δz_world), (c) Frenet torsion τ, (d) residual 강화 form (F2 log-magnitude, sign 통일, τ_past 별도), (e) **v1.1 Tier S** (jerk Frenet, angular velocity ω, saccade binary, ~~Anchor coord Fourier PE — rev2 제거~~, Sinusoidal time PE), (f) **v1.1 Tier A 옵션 B** (STA/LTA ratio, Multi-window stat grid, BCC adjacency, WAP composite, wingbeat-jitter, f0_confidence, anchor-saliency, helicity, Pct-rolling+Peak, v_autocorr — A4 path_sig + A7 learnable embed 제외), (g) **redundancy 제거** (F3 anchor-projection 14D + macro_stat straightness 1D + axis×forward 1D = 16D), (h) **per-channel learnable scale + channel dropout** (LGBM feature_fraction NN 등가, ③ ctx broadcast 128D 만 적용). **anchor 개수·anchor radius·τ_cls(output)·F0 baseline·5-fold split = 모두 plan-022 carry**. *single 변수 = architecture + input FE*. single-config 5-fold OOF + LB 회수.
>
> **합격 기준**: OOF hit_1cm ≥ 0.6628 (= plan-022 winner +0.01) **AND** OOF hit_1.5cm ≥ plan-022 winner (0.8104) **AND** LB ≥ plan-022 carry LB (미박제 시 plan-004 LB 0.6806 floor).
>
> **v1.1 expected lift envelope** (Tier S + Tier A 옵션 B + weighting + channel drop):
>
> *산정식*: `expected_lift = Σ_lever lift_lever × correlation_discount`. correlation_discount = 0.6 (보수, 17 lever 가 부분 상관 가정). individual lever expected lift (mid-band):
> - S1 jerk Frenet: +0.004 · S2 ω Frenet: +0.003 · S3 saccade: +0.003 · ~~S4 Fourier PE~~ **제거 (rev2)** · S5 sinusoidal time PE: +0.003 (Tier S 합 ~+0.013 → × 0.6 = +0.008)
> - A1 STA/LTA: +0.002 · A2 Multi-window: +0.008 · A3 BCC adj: +0.004 · A5 WAP: +0.002 · A6 wingbeat: +0.002 · A8 f0_conf: +0.003 · A9 saliency: +0.003 · A10 Pct-roll+Peak: +0.003 · A11 helicity: +0.001 · A12 autocorr: +0.002 (Tier A 합 ~+0.030 → × 0.6 = +0.018)
> - Redundancy 제거 (F3, straightness, axis×forward): regularization 효과 ~+0.002
> - per-channel learnable scale + channel dropout: regularization gain ~+0.005
> - **mid 합산**: +0.010 + +0.018 + +0.002 + +0.005 ≈ **+0.035** → 0.6528 + 0.035 ≈ **0.688**
> - 비관 (discount 0.4): 0.6528 + 0.020 = **0.6728** (G3 +0.01 통과)
> - 중간 (discount 0.6): 0.6528 + 0.030~0.035 = **0.683~0.688** (LB 0.6806 침투)
> - 낙관 (discount 0.8): 0.6528 + 0.050 = **0.7028** (LB 0.7 가시권)
>
> **out-of-scope**: anchor layout sweep / anchor radius 변경 / τ_cls(output) 변경 / F0 baseline 변경 / corrector reg head 재투입 / ensemble / **path_signature_L2 (A4)** = plan-025 후보 (signatory 의존성) / **Anchor coord Fourier PE (S4) + Learnable anchor embedding (A7)** = plan-025 후보 (둘 다 *anchor identity representation* axis — 14 discrete token 위 fixed Fourier basis vs learnable embedding 의 head-to-head 비교 가치, plan-024 에서 분리 측정 위해 함께 제외) / ideas.md priority 5 추가 = plan-025 후보 / hyperparam sweep (single config 고정).

---

## §0.5 Quick Reference (autonomous loop 매 turn 읽는 section)

> *plan-024 의 commit chain 은 single config 라 plan-022 의 21-cell sweep 대비 단순. STAGE 0 인프라 → STAGE 1 input + label sanity → STAGE 2 model 학습 + 5-fold OOF → STAGE 3 분석 → STAGE 4 LB.*

### 합격 기준 (G-gate sequence)

- **G0**: 5 module (anchor_vocab / seq_builder / cand_builder / model_runner / training) import + smoke + tests green. plan-021 build_input.py + plan-022 anchors / soft_label / PB selector.py import 정상. 위반 시 `infra_drift` severe.
- **G1**: F0 baseline 5-fold concat OOF + plan-022 winner reproduce — F0 hit@1cm ∈ [0.6315, 0.6325] (carry **F0_hit_1cm = 0.6320**, plan-020 `analysis/plan-020/results.md`, plan-022 baseline_carry.json carry) AND hit@1.5cm ∈ [0.8028, 0.8038] (carry **F0_hit_1.5cm = 0.8033**) + plan-022 A6_bcc14_tau001 reproduce: OOF hit_1cm ∈ [0.6520, 0.6536] (carry **0.6528**) AND hit_1.5cm ∈ [0.8096, 0.8112] (carry **0.8104**). 위반 시 `f0_reproduce_drift` / `plan022_reproduce_drift` severe.
- **G2**: cross-attention selector 5-fold OOF 완료 — OOF metric finite + `max_class_ratio < 0.95` + **OOF hit_1cm ≥ 0.6528** (plan-022 winner 최소 동등). 미달 시 `xattn_no_improvement` severe (architecture lever 실패 → plan 종료, decision-note 박제 후 G_final 진입 — submission 생성 안 함).
- **G3 (lift level)**: OOF hit_1cm ≥ **0.6628** (= plan-022 winner +0.01) AND hit_1.5cm ≥ **0.8104** (= plan-022 winner 최소 동등) AND `gap_ranking` ≤ 0.04 (plan-008 base 0.0516 의 절반). 부분 미달 = `xattn_partial_pass` warn (G_final 진입 단 LB 회수만, follow-up plan 강화 axis 박제).
- **G_final**: dacon-submit skill 자율 호출 + 3-file frontmatter sync (status=all_complete + band + best_metric + lb_score) + LB ≥ plan-022 carry LB (미박제 시 plan-004 LB 0.6806 floor). LB 미달 시 `lb_below_floor` warn (severe 아님, results.md 박제 + follow-up axis 박제).

### G-gates

- G0: STAGE 0 인프라 [TODO]
- G1: STAGE 1 F0 + plan-022 reproduce [TODO]
- G2: STAGE 2 cross-attention 5-fold OOF (최소 동등성) [TODO]
- G3: STAGE 3 lift + gap_ranking [TODO]
- G_final: STAGE 4 LB 회수 + 3-file sync [TODO]

### Commit chain (next-up)

| # | type | spec section | status |
|---|---|---|---|
| c1 | docs | `plans/plan-024-cross-attention-anchor-vocab.md` **v1 작성** (initial spec, 4-way sub-agent audit 통합) | [DONE — bd1c4cd] |
| c1.5 | docs | **v1.1 patch** — 4-way ML expert review 결과 반영 (Tier S + Tier A 옵션 B + per-channel weighting + channel dropout + hidden 256→384 + redundancy 제거). §0/§0.5/§4 본문 update + §13 changelog 추가 | [TODO] |
| c2 | code | `analysis/plan-024/anchor_vocab.py` — F 묶음 식 (per past step residual → 14-anchor softmax + magnitude + top1 one-hot + F2 log-magnitude). **sign 통일** (`actual − pred`) + **τ_past 별도 인자**. *F3 anchor-projection 제거* (v1.1, A1+A2 redundancy). spec @ §4.2 | [TODO] |
| c3 | code | `analysis/plan-024/seq_builder.py` — seq **95D** per step (length=7, t=4..10 alignment). ABC + Vz + D residual + angle + pred_F0 coord + E + F/G/H + F2 + I + Δ²res + J + K time + L entropy + M 2nd-best + O speed + turn/curv/direction + torsion τ + **jerk Frenet (S1) + ω Frenet (S2) + saccade binary (S3) + sinusoidal time PE (S5) + anchor-saliency (A9) + helicity (A11) + WAP per-step (A5) + f0_conf per-step (A8)**. spec @ §4.3 | [TODO] |
| c4 | code | `analysis/plan-024/cand_builder.py` — cand_feat **150D** per anchor (14 row): ① 3 + ② 9 (~~Fourier PE 12 제거, rev2~~) + ③ 128 (broadcast: 12 base + STA/LTA 3 + Multi-window 60 + WAP sample-level 5 + wingbeat 3 + f0_conf sample-level 2 + Pct-rolling+Peak 12 + v_autocorr 3 + macro_stat 8(-straightness) + Bz/Tz 2 + regime 18) + ④ 10 (+BCC adjacency 2, -axis×forward 1). spec @ §4.4 | [TODO] |
| c5 | code | `analysis/plan-024/torsion_calc.py` — Frenet torsion τ scalar per step (numerical-safe: collinear mask + sign-flip alignment + ‖v‖ clamp). spec @ §4.5 | [TODO] |
| c5.5 | code | `analysis/plan-024/quantile_carry.py` — train fold quantile 박제 (saccade ω threshold p90 / Peak jerk threshold p90 / Lévy tail threshold). fold-leakage 차단용. spec @ §4.8 | [TODO] |
| c5.7 | code | `analysis/plan-024/feature_weighted_dropout.py` — **per-channel learnable scale** (`cand_scale` 162 + `seq_scale` 95, init=1.0) + **channel dropout** (cand ③ ctx broadcast 영역 128D 만, p=0.3; seq 의 redundant 영역 EWMA J 9D + Multi-window broadcast slice, p=0.2). 보호 영역 (①②④ + seq kinematic) drop X. spec @ §4.6 | [TODO] |
| c5.8 | code | `analysis/plan-024/multiwindow_trim_build.py` — Multi-window stat grid 144→60 trim list 생성 (§4.4.1 deterministic correlation-based greedy column drop). 출력 `multiwindow_trim.json`. spec @ §4.4.1 | [TODO] |
| c6 | code | `analysis/plan-024/model.py` — `CrossAttentionAnchorSelector` (PB framework `CandidateAttentionGRUSelector` 그대로 import + thin wrapper for K=14, cand_dim=**162**, seq_dim=**95**, hidden=**384**) + FeatureWeightedDropout module 의 forward 맨 처음 호출. spec @ §4.6 | [TODO] |
| c7 | code | `analysis/plan-024/run_oof.py` — 5-fold OOF runner (stable_fold_id MD5 carry). data load → quantile_carry build → seq/cand build → fit (2 layer GRU dropout **0.10** + AdamW lr **7e-4** weight_decay **0.02** cosine + epochs pre=**12** fine=**10** + batch 256 + Head MLP dropout **0.15**) → predict → metrics. spec @ §6 | [TODO] |
| c8 | test | `tests/test_plan024_smoke.py` — **10 pytest**: anchor_vocab shape + sign sanity (axis 대칭 invariance) + seq **95D** shape per step + cand **150D** shape per anchor + torsion mask + quantile_carry fold-leakage 차단 + FeatureWeightedDropout weight + channel mask 보호 영역 + model forward smoke (b=4, K=14, T=7) + 1-fold 1-epoch fit finite + G1 reproduce sanity | [TODO] |
| G0 | gate | smoke + tests green (10/10 pytest pass) | [TODO] |
| c9 | exp G1 | F0 + plan-022 winner reproduce → `analysis/plan-024/baseline_carry.json` 박제. F0 hit@{1, 1.5}cm ± 0.0005 / plan-022 A6_bcc14_τ001 hit ± 0.0008 | [TODO] |
| G1 | gate | F0 carry ✓ AND plan-022 reproduce ✓ | [TODO] |
| c10 | exp G2 | cross-attention 5-fold OOF (v1.1 full config **150D cand + 95D seq + hidden=384** + per-channel scale + channel dropout) + `analysis/plan-024/results_xattn.json` (hit_1cm, hit_1.5cm, Δ vs F0, per-fold metric, max_class_ratio, q_true_max, KL, top1_acc, soft_CE, gap_ranking, learnable scale stat). 학습 시간 추정 **~5~7시간** (GPU, 5-fold × 22 epoch, input dim 3배 ↑). | [TODO] |
| G2 | gate | OOF metric finite ✓ + max_class_ratio < 0.95 ✓ + hit_1cm ≥ 0.6528 ✓ | [TODO] |
| c11 | analysis G3 | lift 분석 + gap_ranking 분해 + per-fold variance + paradigm finding → `analysis/plan-024/results.md` (11 항목, plan-022 results.md 형식 carry) + `analysis/plan-024/per_anchor_dist.json` (cf. plan-022 의 per-anchor 분포) | [TODO] |
| G3 | gate | hit_1cm ≥ 0.6628 ✓ AND hit_1.5cm ≥ 0.8104 ✓ AND gap_ranking ≤ 0.04 ✓ (or `xattn_partial_pass` warn 박제) | [TODO] |
| c12 | submission | `analysis/plan-024/submission.csv` 생성 (test set inference, OOF 모델 5-fold 평균 prediction → corrector-free `Σ q · a`). schema = sample_submission.csv 와 정합. | [TODO] |
| c13 | sub-lb | **`dacon-submit` skill 자율 호출** + `analysis/plan-024/lb_log.md` 박제 + **3 파일 frontmatter `lb_score` 동시 갱신** (`plans/plan-024-*.md` top-level + `plans/plan-024-*.results.md` + `analysis/plan-024/results.md`). spec @ §9 + §0.5 AND 조건. | [TODO] |
| c14 | docs | 3-file frontmatter sync (status=all_complete, band, best_metric, lb_score) + follow-up plan 후보 박제 (plan-025 ideas.md priority 5 / plan-026 radius+F0 / plan-027 ensemble) | [TODO] |
| G_final | gate | LB 박제 ✓ + 3-file sync ✓ + §0.5 c1~c14 모두 [DONE] ✓ + follow-up 3건 박제 ✓ | [TODO] |

### Plan-specific severe (WORKFLOW.md §12.3 default 위 추가분)

- `f0_reproduce_drift`: G1 F0 reproduce 가 plan-020/021/022 hard evidence ±0.0005 밖. → halt.
- `plan022_reproduce_drift`: G1 plan-022 A6_bcc14_τ001 reproduce 가 0.6528 ±0.0008 밖. → halt.
- `xattn_no_improvement`: G2 OOF hit_1cm < 0.6528 (plan-022 winner). → architecture lever 실패. plan-009 ranking_loss fail 패턴. **submission 생성 안 함** + decision-note 박제 + results.md 의 `g2_no_improvement` 박제 후 G_final 진입 (LB skip).
- `xattn_partial_pass`: G3 의 hit_1cm ≥ 0.6528 단 < 0.6628 (lift 미달) OR hit_1.5cm < 0.8104 OR gap_ranking > 0.04. warn (severe 아님). G_final 진입 + LB 회수 단 follow-up plan 강화 axis 박제 의무.
- `numerical_collapse`: torsion τ 의 `collinear_rate > 0.7` (sample 의 ≥ 70% 가 collinear → τ 신호 무의미). warn (severe 아님), results.md 박제 후 plan-026 의 redo 후보.
- `lb_below_floor`: G_final LB < plan-022 carry LB OR plan-004 LB 0.6806. warn (severe 아님), results.md 의 `lb_no_improvement` 박제 후 follow-up plan 강화 axis 박제 의무.

### Plan-specific paths (WORKFLOW.md §12.5/§12.6 default 위 추가/제외)

- whitelist 추가: `analysis/plan-024/**`, `tests/test_plan024_*.py`, `plans/plan-024-*.{md,results.md}` (이미 default whitelist 일 가능성, plan-024 명시)
- blacklist 추가: (없음 — plan-021/022 carry 모듈은 read-only / import 만)

### Decision-note 사용 예 (자율 결정 시 commit msg 박제)

- `decision-note: spec-default — GRU hidden=256 (plan-024 §4.6 default; input dim 32→62D + 14→89D 대비 capacity 보강 of PB framework default 128)`
- `decision-note: spec-default — τ_past = 0.003 (plan-024 §4.2; anchor radius 0.005m 의 0.6×, past F0 residual magnitude 5~20mm 스케일 매칭. output τ_cls=0.001 과 분리)`
- `decision-note: spec-default — seq length=7 (t=4..10 alignment; L2/L4 의 7-step coverage 와 일치, t<4 zero-pad/mask 회피)`
- `decision-note: spec-default — torsion τ collinear fallback τ_t := 0 + valid_mask channel (numerical-safe form)`

---

## §1. 배경

### §1.1 carry 박제 — plan-022 winner finding

plan-022 의 21-cell sweep (7 anchor layout × 3 τ_cls) 의 winner = **A6_bcc14_tau001** (14-anchor BCC: 6 axis ±t̂/±n̂/±b̂ + 8 corner ortant, ‖a‖=0.005m 모두):

- OOF hit_1cm = **0.6528** (Δ_F0 = +0.0208)
- OOF hit_1.5cm = 0.8104 (Δ_F0 = +0.0071)
- max_class_ratio = 0.105 (best distributed in sweep)
- q_true_max = 0.097 (자연 분포 최대 anchor share)
- top1_acc = 0.1707 (selector 가 17% sample 의 best anchor 정확 픽)
- soft CE = 2.5346 (vs uniform log(14) = 2.6391)

plan-022 의 paradigm = **selector-only corrector-free** — output = `Σ_k q_k · a_k` (anchor 의 soft-weighted average), corrector reg head 제거 (plan-021 finding: reg_offset ~0.01mm dead). model = LGBM multiclass classifier (170D input, K=14 output).

### §1.2 사용자 narrative — architecture lever

plan-022 결과 + 사용자 통찰 (2026-05-21 session "main"):

1. **plan-022 의 +2pp lift 가 작은 신호** — anchor representation 이 LGBM 안에서 *implicit* 학습 완료, *explicit* anchor coord 입력 lever 가 추가 잠재.
2. **plan-009 ranking_loss G1 fail** — listwise/pairwise *loss* 변경만으로 selector ranking 능력 회수 X (oof_soft_hit 0.6482, gap_ranking 0.108 = base 의 2배 악화). **architecture 가 진짜 lever** 라는 직접 증거.
3. **oracle gap 잔존** — plan-009 oracle_1cm=0.7562, 현 best 0.6528, gap +0.10pp. plan-008 gap_ranking 0.0516 = selector ranking 만으로 회수 가능한 측정된 lever (architecture-extractable headroom).
4. **plan-004 paradigm shift +8pp 증거** — polyfit (B001 LB 0.60) → PB framework (P001 LB 0.6806) 의 +8pp lift = same X, better architecture. 본 plan 의 *역사적 anchor*.

→ plan-024 의 목적 = cross-attention architecture (PB framework `CandidateAttentionGRUSelector` 1:1 carry) 위 anchor coord query + anchor-vocabulary past seq + FE 최대화 (4-way audit 누락 + 강화 form 박제) 로 **+0.01 lift (G3 hit_1cm ≥ 0.6628) 시도**.

### §1.3 4-way sub-agent audit finding (2026-05-21)

**audit A (Q1+Q5 F0 residual / torsion)**:
- **silent bug**: L2 (build_input.py:121) 의 residual = `pred - actual` vs build_soft_label_with_tau (selector_only_model.py:61) 의 residual = `gt - pred = actual - pred`. **opposite sign**. A6 대칭이라 *현재* silent OK, asymmetric anchor 도입 시 bug. → plan-024 구현 시 **`actual - pred`** 통일.
- **τ_past 분리**: output τ_cls=0.001 이 past F0 residual (magnitude 5~20mm) 에 over-sharp → 거의 hard one-hot 정보 손실. → τ_past ∈ {0.002, 0.003, 0.005} 별도. **default τ_past=0.003** (anchor radius 0.5cm 의 0.6×).
- **Frenet torsion τ**: codebase 부재. 신규 계산 모듈 (`torsion_calc.py`, c5).
- **강화 form**: F1 sign-unified, F2 log-magnitude `log(1+‖res‖/τ_ref)`, F3 anchor-projection K-scalar `<a_k/‖a_k‖, r_t>` (softmax-free 안정).

**audit B (Q2+Q3 world frame / z-axis)**:
- Frenet basis = **full 3D Frenet-Serret** (`R = stack([t̂, n̂, b̂], axis=-1)`, build_input.py:75). b̂ = t̂ × n̂ sample-마다 계산. world-z 와 fix 정렬 X.
- **"꺾쇠 Bz, Tz" 의미 해석**: `R_wfn[:, 2, :]` 의 z-row = `[t̂_z, n̂_z, b̂_z]` (Frenet basis 의 world-z component). 핵심: R_wfn 자체가 plan-024 input 에 없음 → **t̂_z, b̂_z 추가 = 순수 redundancy 아닌 새 정보** (gravity-aware).
- **권장**: t̂_z + b̂_z (2D, static broadcast, cand_feat ③ ctx) + seq Vz_world (1D, per step) + cand_feat anchor Δz_world (1D, per (sample, anchor), ④ interactions).

**audit C (Q4 plan-004 / plan-022 input audit)**:
- **누락 family**:
  1. macro_stat 9D (plan-021 `_macro_stat_9d`) — cand_feat ③ ctx broadcast.
  2. L1 EWMA 27D (= p/v/a 의 EWMA 9D × 3α; plan-022 lgbm_extra 의 부분집합) — plan-024 outline 의 seq J 묶음은 *residual* EWMA 9D 만, 두 source 분리 필요. **L1 EWMA 는 sample-level broadcast 로 cand_feat ③ ctx** (27D 추가). **v1.1 decision-note**: L1 EWMA 27D 는 v1.1 의도적 *미박제* (cand ③ ctx 128D 가 이미 macro_stat 8 + Multi-window 60D 로 trajectory-level aggregation 충분 cover, L1 EWMA 의 redundancy 우려) → **plan-025 후보로 분리** (followed_by 의 priority 5 와 함께 ablation).
  3. turn_cos, curvature, direction_flag per step (plan-004 make_seq_features 9D 의 부분) — seq 신규 channel.
  4. regime bin 18-class (plan-004 `assign_regimes`) — cand_feat ③ ctx (18D one-hot).
  5. F0-pred 자체의 Frenet 좌표 per step (현 outline 은 잔차만, pred 위치 자체 빠짐) — seq 신규 묶음.
  6. residual angle (azimuth, elevation) — magnitude/direction decouple.
- **L2 timing**: step i (i=0..6) 는 *t-2 시점의 80ms 예측* vs *t 시점의 실측* 의 retrospective sequence. **seq length=7 (t=4..10) alignment 권장** (t<4 zero-pad 회피).

**audit D (Q6 ideas.md plan-024 통합)**:
- **plan-024 paradigm 안 통합 가능** (priority 5): A1 Multi-window stat (+30~60D, +0.005~0.010) / A6 WAP composite (+4~8D, +0.001~0.003) / B3 STA/LTA ratio (+27D cost 0, +0.001~0.003) / Multi-Parse Input (raw/SG/EMA 3 parallel parse) / B2 Pct-of-rolling-std (+27D, +0.002~0.004).
- **plan-024 out-of-scope** (paradigm 밖, plan-025 후보로 분리): Trajectory-CLIP / KNN pool / MDN / 3-channel ensemble / Jerk regularizer / Path×accel / SE(3) corrector / Voxel head / PointNet / TTA / OHEM / Cascade / VQ / IRM / GMM (모두 anchor pool / output head / loss / corrector / ensemble axis).
- **결정**: priority 5 도 plan-024 의 scope 안에 *모두 박제* 시 학습 시간 + dim explosion 위험. **plan-024 = Stage 1+2 (audit 누락 + 강화 form) 만**. priority 5 = plan-025 후보로 frontmatter 박제.

### §1.4 가설

**H_main**: plan-022 winner (LGBM, 170D) 의 selector 를 cross-attention GRU (**v1.1: 150D cand + 95D seq + hidden=384**) 로 교체 + FE 최대화 시, **gap_ranking 회수** (= 0.0516 → ≤ **0.04**, §3.2 G3 gate 와 통일 — partial 회수 0.012pp 이상) + **hit_1cm +0.01 lift** (0.6528 → 0.6628).

**H_secondary**:
- H1 (architecture lever > loss lever): plan-009 의 listwise loss fail 위에서 cross-attention 의 inner product 가 ranking 능력 직접 학습 → top1_ranking_acc ≥ 0.20 (plan-008 extended 0.17 보다 ↑).
- H2 (anchor-vocab encoding 효과): seq F/G/H 묶음 (anchor-vocabulary re-encoding) 가 input ↔ output 어휘 통일 → soft_CE ≤ 2.50 (plan-022 winner 2.5346 보다 ↓).
- H3 (world-z 정보 lever): t̂_z + b̂_z + Vz + anchor Δz_world 추가가 *순수 redundancy 가 아닌* 새 정보 → ablation 시 contribution ≥ +0.003 (G3 통과 후 plan-025 ablation 후보).
- H4 (Frenet torsion τ lever): muflight low-curvature regime 에서 mostly 0 단 high-τ tail discriminator 로 가치 → contribution ≥ +0.001.

---

## §2. 가설 검증 paradigm (single config 원칙)

plan-022 가 21-cell sweep 으로 *anchor layout* 변수 ablation 했다면, plan-024 는 **single config full FE max** — *architecture + input FE 묶음* 을 한 번에 측정. ablation (각 묶음 contribution 분해) = G3 통과 후 plan-025 영역.

**Single evaluation run, multi-lever simultaneous addition** (v1.1 self-label re-cast): plan-022 winner cell 기준에서 변경 = (1) selector architecture LGBM → cross-attention GRU, (2) input dim 170D 1-vector → **v1.1: 150D cand + 95D seq 2-input 구조** (17 lever 동시 추가 — Tier S 5 + Tier A 옵션 B 10 + redundancy 제거 3 + regularizer 신규 2), (3) hidden 128 PB default → **384** + 5 hyperparam 변경 (dropout / lr / weight_decay / epoch 등). 단 anchor / τ_cls(output) / hit radius / F0 baseline / 5-fold split = 모두 plan-022 carry. **caveat #16 박제**: 17 lever 동시 → G3 fail 시 어느 lever 가 bottleneck 인지 본 plan 안에서 분해 불가, ablation = plan-025.

**Out-of-scope (절대 안 함)**:
- anchor layout 변경 (A6_bcc14 고정)
- anchor radius 변경 (0.5cm 고정)
- τ_cls (output) 변경 (0.001 고정, 단 τ_past 는 별도)
- F0 baseline 변경
- corrector reg head 재투입
- ensemble (plan-022 LGBM 평균)
- ideas.md priority 5 추가 (plan-025 후보)
- hyperparam sweep (GRU hidden / lr / epoch / batch — single config 고정)

---

## §3. 사전 등록 (Pre-registration)

### §3.1 Fold split (plan-020/021/022 carry)

- 5-fold via `stable_fold_id(str(sample_id), 5)` (MD5 prefix mod 5)
- fold distribution = [2020, 2047, 1921, 2020, 1992] (plan-022 carry 박제, N=10000)
- F0 deterministic — fold-leakage 면제 (= plan-021/022 §3.1 carry)
- OOF metric = 5-fold concat hit rate (per-fold 평균 아님)

### §3.2 합격 기준 (정량)

| Gate | 조건 (AND) | severe / warn |
|---|---|---|
| G0 | smoke + tests **10/10** green (v1.1: weight + channel mask 추가) | `infra_drift` severe |
| G1 | F0 hit_1cm ∈ [0.6315, 0.6325] AND hit_1.5cm ∈ [0.8028, 0.8038] AND plan-022 A6_bcc14_τ001 hit_1cm ∈ [0.6520, 0.6536] AND hit_1.5cm ∈ [0.8096, 0.8112] | `f0_reproduce_drift` / `plan022_reproduce_drift` severe |
| G2 | OOF metric finite AND max_class_ratio < 0.95 AND OOF hit_1cm ≥ 0.6528 | `xattn_no_improvement` severe |
| G3 | hit_1cm ≥ 0.6628 AND hit_1.5cm ≥ 0.8104 AND gap_ranking ≤ 0.04 | `xattn_partial_pass` warn |
| G_final | LB 박제 + 3-file sync + follow-up 3건 | `lb_below_floor` warn |

### §3.3 평가 점수

- 1차 metric: OOF hit_1cm (= 5-fold concat OOF 의 ‖pred_world - Y‖ ≤ 0.01m 비율)
- 2차 metric: OOF hit_1.5cm, Δ_F0 (hit_cell - hit_F0), max_class_ratio, q_true_max, KL(probs_avg ‖ q_true_avg), top1_acc, soft_CE, gap_ranking (= oracle_1cm - argmax_hit, plan-008 carry)
- 최종 metric: dacon-submit LB score

### §3.4 Anchor layout (single, plan-022 carry)

- **A6_bcc14** (BCC 14): axis 6 (±t̂/±n̂/±b̂ × 0.005m) + corner 8 (±0.005/√3 m 각 component, 8 ortant). ‖a_k‖ = 0.005m all k. center 없음.
- 정의 출처: `analysis/plan-022/anchors.py:107-119` 그대로 import.

### §3.5 τ_cls (output) 및 τ_past (input F) scan

- τ_cls = **0.001** (plan-022 winner carry, output soft label sharpness — 변경 X)
- τ_past = **0.003** (default; past F0 residual magnitude 5~20mm 스케일 매칭). G1 통과 후 ablation 후보 (plan-025).

### §3.6 gap_ranking 정의 (self-contained, plan-008 carry)

```python
# gap_ranking = oracle_1cm - argmax_hit
# oracle_1cm  = "14 anchor 중 *가장 정답에 가까운* anchor 의 hit@1cm 비율"
#             = (min_k ‖a_k_world - Y‖ ≤ 0.01).mean()
#             where a_k_world = R_wfn @ anchor_k_frenet + pred_F0_world (sample 별)
#             즉 anchor 14개 중 best 를 선택했을 때의 *상한* hit rate
# argmax_hit  = "selector 가 단일 top-1 anchor 만 선택했을 때의 hit@1cm 비율"
#             = (‖a_{argmax_k q_pred[i,k]}_world - Y‖ ≤ 0.01).mean()

oracle_dist  = min_k ‖R_wfn @ anchors[k] + pred_F0_world - Y‖   # (N,)
oracle_1cm   = (oracle_dist <= 0.01).mean()
argmax_idx   = q_pred.argmax(axis=1)                            # (N,)
argmax_pos   = R_wfn @ anchors[argmax_idx] + pred_F0_world      # (N, 3)
argmax_dist  = ‖argmax_pos - Y‖
argmax_hit   = (argmax_dist <= 0.01).mean()
gap_ranking  = oracle_1cm - argmax_hit                          # scalar
```

**reference value** (plan-008 base 27-cand carry): oracle_1cm=0.7562, argmax_hit=0.7046, gap_ranking=**0.0516**. plan-024 의 14-anchor 위 oracle 은 학습 후 직접 측정 (G3 박제).

---

## §4. 핵심 결정 — Input + Architecture spec

### §4.0 frame / R_wfn convention 박제 (v1.1)

본 plan 의 모든 `R_wfn` 참조는 다음 convention 으로 통일 (plan-021 `build_frenet_basis_3d` carry):

- **shape**: `(N, 3, 3)`, float32.
- **column convention**: `R_wfn = np.stack([t̂, n̂, b̂], axis=-1)` — *columns = Frenet basis vectors (in world frame)*. 즉 `R_wfn[:, :, 0] = t̂_world`, `R_wfn[:, :, 1] = n̂_world`, `R_wfn[:, :, 2] = b̂_world`.
- **z-row 추출** (audit B의 `t̂_z, b̂_z`): `R_wfn[:, 2, 0]` = `t̂_world[:, 2]` = `t̂_z`. `R_wfn[:, 2, 2]` = `b̂_z`. *row 2 = world-z component of each Frenet basis vector*.
- **world → Frenet 변환**: `frenet_vec = R_wfn^T @ world_vec` (= `einsum("nij,nj->ni", R_wfn.transpose(0,2,1), world_vec)`).
- **Frenet → world 변환**: `world_vec = R_wfn @ frenet_vec`.
- **per-sample 단일 frame**: `R_wfn` 은 *sample 당 1개* (= `end_idx=10` 기준 *마지막 step 의 Frenet basis*). seq 의 7 step 모두 *같은 R_wfn* 사용 (per-step alignment 아님, audit C 의 timing alignment 와 일관). plan-021 `build_input_common` 의 carry 그대로 (`R_wfn = build_frenet_basis_3d(X, end_idx=X.shape[1]-1)`).
- **origin**: seq A 묶음 `X[:, t] − origin` 의 `origin = X[:, end_idx]` (= `X[:, 10]` 마지막 step 의 world 좌표). plan-021 `build_input_common` 의 carry.

### §4.1 전체 input 구조 (v1.1)

| input | shape | dim | sample-conditional | anchor-conditional |
|---|---|---|---|---|
| **cand_feat** | (b, 14, **150**) | **150D** | ✓ (③ ctx broadcast) | ✓ (①②④) |
| **seq** | (b, 7, **95**) | **95D** | ✓ | ✗ (anchor-agnostic time series) |

**v1 → v1.1-rev2 변화**:
- cand 62D → 150D (+ STA/LTA 3 + Multi-window 60 + WAP sample 5 + wingbeat 3 + f0_conf sample 2 + Pct-rolling+Peak 12 + v_autocorr 3 + BCC adjacency 2 − straightness 1 − axis×forward 1. **rev2: S4 Anchor coord Fourier PE 12 제거** — 14 discrete BCC token 에 continuous-PE 부적합 + plan-025 의 Learnable embedding lever 오염 회피)
- seq 89D → 95D (+ jerk 3 + ω 3 + saccade 2 + time PE 4 + saliency 1 + helicity 1 + WAP per-step 5 + f0_conf per-step 1 − F3 anchor-projection 14)
- per-channel learnable scale (**150 + 95 = 245** params) + channel dropout (cand ③ 128D 만, p=0.3; seq 의 EWMA+Multi-window 영역, p=0.2)

### §4.1.1 build_seq / build_cand signature spec (v1.1, self-contained)

```python
# c3 seq_builder.py
def build_seq(
    X: np.ndarray,                # (N, 11, 3) float32, world coord
    R_wfn: np.ndarray,            # (N, 3, 3) float32, per-sample Frenet basis (§4.0)
    pred_F0_world: np.ndarray,    # (N, 3) float32, end_idx=10 의 80ms 미래 F0 예측
    anchors: np.ndarray,          # (14, 3) float32, ANCHORS_A6 Frenet
    tau_past: float,              # 0.003 (v1.1 default)
    quantile_carry: dict,         # {'omega_p90': float, 'jerk_p90': float, ...} train fold carry
    regimes: np.ndarray | None,   # 미사용 (seq 는 regime feature 없음, cand 만 사용)
) -> np.ndarray:                  # (N, 7, 95) float32, t=4..10 channel ordering §4.3
    ...

# c4 cand_builder.py
def build_cand(
    X: np.ndarray,                # (N, 11, 3) float32
    R_wfn: np.ndarray,            # (N, 3, 3) float32
    pred_F0_world: np.ndarray,    # (N, 3) float32
    anchors: np.ndarray,          # (14, 3) float32
    regimes: np.ndarray,          # (N,) int64, regime 18-class assign (plan-004 carry)
    macro_stat: np.ndarray,       # (N, 9) float32, plan-021 `_macro_stat_9d` carry
                                  #   (v1.1: idx 1 straightness 제외 8 사용, build_cand 내부 slice)
    ewma_alphas: tuple = (0.1, 0.3, 0.5),
    multiwindow_trim_path: str,   # "analysis/plan-024/multiwindow_trim.json" path
    quantile_carry: dict,         # train fold carry
) -> np.ndarray:                  # (N, 14, 150) float32, anchor row × channel §4.4
    ...
```

**호출 순서** (run_oof.py §6.2 안):
1. STAGE 0 (c5.5 quantile_carry.py + c4 cand_builder.py 와 별도 build): `multiwindow_trim_build.py` 실행 → `multiwindow_trim.json` 박제 (per §4.4.1).
2. STAGE 1 (G1 reproduce): macro_stat / EWMA / regime 미리 계산 → cache.
3. STAGE 2 (G2 OOF): per fold k 의 train 위 `quantile_carry.build(...)` → train-only quantile dict 박제 → `build_seq` / `build_cand` 호출 (train + valid fold 동일 quantile_carry 사용, fold-leakage 차단).

### §4.2 anchor_vocab 묶음 식 (c2 `analysis/plan-024/anchor_vocab.py`)

**f0_baseline 시그너처 박제** (plan-020 carry, self-contained):

```python
# analysis/plan-020/baseline_f0.py 의 인터페이스 (carry, 변경 X)
def f0_baseline(sub_x: np.ndarray, end_idx: int) -> np.ndarray:
    """sub_x: (N, T_sub, 3) float64 world coord. end_idx ∈ [2, T_sub-1].
    return: (N, 3) float64 — sub_x 의 end_idx 시점에서 80ms (= 2 step × 40ms) 미래 위치 예측.
    공식: pred = sub_x[:, end_idx] + D1 · (sub_x[:, end_idx] - sub_x[:, end_idx-1]) · PAR
         + perp 보정 (D1=1.98, PAR=1.20, PERP=−0.20, plan-020 §3 carry)."""
```

본 plan-024 사용 패턴 (§4.2 의 anchor_vocab 식 안):
- `sub_x_t = X[:, t-4:t-1, :]` → shape `(N, 3, 3)` (3 past step 의 world 좌표).
- `f0_baseline(sub_x_t, end_idx=2)` → `(N, 3)` world 좌표 (sub_x_t 의 step index 2 = `X[:, t-2]` 시점의 80ms 미래 = `X[:, t]` 의 예측).
- import 시 signature drift 시 G0 smoke test #1 의 `f0_baseline` interface check 로 catch (예: `assert f0_baseline(...).shape == (N, 3)`).

per past step t (t ∈ {4, 5, ..., 10}):

```python
# step t 의 80ms 미래 F0 잔차 (sign-unified: actual - pred)
sub_x_t = X[:, t-4:t-1, :]                           # (N, 3, 3)
pred_t  = f0_baseline(sub_x_t, end_idx=2)            # (N, 3) world
actual_t = X[:, t]                                    # (N, 3) world
residual_w_t = actual_t - pred_t                      # ★ sign 통일 (plan-021 L2 와 opposite)
residual_t_frenet = R_wfn^T @ residual_w_t            # (N, 3) Frenet

# F 묶음: soft assignment (τ_past 별도 인자)
q_past_t = softmax(-‖a_k - residual_t_frenet‖ / τ_past)   # (N, 14)

# G 묶음: magnitude
mag_t = ‖residual_t_frenet‖                            # (N, 1)

# H 묶음: top1 one-hot (argmin distance)
top1_t = argmin_k ‖a_k - residual_t_frenet‖            # (N,)
onehot_t = one_hot(top1_t, K=14)                       # (N, 14)

# F3 묶음: anchor-projection K-scalar (softmax-free)
proj_t = anchors @ residual_t_frenet / 0.005           # (N, 14), unit-normalized

# F2 묶음: log-magnitude (decouple from G)
log_mag_t = log(1 + mag_t / 0.005)                     # (N, 1)
```

- **sign**: `actual - pred` (plan-022 build_soft_label_with_tau 와 통일). L2 import 시 negate.
- **τ_past**: 별도 인자 (default 0.003, hyperparam 박제).
- **F3 unit-normalize**: anchor radius 0.005m 로 나눠 [-1, +1] 범위로 안정화 (모든 a_k 가 ‖a‖=0.005).

### §4.3 seq builder (c3 `analysis/plan-024/seq_builder.py`) — v1.1

per past step t (t ∈ {4, ..., 10}, length=7), **95D** channel:

| 묶음 | source | dim | 식 | v1.1 변화 |
|:--|:--|--:|:--|:--|
| **A** position Frenet | L1 `[p_t, p_n, p_b]` | 3 | `R_wfn^T @ (X[:, t] − origin)` | — |
| **B** velocity Frenet | L1 `[v_t, v_n, v_b]` | 3 | `R_wfn^T @ (X[:, t] − X[:, t-1])` | — |
| **C** acceleration Frenet | L1 `[a_t, a_n, a_b]` | 3 | `R_wfn^T @ (Δv_t − Δv_{t-1})` | — |
| **S1 jerk Frenet** ⭐ | derived (A1+A3) | 3 | `j_t = (a_t − a_{t-1}) / Δt`, Frenet 분해 | **+ v1.1 (cross-confirmed)** |
| **S2 angular velocity ω Frenet** ⭐ | derived (A3) | 3 | `ω_t = R_wfn^T · (v_{t-1} × v_t) / ‖v_t‖²` | **+ v1.1 (Top-1)** |
| **Vz_world** | derived | 1 | `(X[:, t, 2] − X[:, t-1, 2]) / 0.040` (gravity-aware) | — |
| **D** F0 residual Frenet | sign-unified L2 | 3 | `R_wfn^T @ (actual_t − pred_t)` | — |
| **residual angle** | derived | 2 | `[atan2(res_n, res_t), asin(clip(res_b / max(‖res‖, 1e-9), -1.0, 1.0))]` (eps guard + clip for asin domain) | — |
| **pred_F0 Frenet** | derived | 3 | `R_wfn^T @ (pred_t − origin)` | — |
| **E** soft hit | L4 (plan-021 carry) | 2 | `[σ((R_HIT − d)/τ), σ((R_HIT_LOOSE − d)/τ)]` | — |
| **F** anchor-vocab soft (τ_past) | §4.2 | 14 | `softmax(-‖a_k - r_t‖/τ_past)` | — |
| **G** ‖residual_t‖ | §4.2 | 1 | `‖r_t‖` | — |
| **H** anchor-vocab top1 one-hot | §4.2 | 14 | argmin one-hot | — |
| ~~**F3** anchor-projection~~ | ~~§4.2~~ | ~~14~~ | ~~`a_k · r_t / ‖a_k‖`~~ | **− v1.1 (redundancy A1+A2)** |
| **F2** log-magnitude | §4.2 | 1 | `log(1 + ‖r_t‖ / 0.005)` | — |
| **I** Δresidual | derived | 3 | `r_t − r_{t-1}` | — |
| **Δ²residual** | derived | 3 | `r_t − 2·r_{t-1} + r_{t-2}` | — |
| **J** residual EWMA (3α) | plan-021 `_ewma_last` | 9 | α ∈ {0.1, 0.3, 0.5} × Frenet of r | — |
| **K** time offset | positional | 1 | `t / 10` | — |
| **S5 sinusoidal time PE** ⭐ | positional (A4) | 4 | `[sin(2πt/7), cos(2πt/7), sin(4πt/7), cos(4πt/7)]` | **+ v1.1 (Vaswani 2017)** |
| **L** F entropy | derived | 1 | `H(q_past_t) = -Σ_k q log q` | — |
| **M** F 2nd-best mass | derived | 1 | `sort(q_past_t)[-2]` | — |
| **O** speed magnitude | derived | 1 | `‖v_t‖` | — |
| **A9 anchor-saliency prior** ⭐ | derived (A4 Deformable DETR mimic) | 1 | `max_k <a_k/‖a_k‖, r_t>` | **+ v1.1** |
| **A11 helicity** ⭐ | derived (A3) | 1 | `v_t · ω_t` (corkscrew indicator) | **+ v1.1** |
| **A5 WAP per-step** ⭐ | derived (A2 Optiver WAP) | 5 | `[‖v‖²·κ, ‖j‖/(‖a‖+ε), ½‖v‖², ‖v_perp‖·τ, dist·‖a_perp‖]` | **+ v1.1** |
| **A8 f0_conf per-step** ⭐ | derived (A1 PBP) | 1 | `polyfit_residual_norm_t / step_spread_t` — step t 의 *local polyfit* (window = 마지막 3 step, degree=1, 즉 linear extrap residual) divided by step_spread (= `‖v_{t-1} - v_{t-2}‖ / Δt`, 가속도 magnitude proxy). **per-step time-varying**, *cand ③ A8 sample-level 과 구분* (sample-level = end_idx=10 1 시점, per-step = t∈{4..10} 7 시점) | **+ v1.1** |
| **S3 saccade binary** ⭐ | derived (A1+A3) | 2 | `[1{‖ω_t‖ > q90_train}, 1{turn_cos_t < cos(60°)}]` | **+ v1.1 (cross-confirmed)** |
| **turn_cos** | plan-004 carry | 1 | `v_t · v_{t-1} / (‖v_t‖‖v_{t-1}‖)` | — |
| **curvature** | plan-004 carry | 1 | `perp_norm / speed` | — |
| **direction_flag** | plan-004 carry | 1 | constant `+1.0` | — |
| **torsion τ** | §4.5 | 3 | `[τ_t, sign(τ)·log(1+|τ|), valid_mask]` | — |

**total**: 3+3+3+**3+3**+1+3+2+3+2+14+1+14+ ~~14~~ +1+3+3+9+1+**4**+1+1+1+**1+1+5+1+2**+1+1+1+3 = **95D** per step (v1: 89D, Δ = +20 추가 −14 F3 = +6). seq shape = (N, 7, 95).

⭐ = v1.1 추가 (4-way ML expert review 결과 박제).

### §4.4 cand_feat builder (c4 `analysis/plan-024/cand_builder.py`) — v1.1-rev2

per anchor k (k=0..13), per sample, **150D** channel:

| 묶음 | dim | source / 식 | v1.1 변화 |
|:--|--:|:--|:--|
| **① par/perp/dist** (sample × anchor) | 3 | `(a_k - residual_last) → Frenet 분해 par/perp + ‖.‖` | — |
| **② anchor spec** (anchor-static) | **9** | base 9 (Frenet coord 3 + sign 3 + group 2 + idx 1). ~~S4 Anchor coord Fourier PE 12~~ **제거 (v1.1-rev2 사용자 지적)** — 14 BCC token = *discrete fixed ID*, continuous-PE 부적합 (ID one-hot 14D 와 정보량 동등 + expressivity 낮음 + 4πr high-freq term 의 14-sample aliasing 위험). 또한 Learnable anchor embedding (A7, plan-025 후보) 와 *같은 axis 의 fixed-basis 약한 버전* → plan-025 lever 측정 오염 회피. **anchor identity 학습은 ② idx scalar 1D + ④ interactions 의 anchor coord 곱으로 충분** (cross-attention 의 query MLP 가 idx + coord 로 identity 학습). | **(제거, plan-025 lever 분리)** |
| **③ ctx broadcast** (sample × all anchors 같은 값) | **128** | **모두 Frenet frame, last step 기준** (audit B + §4.0 convention). base 12 (last v Frenet 3 + last acc Frenet 3 + last F0 res Frenet 3 + EWMA(α=0.3) of res Frenet 3) + macro_stat **8** (~~straightness~~ 제거, A2 redundancy R4 — plan-021 `_macro_stat_9d` 중 idx 1 straightness 제외 9→8) + Bz/Tz 2 (`[R_wfn[:, 2, 2], R_wfn[:, 2, 0]]` per §4.0) + regime 18 (one-hot) + **A1 STA/LTA ratio 3** (EWMA α=0.5 / EWMA α=0.1 ratio per Frenet axis t̂/n̂/b̂, of F0 residual) + **A2 Multi-window stat grid 60** (`[전체 11, 뒤 7, 뒤 5, 뒤 3] sub-window × [mean, std, slope, max] × 9 channel` = 4×4×9 = 144D 후보, **trim 60D** — 9 channel = Frenet `[p_t, p_n, p_b, v_t, v_n, v_b, a_t, a_n, a_b]`, **trim 절차** §4.4.1 박제) + **A5 WAP sample-level 5** (last-step Frenet `[‖v‖²·κ, ‖j‖/‖a‖, ½‖v‖², ‖v_perp‖·τ_frenet, dist·‖a_perp‖]`) + **A6 wingbeat-jitter envelope 3** (std of `(p_Frenet - EWMA_{α=0.6}(p_Frenet))` per Frenet axis) + **A8 f0_conf sample-level 2** (polyfit residual norm `‖F0_pred - last_step_world_extrap‖` + `step_spread = std(consecutive_speed)`) + **A10 Pct-rolling+Peak 12** (`[pct_{20,50,80}(rolling_std(‖v_Frenet‖, w∈{3,5,7})) → 3×3=9, count_{t=4..10}(‖j_Frenet[t]‖ > quantile_carry.jerk_p90)  // sample-level scalar 1, count_{t=5..10}(sgn(v_Frenet[t, 0]) · sgn(v_Frenet[t-1, 0]) < 0)  // t̂-axis sign flip scalar 1, count_{t=5..10}(turn_cos[t] < 0.5)  // sharp turn scalar 1]` → 9+1+1+1=12) + **A12 v_autocorr 3** (per lag k∈{1,2,3}: `corr(stack([v_Frenet[t, c] for t in range(7)]), stack([v_Frenet[t-k, c] for t in range(k, 7)]))` 의 *3축 (t̂/n̂/b̂) 평균* — `mean_c(Pearson(v_c[k:], v_c[:-k]))` 단일 scalar per k = 3개) | **+88 (A1+A2+A5+A6+A8+A10+A12), -1 straightness** |
| **④ interactions** (sample × anchor) | **10** | base **8** *모두 scalar (1D each)*: (1) anchor·res = `<a_k, r_last_Frenet>` 1, (2) anchor·v = `<a_k, v_last_Frenet>` 1, (3) anchor·acc = `<a_k, acc_last_Frenet>` 1, (4) anchor·EWMA = `<a_k, EWMA(α=0.3)(r_Frenet)>` 1, (5) corner×turn = `is_corner_k · turn_cos_last` 1, (6) sign-agreement = `Σ_c sgn(a_k_c)·sgn(r_last_c)` 1, (7) physics-extrap·anchor = `<a_k, v·Δt + ½·acc·Δt²>` 1 (Δt = 0.080s = 80ms), (8) anchor·Δz_world = `(R_wfn @ a_k)[2]` 1 (= 8 scalar) + **A3 BCC adjacency neighbor pool 2 *모두 scalar* (`[mean_{j∈N(k)}<a_j, r_last>, std_{j∈N(k)}<a_j, r_last>]`, N(k) = anchor k 의 BCC 3-4 nearest neighbor, adjacency precompute static) | **+2 (A3 Set Transformer ISAB mimic), -1 axis×forward** |

**total**: 3 + **9** + **128** + **10** = **150D**. cand_feat shape = (N, 14, 150). (v1: 62D → v1.1: 162D → **v1.1-rev2: 150D** (Fourier PE 12 제거).)

⭐ = v1.1 추가 (4-way ML expert review 결과 박제).

### §4.4.1 Multi-window stat grid 144→60 trim 절차 (v1.1)

**Stage**: STAGE 0 (인프라) 안에서 c4 cand_builder 와 별도 module `analysis/plan-024/multiwindow_trim_build.py` 의 **deterministic** 절차 (random 없음, fold-leakage 없음 — full train set 144D 계산 후 *모든 sample 합산* 위 correlation 추출).

**식**:
1. Full train set (N=10000) 의 144D Multi-window stat 계산 = (N, 144) 행렬.
2. 144×144 absolute Pearson correlation matrix `C` 계산.
3. Greedy column drop: corr > 0.95 인 쌍 중 *variance 작은 column* 제거. drop 후 남은 column 수 ≤ 60 까지 반복. drop 못 해도 60 column 이상 남으면 추가로 variance 낮은 column 부터 drop.
4. 최종 60-col index list 를 `analysis/plan-024/multiwindow_trim.json` 박제: `{"kept_indices": [int × 60], "drop_indices": [int × 84], "corr_threshold": 0.95}`.

**fold-leakage 결정** (단일 path 박제, 결정 미루기 X):
- **채택**: full train (N=10000) 위 계산. trim 결정은 *correlation structure* (각 column 의 sample variance) 만 활용하고 *label* (Y) 안 사용 → leakage scale 미미 (LANL Singer 1st pattern carry, Kaggle PLAsTiCC writeup 의 cross-fold deterministic transform 패턴 일치).
- **mitigation 옵션** (train fold-별 분리 trim) = **거부** — single config 원칙상 *모든 fold 동일 trim* 강제, fold-별 분리 trim 은 (a) 5개 다른 trim_indices → 5개 model 학습 시 input dim 불일치 → ensemble 비교 불가능, (b) decision-note 박제는 *deterministic single trim* 만.
- **carry**: trim 자체가 deterministic 함수라 single config 안에서 *재실행 시 동일 결과* 보장. ablation 없음.

### §4.5 torsion_calc (c5 `analysis/plan-024/torsion_calc.py`)

per past step t (t ∈ {4, ..., 10}, length=7), per sample:

```python
v_t   = X[:, t]   - X[:, t-1]
v_tm1 = X[:, t-1] - X[:, t-2]
cross_t = cross(v_tm1, v_t)                                 # (N, 3) world
cross_norm = ‖cross_t‖                                       # (N,)
# eps-guarded normalize (collinear NaN guard *before* mask)
b_hat_t = cross_t / max(cross_norm, eps_collinear)[:, None]  # (N, 3)

# consecutive b̂ alignment (first valid step initialization)
if t == 4 or b_hat_prev is None:
    pass                                                     # no flip on first step (init carry)
else:
    b_hat_t = sign(dot(b_hat_t, b_hat_prev))[:, None] * b_hat_t   # sign-flip detection

db = b_hat_t - b_hat_prev if b_hat_prev is not None else zeros_like(b_hat_t)
ds = max(‖v_t‖, eps_speed)                                  # arc length proxy (eps guard)
n_hat_t = perp(v_t, b_hat_t)                                # Frenet normal at step t
                                                             # perp(v, b) := normalize(b × v) — n̂ = b̂ × t̂ convention

# torsion scalar (Frenet-Serret 3rd formula)
tau_t = -dot(db / ds[:, None], n_hat_t)                     # sign-aware scalar (N,)

# numerical safety (post-hoc mask, eps guard 후 NaN 발생 X)
valid_mask_t = (cross_norm > eps_collinear) & (‖v_t‖ > eps_speed) & (b_hat_prev is not None)
tau_t = where(valid_mask_t, tau_t, 0.0)

# carry b_hat for next step alignment
b_hat_prev = b_hat_t                                         # for sign-flip + db at t+1

# transform for seq input (3D)
seq_torsion_t = [tau_t, sign(tau_t) * log(1 + |tau_t|), valid_mask_t.float()]
```

- numerical thresholds: `eps_collinear = 1e-6`, `eps_speed = 1e-4`.
- t=4, 5 (insufficient history) → seq_torsion = `[0, 0, 0]` (mask=0).
- 측정 박제: `collinear_rate = (~valid_mask).mean()` → results.md 의 `numerical_collapse` warn trigger.

### §4.6 model spec (c6 `analysis/plan-024/model.py`) — v1.1

**PB framework arch 의 anchor-vocab task fit 정합**: PB framework `CandidateAttentionGRUSelector` 의 원래 task = 27 physics candidate 의 selector (plan-004 LB 0.6822). plan-024 의 task = 14 BCC anchor 의 corrector-free `Σ q · a` (plan-022 carry). 둘 다 **K-cand softmax classifier with GRU-attended past-seq + per-cand spec query** 의 동일 abstraction — PB 의 `cand_count=27, cand_dim=32` 가 plan-024 의 `cand_count=14, cand_dim=150` 로 *parametric swap* 만 필요, *gradient path / loss form / output 구조* 모두 fit. 단 plan-024 의 anchor-vocab encoding (input ↔ output 같은 anchor 어휘) + per-channel learnable scale + channel dropout 은 PB framework 위 *추가 input adaptor 층* 으로 박제 (model.py 안 thin wrapper).

**PB `CandidateAttentionGRUSelector` forward 식 self-contained 박제** (src/pb_0_6822/selector.py:697-727 carry — 본 plan-024 의 구현자가 외부 file read 없이 자족적으로 구현 가능하도록):

```python
class CandidateAttentionGRUSelector(nn.Module):
    def __init__(self, seq_dim, cand_dim, hidden, cand_count):
        self.gru = nn.GRU(seq_dim, hidden, num_layers=2, dropout=0.08, batch_first=True)
        self.query_mlp = nn.Sequential(nn.Linear(cand_dim, hidden), nn.GELU(),
                                        nn.Linear(hidden, hidden))
        self.head = nn.Sequential(nn.Linear(2*hidden + cand_dim, hidden), nn.GELU(),
                                   nn.Dropout(0.10), nn.Linear(hidden, 1))
        self.cand_count = cand_count

    def forward(self, seq, cand_feat):
        # seq: (b, T=7, seq_dim=95). cand_feat: (b, K=14, cand_dim=150).
        out, h = self.gru(seq)            # out: (b, T, hidden=384), h: (2, b, hidden)
        h_final = h[-1]                    # (b, hidden) — last GRU layer final hidden
        query = self.query_mlp(cand_feat) # (b, K, hidden)
        # attention: softmax over T axis (per-cand temporal weight)
        attn_logits = einsum("bth,bkh->bkt", out, query) / sqrt(hidden)  # (b, K, T)
        attn = softmax(attn_logits, dim=-1)                              # (b, K, T)
        event_ctx = einsum("bkt,bth->bkh", attn, out)                    # (b, K, hidden)
        # head: concat(global ctx, per-cand attended ctx, raw cand_feat)
        h_final_broadcast = h_final.unsqueeze(1).expand(-1, cand_count, -1)  # (b, K, hidden)
        head_in = cat([h_final_broadcast, event_ctx, cand_feat], dim=-1)     # (b, K, 2*hidden + cand_dim)
        score = self.head(head_in).squeeze(-1)                               # (b, K)
        return score   # (b, K) logits, softmax(score, dim=-1) → q_pred
```

본 forward 식은 PB framework 의 *exact carry* — plan-024 v1.1 의 `hidden=384` 만 변경, 다른 모든 layer 시그너처 / dim flow 정합. v1.1 의 input adaptor (FeatureWeightedDropout, 위 정의) 는 backbone 의 `forward(seq, cand_feat)` 호출 *직전* 에 (seq, cand_feat) 변환만 추가.

```python
# PB framework 그대로 import (src/pb_0_6822/selector.py:697)
from src.pb_0_6822.selector import CandidateAttentionGRUSelector
from .feature_weighted_dropout import FeatureWeightedDropout

class CrossAttentionAnchorSelector(nn.Module):
    def __init__(self):
        super().__init__()
        # v1.1: per-channel learnable scale + channel dropout (LGBM feature_fraction NN 등가)
        self.fwd = FeatureWeightedDropout(
            cand_dim=150, seq_dim=95,
            cand_drop_p=0.3,            # ③ ctx broadcast 128D 영역만
            seq_drop_p=0.2,             # EWMA J + WAP + Multi-broadcast slice
            cand_drop_start=12,         # ①3 + ②9 = 12 까지 보호 (drop X) — rev2: ② 21→9
            cand_drop_end=140,          # ③ 끝 (12 + 128). 이후 ④10 보호
            seq_drop_indices=list(range(62, 71)) + list(range(81, 86)),
            # = J EWMA(α=0.1/0.3/0.5) 9D channel index [62..70] + A5 WAP per-step 5D channel [81..85] = 14 channel.
            # seq 95D ordering (§4.3 표 순서, 0-indexed): A 0-2 / B 3-5 / C 6-8 / S1 jerk 9-11 /
            # S2 ω 12-14 / Vz 15 / D 16-18 / angle 19-20 / pred_F0 21-23 / E 24-25 /
            # F 26-39 / G 40 / H 41-54 / F2 55 / I 56-58 / Δ² 59-61 / J 62-70 / K 71 /
            # S5 PE 72-75 / L 76 / M 77 / O 78 / A9 79 / A11 80 / A5 WAP 81-85 / A8 86 /
            # S3 saccade 87-88 / turn_cos 89 / curv 90 / dir 91 / torsion 92-94.
            # drop 대상: J EWMA (62-70) + A5 WAP per-step (81-85) — *redundant smoothing/composite*.
            # kinematic (A/B/C/jerk/ω), residual (D/angle/pred/E), anchor-vocab (F/G/H/F2),
            # Δ/Δ², meta (K/S5 PE/L/M/O), saccade-relevant (A9/A11/A8/S3), geometry (turn/curv/dir/torsion) 보호.
        )
        self.backbone = CandidateAttentionGRUSelector(
            seq_dim=95,                 # v1.1 §4.3
            cand_dim=150,               # v1.1 §4.4
            hidden=384,                 # v1: 256 → v1.1: 384 (input dim 추가 확장 대비)
            cand_count=14,
        )

    def forward(self, seq, cand_feat):
        # v1.1: weighting + channel dropout (training only)
        cand_feat, seq = self.fwd(cand_feat, seq, self.training)
        score = self.backbone(seq, cand_feat)              # (b, 14) logits
        q_pred = F.softmax(score, dim=-1)                  # (b, 14) prob
        return q_pred, score
```

**FeatureWeightedDropout module** (c5.7 `analysis/plan-024/feature_weighted_dropout.py`):

```python
class FeatureWeightedDropout(nn.Module):
    """v1.1: per-channel learnable scale + channel-wise dropout (LGBM feature_fraction NN 등가).

    - Learnable scale: model 이 중요 channel 자동 강조 (init=1.0, clamp 0.1~10 안정).
    - Channel dropout: redundant channel 전체 zero → model 이 *대체 channel* 학습.
        cand 의 ③ ctx broadcast 128D 만 drop (①②④ 보호). seq 의 J EWMA 9D + WAP 5D
        + Multi-broadcast slice 만 drop (kinematic A/B/C/jerk/ω/F/G/H 보호).
    """
    def __init__(self, cand_dim, seq_dim, cand_drop_p, seq_drop_p,
                 cand_drop_start, cand_drop_end, seq_drop_indices):
        super().__init__()
        self.cand_scale = nn.Parameter(torch.ones(cand_dim))
        self.seq_scale  = nn.Parameter(torch.ones(seq_dim))
        self.cand_drop_p = cand_drop_p
        self.seq_drop_p  = seq_drop_p
        self.cand_drop_start = cand_drop_start
        self.cand_drop_end   = cand_drop_end
        self.register_buffer("seq_drop_indices",
                             torch.tensor(seq_drop_indices, dtype=torch.long))

    def forward(self, cand, seq, training):
        # weighting (always)
        cand = cand * torch.clamp(self.cand_scale, 0.1, 10.0)[None, None, :]
        seq  = seq  * torch.clamp(self.seq_scale,  0.1, 10.0)[None, None, :]
        if training:
            # cand: drop 적용 영역 (③ ctx broadcast 128D) 만
            mask_cand = torch.ones(cand.shape[-1], device=cand.device)
            drop_region = slice(self.cand_drop_start, self.cand_drop_end)
            mask_cand[drop_region] = (torch.rand(self.cand_drop_end - self.cand_drop_start,
                                                device=cand.device) > self.cand_drop_p).float()
            scale_cand = mask_cand.numel() / (mask_cand.sum() + 1e-6)   # inverted dropout
            cand = cand * mask_cand[None, None, :] * scale_cand
            # seq: 특정 index 만 drop
            mask_seq = torch.ones(seq.shape[-1], device=seq.device)
            drop_keep = (torch.rand(self.seq_drop_indices.numel(),
                                    device=seq.device) > self.seq_drop_p).float()
            mask_seq[self.seq_drop_indices] = drop_keep
            scale_seq = mask_seq.numel() / (mask_seq.sum() + 1e-6)
            seq = seq * mask_seq[None, None, :] * scale_seq
        return cand, seq
```

Hyperparam (v1.1, single config, sweep 안 함):

| param | v1 | **v1.1** | 비고 |
|---|---|---|---|
| GRU hidden | 256 | **384** | input dim 62→150D, 89→95D 대비 capacity 보강 |
| GRU layers | 2 | 2 | PB carry |
| GRU dropout | 0.08 | **0.10** | channel dropout 강한 reg → standard dropout 약화 |
| Head MLP dropout | 0.10 (PB) | **0.15** | head input = 384+384+162=930D 폭증 |
| **per-channel learnable scale** | ✗ | **✓** | cand 150 + seq 95 = 245 params, init=1.0, clamp(0.1, 10) |
| **channel dropout (cand ③ ctx 128D)** | ✗ | **p=0.3** | LGBM feature_fraction NN 등가 (Singer LANL 1st pattern) |
| **channel dropout (seq redundant slice)** | ✗ | **p=0.2** | J EWMA 9D + WAP 5D + Multi-broadcast slice 만 |
| AdamW weight_decay | 0.01 | **0.02** | dim 폭증 대응 |
| Adam optimizer | lr=1e-3 | **lr=7e-4** | dropout 강화 + dim 폭증 시 lr 약간 ↓ |
| LR schedule | warm-up 10% + cosine | carry | — |
| pre-epochs | 10 | **12** | regularize 강화 보완 |
| fine-epochs | 8 | **10** | 동일 |
| batch size | 256 | 256 | PB carry |
| seed | 20260521 | 20260521 | plan-024 박제 |
| 5-fold split | `stable_fold_id` MD5 | carry | plan-020/021/022 carry |

### §4.7 loss & target

- **target (output soft label)**: `q_true = build_soft_label_with_tau(Y, R_wfn, pred_F0_world, ANCHORS_A6, τ_cls=0.001)` — plan-022 carry.
- **loss**: soft cross-entropy `-(q_true · log q_pred).sum(axis=k).mean()` (per sample).
- **prediction (final position)**: corrector-free `pred_world = R_wfn @ (q_pred @ ANCHORS_A6) + pred_F0_world`.

---

## §5. STAGE 0 — 인프라 (c2~c8, G0)

### §5.1 모듈 layout

```
analysis/plan-024/
├── __init__.py
├── anchor_vocab.py                # c2 — F/G/H/F2 묶음 builder (v1.1: F3 제거)
├── seq_builder.py                  # c3 — seq 95D per step assembly (v1.1)
├── cand_builder.py                 # c4 — cand_feat 150D per anchor assembly (v1.1)
├── torsion_calc.py                 # c5 — Frenet torsion τ scalar
├── quantile_carry.py               # c5.5 — train fold quantile 박제 (v1.1, fold-leakage 차단)
├── feature_weighted_dropout.py     # c5.7 — per-channel scale + channel dropout (v1.1)
├── multiwindow_trim_build.py       # c5.8 — Multi-window 144→60 trim list 생성 (v1.1, §4.4.1)
├── model.py                        # c6 — CrossAttentionAnchorSelector wrapper (v1.1 hidden=384)
└── run_oof.py                      # c7 — 5-fold OOF runner + metrics

tests/test_plan024_smoke.py        # c8 — 10 pytest (v1.1: weight + channel mask 추가)
```

### §5.2 tests (c8) — v1.1 (10 pytest)

| # | test | assertion |
|---|---|---|
| 1 | module import smoke | **8 module** import 성공 (anchor_vocab / seq_builder / cand_builder / torsion_calc / quantile_carry / feature_weighted_dropout / model / run_oof) |
| 2 | `build_anchor_vocab` shape | (N=4, K=14, T=7) for F/H (v1.1: F3 제거); (N=4, T=7) for G/F2 |
| 3 | sign convention sanity | A6 axis pair (idx 0 ↔ 1) 의 q_past mass 가 잔차 부호 따라 *역대칭* |
| 4 | seq **95D** shape | `seq.shape == (4, 7, 95)`, no NaN |
| 5 | cand **150D** shape | `cand.shape == (4, 14, 150)`, no NaN |
| 6 | torsion mask | random low-curvature trajectory 에서 `valid_mask.float().mean() < 0.5` |
| 7 | **quantile_carry fold-leakage** (v1.1) | 5-fold quantile 박제, test fold quantile 사용 X (assert keys = 5 fold) |
| 8 | **FeatureWeightedDropout weight + mask** (v1.1) | (a) `cand_scale.shape == (150,)` init=1.0, (b) training=True 시 cand 의 ①+②+④ 영역 (0..23 + 152..161) 은 *항상* 동일 값 (보호 영역 drop X), (c) cand ③ 영역 (24..151) 의 mask 가 random Bernoulli(0.7), (d) eval mode 시 mask 없이 scale 만 적용 |
| 9 | model forward smoke | `CrossAttentionAnchorSelector()(seq, cand)` → `(q_pred, score)` shape (4, 14), `q_pred.sum(-1) ≈ 1` |
| 10 | **2-epoch fit** | (a) 모든 epoch loss finite (no NaN/Inf), (b) `train_loss(epoch=1) < train_loss(epoch=0) − 1e-4` (1e-4 margin = numerical noise floor). 의미: backward path 정상 + gradient descent 일관 |

### §5.3 G0 합격

- **10/10 pytest green** (§5.2 의 10 test 모두 pass, c8 박제 일치)
- 모듈 import 시간 < 10s
- model forward inference (b=256) < 200ms (CPU)

---

## §6. STAGE 1+2 — F0 + plan-022 reproduce + cross-attention OOF (c9~c10, G1~G2)

### §6.1 STAGE 1 (c9, G1) — carry reproduce

`analysis/plan-024/baseline_carry.json` 박제:

```json
{
  "dataset_hash": "<sha256 of X+Y>",
  "n_samples": 10000,
  "fold_dist": [2020, 2047, 1921, 2020, 1992],
  "f0_oof_hit_1cm": <float>,
  "f0_oof_hit_1.5cm": <float>,
  "plan022_winner_oof_hit_1cm": <float>,
  "plan022_winner_oof_hit_1.5cm": <float>,
  "plan022_winner_delta_1cm": <float>,
  "plan022_winner_delta_1.5cm": <float>,
  "g1_pass": <bool>
}
```

**G1 합격**: 두 carry value 모두 박제된 carry tolerance 안 (§3.2 표 참고).

### §6.2 STAGE 2 (c10, G2) — cross-attention 5-fold OOF

per fold k (k=0..4):

```python
tr_idx = where(folds != k)
va_idx = where(folds == k)

q_true_tr = build_soft_label_with_tau(Y[tr], R_wfn[tr], pred_F0_world[tr],
                                       ANCHORS_A6, tau_cls=0.001)

# v1.1: build inputs (sign-unified, τ_past=0.003, quantile carry for saccade/peak threshold)
quantile_carry_train = quantile_carry.build(X[tr], pred_F0_world[tr], ANCHORS_A6)
# = {'omega_p90': float, 'jerk_p90': float, 'levy_tail_threshold': float, ...} — train fold only
seq_tr  = build_seq(X[tr], R_wfn[tr], pred_F0_world[tr], ANCHORS_A6,
                    tau_past=0.003, quantile_carry=quantile_carry_train,
                    regimes=regimes[tr])           # (n_tr, 7, 95)  # v1.1
cand_tr = build_cand(X[tr], R_wfn[tr], pred_F0_world[tr], ANCHORS_A6,
                     regimes=regimes[tr], macro_stat=macro_stat[tr],
                     ewma_alphas=(0.1, 0.3, 0.5),
                     multiwindow_trim_path="analysis/plan-024/multiwindow_trim.json",
                     quantile_carry=quantile_carry_train)            # (n_tr, 14, 162)  # v1.1
# valid fold: train fold 의 quantile_carry 적용 (test fold quantile 사용 X, fold-leakage 차단)
seq_va  = build_seq(X[va], ..., quantile_carry=quantile_carry_train)
cand_va = build_cand(X[va], ..., quantile_carry=quantile_carry_train)

# v1.1 hyperparam (§4.6 표와 일치)
model = CrossAttentionAnchorSelector().cuda()
optim = AdamW(model.parameters(), lr=7e-4, weight_decay=0.02)   # v1.1: lr 1e-3 → 7e-4
scheduler = cosine_warmup(optim, warm=10%, total=epochs)         # total = 12 pre + 10 fine = 22

# v1.1: validation early stop (§12.6 carry, train fold 의 마지막 20% holdout)
# *deterministic ordering*: tr 은 sample_id ascending sort (`sorted(tr, key=lambda i: ids[i])`)
# → split key = sample_id 순. random_state 영향 없음 (fold-leakage 차단 + reproducibility).
tr_sorted = sorted(tr, key=lambda i: ids[i])
val_split_idx = int(len(tr_sorted) * 0.8)
tr_train, tr_val = tr_sorted[:val_split_idx], tr_sorted[val_split_idx:]
best_val_loss = float('inf')
best_state = None
patience = 3
no_improve = 0

# pre + fine training (v1.1: 12 + 10 = 22 epoch)
for epoch in range(12 + 10):
    model.train()
    for batch in DataLoader(seq_tr[tr_train], cand_tr[tr_train],
                            q_true_tr[tr_train], batch_size=256):
        q_pred, score = model(batch.seq, batch.cand)
        loss = -(batch.q_true * log(q_pred + 1e-12)).sum(-1).mean()
        loss.backward(); optim.step(); scheduler.step()
    # validation
    model.eval()
    with torch.no_grad():
        q_val, _ = model(seq_tr[tr_val], cand_tr[tr_val])
        val_loss = -(q_true_tr[tr_val] * log(q_val + 1e-12)).sum(-1).mean()
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        no_improve = 0
    else:
        no_improve += 1
        if no_improve >= patience:
            break  # early stop
# load best state for inference
model.load_state_dict(best_state)

# inference
q_pred_va, _ = model(seq_va, cand_va)              # (n_va, 14)
final_frenet_va = q_pred_va @ ANCHORS_A6           # (n_va, 3)
pred_world_va = einsum("nij,nj->ni", R_wfn[va], final_frenet_va) + pred_F0_world[va]
```

OOF concat → metric computation. `analysis/plan-024/results_xattn.json` 박제 (§3.3 metric 전체).

**G2 합격** (§3.2): OOF metric finite + max_class_ratio < 0.95 + hit_1cm ≥ 0.6528.

---

## §7. STAGE 3 — analysis (c11, G3)

### §7.1 results.md 박제 (`analysis/plan-024/results.md`)

11 항목 (plan-022 results.md 형식 carry):

1. Frontmatter (status, best_metric, lb_score, band)
2. § 한 줄 결론
3. § OOF hit table (1cm / 1.5cm / Δ vs F0 / per-fold variance)
4. § distribution-match diagnostics (max_class_ratio / q_true_max / KL / top1_acc / soft_CE / gap_ranking)
5. § per-anchor distribution (cf. plan-022 의 per-anchor 분포 표 carry 형식)
6. § hyperparam summary (single config 박제)
7. § risk findings (plan-009 ranking_loss fail 패턴 비교 + numerical issue)
8. § ablation slot (plan-025 후보 — 본 plan 은 single config 라 ablation 없음)
9. § comparison table (plan-022 winner vs plan-024)
10. § follow-up plan 후보 (plan-025/026/027)
11. § paths & artifacts

### §7.2 per-anchor distribution 박제

`analysis/plan-024/per_anchor_dist.json` — plan-022 의 dist_per_anchor.json (2026-05-21 main session 박제, A6_bcc14_τ001) 와 *동일 schema* 로 plan-024 의 q_pred 분포 박제. 비교 측정 박제 항목:

- q_true_avg (14-vector, plan-022 carry)
- probs_avg (plan-024 새 값)
- delta_per_anchor = probs_avg - q_true_avg
- pred_top1_share (14-vector, plan-024 새)
- group sum (axis 6 vs corner 8)

### §7.3 G3 합격

§3.2 표: hit_1cm ≥ 0.6628 AND hit_1.5cm ≥ 0.8104 AND gap_ranking ≤ 0.04. 부분 미달 시 `xattn_partial_pass` warn.

---

## §8. STAGE 4 — LB submission (c12~c14, G_final)

### §8.1 submission 생성 (c12)

```python
# test set inference (5-fold trained model 평균)
seq_test  = build_seq(X_test, ..., tau_past=0.003)
cand_test = build_cand(X_test, ..., regimes_test)

q_pred_test = mean([model_k(seq_test, cand_test)[0] for k in range(5)], axis=0)
final_frenet_test = q_pred_test @ ANCHORS_A6
pred_world_test = einsum("nij,nj->ni", R_wfn_test, final_frenet_test) + pred_F0_world_test

# submission.csv = sample_submission.csv 의 ID + x, y, z columns
```

### §8.2 LB 회수 (c13)

`dacon-submit` skill 호출 분기 spec (3 branch, §0.5 severe / warn 일관):

| G2 / G3 결과 | 분기 | dacon-submit |
|:--|:--|:--|
| **G2 fail** (xattn_no_improvement, hit_1cm < 0.6528) | submission 생성 안 함 | **skip** (사용자 confirm 도 없음, §0.5 c12 [SKIPPED] 박제 후 G_final 의 LB skip 진입) |
| **G2 pass + G3 partial** (xattn_partial_pass: hit_1cm ∈ [0.6528, 0.6628) OR hit_1.5cm < 0.8104 OR gap_ranking > 0.04) | submission 생성 ✓ | **사용자 confirm 요청** (memory feedback "dacon-submit user approval required" 일치, CV 마진 부족으로 DACON quota 자동 소모 회피) |
| **G2 + G3 모두 pass** (hit_1cm ≥ 0.6628 AND hit_1.5cm ≥ 0.8104 AND gap_ranking ≤ 0.04) | submission 생성 ✓ | **자율 호출** (CLAUDE.md autonomous policy, 사용자 confirm 없이 자동 1회 quota 소모) |

```bash
# 의사코드 (run_oof.py 의 c12/c13 분기 박제):
if g2_pass and g3_pass_all:
    dacon_submit(submission_csv)                # 자동
elif g2_pass and g3_partial:
    user_confirm = ask_user("G3 partial pass — DACON quota 1회 소모 OK?")
    if user_confirm:
        dacon_submit(submission_csv)
    else:
        log_skipped("g3_partial_no_confirm")
else:  # g2_fail
    log_skipped("xattn_no_improvement, submission 생성 skip")
```

`analysis/plan-024/lb_log.md` 박제:
- submission timestamp / file hash / DACON 응답 점수
- 3-file frontmatter `lb_score` 동시 갱신 (top-level plan / results.md pair / analysis/results.md)

### §8.3 G_final 합격 (v1.1, 3-branch 분기)

G_final 의 *최소 합격* = **§0.5 c1~c11 [DONE] AND 3-file frontmatter sync AND follow-up 3건 박제** (G2/G3 결과와 무관 기본). 추가 분기:

| G2 / G3 결과 | c12/c13 | LB | G_final state |
|:--|:--|:--|:--|
| **G2 fail** (xattn_no_improvement) | **c12 [SKIPPED]** + **c13 [SKIPPED]** + decision-note "submission skip 사유: xattn_no_improvement (hit_1cm=<value> < 0.6528)" 박제 | LB 미회수 | **G_final ✓ pass** (band=`negative`, lb_score=null, results.md 의 `g2_no_improvement_skip` 박제, follow-up plan B/C 강화 axis 권고) |
| **G2 pass + G3 partial** (xattn_partial_pass) | c12 [DONE] + c13 [DONE] (사용자 confirm 후) — 사용자 reject 시 c13 [SKIPPED] | LB 박제 또는 null | **G_final ✓ pass** (band=`partial`, lb_score=value or null, `xattn_partial_pass` warn 박제) |
| **G2 + G3 모두 pass** | c12 [DONE] + c13 [DONE] (자율) | LB 박제 | **G_final ✓ pass** (band=`positive`, lb_score=value) |

LB < plan-022 carry (미박제 시 plan-004 LB 0.6806 floor) → `lb_below_floor` warn (severe 아님, G_final 합격 영향 X).

---

## §9. Out of scope (명시적으로 안 함)

| 항목 | 이유 |
|---|---|
| anchor layout 변경 (A6_bcc14 외) | plan-022 §3.4 sweep 결과 carry. plan-024 = architecture lever 만, anchor axis 별 plan-026. |
| anchor radius ≠ 0.005m | plan-022 §3.5 carry. plan-024 = geometric cap 변경 X (Plan B #3 별 plan). |
| τ_cls (output) ≠ 0.001 | plan-022 winner carry. τ_past 만 별도 (§4.2). |
| F0 baseline 변경 | plan-020/021/022 carry. Plan B #4 별 plan. |
| corrector reg head 재투입 | plan-021 finding "reg head dead" carry. selector-only 유지. |
| ensemble (LGBM + cross-attn 평균) | Plan C #5 별 plan-027. |
| ideas.md priority 5 (A1/A6/B3/Multi-Parse/B2) | plan-025 후보. |
| hyperparam sweep (hidden / lr / epoch / batch / τ_past) | single config 고정 (plan-024 의 측정 power = architecture lever 단일 변수). |
| GRU bidirectional / multi-head attention 변경 | PB framework 1:1 carry, architecture 변경 X. |
| 12-step seq length | L2/L4 의 7-step coverage alignment (audit C). |

---

## §10. 작업량 총 회계 (v1.1)

| 항목 | count |
|---|---|
| commit | **16** (c1, **c1.5**, c2~c5, **c5.5, c5.7**, c6~c14) |
| G-gate | 5 (G0, G1, G2, G3, G_final) |
| OOF training | 5-fold × **22 epoch** ≈ **~5~7h GPU** (input dim 3배 + epoch ↑) |
| LB submission | 1회 (DACON quota 1/5) |
| code module | **9 new** (anchor_vocab / seq_builder / cand_builder / torsion_calc / **quantile_carry** / **feature_weighted_dropout** / **multiwindow_trim_build** / model / run_oof) |
| test | **10 pytest** (1 file, v1.1: weight + channel mask 추가) |
| artifact | results.md + results_xattn.json + baseline_carry.json + per_anchor_dist.json + **quantile_carry.json** + lb_log.md |

---

## §11. results.md 필수 항목 (G_final 박제)

§7.1 의 11 항목 + 다음 3 필수 박제:

1. **plan-009 ranking_loss fail 패턴 비교**: plan-024 가 plan-009 의 G1 fail (oof_soft_hit 0.6482, gap_ranking 0.108) 을 *avoid* 했는지 직접 비교 표.
2. **architecture-extractable headroom 측정**: plan-008 gap_ranking 0.0516 의 plan-024 회수율 = (0.0516 - plan-024 gap_ranking) / 0.0516. 100% 회수 시 oracle 도달 (~0.7562 hit).
3. **LB-OOF gap 측정**: plan-004 carry value (self-contained, 외부 source read 불요):
   - plan-004 boundary OOF soft_hit = **0.6624** (`analysis/plan-004/results.md` carry)
   - plan-004 DACON LB = **0.6806** (frontmatter `lb_score` carry)
   - LB-OOF gap = **+0.0182** (positive → generalization 양호)
   plan-024 의 LB-OOF gap 비교 → positive 면 양호, negative 면 overfit 신호 (caveat #8 의 metric drift caveat 동반 인용 필요).

---

## §12. 통계 함정 & caveats

1. **plan-009 ranking_loss G1 fail 가능성**: cross-attention 도 같은 운명 risk. mitigation = G2 의 *최소 동등성* 조건 (hit_1cm ≥ 0.6528). fail 시 architecture lever 자체 *실패* 박제 + Plan B (radius+F0) 또는 Plan C (ensemble) 로 redirect.
2. **single config 측정 의존**: hyperparam sweep 없음. PB framework default (hidden=128 → 256 만 변경) 가 plan-022 carry 위에서 optimum 일 가능성 미검증. **caveat 박제**: 본 plan 의 +0.01 lift 가 architecture 본질 effect 인지 vs hyperparam optimum 의 우연인지 미분리.
3. **torsion τ numerical collapse risk**: muflight low-curvature 비행에서 `collinear_rate > 0.7` 가능 → τ feature 신호 ≈ 0 (mask 영향). G3 통과 후 plan-025 의 ablation 후보로 분리.
4. **sign convention 통일의 backward compatibility**: plan-021 L2 의 `pred - actual` 그대로 import 시 plan-024 의 anchor-vocab encoding 이 silent bug. **반드시 negate**. tests #3 (sign sanity) 로 검증.
5. **regime 18-class one-hot 의 high dim**: cand_feat ③ ctx 18D = anchor 14 × 18 broadcast = 252 element. GRU hidden=**384** 의 capacity 안 OK 단 sparsity high. **v1.1 default = one-hot 18D 박제** (§4.4 ③ ctx 의 "regime 18" 표기 그대로). scalar `regime_idx/18` 대체는 decision-note 자율 변경 가능 단 default 는 one-hot.
6. **GRU hidden 384 의 overfit risk (v1.1)**: N=10k small dataset. PB framework 가 N=10k 환경에서 hidden=128 로 LB 0.6806 도달 → v1.1 hidden=384 (3배) 가 capacity 과잉 가능. mitigation = GRU dropout 0.10 (v1.1) + Head MLP dropout 0.15 (v1.1) + per-channel learnable scale clamp(0.1, 10) + channel dropout (cand ③ 128D p=0.3 + seq redundant slice p=0.2) + AdamW weight_decay 0.02 (v1.1) + early stop (validation loss 기준, train fold 의 마지막 20% holdout deterministic ordering by sample_id, patience=3).
7. **L2 timing alignment ambiguity**: step i=0..6 의 *target time = t = 4..10* (= -160..0ms relative to end). plan-024 의 K (time offset) 묶음에 `target time / 10` 으로 표현 → positional encoding 일관.
8. **LB-OOF gap 의 plan-004 비교 caveat**: plan-004 의 OOF 측정 framework (selector_soft_hit / boundary_soft_hit) 와 plan-024 의 OOF (corrector-free `Σ q · a` 위 hit) 가 *동일 metric 정의 아님*. plan-004 의 +0.018 gap 을 plan-024 에 직접 적용 시 metric drift caveat.
9. **5-fold split 의 same-seed deterministic carry**: `stable_fold_id` MD5 mod 5 → plan-020/021/022 carry exact. plan-024 의 seed 20260521 은 *model init seed* 만 (data split 영향 X).
10. **anchor_vocab encoding 의 sample-anchor row 폭증 아님**: plan-024 = cross-attention (Q · K^T softmax) — sample×K row expansion 안 필요. LGBM 의 sample-weight expansion 과 다른 paradigm. 학습 시간 ↑ 우려는 epoch 단위 (v1.1 ~5~7h GPU).
11. **(v1.1 신규) dimensionality curse**: cand 14×162 = 2268 element + seq 7×95 = 665 element = sample 당 ~2900 element, N=10k → ratio ~3.4 sample/dim. plan-022 LGBM 170D 대비 18배 빡빡. mitigation = channel dropout 0.3 + weight_decay 0.02 + per-channel scale clamp. caveat 박제: G2 fail 시 dim 축소 (Tier A 일부 제외) 후 재학습 fallback.
12. **(v1.1 신규) learnable scale 의 polar drift**: per-channel scale 이 0 또는 매우 큰 값 가는 instability. mitigation = `clamp(0.1, 10)` parameterization 박제 (§4.6 module spec). 학습 중 scale stat (mean/std/min/max) 매 epoch log → `analysis/plan-024/results_xattn.json` 의 `learnable_scale_stat` key 박제.
13. **(v1.1 신규) channel dropout 의 보호 영역 결정**: ①②④ 보호 (geometric prior + fit signal), ③ ctx broadcast 만 drop. seq 의 J EWMA + WAP per-step + Multi-broadcast slice 만 drop. 보호 영역 선택은 *hand-crafted prior* — 잘못된 prior 도 가능. G3 통과 후 plan-025 의 ablation 후보 (예: ④ interactions 도 drop / ② spec 도 drop).
14. **(v1.1 신규) PB framework default dropout 0.08 변경**: GRU dropout 0.08 → 0.10 (channel dropout 강한 reg 보완), Head MLP 0.10 → 0.15. LB 0.6806 carry 의 hyperparam sensitivity 변경 risk. mitigation = baseline_carry.json 의 plan-022 reproduce step 에서 hyperparam variant 박제 (PB carry exact vs v1.1 변형 비교).
15. **(v1.1 신규) Multi-window stat grid 60D 의 정확한 sub-window / stat 결정**: `[전체 11, 뒤 7, 뒤 5, 뒤 3] × [mean, std, slope, max] × 9 channel = 144D, trim 60D` — trim 기준 = correlation > 0.95 column 제거. trim spec 박제 `analysis/plan-024/multiwindow_trim.json`. trim 자체가 hyperparam — single config 고정 (decision-note 박제).
16. **(v1.1 신규) Tier S + Tier A 동시 적용 → ablation 불가능**: 17 항목 동시 추가 → G3 fail 시 어느 lever 가 bottleneck 인지 분해 불가. mitigation = plan-025 의 *항목별 제외 ablation* 으로 후속 분해.
17. **(v1.1 신규) muflight = mosquito 확정의 FE 영향**: saccade binary (S3), wingbeat-jitter (A6), Lévy long-tail (post-G3 후보) 등 mosquito-specific FE 의 효과는 *task 가 정말 mosquito 일 때만* 의미. 만약 drone trajectory 면 saccade 신호 약함. mitigation = G2 후 saccade flag 의 *실제 활성화 비율* 박제 (`fraction(saccade_flag == 1)` 가 0.05~0.30 범위면 mosquito 가정 valid).

---

## §13. 변경 이력

- **v1.1-rev2 (2026-05-21, 사용자 지적 fix)**:
  - **S4 Anchor coord Fourier PE 12D 제거** — 14 BCC discrete token 에 continuous-PE 부적합 (ID one-hot 14D 와 정보량 동등 + expressivity 낮음 + 4πr high-freq term 의 14-sample aliasing 위험). 또한 Learnable anchor embedding (A7, plan-025 후보) 와 같은 axis 의 fixed-basis 약한 버전 → plan-025 lever 측정 오염 회피. 두 lever (S4 fixed Fourier + A7 learnable embedding) 함께 plan-025 로 분리 = head-to-head 비교 가치 보존.
  - **cand_feat dim**: 162D → **150D** (Δ -12).
  - **lever count**: 17 → **16** (S4 제거).
  - **FeatureWeightedDropout** cand_drop_start 24 → **12** (② 21 → 9), cand_drop_end 152 → **140**.
  - **Tier S 합 lift estimate**: ~+0.016 → **~+0.013** (S4 +0.003 제거).
  - **G2 fail vs G_final** spec 영향 없음 (단 G3 envelope 약간 narrower).
- v1.1 (2026-05-21): **4-way ML expert review** 통합 (Trajectory/Kaggle/Physics/Cross-attention). 사용자 결정 옵션 1 (Tier S + Tier A 옵션 B + per-channel weighting + channel dropout + hidden 384). commit chain c1.5 ~ c1.5f.
- v1 (2026-05-21): 초안 작성 (commit `bd1c4cd`). 4-way sub-agent audit (Q1/Q2/Q3/Q4/Q5/Q6) 결과 통합. plan-022 winner + PB framework arch + FE max input (cand 62D + seq 89D, audit 누락 4 family + 강화 form 4 + Frenet basis world-z + torsion + sign 통일 + τ_past 분리) 박제.
- **v1.1 (2026-05-21)**: **4-way ML expert review** (Trajectory prediction / Kaggle-tabular / Physics-informed / Cross-attention-set-prediction) 결과 반영 — 사용자 결정 옵션 1 (Tier S + Tier A 옵션 B + per-channel weighting + channel dropout). 핵심 변화:
  - **input dim 폭증**: cand 62D → **150D** (rev2, +88, ~~Fourier PE 12 제거~~), seq 89D → 95D (+6 net).
  - **Tier S 추가** (cross-confirmed): jerk Frenet (S1, seq +3/step), angular velocity ω Frenet (S2, seq +3/step), saccade binary (S3, seq +2/step), ~~Anchor coord Fourier PE (S4, cand ② +12) — v1.1-rev2 제거~~, Sinusoidal time PE (S5, seq +4/step).
  - **Tier A 옵션 B 추가**: STA/LTA ratio (A1, cand ③ +3), Multi-window stat grid (A2, cand ③ +60), BCC adjacency neighbor pool (A3, cand ④ +2), WAP composite (A5, cand ③ +5 + seq +5), wingbeat-jitter envelope (A6, cand ③ +3), f0_conf (A8, cand ③ +2 + seq +1), anchor-saliency prior (A9, seq +1/step), Pct-rolling+Peak (A10, cand ③ +12), helicity (A11, seq +1/step), v_autocorr multi-lag (A12, cand ③ +3). **제외**: path_signature_L2 (A4, signatory 의존성), Learnable anchor embedding (A7, model parameter axis 다름) — 모두 plan-025 후보.
  - **Redundancy 제거**: F3 anchor-projection 14D (seq), macro_stat straightness 1D (cand ③), axis×forward 1D (cand ④) — 합계 16D 절감.
  - **Regularization 강화**: per-channel learnable scale (cand 150 + seq 95 = 245 params, init=1.0, clamp 0.1~10), channel dropout (cand ③ ctx 128D p=0.3 + seq redundant slice p=0.2). LGBM `feature_fraction=0.3` NN 등가 (Singer LANL 1st pattern).
  - **Hyperparam**: GRU hidden 256→384, GRU dropout 0.08→0.10, Head MLP dropout 0.10→0.15, weight_decay 0.01→0.02, lr 1e-3→7e-4, epochs 10+8→12+10.
  - **muflight = mosquito 확정** (Agent 3 audit 6 단서, frontmatter `inspired_by` 박제). FE 설계 우선순위: mosquito biomechanics > drone framework > general trajectory.
  - **Module 추가**: `quantile_carry.py` (c5.5, fold-leakage 차단), `feature_weighted_dropout.py` (c5.7, v1.1 핵심 module).
  - **test 추가**: 8 → 10 pytest (weight + channel mask 보호 영역 검증).
  - **G2 학습 시간**: 3.5h → ~5~7h GPU (input dim 3배 + epoch +4).
  - **Lift envelope 재추정**: +0.020 ~ +0.050 (cross-correlation discount 적용). G3 +0.01 통과 envelope 의 lower bound 보강.

---

## §14. 참조 (read-only — path blacklist 예외)

### plan reference
- `plans/plan-022-corrector-free-anchor-layout-sweep.md` (winner cell A6_bcc14_τ001 박제)
- `plans/plan-022-corrector-free-anchor-layout-sweep.results.md`
- `plans/plan-021-frenet-corrector-input-augment.md` (input augment L1/L2/L4 + lgbm_extra carry)
- `plans/plan-004-pb-0-6822-fullrun.md` (PB framework arch + CandidateAttentionGRUSelector)
- `plans/plan-004-pb-0-6822-fullrun.results.md` (LB 0.6806 박제)
- `plans/archive/plan-008-candidate-redefine-corrector-redesign.md` (gap_ranking 0.0516 측정)
- `plans/archive/plan-009-selector-ranking-loss.md` (ranking_loss G1 fail 박제)
- `plans/archive/plan-009-selector-ranking-loss.results.md`
- `plans/archive/plan-020-f0-structural-search.md` (F0 baseline carry)

### module reference (read-only)
- `src/io.py` (load_all_samples, load_labels)
- `src/pb_0_6822/selector.py:697-727` (CandidateAttentionGRUSelector arch)
- `src/pb_0_6822/selector.py:259-294` (turn_context_features, motion_terms)
- `src/pb_0_6822/selector.py:361-377` (assign_regimes 18-class)
- `src/pb_0_6822/selector.py:406-449` (make_seq_features carry)
- `analysis/plan-020/baseline_f0.py` (f0_baseline, D1, PAR, PERP)
- `analysis/plan-021/build_input.py` (build_frenet_basis_3d:35-77, to_frenet, build_input_common, build_input_lgbm_extra, _macro_stat_9d:162-188, _ewma_last:194-213, build_soft_label:219-234)
- `analysis/plan-022/anchors.py` (ANCHORS_A6:107-119)
- `analysis/plan-022/selector_only_model.py:45-67` (build_soft_label_with_tau)

### audit reference (2026-05-21 main session)
- 4-way sub-agent audit 결과 (Q1/Q5 F0 residual+torsion, Q2/Q3 world frame z-axis, Q4 plan-004/022 input audit, Q6 ideas.md plan-024 통합) — sub-agent run output 별 박제 (본 plan §1.3 finding 그대로 carry).
- per-anchor distribution 박제 (plan-022 winner A6_bcc14_τ001): `~/.claude/jobs/deea8ff4/dist_per_anchor.json` (2026-05-21 main session, 14 anchor × {q_true / probs / top1 share}).

### external reference
- `notes/ideas.md` (514 lines, priority 5 plan-025 후보)
- `CLAUDE.md` (Autonomous Execution Policy)
- `WORKFLOW.md` §1~12 (plan / results / commit / push 규약)
- memory `~/.claude/projects/-workspace-daycon-muflight/memory/project_next_plan_direction.md` (2026-05-21 사용자 명시 고정)
