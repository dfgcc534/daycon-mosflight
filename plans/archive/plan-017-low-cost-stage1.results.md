---
plan_id: 017
version: 1.5 (results synthesis)
date: 2026-05-15 (Asia/Seoul)
status: G_final_complete (G0 4/4 pass / G1 pass marginal / G2 fail negative_drop, paradigm-shift 결정점 도달)
based_on:
  - 013 (LB unsubmitted, OOF 0.6381)
  - 014 (LB 0.6628, OOF 0.6425)
  - 015 (= 014 deterministic carry)
  - 016 (LB 0.6638 G1, OOF 0.6452 — best baseline)
followed_by:
  - 018 (parallel work — arch ablation single-model, samdasuu commit b2e1f8c)
  - paradigm-shift 결정점 사용자 confirm 후 추가 plan
scope: low-cost stage 1 batch (ensemble + voxel CE) measured. G1 ensemble marginal positive (+0.0002 LB), G2 voxel CE clear fail (-0.0121 OOF). §3.1 mixed case rule: paradigm-shift #1 (plan-004 2-stage) 권장 — 단 parallel plan-018 (arch ablation) 이 이미 user-launched 인 점 박제.
exp_ids:
  - H057_g0_preflight
  - H058_g1_ensemble
  - H059_g2_voxel_ce
  - H060_g_final_synthesis
lb_score: 0.6640         # G1 ensemble (best of plan-017)
lb_band: positive
band: positive
best_5fold_oof: 0.6452   # plan-016 G1 carry (plan-017 G2 OOF lower)
delta_oof: 0.0000        # G2 didn't improve, baseline plan-016 G1 OOF retained
oof_lb_gap: 0.0188       # G1 ensemble LB - plan-016 G1 OOF (not strict but anchor)
dacon_submits_used: 3    # today (plan-016 G1+G2 + plan-017 G1)
---

# plan-017 v1.4 — Results (band=positive, paradigm-shift 결정점)

## §1. G0~G2 결과 narrative

### G0 — preflight (H057_g0_preflight, e747d69)

4 task 4/4 PASS (단 (d) caveat):
- (a) 3 submission file 존재 + schema (id,x,y,z) + 10000 row.
- (b) plan-016 G1 baseline artifact 정확 일치 (OOF=0.6452, LB=0.6638).
- (c) Voxel CE smoke: logits (16, 343), loss=5.85, backward OK.
- (d) **Voxel coverage measured FAIL trigger** — ±2cm coverage = 85.4% < 90% threshold.
  → Spec v1.4 patch: 5×5×5 (125 voxel) → **7×7×7 (343 voxel)**, ±3cm window (coverage 90.24% caveat zone).

### G1 — Ensemble (H058_g1_ensemble, d7c0413)

3-plan submission 좌표 mean:
- plan_013/submission.csv (OOF 0.6381, LB unknown)
- plan_014_g5_phase4/submission_best.csv (LB 0.6628)
- plan_016_g1_path_a/submission.csv (LB 0.6638)

| metric | value |
|---|---|
| per-source mean dist to mean (m) | plan_013: 0.00175 / plan_014_15: 0.00122 / plan_016_g1: 0.00089 |
| LB measured | **0.6640** |
| Δ vs baseline 0.6638 | **+0.0002** |
| Pass (Δ ≥ 0) | **TRUE (marginal positive)** |
| 가설 ε ~ +0.005 | **falsified** (measured +0.0002 << +0.005) |

해석: framework-disjoint 결합 가설 *부분 검증*. variance reduction lemma 의 ε=+0.005 미달, 그러나 direction 은 positive. 3-plan submission 모두 *highly correlated* — plan-013 가 가장 divergent 였으나 (mean dist 0.18cm) ensemble 효과 marginal. ensemble 단독 paradigm 으로는 paradigm-shift cost 낮춤 효과 미미.

### G2 — Voxel CE Head (H059_g2_voxel_ce, d7c0413)

7×7×7 voxel CE head, 5-seed × 5-fold = 25 models, plan-016 G1 carry config + head 교체 + anchor 무력화.

| metric | value |
|---|---|
| per-seed concat OOF | [0.6329, 0.6328, 0.6324, 0.6325, 0.6323] (매우 consistent across seeds) |
| per-fold (seed-mean) | {0: 0.6517, 1: 0.6273, 2: 0.6427, 3: 0.6278, 4: 0.6159} |
| **multi-seed concat OOF** | **0.6331** |
| Δ vs G1 baseline 0.6452 | **-0.0121** (negative direction) |
| Δ pass (≥ +0.003) | **FALSE** |
| status | **negative_drop** |
| LB | unsubmitted (사용자 결정, quota 보존) |
| training time | 276s (4.6 min) for 25 models |

epoch 분포: 다수 fold 에서 early-stop epoch=1 (val_hit 초기값 vs 학습 후 improvement 없음). voxel CE 가설 (hit metric 직접 정렬) *반례 증거 산출*.

해석:
1. **paradigm 자체 한계 measured** — corrector regression 보다 voxel CE classification 이 *오히려 정보 손실*. 1cm voxel width discretization 이 sub-cm 정확도 학습을 방해.
2. **confound 3종**:
   - 9.76% sample 이 ±3cm window 밖 (clamp 효과, 그 sample 들 voxel argmax 정답 학습 불가).
   - 10000 / 343 = 29 sample/class 학습 부족 (class 수 너무 많음).
   - anchor codebook (K=9) 무력화 — plan-014 G2 measured contribution +0.0066 OOF 손실.
3. **per-seed 매우 consistent** (0.6323-0.6329 range, std 0.0003) → seed variance 아닌 paradigm 한계. 5-seed ensemble 도 paradigm ceiling break 못 함.

## §2. 합격 기준 verdict — §3.1 mixed case 적용

| stage | Pass | direction |
|---|---|---|
| G1 ensemble | **PASS (marginal)** | positive (+0.0002 LB) |
| G2 voxel CE | **FAIL (negative_drop)** | negative (-0.0121 OOF) |

§3.1 mixed case rule (v1.2 박제):
> **G1 pass + G2 fail** → ensemble path 유망, voxel CE head 만 한계 — **paradigm-shift #1 (plan-004 2-stage) 권장** (head 자체보다 architecture 변경).

## §3. Premise verdict — 두 가설 부분 falsified

**§1.1 ensemble premise**:
> framework-disjoint 3 plan 좌표 mean 으로 ε ~ +0.005 LB 회수.

Verdict: **부분 falsified**. direction positive 였으나 ε measured = +0.0002 << +0.005. 3 submission 의 framework-disjointness 가 가정보다 약함 (모두 corrector paradigm 또는 plan-004 simplified family — *fundamentally similar prediction*).

**§1.2 voxel CE premise**:
> 1cm voxel CE classification 이 corrector regression 대비 +0.003~0.005 OOF 회수 (hit metric 직접 정렬).

Verdict: **falsified, opposite direction**. measured ΔOOF = -0.0121 (12배 정도 반대 방향). 1cm voxel width 가 hit threshold 와 align 하지만 *학습 동역학* 측면에서 regression 보다 약함. corrector paradigm 안에서 head reformation 만으로는 ceiling break 불가 확정.

## §4. paradigm-shift 결정점 (§7.3 anchor + 추가 context)

### §4.1 §7.3 anchor 결과별 권장

본 plan G1+G2 결과 매핑:
- G1 pass + G2 fail → **paradigm-shift #1 (plan-004 2-stage corrector) 권장**.
- 사유: head replacement 만으로 paradigm 한계 못 깬 → architecture-level change (selector + boundary corrector 분리) 필요.

### §4.2 추가 context — parallel plan-018 (samdasuu commit b2e1f8c)

본 plan G_final 작성 시점에 user 가 *별도 session 으로* plan-018 시작 박제 확인:
- plan-018 = arch ablation single-model (plan-007 step 4 per-sample coefficient + GRU-attn / 5 arch ablation).
- G_final target: LB ≥ 0.67 (= plan-004 ensemble 0.6822 의 95%).

plan-018 paradigm 은 §7.3 의 후보 A (#1 plan-004 2-stage) 와 후보 B (#2+#3 CLIP+regime) 와 *다른 axis* — arch ablation 가 plan-007 spirit carry + encoder/head 6 variant 비교 (A0 baseline + 5 ablation).

본 plan-017 의 paradigm-shift 결정점 = **사용자 위임**:
- 후보 A: #1 plan-004 2-stage corrector (§7.3 권장).
- 후보 B: #2+#3 Trajectory-CLIP + 486-entry regime bias.
- 후보 C: 기타 (plan-016 §11.4 후보).
- 후보 D: **plan-018 (arch ablation, 이미 user-launched)** 의 결과 우선 확보 후 paradigm-shift 결정.

## §5. measured 값 박제 (외부 reference)

| measure | value | source |
|---|---|---|
| F0 raw hit@1cm (plan-006 frozen reproduce) | 0.6320 | plan-014 G0 |
| plan-014/015 best_stack 5-fold OOF | 0.6425 | plan-014 G5 |
| plan-014/015 best_stack LB | 0.6628 | plan-014 LB |
| plan-016 G1 5-fold OOF (multi-seed) | 0.6452 | plan-016 G1 |
| plan-016 G1 LB | 0.6638 | plan-016 G1 |
| **plan-017 G1 ensemble LB** | **0.6640** ★ (+0.0002 marginal) | plan-017 G1 |
| plan-017 G2 voxel CE 5-fold OOF | 0.6331 | plan-017 G2 (negative_drop) |
| oracle ceiling (E0b Frenet-ortho, hindsight) | 0.8248 | plan-014 G0 |
| Voxel grid ±3cm coverage (plan-014 F0 carry) | 90.24% | plan-017 G0 |
| Voxel grid ±2cm coverage (initial spec v1.0-1.3) | 85.4% (FAIL trigger) | plan-017 G0 |
| F0 → Y dist p95 | 5.16cm | plan-017 G0 |
| F0 → Y dist p99 | 9.85cm | plan-017 G0 |

→ **plan-017 의 measured ceiling = G1 ensemble LB 0.6640** (+0.0002 vs plan-016 G1).
→ corrector paradigm + head reformation paradigm 모두 0.66 plateau.

## §6. LB carry-over

- **plan-017 G1 ensemble submission = LB 0.6640**. plan-014/015/016 LB 0.6628~0.6638 위 +0.0002~0.0012.
- DACON quota used today: **3 / 5** (plan-016 G1 + G2 + plan-017 G1). 남은 quota 2 = paradigm-shift 후속 plan (#1/#2+#3 등) 검정 자원.
- G2 voxel CE submission 산출 완료 (`runs/baseline/plan017_g2_voxel_ce/submission.csv`) — 추후 ensemble 추가 멤버 또는 plan-018 비교용 carry 가능.

## §7. 종료

- G_final 합격 (3 파일 sync):
  - results.md 신규 (본 파일) ✓
  - plan-017 frontmatter sync (status / lb_score / band / followed_by) ★ 별도 commit
  - registry append H060_g_final_synthesis ★ 별도 commit (본 commit)
- §3.1 mixed case rule per paradigm-shift 권장 박제: ✓ (#1 plan-004 2-stage)
- band 분류: **positive** (LB 0.6640 ∈ [0.66, 0.68])
- DACON quota: 2/5 남음
- paradigm-shift 결정점 = **사용자 위임** (후보 A/B/C/D 박제)
