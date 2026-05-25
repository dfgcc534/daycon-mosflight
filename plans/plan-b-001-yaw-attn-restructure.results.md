---
plan_id: plan-b-001
finished_at: 2026-05-26T19:15+09:00
status: all_complete
track: B
exp_ids_completed:
  - B001_f0-baseline
  - B002_kalman-baseline
exp_ids_skipped: []
best_exp_id: B001_f0-baseline
g3_oof_hit_1cm: 0.6296
g3_band: FAIL_regression
g0_5_baseline_hit_1cm: {f0: 0.6320, kalman: 0.5964}
lb_score: {kalman_only: 0.6162}   # DACON comp 236716, OOF→LB gap +0.0198; 노트북 주장 0.6452 재현 실패
band: FAIL_regression
---

# plan-b-001 results — Yaw-frame Anchor-selector + Attention Restructure

## §0. 한 줄 결론

**양 arm 모두 G3 FAIL_regression**. best = B001 (F0 arm) OOF hit_1cm = **0.6296** < F0 baseline 0.6320 (Δ **−0.0024**), plan-030 의 0.6294 와 **사실상 동일 (+0.0002)**. → **"후보 frame 의 Frenet degeneracy 가 plan-030 실패의 carrier" 가설 기각**. yaw frame + attention 재구조(잔차b→bias) + softhit + 3-seed + noise/tier3/log1p 의 6-lever 묶음으로도 F0 floor(0.6320)·plan-024 ceiling(0.6387) 회복 실패. `analysis/plan-029/paradigm_root_cause.md` 의 "main carrier = training procedure (PB multi-phase)" 진단을 **corroborate**.

## §0.5 Result Quick Reference

| arm | baseline | OOF hit_1cm | hit_1p5cm | baseline_hit_1cm (G0.5) | Δ vs 자기 baseline | Δ vs F0 0.6320 | band |
|---|---|---|---|---|---|---|---|
| **B001** | F0 | **0.6296** | 0.8016 | 0.6320 | **−0.0024** | −0.0024 | FAIL_regression |
| B002 | Kalman | 0.6077 | 0.7968 | 0.5964 | **+0.0113** | −0.0243 | FAIL_regression |

- 참조점: plan-030 0.6294 · plan-029 X1 0.6316 · F0 0.6320 · plan-024 ceiling 0.6387.
- elapsed: f0 2119s, kalman 2103s (각 5-fold × 3-seed × 50ep, CPU). 합 ~70min.
- N_total=10000, K=14, n_seeds=3.

## §1. exp 별 산출

### B001_f0-baseline (best)
- OOF hit_1cm **0.6296** / hit_1p5cm 0.8016. fold elapsed [433,419,430,419,418]s.
- F0 standalone(G0.5) = 0.6320 → **모델이 baseline 보다 −0.0024 낮음** (selector 보정이 잘 튜닝된 F0 floor 를 못 넘음).
- 산출: `analysis/plan-b-001/results_g3_f0.json` (+ `.npz` oof_pred).

### B002_kalman-baseline
- OOF hit_1cm **0.6077** / hit_1p5cm 0.7968. fold elapsed [440,408,419,416,420]s.
- Kalman standalone(G0.5) = 0.5964 → **모델이 baseline 보다 +0.0113 높음** (약한 baseline 위에선 잔차 selector 가 가치 추가).
- 단 절대값 0.6077 ≪ F0 arm 0.6296 → Kalman baseline swap 은 net-negative.
- 산출: `analysis/plan-b-001/results_g3_kalman.json` (+ `.npz`).

## §2. 분석

### §2.1 가설 기각 — frame/attention axis 는 carrier 아님
- B001 (0.6296) ≈ plan-030 (0.6294, +0.0002). yaw frame(C1) + 잔차b→bias(F1) + KV 정리(F2) + head 축소(F3) + softhit(①) + 3-seed(②) + noise/tier3/log1p(C2/C3/③) 6-lever 묶음의 net lift ≈ **0**.
- → plan-030 의 0.6294 FAIL 은 후보 frame 의 Frenet degeneracy 나 attention 구조 때문이 **아님**. plan-029 `paradigm_root_cause.md` 진단(carrier = **training procedure**, PB multi-phase + pairwise + prior + distill + reverse-pretrain) 을 corroborate. → plan-031 A-track 이 옳은 방향.

### §2.2 G0.5 baseline 모순 증거 — 확정 (단 **train/OOF split 한정**)
- 같은 hit_1cm metric 에서 **F0 0.6320 > Kalman 0.5964** (Δ 0.0356, train/OOF). plan 작성 단계 flag 한 "Kalman standalone < F0" 가 OOF 로 **확정**.
- **split 명확화** (사용자 verify 2026-05-26): 내 Kalman 0.5964 = 노트북 cell-8 `칼만 train R-Hit` assert 와 **diff 0.0000 정확 재현** (kalman_predict full-train 직접 검증). 노트북 §13 표의 **"Kalman LB 0.6452" 는 test set** 수치 — 이 대회는 **train→LB positive gap 이 큼** (Kalman 0.5964→0.6452 +0.0488; sub_09 OOF 0.6612→LB 0.6778 +0.017). 따라서 "Kalman<F0" 는 OOF 한정 결론이고 **LB 거동은 미측정** (양 baseline 모두 OOF 비교라 내부 일관성은 유지; 본 plan 의 모든 hit_1cm 은 OOF). Kalman 의 노이즈-필터 잔차 가설이 selector 위 OOF floor 손실(−0.036)을 상쇄 못 함은 OOF 기준 사실.
- **LB 실측 (2026-05-26 DACON 제출, comp 236716)**: Kalman-only test → **LB 0.6162**. → 우리 OOF→LB gap = **+0.0198** (0.5964→0.6162; 실재하나 작음). **단 노트북 주장 Kalman LB 0.6452 는 재현 실패** — Kalman 은 결정적이고 우리 train(0.5964)이 노트북 cell-8 과 소수점 일치하므로 같은 config 면 LB 도 동일해야 하나 0.6162 관측. → 노트북 §13 의 0.6452 (및 headline 0.6780) 는 self-report 과대 추정 가능성, 신뢰도 보정 필요. 함의: OOF→LB gap (+0.02)은 method 별로 달라 plan-b-001 OOF FAIL(0.6296)의 LB 외삽 불확실.

### §2.3 arm 비대칭 — F0 floor 는 selector 로 못 넘는다
- F0 arm: 모델 < baseline (−0.0024). Kalman arm: 모델 > baseline (+0.0113).
- 해석: 잘 튜닝된 F0(0.6320)는 anchor selector 의 보정 여지가 거의 없음(이미 hit-optimal 외삽). 약한 Kalman(0.5964)은 보정 여지가 커 selector 가 +0.0113 끌어올리나 출발선이 낮아 절대값 미달.
- 함의: anchor 코드북 양자화 천장 + F0 near-optimality 이중 제약. **연속 회귀 fork** (plan-b-001 에서 미채택, AskUserQuestion 시 "anchor 유지" 선택) 재고 여지.

### §2.4 softhit / 3-seed 효과
- softhit(metric-aligned) loss 가 F0 arm 에서 baseline 초과 lift 를 만들지 못함 — soft_CE+softhit 조합이 plan-030 순수 soft_CE(0.6294) 대비 사실상 동률. metric 정렬만으론 anchor selector 천장 돌파 불가.
- 3-seed 는 분산 완화용 — 단일 G3 OOF 만으론 단독 기여 분리 불가 (ablation 미실행).

## §3. 외부 시스템 결과 (LB)
- DACON 제출 **없음** (plan §6 out-of-scope + CV FAIL — `feedback_dacon_submit_confirmation` 정책상 나쁜 CV 자동 제출 금지).

## §4. 특이사항
- NaN/Inf 0건, training divergence 0건, OOM 0건. 양 arm 5-fold 정상 완주.
- 기존 파일 미수정 (신규 analysis/plan-b-001/ + tests/test_planb001_smoke.py 만) → backward-compat 보존.
- 데이터는 main 체크아웃 추출본 symlink (worktree 는 open.zip 만 committed).

## §5. 다음 단계 후보 (enumeration only — local 권한)
1. **plan-031 A-track (PB multi-phase training)** — 본 plan 이 frame/attention/feature 축 null 을 확정해 training-procedure carrier 진단을 강화. 우선순위 최상.
2. **G3 fail ablation** (§0.5 우선순위 F1>C1>①>feature) — 6-lever 중 어느 것이 미세 +/− 인지 1-fold 분리. 단 net≈0 이라 정보가치 낮음.
3. **연속 회귀 fork** — anchor 코드북 양자화 천장 + F0 near-optimality (§2.3) → selector 폐기·연속 회귀(B-track 변종) 검토.
4. **F0 자체 개선** — selector 가 F0 floor 를 못 넘으므로 baseline(F0) 식 재튜닝이 더 직접적일 수 있음.
