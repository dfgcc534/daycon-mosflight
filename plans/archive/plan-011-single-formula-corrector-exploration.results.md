---
plan_id: 011
plan_title: Single-Formula + Corrector Path Exploration (4-axis breadth ablation)
status: partial (G1 complete, Phase 3+ autonomous skip per §9.3 option a)
date: 2026-05-13 (Asia/Seoul)
best_phase: Phase 1 In axis ID (CNN encoder 64-dim + cf 32-dim)
best_oof_fold0: 0.6450
lb_score: TBD (plan-011.1 carry-over — 할당량 소진 인계)
best_submission: runs/baseline/H012_phase1-input-ablation/sub_ID/submission.csv
g1_pass: false (b condition — 0/4 axes strict positive)
autonomous_branch: option_a_phase3_skip
plan_012_carry_over: true
plan_011_1_carry_over:
  - F3/F4 formula parity fix (cand formula 가 selector.make_candidates 와 numerical 불일치)
  - IC frozen GRU checkpoint 박제 (plan-004 selector.AttnGRUCandidateSelector state_dict load)
  - L axis re-run on In̂=ID anchor (caveat #17 cross-axis bleed 검증)
  - 5-fold OOF reproduce for ID submission
  - submission.csv LB 회수 (dacon-submit)
plan_012_candidates:
  - CNN/transformer encoder 강화 (In axis ID signal 확장, main path)
  - F3/F4 formula parity fix (plan-011.1 carry-over 1순위)
  - KNN/Diffusion paradigm shift (corrector path 의 구조적 제한 인정 후)
detail_results: analysis/plan-011/results.md
detail_attribution: analysis/plan-011/phase1_attribution.md
detail_next_plan: analysis/plan-011/next_plan_candidates.md
---

# plan-011 Results Stub

자세한 결과 → `analysis/plan-011/results.md`.

## 한 줄 요약

Phase 1 24 sub-exp 완료. 0/4 axes strict +0.005 통과 (In axis ID 가장 근접 +0.00495). G1 (b) FAIL → autonomous option a (Phase 3+ skip + G_final 직접 진입). best Phase = In axis ID. LB readout = plan-011.1 carry-over.

**v1.1 post-G_final amendment**: F3/F4 cand formula parity fix 적용. F4 = 0.6431 (+0.0030 vs F0), F axis 진정한 측정 회복. *2 axes (In ID, F F4) sub-threshold positive direction* — P3.1/P3.3 informational 진행 가능성 plan-011.1 carry-over 박제.

**v1.2 post-G_final amendment**: IC (frozen GRU) 활성화 — R001_baseline-residual-gru fold0.pt (2-layer GRU(3,64) same dataset) frozen reuse. spec 32-dim → 64-dim deviation. IC OOF = 0.6446, In̂ = IC (이전 ID 대체). best Phase 갱신 → IC submission.

## 주요 산출

- `analysis/plan-011/preflight.json` — G0 결과 (D001=0.6570 < 0.66, c008 disabled).
- `analysis/plan-011/phase1_{loss,input,arch,formula}_summary.json` — 4 axis summary.
- `analysis/plan-011/phase1_attribution.md` — Phase 1 attribution (10 section).
- `analysis/plan-011/results.md` — 10 section results.
- `analysis/plan-011/next_plan_candidates.md` — plan-012 후보 ≥ 3.
- `runs/baseline/H012_phase1-input-ablation/sub_ID/submission.csv` — best Phase submission.
- `src/pb_0_6822/corrector_redesign_v2.py` — 16 components (smoke 16/16 ✓).
