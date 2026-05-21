---
plan_id: 024
finished_at: 2026-05-21 (Asia/Seoul)
status: all_complete
band: negative
best_metric:
  hit_1cm: 0.6370
  hit_1.5cm: 0.8092
  delta_1cm: +0.0050
  delta_1.5cm: +0.0059
  gap_ranking: 0.1934
exp_ids_completed:
  - Z024_xattn_anchor_vocab
exp_ids_skipped: []
lb_score: null
xattn_no_improvement: true
g_final_state: g2_no_improvement_skip
---

# plan-024.results pair (WORKFLOW.md §11)

핵심 결과는 `analysis/plan-024/results.md` 의 12 항목에 박제. 본 pair file 은 frontmatter 4-way 토큰 일치 (WORKFLOW.md §4 / §11) 의무 충족용 stub.

## 핵심 결과 요약

- **G2 FAIL**: `xattn_no_improvement` severe — cross-attention GRU + 16 lever FE max (cand 150D + seq 95D + hidden 384) 가 plan-022 LGBM winner 보다 −0.0158 below (OOF hit_1cm = 0.6370 vs 0.6528).
- **G_final pass band=negative**: §0.5 / §8.3 분기 — c12/c13 [SKIPPED], LB 미회수, follow-up plan-025/026/027 박제.
- **gap_ranking 0.1934** — plan-009 ranking_loss fail (0.108) 의 1.8× 악화. architecture lever 자체 *underperform*.
- **CPU under-converged 의심**: 학습 167s (spec §10 GPU 5-7h 가정의 100× 빠름) — plan-025 ablation 의 *최우선 변수* = 충분 학습 시간 + epoch 강제.
- **14-anchor oracle = 0.7928** — plan-024 framework 의 상한, plan-022 LGBM 의 0.6528 + plan-008 carry 27-cand 의 0.7562 보다 모두 높음. selector 가 oracle 의 80.4% 만 회수.

## 상세 분석 위치

- `analysis/plan-024/results.md` — 12 항목 G_final 종합 (fail mode 7 가설 + ablation slot + follow-up plan-025/026/027)
- `analysis/plan-024/results_xattn.json` — G2 OOF metric (167s elapsed)
- `analysis/plan-024/baseline_carry.json` — G1 carry pass

## follow-up plan 후보

- **plan-025**: ablation + CPU under-converged fix + hyperparam mini-sweep + path_signature_L2 / Learnable anchor embedding head-to-head
- **plan-026**: anchor radius 확장 (0.5cm → 0.7~1.0cm, oracle 0.7928 → 0.85+ 추정) + F0 baseline ML 화
- **plan-027**: ensemble (plan-022 LGBM + plan-024/025 best variant)
- **plan-028** (가칭): ideas.md ★★★ paradigm shift (Trajectron-CLIP / KNN pool / MDN)
