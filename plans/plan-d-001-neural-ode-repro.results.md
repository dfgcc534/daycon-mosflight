---
plan_id: d-001
version: 1
date: 2026-05-26 (Asia/Seoul)
status: all_complete
lane: d
exp_ids:
  - NODE001_notebook-repro
g_repro_oof_hit_1cm: 0.6330
g_repro_band: PASS
node001_hit_1p5cm: 0.8032
node001_delta_vs_f0: 0.0010
node001_perm_p: 0.7923
overfit_peak_ep3_fold0: 0.6663
---

# plan-d-001 results — Neural ODE 쇼케이스 노트북 재현 (NODE001)

## 결론 (한 줄)

> **NODE001 (Neural ODE 노트북 충실 재현) OOF hit_1cm = 0.6330 — G_repro PASS (F0 floor 0.6320 바로 위)**, 단 **vs F0 Δ+0.0010 p=0.79 = 통계적 동률**. 잔차 paradigm (c-001 F0-GRU 0.6622) 에는 미달 (H2 FAIL). final-epoch(15) overfitting 으로 fold0 peak 0.6663(ep3)→0.6307(ep15) 퇴행이 OOF 를 floor 로 끌어내림 — *학습 물리 paradigm 자체의 잠재력(≈0.66)은 early-stop 시 가용하나, 노트북 spec 그대로는 floor 수렴*.

## 수치

| metric | 값 |
|---|---|
| **NODE001 OOF hit_1cm** | **0.6330** (band PASS) |
| NODE001 OOF hit_1p5cm | 0.8032 |
| F0 floor hit_1cm | 0.6320 |
| Δ (NODE001 − F0) | **+0.0010** |
| paired permutation p (vs F0, 10k) | **0.7923** (동률) |
| fold final val_hit (0~4) | 63.07 / 62.97 / 64.13 / 62.82 / 63.55 % |
| fold0 epoch-curve peak (ep3) | **0.6663** → ep15 0.6307 (overfitting) |
| mean err (m) | (results_node001.json) |

비교 floor: F0 0.6320 · c-001 F0-잔차 0.6622 · a-001 KR002 0.6639 · LB record 0.6854.

## Gate 판정

| gate | 결과 |
|---|---|
| G0 (smoke) | ✅ frame self-check + smoke 3 pass + runner --smoke fold0 1ep 63.02% finite |
| G1 (1-fold full-ep) | ✅ PASS — fold0 ep1 63.02% → final 63.07% (final≥ep1, 비단조 noise 허용) |
| G_repro (full OOF) | ✅ **PASS** (0.6330 ≥ 0.6320). STRONG(0.6622) 미달, EXCELLENT(0.6854) 미달 |
| G_final | ✅ results 박제 + §0.5 sync + main merge |

## 가설 verdict

- **H1 (재현 ≥ F0 0.6320)**: ✅ PASS (0.6330) — *겨우*. 노트북 자칭 "LB 0.6+" 와 floor 수준 정합. 이식 버그·transfer 실패는 아님 (finite, NaN/Inf 0).
- **H2 (paradigm 대조 ≥ 잔차 c-001 0.6622)**: ❌ FAIL (0.6330 ≪ 0.6622). **순수 학습-물리(Neural ODE)는 이 프로젝트에서 "고정 물리 + GRU 잔차 보정" paradigm 에 못 미친다** — final-epoch 기준. plan-020 의 "고정 물리(CTRA/CTRV/Singer) < F0" 에 이어, "학습 물리 ≈ F0 < 잔차" 로 paradigm 서열 확장.
- **H3 (§S2 반증/확증)**: 부분 반증 — OOF 0.6330 ≥ 0.6320 이므로 ideas.md §S2 "Skip 확정"(regular grid·underdetermined·급선회)은 *F0 floor 도달 가능성을 과소평가*. 단 floor 를 의미있게 넘지 못해(p=0.79 동률) §S2 의 회의도 *방향은* 맞음. plan-020 "미시도 ★★" 카드 = **실측 종결: floor 도달하나 잔차 paradigm 미달**.

## 핵심 finding

1. **Overfitting 이 결과를 지배** — fold0 val_hit 가 ep3 0.6663 peak 후 ep15 0.6307 로 단조 퇴행. epochs=15(노트북) + no early-stop + 1-seed 조합이 OOF 를 peak 의 ~95% 로 깎음. **early-stop/epoch 축소 시 ≈0.66 가용** (잔차 paradigm 동급) — 단 §6 out-of-scope (arch sweep).
2. **학습 물리 > 고정 물리** — Neural ODE(0.6330) 가 plan-020 CTRA(0.5070)/CTRV(0.5207)/Singer(0.5951) 를 크게 상회. "힘 법칙을 손으로 고르지 말고 학습" 방향성은 검증됨 (F0 floor 도달).
3. **그러나 잔차 paradigm 미달** — F0 닫힌해(0.6320)+GRU 잔차(c-001 0.6622, a-001 0.6639)의 "물리 뼈대 + 신경망 보정" 분업이 "물리 자체 학습"보다 이 데이터·budget 에서 우세. 현 LB record(KR008 0.6862)와의 격차 큼.
4. **재현 충실성**: model/loss 상수 cell8/10 그대로, extract_features 24D 는 markdown 의미 재구성(byte-exact 불가, §4.2 dim→식 표 박제 — `results_node001.json` 의 DIM_TABLE).

## 다음 방향 (참고, 본 plan out-of-scope)

- **early-stop / epoch 축소** — OOF=final 대신 best-epoch (peak 0.666) 회수 시 잔차 paradigm 동급 가능. 가장 ROI 높은 단일 lever.
- ensemble (multi-seed/multi-config), latent_dim·damping init sweep.
- 잔차화: Neural ODE 를 F0 잔차 위에 학습 (paradigm 융합).

## artifact

- `analysis/plan-d-001/{frame,features,model,losses,run_oof}.py`
- `analysis/plan-d-001/results_node001.json` (DIM_TABLE 포함) + `.npz` (oof_pred/y/err/fold_id)
- `analysis/plan-d-001/g1_node001.json` (epoch curve)
- `tests/test_plan_d001_smoke.py`

decision-note: NODE001 OOF 0.6330 PASS(겨우) — 노트북 spec(epochs15/final-epoch/1-seed) 충실 재현 결과. overfitting peak 0.666 은 early-stop(out-of-scope) 시 가용. DACON LB 미제출 (OOF-only, §6 + quota confirm 정책). H2 FAIL = 학습 물리 < 잔차 paradigm (이 프로젝트 paradigm 서열 박제).
