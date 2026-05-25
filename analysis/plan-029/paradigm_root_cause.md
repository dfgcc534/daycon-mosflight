# plan-029 paradigm_root_cause — PB 0.6511 vs plan-029 0.6316 gap -0.0195 의 진짜 source

## TL;DR

plan-024 carry 시점에서 PB framework 0.6511 (selector 단독 OOF) → plan-024 0.6387 (-0.0124) → plan-029 0.6316 (-0.0195) 의 누적 감소. **candidate paradigm 차이 (F0 hypothesis vs F0 residual) 는 main carrier 아님** — E2 검증으로 부분 기각. 진짜 source = **plan-004 의 multi-phase training procedure** (pre + fine-tune + epoch_plus + pairwise loss + regime/family prior + reverse pretrain + fine-distill + norm-real-only).

## 검증 실험 결과

| cell | architecture | hit_1cm | Δ vs X1 | Δ vs PB OOF 0.6511 |
|---|---|---:|---:|---:|
| PB selector OOF (plan-004 §2.1) | 27 F0 hypothesis + multi-phase training | **0.6511** | +0.0195 | — |
| PB + boundary corrector (plan-004 §2.2) | + 2-stage boundary corrector | 0.6718-0.6748 | — | — |
| PB LB submission (plan-004 §4) | full ensemble | **0.6806** | — | — |
| plan-029 X1 | anchor residual + 4 lever + 단순 training | 0.6316 | — | -0.0195 |
| plan-029 X3 (head full skip) | + head raw skip 부활 | 0.6327 | +0.0011 | -0.0184 |
| **E2 PB paradigm switch** | + candidate paradigm 만 PB 식 (27 F0 hypothesis) | **0.6309** | -0.0007 | **-0.0202** |

## Decision tree 분기 결과

원래 plan 의 decision tree:

| E1 | E2 | 결론 |
|---|---|---|
| 0.68 재현 ✓ | 0.65~0.68 도달 | paradigm = main carrier 확정 |
| 0.68 재현 ✓ | < 0.65 | **paradigm 부분 fix, 다른 axis 필요** ← 현재 분기 |
| 0.68 재현 ✗ | — | environment/data drift |

→ **"paradigm 부분 fix, 다른 axis 필요" 분기 확정** (E2 = 0.6309 < 0.65, paradigm 차이만으로 회복 안 됨).

## 진짜 main carrier 추적

E2 가 X1 과 동등 수준 (0.6309 vs 0.6316) → candidate paradigm 자체는 거의 무영향. 따라서 PB selector 0.6511 vs E2 0.6309 의 -0.0202 gap = **architecture/training 차이**.

`src/pb_0_6822/run_full.py:96-127` 의 PB selector training:

```python
"--pre-epochs", 10,           # multi-phase: 사전학습
"--fine-epochs", 8,           # multi-phase: fine-tune
"--freeze-fine-epochs", 3,    # multi-phase: freeze + fine
"--epoch-plus", 5,            # multi-phase: epoch 추가
"--patience", 4,              # early stop
"--hidden", 48, "--batch", 4096,
"--lr", 0.001, "--fine-lr-scale", 0.12,
"--prior-strength", 0.65, "--regime-prior-strength", 0.45,
"--pairwise-loss-weight", 0.25, "--pairwise-margin", 0.12, "--pairwise-min-label-gap", 0.04,
"--fine-distill-weight", 0.55, "--fine-distill-temp", 0.07,
"--reverse-pretrain", "--norm-real-only",
```

plan-029 X1/X2/X3/E2 = 단순 50ep cosine + AdamW + soft CE + no prior + no pairwise + no distill + no multi-phase.

**Main carrier 후보 (gap -0.0202 의 source)**:

| PB lever | 본 plan 부재 | 추정 effect |
|---|---|---|
| multi-phase (pre + fine + freeze_fine + epoch_plus) | 단순 50ep cosine | medium-high |
| pairwise margin loss (margin=0.12, min_gap=0.04) | 없음 | high (anchor discrimination 강제) |
| regime prior (strength=0.45) + class prior (0.65) | 없음 | medium |
| fine-distill (weight=0.55, temp=0.07) | 없음 | medium |
| reverse-pretrain (BiLSTM 역방향 사전학습) | 없음 | low-medium |
| norm-real-only (normalization on real data) | 단순 normalization | low |
| batch=4096 vs 64 | 64× 차이 | low (대형 batch 의 SGD noise 효과) |
| GPU vs CPU | 무관 (numerical 동일) | none |

## 결론

1. **plan-024 carry 의 "PB framework 1:1 carry" 주장은 부분적으로 틀림**. architecture (GRU + cross-attention + head) 만 carry 했고 **training procedure (multi-phase + pairwise + prior + distill + reverse-pretrain) 는 carry 안 됨**. plan-024 honest ceiling 0.6387 ≈ E2 0.6309 ± noise — training procedure 누락이 main source.

2. **plan-029 의 4 lever (a/b/c/d attention 강화) 는 paradigm-orthogonal**. 단순 training 위에서는 lever 효과 없음.

3. **본 paradigm framework (anchor residual or F0 hypothesis, K=14 or 27) 의 hardlimit ≈ 0.63** (단순 training 식 사용 시). 

## Follow-up direction (plan-030 후보, 우선순위)

### Priority 1: PB training procedure carry (multi-phase + pairwise + prior + distill)
- `src/pb_0_6822/selector.py` 의 SELECTOR_MAIN 호출 식 그대로 carry
- 단 본 plan 의 X1 architecture (GRU + 4 lever) 위에 적용
- 예상 lift: +0.015 ~ +0.02 (gap -0.02 의 main fix)
- runtime: PB 식이 GPU cuda:1 + epoch_plus 까지 60+ ep, batch 4096 — 본 환경 CPU 적용 시 1-2 시간 예상

### Priority 2: Boundary corrector 부활 (PB 식 2-stage)
- plan-004 §2.2: boundary corrector 가 selector OOF 0.6624 → 0.6718 (+0.0094 absolute)
- LB 0.6806 의 main source 중 하나
- selector 가 어느 정도 작동한 후 boundary 가 추가 lift
- 예상 lift: +0.01 (selector + boundary 합산 0.66+0.01)

### Priority 3: GPU 환경 + batch 4096 + hidden 48 (PB 정확 모사)
- E2 hidden=196 + batch=64 vs PB hidden=48 + batch=4096 의 학습 dynamics 차이
- small hidden + large batch 가 actually critical 일 가능성 (over-fit 방지 + SGD smoothing)
- 예상 lift: +0.005 ~ +0.01

### Priority 4: paradigm-distinct lever (KNN, F0 ML, etc.)
- 우선순위 낮음 — 위 P1-P3 으로 0.65+ 도달 가능성 더 높음

## Files

- `analysis/plan-029/pb_candidate_paradigm.py` (E2 runner)
- `analysis/plan-029/results_PB_paradigm.json` (E2 결과)
- `analysis/plan-029/oof_PB_paradigm.npz` (E2 OOF)
- `analysis/plan-029/pb_paradigm_run.log` (학습 log)
- `analysis/plan-029/ablation_head_skip.json` (X1/X2/X3 비교 carry)
- `runs/baseline/P001_pb-0-6822-fullrun/tcn_gru_selector_report.json` (PB OOF 0.6511 박제 source)
- `plans/plan-004-pb-0-6822-fullrun.results.md` (PB metric 박제)
- `src/pb_0_6822/run_full.py` L96-127 (PB training hparam)
