# plan-029 results

→ 본문 `analysis/plan-029/results.md` 참조.

## 요약 (3 줄)

1. **G3 verdict = regression** (hit_1cm = 0.6316 < F0 0.6320 by 0.0004). 4 lever (a query enrichment + b anchor embedding learnable + c key anchor-conditional + d head raw skip 차단) 동시 적용해도 paradigm framework 자체가 F0 baseline 아래.
2. **attention 학습 자체는 정상** (loss 2.64 → 1.5~1.98, grad_norm 정상, top1_acc 0.1266 = random 0.071 의 1.78×). 사용자 진단 (query sample invariant = plan-024 fail root cause) 의 part 1 (query 학습 가능) ✓, part 2 (hit_1cm 회복) ✗.
3. **Follow-up**: paradigm-distinct lever (F0 ML / corrector / KNN) 전환 권장. plan-030 single-lever ablation 우선순위 낮음 (4 lever 모두 적용해도 regression — 단독 ablation 도 유사 예상).

## 메트릭 (간단)

| | plan-029 X1 | F0 | p024 ceiling | p022 winner |
|---|---:|---:|---:|---:|
| hit_1cm | **0.6316** | 0.6320 | 0.6387 | 0.6531 |
| hit_1p5cm | 0.8039 | 0.8033 | — | 0.8108 |
| Δ vs F0 | **-0.0004** | — | — | — |

## Runtime

- 565.3s = 9.4 min (5-fold OOF, spec 7-15 min 범위 내)
- per-fold: 113-117 s
- total param: 478,745

## See also

- `analysis/plan-029/results.md` (full report)
- `analysis/plan-029/paradigm_analysis.{json,md}` (paired Δ + per-anchor distribution)
- `analysis/plan-029/results_X1.json` (raw metric + hparams + fold_logs)
- `plans/plan-029-grunet-input-max.md` (사전 등록 spec)
