# plan-029 paradigm_analysis (c10)

## G3 verdict: **regression** (G2 pass: True)

## Metric

- **hit_1cm** = 0.6316
- **hit_1p5cm** = 0.8039
- max_class_ratio = 0.1328 (1/K = 0.0714, threshold > 0.95 fail)
- top1_acc (argmax vs gt_anchor) = 0.1266

## Paired Δ

- vs F0 (0.6320): Δ = **-0.0004**
- vs plan-022 winner (0.6531): Δ = **-0.0215**  ← G3 임계 0.6528 기준
- vs plan-024 honest ceiling (0.6387): Δ = **-0.0071**
- recall vs 14-anchor oracle (0.7928): 79.7%

## Per-anchor argmax distribution (K=14)

| anchor | ratio |
|---:|---:|
| 0 | 0.0608 |
| 1 | 0.1328 |
| 2 | 0.1119 |
| 3 | 0.1025 |
| 4 | 0.0720 |
| 5 | 0.0780 |
| 6 | 0.0436 |
| 7 | 0.0395 |
| 8 | 0.0494 |
| 9 | 0.0393 |
| 10 | 0.0690 |
| 11 | 0.0660 |
| 12 | 0.0753 |
| 13 | 0.0599 |

## Per-fold gradient norm summary

| fold | ep5 | final | max | min |
|---:|---:|---:|---:|---:|
| 0 | 5.84e-02 | 4.51e-02 | 1.06e-01 | 2.94e-02 |
| 1 | 1.22e-01 | 2.40e-02 | 1.45e-01 | 2.40e-02 |
| 2 | 1.34e-01 | 3.75e-02 | 1.34e-01 | 1.74e-02 |
| 3 | 1.31e-01 | 5.52e-02 | 1.31e-01 | 1.72e-02 |
| 4 | 8.73e-02 | 4.08e-02 | 1.51e-01 | 1.80e-02 |

## Warns

- mode_collapse: False
- mode_collapse_attention: False

## elapsed: 565.3s (9.4 min)

## H4 follow-up
- threshold: cosine off-diag mean < 0.5
- anchor_embed cosine sim matrix 산출은 5-fold model.anchor_embed 박제 필요. 본 c10 은 oof 기반 분석만. G3 PASS 시 follow-up 으로 5-fold model dump.

## Follow-up: plan-030 single-lever ablation 분해 (a/b/c/d 각 단독)
