# plan-006 27-candidate per-formula hit@1cm

train 10,000 samples, end_idx=10, horizon=2

**plan-006 chosen** = `frenet_par120_perp_neg020` (idx 17) → hit = 0.6320, rank = 1/27

| rank | name | hit@1cm | Δ from chosen |
|---|---|---|---|
| 1 | `frenet_par120_perp_neg020` 🔴 | 0.6320 | +0.0000 |
| 2 | `frenet_best` | 0.6296 | -0.0024 |
| 3 | `frenet_par110_perp_neg020` | 0.6284 | -0.0036 |
| 4 | `frenet_par100_perp000` | 0.6283 | -0.0037 |
| 5 | `frenet_par120_perp020` | 0.6280 | -0.0040 |
| 6 | `frenet_par100_perp_neg010` | 0.6276 | -0.0044 |
| 7 | `frenet_par090_perp000` | 0.6259 | -0.0061 |
| 8 | `jerk_small_pos` | 0.6245 | -0.0075 |
| 9 | `frenet_par090_perp020` | 0.6241 | -0.0079 |
| 10 | `jerk_small_neg` | 0.6184 | -0.0136 |
| 11 | `frenet_par070_perp_neg020` | 0.6169 | -0.0151 |
| 12 | `frenet_par080_perp020` | 0.6150 | -0.0170 |
| 13 | `frenet_slow_par100` | 0.6096 | -0.0224 |
| 14 | `frenet_fast_par100` | 0.6078 | -0.0242 |
| 15 | `acc_2d1_060` | 0.6008 | -0.0312 |
| 16 | `acc_2d1_056` | 0.6008 | -0.0312 |
| 17 | `frenet_fast_par120_perp_neg020` | 0.5997 | -0.0323 |
| 18 | `acc_2d1_050` | 0.5993 | -0.0327 |
| 19 | `acc_2d1_040` | 0.5976 | -0.0344 |
| 20 | `p0_2d1` | 0.5787 | -0.0533 |
| 21 | `frenet_slow_par070_perp020` | 0.5732 | -0.0588 |
| 22 | `latency_long_frenet_best_108` | 0.5567 | -0.0753 |
| 23 | `latency_short_frenet_best_092` | 0.5452 | -0.0868 |
| 24 | `latency_long_turn_neg_110` | 0.5173 | -0.1147 |
| 25 | `latency_short_turn_pos_090` | 0.5041 | -0.1279 |
| 26 | `latency_short_frenet_best_085` | 0.4525 | -0.1795 |
| 27 | `latency_long_frenet_best_115` | 0.4323 | -0.1997 |

- mean across 27 = 0.5879
- best individual = 0.6320 (`frenet_par120_perp_neg020`)
- worst individual = 0.4323 (`latency_long_frenet_best_115`)
- std across 27 = 0.0530
