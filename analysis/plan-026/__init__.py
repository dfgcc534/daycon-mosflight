"""plan-026 — Block Ablation (plan-025 input 1080D, block ②/③/④ each-out).

3 ablation cells:
  A1 (no block ②, ctx broadcast 128D 제거)   → 952D per row
  A2 (no block ③, per-anchor 22D 제거)         → 1058D per row
  A3 (no block ④, seq 8-stat 760D 제거)        → 320D per row

baseline = plan-025 G2.C1 (carry from results_C1.json).
attribution: drop_X = baseline_hit_1cm - hit_A{X}.
"""
