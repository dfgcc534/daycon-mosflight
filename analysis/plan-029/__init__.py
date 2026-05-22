"""plan-029 — GRU-attention Input Max (hidden=196, 4 lever).

modules:
  anchor_query_extend  : plan-024 cand_builder 150D + sample×anchor interaction 15ch = 165 (lever a)
  model                : GRUNetX1 — 4 lever (a/b/c/d) 통합 attention model
  train                : 5-fold OOF training loop
  run_oof              : orchestrator + G1 + metric
"""
