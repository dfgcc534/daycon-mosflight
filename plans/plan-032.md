---
plan_id: plan-032
status: draft
based_on: plan-031
title: PB training lever 잔여 후보 — plan-030 에 즉시 적용 가능성 (prior 제외)
---

# plan-032 — PB training lever 잔여 후보

## plan-030 에 즉시 적용 가능성 (prior 제외)

| lever | 적용 가능성 | 작업량 | 비고 |
|---|---|---|---|
| **pairwise margin loss** | **즉시 가능 (★★★★★)** | train.py 에 ~15 줄 추가 | model-agnostic. `score (B, K) + soft_label (B, K)` 만 있으면 됨. plan-030 `GRUNetX2` 가 정확히 이 signature. `selector.py:1296-1305` 의 5 줄 코드 그대로 copy. weight 0.25 / margin 0.12 / min_label_gap 0.04 carry. |
| **multi-phase (pre/fine/freeze_fine/epoch_plus)** | **즉시 가능 (★★★★)** | train.py 의 50ep cosine → 단계 4개로 쪼개기. freeze target 1개 결정 필요 | training schedule level. 단 `freeze_fine` 에서 *어느 layer freeze 할지* 정의 필요. PB 는 GRU freeze + head 만 fine 식 — plan-030 에 매핑: GRU + attention K/V projection 동결 + Q projection + head 만 fine 권장. lr schedule: pre lr=0.001 → fine lr=0.00012 (0.12×). |
| **fine-distill (teacher snapshot)** | **즉시 가능 (★★★★★)** | train.py 에 teacher snapshot save/load + KL term 추가, ~20 줄 | pre phase 끝나면 `model.state_dict()` snapshot. fine phase 에서 teacher forward → `softmax(score/temp=0.07)` 를 KL target. `selector.py:1293-1295` 의 식 그대로. |
| **reverse-pretrain** | **부분 가능 (★★)** | pre phase 전에 reverse augmentation epoch 추가 가능, **단 의미 다름** | PB 는 BiLSTM 의 backward direction weight transfer. plan-030 GRU 는 unidirectional → "BiLSTM backward weight transfer" 1:1 적용 불가. 대안 = sequence 를 시간 역순 (`seq[:, ::-1, :]`) 으로 추가 epoch 학습 (data augmentation). effect 약화 예상 (low-medium → low). |
| **batch=4096** | **부분 가능 (★★)** | train.py batch arg 변경 | CPU 환경 메모리 제약. K=14 × 4096 = 57k row × (97D + 64D + ...) ≈ 수십 MB GPU tensor — CPU RAM 으론 batch 1024-2048 권장. *full 4096 효과 미달*. |
| **hidden=48** | **즉시 가능 (★★★)** | model init arg 변경 | 1 줄 변경. 단 plan-030 의 input dim (GRU 97D, Q 64D, head_summary 51D) 대비 hidden 48 은 capacity 부족 위험. **trade-off 불명** (over-fit 방지 vs under-fit). 권장: 잔차 injection 의 신호 풍부함 고려 hidden 96 중간값으로 절충. |
| **norm-real-only** | **자동 충족 (★★★★★)** | 작업 0 | plan-030 augmentation 없음 (§7 deferred). 자동으로 real-only normalization. trivial. |
