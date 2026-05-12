---
plan_id: 006
based_on:
  - 004
  - 005
finished_at: 2026-05-12T11:26:49+09:00
status: all_complete
exp_ids_completed:
  - E001_minimal-variant-e
lb_exp_id: E001_minimal-variant-e
lb_score: 0.6692
lb_submitted_at: 2026-05-12T11:13:37+09:00
lb_recovered_at: 2026-05-12T11:26:49+09:00
---

# plan-006 results — Minimal Variant E LB Validation

## §1. 한 문장 결론

> plan-005 의 통찰 "PB framework 의 진짜 엔진은 *27 후보 + physics_bias + soft averaging* 뿐" 을 OOF=**0.6524** + LB=**0.6692** 로 직접 검증. plan §0 명제 1 (`lb_score ≥ 0.6606`) **입증 ✓**. full LB 0.6806 대비 -1.14pp 손실로 95% framework 단순화 가능 — **시나리오 A** 채택.

## §2. 핵심 수치

### §2.1 OOF 비교 (plan-005 측정 + 본 plan 측정)

| Variant | OOF hit (soft) | 출처 |
|---|---|---|
| full (GRU + physics + regime) | 0.6599 | plan-005 측정 |
| Variant A (GRU + physics, no regime) | 0.6570 | plan-005 측정 |
| Variant B (physics + regime, no GRU) | 0.6547 | plan-005 측정 |
| **Variant E (physics 만, no GRU/regime)** | **0.6524** | **plan-006 측정** |
| Variant E (raw cands, no corrector) | 0.6250 | plan-006 informational |
| Variant F (uniform, no physics) | 0.6520 | plan-006 sanity check |

### §2.2 Component contribution (OOF 단위)

| Component | Δ OOF |
|---|---|
| corrector (E_raw → E_corrected) | **+0.0274** |
| regime (E → B) | +0.0023 |
| GRU (B → full) | +0.0052 |
| physics_bias (F → E) | **+0.0004** (!) |

**놀라움**: `physics_bias` 의 marginal contribution 이 **+0.0004** (4 hits) 에 불과. `uniform centroid` (F) 만으로도 0.6520 hit — 27 후보의 *기하학적 centroid* 자체가 이미 거의 최적에 가까움. physics_bias 는 *4 hits* 만큼 미세 조정.

→ corrector 가 main lift (`+0.0274`), 다른 component (physics/regime/GRU) 는 *finishing touch* 수준. plan-005 의 "GRU/regime 은 장식" 통찰을 *수치로* 확인.

### §2.3 Per-regime hit (Variant E_corrected)

| regime | n | hit (soft) | regime | n | hit (soft) |
|---|---|---|---|---|---|
| 1 | 629 | **0.9300** | 9 | 562 | 0.7153 |
| 3 | 458 | 0.8865 | 12 | 549 | 0.6120 |
| 0 | 661 | 0.8835 | 4 | 615 | 0.5967 |
| 2 | 663 | 0.8597 | 13 | 916 | 0.5721 |
| 6 | 544 | 0.7978 | 14 | 476 | 0.5105 |
| 7 | 701 | 0.7789 | 15 | 749 | 0.4579 |
| 8 | 592 | 0.7466 | 11 | 355 | 0.4338 |
| 5 | 274 | 0.7080 | 10 | 546 | 0.4103 |
|   |   |   | 17 | 356 | 0.2584 |
|   |   |   | **16** | **354** | **0.2203** |

n 합 = 10000 (assert 통과). worst 3 regime: 16 (0.2203), 17 (0.2584), 10 (0.4103) — plan-005 의 corrector 가 회복하지 못한 high-error 영역과 일치 추정.

### §2.4 physics_bias 해석

- argmax 후보: `frenet_par120_perp_neg020` (idx=17)
- top-5: `frenet_par120_perp_neg020`, `frenet_best`, `frenet_par100_perp000`, `frenet_par120_perp020`, `frenet_par110_perp_neg020`
- 모두 `frenet_*` family (par 100~120% × perp -20~20% 의 8 변형 + best) — 의미: "현재 속도 그대로 20 step 진행한 frenet 좌표 + 약간의 lateral shift". 27 후보 중 *speed extrapolation* 류만 hot.

## §3. plan-005 통찰 LB 입증 — 시나리오 A 채택

**LB lb_score = 0.6692** (2026-05-12 11:26:49 KST 회수).

- 명제 1: `lb_score ≥ 0.6606` (inclusive) → **0.6692 ≥ 0.6606 ✓ 입증**.
- 명제 2: `lb_score < 0.6606` (strict) → 미발동.
- 명제 3: 결과 anchor — 시나리오 A 채택.

severe trigger 점검:
- `lb_anomaly` (`|x − 0.6806| ≥ 0.05`): `|0.6692 − 0.6806| = 0.0114` < 0.05 → 미발동 ✓.
- nominal caveat `[0.62, 0.72]` (§N+3 caveat 7): 0.6692 ∈ [0.62, 0.72] → 미발동 ✓.

## §4. plan-004 full LB (0.6806) 와 비교 (측정)

| Metric | Variant E (plan-006) | full (plan-004) | Δ |
|---|---|---|---|
| OOF (soft) | 0.6524 | 0.6599 | -0.0075 |
| LB | **0.6692** | 0.6806 | **-0.0114** |
| OOF→LB gap | +0.0168 | +0.0207 | -0.0039 |

핵심 관찰:
- **gap variant 간 일관**: Variant E gap (+0.0168) 와 full gap (+0.0207) 의 차이는 -0.0039 (≈ 4 hits) — 거의 동일. plan-005 의 "OOF→LB gap 은 variant 무관" 추정이 검증됨.
- **GRU + regime 의 LB 기여 ≈ +0.0114pp** (LB 단위). plan-005 의 OOF 기여 측정 +0.0075pp 와 거의 동일 — *out-of-sample* 에서 GRU/regime 이 *추가* 손실을 막지 못함. 즉 GRU + regime 의 LB 기여 ≈ OOF 기여 + noise.
- → 후속 plan-007 = **시나리오 A 우선**: 후보 다양화 (A1) + corrector 재설계 (A2).

## §5. Decision-note 박제

- `decision-note: spec-default — boundary.soft_select 는 실제로 selector.soft_select (plan-006 §4.1 L222 spec 의 module path 오기 정정). API: temperature 인자명.`
- `decision-note: spec-default — exp_id=E001_minimal-variant-e (E prefix = Experimental simplification)`
- `decision-note: spec-default — PHYSICS_WEIGHT=0.65, SOFT_TEMP=0.03 (plan-004 default 답습)`
- `decision-note: spec-default — Variant F (uniform) sanity check 는 OOF informational only, LB 제출 X`
- `decision-note: spec-default — LB 제출 1회 (Variant E_corrected 만). CLAUDE.md autonomous policy 가 dacon-submit skill 의 confirm prompt 를 override.`
- `decision-note: spec-default — corrected_*.npz 가 local 에 있음 → fallback 재생성 path 미발동.`
- `decision-note: partial — dacon 응답 (isSubmitted=True, detail=Success) 에 lb_score 부재 → plan §7.2 (True, None) 분기 = TBD carry-over. 점수 회수는 user-driven (DACON dashboard).`

## §5.5 추가 통찰 — "1 후보 고정 픽" 만으로도 LB 의 95%

Variant E 의 argmax 측정값을 다시 보면:

| 방식 | hit (OOF) | 의미 |
|---|---|---|
| **E_corrected argmax** | **0.6491** | physics_bias 1등 후보 (`frenet_par120_perp_neg020`) 의 corrected 좌표를 **모든 sample 에 동일하게** 답으로 제시 |
| E_corrected soft | 0.6524 | 위 + 27 후보 가중평균 보간 |
| LB (soft 제출) | 0.6692 | OOF → LB gap +0.0168 |

→ **argmax 만으로 64.91%**. score 가 sample 무관 상수이므로 argmax 도 sample 무관 → "27 후보 중 train 에서 가장 자주 적중한 단 1 개 공식" 의 좌표를 모든 모기에 적용하는 zero-ML baseline.

```python
# 이 3 줄로 OOF 0.6491 (LB 추정 ~0.66) 재현
best_cand = train_hit_rate.argmax()             # train 1등 후보 1개
pred      = corrected_test[:, best_cand, :]     # 모든 sample 동일 픽
```

함의:
- 모기 비행 데이터의 **64.91% 는 단일 물리 공식 1 개 + corrector 좌표 보정** 만으로 풀린다
- `frenet_par120_perp_neg020` 형태의 "현재 속도 ×1.2 외삽 + 약간의 lateral shift" 공식 1 개가 dominant
- argmax → soft 의 회수 = +0.33pp, GRU + regime 의 회수 = +1.14pp (LB 단위)
- → **27 후보 풀의 1 등 후보 자체가 본 문제의 main lever**. 나머지 26 후보 + 어텐션 + regime 표 등은 *long-tail* (worst regime 16/17 의 0.22/0.26 sample) 회수용 보조 장치
- plan-007 의 후보 다양화 우선순위 강화: long-tail regime 의 hit-rate 를 끌어올리는 새 공식 추가가 ROI 가장 높음

## §6. 다음 plan 후보

→ `analysis/plan-006/next_plan_candidates.md` (LB 점수 회수 후 시나리오별 우선순위 결정)

## §7. 변경 이력

- 2026-05-12 11:13 KST: c5 — dacon-submit 성공 (`isSubmitted: True, detail: Success`).
- 2026-05-12 11:13 KST: c6 — STAGE 4 synthesis. lb_score=TBD (carry-over).
- 2026-05-12 11:26 KST: c5.1 — 사용자 LB 회수 = **0.6692**. 3 파일 frontmatter 동시 갱신 + status `all_complete`. 시나리오 A 채택 (plan-005 통찰 LB 단위 입증).
