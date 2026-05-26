---
plan_id: a-004
finished_at: 2026-05-26T07:00+09:00
status: all_complete
lane: a
exp_ids_completed:
  - KR010_mcl2-decisive
exp_ids_skipped:
  - KR011_gen-axis (G1_decisive KILL → fail-fast skip)
  - KR012_route-K-axis (G1_decisive KILL → fail-fast skip)
best_exp_id: KR010_mcl2-decisive
baseline_exp: KR008 (plan-a-003, LB 0.6862 / OOF 0.6671 / 1-fold 0.6757)
g1_decisive_verdict: KILL
g1_oracle_at_2: 0.6787
g1_realized_hit: 0.6767
g1_gate_threshold: 0.696
lb_score: not_submitted (KR008 못 넘음 — 제출 안 함)
band: KILL_single_gru_conditional_optimal
---

# plan-a-004 results — Multi-Hypothesis 후보 (KR010 G1_decisive KILL)

## §0. 한 줄 결론

**KR010 G1_decisive = KILL.** 2-head MCL(soft_top=2, both heads 생존 per_head [1277,743])의 **oracle@2 = 0.6787** 이 gate 임계 **0.696**(= KR008 1-fold baseline 0.6757 + 0.02) 에 한참 미달 — 단일 head 대비 **고작 +0.003** (noise). realized-hit 0.6767 ≈ KR008 0.6757 (Δ+0.001, p=0.75). → **두 head 가 둘 다 학습·특화돼도 단일 head 보다 의미있게 더 덮지 못함 = 단일 GRU 가 conditional 최적.** §1 핵심 리스크("단일 GRU 가 이미 soft mode-conditioning")가 확증. **fail-fast 설계대로 96초·1-fold 로 multi-hypothesis 전 방향이 무용임을 확정, breadth(KR011/12) skip.** LB 미제출(KR008 못 넘음).

> **★ mode_selectability 와의 화해**: mode_sel 은 "거친 k-means 모드가 95% 선택가능"을 보였으나, KR010 은 그 모드가 **예측-관련 multimodal 분기가 아니라 단일 GRU 가 이미 잡는 residual 군집**이었음을 드러냄. 정밀 per-sample head 로 바꿔도 head 들이 단일 GRU 예측 주변으로 수렴(분화 부족) → oracle 안 오름. "선택 가능"과 "선택해서 이득"은 별개였다.

## §0.5 Result Quick Reference

| 항목 | 값 |
|---|---|
| KR010 (2-head MCL, soft_top=2) | oracle@2 **0.6787** / realized-hit **0.6767** / per_head [1277,743] |
| baseline (KR008 fold0/seed0/cfgA 1-fold) | 0.6757 |
| G1 gate | oracle@2 ≥ **0.696** (= 0.6757+0.02) → **0.6787 미달 = KILL** |
| oracle headroom | +0.003 (단일 대비, noise 수준 — +0.02 요구 대비 1/7) |
| budget | 1-fold fold0/seed0/cfgA, 200ep, GPU L40S, 96s |

## §1. 가설 판정

| 가설 | 판정 | 근거 |
|---|---|---|
| **H1 (multi-head oracle headroom ≥+0.02)** | ❌ **반증** | oracle@2 0.6787 = baseline 0.6757 +0.003 ≪ +0.02. both heads 생존인데도 best-of-2 가 단일 head 를 거의 안 넘음. |
| **H2 (selectability→realized 초월)** | ❌ **반증** | realized 0.6767 ≈ KR008 0.6757 (Δ+0.001 noise). selector 가 잘 골라도 후보 자체가 안 다양함. |
| **메타 (단일 GRU 가 conditional 최적?)** | ✅ **확증** | head 분화 부족 = 단일 GRU 가 이미 per-sample 최적 근방. MoE hard-routing 우위 없음. |

## §2. Gate 판정

| gate | 결과 | band |
|---|---|---|
| G0 인프라+smoke | pytest 5 pass, runner smoke OK, n_heads=1 경로 | green |
| **G1_decisive** | oracle@2 **0.6787** < 0.696 | **KILL** (단일 GRU conditional 최적, 정보) |
| G_mh / G_lb | — | **skip (fail-fast)** |
| G_final | results 박제 + sync + merge (축소) | 완료 |

## §3. 해석 + 함의

1. **Multi-hypothesis 방향 폐기 (확정)**: best-of-2 oracle 이 단일 head 를 +0.003(noise) 밖에 못 넘음 → K head 가 단일 GRU 예측 주변으로 수렴, 분화 부족. soft_top=2 로 both heads 살려도 동일 → 구조적 결론이지 학습 우연 아님.
2. **ceiling_verify 재확인**: plan-a-002 ceiling_verify 가 "0.6862 ≈ 정보 천장, selector gap irreducible" 이라 했고, KR010 이 *학습 head 기반*으로도 그걸 확증. mode_sel 의 95% 선택가능성은 "단일 GRU 가 이미 쓰는 feature 로 residual 군집을 분류"한 것이지 새 정보가 아니었다.
3. **fail-fast 가치**: c1~c3 인프라 + 96초 1-fold 로 multi-hypothesis(생성×선택×K 전 axis)가 무용임을 확정 — breadth sweep(KR011/12, 수십 run) 절약. KILL gate 설계가 정확히 작동.
4. **남은 +0.01 경로 = 새 입력정보뿐**: 미세 lever·corrector·고정 anchor selector·단순 ensemble·**multi-hypothesis** 전부 ROI 0 확정. 현 (N,11,3) 440ms position 데이터의 정보 천장(0.6862). +0.01 은 데이터 확장(더 긴 history·추가 채널)만이 escape.

## §4. Follow-up 후보 (번호 미할당)

- **multi-hypothesis/MoE 재시도 비권장**: oracle@2 가 단일 대비 +0.003 → K↑·생성 mechanism 변경해도 천장 동일 (분화 부족이 구조적). KR011/12 의도적 skip 유지.
- **유일 escape = 새 입력정보**: competition 데이터 사양 확인 (440ms 보다 긴 trajectory·추가 센서·cross-sample). 모델링 아닌 데이터 문제.
- 현 best 고정: **KR008 LB 0.6862 (plan-a-003)** = 프로젝트 top, 정보 천장 부근.

## §4.5 K-sweep 추가 검증 (oracle↔realized 간극 = selection 벽)

KILL 후 "K=3,4 면 oracle 이 단조 증가해 gate 넘지 않나?" 검증 (g1, soft_top=2):

| K | oracle@K | realized-hit | KR008 대비 realized Δ (p) | per_head |
|---|---|---|---|---|
| 1 (KR008) | 0.6757 | 0.6757 | — | — |
| 2 | 0.6787 | 0.6767 | +0.001 (0.75) | [1277,743] |
| **3** | **0.7653** | **0.6347** | **−0.044 (p=0)** | [811,606,603] |
| **4** | **0.7579** | **0.6317** | **−0.047 (p=0)** | [1147,504,314,55] |

- **oracle 는 K 따라 급증** (0.6787→0.7653, gate 0.696 초과) — 후보가 정답을 더 덮음(미래 미세 multimodal 실재).
- **그러나 realized-hit 는 K 따라 폭락** (0.6767→0.6347→0.6317, p=0) — selector 가 늘어난 후보 중 정답 head 를 못 골라, KR008 보다 훨씬 나빠짐.
- **oracle↔realized 간극 = 0.13 (K=3)** → ceiling_verify 의 selection 벽(14-anchor oracle 0.79 vs selector 0.63)과 동일 메커니즘. "정답이 후보에 있음" ≠ "고를 수 있음".
- **G1 gate 정당성 확인**: gate = oracle@2≥0.696 **AND** realized≥0.6707. K=3 은 oracle(0.765) 통과해도 realized(0.6347)<0.6707 로 **여전히 KILL** → K 늘려도 통과 불가, KILL airtight.

## §5. 재현

```
# G1_decisive (KILL gate)
python analysis/plan-a-004/run_oof_mh.py --gate g1 --innov --filtered-v --cv-ca --input-yaw \
    --reflect-aug --noise-aug 0.10 --n-heads 2 --gen mcl --route joint --soft-top 2 \
    --exp KR010 --out g1_kr010.json --compare-to ../plan-a-002/results_kr008.npz
python -m pytest tests/test_plan_a004_smoke.py -q
```
