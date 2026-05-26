---
plan_id: a-003
finished_at: 2026-05-26T05:30+09:00
status: all_complete
lane: a
exp_ids_completed:
  - KR008_reflect-noise-aug
exp_ids_skipped:
  - KR009_reflect-only (deferred — KR008 OOF-neutral·noise-limited 분해, 사용자 confirm 시에만)
best_exp_id: KR008_reflect-noise-aug
baseline_exp: KR003 (plan-a-002, OOF 0.6667 / LB 0.6854 record)
g_aug_oof_hit_1cm: 0.6671
g_aug_delta_vs_kr003: 0.0004
g_aug_p: 0.8668
g_aug_band: no_regression_PASS
kalman_alone_hit_1cm: 0.5964
lb_score: {KR008: not_submitted_user_gated}
lb_note: KR008 OOF 0.6671 = KR003 +0.0004 (p=0.87 neutral, no-regression PASS). submission_kr008.csv 생성 완료. CV-LB 괴리상 OOF-neutral 이 LB 양 가능성(yaw·부산물 전례), 단 LB 노이즈 floor(±0.002~0.003)·quota(오늘 2 남음) 고려해 **제출은 사용자 confirm gated** (자동 안 함). 제출 시 KR003 0.6854 대비 Δ 가 verdict.
band: neutral_no_regression
---

# plan-a-003 results — 반사 + 노이즈 augmentation (KR003 위 대칭/정칙화 lever)

## §0. 한 줄 결론

**반사(yaw frame y→−y, p=0.5) + 노이즈(σ=0.10) train-time augmentation 을 KR003 에 추가한 KR008 = OOF-neutral·무회귀** — OOF hit_1cm **0.6671**, KR003 0.6667 대비 Δ=**+0.0004 (p=0.87, ns)** → `no_regression_PASS` (≥0.6657). 입력 yaw·Kalman 부산물과 **동일한 OOF-neutral 패턴** — augmentation 의 일반화 이득(있다면)은 train 5-fold OOF 가 못 잡음 (CV-LB 괴리). **aug-off 는 KR003 과 bit-identical repro 확증** (RNG guard). 진짜 verdict 는 LB 이나, **노이즈 floor(±0.002~0.003) 근처 + quota 제약**으로 제출은 사용자 confirm gated (§5). 고차 물리 모델은 설계상 명시 배제.

> **★ 메타 결론 — lever 수확 체감 확인**: LB lever 효과가 입력 yaw +0.0060 → Kalman 부산물 +0.0036 → augmentation OOF +0.0004(LB 미정)로 감소. 이 paradigm 은 record(0.6854) 부근에서 노이즈 floor 에 수렴 중. augmentation 은 *무해(무회귀)하나 OOF 로는 추가 이득 불검출* — LB 제출만이 판정 가능하고 그조차 floor 내일 공산.

## §0.5 Result Quick Reference

| exp | OOF hit_1cm | hit_1p5cm | config A / B | vs baseline | band |
|---|---|---|---|---|---|
| KR003 (plan-a-002 baseline) | 0.6667 | 0.8205 | 0.6648 / 0.6676 | — (LB 0.6854 🏆) | — |
| **KR008** (+반사+노이즈 aug) | **0.6671** | 0.8228 | 0.6643 / 0.6677 | KR003 **+0.0004 (p=0.87)** | **no_regression_PASS** |

- baseline: KR003 OOF 0.6667 / **LB 0.6854 record**. KR002 LB 0.6818. F0 0.6320. Kalman-alone 0.5964.
- aug: 반사 p=0.5 online (seq `_y` idx [1,4,7,10,13] + scalar cvca_y idx 41 + 타깃 y 부호 반전), 노이즈 σ=0.10 (표준화 seq 가산). **train-only**, val/test 원본.
- budget: 2cfg(A/B) × 5fold stable_fold_id × 3seed × 200ep, GPU L40S, 850s.
- **aug-off bit-identical**: `--reflect-aug`/`--noise-aug` 없으면 KR003 과 동일 (smoke 0.6637 일치, RNG 미소비 guard).
- paired permutation 10000 resample, N=10000. submission_kr008.csv 생성 (미제출).

## §1. 가설 판정

| 가설 | 판정 | 근거 |
|---|---|---|
| **H1 (반사 → sample efficiency)** | ⚠️ **OOF neutral, LB 보류** | KR008 0.6671 ≥ KR003 0.6667 (무회귀 ✓), Δ+0.0004 p=0.87 ns. OOF 로는 반사 대칭 이득 불검출. CV-LB 괴리상 LB 판정 필요 (yaw·부산물 전례) — 단 사용자 gated. |
| **H2 (노이즈 → robust)** | ⚠️ **OOF neutral (H1 과 묶음)** | KR008 = 반사+노이즈 묶음. 묶음 전체 OOF neutral·무회귀. 노이즈 σ=0.10 가 학습 안정성 안 깨뜨림 (G1 0.6757). |
| **메타 (고차 물리 무의미)** | ✅ **설계 준수** | CTRV/고차 Kalman 배제. lever = 대칭/정칙화 축만. |
| **메타 (수확 체감)** | ✅ **확인** | OOF Δ가 +0.0024(yaw)→+0.0004(부산물,aug) 로 floor 수렴. record 0.6854 부근 노이즈 한계. |

## §2. Gate 판정

| gate | 결과 | band/severity |
|---|---|---|
| G0 인프라+smoke+반사항등성+aug-off repro | green (pytest 4 pass, aug-off 0.6637 bit-identical, aug-on finite) | — |
| G1 KR008 1f vs KR003 1f | 0.6757 vs 0.6762 (Δ−0.0005) | PASS |
| **G_aug** KR008 full vs KR003 | **0.6671**, Δ+0.0004 p=0.87 | **no_regression_PASS** (≥0.6657; positive 미달) |
| **G_lb** KR008 LB vs KR003 0.6854 | submission_kr008.csv 생성, **미제출 (사용자 gated)** | verdict 보류 |
| G_final | results + §0.5 sync + main merge | 완료 |

## §3. exp 별 산출

### KR008_reflect-noise-aug (G_aug no_regression_PASS)
- ensemble OOF hit_1cm **0.6671** / hit_1p5cm 0.8228. config A 0.6643, B 0.6677.
- vs KR003 0.6667: Δ=+0.0004, paired permutation p=0.8668 → 무회귀 PASS, lift 비유의(neutral).
- **aug-off repro 불변 확증**: `--reflect-aug`/`--noise-aug` off 시 train_one RNG 미소비 → KR003 bit-identical (smoke 0.6637 일치). aug-on 만 RNG 소비.
- 반사 대상 audit: seq `_y` (rel/v/a/innov/fv_y, idx 1,4,7,10,13) + scalar cvca_y(41) + 타깃 y. magnitude/cos/norm 불변 (반사 제외 정당, smoke 검증). `flags.reflect_idx_*` in results_kr008.json.
- 산출: `analysis/plan-a-002/results_kr008.json` + `.npz` + `submission_kr008.csv` (uncalibrated, 미제출).

### KR009_reflect-only (deferred)
- 반사 단독 분해 — KR008 이 OOF-neutral 이고 분해는 noise-limited (plan-a-002 박제) → **미실행**. KR008 LB 가 record 갱신 시에만 사용자 confirm 후.

## §4. 해석 + 함의

1. **augmentation = 무해하나 OOF 무이득**: 반사+노이즈가 학습을 안정적으로 유지(무회귀, G1·G_aug)하나 OOF hit 은 KR003 과 통계적 동률. 반사의 경우 — yaw frame 정규화가 회전을 이미 제거했고, GRU 가 좌우 maneuver 를 raw v/a seq 로 이미 학습 가능해 반사 대칭의 *추가* sample efficiency 가 OOF 에선 미미. 노이즈 정칙화도 OOF 분포 내에선 이득 없음.
2. **CV-LB 괴리상 LB 만이 판정**: 입력 yaw·부산물 둘 다 OOF-neutral·LB-positive 였으므로 KR008 도 LB 양 가능성 배제 못 함. 단 **수확 체감 + 노이즈 floor** 로 LB Δ 가 검출 한계 내일 공산이 큼. 그래서 제출을 강행하지 않고 사용자 gated (quota 오늘 2 남음).
3. **paradigm 성숙 신호**: 입력 yaw(+0.006)→부산물(+0.0036)→aug(OOF flat) 로 lever ROI 가 체감. Kalman-잔차 paradigm 은 record 0.6854 부근에서 *대칭/정칙화 축도 소진* — 추가 이득은 노이즈 수준이거나 구조적으로 다른 접근(별 paradigm) 필요.
4. **인프라 자산**: `run_oof.py` 가 이제 `--reflect-aug --noise-aug` 보유 (aug-off repro 불변) → 향후 aug 실험 재사용 가능.

## §5. Follow-up 후보 (번호 미할당)

- **KR008 LB 제출 (사용자 quota confirm gated)**: submission_kr008.csv 준비됨. CV-LB 괴리상 OOF-neutral·LB-positive 가능성 있으나 노이즈 floor·수확 체감으로 inconclusive 공산. 제출 시 KR003 0.6854 대비 Δ 가 verdict (floor 내면 inconclusive 박제). **사용자 명시 confirm 필요** ([[feedback-dacon-submit-confirmation]]).
- **KR009 반사 단독**: KR008 LB record 갱신 시에만 attribution (noise-limited, deferred).
- **σ_aug / 반사확률 grid**: 본 plan 고정값. 단 OOF 무이득이라 grid ROI 낮음.
- **구조적 전환**: paradigm 이 record 부근 노이즈 floor 수렴 → 미세 lever 보다 별 paradigm/ensemble 또는 데이터 자체 재검토가 다음 ROI.

## §6. 재현

```
# G1
python analysis/plan-a-002/run_oof.py --gate g1 --innov --filtered-v --cv-ca --input-yaw \
    --reflect-aug --noise-aug 0.10 --exp KR008 --out g1_kr008.json
# full OOF + submission
python analysis/plan-a-002/run_oof.py --gate full --innov --filtered-v --cv-ca --input-yaw \
    --reflect-aug --noise-aug 0.10 --predict-test --exp KR008 \
    --out results_kr008.json --compare-to results_kr003.npz
# aug-off repro (KR003 bit-identical 확인) + smoke
python analysis/plan-a-002/run_oof.py --gate smoke --innov --filtered-v --cv-ca --input-yaw --exp chk  # → 0.6637
python -m pytest tests/test_plan_a003_smoke.py -q
```
