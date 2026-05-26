---
plan_id: a-002
finished_at: 2026-05-26T04:00+09:00
status: all_complete
lane: a
exp_ids_completed:
  - KR003_kalman-derived-feats
  - KR004_filtered-yaw-frame
best_exp_id: KR003_kalman-derived-feats
baseline_exp: KR002 (plan-a-001, OOF 0.6663 / LB 0.6818)
g_kalman_oof_hit_1cm: 0.6667
g_kalman_delta_vs_kr002: 0.0004
g_kalman_p: 0.8347
g_kalman_band: no_regression_PASS
g_frame_oof_hit_1cm: 0.6652
g_frame_delta_vs_kr003: -0.0015
g_frame_p: 0.1973
g_frame_band: neutral
kalman_alone_hit_1cm: 0.5964
lb_score: {KR003: pending_user_gated, KR004: pending_user_gated}
lb_note: OOF neutral (KR003 Δ+0.0004 ns / KR004 Δ−0.0015 ns). CV-LB 괴리 전제상 OOF neutral 이 폐기 신호 아님 — best OOF config(KR003) LB 제출을 사용자 confirm 후 권장 (입력 yaw 처럼 OOF-neutral·LB-positive 가능성). 본 plan 은 OOF only.
band: neutral_no_regression
---

# plan-a-002 results — Kalman 부산물 입력 feature (innovation · filtered v · CV/CA 불일치)

## §0. 한 줄 결론

**Kalman 부산물 feature 3종(innovation·filtered velocity·CV/CA 불일치)을 KR002 입력에 회수한 KR003 = OOF-neutral·무회귀** — OOF hit_1cm **0.6667**, KR002 baseline 0.6663 대비 Δ=**+0.0004 (p=0.83, ns)** → `no_regression_PASS` (≥0.6653 임계). filtered-yaw frame 으로 바꾼 KR004 = **0.6652**, KR003 대비 Δ=**−0.0015 (p=0.20, ns)** → `neutral` (미미한 비유의 하락). **두 변경 모두 OOF 에선 통계적 0** — 신규 feature 가 학습 안정성을 깨지 않고(무회귀) OOF 일반화를 유의하게 올리지도 않음. **plan-a-001 의 CV-LB 괴리(KR002 입력 yaw = OOF neutral·LB +0.0060) 전제상 OOF neutral 은 폐기 신호가 아니며, best OOF config(KR003)의 LB 제출이 진짜 verdict** (사용자 quota confirm gated, §5).

> **gain/covariance 제외의 정보이론적 정당성 확증**: 선형 시불변 Kalman 에서 K·P 는 측정 독립 = 전 샘플 동일 상수. 구현에서 K(t)·P(t) 를 loop 1회 precompute 후 재사용함으로 실증 (innov·state 만 per-sample). 제외는 휴리스틱 아닌 정당한 결정 — feature 로 넣었어도 정보량 0.

## §0.5 Result Quick Reference

| exp | OOF hit_1cm | hit_1p5cm | config A / B | vs baseline | band |
|---|---|---|---|---|---|
| KR002 (plan-a-001 baseline) | 0.6663 | — | 0.6667 / 0.6671 | — (LB 0.6818 🏆) | — |
| **KR003** (+innov+filtered_v+cv_ca) | **0.6667** | 0.8205 | 0.6648 / 0.6676 | KR002 **+0.0004 (p=0.83)** | **no_regression_PASS** |
| **KR004** (+filtered-yaw frame) | **0.6652** | 0.8209 | 0.6645 / 0.6667 | KR003 **−0.0015 (p=0.20)** | **neutral** |

- baseline: Kalman-alone OOF hit_1cm = **0.5964**. F0 = 0.6320. KR003 vs F0 = **+0.0347**, vs KR001 0.6639 = +0.0028.
- 입력 dim: KR003/KR004 모두 seq **15** (rel/v/a 9 + innov 3 + filtered_v 3), scalar **44** (40 + cv_ca 회전3 + norm1).
- budget: 2 config(A lr5e-4·p0.3 / B lr1e-3·p0.1) × 5-fold stable_fold_id × 3 seed × 200ep, GPU L40S. KR003 618s / KR004 565s.
- paired permutation 10000 resample (sign-flip), N=10000. **LB 미제출 (OOF only, quota 사용자 gated)**.

## §1. 가설 판정

| 가설 | 판정 | 근거 |
|---|---|---|
| **H1 (innovation → maneuver 잔차 개선)** | ⚠️ **OOF neutral** | KR003 0.6667 ≥ KR002 0.6663 (무회귀 ✓) 이나 Δ+0.0004 p=0.83 비유의. innovation 등 부산물이 OOF 일반화를 유의 상승시키지 못함. **단 CV-LB 괴리상 LB 판정 보류** — OOF neutral 이 음 결론 아님. |
| **H2 (filtered v + CV/CA 보조 lift)** | ⚠️ **OOF neutral (H1 과 묶음)** | KR003 = (a)+(b)+(c) 동시 cluster. 묶음 전체 OOF neutral. per-channel attribution 은 약신호(CV-LB 괴리)라 deferred — OOF 만으론 어느 채널 기여 분리 불가. |
| **H3 (filtered-yaw frame 이 LB lever 증폭)** | ⚠️ **OOF 미증폭 (neutral·미미한 하락)** | KR004 vs KR003 Δ=−0.0015 p=0.20 → OOF 에선 filtered heading 이 raw heading 대비 이득 없음 (오히려 비유의 하락). raw v_last yaw(KR003) 가 OOF 상 동등/우월. LB 증폭 가설은 OOF 로 미검증 (CV-LB 괴리상 LB 만이 판정 가능하나 본 plan out-of-scope). |
| **메타 (gain/covariance 제외 = 정보 0)** | ✅ **확증** | K·P 측정독립을 구현에서 precompute-재사용으로 실증. 제외 정당. |

## §2. Gate 판정

| gate | 결과 | band/severity |
|---|---|---|
| G0 인프라+smoke+leakage assert | green (7 pass, innov/filtered t_pred-invariant) | — |
| G1 KR003 1f1s vs KR002 1f | 0.6762 vs 0.6767 (Δ−0.0005, |Δ|≪0.005 tol) | PASS |
| **G_kalman** KR003 full vs KR002 | **0.6667**, Δ+0.0004 p=0.83 | **no_regression_PASS** (≥0.6653; positive 미달) |
| **G_frame** KR004 vs KR003 | **0.6652**, Δ−0.0015 p=0.20 | **neutral** (|Δ|<0.002 ns) |
| G_final | results + §0.5 sync + main merge | 완료 |

## §3. exp 별 산출

### KR003_kalman-derived-feats (G_kalman no_regression_PASS)
- ensemble OOF hit_1cm **0.6667** / hit_1p5cm 0.8205. config A 0.6648, B 0.6676 (ensemble ≈ B).
- vs KR002 0.6663: Δ=+0.0004, paired permutation p=0.8347 → 무회귀 PASS, lift 비유의(neutral).
- **leakage 안전 확증**: innov_seq·filtered_v 는 관측창 X[:, :11] 만의 함수, t_pred 변경에 불변 (G0 assert). 미래 +80ms 관측 X 부재 → 구조적 무참조.
- 신규 feature 회전 audit: seq 15채널 전부 벡터 triplet → rotate_xy(θ) z 보존. cv_ca 회전3 + norm(L2 불변) 1. `rotation_class` 박제 in results_kr003.json.
- 산출: `analysis/plan-a-002/results_kr003.json` + `.npz` (oof_pred/per_sample_hit).

### KR004_filtered-yaw-frame (G_frame neutral)
- ensemble OOF hit_1cm **0.6652** (A 0.6645, B 0.6667). vs KR003 Δ=−0.0015, p=0.1973 → neutral.
- **단일 변수**: θ source 만 `yaw_angle(raw v_last)` → `yaw_angle(filtered v_last)` 교체, feature 집합 불변. filtered velocity 로 heading 산출 시 OOF 에선 raw heading 대비 이득 없음 (미미한 비유의 하락).
- 산출: `analysis/plan-a-002/results_kr004.json` + `.npz`.

## §4. 해석 + 함의

1. **부산물 feature = OOF-neutral·무회귀**: innovation·filtered velocity·CV/CA 불일치를 입력에 회수해도 OOF hit_1cm 은 KR002 대비 통계적 0 (Δ+0.0004 ns). 잔차 GRU 가 이미 raw rel/v/a seq 에서 maneuver 신호를 충분히 추출하고 있어 명시적 부산물 주입의 OOF 한계효용이 0 에 가까움. **단 무회귀** — 저차원 물리기반 feature 라 overfit/발산 없음 (CV-LB 괴리 환경 설계 원칙 적중).
2. **CV-LB 괴리 전제의 일관 적용**: plan-a-001 에서 입력 yaw 회전이 OOF neutral(+0.0024)인데 LB +0.0060 였던 것처럼, KR003 의 OOF neutral 도 LB 양 가능성을 배제 못 함. **OOF 만으로 feature 폐기 금지** (§0.5 plan-specific 주의 준수). 진짜 verdict 는 LB.
3. **filtered-yaw frame 은 OOF 비권장**: KR004 가 KR003 대비 OOF 미미 하락 → raw v_last yaw(KR003) 가 frame source 로 동등/우월. LB 증폭 가설(H3)은 OOF 로 미검증.
4. **gain/covariance 제외 실증**: 측정독립성을 구현 precompute 로 확인 — 정보이론적 제외가 정당했음.

## §5. Follow-up 후보 (번호 미할당)

- **★ KR003 LB 제출 (사용자 quota confirm gated)**: best OOF config KR003(0.6667) 을 test 예측 + DACON 제출로 LB 회수. **CV-LB 괴리 전례상 OOF-neutral·LB-positive 가능성** — 입력 yaw 가 그랬듯 부산물 feature 가 test 에서 일반화 이득을 낼 수 있음. KR002 LB 0.6818 대비 Δ 가 진짜 판정. (DACON 일일 5회 quota, 사용자 명시 confirm 필수 — OOF neutral 이라 더더욱 신중.)
- **per-channel attribution (LB 기반)**: KR003 묶음 중 어느 부산물이 LB lever 인지 — innov-only / filtered_v-only / cv_ca-only 분해 후 각 LB (OOF attribution 은 약신호라 LB 필요).
- **적응형 σ Kalman**: gain 에 정보를 부여하는 sample 별 noise 연동 필터 (본 plan out-of-scope 였던 §6 항목) — gain/covariance 가 per-sample 정보를 갖게 되는 설계.
- **per-step CV/CA 불일치 시퀀스**: 본 plan 은 +80ms 외삽 차 scalar 만. 관측창 전체 시퀀스화로 maneuver 시점 정보 강화.

## §6. 재현

```
# G1 (KR002 baseline 1-fold + KR003 1-fold)
python analysis/plan-a-002/run_oof.py --gate g1 --input-yaw --exp KR002base --out g1_kr002base.json
python analysis/plan-a-002/run_oof.py --gate g1 --innov --filtered-v --cv-ca --input-yaw --exp KR003 --out g1_kr003.json
# full OOF
python analysis/plan-a-002/run_oof.py --gate full --innov --filtered-v --cv-ca --input-yaw \
    --exp KR003 --out results_kr003.json --compare-to ../plan-a-001/results_kr002.npz
python analysis/plan-a-002/run_oof.py --gate full --innov --filtered-v --cv-ca --input-yaw --filtered-yaw \
    --exp KR004 --out results_kr004.json --compare-to results_kr003.npz
# smoke
python -m pytest tests/test_plan_a002_smoke.py -q
```
