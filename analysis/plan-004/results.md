---
plan_id: 004
exp_id: P001_pb-0-6822-fullrun
lb_exp_id: P001_pb-0-6822-fullrun
lb_score: 0.6806
lb_submitted_at: 2026-05-11T20:10:12+09:00
status: all_complete
date: 2026-05-11 (Asia/Seoul)
---

# plan-004 analysis — PB_0.6822 Notebook Full-Fit Results

상세 메트릭 + caveats: `plans/plan-004-pb-0-6822-fullrun.results.md`
18×27 분포 박제: `analysis/plan-004/regime_distribution.{json,md}`
LB 제출 기록: `analysis/plan-004/lb_log.md`

## §1. 한 줄 요약

PB_0.6822 노트북 framework (Attn-GRU selector + 18×27 regime bias + Tiny boundary corrector) 을 우리 데이터에 1:1 적용:

- selector 5-fold OOF soft hit = **0.6511** (10000 rows)
- boundary corrector OOF soft hit = **0.6718** (fold 0 val, +0.0094 lift over selector baseline 0.6624)
- **LB 점수 = 0.6806** (노트북 작성자 주장 0.6822 와 gap 0.0016, 기존 best B001 0.60 대비 +0.0806 lift)
- 18 regime 모두 sample ≥ 274 (degenerate 0개), 19개 hyper-specialized cell — 후속 plan regime 재설계 anchor 정보 확보

## §2. 핵심 비교

| exp_id | OOF hit (soft) | LB |
|---|---|---|
| B001_linear-2pt (plan-001) | n/a | 0.60 |
| R006_combined-winners (plan-003) | n/a | 0.5688 |
| **P001_pb-0-6822-fullrun (plan-004)** | **0.6718** (boundary OOF) | **0.6806** |

본 plan 의 노트북 framework 가 우리 데이터에서 LB 0.6806 — 기존 best (0.60) 대비 +0.0806, R006 (0.5688) 대비 +0.1118 lift 확정. 노트북 작성자 주장 0.6822 와의 gap 0.0016 은 seed/data-split 차이로 설명 가능한 재현 범위. (§0 narrative 갈래 1: 측정 완료.)

## §3. 다음 step

상세는 `plans/plan-004-pb-0-6822-fullrun.results.md §5` 참조. 주요 후보:

1. regime 재설계 — 19개 hyper-specialized cell 의 분포 anchor 활용
2. selector epoch budget 확장 (70% → 100%)
3. boundary corrector 5-fold 확장 (v3 spec §7 의 corrector_no_convergence 자동 판정 활성화)
4. P001 vs R001/B001 ensemble winning-pattern (LB 점수 회수 후 결정)
