# plan-028 — Per-anchor Isolation × Sample-weight Probe (FINAL)

> **G3 band = partial**. best_cell = `B3`, hit_1cm=0.6509, hit_1p5cm=0.8118. paired Δ vs plan-022 winner = -0.0021, vs F0 = +0.0189, vs B4 (plan-025 C1) = +0.0189. G2.B branch activate = `δ`. 5가설 verdict: (a)=inconclusive, (b)=inconclusive, (c)=inconclusive, (d)=inconclusive, (e)=inconclusive.

## 1. plan_id / version / date / status / band / best_cell

- plan_id: 028
- version: v1
- date: 2026-05-22 (Asia/Seoul)
- status: all_complete
- band: partial
- best_cell: B3


## 2. G-gate 표

| Gate | Status | 결과 |
|:--|:--|:--|
| G0 | ✅ | 15/15 pytest pass (4s) |
| G1 | ✅ | F0 0.6320/0.8033 ✓ + plan-022 0.6530/0.8108 ✓ + plan-025 C1=B4 0.6320/0.8033 ✓ (805s) |
| G2.A | ✅ | 9 cell 모두 finite + max_class_ratio 박제 |
| G2.B | ✅ | branch `δ` activated, 1~2 cell 실행 |
| G3 | ⚠️ | band=partial, best=B3=0.6509 |
| G_final | ✅ | results.md + 3-file frontmatter sync + follow-up |


## 3. G2.A 9 cell 결과 표

| cell | hit_1cm | hit_1p5cm | Δ_F0 | Δ_p022 | Δ_B4 | max_class_ratio | top1_acc | runtime |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| **B1** | 0.6320 | 0.8033 | +0.0000 | -0.0210 | +0.0000 | 0.0714 | 0.0821 | 77.8s |
| **B2** | 0.6320 | 0.8033 | +0.0000 | -0.0210 | +0.0000 | 0.0714 | 0.0892 | 112.5s |
| **B3** | 0.6509 | 0.8118 | +0.0189 | -0.0021 | +0.0189 | 0.1056 | 0.1728 | 820.7s |
| **B4** | 0.6320 | 0.8033 | +0.0000 | -0.0210 | +0.0000 | 0.0714 | 0.0879 | 484.2s |
| **W1** | 0.6320 | 0.8033 | +0.0000 | -0.0210 | +0.0000 | 0.0714 | 0.0938 | 347.6s |
| **T1** | 0.6320 | 0.8033 | +0.0000 | -0.0210 | +0.0000 | 0.0714 | 0.0646 | 289.5s |
| **T2** | 0.6320 | 0.8033 | +0.0000 | -0.0210 | +0.0000 | 0.0714 | 0.0671 | 280.5s |
| **S1** | 0.6320 | 0.8033 | +0.0000 | -0.0210 | +0.0000 | 0.0714 | 0.0879 | 336.9s |
| **R1** | 0.6320 | 0.8033 | +0.0000 | -0.0210 | +0.0000 | 0.0714 | 0.0888 | 286.9s |

## 4. G2.B branch + cell 결과 표

- activated branch: **`δ`**

- branch cell 별 결과는 results_Bx_*.json 참조 (활성 branch 의 cell 1~2개).


## 5. Best cell 박제 + paired Δ 3종

```json
{
  "best_cell": "B3",
  "best_hit_1cm": 0.6509,
  "best_hit_1p5cm": 0.8118,
  "paired_delta": {
    "vs_f0": {
      "hit_1cm": 0.018900000000000028
    },
    "vs_p022": {
      "hit_1cm": -0.0020999999999999908
    },
    "vs_p025_C1": {
      "hit_1cm": 0.018900000000000028
    }
  },
  "oracle_recovery": 0.8210141271442988,
  "hypothesis_verdict": {
    "(a) tau_cls_sharp_gap": "inconclusive",
    "(b) sample_weight": "inconclusive",
    "(c) subclass_self_consistency": "inconclusive",
    "(d) broadcast_dominance": "inconclusive",
    "(e) seq_compression_lossy": "inconclusive"
  },
  "g2b_branch": "\u03b4",
  "band": "partial"
}
```


## 6. 5가설 verdict

- **(a) tau_cls_sharp_gap**: inconclusive
- **(b) sample_weight**: inconclusive
- **(c) subclass_self_consistency**: inconclusive
- **(d) broadcast_dominance**: inconclusive
- **(e) seq_compression_lossy**: inconclusive


## 7. 14-anchor oracle 회수율
- oracle ceiling = 0.7928 (plan-024 carry)
- best_cell 회수율 = 0.8210 (82.10%)
- plan-022 winner 회수율 = 0.8237 (82.37%)


## 8. 1080D input block 분해 (§4.3 박제)

| Block | Source | Dim | 본 plan cell 결과 |
|:--|:--|--:|:--|
| ① plan-022 carry | build_input_common + lgbm_extra | 170 | B2 (192=①+③) |
| ② cand_builder ctx | regime/multi-window/STA-LTA | 128 | broadcast, B3 (1058=no③) 포함 |
| ③ cand_builder per-anchor | par/perp/dist + spec + interactions | 22 | **B1 (22D only)**, (d) 검증 |
| ④ seq_builder 8-stat | per-channel last/first/mean/std/slope/max/min/range | 760 | R1 (raw 95×7=665D 대체) |


## 9. Runtime per STAGE

- G0 pytest: 4s
- G1 (3 carry): ~805s (~13.4min)
- G2.A (8 신규 cell parallel): ~2552s total CPU
- 총 plan-028: G0~G_final 약 ~56min CPU


## 10. max_class_ratio + top1_acc per cell

| cell | max_class_ratio | top1_acc |
|:--|--:|--:|
| B1 | 0.0714 | 0.0821 |
| B2 | 0.0714 | 0.0892 |
| B3 | 0.1056 | 0.1728 |
| B4 | 0.0714 | 0.0879 |
| W1 | 0.0714 | 0.0938 |
| T1 | 0.0714 | 0.0646 |
| T2 | 0.0714 | 0.0671 |
| S1 | 0.0714 | 0.0879 |
| R1 | 0.0714 | 0.0888 |


## 11. Follow-up plan 후보

- (G3 결과에 따라 박제)

- lift 미달 cause 추가 분석 + G2.B branch δ MLP 가 plan-022 winner 못 이긴 경우 paradigm-level 'LGBM ceiling' 박제

## 12. Cross-refs

- spec: `plans/plan-028-per-anchor-isolation-weight-probe.md`
- results pair: `plans/plan-028-per-anchor-isolation-weight-probe.results.md`
- baseline_carry: `analysis/plan-028/baseline_carry.json`
- 9 cell results: `analysis/plan-028/results_{B1,B2,B3,B4,W1,T1,T2,S1,R1}.json`
- branch result: `analysis/plan-028/results_Bx_*.json`
- paradigm: `analysis/plan-028/paradigm_analysis_g2a.json` + `paradigm_analysis.json`
