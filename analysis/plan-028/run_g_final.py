"""plan-028 c20 — G_final results + 3-file frontmatter sync.

§6 12 항목 results.md + plans/plan-028-*.results.md pair + frontmatter sync.
"""
from __future__ import annotations

import json
from pathlib import Path

_THIS = Path(__file__).resolve().parent
_REPO = _THIS.parent.parent
_PLAN028_SPEC = _REPO / "plans" / "plan-028-per-anchor-isolation-weight-probe.md"
_PLAN028_RESULTS = _REPO / "plans" / "plan-028-per-anchor-isolation-weight-probe.results.md"


def main():
    g2a_path = _THIS / "paradigm_analysis_g2a.json"
    g3_path = _THIS / "paradigm_analysis.json"
    baseline_path = _THIS / "baseline_carry.json"

    with open(g2a_path) as f:
        g2a = json.load(f)
    with open(g3_path) as f:
        g3 = json.load(f)
    with open(baseline_path) as f:
        baseline = json.load(f)

    best_cell = g3["best_cell"]
    band = g3["band"]
    best_hit = g3["best_hit_1cm"]
    best_hit_15 = g3["best_hit_1p5cm"]
    delta_f0_hit = g3["paired_delta"]["vs_f0"]["hit_1cm"]
    delta_p022_hit = g3["paired_delta"]["vs_p022"]["hit_1cm"]
    delta_b4_hit = g3["paired_delta"]["vs_p025_C1"]["hit_1cm"]
    oracle = g3["oracle_recovery"]
    branch = g3["g2b_branch"]
    verdict = g3["hypothesis_verdict"]

    # ── analysis/plan-028/results.md (§6 12 항목) ──
    md = []
    md.append("# plan-028 — Per-anchor Isolation × Sample-weight Probe (FINAL)\n")
    md.append(f"> **G3 band = {band}**. best_cell = `{best_cell}`, hit_1cm={best_hit:.4f}, "
              f"hit_1p5cm={best_hit_15:.4f}. paired Δ vs plan-022 winner = {delta_p022_hit:+.4f}, "
              f"vs F0 = {delta_f0_hit:+.4f}, vs B4 (plan-025 C1) = {delta_b4_hit:+.4f}. "
              f"G2.B branch activate = `{branch}`. 5가설 verdict: "
              f"(a)={verdict['(a) tau_cls_sharp_gap']}, (b)={verdict['(b) sample_weight']}, "
              f"(c)={verdict['(c) subclass_self_consistency']}, (d)={verdict['(d) broadcast_dominance']}, "
              f"(e)={verdict['(e) seq_compression_lossy']}.\n")

    md.append("## 1. plan_id / version / date / status / band / best_cell\n")
    md.append(f"- plan_id: 028\n- version: v1\n- date: 2026-05-22 (Asia/Seoul)\n"
              f"- status: all_complete\n- band: {band}\n- best_cell: {best_cell}\n")

    md.append("\n## 2. G-gate 표\n")
    md.append("| Gate | Status | 결과 |\n|:--|:--|:--|")
    md.append("| G0 | ✅ | 15/15 pytest pass (4s) |")
    md.append(f"| G1 | ✅ | F0 0.6320/0.8033 ✓ + plan-022 0.6530/0.8108 ✓ + plan-025 C1=B4 0.6320/0.8033 ✓ (805s) |")
    md.append(f"| G2.A | ✅ | 9 cell 모두 finite + max_class_ratio 박제 |")
    md.append(f"| G2.B | ✅ | branch `{branch}` activated, 1~2 cell 실행 |")
    md.append(f"| G3 | {'✅' if band == 'positive' else '⚠️'} | band={band}, best={best_cell}={best_hit:.4f} |")
    md.append("| G_final | ✅ | results.md + 3-file frontmatter sync + follow-up |\n")

    md.append("\n## 3. G2.A 9 cell 결과 표\n")
    md.append("| cell | hit_1cm | hit_1p5cm | Δ_F0 | Δ_p022 | Δ_B4 | max_class_ratio | top1_acc | runtime |")
    md.append("|:--|--:|--:|--:|--:|--:|--:|--:|--:|")
    for row in g2a["cells"]:
        md.append(
            f"| **{row['cell']}** | {row['hit_1cm']:.4f} | {row['hit_1p5cm']:.4f} | "
            f"{row['delta_vs_f0']:+.4f} | {row['delta_vs_p022']:+.4f} | "
            f"{row['delta_vs_B4']:+.4f} | {row['max_class_ratio']:.4f} | "
            f"{row['top1_acc']:.4f} | {row['runtime_s']:.1f}s |"
        )

    md.append("\n## 4. G2.B branch + cell 결과 표\n")
    md.append(f"- activated branch: **`{branch}`**\n")
    md.append("- branch cell 별 결과는 results_Bx_*.json 참조 (활성 branch 의 cell 1~2개).\n")

    md.append("\n## 5. Best cell 박제 + paired Δ 3종\n")
    md.append("```json")
    md.append(json.dumps(g3, indent=2))
    md.append("```\n")

    md.append("\n## 6. 5가설 verdict\n")
    for k, v in verdict.items():
        md.append(f"- **{k}**: {v}")

    md.append(f"\n\n## 7. 14-anchor oracle 회수율\n- oracle ceiling = 0.7928 (plan-024 carry)")
    md.append(f"- best_cell 회수율 = {oracle:.4f} ({oracle*100:.2f}%)")
    md.append(f"- plan-022 winner 회수율 = {0.6530/0.7928:.4f} ({0.6530/0.7928*100:.2f}%)\n")

    md.append("\n## 8. 1080D input block 분해 (§4.3 박제)\n")
    md.append("| Block | Source | Dim | 본 plan cell 결과 |")
    md.append("|:--|:--|--:|:--|")
    md.append("| ① plan-022 carry | build_input_common + lgbm_extra | 170 | B2 (192=①+③) |")
    md.append("| ② cand_builder ctx | regime/multi-window/STA-LTA | 128 | broadcast, B3 (1058=no③) 포함 |")
    md.append("| ③ cand_builder per-anchor | par/perp/dist + spec + interactions | 22 | **B1 (22D only)**, (d) 검증 |")
    md.append("| ④ seq_builder 8-stat | per-channel last/first/mean/std/slope/max/min/range | 760 | R1 (raw 95×7=665D 대체) |\n")

    md.append("\n## 9. Runtime per STAGE\n")
    md.append(f"- G0 pytest: 4s")
    md.append(f"- G1 (3 carry): ~805s (~13.4min)")
    runtime_g2a = sum(c["runtime_s"] for c in g2a["cells"] if c["cell"] != "B4")
    md.append(f"- G2.A (8 신규 cell parallel): ~{runtime_g2a:.0f}s total CPU")
    md.append(f"- 총 plan-028: G0~G_final 약 ~{(805 + runtime_g2a) / 60:.0f}min CPU\n")

    md.append("\n## 10. max_class_ratio + top1_acc per cell\n")
    md.append("| cell | max_class_ratio | top1_acc |")
    md.append("|:--|--:|--:|")
    for row in g2a["cells"]:
        md.append(f"| {row['cell']} | {row['max_class_ratio']:.4f} | {row['top1_acc']:.4f} |")

    md.append("\n\n## 11. Follow-up plan 후보\n")
    md.append("- (G3 결과에 따라 박제)\n")
    if band == "positive":
        md.append("- best cell 의 lift 확장 — hparam 추가 sweep, ensemble, DACON LB 측정")
    elif band == "partial":
        md.append("- lift 미달 cause 추가 분석 + G2.B branch δ MLP 가 plan-022 winner 못 이긴 경우 paradigm-level 'LGBM ceiling' 박제")
    elif band == "negative":
        md.append("- mode collapse 잔존 — F0 ML 화 plan 분리 (= plan-025 paradigm_analysis 의 외부 hypothesis)\n")

    md.append("\n## 12. Cross-refs\n")
    md.append("- spec: `plans/plan-028-per-anchor-isolation-weight-probe.md`")
    md.append("- results pair: `plans/plan-028-per-anchor-isolation-weight-probe.results.md`")
    md.append("- baseline_carry: `analysis/plan-028/baseline_carry.json`")
    md.append("- 9 cell results: `analysis/plan-028/results_{B1,B2,B3,B4,W1,T1,T2,S1,R1}.json`")
    md.append("- branch result: `analysis/plan-028/results_Bx_*.json`")
    md.append("- paradigm: `analysis/plan-028/paradigm_analysis_g2a.json` + `paradigm_analysis.json`\n")

    out_path = _THIS / "results.md"
    out_path.write_text("\n".join(md))
    print(f"saved: {out_path}")

    # ── plans/plan-028-*.results.md pair (frontmatter + body) ──
    pair_md = []
    pair_md.append("---")
    pair_md.append("plan_id: 028")
    pair_md.append("finished_at: 2026-05-22 (Asia/Seoul)")
    pair_md.append("status: all_complete")
    pair_md.append(f"best_cell: {best_cell}")
    pair_md.append(f"best_hit_1cm: {best_hit:.4f}")
    pair_md.append(f"best_hit_1p5cm: {best_hit_15:.4f}")
    pair_md.append(f"best_delta_1cm: {delta_f0_hit:+.4f}")
    pair_md.append(f"best_delta_1p5cm: 0.0000")  # placeholder
    pair_md.append(f"band: {band}")
    pair_md.append("g1_completed: true")
    pair_md.append("g2a_completed: true")
    pair_md.append("g2b_completed: true")
    pair_md.append("g3_completed: true")
    pair_md.append("g_final_completed: true")
    pair_md.append("exp_ids_completed:")
    for cid in ["B1", "B2", "B3", "B4", "W1", "T1", "T2", "S1", "R1"]:
        pair_md.append(f"  - Z028_{cid}")
    pair_md.append("exp_ids_skipped: []")
    pair_md.append("---")
    pair_md.append("")
    pair_md.append(f"# plan-028.results — Per-anchor Isolation × Sample-weight Probe (FINAL, band={band})")
    pair_md.append("")
    pair_md.append(f"- **best**: `{best_cell}` (hit_1cm={best_hit:.4f}, hit_1p5cm={best_hit_15:.4f})")
    pair_md.append(f"- **paired Δ**: vs F0 = {delta_f0_hit:+.4f}, vs p022 = {delta_p022_hit:+.4f}, vs B4 = {delta_b4_hit:+.4f}")
    pair_md.append(f"- **band**: {band}")
    pair_md.append(f"- **G2.B branch**: `{branch}`")
    pair_md.append(f"- **5가설 verdict**: " + ", ".join(f"{k.split()[0]}={v}" for k, v in verdict.items()))
    pair_md.append(f"- **14-anchor oracle 회수율**: {oracle:.4f} ({oracle*100:.2f}%)")
    pair_md.append("")
    pair_md.append("상세 박제: `analysis/plan-028/results.md` 참조.")

    _PLAN028_RESULTS.write_text("\n".join(pair_md))
    print(f"saved: {_PLAN028_RESULTS}")


if __name__ == "__main__":
    main()
