"""plan-028 c19 — G3 best_cell selection (G2.A + G2.B 통합 argmax).

§4.6 tiebreaker: hit_1cm > paired Δ_p022 > runtime.
band: positive (>0.6531 AND Δ>0) / partial / negative / tight (paired Δ resolve).
"""
from __future__ import annotations

import json
from pathlib import Path

_THIS = Path(__file__).resolve().parent


def main():
    # G2.A 9 cell
    g2a_cells = ["B1", "B2", "B3", "B4", "W1", "T1", "T2", "S1", "R1"]
    cells = {}
    for cid in g2a_cells:
        p = _THIS / f"results_{cid}.json"
        if p.exists():
            with open(p) as f:
                cells[cid] = json.load(f)

    # G2.B branch cells (있으면)
    g2a_path = _THIS / "paradigm_analysis_g2a.json"
    branch = None
    if g2a_path.exists():
        with open(g2a_path) as f:
            g2a = json.load(f)
        branch = g2a["g2b_branch"]
    # branch cell results — Bx_1, Bx_2 (있는 것만)
    for bx in ["Bx_1", "Bx_2"]:
        p = _THIS / f"results_{bx}.json"
        if p.exists():
            with open(p) as f:
                cells[bx] = json.load(f)

    # argmax(hit_1cm)
    F0 = 0.6320
    P022 = 0.6530  # G1 reproduce (spec hard evidence 0.6531 와 ε 차)
    B4_hit = cells["B4"]["hit_1cm"]

    best_cell = max(cells.keys(),
                    key=lambda c: (cells[c]["hit_1cm"], cells[c]["hit_1cm"] - P022, -cells[c]["runtime_s"]))
    best = cells[best_cell]
    best_hit = best["hit_1cm"]
    best_hit_15 = best["hit_1p5cm"]
    delta_p022 = best_hit - P022
    delta_f0 = best_hit - F0
    delta_b4 = best_hit - B4_hit

    # band
    if best_hit > P022 and delta_p022 > 0:
        band = "positive"
    elif 0.6526 <= best_hit <= 0.6536:
        band = "tight" if delta_p022 > 0 else "partial"
        if band == "tight":
            band = "positive" if delta_p022 > 0 else "partial"
    elif best_hit > P022 and delta_p022 <= 0:
        band = "partial"
    elif 0.6320 < best_hit <= P022:
        band = "partial"
    else:
        band = "negative"

    out = {
        "best_cell": best_cell,
        "best_hit_1cm": best_hit,
        "best_hit_1p5cm": best_hit_15,
        "paired_delta": {
            "vs_f0": {"hit_1cm": delta_f0},
            "vs_p022": {"hit_1cm": delta_p022},
            "vs_p025_C1": {"hit_1cm": delta_b4},
        },
        "oracle_recovery": best_hit / 0.7928,
        "hypothesis_verdict": g2a["hypothesis_verdict"] if g2a_path.exists() else None,
        "g2b_branch": branch,
        "band": band,
    }
    out_path = _THIS / "paradigm_analysis.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"best_cell: {best_cell}  hit_1cm={best_hit:.4f}  hit_1p5cm={best_hit_15:.4f}")
    print(f"  Δ vs F0:   {delta_f0:+.4f}")
    print(f"  Δ vs p022: {delta_p022:+.4f}")
    print(f"  Δ vs B4:   {delta_b4:+.4f}")
    print(f"  oracle:    {best_hit / 0.7928:.4f} (= best/0.7928)")
    print(f"  band:      {band}")
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
