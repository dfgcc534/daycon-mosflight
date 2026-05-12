"""plan-007 post-hoc — per-candidate hit@1cm on train (10K, end_idx=10, horizon=2).

For each of the 27 candidates in plan-004's CANDIDATES list (= plan-006 후보 풀), measure
the fraction of train samples where ||cand_pred - train_y|| ≤ 0.01m.

Outputs:
  - analysis/plan-007/per_candidate_hit.json
  - analysis/plan-007/per_candidate_hit.png  (bar chart, 27 candidates)
  - analysis/plan-007/per_candidate_hit.md   (table + 해석)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.pb_0_6822 import selector

DATA_ROOT = Path("data")
OUT_DIR = Path("analysis/plan-007")
R_HIT = 0.01


def main() -> None:
    ids, train_y = selector.read_labels(DATA_ROOT / "train_labels.csv")
    train_x = selector.load_stack(DATA_ROOT / "train", ids)
    cands = selector.make_candidates(train_x, train_x.shape[1] - 1, horizon=2)  # (N, 27, 3)
    n_samples, n_cand, _ = cands.shape
    assert n_cand == 27

    err = np.linalg.norm(cands - train_y[:, None, :], axis=2)  # (N, 27)
    hit_per = (err <= R_HIT).mean(axis=0)                       # (27,)
    names = [spec.name for spec in selector.CANDIDATES]

    # Sort by hit (descending) for visual clarity
    order = np.argsort(-hit_per)
    sorted_hit = hit_per[order]
    sorted_names = [names[i] for i in order]
    rank_map = {names[i]: int(order.tolist().index(i)) for i in range(n_cand)}

    # ── Plot ──
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(n_cand)
    bars = ax.bar(x, sorted_hit, color="steelblue", edgecolor="black", linewidth=0.5)

    # Highlight plan-006's chosen single formula (frenet_par120_perp_neg020 = index 17)
    best_name = "frenet_par120_perp_neg020"
    if best_name in sorted_names:
        i = sorted_names.index(best_name)
        bars[i].set_color("crimson")
        bars[i].set_edgecolor("black")

    # Annotate hit values on top
    for bar, h in zip(bars, sorted_hit):
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005, f"{h:.3f}",
                ha="center", va="bottom", fontsize=7, rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels(sorted_names, rotation=75, ha="right", fontsize=8)
    ax.set_ylabel(f"hit@{R_HIT}m (fraction of {n_samples:,} train samples)")
    ax.set_title(
        f"plan-006 27-candidate per-formula hit@{R_HIT}m (train end_idx=10, horizon=2)\n"
        f"Red bar = plan-006 chosen single formula `frenet_par120_perp_neg020` (index 17)"
    )
    ax.axhline(hit_per.mean(), ls="--", color="gray", lw=0.8,
               label=f"mean across 27 = {hit_per.mean():.3f}")
    ax.axhline(hit_per.max(), ls=":", color="darkgreen", lw=0.8,
               label=f"best individual = {hit_per.max():.3f}")
    ax.set_ylim(0, max(sorted_hit) * 1.18)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    png_path = OUT_DIR / "per_candidate_hit.png"
    fig.savefig(png_path, dpi=140)
    plt.close(fig)

    # ── JSON + MD ──
    result = {
        "r_hit": R_HIT,
        "n_samples": int(n_samples),
        "end_idx": int(train_x.shape[1] - 1),
        "horizon": 2,
        "candidates": [
            {"index": int(i), "name": names[i], "hit": float(hit_per[i]),
             "rank": int(rank_map[names[i]])}
            for i in range(n_cand)
        ],
        "summary": {
            "best": {"name": sorted_names[0], "hit": float(sorted_hit[0])},
            "worst": {"name": sorted_names[-1], "hit": float(sorted_hit[-1])},
            "mean": float(hit_per.mean()),
            "median": float(np.median(hit_per)),
            "std": float(hit_per.std()),
            "plan_006_chosen": {
                "name": best_name,
                "hit": float(hit_per[17]),
                "rank": rank_map[best_name] + 1,  # 1-indexed
            },
        },
    }
    (OUT_DIR / "per_candidate_hit.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))

    md_lines = [
        "# plan-006 27-candidate per-formula hit@1cm",
        "",
        f"train {n_samples:,} samples, end_idx={result['end_idx']}, horizon={result['horizon']}",
        "",
        f"**plan-006 chosen** = `{best_name}` (idx 17) → hit = {hit_per[17]:.4f}, rank = {rank_map[best_name] + 1}/{n_cand}",
        "",
        "| rank | name | hit@1cm | Δ from chosen |",
        "|---|---|---|---|",
    ]
    for rk, (name, h) in enumerate(zip(sorted_names, sorted_hit), start=1):
        delta = h - hit_per[17]
        marker = " 🔴" if name == best_name else ""
        md_lines.append(f"| {rk} | `{name}`{marker} | {h:.4f} | {delta:+.4f} |")
    md_lines.append("")
    md_lines.append(f"- mean across 27 = {hit_per.mean():.4f}")
    md_lines.append(f"- best individual = {hit_per.max():.4f} (`{sorted_names[0]}`)")
    md_lines.append(f"- worst individual = {hit_per.min():.4f} (`{sorted_names[-1]}`)")
    md_lines.append(f"- std across 27 = {hit_per.std():.4f}")
    (OUT_DIR / "per_candidate_hit.md").write_text("\n".join(md_lines) + "\n")

    # ── stdout summary ──
    print(f"saved: {png_path}")
    print(f"top 5:")
    for rk in range(5):
        print(f"  {rk+1}. {sorted_names[rk]} → {sorted_hit[rk]:.4f}")
    print(f"plan-006 chosen `{best_name}` (idx 17): hit={hit_per[17]:.4f}, rank={rank_map[best_name]+1}/27")
    print(f"mean across 27 = {hit_per.mean():.4f}, oracle (best of 27) = {hit_per.max():.4f}")


if __name__ == "__main__":
    main()
