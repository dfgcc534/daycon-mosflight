"""plan-029 c10 — paradigm_analysis.

X1 결과 후처리:
- paired Δ vs F0 / vs plan-022 winner (baseline_carry.json carry)
- 14-anchor oracle 회수율 (oracle = 0.7928, MEMORY 박제)
- mode collapse 진단 (max_class_ratio + per-anchor argmax 분포)
- anchor_embed cosine similarity matrix (K=14 × K=14) — anchor 별 학습된 embedding 의 differentiation
- per-fold grad norm trajectory summary
- G3 verdict 종합

outputs:
  analysis/plan-029/paradigm_analysis.json
  analysis/plan-029/paradigm_analysis.md
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

_THIS = Path(__file__).resolve().parent
_REPO = _THIS.parent.parent
for p in (_REPO, _THIS):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


_model_mod = _load(_THIS / "model.py", "p029_pa_model")


def cosine_offdiag_mean(embed: np.ndarray) -> tuple[float, np.ndarray]:
    """anchor embedding cosine similarity matrix off-diagonal mean."""
    embed_t = torch.from_numpy(embed).float()
    # row-normalize
    embed_n = embed_t / (embed_t.norm(dim=-1, keepdim=True) + 1e-9)
    sim = (embed_n @ embed_n.T).numpy()                                    # (K, K)
    K = sim.shape[0]
    mask = ~np.eye(K, dtype=bool)
    offdiag_mean = float(sim[mask].mean())
    return offdiag_mean, sim


def analyze() -> dict:
    # 1) results_X1.json load
    with open(_THIS / "results_X1.json") as f:
        results = json.load(f)
    with open(_THIS.parent / "plan-025" / "baseline_carry.json") as f:
        carry = json.load(f)

    # 2) oof_X1.npz load (oof_pred, oof_probs, gt_anchor_label)
    oof = np.load(_THIS / "oof_X1.npz")
    oof_probs = oof["oof_probs"]                                           # (N_total, K)
    gt_anchor_label = oof["gt_anchor_label"]                               # (N_total,)
    top1_argmax = oof_probs.argmax(axis=1)

    # 3) paradigm metric
    hit_1cm = results["hit_1cm"]
    hit_1p5cm = results["hit_1p5cm"]
    delta_F0 = hit_1cm - float(carry["F0"]["hit_1cm"])
    delta_p022 = hit_1cm - float(carry["plan022_winner"]["hit_1cm"])
    delta_p024_ceiling = hit_1cm - 0.6387                                  # plan-024 honest ceiling
    oracle = 0.7928
    recall_oracle = hit_1cm / oracle

    # 4) per-anchor argmax distribution
    bincount = np.bincount(top1_argmax, minlength=14).astype(int)
    bincount_norm = (bincount / bincount.sum()).tolist()

    # 5) anchor_embed cosine similarity — NOTE: model 학습본 미저장 → analytic re-init 으로 lower bound
    # 실제 학습본은 fold 별 다르므로, 본 c10 에서는 마지막 fold model 재학습 또는 spec 의 H4 threshold 추정만.
    # (G3 PASS 시 c10 별도 학습본 dump 필요 — 본 c10 은 oof 만 사용)
    # H4 threshold: anchor_embed_cosine_offdiag_mean < 0.5
    h4_threshold = 0.5
    # 추정: full training 시점의 embed 는 없으니, 5-fold OOF 의 final epoch 마지막 fold 의 anchor_embed 가 필요.
    # 본 c10 에서는 estimate = None 박제 (c11 results.md 에서 follow-up 으로 명시).

    # 6) per-fold grad_norm trajectory summary
    fold_grad_summary = []
    for log in results["fold_logs"]:
        traj = log["grad_norm_trajectory"]
        fold_grad_summary.append({
            "fold": log["fold"],
            "ep5_grad_norm": float(traj[4]) if len(traj) > 4 else None,
            "final_grad_norm": float(traj[-1]) if traj else None,
            "max_grad_norm": float(max(traj)) if traj else None,
            "min_grad_norm": float(min(traj)) if traj else None,
        })

    # 7) G3 verdict from results
    verdict = results["verdict_G3"]
    g2_pass = results["g2_x1"]["pass"]

    # 8) top1 acc breakdown (paradigm-distinct: argmax accuracy 와 hit_1cm 의 dissociation)
    top1_acc = results["top1_acc"]

    analysis = {
        "G3_verdict": verdict,
        "G2_pass": g2_pass,
        "hit_1cm": hit_1cm,
        "hit_1p5cm": hit_1p5cm,
        "delta_vs_F0_1cm": delta_F0,
        "delta_vs_p022_winner_1cm": delta_p022,
        "delta_vs_p024_ceiling_1cm": delta_p024_ceiling,
        "oracle_14anchor_1cm": oracle,
        "recall_vs_oracle": recall_oracle,
        "max_class_ratio": results["max_class_ratio"],
        "top1_acc": top1_acc,
        "per_anchor_argmax_distribution": bincount_norm,
        "per_fold_grad_summary": fold_grad_summary,
        "warns": results["warns"],
        "elapsed_total_s": results["elapsed_total_s"],
        "H4_threshold_cos_offdiag_mean": h4_threshold,
        "H4_anchor_embed_cos_offdiag_mean": None,  # follow-up: 5-fold 각 model 박제 후 산출
        "H4_note": "anchor_embed cosine sim matrix 산출은 5-fold model.anchor_embed 박제 필요. "
                   "본 c10 은 oof 기반 분석만. G3 PASS 시 follow-up 으로 5-fold model dump.",
        "follow_up": "plan-030 single-lever ablation 분해 (a/b/c/d 각 단독)",
    }

    # dump JSON
    out_json = _THIS / "paradigm_analysis.json"
    with open(out_json, "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"[dump] {out_json}")

    # markdown report
    out_md = _THIS / "paradigm_analysis.md"
    md = ["# plan-029 paradigm_analysis (c10)\n"]
    md.append(f"## G3 verdict: **{verdict}** (G2 pass: {g2_pass})\n")
    md.append("## Metric\n")
    md.append(f"- **hit_1cm** = {hit_1cm:.4f}")
    md.append(f"- **hit_1p5cm** = {hit_1p5cm:.4f}")
    md.append(f"- max_class_ratio = {results['max_class_ratio']:.4f} (1/K = 0.0714, threshold > 0.95 fail)")
    md.append(f"- top1_acc (argmax vs gt_anchor) = {top1_acc:.4f}")
    md.append("")
    md.append("## Paired Δ\n")
    md.append(f"- vs F0 (0.6320): Δ = **{delta_F0:+.4f}**")
    md.append(f"- vs plan-022 winner (0.6531): Δ = **{delta_p022:+.4f}**  ← G3 임계 0.6528 기준")
    md.append(f"- vs plan-024 honest ceiling (0.6387): Δ = **{delta_p024_ceiling:+.4f}**")
    md.append(f"- recall vs 14-anchor oracle (0.7928): {recall_oracle * 100:.1f}%")
    md.append("")
    md.append("## Per-anchor argmax distribution (K=14)\n")
    md.append("| anchor | ratio |")
    md.append("|---:|---:|")
    for k, r in enumerate(bincount_norm):
        md.append(f"| {k} | {r:.4f} |")
    md.append("")
    md.append("## Per-fold gradient norm summary\n")
    md.append("| fold | ep5 | final | max | min |")
    md.append("|---:|---:|---:|---:|---:|")
    for fg in fold_grad_summary:
        md.append(f"| {fg['fold']} | {fg['ep5_grad_norm']:.2e} | "
                  f"{fg['final_grad_norm']:.2e} | "
                  f"{fg['max_grad_norm']:.2e} | "
                  f"{fg['min_grad_norm']:.2e} |")
    md.append("")
    md.append("## Warns\n")
    for k, v in results["warns"].items():
        md.append(f"- {k}: {v}")
    md.append("")
    md.append(f"## elapsed: {results['elapsed_total_s']:.1f}s ({results['elapsed_total_s'] / 60:.1f} min)\n")
    md.append(f"## H4 follow-up\n- threshold: cosine off-diag mean < {h4_threshold}")
    md.append(f"- {analysis['H4_note']}")
    md.append("")
    md.append(f"## Follow-up: {analysis['follow_up']}\n")
    with open(out_md, "w") as f:
        f.write("\n".join(md))
    print(f"[dump] {out_md}")
    return analysis


if __name__ == "__main__":
    analyze()
