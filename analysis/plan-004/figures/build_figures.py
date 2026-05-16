"""Build 8 explanatory figures for the plan-004 newcomer-friendly PDF.

stdlib + numpy + matplotlib only (no pandas / torch).
Run: python3 analysis/plan-004/figures/build_figures.py
Output: analysis/plan-004/figures/fig0{1..8}_*.png
"""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "data"
OUT = REPO / "analysis/plan-004/figures"
OUT.mkdir(parents=True, exist_ok=True)

# Use a font that handles Korean if available; fall back gracefully.
for cand in ("AppleGothic", "Apple SD Gothic Neo", "NanumGothic", "Noto Sans CJK KR"):
    if cand in {f.name for f in matplotlib.font_manager.fontManager.ttflist}:
        plt.rcParams["font.family"] = cand
        break
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 150
plt.rcParams["savefig.bbox"] = "tight"


def load_train_sequence(sample_id):
    """Returns (11, 4) array: [timestep_ms, x, y, z]."""
    path = DATA / "train" / f"{sample_id}.csv"
    rows = []
    with open(path) as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append([float(row["timestep_ms"]), float(row["x"]),
                         float(row["y"]), float(row["z"])])
    return np.array(rows)


def load_train_labels():
    """Returns dict id -> (x, y, z)."""
    out = {}
    with open(DATA / "train_labels.csv") as f:
        r = csv.DictReader(f)
        for row in r:
            out[row["id"]] = np.array([float(row["x"]), float(row["y"]),
                                       float(row["z"])])
    return out


# ============================================================
# Figure 1 — task timeline
# ============================================================
def fig01_task_timeline():
    fig, ax = plt.subplots(figsize=(10, 3.2))
    obs_t = np.arange(-400, 1, 40)
    pred_t = 80
    ax.scatter(obs_t, np.zeros_like(obs_t), s=120, c="#2b7bb9", zorder=3,
               label="관측 (Observation, 11 points)")
    ax.scatter([pred_t], [0], s=300, marker="*", c="#e74c3c", zorder=4,
               label="예측 대상 (Prediction target, +80 ms)")
    ax.axvline(0, color="#888", linestyle="--", linewidth=0.8)
    ax.axhline(0, color="#222", linewidth=1)
    ax.annotate("관측 종료 (t=0)", xy=(0, 0), xytext=(0, 0.35), ha="center",
                fontsize=9, color="#444")
    ax.annotate("+80 ms 후 좌표\n(예측 대상)", xy=(80, 0), xytext=(80, 0.45),
                ha="center", fontsize=10, color="#c0392b",
                arrowprops=dict(arrowstyle="->", color="#c0392b"))
    for t in obs_t:
        ax.annotate(f"{int(t)}", xy=(t, 0), xytext=(t, -0.25), ha="center",
                    fontsize=7, color="#555")
    ax.set_xlim(-450, 150)
    ax.set_ylim(-0.6, 0.9)
    ax.set_yticks([])
    ax.set_xlabel("시간 (ms, 0 = 관측 종료 시점)")
    ax.set_title("Task: 11 LiDAR 관측점 → +80 ms 미래 좌표 예측\n"
                 "(Predict the (x, y, z) at +80 ms from 11 past observations at 40 ms intervals)",
                 fontsize=11)
    ax.legend(loc="upper left", fontsize=9)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    fig.savefig(OUT / "fig01_task_timeline.png")
    plt.close(fig)


# ============================================================
# Figure 2 — sample trajectories (3D scatter)
# ============================================================
def fig02_sample_trajectories():
    labels = load_train_labels()
    rng = np.random.default_rng(20260516)
    # 10 random training sample ids (TRAIN_00001 ~ TRAIN_10000)
    pick_ids = [f"TRAIN_{i:05d}" for i in rng.choice(np.arange(1, 10001),
                                                     size=8, replace=False)]
    fig = plt.figure(figsize=(9, 7.5))
    ax = fig.add_subplot(111, projection="3d")
    colors = plt.cm.tab10(np.linspace(0, 1, len(pick_ids)))
    for sid, c in zip(pick_ids, colors):
        seq = load_train_sequence(sid)
        ax.plot(seq[:, 1], seq[:, 2], seq[:, 3], "-o", color=c,
                markersize=3, linewidth=1.2, alpha=0.85)
        gt = labels[sid]
        # connect last observation to ground truth
        ax.plot([seq[-1, 1], gt[0]], [seq[-1, 2], gt[1]], [seq[-1, 3], gt[2]],
                "--", color=c, linewidth=1.0, alpha=0.6)
        ax.scatter([gt[0]], [gt[1]], [gt[2]], marker="*", s=120, color=c,
                   edgecolors="black", linewidths=0.5)
    ax.set_xlabel("x (m, forward)")
    ax.set_ylabel("y (m, left)")
    ax.set_zlabel("z (m, up)")
    ax.set_title("Sample Trajectories (8 random training samples)\n"
                 "● = 관측 11점,  ★ = 정답 (+80 ms),  실선 = 관측,  점선 = 마지막 → 정답",
                 fontsize=10)
    fig.savefig(OUT / "fig02_sample_trajectories.png")
    plt.close(fig)


# ============================================================
# Figure 3 — architecture block diagram
# ============================================================
def _box(ax, xy, w, h, text, fc="#e8f1fb", ec="#2b7bb9", fontsize=9):
    x, y = xy
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.04",
                         linewidth=1.3, facecolor=fc, edgecolor=ec)
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, wrap=True)


def _arrow(ax, p1, p2, color="#444", lw=1.2):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=14,
                                 color=color, linewidth=lw))


def fig03_architecture():
    fig, ax = plt.subplots(figsize=(13, 6.8))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 7)
    ax.set_axis_off()

    # Title
    ax.text(6.5, 6.7, "Plan-004 Architecture (Physics Ladder)",
            fontsize=14, ha="center", weight="bold")
    ax.text(6.5, 6.3,
            "신경망 = 후보 선택 (ranker)   ·   물리 공식 = 후보 좌표 (regressor)   ·   Tiny corrector = 1cm boundary 회수",
            fontsize=9, ha="center", color="#555")

    # Left column: input
    _box(ax, (0.2, 2.8), 1.9, 0.9,
         "입력 Input\n(11×3 좌표)\nLiDAR observations",
         fc="#fdf3e6", ec="#e67e22")

    # Mid-left: 27 candidates
    _box(ax, (2.5, 2.8), 2.4, 0.9,
         "27 Physics Candidates\np0, acc, frenet,\njerk, latency …",
         fc="#e8f5e8", ec="#27ae60")

    # Middle col (3 boxes stacked): selector / regime bias / physics bias
    _box(ax, (5.4, 4.2), 2.5, 1.0,
         "Attn-GRU Selector\nhidden=48, 2-layer GRU\n+ attention head",
         fc="#e8f1fb", ec="#2b7bb9")
    _box(ax, (5.4, 2.8), 2.5, 1.0,
         "18-Regime × 27 Bias\n(empirical Bayes\nshrinkage table)",
         fc="#f5e8fb", ec="#8e44ad")
    _box(ax, (5.4, 1.4), 2.5, 1.0,
         "Physics bias\n(prior 0.65)",
         fc="#fdf3e6", ec="#e67e22")

    # Right col: softmax blend → corrector
    _box(ax, (8.4, 3.0), 2.4, 1.7,
         "Softmax + Bias 가산\n→ soft blend\n(27 후보 가중 평균)",
         fc="#e8f1fb", ec="#2b7bb9", fontsize=9)
    _box(ax, (8.4, 0.9), 2.4, 1.5,
         "Tiny MLP Corrector\nFrenet local frame\n±0.6 cm cap\nzero-init delta",
         fc="#fde8e8", ec="#c0392b", fontsize=9)

    # Far right: output
    _box(ax, (11.2, 2.8), 1.6, 1.0,
         "최종 Output\n(x, y, z)\nat +80 ms",
         fc="#fff8c5", ec="#b58900", fontsize=10)

    # Arrows: input → candidates → 3 middle boxes
    _arrow(ax, (2.1, 3.25), (2.5, 3.25))
    _arrow(ax, (4.9, 3.25), (5.4, 4.7))
    _arrow(ax, (4.9, 3.25), (5.4, 3.3))
    _arrow(ax, (4.9, 3.25), (5.4, 1.9))
    # 3 middle boxes → softmax
    _arrow(ax, (7.9, 4.7), (8.4, 4.2))
    _arrow(ax, (7.9, 3.3), (8.4, 3.85))
    _arrow(ax, (7.9, 1.9), (8.4, 3.5))
    # Softmax → Corrector (downward, red)
    _arrow(ax, (9.6, 3.0), (9.6, 2.4), color="#c0392b")
    # Softmax → output (right)
    _arrow(ax, (10.8, 3.85), (11.2, 3.3))
    # Corrector → output (right + up)
    _arrow(ax, (10.8, 1.65), (11.2, 3.0), color="#c0392b")

    fig.savefig(OUT / "fig03_architecture.png")
    plt.close(fig)


# ============================================================
# Figure 4 — example candidates for one real sample
# ============================================================
def _frenet_basis(obs):
    """obs shape (11, 3). Returns (tangent, normal, binormal) at last obs."""
    v = obs[-1] - obs[-2]
    nv = np.linalg.norm(v)
    t = v / (nv + 1e-9)
    a = (obs[-1] - 2 * obs[-2] + obs[-3])
    a_perp = a - np.dot(a, t) * t
    na = np.linalg.norm(a_perp)
    if na < 1e-6:
        # arbitrary perpendicular
        ref = np.array([0.0, 0.0, 1.0]) if abs(t[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
        n = ref - np.dot(ref, t) * t
        n /= np.linalg.norm(n) + 1e-9
    else:
        n = a_perp / na
    b = np.cross(t, n)
    return t, n, b


def _illustrative_candidates(obs):
    """Generate ~8 illustrative candidate positions for +80ms (2 steps).
    Mirrors the *categories* in plan-004 (constant-vel, accel, frenet, jerk, latency)
    without invoking the full 27-candidate codebase.
    """
    dt = 0.040  # s per step
    horizon = 2  # +80ms = 2 × 40ms
    last = obs[-1]
    v = (obs[-1] - obs[-2]) / dt
    a = (obs[-1] - 2 * obs[-2] + obs[-3]) / (dt ** 2)
    t, n, _b = _frenet_basis(obs)
    out = {}
    # p0_2d1 — constant velocity (no accel)
    out["p0_2d1\n(constant vel)"] = last + v * (horizon * dt)
    # acc_2d1_050 — half accel
    out["acc_2d1_050\n(half accel)"] = last + v * (horizon * dt) + 0.5 * a * (horizon * dt) ** 2 * 0.5
    # acc_2d1_060
    out["acc_2d1_060\n(0.6× accel)"] = last + v * (horizon * dt) + 0.5 * a * (horizon * dt) ** 2 * 0.6
    # frenet_par100_perp000 — pure tangent extension
    speed = np.linalg.norm(v)
    out["frenet_par100\nperp000"] = last + t * speed * (horizon * dt) * 1.0
    # frenet_par090_perp020 — slight curve
    out["frenet_par090\nperp020"] = last + t * speed * horizon * dt * 0.9 + n * speed * horizon * dt * 0.20
    # latency_short_085 — assume time scale 0.85
    out["latency_short\n(time×0.85)"] = last + v * (horizon * dt) * 0.85 + 0.5 * a * (horizon * dt) ** 2 * 0.85
    # latency_long_115 — assume time scale 1.15
    out["latency_long\n(time×1.15)"] = last + v * (horizon * dt) * 1.15 + 0.5 * a * (horizon * dt) ** 2 * 1.15
    # jerk_small_pos
    jerk_dir = a / (np.linalg.norm(a) + 1e-9)
    out["jerk_small_pos"] = last + v * horizon * dt + 0.5 * a * (horizon * dt) ** 2 + jerk_dir * 0.003
    return out


def fig04_candidates():
    labels = load_train_labels()
    # pick a moderately-curving trajectory: TRAIN_00007 (arbitrary, deterministic)
    sid = "TRAIN_00007"
    seq = load_train_sequence(sid)
    obs = seq[:, 1:4]
    gt = labels[sid]

    cands = _illustrative_candidates(obs)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(obs[:, 0], obs[:, 1], "-o", color="#2b7bb9", markersize=4,
            linewidth=1.5, label="관측 (11 points)")
    ax.scatter([gt[0]], [gt[1]], marker="*", s=400, color="#e74c3c",
               edgecolors="black", linewidths=0.8, zorder=5,
               label=f"정답 ground truth ({sid})")

    # draw 1cm hit circle around ground truth
    circle = plt.Circle((gt[0], gt[1]), 0.01, fill=False, color="#c0392b",
                        linestyle="--", linewidth=1.0, label="hit radius = 1 cm")
    ax.add_patch(circle)

    colors = plt.cm.tab10(np.linspace(0, 1, len(cands)))
    for (name, p), c in zip(cands.items(), colors):
        ax.scatter([p[0]], [p[1]], s=70, color=c, edgecolors="black",
                   linewidths=0.3, zorder=4)
        ax.annotate(name, xy=(p[0], p[1]), xytext=(8, 6),
                    textcoords="offset points", fontsize=7, color=c)

    ax.set_xlabel("x (m, forward)")
    ax.set_ylabel("y (m, left)")
    ax.set_title("Figure 4. 같은 입력에 대한 *8개 대표 candidate* 의 분포 (top-down view)\n"
                 "(plan-004의 27개 candidate 중 5개 family 의 대표 — full set 은 regime_distribution.md 참조)",
                 fontsize=10)
    ax.legend(loc="upper left", fontsize=9)
    ax.set_aspect("equal", adjustable="datalim")
    # zoom to show GT, last obs, and candidates clearly
    all_pts = np.vstack(list(cands.values()) + [gt, obs[-1]])
    pad = 0.02
    ax.set_xlim(all_pts[:, 0].min() - pad, all_pts[:, 0].max() + pad)
    ax.set_ylim(all_pts[:, 1].min() - pad, all_pts[:, 1].max() + pad)
    fig.savefig(OUT / "fig04_27_candidates.png")
    plt.close(fig)


# ============================================================
# Figure 5 — 18×27 regime hit-rate heatmap
# ============================================================
def fig05_regime_heatmap():
    rd = json.load(open(REPO / "analysis/plan-004/regime_distribution.json"))
    hit = np.array(rd["hit_table"])  # (18, 27)
    cand_names = rd["candidate_names"]
    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(hit, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(range(27))
    ax.set_xticklabels([f"c{i:02d} {n}" for i, n in enumerate(cand_names)],
                       rotation=80, fontsize=7, ha="right")
    ax.set_yticks(range(18))
    ax.set_yticklabels([f"r{r:02d} (n={c})"
                       for r, c in enumerate(rd["regime_histogram"])],
                       fontsize=9)
    # Annotate each cell with its value
    for r in range(18):
        for c in range(27):
            v = hit[r, c]
            color = "white" if v < 0.35 or v > 0.75 else "black"
            ax.text(c, r, f"{v:.2f}", ha="center", va="center",
                    fontsize=5.5, color=color)
    ax.set_xlabel("Candidate (27)", fontsize=10)
    ax.set_ylabel("Regime (18 = speed × curvature × speed_slope)", fontsize=10)
    ax.set_title("Figure 5. 18 regime × 27 candidate hit-rate (@ 1 cm) — 실측 train OOF\n"
                 "녹색=잘 맞춤, 빨강=거의 못 맞춤.  selector logit 에 이 표 기반 regime_bias(0.45) 가산.",
                 fontsize=10)
    cbar = fig.colorbar(im, ax=ax, fraction=0.030, pad=0.02)
    cbar.set_label("hit rate @ 1 cm", fontsize=9)
    fig.savefig(OUT / "fig05_regime_heatmap.png")
    plt.close(fig)


# ============================================================
# Figure 6 — 2-stage training pipeline
# ============================================================
def fig06_training_pipeline():
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.set_axis_off()

    ax.text(6, 5.6, "2-stage Sequential Training (누수 차단을 위한 OOF interface)",
            fontsize=12, ha="center", weight="bold")

    # Stage 1
    _box(ax, (0.3, 3.2), 2.2, 1.4,
         "Stage 1\n5-fold CV Selector\nfold k=0..4\n각 fold: train→val(OOF)",
         fc="#e8f1fb", ec="#2b7bb9", fontsize=9)
    _box(ax, (3.0, 3.2), 2.2, 1.4,
         "OOF Score Bank\noof_selector_scores.npz\nshape (N_train, 27)",
         fc="#fff8c5", ec="#b58900", fontsize=9)
    _box(ax, (5.7, 3.2), 2.2, 1.4,
         "Stage 2\nCorrector full-fit\non *frozen* selector\nscores (zero leak)",
         fc="#fde8e8", ec="#c0392b", fontsize=9)
    _box(ax, (8.4, 3.2), 3.3, 1.4,
         "최종 산출\nsubmission_boundary_tiny_soft.csv\n(soft probability-weighted blend)",
         fc="#fff8c5", ec="#b58900", fontsize=9)
    _arrow(ax, (2.5, 3.9), (3.0, 3.9))
    _arrow(ax, (5.2, 3.9), (5.7, 3.9))
    _arrow(ax, (7.9, 3.9), (8.4, 3.9))

    # Bottom row: why 2-stage
    _box(ax, (0.3, 0.8), 5.5, 1.5,
         "[X] End-to-end 학습\nCorrector 가 selector 의 train-set\noverfit score 를 보고 학습 -> 누수\n-> OOF metric 신뢰 X",
         fc="#fde8e8", ec="#c0392b", fontsize=9)
    _box(ax, (6.2, 0.8), 5.5, 1.5,
         "[O] 2-stage Sequential\nSelector OOF score = 'fold k 의 val 은\n다른 fold 들의 train 으로 만든 모델 점수'\n-> 후속 corrector 학습 신호가 공정",
         fc="#e8f5e8", ec="#27ae60", fontsize=9)

    fig.savefig(OUT / "fig06_training_pipeline.png")
    plt.close(fig)


# ============================================================
# Figure 7 — L2 loss vs hit@1cm metric mismatch
# ============================================================
def fig07_metric_vs_l2():
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    rng = np.random.default_rng(20260516)
    # Scenario A: predictions tightly clustered around GT but with one outlier
    gt = np.array([0.0, 0.0])
    pred_A = rng.normal(0, 0.006, size=(50, 2))  # mostly within 1cm
    pred_A[0] = [0.05, 0.05]  # outlier pulls L2 mean up
    # Scenario B: same total L2 mean, but more samples just outside 1cm
    pred_B = rng.normal(0, 0.014, size=(50, 2))  # spread, fewer hits

    for ax, pred, title, color in [(axes[0], pred_A, "A. 1개 outlier + 49개 hit",
                                    "#2b7bb9"),
                                   (axes[1], pred_B, "B. 다수가 1cm 살짝 바깥",
                                    "#e67e22")]:
        ax.scatter(pred[:, 0], pred[:, 1], color=color, s=25, alpha=0.7,
                   edgecolors="black", linewidths=0.3)
        ax.scatter([0], [0], marker="*", s=300, color="#e74c3c",
                   edgecolors="black", linewidths=0.8, label="정답 GT", zorder=5)
        circle = plt.Circle((0, 0), 0.01, fill=False, color="#c0392b",
                            linestyle="--", linewidth=1.2)
        ax.add_patch(circle)
        ax.set_xlim(-0.06, 0.06)
        ax.set_ylim(-0.06, 0.06)
        ax.set_aspect("equal")
        ax.set_xlabel("x error (m)")
        ax.set_ylabel("y error (m)")
        l2_mean = np.linalg.norm(pred, axis=1).mean()
        hit_rate = (np.linalg.norm(pred, axis=1) <= 0.01).mean()
        ax.set_title(f"{title}\nL2 mean = {l2_mean:.4f} m  ·  hit@1cm = {hit_rate:.2f}",
                     fontsize=10)
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(alpha=0.3)

    fig.suptitle("Figure 7. L2 평균 vs hit@1cm: 평균은 비슷해도 hit 가 갈린다\n"
                 "(평균 거리 최소화 ≠ hit rate 최대화 — 메트릭이 설계를 결정)",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "fig07_metric_vs_l2.png")
    plt.close(fig)


# ============================================================
# Figure 8 — corrector lift bar chart
# ============================================================
def fig08_corrector_lift():
    bars = [
        ("기존 best\n(B001 polyfit)", 0.6000, "#aaaaaa"),
        ("Selector OOF\n(soft)", 0.6624, "#2b7bb9"),
        ("Corrector OOF\n(soft)", 0.6718, "#c0392b"),
        ("Final LB\n(0.6806)", 0.6806, "#27ae60"),
        ("Oracle bound\n(best-of-27)", 0.7277, "#888888"),
    ]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    xs = np.arange(len(bars))
    vals = [v for _, v, _ in bars]
    cols = [c for _, _, c in bars]
    bars_plot = ax.bar(xs, vals, color=cols, edgecolor="black", linewidth=0.6)
    for x, v, (label, _, _) in zip(xs, vals, bars):
        ax.text(x, v + 0.005, f"{v:.4f}", ha="center", fontsize=10)
    ax.set_xticks(xs)
    ax.set_xticklabels([b[0] for b in bars], fontsize=9)
    ax.set_ylim(0.55, 0.78)
    ax.set_ylabel("hit rate @ 1 cm")
    ax.set_title("Figure 8. plan-004 가 만든 lift 의 분해\n"
                 "(0.60 → 0.6806 = +0.0806;  남은 oracle 헤드룸 = 0.0471)",
                 fontsize=10)
    ax.axhline(0.7277, color="#888", linestyle=":", linewidth=1)
    ax.text(4.4, 0.7305, "oracle = 후보 27개 중 항상 best 선택 시", fontsize=8,
            color="#666", ha="right")
    ax.grid(axis="y", alpha=0.3)
    fig.savefig(OUT / "fig08_corrector_lift.png")
    plt.close(fig)


# ============================================================
# Driver
# ============================================================
def main():
    print("Building 8 figures →", OUT)
    fig01_task_timeline();             print("  ✓ fig01_task_timeline.png")
    fig02_sample_trajectories();       print("  ✓ fig02_sample_trajectories.png")
    fig03_architecture();              print("  ✓ fig03_architecture.png")
    fig04_candidates();                print("  ✓ fig04_27_candidates.png")
    fig05_regime_heatmap();            print("  ✓ fig05_regime_heatmap.png")
    fig06_training_pipeline();         print("  ✓ fig06_training_pipeline.png")
    fig07_metric_vs_l2();              print("  ✓ fig07_metric_vs_l2.png")
    fig08_corrector_lift();            print("  ✓ fig08_corrector_lift.png")
    print("Done.")


if __name__ == "__main__":
    main()
