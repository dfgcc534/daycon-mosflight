"""plan-016 Path C 공통 runner (Feature B/C/D 단독, base=G1 since G2 dropped).

§7 / §8 / §9 spec carry. G3/G4/G5 driver 들이 import 해서 사용.

base config = G1 carry (E0c K=9 + boundary_weight, F0 frozen, monitor=val_hit, 5-seed × 5-fold).
변경 변수 = feature_flags 의 단일 True (B/C/D 중 하나).
threshold = +0.003 (vs G1, per §7.3 v1.5 fix: G2 dropped → base=G1).
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from src.io import load_all_samples, load_labels
from src.pb_0_6822 import plan014_paradigm as pp
from src.pb_0_6822 import plan016_ensemble as pe


PATH_A_SEEDS = [20260514, 20260515, 20260516, 20260517, 20260518]
PATH_C_DELTA_THRESHOLD = 0.003


def run_path_c(
    feature: str,
    feature_flag: dict[str, bool],
    expected_dim: int,
    exp_id: str,
    out_json: Path,
    run_dir: Path,
    g1_json: Path = Path("analysis/plan-016/g1_path_a.json"),
    epochs: int = 20,
    patience: int = pp.DEFAULT_PATIENCE,
):
    """Path C 단독 feature stage runner.

    Args:
        feature: "B" | "C" | "D"
        feature_flag: {"A":False, "B":False, "C":False, "D":False} 중 하나만 True.
        expected_dim: 10 (B) / 18 (C) / 15 (D)
    """
    g1 = json.loads(g1_json.read_text())
    g1_oof = g1["overall_oof_hit_1cm"]
    g1_lb = g1.get("lb_score")
    print(f"[plan-016 Path C-{feature}] G1 baseline: OOF={g1_oof:.4f} LB={g1_lb}", flush=True)
    print(f"  feature_flag={feature_flag} expected_dim={expected_dim}", flush=True)

    t_start = time.time()
    print(f"[plan-016 Path C-{feature}] loading data ...", flush=True)
    ids_train, X_train = load_all_samples("train")
    ids_test, X_test = load_all_samples("test")
    label_ids, Y_train = load_labels()
    assert ids_train == label_ids
    X_train = X_train.astype(np.float32); Y_train = Y_train.astype(np.float32)
    X_test = X_test.astype(np.float32)

    config_base = pp.TrainConfig(
        name=f"path_c_{feature.lower()}", K=9, encoder_name="bigru", codebook="kmeans",
        use_reg_head=True, use_hinge=False,
        temperature=0.03, r0_logit_prior=0.0, boundary_weight_on=True,
        lr=pp.DEFAULT_LR, batch_size=pp.DEFAULT_BATCH,
        epochs=epochs, patience=patience, seed=PATH_A_SEEDS[0],
        monitor="val_hit",  # G1 carry (G2 dropped → val_hit, NOT val_loss)
    )

    def progress(si, f, seed, res, elapsed):
        print(f"  seed={seed} fold={f}: val_hit={res['best_val_hit']:.4f} "
              f"epoch={res['best_epoch']}/{epochs} feature_dim={res['feature_dim']} "
              f"elapsed={elapsed:.1f}s", flush=True)

    print(f"\n[plan-016 Path C-{feature}] === 5-seed × 5-fold = 25 models ===", flush=True)
    ensemble_result = pe.run_multiseed_kfold_v2(
        ids_train, X_train, Y_train, ids_test, X_test,
        config_base=config_base, seeds=PATH_A_SEEDS,
        feature_flags=feature_flag,
        f0_function=pp.Plan014F0Function(),
        progress_cb=progress,
    )

    overall_oof = ensemble_result["overall_oof_hit_1cm"]
    delta_oof = overall_oof - g1_oof
    oof_pass = delta_oof >= PATH_C_DELTA_THRESHOLD
    per_seed_oof = ensemble_result["per_seed_oof_hit_1cm"]
    fold_oof = ensemble_result["fold_oof_hit_per_fold"]
    feature_dim = ensemble_result["feature_dim"]
    assert feature_dim == expected_dim, f"feature_dim mismatch: {feature_dim} != {expected_dim}"

    print(f"\n[plan-016 Path C-{feature}] === final ===", flush=True)
    print(f"  per-seed OOF = {per_seed_oof}", flush=True)
    print(f"  per-fold (seed-mean) OOF = {fold_oof}", flush=True)
    print(f"  multi-seed concat OOF = {overall_oof:.4f}", flush=True)
    print(f"  Δ vs G1 0.6452 = {delta_oof:+.4f} (threshold +{PATH_C_DELTA_THRESHOLD})", flush=True)
    print(f"  OOF Δ pass = {oof_pass}", flush=True)

    # Submission write (but submit decision deferred to user)
    run_dir.mkdir(parents=True, exist_ok=True)
    sample_sub = pd.read_csv("data/sample_submission.csv")
    sample_ids = sample_sub["id"].tolist()
    id_to_idx = {sid: i for i, sid in enumerate(ids_test)}
    test_pred = ensemble_result["test_pred"]
    ordered = np.array([test_pred[id_to_idx[sid]] for sid in sample_ids], dtype=np.float64)
    submission_path = run_dir / "submission.csv"
    df = pd.DataFrame({
        "id": sample_ids,
        "x": [f"{v:.6f}" for v in ordered[:, 0]],
        "y": [f"{v:.6f}" for v in ordered[:, 1]],
        "z": [f"{v:.6f}" for v in ordered[:, 2]],
    })
    df.to_csv(submission_path, index=False)
    print(f"  submission -> {submission_path}", flush=True)

    elapsed_total = time.time() - t_start
    artifact = {
        "exp_id": exp_id,
        "plan_version": "v1.5",
        "feature": feature,
        "feature_flag": feature_flag,
        "feature_dim": feature_dim,
        "config_base": asdict(config_base),
        "seeds": PATH_A_SEEDS,
        "n_models": 25,
        "per_seed_oof_hit_1cm": per_seed_oof,
        "fold_oof_hit_per_fold": fold_oof,
        "overall_oof_hit_1cm": overall_oof,
        "g1_oof": g1_oof,
        "g1_lb": g1_lb,
        "delta_oof_vs_g1": delta_oof,
        "delta_threshold": PATH_C_DELTA_THRESHOLD,
        "oof_pass": oof_pass,
        "lb_pass": None,
        "lb_score": None,
        "status": None,
        "fold_results": ensemble_result["fold_results"],
        "submission_path": str(submission_path),
        "elapsed_total_seconds": elapsed_total,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(artifact, indent=2, ensure_ascii=False))
    print(f"\n[plan-016 Path C-{feature}] elapsed_total={elapsed_total:.1f}s", flush=True)
    print(f"[plan-016 Path C-{feature}] artifact -> {out_json}", flush=True)
    return artifact
