"""plan-015 train function — plan-014 train_one_fold 의 wrapper.

plan-014 module *import* + make_seq_features_v2 사용. Plan014HybridHead 의
encoder input_dim 을 feature dim 에 맞춰 overwrite (plan-014 module 무수정).

§7.2 weight 재초기화 spec carry: 전체 model state Kaiming default +
seed=20260514, plan-014 weight transfer 안 함.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from src.pb_0_6822 import plan014_paradigm as pp
from src.pb_0_6822.plan015_features import make_seq_features_v2


def train_one_fold_v2(
    config: pp.TrainConfig,
    fold_id: int,
    X_train: np.ndarray, Y_train: np.ndarray,
    X_val: np.ndarray, Y_val: np.ndarray,
    f0_function: pp.Plan014F0Function,
    feature_flags: dict[str, bool],
    X_test: np.ndarray | None = None,
) -> dict[str, Any]:
    """plan-015 train_one_fold variant. plan-014 train_one_fold copy + feature swap.

    feature_flags = {"A": bool, "B": bool, "C": bool, "D": bool}.

    plan-016 carry: X_test 가 None 이 아니면 best-state restore 후 test prediction
    `(N_test, 3)` 을 `test_pred` 키로 반환.
    """
    cfg = config
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device(cfg.device)

    # Anchors (plan-014 carry: E0c K-Means K=9 per fold)
    anchors_local, R_train = pp.get_anchors_for_fold(
        cfg.codebook, X_train, Y_train, fold_id, K=cfg.K, f0_function=f0_function
    )
    R_val = pp.build_frenet_basis_3d(X_val, end_idx=cfg.end_idx) if cfg.codebook != "absolute" else None

    # Sequence features (plan-015 v2)
    seq_train = make_seq_features_v2(X_train, end_idx=cfg.end_idx, feature_flags=feature_flags)
    seq_val = make_seq_features_v2(X_val, end_idx=cfg.end_idx, feature_flags=feature_flags)
    feature_dim = seq_train.shape[-1]

    # F0 frozen
    F0_train_np = f0_function(X_train)
    F0_val_np = f0_function(X_val)

    # Boundary weight (E6 carry)
    sw_train = pp._boundary_weight(F0_train_np, Y_train) if cfg.boundary_weight_on else None

    # To torch
    seq_train_t = torch.from_numpy(seq_train).to(device)
    seq_val_t = torch.from_numpy(seq_val).to(device)
    Y_train_t = torch.from_numpy(Y_train.astype(np.float32)).to(device)
    Y_val_t = torch.from_numpy(Y_val.astype(np.float32)).to(device)
    F0_train_t = torch.from_numpy(F0_train_np.astype(np.float32)).to(device)
    F0_val_t = torch.from_numpy(F0_val_np.astype(np.float32)).to(device)
    anchors_t = torch.from_numpy(anchors_local.astype(np.float32)).to(device)
    R_train_t = torch.from_numpy(R_train.astype(np.float32)).to(device) if R_train is not None else None
    R_val_t = torch.from_numpy(R_val.astype(np.float32)).to(device) if R_val is not None else None
    sw_t = torch.from_numpy(sw_train).to(device) if sw_train is not None else None

    # Build corrector with feature_dim input (encoder input_dim overwrite)
    model = pp.Plan014HybridHead(K=anchors_local.shape[0], encoder_name=cfg.encoder_name).to(device)
    # Overwrite encoder with correct input_dim (Kaiming re-init, seed carry)
    torch.manual_seed(cfg.seed)
    model.encoder = pp.Plan014BiGRUEncoder(
        input_dim=feature_dim, hidden=128, num_layers=2, dropout=0.1
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    n_train = X_train.shape[0]

    # Initial val_hit
    model.eval()
    with torch.no_grad():
        hybrid_pred_init = model.hybrid_predict(
            seq_val_t, anchors_t, R_val_t, F0_val_t,
            temperature=cfg.temperature, use_reg_head=cfg.use_reg_head,
            r0_logit_prior=cfg.r0_logit_prior,
        )
        initial_val_hit = pp._hit_rate_torch(hybrid_pred_init, Y_val_t, threshold_m=0.01)

    best_val_hit = -1.0
    best_val_loss = float("inf")
    best_state = None
    best_epoch = -1
    patience_left = cfg.patience
    epoch_log: list[dict] = []

    for epoch in range(cfg.epochs):
        model.train()
        perm = torch.randperm(n_train, device=device)
        epoch_losses = []
        for i in range(0, n_train, cfg.batch_size):
            idx = perm[i:i + cfg.batch_size]
            seq_b = seq_train_t[idx]
            Y_b = Y_train_t[idx]
            F0_b = F0_train_t[idx]
            anchors_world_b = (anchors_t.unsqueeze(0).expand(idx.shape[0], -1, -1)
                                if R_train_t is None
                                else torch.einsum("bij,kj->bki", R_train_t[idx], anchors_t))
            sw_b = sw_t[idx] if sw_t is not None else None

            opt.zero_grad()
            logits, reg_offset = model.forward(seq_b, anchors_t)
            loss, _ = pp.hybrid_combined_loss(
                logits, reg_offset, F0_b, anchors_world_b, Y_b,
                sample_weight=sw_b,
                use_hinge=cfg.use_hinge, use_reg_head=cfg.use_reg_head,
                temperature=cfg.temperature,
            )
            loss.backward()
            opt.step()
            epoch_losses.append(float(loss.item()))

        train_loss = float(np.mean(epoch_losses))

        # Validation
        model.eval()
        with torch.no_grad():
            anchors_val_world = (anchors_t.unsqueeze(0).expand(seq_val_t.shape[0], -1, -1)
                                  if R_val_t is None
                                  else torch.einsum("bij,kj->bki", R_val_t, anchors_t))
            logits_v, reg_v = model.forward(seq_val_t, anchors_t)
            val_loss, _ = pp.hybrid_combined_loss(
                logits_v, reg_v, F0_val_t, anchors_val_world, Y_val_t,
                use_hinge=cfg.use_hinge, use_reg_head=cfg.use_reg_head,
                temperature=cfg.temperature,
            )
            hybrid_pred_val = model.hybrid_predict(
                seq_val_t, anchors_t, R_val_t, F0_val_t,
                temperature=cfg.temperature, use_reg_head=cfg.use_reg_head,
                r0_logit_prior=cfg.r0_logit_prior,
            )
            val_hit = pp._hit_rate_torch(hybrid_pred_val, Y_val_t, threshold_m=0.01)
            dcm = float(torch.linalg.norm(hybrid_pred_val - F0_val_t, dim=1).mean().item())

        val_loss_f = float(val_loss.item())
        epoch_log.append({"epoch": epoch + 1, "train_loss": train_loss,
                          "val_loss": val_loss_f, "val_hit": val_hit, "dcm": dcm})

        # monitor = cfg.monitor ("val_hit" ascending / "val_loss" descending) — plan-016 Path B
        monitor = getattr(cfg, "monitor", "val_hit")
        if monitor == "val_hit":
            improved = val_hit > best_val_hit
        elif monitor == "val_loss":
            improved = val_loss_f < best_val_loss
        else:
            raise ValueError(f"Unknown monitor: {monitor!r} (expected 'val_hit' or 'val_loss')")
        if improved:
            best_val_loss = val_loss_f
            best_val_hit = val_hit
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch + 1
            patience_left = cfg.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    # Restore best
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        hybrid_pred_val = model.hybrid_predict(
            seq_val_t, anchors_t, R_val_t, F0_val_t,
            temperature=cfg.temperature, use_reg_head=cfg.use_reg_head,
            r0_logit_prior=cfg.r0_logit_prior,
        )
        oof_pred = hybrid_pred_val.cpu().numpy()
        final_dcm = float(torch.linalg.norm(hybrid_pred_val - F0_val_t, dim=1).mean().item())

        # plan-016 carry: optional test prediction (best-state model)
        test_pred = None
        if X_test is not None:
            seq_test = make_seq_features_v2(X_test, end_idx=cfg.end_idx, feature_flags=feature_flags)
            F0_test_np = f0_function(X_test)
            seq_test_t = torch.from_numpy(seq_test).to(device)
            F0_test_t = torch.from_numpy(F0_test_np.astype(np.float32)).to(device)
            R_test_t = None
            if cfg.codebook != "absolute":
                R_test = pp.build_frenet_basis_3d(X_test, end_idx=cfg.end_idx)
                R_test_t = torch.from_numpy(R_test.astype(np.float32)).to(device)
            hybrid_pred_test = model.hybrid_predict(
                seq_test_t, anchors_t, R_test_t, F0_test_t,
                temperature=cfg.temperature, use_reg_head=cfg.use_reg_head,
                r0_logit_prior=cfg.r0_logit_prior,
            )
            test_pred = hybrid_pred_test.cpu().numpy()

    result = {
        "fold_id": fold_id,
        "n_train": int(n_train),
        "n_val": int(X_val.shape[0]),
        "best_val_loss": best_val_loss,
        "best_val_hit": best_val_hit,
        "best_epoch": best_epoch,
        "initial_val_hit": initial_val_hit,
        "dcm": final_dcm,
        "oof_pred": oof_pred,
        "epoch_log": epoch_log,
        "anchors_local": anchors_local.tolist(),
        "feature_dim": feature_dim,
    }
    if test_pred is not None:
        result["test_pred"] = test_pred
    return result


def run_kfold_oof_v2(
    ids: list[str],
    X: np.ndarray,
    Y: np.ndarray,
    config: pp.TrainConfig,
    feature_flags: dict[str, bool],
    f0_function: pp.Plan014F0Function | None = None,
    folds: list[int] | None = None,
    progress_cb=None,
) -> dict[str, Any]:
    """plan-015 K-fold OOF runner."""
    if f0_function is None:
        f0_function = pp.Plan014F0Function()

    fold_of = np.array([pp.stable_hash_fold(sid) for sid in ids])
    if folds is None:
        folds = list(range(pp.N_FOLDS))

    oof_pred = np.full_like(Y, fill_value=np.nan, dtype=np.float32)
    fold_results = []
    for f in folds:
        train_mask = (fold_of != f)
        val_mask = (fold_of == f)
        res = train_one_fold_v2(
            config, f,
            X[train_mask], Y[train_mask],
            X[val_mask], Y[val_mask],
            f0_function=f0_function,
            feature_flags=feature_flags,
        )
        oof_pred[val_mask] = res["oof_pred"]
        fold_results.append({k: v for k, v in res.items() if k != "oof_pred"})
        if progress_cb is not None:
            progress_cb(f, res)

    completed_mask = ~np.isnan(oof_pred).any(axis=1)
    if completed_mask.sum() > 0:
        overall_oof_hit = float(np.mean(
            np.linalg.norm(oof_pred[completed_mask] - Y[completed_mask], axis=-1) <= 0.01
        ))
    else:
        overall_oof_hit = float("nan")

    return {
        "overall_oof_hit_1cm": overall_oof_hit,
        "fold_results": fold_results,
        "oof_pred": oof_pred,
        "fold_of": fold_of.tolist(),
    }
