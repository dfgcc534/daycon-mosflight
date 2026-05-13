"""plan-012 v2 — Codebook Bake-off Classification + Regression Hybrid (3D).

§4 spec 의 7 컴포넌트 self-contained 구현. plan-004 selector.CandidateAttentionGRUSelector encoder
reuse + 분류 head (K logit) + regression head (sample-wise, K mode × 3D offset, ±reg_head_scale_m bound).

External invariants vs plan §4:
- base.CandidateAttentionGRUSelector attribute 실제 명 = (gru, query, ctx_norm, event_norm, head)
  plan §4 의 invariant 박제 (gru, cand_proj, cand_attn, score_head) 와 부분 mismatch.
  실제 base 모듈 명 그대로 reuse. (plan spec 의 attribute 명 invariant 는 c2.1 §0.5 sync 시 spot-fix.)
"""

from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans

from src.pb_0_6822 import selector as base


EPS = 1e-8


# ──────────────────────────────────────────────────────────────────────────────
# 컴포넌트 1: 3 codebook generator (+ clip_norm helper)
# ──────────────────────────────────────────────────────────────────────────────


def clip_norm(vecs: np.ndarray, max_norm: float) -> np.ndarray:
    """Per-row L2-norm clip. (M, 3) → (M, 3). 각 행의 ‖·‖₂ > max_norm 시 max_norm 으로 축소."""
    norms = np.linalg.norm(vecs, axis=-1, keepdims=True)
    scale = np.minimum(1.0, max_norm / np.maximum(norms, 1e-12))
    return (vecs * scale).astype(vecs.dtype)


def compute_anchors_absolute(radius_m: float = 0.005) -> np.ndarray:
    """E0a Absolute-7Way: world frame ±x, ±y, ±z + center. Returns (7, 3) float64."""
    r = float(radius_m)
    anchors = np.array(
        [
            [0.0, 0.0, 0.0],   # center
            [+r,  0.0, 0.0],   # +x
            [-r,  0.0, 0.0],   # -x
            [0.0, +r,  0.0],   # +y
            [0.0, -r,  0.0],   # -y
            [0.0, 0.0, +r],    # +z
            [0.0, 0.0, -r],    # -z
        ],
        dtype=np.float64,
    )
    return anchors


def compute_anchors_frenet_orthogonal(radius_m: float = 0.005) -> np.ndarray:
    """E0b Frenet-Orthogonal-7Way: ±t̂, ±n̂, ±b̂ + center, Frenet local coord 값.

    좌표 *형식* = E0a 와 동일 (7 × 3 직교 set, basis 가 trajectory-aligned 인 점만 caller 가 회전).
    """
    return compute_anchors_absolute(radius_m=radius_m)  # 값 동일, 의미만 다름


def compute_anchors_kmeans(
    train_residuals_world: np.ndarray,
    R_world_from_frenet: np.ndarray,
    fold_id: np.ndarray,
    K: int = 7,
    radius_clip_m: float = 0.020,
    n_init: int = 10,
    random_state: int = 20260606,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """E0c K-Means-7Way: train residuals 의 Frenet 변환 위 K-Means K cluster + center origin.

    fold-aware fit: 각 fold k 마다 fold_id != k partition 으로 fit. val partition 은 assign-only.
    Returns:
        centers_per_fold       (5, K, 3) — fold-별 cluster centers in Frenet frame
        cluster_sizes_per_fold (5, K) int — cluster 별 sample 수
        fit_meta dict          — inertia, silhouette 등
    """
    assert train_residuals_world.shape[1] == 3, "residuals must be (N, 3)"
    assert R_world_from_frenet.shape == (train_residuals_world.shape[0], 3, 3)
    assert fold_id.shape == (train_residuals_world.shape[0],)
    assert K >= 2, "K >= 2 required"
    n_folds = 5

    centers_per_fold = np.zeros((n_folds, K, 3), dtype=np.float64)
    cluster_sizes_per_fold = np.zeros((n_folds, K), dtype=np.int64)
    inertia = []
    silhouette = []

    for k in range(n_folds):
        train_mask = fold_id != k
        n_tr = int(train_mask.sum())
        if n_tr < (K - 1) * 10:
            raise ValueError(f"compute_anchors_kmeans: insufficient train samples in fold {k} (n_tr={n_tr})")

        # world residual → Frenet frame: r_f = R_wfn.T @ r_w
        residuals_frenet_train = (
            R_world_from_frenet[train_mask].transpose(0, 2, 1)
            @ train_residuals_world[train_mask, :, None]
        ).squeeze(-1).astype(np.float64)

        km = KMeans(n_clusters=K - 1, n_init=n_init, random_state=random_state).fit(residuals_frenet_train)
        labels = km.labels_

        centers_per_fold[k, 0] = (0.0, 0.0, 0.0)
        centers_per_fold[k, 1:K] = km.cluster_centers_
        centers_per_fold[k, 1:K] = clip_norm(centers_per_fold[k, 1:K], max_norm=radius_clip_m)

        cluster_sizes_per_fold[k, 0] = 0  # explicit center anchor — no train sample assigned
        for j in range(K - 1):
            cluster_sizes_per_fold[k, j + 1] = int(np.sum(labels == j))

        inertia.append(float(km.inertia_))
        # silhouette is expensive — skip for large N; compute only on subsample if needed
        silhouette.append(float("nan"))

    fit_meta = {
        "inertia_per_fold": inertia,
        "silhouette_per_fold": silhouette,
        "n_folds": n_folds,
        "K": K,
        "n_init": n_init,
        "random_state": random_state,
    }
    return centers_per_fold, cluster_sizes_per_fold, fit_meta


# ──────────────────────────────────────────────────────────────────────────────
# 컴포넌트 2: 3D Frenet basis @ F0 prediction point
# ──────────────────────────────────────────────────────────────────────────────


def build_frenet_basis_3d(trajectory_x: np.ndarray, end_idx: int) -> np.ndarray:
    """Compute 3D Frenet basis (t̂, n̂, b̂) per sample.

    trajectory_x: (N, T, 3) world coords. end_idx: int, the F0 prediction point.
    Returns: R_world_from_frenet (N, 3, 3), columns = (t̂, n̂, b̂).

    Degenerate (‖v‖<EPS or ‖n_unnorm‖<EPS) → identity basis fallback (logged via return n_fallback count
    is omitted here; caller can post-detect via determinant or warn).
    """
    assert trajectory_x.ndim == 3 and trajectory_x.shape[-1] == 3
    assert end_idx >= 2, "end_idx >= 2 required for acceleration"
    N = trajectory_x.shape[0]

    v = trajectory_x[:, end_idx] - trajectory_x[:, end_idx - 1]                       # (N, 3)
    a_prev = trajectory_x[:, end_idx - 1] - trajectory_x[:, end_idx - 2]               # (N, 3)
    a = v - a_prev                                                                      # (N, 3) acceleration

    v_norm = np.linalg.norm(v, axis=1, keepdims=True)                                   # (N, 1)
    deg_v = v_norm[:, 0] < EPS
    safe_v_norm = np.where(deg_v[:, None], np.ones_like(v_norm), v_norm)
    t_hat = v / safe_v_norm                                                              # (N, 3)

    a_par = (np.sum(a * t_hat, axis=1, keepdims=True)) * t_hat                          # (N, 3)
    n_unnorm = a - a_par                                                                 # (N, 3)
    n_norm = np.linalg.norm(n_unnorm, axis=1, keepdims=True)
    deg_n = n_norm[:, 0] < EPS
    safe_n_norm = np.where(deg_n[:, None], np.ones_like(n_norm), n_norm)
    n_hat = n_unnorm / safe_n_norm

    b_hat = np.cross(t_hat, n_hat, axis=1)

    R = np.stack([t_hat, n_hat, b_hat], axis=-1).astype(np.float64)                     # (N, 3, 3) cols = t,n,b

    # degenerate fallback: identity
    fallback = (deg_v | deg_n)
    if fallback.any():
        R[fallback] = np.eye(3, dtype=np.float64)
    return R


# ──────────────────────────────────────────────────────────────────────────────
# 컴포넌트 3: anchor → world frame deltas
# ──────────────────────────────────────────────────────────────────────────────


def anchors_to_world(
    anchors_local: np.ndarray,
    R_world_from_frenet: np.ndarray | None,
    N: int,
) -> np.ndarray:
    """Returns (N, K, 3) — anchor *deltas* in world frame (F0_pred 미포함).

    E0a (Absolute): R_world_from_frenet=None → broadcast.
    E0b/E0c (Frenet-based): apply R per-sample.
    """
    assert anchors_local.ndim == 2 and anchors_local.shape[1] == 3
    K = anchors_local.shape[0]

    if R_world_from_frenet is None:
        out = np.broadcast_to(anchors_local[None, :, :], (N, K, 3)).copy()              # (N, K, 3)
        return out

    assert R_world_from_frenet.shape == (N, 3, 3)
    # world_vec = R @ frenet_vec — for each sample, each anchor
    # out[i, k, :] = R[i] @ anchors_local[k]
    out = np.einsum("nij,kj->nki", R_world_from_frenet, anchors_local)                  # (N, K, 3)
    return out


# ──────────────────────────────────────────────────────────────────────────────
# 컴포넌트 4: candidate feature builder (cand_dim = 11)
# ──────────────────────────────────────────────────────────────────────────────


def make_codebook_candidate_features(
    cand_pos_world: np.ndarray | None,
    anchors_local: np.ndarray,
    codebook_id: str,
    R_world_from_frenet: np.ndarray | None,
    F0_pred_world: np.ndarray,
) -> np.ndarray:
    """Build candidate feature tensor for hybrid scorer head. Returns (N, K, 11) float32.

    11-dim features = sample-invariant codebook-level constants only.
    sample-wise variation 은 HybridScorerHead._extract_seq_hidden 으로 주입.
    cand_pos_world / R_world_from_frenet 인자는 호출 통일 + 향후 확장용 — 본 v2 spec 에서 미사용.
    """
    del cand_pos_world, R_world_from_frenet  # 호출 통일용 — 향후 plan-013 확장 자리
    assert anchors_local.ndim == 2 and anchors_local.shape[1] == 3
    assert codebook_id in {"absolute", "frenet_orthogonal", "kmeans"}
    N = F0_pred_world.shape[0]
    K = anchors_local.shape[0]

    radius_local = np.linalg.norm(anchors_local, axis=1)                                # (K,)
    is_origin = (radius_local < 1e-6).astype(np.float32)                                # (K,)
    anchor_unit = anchors_local / 0.005                                                  # (K, 3) unit-scaled (assumes 0.005m radius default)

    cb_abs = 1.0 if codebook_id == "absolute" else 0.0
    cb_fr = 1.0 if codebook_id == "frenet_orthogonal" else 0.0
    cb_km = 1.0 if codebook_id == "kmeans" else 0.0

    per_cand = np.concatenate(
        [
            anchors_local.astype(np.float32),                            # [0:3]
            radius_local.astype(np.float32)[:, None],                    # [3]
            is_origin[:, None],                                          # [4]
            anchor_unit.astype(np.float32),                              # [5:8]
            np.full((K, 1), cb_abs, dtype=np.float32),                   # [8]
            np.full((K, 1), cb_fr, dtype=np.float32),                    # [9]
            np.full((K, 1), cb_km, dtype=np.float32),                    # [10]
        ],
        axis=1,
    )                                                                     # (K, 11)
    # broadcast to (N, K, 11)
    return np.broadcast_to(per_cand[None, :, :], (N, K, 11)).astype(np.float32).copy()


# ──────────────────────────────────────────────────────────────────────────────
# 컴포넌트 5a: HybridScorerHead (Attn-GRU scorer)
# ──────────────────────────────────────────────────────────────────────────────


class HybridScorerHead(nn.Module):
    """plan-004 CandidateAttentionGRUSelector + classifier (K logit) + regression head (3D offset)."""

    def __init__(
        self,
        K: int = 7,
        hidden: int = 64,
        cand_dim: int = 11,
        reg_head_scale_m: float = 0.005,
        encoder_pretrained_path: str | None = None,
        seq_dim: int = 9,
    ):
        super().__init__()
        self.K = K
        self.hidden = hidden
        self.cand_dim = cand_dim
        self.reg_head_scale_m = reg_head_scale_m
        self.scorer = base.CandidateAttentionGRUSelector(
            seq_dim=seq_dim, cand_dim=cand_dim, hidden=hidden, cand_count=K,
        )
        self.reg_head = nn.Sequential(
            nn.Linear(hidden + cand_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 3),
        )
        if encoder_pretrained_path is not None:
            self._load_encoder_weights(encoder_pretrained_path)

    def _load_encoder_weights(self, path: str) -> None:
        """Partial-load plan-004 GRU weights. cand_count-dependent layers (K=27→7) skip."""
        ckpt = torch.load(path, map_location="cpu")
        state = ckpt.get("model_state_dict", ckpt)
        # filter: keep only keys prefixed with attr names that exist on self.scorer with matching shape
        own = self.scorer.state_dict()
        filtered = {}
        for k, v in state.items():
            if k in own and own[k].shape == v.shape:
                filtered[k] = v
        missing, unexpected = self.scorer.load_state_dict(filtered, strict=False)
        # log via attribute for downstream introspection
        self._load_result = {"loaded": list(filtered.keys()), "missing": list(missing), "unexpected": list(unexpected)}

    def _extract_seq_hidden(self, seq: torch.Tensor) -> torch.Tensor:
        """seq (B, T, seq_dim) → (B, hidden) sample-wise representation.

        attn_gru scorer (default): GRU last-layer last-step hidden via ctx_norm.
        last_step_mlp scorer (E7.B): LastStepMLPScorer.extract_seq_hidden (seq_mlp(seq[:,-1,:])).
        """
        if hasattr(self.scorer, "extract_seq_hidden"):
            # LastStepMLPScorer or any alternate scorer providing its own extractor
            return self.scorer.extract_seq_hidden(seq)
        _, h = self.scorer.gru(seq)
        final_ctx = self.scorer.ctx_norm(h[-1])
        return final_ctx

    def forward(self, seq: torch.Tensor, cand_feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """seq (B, T, seq_dim), cand_feat (B, K, cand_dim) → (logits (B, K), reg_offset (B, K, 3))."""
        logits = self.scorer(seq, cand_feat)                                              # (B, K)
        seq_hidden = self._extract_seq_hidden(seq)                                        # (B, hidden)
        seq_hidden_b = seq_hidden[:, None, :].expand(-1, self.K, -1)                       # (B, K, hidden)
        reg_in = torch.cat([seq_hidden_b, cand_feat], dim=-1)                              # (B, K, hidden+cand_dim)
        reg_raw = self.reg_head(reg_in)                                                    # (B, K, 3)
        reg_offset = torch.tanh(reg_raw) * self.reg_head_scale_m                           # bounded to ±reg_head_scale_m
        return logits, reg_offset

    def freeze_encoder(self) -> None:
        """E7 sub-exp B (frozen GRU). encoder + cand attention 만 freeze; head + reg_head trainable."""
        for p in self.scorer.gru.parameters():
            p.requires_grad = False
        for p in self.scorer.query.parameters():
            p.requires_grad = False
        for p in self.scorer.ctx_norm.parameters():
            p.requires_grad = False
        for p in self.scorer.event_norm.parameters():
            p.requires_grad = False
        # self.scorer.head 의 score head 는 trainable 유지 (classifier head)
        # self.reg_head 는 trainable


# ──────────────────────────────────────────────────────────────────────────────
# 컴포넌트 5b: LastStepMLPScorer (E7 sub-exp B controlled comparison)
# ──────────────────────────────────────────────────────────────────────────────


class LastStepMLPScorer(nn.Module):
    """E7 sub-exp B — GRU 우회, last-step features only. drop-in for base.CandidateAttentionGRUSelector."""

    def __init__(self, seq_dim: int = 9, cand_dim: int = 11, hidden: int = 64, cand_count: int = 7):
        super().__init__()
        self.cand_count = cand_count
        self.hidden = hidden
        self.seq_mlp = nn.Sequential(
            nn.Linear(seq_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
        )
        self.cand_proj = nn.Sequential(
            nn.Linear(cand_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        # alias attributes for compatibility with HybridScorerHead.{_extract_seq_hidden, freeze_encoder}
        self.gru = nn.Identity()                          # placeholder (E7 sub-exp B 는 freeze_encoder 미사용)
        self.ctx_norm = nn.Identity()
        self.event_norm = nn.Identity()
        self.query = nn.Identity()
        self.head = nn.Identity()

    def forward(self, seq: torch.Tensor, cand_feat: torch.Tensor) -> torch.Tensor:
        seq_last = seq[:, -1, :]                                                           # (B, seq_dim)
        h = self.seq_mlp(seq_last)                                                         # (B, hidden)
        cand_h = self.cand_proj(cand_feat)                                                 # (B, K, hidden)
        logits = (cand_h * h[:, None, :]).sum(dim=-1)                                      # (B, K)
        return logits

    def extract_seq_hidden(self, seq: torch.Tensor) -> torch.Tensor:
        return self.seq_mlp(seq[:, -1, :])


# ──────────────────────────────────────────────────────────────────────────────
# 컴포넌트 6: L7 hit-aware hinge + huber + classifier CE + hybrid combined loss
# ──────────────────────────────────────────────────────────────────────────────


def hit_aware_hinge(
    corrected_pos: torch.Tensor,
    target: torch.Tensor,
    R_HIT: float = 0.01,
    smooth: float = 0.005,
) -> torch.Tensor:
    """plan-011 §5.1 hit-aware smooth hinge, 3D. (B, 3) → (B,)."""
    excess = torch.norm(corrected_pos - target, dim=-1) - R_HIT
    linear_hinge = F.softplus(excess / smooth) * smooth
    return linear_hinge ** 2


def huber_loss_3d(pred: torch.Tensor, target: torch.Tensor, beta: float = 0.005) -> torch.Tensor:
    """3D huber. (B, 3) → (B,)."""
    return F.smooth_l1_loss(pred, target, beta=beta, reduction="none").sum(dim=-1)


def classifier_ce_loss(
    logits: torch.Tensor,
    F0_pred_world: torch.Tensor,
    anchors_world: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Hard-label CE on argmin-by-distance mode."""
    cand_pos = F0_pred_world[:, None, :] + anchors_world                                  # (B, K, 3)
    distances = torch.norm(cand_pos - target[:, None, :], dim=-1)                          # (B, K)
    hard_label = distances.argmin(dim=-1)                                                  # (B,)
    return F.cross_entropy(logits, hard_label)


def hybrid_combined_loss(
    logits: torch.Tensor,
    reg_offset: torch.Tensor,
    anchors_world: torch.Tensor,
    F0_pred_world: torch.Tensor,
    target: torch.Tensor,
    temperature: float = 0.03,
    R_HIT: float = 0.01,
    smooth: float = 0.005,
    lambda_ce: float = 1.0,
    lambda_hinge: float = 1.0,
    use_reg_head: bool = True,
    use_hinge: bool = True,
) -> torch.Tensor:
    """Combined loss = λ_ce·CE + λ_hinge·(0.5·huber + 0.5·hinge) on hybrid_pred_pos.

    use_reg_head=False → reg_offset 항 무시 (E5.A sub-exp).
    use_hinge=False    → hinge 제거 (E4.B sub-exp, huber only).
    """
    ce = classifier_ce_loss(logits, F0_pred_world, anchors_world, target)

    if temperature <= 1e-8:
        mode_k = logits.argmax(dim=-1)                                                     # (B,)
        sel_anchor = anchors_world.gather(1, mode_k[:, None, None].expand(-1, -1, 3)).squeeze(1)
        if use_reg_head:
            sel_reg = reg_offset.gather(1, mode_k[:, None, None].expand(-1, -1, 3)).squeeze(1)
        else:
            sel_reg = torch.zeros_like(sel_anchor)
        hybrid_pos = F0_pred_world + sel_anchor + sel_reg
    else:
        prob = F.softmax(logits / temperature, dim=-1)                                     # (B, K)
        anchor_blend = (prob[:, :, None] * anchors_world).sum(dim=1)                       # (B, 3)
        if use_reg_head:
            reg_blend = (prob[:, :, None] * reg_offset).sum(dim=1)
        else:
            reg_blend = torch.zeros_like(anchor_blend)
        hybrid_pos = F0_pred_world + anchor_blend + reg_blend

    huber = huber_loss_3d(hybrid_pos, target)
    if use_hinge:
        hinge = hit_aware_hinge(hybrid_pos, target, R_HIT=R_HIT, smooth=smooth)
        pos = 0.5 * huber + 0.5 * hinge
    else:
        pos = huber
    pos_loss = pos.mean()

    return lambda_ce * ce + lambda_hinge * pos_loss


# ──────────────────────────────────────────────────────────────────────────────
# 컴포넌트 7: hybrid inference
# ──────────────────────────────────────────────────────────────────────────────


def hybrid_predict(
    logits: torch.Tensor,
    reg_offset: torch.Tensor,
    anchors_world: torch.Tensor,
    F0_pred_world: torch.Tensor,
    temperature: float = 0.03,
    use_reg_head: bool = True,
    r0_logit_prior: float = 0.0,
) -> torch.Tensor:
    """Inference: (B, K)+(B, K, 3)+(B, K, 3)+(B, 3) → (B, 3) world position."""
    prior = torch.zeros(logits.shape[-1], device=logits.device, dtype=logits.dtype)
    prior[0] = r0_logit_prior
    logits_prior = logits + prior

    if temperature <= 1e-8:
        mode_k = logits_prior.argmax(dim=-1)
        sel_anchor = anchors_world.gather(1, mode_k[:, None, None].expand(-1, -1, 3)).squeeze(1)
        if use_reg_head:
            sel_reg = reg_offset.gather(1, mode_k[:, None, None].expand(-1, -1, 3)).squeeze(1)
        else:
            sel_reg = torch.zeros_like(sel_anchor)
        return F0_pred_world + sel_anchor + sel_reg

    prob = F.softmax(logits_prior / temperature, dim=-1)
    anchor_blend = (prob[:, :, None] * anchors_world).sum(dim=1)
    if use_reg_head:
        reg_blend = (prob[:, :, None] * reg_offset).sum(dim=1)
    else:
        reg_blend = torch.zeros_like(anchor_blend)
    return F0_pred_world + anchor_blend + reg_blend


# ──────────────────────────────────────────────────────────────────────────────
# F0 단일공식 — frenet_par120_perp_neg020 (plan-006 CANDIDATES[17])
#
# corrector_redesign_v2.LearnableSingleCandidate (init_coef = 1.98, 0.0, 1.20,
# -0.20, 0.0, 1.0) 와 numerically exact:
#   cand = p0 + d1·v_last + d2·v_prev + par·acc_par_vec + perp·acc_perp_vec
#          + jerk·jerk_vec
# horizon=2, time_scale=1 → v_scale = acc_scale = 1.
#
# (decision-note: spec-default — plan §1.4.1 의 박제 식 (par·v_par, perp·v_perp,
# binorm·v_binorm 만) 은 단순화되어 plan-006 CANDIDATES[17] 와 mismatch. 본 실제
# 식이 plan-006 의 source-of-truth. F0 raw hit 측정 시 본 식 사용. plan §1.4.1
# spot-fix 는 추후 §0.5 sync 시 처리.)
# ──────────────────────────────────────────────────────────────────────────────


def f0_predict_frenet_par120_perp_neg020(
    trajectory_x: np.ndarray,
    end_idx: int | None = None,
    d1_coef: float = 1.98,
    d2_coef: float = 0.0,
    par_coef: float = 1.20,
    perp_coef: float = -0.20,
    jerk_coef: float = 0.0,
    horizon: float = 2.0,
    time_scale: float = 1.0,
) -> np.ndarray:
    """F0 = frenet_par120_perp_neg020. Returns (N, 3) F0_pred_world.

    cand = p0 + d1·v_scale·v_last + d2·v_scale·v_prev
              + par·acc_scale·acc_par_vec + perp·acc_scale·acc_perp_vec
              + jerk·acc_scale·jerk_vec
    where v_scale = (horizon/2)·time_scale, acc_scale = (horizon/2)²·time_scale².
    """
    assert trajectory_x.ndim == 3 and trajectory_x.shape[-1] == 3
    T = trajectory_x.shape[1]
    if end_idx is None:
        end_idx = T - 1
    assert end_idx >= 3, "end_idx >= 3 required (need x[end-3] for prev_acc)"

    p0 = trajectory_x[:, end_idx]                                                            # (N, 3)
    v_last = p0 - trajectory_x[:, end_idx - 1]                                               # (N, 3)
    v_prev = trajectory_x[:, end_idx - 1] - trajectory_x[:, end_idx - 2]                     # (N, 3)
    prev_v = trajectory_x[:, end_idx - 2] - trajectory_x[:, end_idx - 3]                     # (N, 3)
    acc = v_last - v_prev
    prev_acc = v_prev - prev_v
    jerk_vec = acc - prev_acc

    v_norm = np.linalg.norm(v_last, axis=-1, keepdims=True)
    safe_v_norm = np.where(v_norm < EPS, np.ones_like(v_norm), v_norm)
    t_hat = v_last / safe_v_norm
    acc_par_scalar = np.sum(acc * t_hat, axis=-1, keepdims=True)
    acc_par_vec = acc_par_scalar * t_hat
    acc_perp_vec = acc - acc_par_vec

    half_h = horizon / 2.0
    v_scale = half_h * time_scale
    acc_scale = (half_h ** 2) * (time_scale ** 2)

    F0 = (
        p0
        + d1_coef * v_scale * v_last
        + d2_coef * v_scale * v_prev
        + par_coef * acc_scale * acc_par_vec
        + perp_coef * acc_scale * acc_perp_vec
        + jerk_coef * acc_scale * jerk_vec
    )
    return F0.astype(np.float64)
