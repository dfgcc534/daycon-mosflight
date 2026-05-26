"""천장 검증 (정보이론) — read-only, GPU 0.

질문: 0.6862 위 oracle headroom(union 0.69 / anchor 0.79)이 irreducible 미래 불확실성인가,
learnable 인가? 세 각도:
 (1) k-NN Bayes hit: 유사 과거 이웃의 미래 local-mean 예측 → 어떤 모델도 못 넘는 천장 추정.
 (2) 조건부 future-spread: 이웃 미래 residual RMS > 1cm = irreducible 비율.
 (3) gt_anchor_label 학습가능성: 강한 GBM 이 oracle-best anchor 를 현 selector 보다 맞추나.
leakage 차단: 이웃·학습은 train fold 만 (fold_ids).
"""
from __future__ import annotations

import importlib.util as _u
import sys
from pathlib import Path

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

_THIS = Path(__file__).resolve().parent
_REPO = _THIS.parent.parent
sys.path.insert(0, str(_REPO))
from src.io import load_all_samples  # noqa: E402

R = 0.01


def _load(p, n):
    s = _u.spec_from_file_location(n, p); m = _u.module_from_spec(s); s.loader.exec_module(m); return m


_kf = _load(_THIS / "kalman_features.py", "kf")


def hit(p, y):
    return (np.linalg.norm(p - y, axis=-1) <= R)


def main():
    ids, X = load_all_samples("train")
    z8 = np.load(_THIS / "results_kr008.npz")
    y = z8["y"]; kal = z8["kalman_main"]; fid = z8["fold_ids"]; h8 = hit(z8["oof_pred"], y)
    n = len(y)
    dt = 0.04
    v = (X[:, -1] - X[:, -2]) / dt
    a = (X[:, -1] - 2 * X[:, -2] + X[:, -3]) / dt**2
    cc = _kf.cv_ca_disagreement(X)
    # 과거 descriptor (최근 동역학 — +80ms 미래 결정 요인). 회전불변 위해 raw 사용(k-NN 거리용).
    rel3 = (X[:, -3:] - X[:, -1:]).reshape(n, -1)   # 최근 3 step rel (9)
    past = np.concatenate([v, a, cc, rel3], axis=1).astype(np.float64)  # 18D
    r = y - kal                                      # 모델이 맞춰야 할 residual

    print(f"# 천장 검증 (n={n}, KR008 hit={h8.mean():.4f})\n")

    # (1)+(2) k-NN Bayes hit + 조건부 spread (cross-fold, leakage 0)
    print("## (1) k-NN Bayes hit-rate (유사 과거 이웃 local-mean 예측 = 천장 추정)")
    print("## (2) 조건부 future-spread (이웃 residual RMS > 1cm = irreducible)")
    for K in [30, 100, 300]:
        bayes_pred = np.zeros_like(y); local_rms = np.zeros(n)
        for f in range(5):
            tr = np.where(fid != f)[0]; va = np.where(fid == f)[0]
            sc = StandardScaler().fit(past[tr])
            nn = NearestNeighbors(n_neighbors=K).fit(sc.transform(past[tr]))
            _, idx = nn.kneighbors(sc.transform(past[va]))
            nbr_r = r[tr][idx]                       # (|va|, K, 3) 이웃 residual
            mean_r = nbr_r.mean(1)
            bayes_pred[va] = kal[va] + mean_r        # Bayes proxy 예측
            local_rms[va] = np.sqrt(((nbr_r - mean_r[:, None]) ** 2).sum(-1).mean(1))  # 이웃 미래 spread
        bh = hit(bayes_pred, y).mean()
        irreducible = (local_rms > R).mean()
        print(f"  K={K:3d}: k-NN Bayes hit={bh:.4f} (vs KR008 {h8.mean():.4f}, Δ {bh-h8.mean():+.4f}) | "
              f"local-spread>1cm 비율={irreducible:.4f}")
    # missed 샘플의 spread
    print(f"  → KR008 miss 샘플의 local-spread>1cm 비율 (K=100 기준 아래 별도)")

    # (3) gt_anchor_label 학습가능성
    print("\n## (3) gt_anchor_label 학습가능성 (강한 GBM vs 현 selector)")
    zx = np.load("analysis/plan-029/oof_X1.npz")
    if "gt_anchor_label" in zx.files and "oof_probs" in zx.files:
        gt = zx["gt_anchor_label"].astype(int); probs = zx["oof_probs"]
        sel_acc = (probs.argmax(1) == gt).mean()     # 현 selector top-1 anchor 정확도
        from sklearn.ensemble import GradientBoostingClassifier
        feats = np.concatenate([v, a, cc, np.linalg.norm(v, axis=1, keepdims=True),
                                np.linalg.norm(cc, axis=1, keepdims=True), rel3], axis=1)
        gbm_pred = np.zeros(n, int)
        for f in range(5):
            tr = fid != f; va = fid == f
            clf = GradientBoostingClassifier(max_depth=3, n_estimators=150, subsample=0.7)
            clf.fit(feats[tr], gt[tr]); gbm_pred[va] = clf.predict(feats[va])
        gbm_acc = (gbm_pred == gt).mean()
        chance = np.bincount(gt).max() / n
        print(f"  현 selector anchor 정확도={sel_acc:.4f} | 신규 GBM={gbm_acc:.4f} | "
              f"majority-class chance={chance:.4f}")
        print(f"  → GBM−selector Δ={gbm_acc-sel_acc:+.4f} (양 크면 selector gap learnable, ≈0 면 anchor 선택 정보 부재)")

    print("\n## 해석: k-NN Bayes ≈ KR008 면 천장 확정(irreducible). GBM≈selector 면 anchor 선택 비-learnable.")


if __name__ == "__main__":
    main()
