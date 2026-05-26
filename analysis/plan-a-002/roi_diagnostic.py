"""ROI 방향 진단 (A/B/C 회수율) — read-only, GPU 0.

+0.01 목표 위해 어느 방향이 현실 capturable 한지 commit 전 추정:
 A. boundary corrector — sklearn corrector(Ridge/GBM)로 KR008 residual 회수 가능량 (OOF, lower bound)
 B. cross-paradigm stacking — KR008 ⊕ 탈상관 partner: oracle vs best-blend vs feature-selector
 C. anchor 부활 — 14-anchor oracle(기지 0.7928) vs selector 한계 gap

모두 기존 OOF npz + X feature 재계산만. 결론 = 방향별 예상 capturable ΔOOF.
"""
from __future__ import annotations

import importlib.util as _u
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge

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
    ids = np.array(ids)
    z8 = np.load(_THIS / "results_kr008.npz")
    y = z8["y"]; pred8 = z8["oof_pred"]; kal = z8["kalman_main"]; fid = z8["fold_ids"]
    n = len(y)
    print(f"# ROI 진단 (n={n}, KR008 OOF hit={hit(pred8, y).mean():.4f})\n")

    # ---------- A. boundary corrector capturability ----------
    print("## A. Boundary corrector (sklearn, OOF lower-bound)")
    # corrector features (관측창만 — leakage-safe)
    dt = 0.04
    v = (X[:, -1] - X[:, -2]) / dt
    a = (X[:, -1] - 2 * X[:, -2] + X[:, -3]) / dt**2
    cv_ca = _kf.cv_ca_disagreement(X)
    _, innov, fv = _kf.kalman_with_internals(X)
    gru_corr = pred8 - kal                      # GRU 가 kalman 에 가한 보정
    feats = np.concatenate([
        v, a, cv_ca,                            # 9: 속도/가속/CA-CV 불일치
        np.linalg.norm(v, axis=1, keepdims=True),
        np.linalg.norm(a, axis=1, keepdims=True),
        np.linalg.norm(cv_ca, axis=1, keepdims=True),
        np.linalg.norm(innov[:, -3:], axis=2).mean(1, keepdims=True),  # 최근 maneuver
        gru_corr, np.linalg.norm(gru_corr, axis=1, keepdims=True),     # GRU 보정 크기/방향
        pred8 - kal,                            # = gru_corr (중복이나 GBM 무해)
    ], axis=1).astype(np.float64)
    tgt = (y - pred8)                           # corrector 가 맞춰야 할 Δ
    for name, mk in [("Ridge", lambda: Ridge(alpha=1.0)),
                     ("GBM", lambda: GradientBoostingRegressor(max_depth=3, n_estimators=200, subsample=0.7))]:
        corr = np.zeros_like(tgt)
        for f in range(5):
            tr = fid != f; va = fid == f
            for j in range(3):
                mdl = mk(); mdl.fit(feats[tr], tgt[tr, j]); corr[va, j] = mdl.predict(feats[va])
        new = pred8 + corr
        # 과보정 방지: 보정 후 hit 만 채택하는 건 oracle → 순수 corrector 결과 보고
        print(f"  {name:6s}: corrected OOF hit = {hit(new, y).mean():.4f}  (Δ vs KR008 {hit(new,y).mean()-hit(pred8,y).mean():+.4f})")
    # near-miss systematic 진단: 1-1.5cm 잔차의 feature 설명력 (GBM R² 대용 — corrected 개선폭이 곧 신호)
    d = np.linalg.norm(pred8 - y, axis=1)
    nm = (d > 0.01) & (d <= 0.015)
    print(f"  near-miss(1-1.5cm) {nm.sum()}마리 — 위 corrected Δ 가 양이면 체계적 신호 존재")

    # ---------- B. cross-paradigm stacking ----------
    print("\n## B. Cross-paradigm stacking (KR008 ⊕ partner)")
    partners = {
        "F014_corrector": ("runs/baseline/F014_ebip-base/oof_predictions.npz", "oof_pred", "ids"),
        "KR001": ("analysis/plan-a-001/results_kr001.npz", "oof_pred", None),
    }
    for pname, (pth, key, idkey) in partners.items():
        zp = np.load(_REPO / pth if not Path(pth).is_absolute() else pth, allow_pickle=False) \
            if (_REPO / pth).exists() else np.load(pth)
        pp = zp[key]
        if idkey is not None and idkey in zp.files:          # id 정렬
            order = {i: k for k, i in enumerate(zp[idkey])}
            idx = np.array([order[i] for i in ids]); pp = pp[idx]
        hp = hit(pp, y); h8 = hit(pred8, y)
        oracle = (h8 | hp).mean()
        # best fixed blend
        best_blend = max(hit(w * pred8 + (1 - w) * pp, y).mean() for w in np.linspace(0, 1, 11))
        # feature-selector proxy: GBM 분류 "KR008 이 더 가까운가" 5-fold
        closer8 = (np.linalg.norm(pred8 - y, axis=1) <= np.linalg.norm(pp - y, axis=1)).astype(int)
        sel = np.zeros(n, int)
        from sklearn.ensemble import GradientBoostingClassifier
        for f in range(5):
            tr = fid != f; va = fid == f
            clf = GradientBoostingClassifier(max_depth=3, n_estimators=150, subsample=0.7)
            clf.fit(feats[tr], closer8[tr]); sel[va] = clf.predict(feats[va])
        sel_pred = np.where(sel[:, None] == 1, pred8, pp)
        sel_hit = hit(sel_pred, y).mean()
        print(f"  {pname:15s}: partner {hp.mean():.4f} | oracle {oracle:.4f} | best-blend {best_blend:.4f} | "
              f"feat-selector {sel_hit:.4f} (Δ {sel_hit-h8.mean():+.4f})")

    # ---------- C. anchor 부활 ----------
    print("\n## C. Anchor/selector (기지 oracle vs selector 한계)")
    zx = np.load("analysis/plan-029/oof_X1.npz")
    sel_hit_x = hit(zx["oof_pred"], y).mean()
    print(f"  14-anchor oracle (기지, plan-024) = 0.7928  |  현 selector OOF hit = {sel_hit_x:.4f}  "
          f"|  미실현 gap = {0.7928 - sel_hit_x:.4f}")
    print(f"  → KR008(0.6862) 후보 추가해도 oracle ≈ 0.79+ (이미 anchor 가 79% 포함). 병목 = selector.")

    print("\n## 결론: 위 A corrected Δ / B feat-selector Δ / C gap 으로 방향 ROI 비교")


if __name__ == "__main__":
    main()
