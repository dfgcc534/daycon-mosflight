"""plan-016 c5 (STAGE 3, G3, Path C-B) — Feature B 단독 (binormal split, 10D).

base = G1 carry (G2 dropped per §6.3 negative_drop). monitor=val_hit.
변경 = feature_flags={"B": True} only.
threshold = +0.003 vs G1 (per §7.3 v1.5 fix, G2 drop case → base=G1).

Usage:
    python analysis/plan-016/g3_path_c_b.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _path_c_runner import run_path_c  # noqa: E402


def main():
    run_path_c(
        feature="B",
        feature_flag={"A": False, "B": True, "C": False, "D": False},
        expected_dim=10,
        exp_id="H052_g3_path_c_b",
        out_json=Path("analysis/plan-016/g3_path_c_b.json"),
        run_dir=Path("runs/baseline/plan016_g3_path_c_b"),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
