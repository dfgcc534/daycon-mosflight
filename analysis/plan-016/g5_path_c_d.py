"""plan-016 c7 (STAGE 5, G5, Path C-D) — Feature D 단독 (pairwise cross-step, 15D)."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _path_c_runner import run_path_c  # noqa: E402


def main():
    run_path_c(
        feature="D",
        feature_flag={"A": False, "B": False, "C": False, "D": True},
        expected_dim=15,
        exp_id="H054_g5_path_c_d",
        out_json=Path("analysis/plan-016/g5_path_c_d.json"),
        run_dir=Path("runs/baseline/plan016_g5_path_c_d"),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
