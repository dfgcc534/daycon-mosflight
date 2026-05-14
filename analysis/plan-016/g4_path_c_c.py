"""plan-016 c6 (STAGE 4, G4, Path C-C) — Feature C 단독 (multi-scale stride, 18D)."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _path_c_runner import run_path_c  # noqa: E402


def main():
    run_path_c(
        feature="C",
        feature_flag={"A": False, "B": False, "C": True, "D": False},
        expected_dim=18,
        exp_id="H053_g4_path_c_c",
        out_json=Path("analysis/plan-016/g4_path_c_c.json"),
        run_dir=Path("runs/baseline/plan016_g4_path_c_c"),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
