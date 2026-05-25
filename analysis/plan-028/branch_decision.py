"""plan-028 c5 — G2.B branch decision function (§4.5 박제).

decide_branch(B1, B2, B3, B4, W1) → Literal["α", "β", "γ", "δ"]

본 함수는 G2.A 의 (b)+(d) cell 5개 hit_1cm 만 input — T1/T2/S1/R1 (가설 (a)/(c)/(e))
verdict 는 §4.6 verdict 함수로 별도 산출, G2.B branch decision 에 영향 X (의도적 분리).
"""
from __future__ import annotations

from typing import Literal


P022 = 0.6531  # plan-022 winner hit_1cm (carry)


def decide_branch(
    B1: float, B2: float, B3: float, B4: float, W1: float,
) -> Literal["α", "β", "γ", "δ"]:
    """G2.A 의 (b)+(d) 5 cell (B1, B2, B3, B4, W1) hit_1cm 입력 → 1 branch activate.

    T1/T2/S1/R1 (가설 (a)/(c)/(e)) 결과는 본 함수 input 미포함 — §4.6 verdict 별도 산출.

    우선순위: α > β > γ > δ. 복수 조건 만족 시 우선순위 높은 것만 activate.
    """
    # α: input dim sweet spot (B2 ≥ p022 또는 B2 가 p022 직전 + B4 위 lift)
    if B2 >= P022 or (B2 > P022 - 0.005 and B2 > B4 + 0.005):
        return "α"

    # β: per-anchor 22D 단독 회복 (B1 > B3+0.005 AND B1 < p022)
    if B1 > B3 + 0.005 and B1 < P022:
        return "β"

    # γ: sample-weight 가 진짜 원인
    if W1 > B4 + 0.005:
        return "γ"

    # δ: default — α/β/γ 모두 미충족 → fallback.
    # §4.4 표 활성 조건 "모든 cell ≤ B4 + 0.003" 은 implicit (= α/β/γ 모두 false
    # 이면 자동으로 lift cell 부재 의미, mode collapse 잔존). explicit check
    # 별도 안 함. (a)/(c)/(e) 가설 cell (T1/T2/S1/R1) 결과는 branch decision 에
    # 미반영 — §4.6 verdict 함수에서만 산출, G3 best_cell selection 에서 통합 argmax.
    return "δ"
