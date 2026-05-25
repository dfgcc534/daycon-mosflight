"""plan-031 c3 — GRUNetX3.

plan-030 GRUNetX2 그대로 carry. 유일 변경 = head_hidden 384 → 196 (over-param 완화).

plan-030 G3 FAIL root cause 2: head MLP input 382D 의 sample_summary broadcast 65% dominant
+ head_hidden 384 으로 over-parameterized → fold 별 spurious pattern overfitting.

slim 효과:
  head MLP params: 382×384 + 384 + 384×1 + 1 = 147,073
                 → 382×196 + 196 + 196×1 + 1 = 75,069  (= 51% 감소)
  total params 추정: 586,765 → ~515,000
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

_THIS = Path(__file__).resolve().parent
_PLAN030 = _THIS.parent / "plan-030"
if str(_PLAN030.parent.parent) not in sys.path:
    sys.path.insert(0, str(_PLAN030.parent.parent))


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


_p030_model = _load(_PLAN030 / "model.py", "p031_carry_model")
GRUNetX2 = _p030_model.GRUNetX2


class GRUNetX3(GRUNetX2):
    """plan-031 = plan-030 GRUNetX2 carry + head_hidden 196 (over-param 완화)."""

    def __init__(self, head_hidden: int = 196, **kwargs):
        super().__init__(head_hidden=head_hidden, **kwargs)


# ── smoke ────────────────────────────────────────────────────────────


def _smoke() -> None:
    torch.manual_seed(20260524)
    B, T, K = 4, 7, 14

    anchors = torch.randn(K, 3) * 0.01
    model = GRUNetX3(anchors=anchors)
    model.train()

    # head_in_dim 동일 (382), head_hidden 만 변경 (196)
    assert model.head_in_dim == 382
    assert isinstance(model.head_mlp[0], torch.nn.Linear)
    assert model.head_mlp[0].out_features == 196, f"head_hidden={model.head_mlp[0].out_features}"
    assert model.head_mlp[-1].in_features == 196
    assert model.head_mlp[-1].out_features == 1

    # forward signature 정합 (plan-030 carry)
    seq_97 = torch.randn(B, T, 97)
    res_kv = torch.randn(B, T, 5)
    q64 = torch.randn(B, K, 64)
    hs = torch.randn(B, 51)
    slim7 = torch.randn(B, K, 7)
    F0 = torch.randn(B, 3)
    R = torch.eye(3).unsqueeze(0).expand(B, -1, -1).contiguous()

    world_pred, probs = model(seq_97, res_kv, q64, hs, slim7, F0, R)
    assert world_pred.shape == (B, 3)
    assert probs.shape == (B, K)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(B), atol=1e-5)

    # param 합산
    n_params = sum(p.numel() for p in model.parameters())
    # plan-030 GRUNetX2 default = 586,765
    assert n_params < 586765, f"slim 효과 없음: {n_params}"

    print(f"[smoke] GRUNetX3 OK — head_hidden={model.head_mlp[0].out_features}, "
          f"head_in_dim={model.head_in_dim}, n_params={n_params}")


if __name__ == "__main__":
    _smoke()
