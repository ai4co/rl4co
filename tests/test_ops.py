import pytest
import torch
from rl4co.utils.ops import batchify, unbatchify
from tensordict import TensorDict


@pytest.mark.parametrize(
    "a",
    [
        torch.randn(10, 20, 2),
        TensorDict(
            {"a": torch.randn(10, 20, 2), "b": torch.randn(10, 20, 2)}, batch_size=10
        ),
    ],
)
def test_batchify(a):
    a_batch = batchify(a, 5)
    a_unbatch = unbatchify(a_batch, 5)
    if isinstance(a, TensorDict):
        a, a_unbatch = a["a"], a_unbatch["a"]
    assert torch.allclose(a, a_unbatch[:, 0])
