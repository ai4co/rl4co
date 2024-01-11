import pytest
import torch

from tensordict import TensorDict

from rl4co.utils.ops import batchify, unbatchify


@pytest.mark.parametrize(
    "a",
    [
        torch.randn(10, 20, 2),
        TensorDict(
            {"a": torch.randn(10, 20, 2), "b": torch.randn(10, 20, 2)}, batch_size=10
        ),
    ],
)
@pytest.mark.parametrize("shape", [(2,), (2, 2), (2, 2, 2)])
def test_batchify(a, shape):
    # batchify: [b, ...] -> [b * prod(shape), ...]
    # unbatchify: [b * prod(shape), ...] -> [b, shape[0], shape[1], ...]
    a_batch = batchify(a, shape)
    a_unbatch = unbatchify(a_batch, shape)
    if isinstance(a, TensorDict):
        a, a_unbatch = a["a"], a_unbatch["a"]
    index = (slice(None),) + (0,) * len(shape)  # (slice(None), 0, 0, ..., 0)
    assert torch.allclose(a, a_unbatch[index])
