import pytest

from rl4co.models import AttentionModel, AttentionModelPolicy
from rl4co.models import PointerNetwork, PointerNetworkPolicy
from rl4co.utils.test_utils import generate_env_data


@pytest.mark.parametrize("size", [10, 50])
def test_am(size):
    env, x = generate_env_data("tsp", size)
    td = env.reset(x)
    model = AttentionModel(env)
    out = model(td, decode_type="sampling")
    assert out["reward"].shape == (2,)


@pytest.mark.parametrize("size", [10, 50])
def test_ptrnet(size):
    env, x = generate_env_data("tsp", size)
    td = env.reset()
    model = PointerNetwork(env)
    out = model(td, decode_type="sampling")
    assert out["reward"].shape == (2,)
