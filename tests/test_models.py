import pytest

from rl4co.models import AttentionModel, AttentionModelPolicy
from rl4co.models import PointerNetwork, PointerNetworkPolicy
from rl4co.utils.test_utils import generate_env_data


@pytest.mark.parametrize("size", [10, 50])
def test_am(size):
    batch_size = 2
    env, x = generate_env_data("tsp", size, batch_size)    
    td = env.reset(x)
    model = AttentionModel(env)
    out = model(td, decode_type="sampling")
    assert out["reward"].shape == (batch_size,)


@pytest.mark.parametrize("size", [10, 50])
def test_ptrnet(size):
    batch_size = 2
    env, x = generate_env_data("tsp", size, batch_size)
    td = env.reset(x)
    model = PointerNetwork(env)
    out = model(td, decode_type="sampling")
    assert out["reward"].shape == (batch_size,)
