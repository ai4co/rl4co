import pytest

from ncobench.models import AttentionModel, AttentionModelPolicy
from ncobench.models import PointerNetwork, PointerNetworkPolicy
from ncobench.utils.test_utils import generate_env_data


@pytest.mark.parametrize("size", [10, 50])
def test_am(size):
    env, x = generate_env_data("tsp", size)
    td = env.reset(init_obs=x)
    model = AttentionModel(env)
    out = model(td, decode_type="sampling")
    assert out["reward"].shape == (2,)


@pytest.mark.parametrize("size", [10, 50])
def test_ptrnet(size):
    env, x = generate_env_data("tsp", size)
    td = env.reset(init_obs=x)
    model = PointerNetwork(env)
    out = model(td, decode_type="sampling")
    assert out["reward"].shape == (2,)
