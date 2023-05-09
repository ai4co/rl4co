import pytest

from rl4co.models import AttentionModel, PointerNetwork, POMO, SymNCO
from rl4co.models import SymNCOPolicy
from rl4co.utils.test_utils import generate_env_data


@pytest.mark.parametrize("size", [10])
@pytest.mark.parametrize("env_name", ["tsp", "cvrp", "sdvrp"]) # TODO: dpp
def test_am(size, env_name, batch_size=2):
    env, x = generate_env_data(env_name, size, batch_size)
    td = env.reset(x)
    model = AttentionModel(env)
    out = model(td, decode_type="sampling")
    assert out["reward"].shape == (batch_size,)


@pytest.mark.parametrize("size", [10])
def test_ptrnet(size, batch_size=2):
    env, x = generate_env_data("tsp", size, batch_size)
    td = env.reset(x)
    model = PointerNetwork(env)
    out = model(td, decode_type="sampling")
    assert out["reward"].shape == (batch_size,)


@pytest.mark.parametrize("size", [10])
def test_pomo(size,  batch_size = 2):
    env, x = generate_env_data("tsp", size, batch_size)
    td = env.reset(x)
    model = POMO(env)
    model.policy.num_pomo = num_pomo = 10
    out = model(td, decode_type="sampling")
    assert out["reward"].shape == (batch_size * num_pomo,)


@pytest.mark.parametrize("size", [10])
def test_symnco(size, batch_size = 2, num_augment = 8, num_starts = 10):
    env, x = generate_env_data("tsp", size, batch_size)
    td = env.reset(x)
    policy = SymNCOPolicy(env, num_starts=num_starts)
    model = SymNCO(env, policy, num_augment=num_augment)
    out = model(td, decode_type="sampling")
    assert out["reward"].shape == (batch_size * num_augment * num_starts,)