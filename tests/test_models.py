import pytest

from rl4co.models import (
    POMO,
    AttentionModel,
    HeterogeneousAttentionModel,
    MDAMPolicy,
    PointerNetwork,
    SymNCO,
    SymNCOPolicy,
)
from rl4co.utils.test_utils import generate_env_data


@pytest.mark.parametrize("size", [20])
@pytest.mark.parametrize(
    "env_name", ["tsp", "cvrp", "sdvrp", "mtsp", "op", "pctsp", "spctsp", "dpp", "mdpp"]
)  # todo: sdvrp
def test_am(size, env_name, batch_size=2):
    env, x = generate_env_data(env_name, size, batch_size)
    td = env.reset(x)
    model = AttentionModel(env)
    out = model(td, decode_type="sampling")
    assert out["reward"].shape == (batch_size,)


@pytest.mark.parametrize("size", [20])
def test_ptrnet(size, batch_size=2):
    env, x = generate_env_data("tsp", size, batch_size)
    td = env.reset(x)
    model = PointerNetwork(env)
    out = model(td, decode_type="sampling")
    assert out["reward"].shape == (batch_size,)


@pytest.mark.parametrize("size", [20])
def test_pomo(size, batch_size=2):
    env, x = generate_env_data("tsp", size, batch_size)
    td = env.reset(x)
    model = POMO(env, num_starts=size)
    out = model(td, decode_type="sampling")
    assert out["reward"].shape == (batch_size * size,)


@pytest.mark.parametrize("size", [20])
def test_symnco(size, batch_size=2, num_augment=8, num_starts=20):
    env, x = generate_env_data("tsp", size, batch_size)
    td = env.reset(x)
    policy = SymNCOPolicy(env, num_starts=num_starts)
    model = SymNCO(env, policy, num_augment=num_augment)
    out = model(td, decode_type="sampling")
    assert out["reward"].shape == (batch_size * num_augment * num_starts,)


@pytest.mark.parametrize("size", [20])
def test_ham(size, batch_size=2):
    env, x = generate_env_data("pdp", size, batch_size)
    td = env.reset(x)
    model = HeterogeneousAttentionModel(env)
    out = model(td, decode_type="sampling")
    assert out["reward"].shape == (batch_size,)


@pytest.mark.parametrize("size", [20])
def test_mdam(size, batch_size=2, num_paths=5):
    env, x = generate_env_data("tsp", size, batch_size)
    td = env.reset(x)
    model = MDAMPolicy(env, num_paths=num_paths)
    out = model(td, decode_type="sampling")
    print(out["reward"].shape)
    assert out["reward"].shape == (
        num_paths,
        batch_size,
    )
