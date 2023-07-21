import pytest

from rl4co.models import AutoregressivePolicy, PointerNetworkPolicy
from rl4co.utils.test_utils import generate_env_data


# Main autorergressive policy: rollout over multiple envs since it is the base
@pytest.mark.parametrize(
    "env_name", ["tsp", "cvrp", "sdvrp", "mtsp", "op", "pctsp", "spctsp", "dpp", "mdpp"]
)
def test_base_policy(env_name, size=20, batch_size=2):
    env, x = generate_env_data(env_name, size, batch_size)
    td = env.reset(x)
    policy = AutoregressivePolicy(env.name)
    out = policy(td, env, decode_type="greedy")
    assert out["reward"].shape == (batch_size,)


@pytest.mark.parametrize("env_name", ["tsp", "cvrp", "pctsp", "spctsp"])
def test_base_policy_multistart(env_name, size=20, batch_size=2):
    env, x = generate_env_data(env_name, size, batch_size)
    td = env.reset(x)
    policy = AutoregressivePolicy(env.name)
    out = policy(td, env, decode_type="greedy_multistart", num_starts=size)
    assert out["reward"].shape == (
        batch_size * size,
    )  # to evaluate, we could just unbatchify


def test_pointer_network(size=20, batch_size=2):
    env, x = generate_env_data("tsp", size, batch_size)
    td = env.reset(x)
    policy = PointerNetworkPolicy(env.name)
    out = policy(td, env, decode_type="greedy")
    assert out["reward"].shape == (batch_size,)
