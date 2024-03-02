import pytest

from rl4co.models import AutoregressivePolicy, PointerNetworkPolicy
from rl4co.utils.test_utils import generate_env_data
from rl4co.utils.ops import select_start_nodes


# Main autorergressive policy: rollout over multiple envs since it is the base
@pytest.mark.parametrize(
    "env_name",
    ["tsp", "cvrp", "sdvrp", "mtsp", "op", "pctsp", "spctsp", "dpp", "mdpp", "smtwtp"],
)
def test_base_policy(env_name, size=20, batch_size=2):
    env, x = generate_env_data(env_name, size, batch_size)
    td = env.reset(x)
    policy = AutoregressivePolicy(env.name)
    out = policy(td, env, decode_type="greedy")
    assert out["reward"].shape == (batch_size,)


@pytest.mark.parametrize(
    "env_name", ["tsp", "cvrp", "pctsp", "spctsp", "sdvrp", "op", "pdp"]
)
def test_base_policy_multistart(env_name, size=20, batch_size=2):
    env, x = generate_env_data(env_name, size, batch_size)
    td = env.reset(x)
    policy = AutoregressivePolicy(env.name)
    num_starts = size // 2 if env.name in ["pdp"] else size
    out = policy(
        td,
        env,
        decode_type="multistart_greedy",
        num_starts=num_starts,
        select_start_nodes_fn=select_start_nodes,
    )
    assert out["reward"].shape == (
        batch_size * num_starts,
    )  # to evaluate, we could just unbatchify


@pytest.mark.parametrize(
    "env_name",
    ["tsp", "cvrp", "pctsp", "spctsp", "sdvrp", "op", "pdp"],
)
@pytest.mark.parametrize("select_best", [True, False])
def test_base_policy_beam_search(env_name, select_best, size=20, batch_size=2):
    env, x = generate_env_data(env_name, size, batch_size)
    td = env.reset(x)
    policy = AutoregressivePolicy(env.name)
    beam_width = size // 2 if env.name in ["pdp"] else size
    out = policy(
        td, env, decode_type="beam_search", beam_width=beam_width, select_best=select_best
    )

    if select_best:
        expected_shape = (batch_size,)
    else:
        expected_shape = (batch_size * beam_width,)
    assert out["reward"].shape == expected_shape


def test_pointer_network(size=20, batch_size=2):
    env, x = generate_env_data("tsp", size, batch_size)
    td = env.reset(x)
    policy = PointerNetworkPolicy(env.name)
    out = policy(td, env, decode_type="greedy")
    assert out["reward"].shape == (batch_size,)
