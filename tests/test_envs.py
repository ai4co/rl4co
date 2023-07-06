import pytest

from rl4co.envs import ATSPEnv, CVRPEnv, DPPEnv, MTSPEnv, PDPEnv, SDVRPEnv, TSPEnv
from rl4co.models.nn.utils import random_policy, rollout


@pytest.mark.parametrize("size, batch_size", [(20, 2)])
def test_tsp(size, batch_size):
    env = TSPEnv(num_loc=size)
    reward = rollout(env, env.reset(batch_size=[batch_size]), random_policy)
    assert reward.shape == (batch_size,)


@pytest.mark.parametrize("size, batch_size", [(20, 2)])
def test_atsp(size, batch_size):
    env = ATSPEnv(num_loc=size)
    reward = rollout(env, env.reset(batch_size=[batch_size]), random_policy)
    assert reward.shape == (batch_size,)


@pytest.mark.parametrize("size, batch_size", [(20, 2)])
def test_dpp(size, batch_size):
    env = DPPEnv()
    reward = rollout(env, env.reset(batch_size=[batch_size]), random_policy)
    assert reward.shape == (batch_size,)


@pytest.mark.parametrize("size, batch_size", [(20, 2)])
def test_cvrp(size, batch_size):
    env = CVRPEnv()
    reward = rollout(env, env.reset(batch_size=[batch_size]), random_policy)
    assert reward.shape == (batch_size,)


@pytest.mark.parametrize("size, batch_size", [(20, 2)])
def test_sdvrp(size, batch_size):
    env = SDVRPEnv()
    reward = rollout(env, env.reset(batch_size=[batch_size]), random_policy)
    assert reward.shape == (batch_size,)


@pytest.mark.parametrize("size, batch_size", [(20, 2)])
def test_pdp(size, batch_size):
    env = PDPEnv(num_loc=size)
    reward = rollout(env, env.reset(batch_size=[batch_size]), random_policy)
    assert reward.shape == (batch_size,)


@pytest.mark.parametrize("size, batch_size", [(20, 2)])
def test_mtsp(size, batch_size):
    env = MTSPEnv(num_loc=size)
    reward = rollout(env, env.reset(batch_size=[batch_size]), random_policy)
    assert reward.shape == (batch_size,)
