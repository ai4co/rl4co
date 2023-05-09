import pytest
import torch

from rl4co.envs import TSPEnv, ATSPEnv, DPPEnv, CVRPEnv, SDVRPEnv, PDPEnv


def policy(td):
    """Helper function to select a random action from available actions"""
    action = torch.multinomial(td["action_mask"].float(), 1).squeeze(-1)
    td.set("action", action)
    return td


def rollout(env, td, policy):
    """Helper function to rollout a policy"""
    actions = []
    while not td["done"].any():
        td = policy(td)
        actions.append(td["action"])
        td = env.step(td)["next"]
    return env.get_reward(td, torch.stack(actions, dim=1))


@pytest.mark.parametrize("size, batch_size", [(10, 2)])
def test_tsp(size, batch_size):
    env = TSPEnv(num_loc=size)
    reward = rollout(env, env.reset(batch_size=[batch_size]), policy)
    assert reward.shape == (batch_size,)


@pytest.mark.parametrize("size, batch_size", [(10, 2)])
def test_atsp(size, batch_size):
    env = ATSPEnv(num_loc=size)
    reward = rollout(env, env.reset(batch_size=[batch_size]), policy)
    assert reward.shape == (batch_size,)


@pytest.mark.parametrize("size, batch_size", [(10, 2)])
def test_dpp(size, batch_size):
    env = DPPEnv()
    reward = rollout(env, env.reset(batch_size=[batch_size]), policy)
    assert reward.shape == (batch_size,)


@pytest.mark.parametrize("size, batch_size", [(10, 2)])
def test_cvrp(size, batch_size):
    env = CVRPEnv()
    reward = rollout(env, env.reset(batch_size=[batch_size]), policy)
    assert reward.shape == (batch_size,)


@pytest.mark.parametrize("size, batch_size", [(10, 2)])
def test_sdvrp(size, batch_size):
    env = SDVRPEnv()
    reward = rollout(env, env.reset(batch_size=[batch_size]), policy)
    assert reward.shape == (batch_size,)


@pytest.mark.parametrize("size, batch_size", [(10, 2)])
def test_pdp(size, batch_size):
    env = PDPEnv(num_loc=size)
    reward = rollout(env, env.reset(batch_size=[batch_size]), policy)
    assert reward.shape == (batch_size,)
