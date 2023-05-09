import pytest
import torch

from rl4co.envs import TSPEnv, ATSPEnv, DPPEnv, CVRPEnv, SDVRPEnv


@pytest.mark.parametrize("size, batch_size", [(10, 2)])
def test_tsp(size, batch_size):
    env = TSPEnv(num_loc=size)
    actions = torch.stack([torch.randperm(size) for _ in range(batch_size)])
    td = env.reset(batch_size=[batch_size])
    for i in range(size):
        td.set("action", actions[:, i])
        td = env.step(td)["next"]
    reward = env.get_reward(td, actions)
    assert reward.shape == (batch_size,)


@pytest.mark.parametrize("size, batch_size", [(10, 2)])
def test_atsp(size, batch_size):
    env = ATSPEnv(num_loc=size)
    actions = torch.stack([torch.randperm(size) for _ in range(batch_size)])
    td = env.reset(batch_size=[batch_size])
    for i in range(size):
        td.set("action", actions[:, i])
        td = env.step(td)["next"]
    reward = env.get_reward(td, actions)
    assert reward.shape == (batch_size,)


@pytest.mark.parametrize("size, batch_size", [(10, 2)])
def test_dpp(size, batch_size):
    env = DPPEnv()
    actions = torch.stack([torch.randperm(size) for _ in range(batch_size)])
    td = env.reset(batch_size=[batch_size])
    for i in range(size):
        td.set("action", actions[:, i])
        td = env.step(td)["next"]
    reward = env.get_reward(td, actions)
    assert reward.shape == (batch_size,)


@pytest.mark.parametrize("size, batch_size", [(10, 2)])
def test_cvrp(size, batch_size):
    # NOTE: we may need to select random actions in a different way
    env = CVRPEnv()
    actions = torch.stack([torch.randperm(size) for _ in range(batch_size)])
    td = env.reset(batch_size=[batch_size])
    for i in range(size):
        td.set("action", actions[:, i])
        td = env.step(td)["next"]
    reward = env.get_reward(td, actions)
    assert reward.shape == (batch_size,)


@pytest.mark.parametrize("size, batch_size", [(10, 2)])
def test_sdvrp(size, batch_size):
    # NOTE: we may need to select random actions in a different way
    env = SDVRPEnv()
    actions = torch.stack([torch.randperm(size) for _ in range(batch_size)])
    td = env.reset(batch_size=[batch_size])
    for i in range(size):
        td.set("action", actions[:, i])
        td = env.step(td)["next"]
    reward = env.get_reward(td, actions)
    assert reward.shape == (batch_size,)