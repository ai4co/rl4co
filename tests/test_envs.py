import pytest
import torch

from rl4co.envs import TSPEnv, ATSPEnv, DPPEnv


@pytest.mark.parametrize("size, batch_size", [(10, 2)])
def test_tsp(size, batch_size):
    env = TSPEnv(num_loc=size)
    actions = torch.stack([torch.randperm(size) for _ in range(batch_size)])
    td = env.reset(batch_size=[batch_size])
    for i in range(size):
        td.set("action", actions[:, i])
        td = env.step(td)['next']
    reward = env.get_reward(td, actions)
    assert reward.shape == (batch_size,)
    

@pytest.mark.parametrize("size, batch_size", [(10, 2)])
def test_atsp(size, batch_size):
    env = ATSPEnv(num_loc=size)
    actions = torch.stack([torch.randperm(size) for _ in range(batch_size)])
    td = env.reset(batch_size=[batch_size])
    for i in range(size):
        td.set("action", actions[:, i])
        td = env.step(td)['next']
    reward = env.get_reward(td, actions)
    assert reward.shape == (batch_size,)


@pytest.mark.parametrize("size, batch_size", [(10, 2)])
def test_dpp(size, batch_size):
    env = DPPEnv()
    bs = 2
    actions = torch.stack([torch.randperm(10) for _ in range(batch_size)])
    td = env.reset(batch_size=[batch_size])
    for i in range(10):
        td.set("action", actions[:, i])
        td = env.step(td)['next']
    reward = env.get_reward(td, actions)
    assert reward.shape == (batch_size,)