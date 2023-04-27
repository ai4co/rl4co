import pytest
import torch

from rl4co.envs import TSPEnv, ATSPEnv 


@pytest.mark.parametrize("size", [10])
def test_tsp(size):
    env = TSPEnv(num_loc=size)
    bs = 2
    actions = torch.stack([torch.randperm(size) for _ in range(bs)])
    td = env.reset(batch_size=[bs])
    for i in range(size):
        td.set("action", actions[:, i])
        td = env.step(td)['next']
    reward = env.get_reward(td, actions)
    assert reward.shape == (bs,)
    

@pytest.mark.parametrize("size", [10])
def test_atsp(size):
    env = ATSPEnv(num_loc=size)
    bs = 2
    actions = torch.stack([torch.randperm(size) for _ in range(bs)])
    td = env.reset(batch_size=[bs])
    for i in range(size):
        td.set("action", actions[:, i])
        td = env.step(td)['next']
    reward = env.get_reward(td, actions)
    assert reward.shape == (bs,)
