import warnings

import matplotlib.pyplot as plt
import pytest
import torch

from tensordict import TensorDict

from rl4co.envs import (
    ATSPEnv,
    CVRPEnv,
    CVRPTWEnv,
    DPPEnv,
    FFSPEnv,
    JSSPEnv,
    MDPPEnv,
    MTSPEnv,
    OPEnv,
    PCTSPEnv,
    PDPEnv,
    SDVRPEnv,
    SMTWTPEnv,
    SPCTSPEnv,
    SVRPEnv,
    TSPEnv,
)
from rl4co.models.nn.utils import random_policy, rollout

# Switch to non-GUI backend for testing
plt.switch_backend("Agg")
warnings.filterwarnings("ignore", "Matplotlib is currently using agg")


@pytest.mark.parametrize(
    "env_cls",
    [
        TSPEnv,
        CVRPEnv,
        CVRPTWEnv,
        SVRPEnv,
        SDVRPEnv,
        PCTSPEnv,
        SPCTSPEnv,
        OPEnv,
        PDPEnv,
        MTSPEnv,
        ATSPEnv,
    ],
)
def test_routing(env_cls, batch_size=2, size=20):
    env = env_cls(num_loc=size)
    reward, td, actions = rollout(env, env.reset(batch_size=[batch_size]), random_policy)
    env.render(td, actions)
    assert reward.shape == (batch_size,)


@pytest.mark.parametrize("env_cls", [DPPEnv, MDPPEnv])
def test_eda(env_cls, batch_size=2, max_decaps=5):
    env = env_cls(max_decaps=max_decaps)
    reward, td, actions = rollout(env, env.reset(batch_size=[batch_size]), random_policy)
    ## Note: we skip rendering for now because we need to collect extra data. TODO
    # env.render(td, actions)
    assert reward.shape == (batch_size,)


@pytest.mark.parametrize("env_cls", [FFSPEnv])
def test_scheduling(env_cls, batch_size=2):
    env = env_cls(
        num_stage=2,
        num_machine=3,
        num_job=4,
        batch_size=[batch_size],
    )
    td = env.reset()
    td["action"] = torch.tensor([1, 1])
    td = env._step(td)


@pytest.mark.parametrize("env_cls", [SMTWTPEnv])
def test_smtwtp(env_cls, batch_size=2):
    env = env_cls(num_job=4)
    reward, td, actions = rollout(env, env.reset(batch_size=[batch_size]), random_policy)
    assert reward.shape == (batch_size,)


@pytest.mark.parametrize("env_cls", [JSSPEnv])
def test_jssp(env_cls, batch_size=2):
    env = env_cls(num_jobs=4, num_machines=5, batch_size=[batch_size])
    reward, td, actions = rollout(env, env.reset(), random_policy)
    assert reward.shape == (batch_size,)


@pytest.mark.parametrize("env_cls", [JSSPEnv])
def test_jssp_lb(env_cls):
    env = env_cls(num_jobs=2, num_machines=2, batch_size=[1])
    td = TensorDict(
        {
            "durations": torch.tensor([[[1, 2], [3, 4]]], dtype=torch.float32),
            "machines": torch.tensor([[[0, 1], [1, 0]]], dtype=torch.long),
        },
        batch_size=[1],
    )
    td.set("ops", td["machines"].argsort(2))

    td = env._reset(td)

    actions = [0, 1, 1]
    for action in actions:
        td.set("action", torch.tensor([action], dtype=torch.long))
        td = env._step(td)

    lb_expected = torch.tensor([[1, 3], [3, 7]], dtype=torch.float32)
    adj_expected = torch.tensor(
        [[1, 0, 0, 0], [1, 1, 0, 0], [0, 0, 1, 0], [1, 0, 1, 1]], dtype=torch.float32
    ).unsqueeze(0)
    assert torch.allclose(td["lower_bounds"], lb_expected)
    assert torch.allclose(td["adjacency"], adj_expected)
