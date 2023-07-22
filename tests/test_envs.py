import warnings

import matplotlib.pyplot as plt
import pytest
import torch

from rl4co.envs import (
    ATSPEnv,
    CVRPEnv,
    DPPEnv,
    FFSPEnv,
    MDPPEnv,
    MTSPEnv,
    OPEnv,
    PCTSPEnv,
    PDPEnv,
    SDVRPEnv,
    SPCTSPEnv,
    TSPEnv,
)
from rl4co.models.nn.utils import random_policy, rollout

# Switch to non-GUI backend for testing
plt.switch_backend("Agg")
warnings.filterwarnings("ignore", "Matplotlib is currently using agg")


@pytest.mark.parametrize(
    "env_cls",
    [TSPEnv, CVRPEnv, SDVRPEnv, PCTSPEnv, SPCTSPEnv, OPEnv, PDPEnv, MTSPEnv, ATSPEnv],
)
def test_routing(env_cls, batch_size=2, size=20):
    env = env_cls(num_loc=size)
    reward, td, actions = rollout(env, env.reset(batch_size=[batch_size]), random_policy)
    env.render(td, actions)
    assert reward.shape == (batch_size,)


@pytest.mark.parametrize("env_cls", [DPPEnv, MDPPEnv])
def test_eda(env_cls, batch_size=2, size=20):
    env = env_cls(num_loc=size)
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
    td["job_idx"] = torch.tensor([1, 1])
    td = env._step(td)
