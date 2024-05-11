from typing import Union

from rl4co.envs.common.base import RL4COEnvBase
from rl4co.models.rl import REINFORCE
from rl4co.models.rl.reinforce.baselines import REINFORCEBaseline

from .policy import HetGNNPolicy


class HetGNNModel(REINFORCE):
    """Heterogenous Graph Neural Network Model as described by Song et al. (2022):
    'Flexible Job Shop Scheduling via Graph Neural Network and Deep Reinforcement Learning'

    Args:
        env: Environment to use for the algorithm
        policy: Policy to use for the algorithm
        baseline: REINFORCE baseline. Defaults to rollout (1 epoch of exponential, then greedy rollout baseline)
        policy_kwargs: Keyword arguments for policy
        baseline_kwargs: Keyword arguments for baseline
        **kwargs: Keyword arguments passed to the superclass
    """

    def __init__(
        self,
        env: RL4COEnvBase,
        policy: HetGNNPolicy = None,
        baseline: Union[REINFORCEBaseline, str] = "rollout",
        policy_kwargs={},
        baseline_kwargs={},
        **kwargs,
    ):
        assert (
            env.name == "fjsp"
        ), "HetGNNModel currently only works for FJSP (Flexible Job-Shop Scheduling Problem)"
        if policy is None:
            policy = HetGNNPolicy(env_name=env.name, **policy_kwargs)

        super().__init__(env, policy, baseline, baseline_kwargs, **kwargs)
