from typing import Union

from rl4co.envs.common.base import RL4COEnvBase
from rl4co.models.rl import REINFORCE
from rl4co.models.rl.reinforce.baselines import REINFORCEBaseline
from rl4co.models.zoo.deepaco.policy import DeepACOPolicy


class DeepACO(REINFORCE):
    def __init__(
        self,
        env: RL4COEnvBase,
        baseline: Union[REINFORCEBaseline, str] = "exponential",
        policy_kwargs: dict = {},
        baseline_kwargs: dict = {},
        **kwargs,
    ):
        policy = DeepACOPolicy(env.name, **policy_kwargs)

        super().__init__(env, policy, baseline, baseline_kwargs, **kwargs)
