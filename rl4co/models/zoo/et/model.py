from typing import Optional, Union

from rl4co.envs.common.base import RL4COEnvBase
from rl4co.models.rl import REINFORCE
from rl4co.models.rl.reinforce.baselines import REINFORCEBaseline
from rl4co.models.zoo.am.policy import AttentionModelPolicy


class EquityTransformer(REINFORCE):
    """Equity Transformer from Son et al., 2023.
    Reference: https://arxiv.org/abs/2306.02689

    Warning:
        This implementation is under development and subject to change.
    """

    def __init__(
        self,
        env: RL4COEnvBase,
        policy: Optional(AttentionModelPolicy) = None,
        baseline: Union[REINFORCEBaseline, str] = "rollout",
        policy_kwargs={},
        baseline_kwargs={},
        **kwargs,
    ):
        if policy is None:
            policy = AttentionModelPolicy(env.name, **policy_kwargs)

        super().__init__(env, policy, baseline, baseline_kwargs, **kwargs)
