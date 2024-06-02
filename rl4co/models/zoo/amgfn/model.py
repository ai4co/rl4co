from typing import Union

from rl4co.envs.common.base import RL4COEnvBase
from rl4co.models.rl import TB
from rl4co.models.rl.reinforce.baselines import REINFORCEBaseline
from rl4co.models.zoo.amgfn.policy import AttentionGFNModelPolicy


class AttentionGFNModel(TB):
    """Attention Model based on REINFORCE: https://arxiv.org/abs/1803.08475.
    Check :class:`REINFORCE` and :class:`rl4co.models.RL4COLitModule` for more details such as additional parameters  including batch size.

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
        policy: AttentionGFNModelPolicy = None,
        baseline: Union[REINFORCEBaseline, str] = "no",
        policy_kwargs={},
        baseline_kwargs={},
        beta: int = 1,
        gfn_epochs: int = 2,
        **kwargs,
    ):
        if policy is None:
            policy = AttentionGFNModelPolicy(env_name=env.name, **policy_kwargs)

        super().__init__(env, policy, baseline, baseline_kwargs, beta, gfn_epochs, **kwargs)
