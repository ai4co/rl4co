
from typing import Union

from rl4co.envs.common.base import RL4COEnvBase
from rl4co.models.rl import REINFORCE
from rl4co.models.rl.reinforce.baselines import REINFORCEBaseline
from rl4co.models.zoo.mdam.policy import MDAMPolicy


class MDAM(REINFORCE):
    """ Multi-Decoder Attention Model (MDAM) is a model
    to train multiple diverse policies, which effectively increases the chance of finding 
    good solutions compared with existing methods that train only one policy.
    Reference link: https://arxiv.org/abs/2012.10638; 
    Implementation reference: https://github.com/liangxinedu/MDAM.

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
            policy: MDAMPolicy = None, 
            baseline: Union[REINFORCEBaseline, str] = "rollout", 
            policy_kwargs={},
            baseline_kwargs={},
            **kwargs
        ):
        if policy is None:
            policy = MDAMPolicy(env.name, **policy_kwargs) 

        super().__init__(env, policy, baseline, baseline_kwargs, **kwargs)