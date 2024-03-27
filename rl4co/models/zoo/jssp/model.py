from rl4co.envs.common.base import RL4COEnvBase
from rl4co.models.rl import PPO
from rl4co.models.zoo.jssp.policy import L2DPolicy


class L2DModel(PPO):
    """Attention Model based on REINFORCE: https://arxiv.org/abs/1803.08475.

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
        policy: L2DPolicy = None,
        policy_kwargs={},
        **kwargs,
    ):
        if policy is None:
            policy = L2DPolicy(env.name, **policy_kwargs)
        critic = None  # TODO DEFINE CRITIC
        super().__init__(env, policy, critic, **kwargs)
