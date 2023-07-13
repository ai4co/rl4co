from typing import Union

from rl4co.algos.reinforce.baselines import REINFORCEBaseline
from rl4co.algos.reinforce.reinforce import REINFORCE
from rl4co.envs.common.base import RL4COEnvBase
from rl4co.models.zoo.am.policy import AttentionModelPolicy


class AttentionModel(REINFORCE):
    """Attention Model based on REINFORCE.

    Args:
        env: Environment to use for the algorithm
        policy: Policy to use for the algorithm
        baseline: REINFORCE baseline
        policy_kwargs: Keyword arguments for policy
        baseline_kwargs: Keyword arguments for baseline
        **kwargs: Keyword arguments passed to the superclass
    """

    def __init__(
        self,
        env: RL4COEnvBase,
        policy: AttentionModelPolicy = None,
        baseline: Union[REINFORCEBaseline, str] = "rollout",
        policy_kwargs={},
        baseline_kwargs={},
        **kwargs,
    ):
        if policy is None:
            policy = AttentionModelPolicy(env.name, **policy_kwargs)

        super().__init__(env, policy, baseline, baseline_kwargs, **kwargs)


if __name__ == "__main__":
    from rl4co.envs import TSPEnv

    env = TSPEnv()
    model = AttentionModel(env)

    td = env.reset(batch_size=10)

    out = model(td)

    print(out["reward"].shape)
