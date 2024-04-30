from typing import Optional, Union

from rl4co.envs.common.base import RL4COEnvBase
from rl4co.models.rl import REINFORCE
from rl4co.models.rl.reinforce.baselines import REINFORCEBaseline
from rl4co.models.zoo.am.policy import AttentionModelPolicy
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


class EquityTransformer(REINFORCE):
    """Equity Transformer from Son et al., 2024.
    Reference: https://arxiv.org/abs/2306.02689

    Note that you may find the embeddings proposed in the paper under
    :class:`rl4co.models.models.nn.env_embeddings`.

    Warning:
        This implementation is under development and subject to change.
    """

    def __init__(
        self,
        env: RL4COEnvBase,
        policy: Optional[AttentionModelPolicy] = None,
        baseline: Union[REINFORCEBaseline, str] = "rollout",
        policy_kwargs={},
        baseline_kwargs={},
        **kwargs,
    ):
        if policy is None:
            policy = AttentionModelPolicy(env_name=env.name, **policy_kwargs)

        if env.name not in ["mtsp", "mpdp"]:
            log.error(f"env_name {env.name } is not originally implemented in ET")

        super().__init__(env, policy, baseline, baseline_kwargs, **kwargs)
