import torch.nn as nn

from rl4co.envs.common.base import RL4COEnvBase
from rl4co.models.rl.common.critic import CriticNetwork, create_critic_from_actor
from rl4co.models.rl.reinforce.baselines import CriticBaseline
from rl4co.models.rl.reinforce.reinforce import REINFORCE
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


class A2C(REINFORCE):
    """Advantage Actor Critic (A2C) algorithm.
    A2C is a variant of REINFORCE where a baseline is provided by a critic network.

    Args:
        env: Environment to use for the algorithm
        policy: Policy to use for the algorithm
        critic: Critic network to use for the algorithm
        critic_kwargs: Keyword arguments to pass to the critic network
        **kwargs: Keyword arguments passed to the superclass
    """

    def __init__(
        self,
        env: RL4COEnvBase,
        policy: nn.Module,
        critic: CriticNetwork = None,
        critic_kwargs: dict = {},
        **kwargs,
    ):
        if critic is None:
            log.info("Creating critic network for {}".format(env.name))
            critic = create_critic_from_actor(policy, **critic_kwargs)

        super().__init__(env, policy, baseline=CriticBaseline(critic), **kwargs)
