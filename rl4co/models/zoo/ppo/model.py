import torch.nn as nn

from rl4co.envs import RL4COEnvBase
from rl4co.models.rl import PPO
from rl4co.models.rl.common.critic import CriticNetwork
from rl4co.models.zoo.am.policy import AttentionModelPolicy


class PPOModel(PPO):
    """PPO Model based on Proximal Policy Optimization (PPO).
    We default to the attention model policy and the Attention Critic Network.

    Args:
        env: Environment to use for the algorithm
        policy: Policy to use for the algorithm
        critic: Critic to use for the algorithm
        policy_kwargs: Keyword arguments for policy
        critic_kwargs: Keyword arguments for critic
    """

    def __init__(
        self,
        env: RL4COEnvBase,
        policy: nn.Module = None,
        critic: CriticNetwork = None,
        policy_kwargs: dict = {},
        critic_kwargs: dict = {},
        **kwargs,
    ):
        if policy is None:
            policy = AttentionModelPolicy(env_name=env.name, **policy_kwargs)

        if critic is None:
            critic = CriticNetwork(env_name=env.name, **critic_kwargs)

        super().__init__(env, policy, critic, **kwargs)
