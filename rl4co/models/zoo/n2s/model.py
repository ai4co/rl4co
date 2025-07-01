import torch.nn as nn

from rl4co.envs import RL4COEnvBase
from rl4co.models.nn.graph.attnnet import MultiHeadAttentionLayer
from rl4co.models.rl import n_step_PPO
from rl4co.models.rl.common.critic import CriticNetwork
from rl4co.models.zoo.n2s.decoder import CriticDecoder
from rl4co.models.zoo.n2s.policy import N2SPolicy
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


class N2S(n_step_PPO):
    """N2S Model based on n_step Proximal Policy Optimization (PPO) with an N2S model policy.
    We default to the N2S model policy and the improvement Critic Network.

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
            policy = N2SPolicy(env_name=env.name, **policy_kwargs)

        if critic is None:
            embed_dim = (
                policy_kwargs["embed_dim"] if "embed_dim" in policy_kwargs else 128
            )  # the critic's embed_dim must be as policy's

            encoder = MultiHeadAttentionLayer(
                embed_dim,
                critic_kwargs["num_heads"] if "num_heads" in critic_kwargs else 4,
                critic_kwargs["feedforward_hidden"]
                if "feedforward_hidden" in critic_kwargs
                else 128,
                critic_kwargs["normalization"]
                if "normalization" in critic_kwargs
                else "layer",
                bias=False,
            )
            value_head = CriticDecoder(embed_dim)

            critic = CriticNetwork(
                encoder=encoder,
                value_head=value_head,
                customized=True,
            )

        super().__init__(env, policy, critic, **kwargs)
