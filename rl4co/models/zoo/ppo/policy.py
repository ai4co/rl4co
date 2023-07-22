from typing import Tuple, Union

from tensordict import TensorDict
from torch import Tensor

from rl4co.envs import RL4COEnvBase
from rl4co.models.zoo.common.autoregressive import AutoregressivePolicy
from rl4co.models.zoo.ppo.decoder import PPODecoder


class PPOPolicy(AutoregressivePolicy):
    """PPO Policy based on Kool et al. (2019): https://arxiv.org/abs/1803.08475.
    PPOPolicy supports 'evaluate_action' method to evaluate the action probability

    Args:
        env_name: Name of the environment used to initialize embeddings
        embedding_dim: Dimension of the node embeddings
        num_encoder_layers: Number of layers in the encoder
        num_heads: Number of heads in the attention layers
        normalization: Normalization type in the attention layers
        **kwargs: keyword arguments passed to the `AutoregressivePolicy`
    """

    def __init__(
        self,
        env_name: str,
        embedding_dim: int = 128,
        num_encoder_layers: int = 3,
        num_heads: int = 8,
        normalization: str = "batch",
        **kwargs,
    ):
        super(PPOPolicy, self).__init__(
            env_name=env_name,
            decoder=PPODecoder(
                env_name=env_name,
                embedding_dim=embedding_dim,
                num_heads=num_heads,
                **kwargs,
            ),  # override decoder with PPODecoder to support 'evaluate_action"
            embedding_dim=embedding_dim,
            num_encoder_layers=num_encoder_layers,
            num_heads=num_heads,
            normalization=normalization,
            **kwargs,
        )

    def evaluate_action(
        self,
        td: TensorDict,
        action: Tensor,
        env: Union[str, RL4COEnvBase] = None,
    ) -> Tuple[Tensor, Tensor]:
        embeddings, _ = self.encoder(td)
        ll, entropy = self.decoder.evaluate_action(td, embeddings, action, env)
        return ll, entropy
