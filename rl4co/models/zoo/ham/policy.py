import torch.nn as nn
from rl4co.models.zoo.common.autoregressive import AutoregressivePolicy
from rl4co.models.zoo.ham.encoder import GraphHeterogeneousAttentionEncoder


class HeterogeneousAttentionModelPolicy(AutoregressivePolicy):
    """Heterogeneous Attention Model Policy based on Kool et al. (2019): https://arxiv.org/abs/1803.08475.
    We re-declare the most important arguments here for convenience as in the paper.
    See `AutoregressivePolicy` superclass for more details.

    Args:
        env_name: Name of the environment used to initialize embeddings
        encoder: Encoder to use for the policy
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
        super(HeterogeneousAttentionModelPolicy, self).__init__(
            env_name=env_name,
            encoder=GraphHeterogeneousAttentionEncoder(
                num_heads=num_heads,
                embedding_dim=embedding_dim,
                num_encoder_layers=num_encoder_layers,
                env_name=env_name,
                normalization=normalization,
            ),
            embedding_dim=embedding_dim,
            num_encoder_layers=num_encoder_layers,
            num_heads=num_heads,
            normalization=normalization,
            **kwargs,
        )