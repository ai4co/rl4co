from typing import Callable, Optional

import torch.nn as nn

from rl4co.models.zoo.am import AttentionModelPolicy
from rl4co.models.zoo.ham.encoder import GraphHeterogeneousAttentionEncoder


class HeterogeneousAttentionModelPolicy(AttentionModelPolicy):
    """Heterogeneous Attention Model Policy based on # TODO
    We re-declare the most important arguments here for convenience as in the paper.
    See `AutoregressivePolicy` superclass for more details.

    Args:
        env_name: Name of the environment used to initialize embeddings
        encoder: Encoder to use for the policy
        embed_dim: Dimension of the node embeddings
        num_encoder_layers: Number of layers in the encoder
        num_heads: Number of heads in the attention layers
        normalization: Normalization type in the attention layers
        **kwargs: keyword arguments passed to the `AutoregressivePolicy`
    """

    def __init__(
        self,
        encoder: nn.Module = None,
        env_name: str = "pdp",
        init_embedding: nn.Module = None,
        embed_dim: int = 128,
        num_encoder_layers: int = 3,
        num_heads: int = 8,
        normalization: str = "batch",
        feedforward_hidden: int = 512,
        sdpa_fn: Optional[Callable] = None,
        **kwargs,
    ):
        if encoder is None:
            encoder = GraphHeterogeneousAttentionEncoder(
                init_embedding=init_embedding,
                num_heads=num_heads,
                embed_dim=embed_dim,
                num_encoder_layers=num_encoder_layers,
                env_name=env_name,
                normalization=normalization,
                feedforward_hidden=feedforward_hidden,
                sdpa_fn=sdpa_fn,
            )
        else:
            encoder = encoder

        super(HeterogeneousAttentionModelPolicy, self).__init__(
            env_name=env_name,
            encoder=encoder,
            embed_dim=embed_dim,
            num_encoder_layers=num_encoder_layers,
            num_heads=num_heads,
            normalization=normalization,
            **kwargs,
        )
