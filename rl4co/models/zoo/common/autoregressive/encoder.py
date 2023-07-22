from typing import Tuple, Union

import torch.nn as nn

from tensordict import TensorDict
from torch import Tensor

from rl4co.models.nn.env_embeddings import env_init_embedding
from rl4co.models.nn.graph.attnnet import GraphAttentionNetwork


class GraphAttentionEncoder(nn.Module):
    """Graph Attention Encoder as in Kool et al. (2019).

    Args:
        env_name: environment name to solve
        num_heads: Number of heads for the attention
        embedding_dim: Dimension of the embeddings
        num_layers: Number of layers for the encoder
        normalization: Normalization to use for the attention
        feed_forward_hidden: Hidden dimension for the feed-forward network
        force_flash_attn: Whether to force the use of flash attention. If True, cast to fp16
    """

    def __init__(
        self,
        env_name: str,
        num_heads: int,
        embedding_dim: int,
        num_layers: int,
        normalization: str = "batch",
        feed_forward_hidden: int = 512,
        force_flash_attn: bool = False,
    ):
        super(GraphAttentionEncoder, self).__init__()

        self.env_name = env_name
        self.init_embedding = env_init_embedding(
            self.env_name, {"embedding_dim": embedding_dim}
        )
        self.net = GraphAttentionNetwork(
            num_heads,
            embedding_dim,
            num_layers,
            normalization,
            feed_forward_hidden,
            force_flash_attn,
        )

    def forward(
        self, td: TensorDict, mask: Union[Tensor, None] = None
    ) -> Tuple[Tensor, Tensor]:
        """Forward pass of the encoder.
        Transform the input TensorDict into a latent representation.

        Args:
            td: Input TensorDict containing the environment state
            mask: Mask to apply to the attention

        Returns:
            h: Latent representation of the input
            init_h: Initial embedding of the input
        """
        # Transfer to embedding space
        init_h = self.init_embedding(td)

        # Process embedding
        h = self.net(init_h, mask)

        # Return latent representation and initial embedding
        return h, init_h
