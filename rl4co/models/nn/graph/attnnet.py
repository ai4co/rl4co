from typing import Optional

import torch.nn as nn

from torch import Tensor

from rl4co.models.nn.attention import MultiHeadAttention
from rl4co.models.nn.ops import Normalization, SkipConnection
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


class MultiHeadAttentionLayer(nn.Sequential):
    """Multi-Head Attention Layer with normalization and feed-forward layer

    Args:
        num_heads: number of heads in the MHA
        embed_dim: dimension of the embeddings
        feed_forward_hidden: dimension of the hidden layer in the feed-forward layer
        normalization: type of normalization to use (batch, layer, none)
        force_flash_attn: whether to force FlashAttention (move to half precision)
    """

    def __init__(
        self,
        num_heads: int,
        embed_dim: int,
        feed_forward_hidden: int = 512,
        normalization: Optional[str] = "batch",
        force_flash_attn: bool = False,
    ):
        super(MultiHeadAttentionLayer, self).__init__(
            SkipConnection(
                MultiHeadAttention(
                    embed_dim, num_heads, force_flash_attn=force_flash_attn
                )
            ),
            Normalization(embed_dim, normalization),
            SkipConnection(
                nn.Sequential(
                    nn.Linear(embed_dim, feed_forward_hidden),
                    nn.ReLU(),
                    nn.Linear(feed_forward_hidden, embed_dim),
                )
                if feed_forward_hidden > 0
                else nn.Linear(embed_dim, embed_dim)
            ),
            Normalization(embed_dim, normalization),
        )


class GraphAttentionNetwork(nn.Module):
    """Graph Attention Network to encode embeddings with a series of MHA layers consisting of a MHA layer,
    normalization, feed-forward layer, and normalization. Similar to Transformer encoder, as used in Kool et al. (2019).

    Args:
        num_heads: number of heads in the MHA
        embedding_dim: dimension of the embeddings
        num_layers: number of MHA layers
        normalization: type of normalization to use (batch, layer, none)
        feed_forward_hidden: dimension of the hidden layer in the feed-forward layer
        force_flash_attn: whether to force FlashAttention (move to half precision)
    """

    def __init__(
        self,
        num_heads: int,
        embedding_dim: int,
        num_layers: int,
        normalization: str = "batch",
        feed_forward_hidden: int = 512,
        force_flash_attn: bool = False,
    ):
        super(GraphAttentionNetwork, self).__init__()

        self.layers = nn.Sequential(
            *(
                MultiHeadAttentionLayer(
                    num_heads,
                    embedding_dim,
                    feed_forward_hidden=feed_forward_hidden,
                    normalization=normalization,
                    force_flash_attn=force_flash_attn,
                )
                for _ in range(num_layers)
            )
        )

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """Forward pass of the encoder

        Args:
            x: [batch_size, graph_size, embed_dim] initial embeddings to process
            mask: [batch_size, graph_size, graph_size] mask for the input embeddings. Unused for now.
        """
        assert mask is None, "Mask not yet supported!"
        h = self.layers(x)
        return h
