import math
import torch
import torch.nn as nn

from rl4co.models.nn.attention import NativeFlashMHA
from rl4co.models.nn.ops import Normalization, SkipConnection


class MultiHeadAttentionLayer(nn.Sequential):
    def __init__(
        self,
        num_heads,
        embed_dim,
        feed_forward_hidden=512,
        normalization="batch",
        force_flash_attn=False,
    ):
        super(MultiHeadAttentionLayer, self).__init__(
            SkipConnection(
                NativeFlashMHA(embed_dim, num_heads, force_flash_attn=force_flash_attn)
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


class GraphAttentionEncoder(nn.Module):
    def __init__(
        self,
        num_heads,
        embed_dim,
        num_layers,
        node_dim=None,
        normalization="batch",
        feed_forward_hidden=512,
        force_flash_attn=False,
    ):
        super(GraphAttentionEncoder, self).__init__()

        # To map input to embedding space
        self.init_embed = (
            nn.Linear(node_dim, embed_dim) if node_dim is not None else None
        )

        self.layers = nn.Sequential(
            *(
                MultiHeadAttentionLayer(
                    num_heads,
                    embed_dim,
                    feed_forward_hidden,
                    normalization,
                    force_flash_attn,
                )
                for _ in range(num_layers)
            )
        )

    def forward(self, x, mask=None):
        assert mask is None, "Mask not yet supported!"

        # Batch multiply to get initial embeddings of nodes
        h = (
            self.init_embed(x.view(-1, x.size(-1))).view(*x.size()[:2], -1)
            if self.init_embed is not None
            else x
        )

        h = self.layers(h)
        return h  # (batch_size, graph_size, embed_dim)
