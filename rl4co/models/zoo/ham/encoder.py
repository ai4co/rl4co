import torch
import torch.nn as nn
import math


from rl4co.models.nn.graph import SkipConnection, Normalization
from rl4co.models.zoo.ham.attention import HeterogenousMHA


class HeterogeneuousMHALayer(nn.Sequential):
    def __init__(
        self,
        num_heads,
        embed_dim,
        feed_forward_hidden=512,
        normalization="batch",
    ):
        super(HeterogeneuousMHALayer, self).__init__(
            SkipConnection(HeterogenousMHA(num_heads, embed_dim, embed_dim)),
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


class GraphHeterogeneousAttentionEncoder(nn.Module):
    def __init__(
        self,
        num_heads,
        embed_dim,
        num_layers,
        node_dim=None,
        normalization="batch",
        feed_forward_hidden=512,
    ):
        super(GraphHeterogeneousAttentionEncoder, self).__init__()

        # To map input to embedding space
        self.init_embed = (
            nn.Linear(node_dim, embed_dim) if node_dim is not None else None
        )

        self.layers = nn.Sequential(
            *(
                HeterogeneuousMHALayer(
                    num_heads,
                    embed_dim,
                    feed_forward_hidden,
                    normalization,
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
