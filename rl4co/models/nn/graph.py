import math
import torch
import torch.nn as nn

from rl4co.models.nn.attention import NativeFlashMHA
from rl4co.models.nn.ops import Normalization, SkipConnection
from rl4co.models.nn.env_embedding import env_init_embedding


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
        embedding_dim,
        num_layers,
        env_name="tsp",
        normalization="batch",
        feed_forward_hidden=512,
        force_flash_attn=False,
    ):
        super(GraphAttentionEncoder, self).__init__()

        # To map input to embedding space
        self.init_embedding = env_init_embedding(
            env_name, {"embedding_dim": embedding_dim}
        )

        self.layers = nn.Sequential(
            *(
                MultiHeadAttentionLayer(
                    num_heads,
                    embedding_dim,
                    feed_forward_hidden,
                    normalization,
                    force_flash_attn,
                )
                for _ in range(num_layers)
            )
        )

    def forward(self, x, mask=None):
        assert mask is None, "Mask not yet supported!"
        # initial Embedding from features
        init_embeds = self.init_embedding(x)  # (batch_size, graph_size, embed_dim)
        # layers  (batch_size, graph_size, embed_dim)
        embeds = self.layers(init_embeds)
        return embeds, init_embeds
