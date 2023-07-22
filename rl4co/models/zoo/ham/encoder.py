import torch.nn as nn

from rl4co.models.nn.env_embeddings import env_init_embedding
from rl4co.models.nn.graph.attnnet import Normalization, SkipConnection
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
        embedding_dim,
        num_encoder_layers,
        env_name=None,
        normalization="batch",
        feed_forward_hidden=512,
        force_flash_attn=False,
    ):
        super(GraphHeterogeneousAttentionEncoder, self).__init__()

        # Map input to embedding space
        self.init_embedding = env_init_embedding(
            env_name, {"embedding_dim": embedding_dim}
        )

        self.layers = nn.Sequential(
            *(
                HeterogeneuousMHALayer(
                    num_heads,
                    embedding_dim,
                    feed_forward_hidden,
                    normalization,
                )
                for _ in range(num_encoder_layers)
            )
        )

    def forward(self, x, mask=None):
        assert mask is None, "Mask not yet supported!"
        # initial Embedding from features
        init_embeds = self.init_embedding(x)  # (batch_size, graph_size, embed_dim)
        # layers  (batch_size, graph_size, embed_dim)
        embeds = self.layers(init_embeds)
        return embeds, init_embeds
