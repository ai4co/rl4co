import torch.nn as nn

from rl4co.models.nn.attention import NativeFlashMHA
from rl4co.models.nn.env_embedding import env_init_embedding
from rl4co.models.nn.kooltention import MultiHeadAttention
from rl4co.models.nn.ops import Normalization, SkipConnection
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


class MultiHeadAttentionLayer(nn.Sequential):
    def __init__(
        self,
        num_heads,
        embed_dim,
        feed_forward_hidden=512,
        normalization="batch",
        force_flash_attn=False,
    ):
        # monkey patch to use Kool's attention implementation.

        super(MultiHeadAttentionLayer, self).__init__(
            SkipConnection(
                MultiHeadAttention(
                    n_heads=num_heads,
                    input_dim=embed_dim,
                    embed_dim=embed_dim,
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


class GraphAttentionEncoder(nn.Module):
    def __init__(
        self,
        num_heads,
        embedding_dim,
        num_layers,
        env=None,
        normalization="batch",
        feed_forward_hidden=512,
        force_flash_attn=False,
        disable_init_embedding=False,
    ):
        super(GraphAttentionEncoder, self).__init__()

        # To map input to embedding space
        if not disable_init_embedding:
            self.init_embedding = env_init_embedding(env, {"embedding_dim": embedding_dim})
        else:
            log.warning("Disabling init embedding manually for GraphAttentionEncoder")
            self.init_embedding = nn.Identity()  # do nothing

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
        init_embeds = self.init_embedding(x)
        # layers  (batch_size, graph_size, embed_dim)
        embeds = self.layers(init_embeds)
        return embeds, init_embeds
