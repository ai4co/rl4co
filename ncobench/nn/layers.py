import torch.nn as nn

from ncobench.nn.attention import MHA
from ncobench.nn.components import SkipConnection, Normalization


class MultiHeadAttentionLayer(nn.Sequential):
    def __init__(
        self,
        n_heads,
        embed_dim,
        feed_forward_hidden=512,
        normalization="batch",
        use_flash_attn=False,
    ):
        super(MultiHeadAttentionLayer, self).__init__(
            SkipConnection(MHA(embed_dim, n_heads, use_flash_attn=use_flash_attn)),
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
