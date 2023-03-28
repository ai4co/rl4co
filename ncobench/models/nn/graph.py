from torch import nn

from ncobench.models.nn.layers import MultiHeadAttentionLayer


class GraphAttentionEncoder(nn.Module):
    def __init__(
        self,
        n_heads,
        embed_dim,
        n_layers,
        node_dim=None,
        normalization="batch",
        feed_forward_hidden=512,
        use_flash_attn=False,
    ):
        super(GraphAttentionEncoder, self).__init__()

        # To map input to embedding space
        self.init_embed = (
            nn.Linear(node_dim, embed_dim) if node_dim is not None else None
        )

        self.layers = nn.Sequential(
            *(
                MultiHeadAttentionLayer(
                    n_heads,
                    embed_dim,
                    feed_forward_hidden,
                    normalization,
                    use_flash_attn,
                )
                for _ in range(n_layers)
            )
        )

    def forward(self, x, mask=None):
        assert mask is None, "TODO mask not yet supported!"

        # Batch multiply to get initial embeddings of nodes
        h = (
            self.init_embed(x.view(-1, x.size(-1))).view(*x.size()[:2], -1)
            if self.init_embed is not None
            else x
        )

        h = self.layers(h)

        return (
            h,  # (batch_size, graph_size, embed_dim)
            h.mean(dim=1),  # average to get embedding of graph, (batch_size, embed_dim)
        )
